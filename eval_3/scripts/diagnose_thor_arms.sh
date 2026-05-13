#!/bin/bash
# One-shot diagnostic: capture EVERYTHING relevant about thor's USB state +
# arm motor bus reachability. Run this with BOTH arms plugged into thor.
#
# Output: /tmp/thor_arm_diag.txt + console echo.
#
# What we're trying to distinguish:
#   - Thor USB power / autosuspend dropping data on one PCB
#   - Tegra cdc_acm driver quirk
#   - Hub power budget exceeded
#   - Specific USB cable/port being unreliable
set +e
OUT=/tmp/thor_arm_diag.txt
exec > >(tee "$OUT") 2>&1

echo "==============================================================="
echo "  thor SO-101 arm diagnostic  ($(date))"
echo "==============================================================="

echo "[1] Device nodes:"
ls -la /dev/so101-* /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
echo

echo "[2] udev: serial → port map:"
for d in /dev/ttyACM* /dev/ttyUSB*; do
  [[ -e "$d" ]] || continue
  printf "  %s  serial=%s  vendor=%s\n" \
    "$d" \
    "$(udevadm info "$d" 2>/dev/null | awk -F= '/ID_SERIAL_SHORT/{print $2}')" \
    "$(udevadm info "$d" 2>/dev/null | awk -F= '/ID_VENDOR=/{print $2; exit}')"
done
echo

echo "[3] USB topology + speeds:"
lsusb -t | grep -E "tegra|hub|cdc_acm" | head -20
echo

echo "[4] Power info per arm:"
for d in /sys/bus/usb/devices/*; do
  if [[ -f "$d/idVendor" && "$(cat "$d/idVendor" 2>/dev/null)" == "1a86" ]]; then
    echo "  $(basename "$d"):"
    echo "    serial:        $(cat "$d/serial" 2>/dev/null)"
    echo "    bMaxPower:     $(cat "$d/bMaxPower" 2>/dev/null)"
    echo "    power/control: $(cat "$d/power/control" 2>/dev/null)"
    echo "    power/runtime_active_time: $(cat "$d/power/runtime_active_time" 2>/dev/null) ms"
    echo "    power/runtime_suspended_time: $(cat "$d/power/runtime_suspended_time" 2>/dev/null) ms"
    echo "    power/autosuspend_delay_ms: $(cat "$d/power/autosuspend_delay_ms" 2>/dev/null)"
    echo "    devpath:       $(cat "$d/devpath" 2>/dev/null)"
    echo "    busnum/devnum: $(cat "$d/busnum" 2>/dev/null)/$(cat "$d/devnum" 2>/dev/null)"
  fi
done
echo

echo "[5] Hub power budget upstream of the arms:"
for d in /sys/bus/usb/devices/*; do
  if [[ -f "$d/bDeviceClass" && "$(cat "$d/bDeviceClass" 2>/dev/null)" == "09" ]]; then
    echo "  HUB $(basename "$d"):"
    echo "    bcdUSB:        $(cat "$d/version" 2>/dev/null)"
    echo "    bMaxPower:     $(cat "$d/bMaxPower" 2>/dev/null)"
    echo "    speed:         $(cat "$d/speed" 2>/dev/null) Mb/s"
    echo "    devpath:       $(cat "$d/devpath" 2>/dev/null)"
  fi
done
echo

echo "[6] Live motor ping (1 Mbps Feetech default):"
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate lemonkey 2>/dev/null
python3 <<'PY'
import serial, time, os, sys
for p in ["/dev/so101-leader", "/dev/so101-follower"]:
    if not os.path.exists(p):
        print(f"  {p}: NOT PRESENT"); continue
    try:
        s = serial.Serial(p, 1_000_000, timeout=0.3)
        # broadcast ping
        s.write(bytes([0xFF, 0xFF, 0xFE, 0x02, 0x01, ~(0xFE+2+1)&0xFF]))
        time.sleep(0.3)
        r = s.read(256)
        print(f"  {p}: {len(r)} bytes  {('reply ' + r.hex()) if r else '(SILENT)'}")
        s.close()
    except Exception as e:
        print(f"  {p}: error {e}")
PY
echo

echo "[7] dmesg (most recent USB events):"
sudo dmesg 2>/dev/null | grep -iE "ttyACM|usb.*disc|usb.*conn|cdc_acm|under.*volt|over.*current|tegra-xusb" | tail -15
echo "(if you didn't sudo this, kernel log is empty above)"
echo

echo "==============================================================="
echo "  diagnostic saved to $OUT"
echo "==============================================================="
