#!/usr/bin/env bash
# Launch the SmolVLA Eval 2 training as a transient systemd user service.
# This places it in /user.slice/user-1001.slice (which is lingered),
# guaranteeing survival across SSH disconnect, laptop close, etc.
#
# After launch:
#   ~/training_status.sh                       # one-shot snapshot
#   ~/follow_training.sh                       # live tail of log
#   journalctl --user -u lerobot-train-eval2 -f
#   systemctl --user status lerobot-train-eval2
#   systemctl --user stop lerobot-train-eval2  # cancel cleanly
set -e

UNIT=lerobot-train-eval2

# When invoked from a non-PAM shell (cron, Claude Code, etc.), XDG_RUNTIME_DIR
# isn't set so `systemctl --user` can't reach the user manager. Point at it.
UID_NUM=$(id -u "$USER")
export XDG_RUNTIME_DIR="/run/user/$UID_NUM"
export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"

# 1. Verify lingering - without it, the user manager dies on logout and so
#    does this transient service.
LINGER_STATE=$(loginctl show-user "$USER" --property=Linger 2>/dev/null | cut -d= -f2)
if [ "$LINGER_STATE" != "yes" ]; then
  echo "ERROR: user lingering is NOT enabled for $USER."
  echo "  Without it, the systemd --user manager dies on logout."
  echo "  Fix:    sudo loginctl enable-linger $USER"
  exit 1
fi

# 2. Verify systemd --user manager is up.
if ! systemctl --user is-active --quiet default.target; then
  echo "ERROR: user systemd manager isn't running default.target."
  echo "  Try:  systemctl --user start default.target"
  exit 1
fi

# 3. Kill any prior run.
if systemctl --user is-active --quiet "$UNIT.service" 2>/dev/null; then
  echo "==> stopping previous $UNIT.service"
  systemctl --user stop "$UNIT.service" || true
  sleep 2
fi

# Also kill any orphan lerobot-train (from older tmux runs etc).
if ps -ef | grep "lerobot-train" | grep -v grep >/dev/null; then
  echo "==> killing orphan lerobot-train processes"
  pkill -9 -f "lerobot-train" 2>/dev/null || true
  sleep 2
fi

# Wipe stale log so fresh run starts clean.
rm -f ~/outputs/train/so101_smolvla_eval2.log

# 4. Launch as a transient user service inside user-1001.slice.
echo "==> launching $UNIT as a transient user service"
systemd-run \
  --user \
  --unit="$UNIT" \
  --description="LeRobot SmolVLA Eval 2 training (compositional, 180 ep)" \
  --property=Type=simple \
  --property=KillMode=control-group \
  --property=KillSignal=SIGTERM \
  --property=Restart=no \
  bash ~/run_training.sh

# 5. Give it a moment to start, then verify cgroup.
sleep 3
echo
echo "==> service status:"
systemctl --user status "$UNIT.service" --no-pager 2>&1 | head -8
echo
TRAIN_PID=$(systemctl --user show "$UNIT.service" --property=MainPID | cut -d= -f2)
if [ -n "$TRAIN_PID" ] && [ "$TRAIN_PID" != "0" ]; then
  CG=$(cat /proc/$TRAIN_PID/cgroup 2>/dev/null | head -1)
  echo "==> training cgroup: $CG"
  case "$CG" in
    *user-*.slice*) echo "==> [OK] running in user slice - will survive disconnect" ;;
    *) echo "==> [WARN] unexpected cgroup; expected user-1001.slice" ;;
  esac
fi
echo
echo "next steps:"
echo "  ~/training_status.sh                       # one-shot snapshot"
echo "  ~/follow_training.sh                       # live tail of log"
echo "  journalctl --user -u $UNIT -f              # journal stream"
echo "  systemctl --user status $UNIT              # service health"
echo "  systemctl --user stop $UNIT                # cancel cleanly"
