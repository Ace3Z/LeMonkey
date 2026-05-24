#!/usr/bin/env bash
# Launch a lerobot-train run as a transient systemd --user service so it
# survives SSH disconnect / laptop close (the unit lands in user-NNNN.slice
# which is lingered). Wipes the previous log, stops any prior instance of
# the same UNIT, and verifies the new pid is in the user slice.
#
# Required env (no defaults; every eval picks its own):
#   UNIT            systemd --user unit name           e.g. lerobot-train-eval3
#   DESCRIPTION     unit description string            e.g. "LeRobot SmolVLA Eval 3 (image-as-prompt)"
#   TRAIN_SCRIPT    absolute path to the bash trainer  e.g. $REPO_ROOT/eval_3/scripts/brev/train_smolvla_broad.sh
#   LOG_FILE        log path to wipe before launch     e.g. $HOME/outputs/train/so101_smolvla_eval3_broad.log
#
# Optional env:
#   LIMIT_NOFILE    override LimitNOFILE on the unit   e.g. 524288  (large multi-mp4 datasets)
#
# Usage:
#   UNIT=lerobot-train-eval3 \
#   DESCRIPTION="LeRobot SmolVLA Eval 3 training (image-as-prompt Coke-on-celebrity)" \
#   TRAIN_SCRIPT=$REPO_ROOT/eval_3/scripts/brev/train_smolvla_broad.sh \
#   LOG_FILE=$HOME/outputs/train/so101_smolvla_eval3_broad.log \
#   LIMIT_NOFILE=524288 \
#       bash scripts/brev/start_training.sh
set -e

UNIT="${UNIT:-}"
DESCRIPTION="${DESCRIPTION:-}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-}"
LOG_FILE="${LOG_FILE:-}"
LIMIT_NOFILE="${LIMIT_NOFILE:-}"

if [ -z "$UNIT" ] || [ -z "$DESCRIPTION" ] || [ -z "$TRAIN_SCRIPT" ] || [ -z "$LOG_FILE" ]; then
  echo "[FATAL] UNIT, DESCRIPTION, TRAIN_SCRIPT, LOG_FILE all required." >&2
  echo "        See the header of this script for the invocation pattern." >&2
  exit 2
fi
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "[FATAL] TRAIN_SCRIPT does not exist: $TRAIN_SCRIPT" >&2
  exit 2
fi

# When invoked from a non-PAM shell (cron, Claude Code, etc.), XDG_RUNTIME_DIR
# isn't set so `systemctl --user` can't reach the user manager. Point at it.
UID_NUM=$(id -u "$USER")
export XDG_RUNTIME_DIR="/run/user/$UID_NUM"
export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"

# 1. Verify lingering. Without it, the user manager dies on logout and so
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

if ps -ef | grep "lerobot-train" | grep -v grep >/dev/null; then
  echo "==> killing orphan lerobot-train processes"
  pkill -9 -f "lerobot-train" 2>/dev/null || true
  sleep 2
fi

# Wipe stale log so a fresh tail starts clean.
rm -f "$LOG_FILE"

# 4. Launch as a transient user service inside the lingered user slice.
echo "==> launching $UNIT as a transient user service"
SYSTEMD_RUN_ARGS=(
  --user
  --unit="$UNIT"
  --description="$DESCRIPTION"
  --property=Type=simple
  --property=KillMode=control-group
  --property=KillSignal=SIGTERM
  --property=Restart=no
)
if [ -n "$LIMIT_NOFILE" ]; then
  SYSTEMD_RUN_ARGS+=(--property=LimitNOFILE="$LIMIT_NOFILE")
fi
systemd-run "${SYSTEMD_RUN_ARGS[@]}" bash "$TRAIN_SCRIPT"

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
    *) echo "==> [WARN] unexpected cgroup; expected user-NNNN.slice" ;;
  esac
fi
echo
echo "next steps:"
echo "  bash scripts/brev/training_status.sh                # one-shot snapshot (set LOG/UNIT/CHECKPOINT_DIR)"
echo "  bash scripts/brev/follow_training.sh $LOG_FILE      # live tail of log"
echo "  journalctl --user -u $UNIT -f                       # journal stream"
echo "  systemctl --user status $UNIT                       # service health"
echo "  systemctl --user stop $UNIT                         # cancel cleanly"
