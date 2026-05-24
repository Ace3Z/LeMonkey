#!/usr/bin/env bash
# Follow a lerobot-train log live. Colourises matched patterns by severity:
# red = traceback / OOM / SIGTERM, yellow = warnings, green = loss / step /
# checkpoint events, cyan = the periodic % progress line.
#
# Usage:
#   bash scripts/brev/follow_training.sh /path/to/lerobot-train.log
#
# Or:
#   LOG=/path/to/file.log bash scripts/brev/follow_training.sh
#
# This is the shared launcher invoked by each eval's brev README. The eval-
# specific defaults (log filename, systemd unit, checkpoint dir) live next
# to the README that documents them; this script takes them as a CLI arg.
set -u

LOG="${1:-${LOG:-}}"
if [ -z "$LOG" ]; then
  echo "[FATAL] no log path supplied; pass as argv[1] or set LOG=…" >&2
  exit 2
fi

B=$(tput bold 2>/dev/null || true)
R=$(tput setaf 1 2>/dev/null || true)
G=$(tput setaf 2 2>/dev/null || true)
Y=$(tput setaf 3 2>/dev/null || true)
C=$(tput setaf 6 2>/dev/null || true)
N=$(tput sgr0 2>/dev/null || true)

if [ ! -f "$LOG" ]; then
  echo "${Y}waiting for $LOG to appear...${N}"
  until [ -f "$LOG" ]; do sleep 1; done
fi

echo "${B}following: $LOG${N}"
echo "${B}gpu snapshot:${N} $(nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable')"
echo "${B}─────────────────────────────────────────────${N}"

tail -c 200000 -F "$LOG" 2>/dev/null \
  | stdbuf -oL tr '\r' '\n' \
  | stdbuf -oL awk -v R="$R" -v G="$G" -v Y="$Y" -v C="$C" -v B="$B" -v N="$N" '
      /Training: +[0-9]+%/ {
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^[0-9]+\/[0-9]+/) {
            split($i, a, "/")
            step = a[1] + 0
            total = a[2] + 0
            if (total > 0 && step != last_step && (step % 20 == 0 || step <= 5)) {
              pct = (step / total) * 100
              printf "%s[%5.1f%%]%s %s\n", C, pct, N, $0
              fflush()
              last_step = step
            }
            next
          }
        }
        next
      }
      /[Tt]raceback|RuntimeError|CUDA out of memory|Killed|SIGTERM|Errno|FAILED/ {
        printf "%s%s%s\n", R, $0, N; fflush(); next
      }
      /\[WARN\]|UserWarning|FutureWarning|DeprecationWarning/ {
        printf "%s%s%s\n", Y, $0, N; fflush(); next
      }
      /step:[0-9]+|smpl:|loss:|saved|checkpoint|complete|Finished|done/ {
        printf "%s%s%s\n", G, $0, N; fflush(); next
      }
      /^INFO/ { print; fflush(); next }
    '
