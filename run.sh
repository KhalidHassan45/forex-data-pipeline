#!/usr/bin/env bash
# Wrapper used by Coolify Scheduled Tasks and manual runs.
#   run.sh backfill [extra args]   → full history (first run)
#   run.sh update   [extra args]   → daily incremental
#   run.sh status                  → print last_run.json
# Prevents overlapping runs (flock), logs to $DATA_DIR/logs, optional Telegram alert.
set -uo pipefail

DATA_DIR="${DATA_DIR:-/data}"
LOG_DIR="$DATA_DIR/logs"
mkdir -p "$LOG_DIR"
MODE="${1:-update}"; shift || true

if [[ "$MODE" == "status" ]]; then
  cat "$LOG_DIR/last_run.json" 2>/dev/null || echo "no run yet"
  exit 0
fi

LOG="$LOG_DIR/${MODE}_$(date -u +%Y%m%d_%H%M%S).log"

notify() {
  [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]] && return 0
  curl -s -m 15 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
    --data-urlencode "text=$1" >/dev/null || true
}

exec 9>"$DATA_DIR/.forex.lock"
if ! flock -n 9; then
  echo "another run is in progress — exiting" | tee -a "$LOG"
  exit 0
fi

START=$(date +%s)
python /app/fetch_forex.py "$MODE" "$@" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
MIN=$(( ($(date +%s) - START) / 60 ))

# keep the last 30 logs only
ls -1t "$LOG_DIR"/*.log 2>/dev/null | tail -n +31 | xargs -r rm -f

SUMMARY=$(python - <<'PY'
import json, os
p = os.path.join(os.environ.get("DATA_DIR", "/data"), "logs", "last_run.json")
try:
    r = json.load(open(p))
    ok = len([p for p in r["pairs"] if p not in r["failed"]]); rows = sum(v.get("rows", 0) for v in r["pairs"].values())
    ins = sum(v.get("upserted", 0) for v in r["pairs"].values())
    print(f"pairs ok={ok} rows={rows:,} upserted={ins:,} failed={r['failed'] or '-'}")
except Exception as e:
    print(f"no summary ({e})")
PY
)

if [[ $RC -eq 0 ]]; then
  notify "✅ Forex ${MODE} done in ${MIN}m — ${SUMMARY}"
else
  notify "⚠️ Forex ${MODE} rc=${RC} after ${MIN}m — ${SUMMARY} — log: ${LOG}"
fi
exit $RC
