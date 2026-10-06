#!/usr/bin/env bash
# Wrapper: run the strategy lab inside forex-worker, no overlap, log + optional Telegram digest.
#   run_lab.sh [run_backtest.py args...]
#   run_lab.sh latest        → print the latest summary.json
set -uo pipefail
DATA_DIR="${DATA_DIR:-/data}"
mkdir -p "$DATA_DIR/logs" "$DATA_DIR/reports"

if [[ "${1:-}" == "latest" ]]; then
  D=$(cat "$DATA_DIR/reports/LATEST" 2>/dev/null) && cat "$D/summary.json" || echo "no report yet"
  exit 0
fi

exec 8>"$DATA_DIR/.lab.lock"
flock -n 8 || { echo "a backtest is already running"; exit 0; }
# never run while the data pipeline is writing parquet files
exec 9>"$DATA_DIR/.forex.lock"
flock -w 3600 9 || { echo "data pipeline busy for >1h — aborting"; exit 1; }
flock -u 9

LOG="$DATA_DIR/logs/lab_$(date -u +%Y%m%d_%H%M%S).log"
python /app/backtest/run_backtest.py "$@" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}

if [[ -n "${TELEGRAM_BOT_TOKEN:-}" && -n "${TELEGRAM_CHAT_ID:-}" ]]; then
  MSG=$(python - <<'PY'
import json, os
d = open(os.path.join(os.environ.get("DATA_DIR", "/data"), "reports", "LATEST")).read().strip()
s = json.load(open(os.path.join(d, "summary.json")))
lines = [f"🧪 Strategy lab ({s['tf']}) — {len(s['passed'])} PASS من {s['total_combos']} توليفة"]
for p in s["passed"][:8]:
    lines.append(f"• {p['strategy']} {p['pair']}: Sharpe {p['wf_sharpe']} | DD {p['wf_max_dd_pct']}% | PF {p['wf_profit_factor']}")
lines.append(f"فجوة التفاؤل IS−WF: {s.get('avg_is_minus_wf_sharpe')}")
print("\n".join(lines))
PY
)
  curl -s -m 15 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=${MSG}" >/dev/null || true
fi
exit $RC
