#!/usr/bin/env bash
# Wrapper for the research cycle inside forex-worker.
#   research.sh scan [scan.py args]     → pattern scan (waits for the data pipeline lock)
#   research.sh list                    → candidates awaiting the vault
#   research.sh latest                  → latest scan summary.json
set -uo pipefail
DATA_DIR="${DATA_DIR:-/data}"; R=/app/research
mkdir -p "$DATA_DIR/logs" "$DATA_DIR/research"
CMD="${1:-scan}"; shift || true
case "$CMD" in
  list)   exec python $R/vault.py --list ;;
  latest) D=$(cat "$DATA_DIR/research/LATEST" 2>/dev/null) && cat "$D/summary.json" || echo "no scan yet"; exit 0 ;;
  scan)   ;;
  *) echo "usage: research.sh scan|list|latest"; exit 2 ;;
esac
exec 7>"$DATA_DIR/.research.lock"; flock -n 7 || { echo "research already running"; exit 0; }
exec 9>"$DATA_DIR/.forex.lock"; flock -w 3600 9 || { echo "data pipeline busy >1h"; exit 1; }; flock -u 9
LOG="$DATA_DIR/logs/research_$(date -u +%Y%m%d_%H%M%S).log"
python $R/scan.py "$@" 2>&1 | tee "$LOG"; RC=${PIPESTATUS[0]}
if [[ -n "${TELEGRAM_BOT_TOKEN:-}" && -n "${TELEGRAM_CHAT_ID:-}" ]]; then
  MSG=$(python - <<'PY'
import json, os
try:
    d = open(os.path.join(os.environ.get("DATA_DIR", "/data"), "research", "LATEST")).read().strip()
    s = json.load(open(os.path.join(d, "summary.json")))
    L = [f"🔬 مسح الأنماط: {s['tested_this_run']} فرضية (إجمالي {s['lifetime_tests']})",
         f"صدفة متوقعة بلا تصحيح: {s['expected_false_positives_at_5pct_without_correction']} | مرشّحة: {len(s['candidates'])}"]
    L += [f"• {c['id']} — t={c['t']} ثبات={c['stability']}" for c in s["candidates"][:6]]
    if s["candidates"]: L.append("بانتظار موافقتك لاختبار الخزنة.")
    print("\n".join(L))
except Exception as e:
    print(f"🔬 research run finished (no summary: {e})")
PY
)
  curl -s -m 15 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=${MSG}" >/dev/null || true
fi
exit $RC
