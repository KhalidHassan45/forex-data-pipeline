#!/usr/bin/env bash
# ml-lab wrapper inside forex-worker.
#   ml.sh build                     → feature store + default labels for all pairs
#   ml.sh leakcheck [--selftest]    → certify current feature code (required before training)
#   ml.sh train --pairs EURUSD --why "..." [train.py args]
#   ml.sh list                      → CANDIDATE models awaiting the vault
#   ml.sh vault --id "ml|..." --approved-by Khalid
#   ml.sh paper approve --id "ml|..." --approved-by Khalid | paper monitor | paper status
#   ml.sh latest                    → last training batch summary
set -uo pipefail
DATA_DIR="${DATA_DIR:-/data}"; M=/app/ml
[[ -d $M ]] || M="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DATA_DIR/logs" "$DATA_DIR/ml"
CMD="${1:-}"; shift || true
case "$CMD" in
  list)   exec python $M/vault.py --list ;;
  latest) B=$(cat "$DATA_DIR/ml/LATEST" 2>/dev/null) && cat "$B/summary.json" || echo "no training yet"; exit 0 ;;
  build|leakcheck|train|vault|paper) ;;
  *) echo "usage: ml.sh build|leakcheck|train|list|vault|paper|latest"; exit 2 ;;
esac
exec 7>"$DATA_DIR/.ml.lock"; flock -n 7 || { echo "ml-lab already running"; exit 0; }
exec 9>"$DATA_DIR/.forex.lock"; flock -w 3600 9 || { echo "data pipeline busy >1h"; exit 1; }; flock -u 9
LOG="$DATA_DIR/logs/ml_${CMD}_$(date -u +%Y%m%d_%H%M%S).log"
case "$CMD" in
  build)     MLDIR=$M python - "$@" <<'PY' 2>&1 | tee "$LOG"; RC=${PIPESTATUS[0]} ;;
import os, sys; sys.path.insert(0, os.environ["MLDIR"])
import common as C, features as F, labels as LB
D = C.load_all("h1"); C.vault_start(D)
print("features:", {p: str(f) for p, f in F.build(D, "h1", force="--force" in sys.argv).items()})
print("labels:", {p: str(f) for p, f in LB.build(D, "h1").items()})
PY
  leakcheck) python $M/leakage.py "$@" 2>&1 | tee "$LOG"; RC=${PIPESTATUS[0]} ;;
  train)     nice -n 10 python $M/train.py "$@" 2>&1 | tee "$LOG"; RC=${PIPESTATUS[0]} ;;
  vault)     python $M/vault.py "$@" 2>&1 | tee "$LOG"; RC=${PIPESTATUS[0]} ;;
  paper)     python $M/paper.py "$@" 2>&1 | tee "$LOG"; RC=${PIPESTATUS[0]} ;;
esac
if [[ -n "${TELEGRAM_BOT_TOKEN:-}" && -n "${TELEGRAM_CHAT_ID:-}" && "$CMD" =~ ^(train|vault|paper)$ ]]; then
  MSG=$(CMD="$CMD" LOG="$LOG" python - <<'PY'
import json, os
cmd, d = os.environ["CMD"], os.environ.get("DATA_DIR", "/data")
try:
    if cmd == "train":
        s = json.load(open(os.path.join(open(os.path.join(d, "ml", "LATEST")).read().strip(), "summary.json")))
        L = [f"🤖 تدريب ML: {len(s['results'])} نموذج"]
        L += [f"• {r['pair']} — {'✅ مرشّح' if r['status']=='CANDIDATE' else '❌ مرفوض'} | SR={r['net_sharpe']} DSR={r['dsr']} null+{r['edge_over_null']}" for r in s["results"]]
        if any(r["status"] == "CANDIDATE" for r in s["results"]): L.append("المرشّح يحتاج موافقتك لاختبار الخزنة.")
    elif cmd == "vault":
        L = ["🔐 خزنة ML: " + open(os.environ["LOG"]).read()[-600:]]
    else:
        st = json.load(open(os.path.join(d, "ml", "paper", "status.json")))
        L = ["📒 Paper (ظلّي، بلا وسيط):"]
        for m in st["models"]:
            g = m.get("ledger", {})
            L.append(f"• {m['pair']} {m['status']} | صفقات={g.get('net_trades',0)} PF={g.get('net_profit_factor','-')} DD={g.get('net_max_dd_pct','-')}%"
                     + (f" ⚠️ {m['alert']}" if m.get("alert") else "") + (" 🔁 انحراف" if m.get("retrain_suggested") else ""))
    print("\n".join(L))
except Exception as e:
    print(f"🤖 ml-lab {cmd} finished ({e})")
PY
)
  curl -s -m 15 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=${MSG}" >/dev/null || true
fi
[[ -x /app/dashboard/build.sh ]] && /app/dashboard/build.sh
exit $RC
