#!/usr/bin/env bash
# Acceptance test on synthetic data with known truth. ~8 minutes on 1 CPU. Never touches /data.
#   bash tests/run_tests.sh            (from the ml-lab folder, or /app/ml/tests inside the worker)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; M="$(cd "$HERE/../ml" 2>/dev/null && pwd || echo /app/ml)"
export DATA_DIR="$(mktemp -d)"; trap 'rm -rf "$DATA_DIR"' EXIT
python "$HERE/make_synthetic.py" "$DATA_DIR" >/dev/null
cd "$M"
ok(){ echo "✔ $1"; }; bad(){ echo "✘ $1"; exit 1; }
python leakage.py --selftest >/dev/null && ok "leakage test catches planted look-ahead, no false alarms" || bad "leakage selftest"
python leakage.py >/dev/null
python train.py --pairs EURUSD GBPUSD USDJPY AUDUSD --why "synthetic: London reversal after 24h selloff" > "$DATA_DIR/t.log"
st(){ python -c "import json,sys;[print(r['status']) for r in map(json.loads,open('$DATA_DIR/ml/registry.jsonl')) if r.get('event')=='TRAIN' and r['pair']=='$1']"; }
[[ $(st EURUSD) == CANDIDATE ]] && ok "real edge (EURUSD) → CANDIDATE" || bad "EURUSD should be CANDIDATE"
[[ $(st GBPUSD) == CANDIDATE ]] && ok "edge alive in research (GBPUSD) → CANDIDATE" || bad "GBPUSD should be CANDIDATE"
[[ $(st USDJPY) == REJECTED && $(st AUDUSD) == REJECTED ]] && ok "pure noise → REJECTED" || bad "noise should be REJECTED"
id(){ python -c "import json;print([r['id'] for r in map(json.loads,open('$DATA_DIR/ml/registry.jsonl')) if r.get('event')=='TRAIN' and r['pair']=='$1'][-1])"; }
OUT=$(python vault.py --id "$(id EURUSD)" --approved-by test 2>&1 || true); grep -q VAULT_PASS <<<"$OUT" && ok "real edge passes the vault" || bad "EURUSD vault"
OUT=$(python vault.py --id "$(id GBPUSD)" --approved-by test 2>&1 || true); grep -q VAULT_FAIL <<<"$OUT" && ok "edge that died fails the vault" || bad "GBPUSD vault"
OUT=$(python vault.py --id "$(id EURUSD)" --approved-by test 2>&1 || true); grep -q "already used" <<<"$OUT" && ok "second vault attempt refused" || bad "vault reuse"
OUT=$(python vault.py --id "$(id USDJPY)" --approved-by test 2>&1 || true); grep -q "not a CANDIDATE" <<<"$OUT" && ok "rejected model cannot open the vault" || bad "vault gate"
OUT=$(python paper.py approve --id "$(id GBPUSD)" --approved-by test 2>&1 || true); grep -q refused <<<"$OUT" && ok "vault-failed model cannot paper trade" || bad "paper gate"
OUT=$(python paper.py approve --id "$(id EURUSD)" --approved-by test 2>&1 || true); grep -q PAPER_ACTIVE <<<"$OUT" && ok "paper approval works" || bad "paper approve"
python paper.py monitor >/dev/null && ok "monitor runs (ledger + drift)" || bad "monitor"
echo "ALL ACCEPTANCE TESTS PASSED"
