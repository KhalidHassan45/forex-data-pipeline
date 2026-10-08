#!/usr/bin/env python3
"""Automated look-ahead test for the feature store — a feature is guilty until proven causal.

Truncation test: compute features on a window, then again on the same window cut at several points.
A causal feature has IDENTICAL values on the shared bars; anything that peeks (shift(-1), centred windows,
full-sample normalisation, resampling that labels by the bar end, ...) changes and is flagged.
Uses research-window bars only (never the vault).

    python leakage.py              # certify current feature version → /data/ml/leakage_cert.json
    python leakage.py --selftest   # plant two known leaks and prove the test catches them
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
import features as F  # noqa: E402

WINDOW, WARMUP, CUTS = 3000, 800, (0.55, 0.75, 0.92)


def leaky_columns(D, pair, compute=F.compute, seed=0):
    df = D[pair]
    if len(df) < WINDOW + 10:
        raise SystemExit(f"{pair}: not enough research data for the leakage test")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(df) - WINDOW, size=2)
    bad = set()
    for s0 in starts:
        sl = df.iloc[s0:s0 + WINDOW]
        t0, t1 = sl.index[0], sl.index[-1]
        Dw = {p: d[(d.index >= t0) & (d.index <= t1)] for p, d in D.items()}
        full = compute(sl, Dw, pair)
        for cut in CUTS:
            ct = sl.index[int(len(sl) * cut)]
            Dc = {p: d[d.index <= ct] for p, d in Dw.items()}
            part = compute(sl[sl.index <= ct], Dc, pair)
            a = full.loc[part.index].iloc[WARMUP:]
            b = part.iloc[WARMUP:]
            for col in full.columns:
                if col not in b:
                    bad.add(col); continue
                x, y = a[col].to_numpy(float), b[col].to_numpy(float)
                both_nan = np.isnan(x) & np.isnan(y)
                close = np.isclose(x, y, rtol=1e-4, atol=1e-6) | both_nan
                if not close.all():
                    bad.add(col)
    return sorted(bad)


def certify(tf="h1", pairs=None):
    D = C.load_all(tf, pairs)
    vs = C.vault_start(D)
    D = {p: d[d.index < vs] for p, d in D.items()}           # research window only
    test_pairs = sorted(D)[:3] if not pairs else sorted(D)
    leaks = {}
    for p in test_pairs:
        leaks[p] = leaky_columns(D, p)
    allbad = sorted({c for v in leaks.values() for c in v})
    cert = {"version": F.version(), "tf": tf, "ts": C.now(), "pairs_tested": test_pairs,
            "status": "PASS" if not allbad else "PARTIAL", "leaky_columns": allbad,
            "note": "leaky columns are excluded from every training run automatically"}
    C.ML_DIR.mkdir(parents=True, exist_ok=True)
    allc = json.loads(C.CERT_FILE.read_text()) if C.CERT_FILE.exists() else {}
    allc[f"{F.version()}|{tf}"] = cert
    C.CERT_FILE.write_text(json.dumps(allc, indent=2))
    return cert


def get_cert(tf):
    if not C.CERT_FILE.exists():
        return None
    return json.loads(C.CERT_FILE.read_text()).get(f"{F.version()}|{tf}")


def selftest(tf="h1"):
    D = C.load_all(tf)
    vs = C.vault_start(D)
    D = {p: d[d.index < vs] for p, d in D.items()}
    pair = sorted(D)[0]

    def planted(df, D_, p):
        X = F.compute(df, D_, p)
        X["LEAK_next_close"] = (df["close"].shift(-1) / df["close"] - 1).astype("float32")      # peeks 1 bar
        X["LEAK_fullsample_z"] = ((df["close"] - df["close"].mean()) / df["close"].std()).astype("float32")
        X["OK_lagged"] = (df["close"].shift(1) / df["close"].shift(2) - 1).astype("float32")
        return X

    bad = leaky_columns(D, pair, compute=planted)
    caught = {"LEAK_next_close", "LEAK_fullsample_z"} <= set(bad)
    clean = "OK_lagged" not in bad and not [b for b in bad if not b.startswith("LEAK_")]
    print(json.dumps({"pair": pair, "flagged": bad, "caught_planted_leaks": caught,
                      "no_false_alarms_on_real_features": clean}, indent=2))
    return 0 if caught and clean else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="h1")
    ap.add_argument("--pairs", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest(a.tf))
    c = certify(a.tf, [p.upper() for p in a.pairs] if a.pairs else None)
    print(json.dumps(c, indent=2))
    sys.exit(0)
