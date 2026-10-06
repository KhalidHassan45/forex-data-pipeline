#!/usr/bin/env python3
"""
Deep validation of the EURUSD-USDCHF pairs ensemble.
- data quality: per-year bar counts (detects short/gapped history)
- stability    : per-year Sharpe, positive-year fraction, rolling 2y Sharpe
- 2nd split    : first-half vs second-half
Research only.
"""
import os
import numpy as np
import pandas as pd

PARQ = os.environ.get("PARQUET_DIR", "/data/parquet")
TF = os.environ.get("FOREX_TF", "h1")
WINS = [30, 60, 120]
ZES = [1.5, 2.0, 2.5]
Z_EXIT = 0.5
COST = 0.5 / 10000.0
BPY = 6048


def load(p):
    f = os.path.join(PARQ, f"{p.lower()}_{TF}.parquet")
    return pd.read_parquet(f)[["ts", "close"]].set_index("ts") if os.path.exists(f) else None


def zs(df, win):
    la, lb = np.log(df["a"]), np.log(df["b"])
    beta = la.rolling(win).cov(lb) / lb.rolling(win).var()
    sp = la - beta * lb
    z = (sp - sp.rolling(win).mean()) / sp.rolling(win).std()
    return sp.to_numpy(), z.to_numpy(), beta.to_numpy()


def pos_of(z, ze):
    p = 0.0
    out = np.zeros(len(z))
    for i in range(1, len(z)):
        zt = z[i - 1]
        if np.isnan(zt):
            out[i] = p
            continue
        if p == 0:
            p = -1.0 if zt > ze else 1.0 if zt < -ze else 0.0
        elif p > 0 and zt >= -Z_EXIT:
            p = 0.0
        elif p < 0 and zt <= Z_EXIT:
            p = 0.0
        out[i] = p
    return out


def sh(x):
    x = np.asarray(x)[~np.isnan(np.asarray(x, dtype=float))]
    if len(x) < 50 or x.std() == 0:
        return None
    return float(x.mean() / x.std() * np.sqrt(BPY))


def ensemble(a_ser, b_ser):
    df = pd.concat({"a": a_ser, "b": b_ser}, axis=1).dropna()
    idx = df.index
    psum = np.zeros(len(df))
    for w in WINS:
        sp, z, beta = zs(df, w)
        for ze in ZES:
            psum += pos_of(z, ze)
    er = psum / (len(WINS) * len(ZES))
    ep = np.zeros(len(er))
    h = 0.0
    for i in range(len(er)):
        if er[i] > 0.5:
            h = 1.0
        elif er[i] < -0.5:
            h = -1.0
        elif h > 0 and er[i] < 0.1:
            h = 0.0
        elif h < 0 and er[i] > -0.1:
            h = 0.0
        ep[i] = h
    sp, z, beta = zs(df, 60)
    pnl = ep * np.diff(sp, prepend=sp[0]) - np.abs(np.diff(ep, prepend=0.0)) * COST * (1 + np.abs(beta))
    return idx, pd.Series(pnl, index=idx)


def main():
    a, b = load("EURUSD"), load("USDCHF")
    print("=== DATA QUALITY ===")
    for nm, d in [("EURUSD", a), ("USDCHF", b)]:
        if d is None:
            print(f"{nm}: MISSING"); continue
        yr = d.groupby(d.index.year).size()
        print(f"{nm}: {len(d)} bars {d.index.min().date()} -> {d.index.max().date()}")
        print(f"   bars/year: {dict((int(k), int(v)) for k, v in yr.items())}")
    if a is None or b is None:
        return
    idx, pnl = ensemble(a["close"], b["close"])
    print("\n=== ENSEMBLE (EURUSD-USDCHF) ===")
    print("overall sharpe:", round(sh(pnl.to_numpy()), 3))
    ys = {int(y): (round(sh(pnl[pnl.index.year == y].to_numpy()), 2)) for y in sorted(set(idx.year))}
    print("per-year sharpe:", ys)
    pos_y = sum(1 for v in ys.values() if v and v > 0)
    print(f"positive years: {pos_y}/{len(ys)}")
    for label, lo, hi in [("2003-2014", "2003", "2014"), ("2014-2025", "2014", "2026")]:
        seg = pnl[(pnl.index.year >= int(lo)) & (pnl.index.year < int(hi))]
        print(f"split {label}: sharpe={round(sh(seg.to_numpy()), 3) if seg.size else None} n={seg.size}")
    roll = pnl.rolling(2 * BPY).mean() / pnl.rolling(2 * BPY).std() * np.sqrt(BPY)
    print("rolling2y sharpe min/med/max:", round(float(np.nanmin(roll)), 2),
          round(float(np.nanmedian(roll)), 2), round(float(np.nanmax(roll)), 2))


if __name__ == "__main__":
    main()
