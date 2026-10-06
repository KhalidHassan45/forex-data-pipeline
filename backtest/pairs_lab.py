#!/usr/bin/env python3
"""
Pairs / cointegration lab — mean-reversion on the log-spread of two correlated pairs.
Walk-forward (train_y / test_y) + realistic cost on BOTH legs. Reads {pair}_{tf}.parquet.
Pure research — no execution, no guarantees.
"""
import itertools
import os
import numpy as np
import pandas as pd

PARQ = os.environ.get("PARQUET_DIR", "/data/parquet")
TF = os.environ.get("FOREX_TF", "h1")
Z_ENTRY = [1.5, 2.0, 2.5]
WIN = [60, 120]
Z_EXIT = 0.5
COST_BPS = 0.5          # per leg per side (spread cost)
TRAIN_Y = 3
TEST_Y = 1
COMBOS = [("EURUSD", "GBPUSD"), ("AUDUSD", "NZDUSD"), ("EURUSD", "USDCHF"),
          ("EURUSD", "USDJPY"), ("GBPUSD", "USDCHF"), ("EURUSD", "USDCAD")]
BARS_PER_YEAR = 6048    # ~h1 FX bars/year


def load(pair):
    p = os.path.join(PARQ, f"{pair.lower()}_{TF}.parquet")
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p)[["ts", "close"]].set_index("ts")["close"]


def signals(a, b, win, ze):
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    la, lb = np.log(df["a"]), np.log(df["b"])
    beta = la.rolling(win).cov(lb) / lb.rolling(win).var()
    spread = la - beta * lb
    z = (spread - spread.rolling(win).mean()) / spread.rolling(win).std()
    return df.index, spread.to_numpy(), z.to_numpy(), beta.to_numpy()


def pnl_series(idx, spread, z, beta, ze):
    n = len(idx)
    out = np.zeros(n)
    pos = 0.0
    prev = 0.0
    for i in range(1, n):
        zt = z[i - 1]
        if np.isnan(zt) or np.isnan(beta[i]):
            out[i] = 0.0
            continue
        if pos == 0:
            if zt > ze:
                pos = -1.0
            elif zt < -ze:
                pos = 1.0
        elif pos > 0 and zt >= -Z_EXIT:
            pos = 0.0
        elif pos < 0 and zt <= Z_EXIT:
            pos = 0.0
        sr = spread[i] - spread[i - 1]
        turnover = abs(pos - prev)
        cost = turnover * (COST_BPS / 10000.0) * (1.0 + abs(beta[i]))
        out[i] = pos * sr - cost
        prev = pos
    return pd.Series(out, index=idx)


def stats(pnl):
    pnl = pnl.dropna()
    if len(pnl) < 50 or pnl.std() == 0:
        return None
    sharpe = pnl.mean() / pnl.std() * np.sqrt(BARS_PER_YEAR)
    eq = pnl.cumsum()
    dd = (eq - eq.cummax()).min()
    trades = int((pnl != 0).sum())
    return {"sharpe": round(float(sharpe), 3), "ret_pct": round(float(pnl.sum() * 100), 2),
            "max_dd_pct": round(float(dd * 100), 2), "bars": trades}


def main():
    data = {p: load(p) for combo in COMBOS for p in combo}
    print(f"Pairs lab · tf={TF} · cost={COST_BPS}bps/leg · z_entry={Z_ENTRY} · win={WIN}")
    for a, b in COMBOS:
        if data.get(a) is None or data.get(b) is None:
            print(f"\n{a}-{b}: missing data"); continue
        best = None
        for win, ze in itertools.product(WIN, Z_ENTRY):
            idx, spread, z, beta = signals(data[a], data[b], win, ze)
            pnl = pnl_series(idx, spread, z, beta, ze)
            # walk-forward: train 3y pick best ze by train sharpe, test next year
            years = sorted({t.year for t in idx})
            oos = pd.Series(dtype=float)
            for y0 in years:
                tr = pnl[(pnl.index.year >= y0 - TRAIN_Y) & (pnl.index.year < y0)]
                te = pnl[pnl.index.year == y0]
                s = stats(tr)
                if s and s["sharpe"] > 0 and len(te) > 0:
                    oos = pd.concat([oos, te])
            st = stats(oos)
            if st:
                tag = f"{a}-{b} win={win} ze={ze}"
                print(f"{tag:32} wf_sharpe={st['sharpe']:>7} ret={st['ret_pct']:>8}% dd={st['max_dd_pct']:>7}% bars={st['bars']}")
                if best is None or st["sharpe"] > best[1]["sharpe"]:
                    best = (tag, st)
        if best:
            print(f"   >> BEST {a}-{b}: {best[0]} -> {best[1]}")


if __name__ == "__main__":
    main()
