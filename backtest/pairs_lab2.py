#!/usr/bin/env python3
"""
Pairs lab v2 — ROBUSTNESS focus. Addresses parameter sensitivity 3 ways:
  (a) plateau   : fraction of the (win,z) grid that is positive OOS (not one lucky cell)
  (b) ensemble  : average position across the WHOLE grid -> one robust series (no param pick)
  (c) cost stress: ensemble must survive at 1.5x cost
Reads {pair}_{tf}.parquet. Research only.
"""
import itertools
import os
import numpy as np
import pandas as pd

PARQ = os.environ.get("PARQUET_DIR", "/data/parquet")
TF = os.environ.get("FOREX_TF", "h1")
WINS = [30, 60, 120]
ZES = [1.5, 2.0, 2.5]
Z_EXIT = 0.5
COST = 0.5 / 10000.0
WARM_Y = 3          # years skipped for warm-up / OOS start
BPY = 6048          # ~h1 FX bars/year
COMBOS = [("EURUSD", "GBPUSD"), ("EURUSD", "USDCHF"), ("EURUSD", "USDJPY"),
          ("EURUSD", "USDCAD"), ("GBPUSD", "USDCHF"), ("AUDUSD", "NZDUSD"),
          ("AUDUSD", "EURUSD"), ("NZDUSD", "EURUSD"), ("EURJPY", "USDJPY"),
          ("EURGBP", "EURUSD")]


def load(p):
    f = os.path.join(PARQ, f"{p.lower()}_{TF}.parquet")
    return pd.read_parquet(f)[["ts", "close"]].set_index("ts")["close"] if os.path.exists(f) else None


def zseries(df, win):
    la, lb = np.log(df["a"]), np.log(df["b"])
    beta = la.rolling(win).cov(lb) / lb.rolling(win).var()
    spread = la - beta * lb
    z = (spread - spread.rolling(win).mean()) / spread.rolling(win).std()
    return spread.to_numpy(), z.to_numpy(), beta.to_numpy()


def positions(z, ze):
    pos = np.zeros(len(z))
    p = 0.0
    for i in range(1, len(z)):
        zt = z[i - 1]
        if np.isnan(zt):
            pos[i] = p
            continue
        if p == 0:
            p = -1.0 if zt > ze else 1.0 if zt < -ze else 0.0
        elif p > 0 and zt >= -Z_EXIT:
            p = 0.0
        elif p < 0 and zt <= Z_EXIT:
            p = 0.0
        pos[i] = p
    return pos


def pnl(spread, pos, beta, cost=COST):
    sr = np.diff(spread, prepend=spread[0])
    turn = np.abs(np.diff(pos, prepend=0.0))
    return pos * sr - turn * cost * (1.0 + np.abs(beta))


def sharpe(x):
    x = x[~np.isnan(x)]
    if len(x) < 50 or x.std() == 0:
        return None
    return float(x.mean() / x.std() * np.sqrt(BPY))


def main():
    print(f"Pairs lab v2 · tf={TF} · wins={WINS} zes={ZES} · cost={COST*10000:.1f}bps/leg")
    for a, b in COMBOS:
        da, db = load(a), load(b)
        if da is None or db is None:
            print(f"\n{a}-{b}: MISSING"); continue
        df = pd.concat({"a": da, "b": db}, axis=1).dropna()
        idx = df.index
        warm = idx >= idx[0] + pd.Timedelta(days=365 * WARM_Y)
        pos_sum = np.zeros(len(df))
        grid_pos = 0
        grid_n = 0
        best = None
        for win, ze in itertools.product(WINS, ZES):
            spread, z, beta = zseries(df, win)
            ps = positions(z, ze)
            base = ps * np.diff(spread, prepend=spread[0]) - np.abs(np.diff(ps, prepend=0.0)) * COST * (1 + np.abs(beta))
            s_ = sharpe(base[warm])
            grid_n += 1
            if s_ and s_ > 0:
                grid_pos += 1
            if s_ and (best is None or s_ > best[0]):
                best = (s_, win, ze)
            pos_sum += ps
        ens_pos = pos_sum / (len(WINS) * len(ZES))
        spread, z, beta = zseries(df, 60)
        ens = ens_pos * np.diff(spread, prepend=spread[0]) - np.abs(np.diff(ens_pos, prepend=0.0)) * COST * (1 + np.abs(beta))
        ens_s = sharpe(ens[warm])
        ens_s15 = sharpe(ens_pos * np.diff(spread, prepend=spread[0]) - np.abs(np.diff(ens_pos, prepend=0.0)) * COST * 1.5 * (1 + np.abs(beta))) if ens_s is not None else None
        plat = f"{grid_pos}/{grid_n}"
        bs = f"{best[0]:+.2f}@w{best[1]},z{best[2]}" if best else "none"
        print(f"\n{a}-{b}: n={len(df)}")
        print(f"   plateau(positive cells) = {plat} | best cell = {bs}")
        print(f"   ENSEMBLE oos_sharpe = {ens_s} | @1.5x cost = {ens_s15}")


if __name__ == "__main__":
    main()
