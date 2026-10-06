#!/usr/bin/env python3
"""
Deep pairs analysis on AVAILABLE data:
  - Cointegration ADF test on the log-spread (is it stationary?)
  - Half-life of mean reversion (OU) -> informs the z window
  - Kalman-filter hedge ratio (dynamic beta) -> spread -> z-reversion ensemble
  - Ensemble OOS Sharpe (3y warm-up) + 1.5x cost stress
Research only, no execution.
"""
import os
import numpy as np
import pandas as pd

PARQ = os.environ.get("PARQUET_DIR", "/data/parquet")
TF = os.environ.get("FOREX_TF", "h1")
WINS = [60, 120]
ZES = [1.5, 2.0, 2.5]
Z_EXIT = 0.5
COST = 0.5 / 10000.0
WARM_Y = 3
BPY = 6048
ADF_CRIT_5 = -2.86
COMBOS = [("EURUSD", "GBPUSD"), ("EURUSD", "USDCHF"), ("EURUSD", "USDJPY"),
          ("EURUSD", "USDCAD"), ("GBPUSD", "USDCHF"), ("AUDUSD", "NZDUSD"),
          ("AUDUSD", "EURUSD"), ("NZDUSD", "EURUSD"), ("EURJPY", "USDJPY"),
          ("EURGBP", "EURUSD")]


def load(p):
    f = os.path.join(PARQ, f"{p.lower()}_{TF}.parquet")
    return pd.read_parquet(f)[["ts", "close"]].set_index("ts")["close"] if os.path.exists(f) else None


def adf_tstat(y, maxlag=4):
    y = np.asarray(y, float)
    dy = np.diff(y)
    n = len(dy)
    cols = [np.ones(n), y[:-1]]
    for i in range(1, maxlag + 1):
        cols.append(np.concatenate([np.full(i, np.nan), dy[:-i]]))
    X = np.column_stack(cols)
    mask = ~np.isnan(X).any(axis=1)
    Xm, ym = X[mask], dy[mask]
    beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    resid = ym - Xm @ beta
    dof = len(ym) - Xm.shape[1]
    s2 = float((resid ** 2).sum() / dof)
    se = np.sqrt(s2 * np.linalg.inv(Xm.T @ Xm)[1, 1])
    return float(beta[1] / se)


def half_life(spread):
    y = np.asarray(spread, float)
    dy = np.diff(y)
    yl = y[:-1]
    mask = ~np.isnan(yl) & ~np.isnan(dy)
    X = np.column_stack([np.ones(int(mask.sum())), yl[mask]])
    beta, *_ = np.linalg.lstsq(X, dy[mask], rcond=None)
    lam = float(beta[1])
    return float(-np.log(2) / lam) if lam < 0 else float("inf")


def kalman_beta(a, b, q=1e-5, r=1e-3):
    n = len(a)
    out = np.zeros(n)
    bet, P = 0.0, 1.0
    for i in range(n):
        P += q
        ai, bi = a[i], b[i]
        if not np.isnan(ai) and not np.isnan(bi) and bi != 0:
            K = P * bi / (bi * bi * P + r)
            bet += K * (ai - bet * bi)
            P *= (1 - K * bi)
        out[i] = bet
    return out


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


def sharpe(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 50 or x.std() == 0:
        return None
    return round(float(x.mean() / x.std() * np.sqrt(BPY)), 3)


def main():
    print(f"Deep pairs · tf={TF} · cost={COST * 10000:.1f}bps/leg · Kalman hedge")
    for a, b in COMBOS:
        da, db = load(a), load(b)
        if da is None or db is None:
            print(f"\n{a}-{b}: MISSING"); continue
        df = pd.concat({"a": da, "b": db}, axis=1).dropna()
        idx = df.index
        la, lb = np.log(df["a"]), np.log(df["b"])
        beta_ols = float(la.cov(lb) / lb.var())
        sp_ols = (la - beta_ols * lb).to_numpy()[-30000:]
        t = adf_tstat(sp_ols)
        hl = half_life(sp_ols)
        kb = kalman_beta(la.to_numpy(), lb.to_numpy())
        sp = la.to_numpy() - kb * lb.to_numpy()
        sp_s = pd.Series(sp, index=idx)
        psum = np.zeros(len(df))
        for w in WINS:
            z = ((sp_s - sp_s.rolling(w).mean()) / sp_s.rolling(w).std()).to_numpy()
            for ze in ZES:
                psum += positions(z, ze)
        er = psum / (len(WINS) * len(ZES))
        ep = np.zeros(len(er)); h = 0.0
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
        sr = np.diff(sp, prepend=sp[0])
        turn = np.abs(np.diff(ep, prepend=0.0))
        warm = idx >= idx[0] + pd.Timedelta(days=365 * WARM_Y)
        ens = sharpe((ep * sr - turn * COST * (1 + np.abs(kb)))[warm])
        ens15 = sharpe((ep * sr - turn * COST * 1.5 * (1 + np.abs(kb)))[warm])
        coint = "YES" if t < ADF_CRIT_5 else "no"
        print(f"\n{a}-{b}: n={len(df)}  ADF t={t:.2f} ({coint} @5%)  half-life={hl:.0f} bars")
        print(f"   Kalman-hedge ensemble OOS sharpe = {ens}  |  @1.5x cost = {ens15}")


if __name__ == "__main__":
    main()
