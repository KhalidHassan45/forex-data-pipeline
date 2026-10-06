"""
Classic, transparent strategies. Each returns the DESIRED position decided at the
close of each bar: +1 long, -1 short, 0 flat. The engine executes it on the next bar,
so a strategy may use the current bar's close without look-ahead.

All indicators are causal (rolling / shifted) — never use .shift(-n) here.
"""
from itertools import product

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


# ---------------------------------------------------------------- trend following
def sma_cross(df, fast=20, slow=100):
    f, s = df["close"].rolling(fast).mean(), df["close"].rolling(slow).mean()
    pos = np.where(f > s, 1.0, -1.0)
    pos[s.isna().to_numpy()] = 0.0
    return pd.Series(pos, index=df.index)


def donchian_breakout(df, entry=55, exit=20):
    """Turtle-style: enter on N-bar breakout, exit on opposite M-bar extreme."""
    c = df["close"].to_numpy()
    hi = df["high"].rolling(entry).max().shift().to_numpy()
    lo = df["low"].rolling(entry).min().shift().to_numpy()
    xhi = df["high"].rolling(exit).max().shift().to_numpy()
    xlo = df["low"].rolling(exit).min().shift().to_numpy()
    pos = np.zeros(len(c))
    p = 0.0
    for i in range(len(c)):
        if np.isnan(hi[i]):
            continue
        if p == 0:
            p = 1.0 if c[i] > hi[i] else -1.0 if c[i] < lo[i] else 0.0
        elif p > 0 and c[i] < xlo[i]:
            p = -1.0 if c[i] < lo[i] else 0.0
        elif p < 0 and c[i] > xhi[i]:
            p = 1.0 if c[i] > hi[i] else 0.0
        pos[i] = p
    return pd.Series(pos, index=df.index)


# ---------------------------------------------------------------- mean reversion
def rsi_reversion(df, n=14, band=30, exit=50):
    r = rsi(df["close"], n).to_numpy()
    lo, hi = band, 100 - band
    pos = np.zeros(len(r))
    p = 0.0
    for i in range(len(r)):
        if np.isnan(r[i]):
            continue
        if p == 0:
            p = 1.0 if r[i] < lo else -1.0 if r[i] > hi else 0.0
        elif p > 0 and r[i] > exit:
            p = 0.0
        elif p < 0 and r[i] < exit:
            p = 0.0
        pos[i] = p
    return pd.Series(pos, index=df.index)


def bollinger_reversion(df, n=20, k=2.0):
    c = df["close"]
    mid = c.rolling(n).mean()
    sd = c.rolling(n).std()
    up, dn = (mid + k * sd).to_numpy(), (mid - k * sd).to_numpy()
    m, cc = mid.to_numpy(), c.to_numpy()
    pos = np.zeros(len(cc))
    p = 0.0
    for i in range(len(cc)):
        if np.isnan(m[i]):
            continue
        if p == 0:
            p = 1.0 if cc[i] < dn[i] else -1.0 if cc[i] > up[i] else 0.0
        elif p > 0 and cc[i] >= m[i]:
            p = 0.0
        elif p < 0 and cc[i] <= m[i]:
            p = 0.0
        pos[i] = p
    return pd.Series(pos, index=df.index)


# ---------------------------------------------------------------- session (intraday only)
def london_breakout(df, range_end=7, flat_hour=16):
    """Asian range 00:00→range_end UTC; trade first breakout until flat_hour UTC; flat overnight."""
    idx = df.index
    hours = idx.hour.to_numpy()
    days = idx.normalize().to_numpy()
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    pos = np.zeros(len(c))
    cur_day, rh, rl, p, traded = None, -np.inf, np.inf, 0.0, False
    for i in range(len(c)):
        if days[i] != cur_day:
            cur_day, rh, rl, p, traded = days[i], -np.inf, np.inf, 0.0, False
        if hours[i] < range_end:
            rh, rl = max(rh, h[i]), min(rl, l[i])
            p = 0.0
        elif hours[i] < flat_hour and np.isfinite(rh):
            if p == 0 and not traded:
                if c[i] > rh:
                    p, traded = 1.0, True
                elif c[i] < rl:
                    p, traded = -1.0, True
        else:
            p = 0.0
        pos[i] = p
    return pd.Series(pos, index=df.index)


# ---------------------------------------------------------------- registry
def _grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, v)) for v in product(*axes.values())]


STRATEGIES = {
    "sma_cross": (sma_cross, [g for g in _grid(fast=[10, 20, 50], slow=[100, 200]) if g["fast"] < g["slow"]], "trend"),
    "donchian_breakout": (donchian_breakout, _grid(entry=[20, 55, 100], exit=[10, 20]), "trend"),
    "rsi_reversion": (rsi_reversion, _grid(n=[7, 14], band=[20, 30], exit=[50]), "reversion"),
    "bollinger_reversion": (bollinger_reversion, _grid(n=[20, 50], k=[2.0, 2.5]), "reversion"),
    "london_breakout": (london_breakout, _grid(range_end=[6, 7], flat_hour=[14, 16]), "session"),
}
INTRADAY_ONLY = {"london_breakout"}
STOP_GRID = [0.0, 3.0]   # ATR multiples; 0 = no stop. Added to every strategy's grid.
