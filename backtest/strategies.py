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


# ---------------------------------------------------------------- momentum / volatility (new)
def macd_trend(df, fast=12, slow=26, sig=9):
    """MACD above its signal line = long, below = short."""
    c = df["close"]
    macd = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    signal = macd.ewm(span=sig, adjust=False).mean()
    pos = np.where(macd > signal, 1.0, -1.0)
    pos[signal.isna().to_numpy()] = 0.0
    return pd.Series(pos, index=df.index)


def rsi_trend_filter(df, n=14, band=30, exit=50, trend=200):
    """RSI reversion, but ONLY with the long-term trend (cuts whipsaw)."""
    c = df["close"]
    r = rsi(c, n).to_numpy()
    ma = c.rolling(trend).mean().to_numpy()
    cc = c.to_numpy()
    up = cc > ma
    lo, hi = band, 100 - band
    pos = np.zeros(len(cc))
    p = 0.0
    for i in range(len(cc)):
        if np.isnan(r[i]) or np.isnan(ma[i]):
            continue
        if p == 0:
            if up[i] and r[i] < lo:
                p = 1.0
            elif (not up[i]) and r[i] > hi:
                p = -1.0
        elif p > 0 and r[i] > exit:
            p = 0.0
        elif p < 0 and r[i] < exit:
            p = 0.0
        pos[i] = p
    return pd.Series(pos, index=df.index)


def keltner_breakout(df, n=20, mult=1.5):
    """Volatility-channel breakout: close beyond the Keltner band = breakout trade."""
    c = df["close"]
    ema = c.ewm(span=n, adjust=False).mean()
    a = atr(df, n)
    up, dn = (ema + mult * a).to_numpy(), (ema - mult * a).to_numpy()
    m, cc = ema.to_numpy(), c.to_numpy()
    pos = np.zeros(len(cc))
    p = 0.0
    for i in range(len(cc)):
        if np.isnan(m[i]):
            continue
        if p == 0:
            p = 1.0 if cc[i] > up[i] else -1.0 if cc[i] < dn[i] else 0.0
        elif p > 0 and cc[i] < m[i]:
            p = 0.0
        elif p < 0 and cc[i] > m[i]:
            p = 0.0
        pos[i] = p
    return pd.Series(pos, index=df.index)


def adx(df, n=14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = ((up > dn) & (up > 0)) * up
    minus = ((dn > up) & (dn > 0)) * dn
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / atr_.replace(0, np.nan)
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / atr_.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def triple_rsi(df, r1=5, r2=14, r3=28, lo=30, hi=70, exit=50):
    """Triple RSI (The Forex Geek): all three RSIs oversold = long, overbought = short."""
    a = rsi(df["close"], r1).to_numpy()
    b = rsi(df["close"], r2).to_numpy()
    d = rsi(df["close"], r3).to_numpy()
    pos = np.zeros(len(a))
    p = 0.0
    for i in range(len(a)):
        if np.isnan(d[i]):
            continue
        if p == 0:
            if a[i] < lo and b[i] < lo and d[i] < lo:
                p = 1.0
            elif a[i] > hi and b[i] > hi and d[i] > hi:
                p = -1.0
        elif p > 0 and b[i] > exit:
            p = 0.0
        elif p < 0 and b[i] < exit:
            p = 0.0
        pos[i] = p
    return pd.Series(pos, index=df.index)


def adx_trend(df, fast=20, slow=50, adx_n=14, th=25):
    """MA trend traded ONLY when ADX confirms a trend (regime filter)."""
    c = df["close"]
    f = c.ewm(span=fast, adjust=False).mean()
    s = c.ewm(span=slow, adjust=False).mean()
    a = adx(df, adx_n).to_numpy()
    ff, ss = f.to_numpy(), s.to_numpy()
    pos = np.zeros(len(c))
    for i in range(len(c)):
        if np.isnan(ff[i]) or np.isnan(a[i]):
            continue
        pos[i] = (1.0 if ff[i] > ss[i] else -1.0) if a[i] > th else 0.0
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
STRATEGIES["macd_trend"] = (macd_trend, _grid(fast=[12], slow=[26, 50], sig=[9]), "trend")
STRATEGIES["rsi_trend_filter"] = (rsi_trend_filter, _grid(n=[14], band=[25, 30], exit=[50], trend=[100, 200]), "reversion")
STRATEGIES["keltner_breakout"] = (keltner_breakout, _grid(n=[20, 50], mult=[1.5, 2.0]), "trend")
STRATEGIES["triple_rsi"] = (triple_rsi, _grid(lo=[25, 30], hi=[70, 75]), "reversion")
STRATEGIES["adx_trend"] = (adx_trend, _grid(fast=[10, 20], slow=[50, 100], th=[20, 25]), "trend")
INTRADAY_ONLY = {"london_breakout"}
STOP_GRID = [0.0, 3.0]   # ATR multiples; 0 = no stop. Added to every strategy's grid.
