"""
Backtest core: execution with realistic costs, optional ATR stop, metrics, walk-forward.

Execution model (conservative, stated plainly in every report):
  * signal decided at close of bar t → position held during bar t+1 (no look-ahead)
  * every unit of position change pays half-spread + slippage (a reversal pays twice)
  * ATR stop is checked on bar high/low but exits at that bar's close (worse than a
    resting stop order in fast markets — intentionally pessimistic)
  * returns are % of notional, unlevered. Leverage multiplies returns AND drawdowns.
"""
import numpy as np
import pandas as pd

from strategies import atr

# typical retail ECN spreads in pips (override with --spread-mult)
SPREAD_PIPS = {
    "EURUSD": 1.0, "GBPUSD": 1.4, "USDJPY": 1.2, "USDCHF": 1.6, "AUDUSD": 1.3,
    "USDCAD": 1.7, "NZDUSD": 1.8, "EURGBP": 1.5, "EURJPY": 2.0, "GBPJPY": 3.0,
    "XAUUSD": 30.0,   # 0.30 USD with pip = 0.01
}
DEFAULT_SPREAD = 2.5
SLIPPAGE_PIPS = 0.3


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") or pair.startswith("XAU") else 0.0001


def apply_atr_stop(df: pd.DataFrame, pos: pd.Series, k: float) -> pd.Series:
    """Flatten a position once price moves k*ATR(entry) against it; re-arm on a new signal."""
    if not k:
        return pos
    a = atr(df).to_numpy()
    p = pos.to_numpy().copy()
    c, h, l = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    out = np.zeros_like(p)
    held, entry, risk, stopped_sig = 0.0, 0.0, 0.0, None
    for i in range(len(p)):
        sig = p[i]
        if stopped_sig is not None:
            if sig == stopped_sig:          # still the same signal that got stopped
                out[i] = 0.0
                continue
            stopped_sig = None
        if held != 0 and ((held > 0 and l[i] <= entry - risk) or (held < 0 and h[i] >= entry + risk)):
            stopped_sig, held = sig, 0.0
            out[i] = 0.0
            continue
        if sig != held:
            held = sig
            if sig != 0:
                entry = c[i]
                risk = k * (a[i] if not np.isnan(a[i]) else (h[i] - l[i]))
        out[i] = held
    return pd.Series(out, index=pos.index)


def run(df: pd.DataFrame, pos: pd.Series, pair: str, spread_mult: float = 1.0):
    """Return per-bar net returns and the held position."""
    held = pos.reindex(df.index).fillna(0).shift(1).fillna(0)
    ret = df["close"].pct_change().fillna(0)
    turnover = held.diff().abs().fillna(held.abs())
    half_cost = (SPREAD_PIPS.get(pair, DEFAULT_SPREAD) * spread_mult / 2 + SLIPPAGE_PIPS) * pip_size(pair)
    net = held * ret - turnover * half_cost / df["close"].shift(1).fillna(df["close"])
    return net, held


def trades(net: pd.Series, held: pd.Series) -> np.ndarray:
    """P&L of each trade (contiguous run of the same non-zero position)."""
    h = held.to_numpy()
    n = net.to_numpy().copy()
    # exit cost is charged on the first flat bar after a trade → move it into that trade
    exit_bar = np.r_[False, (h[1:] == 0) & (h[:-1] != 0)]
    n[np.r_[exit_bar[1:], False]] += n[exit_bar]
    grp = np.cumsum(np.r_[True, h[1:] != h[:-1]])
    df = pd.DataFrame({"g": grp, "h": h, "n": n})
    df = df[df["h"] != 0]
    if df.empty:
        return np.array([])
    return df.groupby("g")["n"].sum().to_numpy()


def metrics(net: pd.Series, held: pd.Series) -> dict:
    if len(net) < 2:
        return {}
    years = max((net.index[-1] - net.index[0]).days / 365.25, 1e-9)
    bpy = len(net) / years
    eq = (1 + net).cumprod()
    tr = trades(net, held)
    wins, losses = tr[tr > 0], tr[tr <= 0]
    sd = net.std()
    return {
        "years": round(years, 2),
        "total_return_pct": round((eq.iloc[-1] - 1) * 100, 2),
        "cagr_pct": round((eq.iloc[-1] ** (1 / years) - 1) * 100, 2) if eq.iloc[-1] > 0 else -100.0,
        "sharpe": round(net.mean() / sd * np.sqrt(bpy), 2) if sd > 0 else 0.0,
        "max_dd_pct": round((eq / eq.cummax() - 1).min() * 100, 2),
        "trades": int(len(tr)),
        "trades_per_year": round(len(tr) / years, 1),
        "win_rate_pct": round(len(wins) / len(tr) * 100, 1) if len(tr) else 0.0,
        "profit_factor": round(wins.sum() / -losses.sum(), 2) if losses.sum() < 0 else (99.0 if len(wins) else 0.0),
        "avg_trade_pct": round(tr.mean() * 100, 4) if len(tr) else 0.0,
        "exposure_pct": round((held != 0).mean() * 100, 1),
    }


def walk_forward(nets: dict, helds: dict, train_years: int, test_years: int, min_trades: int):
    """
    nets/helds: {param_key: series over the full history} (signals are causal, so
    computing them once on the full history is equivalent to recomputing per window).
    For each test window choose the params with the best TRAIN Sharpe, keep only its
    TEST returns, and stitch the out-of-sample pieces together.
    """
    any_key = next(iter(nets))
    idx = nets[any_key].index
    y0, y1 = idx[0].year, idx[-1].year
    oos_net, oos_held, chosen = [], [], []
    y = y0 + train_years
    while y <= y1:
        tr_lo, tr_hi = pd.Timestamp(f"{y - train_years}-01-01", tz="UTC"), pd.Timestamp(f"{y}-01-01", tz="UTC")
        te_hi = pd.Timestamp(f"{y + test_years}-01-01", tz="UTC")
        best, best_s = None, -np.inf
        for k in nets:
            m = metrics(nets[k][tr_lo:tr_hi - pd.Timedelta("1ns")], helds[k][tr_lo:tr_hi - pd.Timedelta("1ns")])
            if m and m["trades"] >= min_trades and m["sharpe"] > best_s:
                best, best_s = k, m["sharpe"]
        if best is not None:
            seg = slice(tr_hi, te_hi - pd.Timedelta("1ns"))
            oos_net.append(nets[best][seg])
            oos_held.append(helds[best][seg])
            chosen.append({"test_from": y, "params": best, "train_sharpe": best_s})
        y += test_years
    if not oos_net:
        return None, None, chosen
    return pd.concat(oos_net), pd.concat(oos_held), chosen
