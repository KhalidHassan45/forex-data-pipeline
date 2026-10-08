"""Triple-barrier labelling, NET of costs.

For every bar t we ask: entering at the NEXT bar's open, with a take-profit of pt×ATR and a stop of sl×ATR
and a time limit of `horizon` bars — which happens first, for a long and for a short?

    label = +1  the long hits its take-profit first        (a buying opportunity)
    label = -1  the short hits its take-profit first       (a selling opportunity)
    label =  0  neither: stop, timeout, or the move is too small to pay costs (no opportunity)

Also stored per bar: ret_long / ret_short = the realised NET return of each trade, cost, and t1 = when the
later of the two trades ends. t1 is what Purged CV uses to stop overlapping labels leaking between folds.
Conservative rules: a bar touching both barriers counts as the stop; a gap through the stop fills at the open.
"""
import numpy as np
import pandas as pd

import common as C

DEFAULT = {"horizon": 24, "pt": 1.5, "sl": 1.0, "atr_n": 24, "min_move_cost_ratio": 3.0}


def atr(df, n):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def triple_barrier(df: pd.DataFrame, pair: str, cfg=None) -> pd.DataFrame:
    cfg = {**DEFAULT, **(cfg or {})}
    h, pt, sl = int(cfg["horizon"]), float(cfg["pt"]), float(cfg["sl"])
    if pt < sl:
        raise ValueError("pt must be >= sl so a long-win and a short-win cannot both happen")
    o, hi, lo, cl = (df[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    a = atr(df, int(cfg["atr_n"])).to_numpy(float)
    n = len(df)
    m = n - h - 1
    if m <= 0:
        return pd.DataFrame()
    t = np.arange(m)
    entry = o[t + 1]
    A = a[t]
    cost = C.roundtrip_cost_px(pair)
    L_tp, L_sl = entry + pt * A, entry - sl * A
    S_tp, S_sl = entry - pt * A, entry + sl * A
    L_done = np.zeros(m, bool); S_done = np.zeros(m, bool)
    L_px = np.full(m, np.nan); S_px = np.full(m, np.nan)
    L_k = np.full(m, h); S_k = np.full(m, h)
    L_win = np.zeros(m, bool); S_win = np.zeros(m, bool)
    for k in range(1, h + 1):
        j = t + k
        hj, lj, oj = hi[j], lo[j], o[j]
        # long
        stop = ~L_done & (lj <= L_sl)
        take = ~L_done & (hj >= L_tp) & ~stop
        L_px[stop] = np.minimum(L_sl[stop], oj[stop]); L_px[take] = L_tp[take]
        L_k[stop | take] = k; L_win[take] = True; L_done |= stop | take
        # short
        stop = ~S_done & (hj >= S_sl)
        take = ~S_done & (lj <= S_tp) & ~stop
        S_px[stop] = np.maximum(S_sl[stop], oj[stop]); S_px[take] = S_tp[take]
        S_k[stop | take] = k; S_win[take] = True; S_done |= stop | take
    end_px = cl[t + h]
    L_px[~L_done] = end_px[~L_done]; S_px[~S_done] = end_px[~S_done]
    ret_l = (L_px - entry - cost) / entry
    ret_s = (entry - S_px - cost) / entry
    tradeable = pt * A >= cfg["min_move_cost_ratio"] * cost
    lab = np.where(L_win & (ret_l > 0) & tradeable, 1, np.where(S_win & (ret_s > 0) & tradeable, -1, 0))
    idx = df.index
    out = pd.DataFrame({
        "label": lab.astype(np.int8),
        "ret_long": ret_l, "ret_short": ret_s,
        "exit_long": idx[t + L_k], "exit_short": idx[t + S_k],
        "cost_frac": cost / entry, "atr": A, "tradeable": tradeable,
    }, index=idx[:m])
    out["t1"] = idx[t + np.maximum(L_k, S_k)]
    return out[np.isfinite(A)]


def label_path(pair, tf, cfg):
    return C.LABEL_DIR / f"{pair}_{tf}_{C.cfg_hash({**DEFAULT, **(cfg or {})})}.parquet"


def build(D, tf, cfg=None, force=False):
    C.LABEL_DIR.mkdir(parents=True, exist_ok=True)
    res = {}
    for p, df in D.items():
        f = label_path(p, tf, cfg)
        if f.exists() and not force and pd.read_parquet(f, columns=["label"]).index.max() >= df.index[-30]:
            res[p] = f; continue
        L = triple_barrier(df, p, cfg)
        L.to_parquet(f)
        res[p] = f
    return res


def load(pair, tf, cfg=None):
    return pd.read_parquet(label_path(pair, tf, cfg))
