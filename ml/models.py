"""Model fitting, thresholds, meta-labelling and trade simulation (shared by train / vault / paper)."""
import numpy as np
import pandas as pd

import common as C

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:  # pragma: no cover
    HAVE_LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

CLASSES = np.array([-1, 0, 1])

PRIMARY_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                      subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                      class_weight="balanced", random_state=7, verbose=-1)
META_PARAMS = dict(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=50,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                   random_state=11, verbose=-1)


def _clf(params, multiclass):
    if HAVE_LGB:
        p = dict(params, n_jobs=C.THREADS)
        if multiclass:
            p["objective"] = "multiclass"
        return lgb.LGBMClassifier(**p)
    return HistGradientBoostingClassifier(max_iter=params["n_estimators"], learning_rate=params["learning_rate"],
                                          max_leaf_nodes=params["num_leaves"], min_samples_leaf=params["min_child_samples"],
                                          class_weight=params.get("class_weight"), random_state=params["random_state"])


def fit_primary(X, y):
    m = _clf(PRIMARY_PARAMS, True)
    m.fit(X, y)
    return m


def proba(m, X):
    P = m.predict_proba(X)
    cl = list(m.classes_)
    get = lambda c: P[:, cl.index(c)] if c in cl else np.zeros(len(X))
    return pd.DataFrame({"p_short": get(-1), "p_none": get(0), "p_long": get(1)}, index=X.index)


def thresholds(P, q):
    return {"long": float(np.quantile(P["p_long"], q)), "short": float(np.quantile(P["p_short"], q))}


def signals(P, thr):
    lo = (P["p_long"] >= thr["long"]) & (P["p_long"] > P["p_short"])
    sh = (P["p_short"] >= thr["short"]) & (P["p_short"] > P["p_long"])
    return pd.Series(np.where(lo, 1, np.where(sh, -1, 0)), index=P.index, dtype=np.int8)


def fit_meta(X, P, side, outcome):
    """Secondary model: given that the primary fired, will THIS trade make money? (López de Prado)"""
    Z = meta_X(X, P, side)
    m = _clf(META_PARAMS, False)
    m.fit(Z, outcome.astype(int))
    return m


def meta_X(X, P, side):
    Z = X.copy()
    Z["m_side"] = side.astype(float)
    Z["m_p_side"] = np.where(side > 0, P["p_long"], P["p_short"])
    Z["m_p_gap"] = (P["p_long"] - P["p_short"]).abs()
    return Z


def meta_size(p):
    """0 below 0.5, linearly to full size at 0.75."""
    return np.clip((np.asarray(p) - 0.5) * 4, 0, 1)


def simulate(sig: pd.Series, L: pd.DataFrame, size=None, extra_cost_mult=0.0):
    """One position at a time. A trade entered at bar t uses the labeller's realised net return,
    so the evaluation uses exactly the costs, barriers and conservative fills of the labels."""
    size = pd.Series(1.0, index=sig.index) if size is None else size.reindex(sig.index).fillna(0.0)
    fire = sig[(sig != 0) & (size > 0)]
    rows, busy_until = [], None
    idx = L.index
    for ts, sd in fire.items():
        if busy_until is not None and ts < busy_until:
            continue
        r = L.at[ts, "ret_long"] if sd > 0 else L.at[ts, "ret_short"]
        ex = L.at[ts, "exit_long"] if sd > 0 else L.at[ts, "exit_short"]
        w = float(size.at[ts])
        r = (r - extra_cost_mult * L.at[ts, "cost_frac"]) * w
        pos = idx.get_loc(ts)
        entry = idx[pos + 1] if pos + 1 < len(idx) else ts
        rows.append((entry, ex, int(sd), w, float(r), ts))
        busy_until = ex
    return pd.DataFrame(rows, columns=["entry_ts", "exit_ts", "side", "size", "ret", "signal_ts"])
