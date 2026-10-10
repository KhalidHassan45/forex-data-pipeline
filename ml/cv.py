"""Validation splitters that respect time and overlapping labels.

purged_kfold: López de Prado's Purged K-Fold with embargo — training rows whose label window [t, t1]
overlaps the test block are removed, and so are rows in an embargo after it.
walk_forward: expanding window by calendar year; the boundary is purged the same way.
"""
import numpy as np
import pandas as pd

import common as C


def _purge(train_mask, idx, t1, test_start, test_end, embargo):
    # label windows overlapping [test_start, test_end] leak information about the test block
    overlap = (idx <= test_end) & (t1 >= test_start)
    emb = (idx > test_end) & (idx <= test_end + embargo)
    return train_mask & ~overlap & ~emb


def purged_kfold(idx: pd.DatetimeIndex, t1: pd.Series, n_splits=5, embargo_pct=0.01):
    n = len(idx)
    t1v = C.utc(t1)
    span = idx[-1] - idx[0]
    embargo = span * embargo_pct
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    for k in range(n_splits):
        te = np.zeros(n, bool); te[bounds[k]:bounds[k + 1]] = True
        ts_, te_ = idx[bounds[k]], t1v[bounds[k]:bounds[k + 1]].max()
        tr = _purge(~te, idx, t1v, ts_, te_, embargo)
        yield np.where(tr)[0], np.where(te)[0]


def walk_forward(idx: pd.DatetimeIndex, t1: pd.Series, min_train_years=3, embargo=pd.Timedelta(days=2)):
    """Yields (year, train_idx, test_idx). Train = everything before Jan 1 of the test year (purged)."""
    t1v = C.utc(t1)
    years = sorted(set(idx.year))
    for y in years[min_train_years:]:
        start = pd.Timestamp(year=y, month=1, day=1, tz="UTC")
        end = pd.Timestamp(year=y + 1, month=1, day=1, tz="UTC")
        te = (idx >= start) & (idx < end)
        if te.sum() < 200:      # ~1 trading year on d1 (~250 bars); h1 has ~6000/yr so unaffected
            continue
        tr = (idx < start) & (t1v < start)        # purge: a training label may not end inside the test year
        tr &= idx < start - embargo
        yield y, np.where(tr)[0], np.where(te)[0]
