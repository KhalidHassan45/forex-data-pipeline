"""Synthetic h1 market with KNOWN truth, to prove the lab finds real edges and rejects noise.
EURUSD: planted edge for the whole history (after a 24h selloff, London 07-10 UTC drifts up 8 bars).
GBPUSD: same edge but it DIES at the vault start (must pass research, then fail the vault).
All other pairs: pure noise with volatility clustering (must be rejected)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(sys.argv[1]) / "parquet"; out.mkdir(parents=True, exist_ok=True)
idx = pd.date_range("2015-01-01", "2026-09-30 23:00", freq="h", tz="UTC")
idx = idx[(idx.dayofweek < 5)]
n = len(idx)
hours = idx.hour.to_numpy()
vault = pd.Timestamp("2024-09-01", tz="UTC")
start_px = {"EURUSD": 1.10, "GBPUSD": 1.30, "USDJPY": 120.0, "AUDUSD": 0.72, "USDCAD": 1.30, "XAUUSD": 1800.0}
for k, (pair, p0) in enumerate(start_px.items()):
    rng = np.random.default_rng(100 + k)
    base_sig = 0.0011 * (1 + 0.6 * np.isin(hours, range(7, 17)))
    r = np.zeros(n); s2 = 1.0; boost = np.zeros(n)
    z = rng.standard_normal(n)
    for i in range(1, n):
        s2 = 0.05 + 0.90 * s2 + 0.05 * (r[i - 1] / base_sig[i - 1]) ** 2
        planted = pair == "EURUSD" or (pair == "GBPUSD" and idx[i] < vault)
        if planted and hours[i] in (7, 8, 9, 10) and boost[i] == 0 and i > 24 and r[i - 24:i].sum() < -0.004:
            boost[i + 1:i + 9] = 0.30
        r[i] = base_sig[i] * np.sqrt(s2) * (z[i] + boost[i])
    c = p0 * np.exp(np.cumsum(r))
    o = np.r_[p0, c[:-1]]
    wig = np.abs(rng.standard_normal((n, 2))) * base_sig[:, None] * 0.6 * c[:, None]
    h = np.maximum(o, c) + wig[:, 0]; l = np.minimum(o, c) - wig[:, 1]
    pd.DataFrame({"ts": idx, "open": o, "high": h, "low": l, "close": c}).to_parquet(out / f"{pair.lower()}_h1.parquet")
    print(pair, n, "events" if pair in ("EURUSD", "GBPUSD") else "", int((boost[1:] > 0).sum() / 8))
