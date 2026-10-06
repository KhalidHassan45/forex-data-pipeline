import glob
import os

import pandas as pd

P = os.environ.get("PARQUET_DIR", "/data/parquet")
print(f"{'pair':8}{'bars':>9}  {'span':22}{'yrs':>4}  gaps(missing years) / thin(<3000)")
for f in sorted(glob.glob(os.path.join(P, "*_h1.parquet"))):
    df = pd.read_parquet(f)
    if "ts" not in df.columns:
        continue
    d = df.groupby(df["ts"].dt.year).size()
    pair = os.path.basename(f).split("_")[0].upper()
    y0, y1 = int(d.index.min()), int(d.index.max())
    missing = [y for y in range(y0, y1 + 1) if y not in d.index]
    thin = [int(y) for y, n in d.items() if n < 3000]
    print(f"{pair:8}{len(df):>9}  {str(y0)+'-'+str(y1):22}{len(d):>4}  {missing} / {thin}")
