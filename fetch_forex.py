#!/usr/bin/env python3
"""
Forex historical data pipeline — Dukascopy (free) → Parquet → Supabase/Postgres.

Modes:
    backfill : download all years in [--from, --to], skip yearly files already on disk.
    update   : re-download the current year only, rebuild Parquet, upload current-year rows.

Examples:
    python fetch_forex.py backfill                       # 10 years, h1, default pairs
    python fetch_forex.py backfill --tf d1 --pairs eurusd xauusd
    python fetch_forex.py update                         # daily incremental
    python fetch_forex.py backfill --no-db               # Parquet only

Env:
    DATA_DIR          (default ./data)       raw CSV + parquet + logs live here
    SUPABASE_DB_URL   Postgres DSN (required unless --no-db)
    FOREX_PAIRS       space-separated override of default pairs
    FOREX_TF          default timeframe (h1)
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_PAIRS = os.getenv(
    "FOREX_PAIRS",
    "eurusd gbpusd usdjpy usdchf audusd usdcad nzdusd eurgbp eurjpy gbpjpy xauusd",
).split()
VALID_TF = ["m1", "m5", "m15", "m30", "h1", "h4", "d1"]
TF_STEP = {"m1": "1min", "m5": "5min", "m15": "15min", "m30": "30min",
           "h1": "1h", "h4": "4h", "d1": "1D"}

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "parquet"
LOG_DIR = DATA_DIR / "logs"

DUKA = ["dukascopy-node"] if shutil.which("dukascopy-node") else ["npx", "--yes", "dukascopy-node"]


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


def raw_path(pair: str, tf: str, price: str, year: int) -> Path:
    return RAW_DIR / f"{pair}_{tf}_{price}_{year}.csv"


def download_year(pair: str, year: int, tf: str, price: str, force: bool = False,
                  retries: int = 3) -> Path | None:
    """Download one pair-year. Yearly split = resumable + smaller failures."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = raw_path(pair, tf, price, year)
    if out.exists() and out.stat().st_size > 0 and not force:
        log(f"  = cached {out.name}")
        return out

    today = date.today()
    # date-to is treated as an upper bound; overlaps are de-duplicated later.
    to = f"{year + 1}-01-01" if year < today.year else "now"
    cmd = DUKA + [
        "-i", pair, "-from", f"{year}-01-01", "-to", to,
        "-t", tf, "-p", price, "-f", "csv", "-v",
        "-dir", str(RAW_DIR), "-fn", out.stem,
        "-r", "3", "-re", "-rp", "2000",
        "-bs", "10", "-bp", "1000", "-s",
    ]
    tmp_backup = None
    if out.exists() and force:
        tmp_backup = out.with_suffix(".csv.bak")
        out.replace(tmp_backup)

    for attempt in range(1, retries + 1):
        try:
            subprocess.run(cmd, check=True, timeout=3600,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if out.exists() and out.stat().st_size > 0:
                log(f"  + {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
                if tmp_backup and tmp_backup.exists():
                    tmp_backup.unlink()
                return out
            log(f"  ! no data for {out.stem} — skipping (empty year / blocked)")
            return None
        except subprocess.CalledProcessError as e:
            log(f"  ! attempt {attempt} failed for {out.stem}: {(e.stderr or '')[-300:]}")
        except subprocess.TimeoutExpired:
            log(f"  ! attempt {attempt} timed out for {out.stem}")
        if out.exists() and out.stat().st_size == 0:
            out.unlink()
        time.sleep(10 * attempt)

    if tmp_backup and tmp_backup.exists():          # keep the old data if refresh failed
        tmp_backup.replace(out)
        log(f"  ~ refresh failed, kept previous {out.name}")
        return out
    log(f"  x giving up on {out.stem}")
    return None


def build_frame(pair: str, tf: str, price: str) -> pd.DataFrame:
    """Merge every yearly CSV on disk for this pair/tf/price into one clean frame."""
    files = sorted(RAW_DIR.glob(f"{pair}_{tf}_{price}_*.csv"))
    frames = [pd.read_csv(f) for f in files if f.stat().st_size > 0]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "volume" not in df.columns:
        df["volume"] = pd.NA
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = (df.drop(columns=["timestamp"])
            .drop_duplicates(subset="ts", keep="last")
            .sort_values("ts").reset_index(drop=True))
    df.insert(0, "pair", pair.upper())
    df.insert(1, "tf", tf)
    return df[["pair", "tf", "ts", "open", "high", "low", "close", "volume"]]


def quality(df: pd.DataFrame, tf: str) -> dict:
    gaps = df["ts"].diff()
    big_gaps = int(((gaps > pd.Timedelta(TF_STEP[tf]) * 3) & (gaps > pd.Timedelta("3D"))).sum())
    bad = int(((df["high"] < df["low"]) | (df["close"] <= 0)).sum())
    return {"rows": len(df), "first": f"{df['ts'].min():%Y-%m-%d}",
            "last": f"{df['ts'].max():%Y-%m-%d %H:%M}", "gaps_gt_3d": big_gaps, "bad_candles": bad}


def upload(df: pd.DataFrame, dsn: str) -> int:
    """COPY into temp table → upsert; only new or changed bars are written (idempotent)."""
    import psycopg
    with psycopg.connect(dsn, connect_timeout=20) as conn, conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE _stage (LIKE public.forex_ohlcv INCLUDING DEFAULTS) ON COMMIT DROP;")
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False)
        buf.seek(0)
        with cur.copy("COPY _stage (pair, tf, ts, open, high, low, close, volume) FROM STDIN WITH CSV") as cp:
            while chunk := buf.read(1 << 20):
                cp.write(chunk)
        cur.execute("""
            INSERT INTO public.forex_ohlcv (pair, tf, ts, open, high, low, close, volume)
            SELECT pair, tf, ts, open, high, low, close, volume FROM _stage
            ON CONFLICT (pair, tf, ts) DO UPDATE SET
                open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                close = EXCLUDED.close, volume = EXCLUDED.volume
            WHERE (forex_ohlcv.open, forex_ohlcv.high, forex_ohlcv.low, forex_ohlcv.close, forex_ohlcv.volume)
                  IS DISTINCT FROM (EXCLUDED.open, EXCLUDED.high, EXCLUDED.low, EXCLUDED.close, EXCLUDED.volume);""")
        n = cur.rowcount
        conn.commit()
    return n


def main() -> int:
    y = date.today().year
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["backfill", "update"])
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    ap.add_argument("--tf", default=os.getenv("FOREX_TF", "h1"), choices=VALID_TF)
    ap.add_argument("--from", dest="start", type=int, default=y - 10)
    ap.add_argument("--to", dest="end", type=int, default=y)
    ap.add_argument("--price", default="bid", choices=["bid", "ask"])
    ap.add_argument("--no-db", action="store_true")
    a = ap.parse_args()

    if a.mode == "update":
        a.start = a.end = y
    dsn = os.getenv("SUPABASE_DB_URL")
    if not a.no_db and not dsn:
        log("SUPABASE_DB_URL missing (or pass --no-db)")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"mode={a.mode} tf={a.tf} years={a.start}-{a.end} pairs={len(a.pairs)} bin={DUKA[0]}")

    report = {"mode": a.mode, "tf": a.tf, "started": datetime.now(timezone.utc).isoformat(),
              "pairs": {}, "failed": []}
    for pair in (p.lower() for p in a.pairs):
        log(f"> {pair.upper()}")
        got = []
        for yr in range(a.start, a.end + 1):
            got.append(download_year(pair, yr, a.tf, a.price, force=(yr == y)))
            time.sleep(2 + (yr % 3))        # pace requests (avoid Dukascopy 429)
        df = build_frame(pair, a.tf, a.price)
        if df.empty or not any(got):
            report["failed"].append(pair.upper())
            log("  x no data")
            continue
        q = quality(df, a.tf)
        df.to_parquet(OUT_DIR / f"{pair}_{a.tf}.parquet", index=False)
        if not a.no_db:
            part = df[df["ts"] >= pd.Timestamp(f"{a.start}-01-01", tz="UTC")]
            try:
                q["upserted"] = upload(part, dsn)
            except Exception as e:  # keep going with other pairs
                q["db_error"] = str(e)[:300]
                report["failed"].append(pair.upper())
        report["pairs"][pair.upper()] = q
        log(f"  {q}")
        time.sleep(5)                       # brief pause between pairs

    report["finished"] = datetime.now(timezone.utc).isoformat()
    (LOG_DIR / "last_run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    ok = [p for p in report["pairs"] if p not in report["failed"]]
    log(f"done. ok={len(ok)} failed={report['failed']}")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
