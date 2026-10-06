#!/usr/bin/env python3
"""
Live signal runner — compute the CURRENT signal for configured strategy×pair from the
latest bars in Supabase, and push CHANGES to WhatsApp via Evolution.

Env:
  SUPABASE_DB_URL   required (same DSN as the pipeline)
  SIGNAL_CONFIG     "EURUSD:rsi_trend_filter,GBPUSD:keltner_breakout"  (pair:strategy, comma-sep)
  FOREX_TF          default h1
  SIGNAL_WA_NUMBER  destination WhatsApp number, e.g. 966502919936
  EVO_URL           default https://evo.biggrowth.io
  EVO_INSTANCE      default "BIG GROWTH"
  EVO_KEY           Evolution instance API key
"""
import json
import os
import sys
import urllib.parse
import urllib.request

import pandas as pd
import psycopg

sys.path.insert(0, "/app/backtest")
import strategies as S  # noqa: E402

STATE = os.environ.get("SIGNAL_STATE", "/data/signals_state.json")
LOOKBACK = 500
LABELS = {1.0: "🟢 شراء (Long)", -1.0: "🔴 بيع (Short)", 0.0: "⚪ خروج (Flat)"}


def send_wa(text: str) -> None:
    key, num = os.environ.get("EVO_KEY"), os.environ.get("SIGNAL_WA_NUMBER")
    if not (key and num):
        print("whatsapp not configured (EVO_KEY / SIGNAL_WA_NUMBER)")
        return
    base = os.environ.get("EVO_URL", "https://evo.biggrowth.io")
    inst = urllib.parse.quote(os.environ.get("EVO_INSTANCE", "BIG GROWTH"))
    body = json.dumps({"number": num, "text": text}).encode()
    req = urllib.request.Request(base + "/message/sendText/" + inst, data=body, method="POST",
                                 headers={"apikey": key, "Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=30)
        print("  -> whatsapp sent")
    except Exception as e:  # noqa: BLE001
        print("  -> whatsapp error:", e)


def main() -> None:
    dsn = os.environ["SUPABASE_DB_URL"]
    tf = os.environ.get("FOREX_TF", "h1")
    config = os.environ.get("SIGNAL_CONFIG", "").strip()
    if not config:
        print("SIGNAL_CONFIG empty — nothing to do")
        return
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    with psycopg.connect(dsn, connect_timeout=20) as conn:
        for item in config.split(","):
            pair, strat = (x.strip() for x in item.split(":"))
            if strat not in S.STRATEGIES:
                print("unknown strategy:", strat)
                continue
            fn = S.STRATEGIES[strat][0]
            df = pd.read_sql(
                "select ts,open,high,low,close,volume from public.forex_ohlcv "
                "where pair=%s and tf=%s order by ts desc limit %s",
                conn, params=[pair.upper(), tf, LOOKBACK]).iloc[::-1].set_index("ts")
            if df.empty:
                print("no data:", pair)
                continue
            desired = float(fn(df).iloc[-1])
            bar_ts = str(df.index[-1])
            k = f"{pair.upper()}:{strat}"
            prev = state.get(k, {}).get("pos")
            if prev is None or desired != prev:
                msg = (f"📣 إشارة · {strat} · {pair.upper()} ({tf})\n"
                       f"{LABELS.get(desired, desired)}\nشمعة: {bar_ts}\n"
                       f"⚠️ إشارة بحثية — ليست نصيحة تداول.")
                print("signal:", msg.replace("\n", " | "))
                send_wa(msg)
                state[k] = {"pos": desired, "ts": bar_ts}
            else:
                print(f"{k}: unchanged ({LABELS.get(desired, desired)})")
    json.dump(state, open(STATE, "w"), ensure_ascii=False, indent=2)
    print("done")


if __name__ == "__main__":
    main()
