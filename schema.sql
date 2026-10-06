-- شغّله مرة واحدة في Supabase → SQL Editor
CREATE TABLE IF NOT EXISTS public.forex_ohlcv (
    pair    text        NOT NULL,          -- مثل EURUSD
    tf      text        NOT NULL,          -- m1, m5, h1, d1 ...
    ts      timestamptz NOT NULL,          -- بتوقيت UTC
    open    double precision NOT NULL,
    high    double precision NOT NULL,
    low     double precision NOT NULL,
    close   double precision NOT NULL,
    volume  double precision,
    PRIMARY KEY (pair, tf, ts)
);

CREATE INDEX IF NOT EXISTS forex_ohlcv_ts_idx ON public.forex_ohlcv (ts);

-- عرض جاهز: العائد اليومي والتذبذب الشهري لكل زوج
CREATE OR REPLACE VIEW public.forex_monthly_stats AS
WITH daily AS (
    SELECT pair,
           date_trunc('day', ts) AS d,
           (array_agg(close ORDER BY ts DESC))[1] AS close
    FROM public.forex_ohlcv
    GROUP BY pair, date_trunc('day', ts)
), rets AS (
    SELECT pair, d,
           close / lag(close) OVER (PARTITION BY pair ORDER BY d) - 1 AS ret
    FROM daily
)
SELECT pair,
       date_trunc('month', d)            AS month,
       round((exp(sum(ln(1 + ret))) - 1)::numeric * 100, 3) AS return_pct,
       round((stddev(ret) * sqrt(252))::numeric * 100, 3)   AS ann_vol_pct
FROM rets
WHERE ret IS NOT NULL
GROUP BY pair, date_trunc('month', d)
ORDER BY pair, month;
