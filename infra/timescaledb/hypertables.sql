-- TimescaleDB hypertable setup — run after init.sql
-- Converts time-series tables to hypertables for 10-100x query performance

-- OHLCV: partition by 1 week (intraday data → many rows)
SELECT create_hypertable('ohlcv', 'time',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- Macro data: partition by 1 month (lower cardinality)
SELECT create_hypertable('macro_data', 'time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- News: partition by 1 week
SELECT create_hypertable('news_articles', 'published_at',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- Data health events: partition by 1 day
SELECT create_hypertable('data_health_events', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Continuous aggregate for daily OHLCV from minute data
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS day,
    figi,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume,
    avg(vwap) AS vwap,
    count(*) AS bar_count
FROM ohlcv
WHERE interval IN ('1m', '5m', '15m', '30m', '1h')
GROUP BY day, figi
WITH NO DATA;

-- Refresh policy for daily aggregate
SELECT add_continuous_aggregate_policy('ohlcv_daily',
    start_offset => INTERVAL '7 days',
    end_offset   => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Data retention: keep raw minute bars for 90 days, daily bars forever
SELECT add_retention_policy('ohlcv',
    INTERVAL '90 days',
    if_not_exists => TRUE
);

-- Compression for older OHLCV chunks
ALTER TABLE ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'figi, interval',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('ohlcv',
    INTERVAL '7 days',
    if_not_exists => TRUE
);
