-- SENTINEL PostgreSQL Schema
-- Run order: 01_init.sql → 02_hypertables.sql

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- Fuzzy text search

-- ─── Instrument Master ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS instruments (
    figi            VARCHAR(12) PRIMARY KEY,
    ticker          VARCHAR(20) NOT NULL,
    isin            VARCHAR(12),
    cusip           VARCHAR(9),
    sedol           VARCHAR(7),
    name            VARCHAR(256) NOT NULL DEFAULT '',
    exchange        VARCHAR(20) NOT NULL DEFAULT '',
    asset_class     VARCHAR(30) NOT NULL DEFAULT 'equity',
    currency        VARCHAR(3) NOT NULL DEFAULT 'USD',
    security_type   VARCHAR(50),
    sector          VARCHAR(100),
    industry        VARCHAR(100),
    gics_sector     VARCHAR(50),
    sic_code        VARCHAR(6),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_instruments_ticker ON instruments (ticker);
CREATE INDEX IF NOT EXISTS ix_instruments_isin   ON instruments (isin) WHERE isin IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_instruments_cusip  ON instruments (cusip) WHERE cusip IS NOT NULL;

-- ─── OHLCV (TimescaleDB hypertable — created in 02_hypertables.sql) ──────────
CREATE TABLE IF NOT EXISTS ohlcv (
    time        TIMESTAMPTZ NOT NULL,
    figi        VARCHAR(12) NOT NULL REFERENCES instruments(figi) ON DELETE CASCADE,
    open        NUMERIC(20,8) NOT NULL,
    high        NUMERIC(20,8) NOT NULL,
    low         NUMERIC(20,8) NOT NULL,
    close       NUMERIC(20,8) NOT NULL,
    volume      NUMERIC(24,4) NOT NULL DEFAULT 0,
    vwap        NUMERIC(20,8),
    source      VARCHAR(50) NOT NULL,
    interval    VARCHAR(10) NOT NULL DEFAULT '1d',
    PRIMARY KEY (time, figi, interval)
);

-- ─── Financial Facts (EDGAR XBRL) ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS financial_facts (
    id          BIGSERIAL PRIMARY KEY,
    cik         VARCHAR(10) NOT NULL,
    figi        VARCHAR(12) NOT NULL DEFAULT '',
    concept     VARCHAR(200) NOT NULL,
    label       VARCHAR(100) NOT NULL,
    value       NUMERIC(28,4) NOT NULL,
    unit        VARCHAR(30) NOT NULL,
    period_start DATE,
    period_end  DATE NOT NULL,
    form        VARCHAR(20) NOT NULL DEFAULT '',
    filed       DATE,
    accession   VARCHAR(25),
    frame       VARCHAR(30),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_facts_cik_label ON financial_facts (cik, label);
CREATE INDEX IF NOT EXISTS ix_facts_figi_label ON financial_facts (figi, label) WHERE figi != '';
CREATE INDEX IF NOT EXISTS ix_facts_period ON financial_facts (period_end);

-- ─── Insider Transactions (Form 4) ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS insider_transactions (
    id              BIGSERIAL PRIMARY KEY,
    cik             VARCHAR(10) NOT NULL,
    figi            VARCHAR(12) NOT NULL DEFAULT '',
    ticker          VARCHAR(20) NOT NULL,
    owner_name      VARCHAR(200) NOT NULL,
    owner_cik       VARCHAR(10),
    role            VARCHAR(30) NOT NULL,
    security_title  VARCHAR(100),
    tx_date         DATE NOT NULL,
    tx_code         VARCHAR(20) NOT NULL,
    shares          NUMERIC(20,4) NOT NULL,
    price_per_share NUMERIC(20,8),
    value           NUMERIC(24,4),
    shares_owned_after NUMERIC(20,4),
    is_derivative   BOOLEAN NOT NULL DEFAULT FALSE,
    exercise_price  NUMERIC(20,8),
    expiry_date     DATE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_insider_figi_date ON insider_transactions (figi, tx_date);
CREATE INDEX IF NOT EXISTS ix_insider_ticker ON insider_transactions (ticker);

-- ─── Institutional Holdings (13F-HR) ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS institutional_holdings (
    id                  BIGSERIAL PRIMARY KEY,
    manager_cik         VARCHAR(10) NOT NULL,
    issuer_name         VARCHAR(256),
    cusip               VARCHAR(9),
    ticker              VARCHAR(20),
    figi                VARCHAR(12),
    period_of_report    DATE NOT NULL,
    filed_date          DATE,
    market_value        NUMERIC(20,2),
    shares              NUMERIC(20,4),
    share_type          VARCHAR(5),
    put_call            VARCHAR(4),
    investment_discretion VARCHAR(10),
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_holdings_manager_period ON institutional_holdings (manager_cik, period_of_report);
CREATE INDEX IF NOT EXISTS ix_holdings_cusip ON institutional_holdings (cusip) WHERE cusip IS NOT NULL;

-- ─── Congressional Trades (STOCK Act) ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS congressional_trades (
    id              BIGSERIAL PRIMARY KEY,
    politician_name VARCHAR(200) NOT NULL,
    chamber         VARCHAR(10) NOT NULL,
    party           VARCHAR(5),
    state           VARCHAR(5),
    ticker          VARCHAR(20),
    figi            VARCHAR(12),
    asset_name      VARCHAR(500),
    tx_date         DATE NOT NULL,
    filed_date      DATE NOT NULL,
    tx_type         VARCHAR(20) NOT NULL,
    amount_low      NUMERIC(20,2),
    amount_high     NUMERIC(20,2),
    filing_lag_days INTEGER,
    late_filing     BOOLEAN DEFAULT FALSE,
    source          VARCHAR(30) NOT NULL,
    disclosure_url  TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_congress_ticker ON congressional_trades (ticker) WHERE ticker IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_congress_date ON congressional_trades (tx_date);

-- ─── Macro Data Points (FRED) ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS macro_data (
    time        TIMESTAMPTZ NOT NULL,
    series_id   VARCHAR(30) NOT NULL,
    value       NUMERIC(20,8) NOT NULL,
    vintage     TIMESTAMPTZ,   -- ALFRED point-in-time
    PRIMARY KEY (time, series_id)
);

-- ─── News Articles ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS news_articles (
    id              BIGSERIAL NOT NULL,
    headline        TEXT NOT NULL,
    summary         TEXT,
    source          VARCHAR(100),
    url             TEXT,
    published_at    TIMESTAMPTZ NOT NULL,
    tickers         VARCHAR(20)[],
    sentiment_label VARCHAR(20),
    sentiment_score NUMERIC(6,4),
    embedding       vector(384),  -- sentence-transformers/all-MiniLM-L6-v2
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (published_at, id)
);
CREATE INDEX IF NOT EXISTS ix_news_published ON news_articles (published_at);
CREATE INDEX IF NOT EXISTS ix_news_tickers ON news_articles USING GIN (tickers);
-- Vector similarity index created after embedding population:
-- CREATE INDEX ON news_articles USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ─── Backtest Results ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS backtest_results (
    id                  BIGSERIAL PRIMARY KEY,
    strategy_id         VARCHAR(200) NOT NULL,
    ticker              VARCHAR(20),
    start_date          DATE,
    end_date            DATE,
    interval            VARCHAR(10),
    total_return        NUMERIC(12,6),
    cagr                NUMERIC(12,6),
    sharpe_ratio        NUMERIC(10,4),
    sortino_ratio       NUMERIC(10,4),
    calmar_ratio        NUMERIC(10,4),
    deflated_sharpe     NUMERIC(10,4),
    max_drawdown        NUMERIC(10,6),
    win_rate            NUMERIC(8,4),
    volatility          NUMERIC(10,6),
    var_95              NUMERIC(10,6),
    cvar_95             NUMERIC(10,6),
    beta                NUMERIC(10,4),
    alpha               NUMERIC(10,6),
    n_trials            INTEGER DEFAULT 1,
    params              JSONB,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Strategy Registry ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS strategy_registry (
    strategy_id     VARCHAR(200) PRIMARY KEY,
    name            VARCHAR(200) NOT NULL,
    description     TEXT,
    status          VARCHAR(30) NOT NULL DEFAULT 'BACKTEST',
    status_since    DATE,
    kill_switch     BOOLEAN NOT NULL DEFAULT FALSE,
    params          JSONB,
    tags            TEXT[],
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Strategy Promotion Log ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS promotion_log (
    id              BIGSERIAL PRIMARY KEY,
    strategy_id     VARCHAR(200) NOT NULL REFERENCES strategy_registry(strategy_id),
    from_status     VARCHAR(30) NOT NULL,
    to_status       VARCHAR(30) NOT NULL,
    approved        BOOLEAN NOT NULL,
    evaluated_at    TIMESTAMPTZ NOT NULL,
    gates           JSONB,
    rejection_reason TEXT,
    notes           TEXT
);

-- ─── Data Health Events ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS data_health_events (
    id          BIGSERIAL NOT NULL,
    time        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    adapter     VARCHAR(50) NOT NULL,
    event_type  VARCHAR(50) NOT NULL,
    severity    VARCHAR(20) NOT NULL,
    details     JSONB,
    PRIMARY KEY (time, id)
);
CREATE INDEX IF NOT EXISTS ix_health_adapter_time ON data_health_events (adapter, time);

-- ─── Corporate Actions (SDS Gen 0) ───────────────────────────────────────────
-- Raw OHLCV is NEVER modified. adj_factor is computed on read from this table.
CREATE TABLE IF NOT EXISTS corporate_actions (
    id              BIGSERIAL PRIMARY KEY,
    figi            VARCHAR(12) NOT NULL,
    ticker          VARCHAR(20) NOT NULL,
    action_type     VARCHAR(30) NOT NULL,   -- split / reverse_split / dividend / spin_off
    ex_date         DATE NOT NULL,
    ratio_new       NUMERIC(20, 8) NOT NULL DEFAULT 1,
    ratio_old       NUMERIC(20, 8) NOT NULL DEFAULT 1,
    factor          NUMERIC(20, 8) NOT NULL, -- precomputed backward-adjustment factor
    source          VARCHAR(50) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (figi, action_type, ex_date)
);
CREATE INDEX IF NOT EXISTS ix_ca_figi_exdate ON corporate_actions (figi, ex_date);
CREATE INDEX IF NOT EXISTS ix_ca_ticker       ON corporate_actions (ticker);

-- ─── Survivorship Registry (SDS Gen 0) ───────────────────────────────────────
-- Known delisted/bankrupt securities — prevents survivorship bias in backtests.
CREATE TABLE IF NOT EXISTS survivorship_registry (
    id              BIGSERIAL PRIMARY KEY,
    cik             VARCHAR(10) NOT NULL,
    figi            VARCHAR(12),
    ticker          VARCHAR(20) NOT NULL,
    company_name    VARCHAR(256) NOT NULL,
    delist_date     DATE NOT NULL,
    delist_reason   VARCHAR(30) NOT NULL DEFAULT 'unknown',
    exchange        VARCHAR(20),
    notes           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (cik)
);
CREATE INDEX IF NOT EXISTS ix_surv_ticker  ON survivorship_registry (ticker);
CREATE INDEX IF NOT EXISTS ix_surv_delist  ON survivorship_registry (delist_date);

-- ─── Data Provenance (SDS Gen 0) ─────────────────────────────────────────────
-- Append-only SHA-256 receipt chain — every ingestion batch is auditable.
CREATE TABLE IF NOT EXISTS data_provenance (
    id                   BIGSERIAL PRIMARY KEY,
    batch_id             UUID NOT NULL DEFAULT gen_random_uuid(),
    source               VARCHAR(50) NOT NULL,
    ticker               VARCHAR(20) NOT NULL,
    figi                 VARCHAR(12),
    interval             VARCHAR(10) NOT NULL,
    start_time           TIMESTAMPTZ NOT NULL,
    end_time             TIMESTAMPTZ NOT NULL,
    bar_count            INTEGER NOT NULL,
    sha256               VARCHAR(64) NOT NULL,
    prev_hash            VARCHAR(64),
    validated            BOOLEAN NOT NULL DEFAULT FALSE,
    validation_delta_pct NUMERIC(8, 4),
    ingested_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (sha256)
);
CREATE INDEX IF NOT EXISTS ix_prov_ticker_source ON data_provenance (ticker, source, interval);
CREATE INDEX IF NOT EXISTS ix_prov_ingested      ON data_provenance (ingested_at);
