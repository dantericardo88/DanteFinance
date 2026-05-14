# SENTINEL Technical Specification Document (TSD)
## Version 1.0 — Construction Blueprint for DanteForge
**Classification:** DanteForge Build Input — Generation 0 Ready  
**Follows:** SENTINEL Founding PRD (May 2026)  
**Owner:** Ricky Porras / Dante Ecosystem  
**Hardware Target:** Mac mini Apple Silicon (M-series), 16–32 GB RAM, 1 TB SSD  
**Philosophy:** Sovereign. Self-hosted. AI-native. No mandatory cloud. Free by default, paid by choice.

---

## PART 0 — PRD GAP ANALYSIS & ADDITIONS

Before the technical spec, six material gaps from the PRD are promoted to first-class modules or subsystems:

### Gap 1 — News & Media Intelligence Engine (New: Module SNM)
The PRD mentioned news as a data feed. It is a full organ. SENTINEL needs:
- Real-time financial news ingestion (multi-source)
- Earnings call live transcription + entity tagging
- Podcast transcription pipeline (Whisper-based)
- Fed/ECB/BoE speech and FOMC minutes intelligence
- Congressional trading disclosures (STOCK Act)
- News-to-signal pipeline with latency tracking

### Gap 2 — CFTC Commitment of Traders (COT) Module (New: subsystem of SMA)
Free every Friday from cftc.gov. Commercial vs. large speculator vs. small speculator net positioning across 150+ futures markets. One of the most powerful free macro-positioning datasets. Required for commodity trend and cross-asset macro strategies. Fully integrated into SMA.

### Gap 3 — Congressional & Insider Intelligence (New: subsystem of SOD)
STOCK Act disclosures (eFD filings) from Members of Congress. Senate eFD API and House periodic transaction reports. Research by Jochec (2020) and others shows 5–15% annualized abnormal returns from following informed congressional trades. Added to SOD alongside Form 4.

### Gap 4 — Options Flow Intelligence (New: subsystem of SSE)
CBOE put/call ratio (free, daily), VIX term structure (FRED free), unusual options activity screener (high volume/OI ratio, large premium, short-dated). Added as a screening filter category in SSE.

### Gap 5 — Data Quality Monitoring Layer (New: cross-cutting concern)
Every data adapter emits a `DataHealthEvent`. A background monitor checks staleness, unexpected gaps, schema drift, and silent throttling. Alerts surface in the terminal.

### Gap 6 — Strategy Promotion Framework (New: cross-cutting process)
The exact gates a strategy must pass to move from backtest → paper → capped-live → full-autonomous. Codified as a state machine in SEE. No strategy reaches live without passing all gates.

---

## PART 1 — PROJECT STRUCTURE

### 1.1 — Repository Layout

Every file under 500 lines (KiloCode). Every module is a Python package. No circular imports. No stubs — every file ships complete.

```
sentinel/
├── core/                          # Shared types, bus, config
│   ├── __init__.py
│   ├── config.py                  # Settings via pydantic-settings
│   ├── bus.py                     # Internal event bus (asyncio queues + Redis Streams bridge)
│   ├── types.py                   # All shared Pydantic v2 schemas
│   ├── health.py                  # Data health monitor
│   ├── logging.py                 # Structured JSON logging (structlog)
│   └── security.py                # Secret store wrapper (no plaintext keys in code)
│
├── sds/                           # SENTINEL Data Spine
│   ├── __init__.py
│   ├── base_adapter.py            # Abstract adapter interface
│   ├── adapters/
│   │   ├── yfinance_adapter.py
│   │   ├── finnhub_adapter.py
│   │   ├── alphavantage_adapter.py
│   │   ├── fmp_adapter.py
│   │   ├── fred_adapter.py
│   │   ├── ccxt_adapter.py
│   │   ├── polygon_adapter.py
│   │   ├── eodhd_adapter.py
│   │   ├── alpaca_adapter.py
│   │   ├── treasury_adapter.py
│   │   └── edgar_adapter.py
│   ├── normalizer.py              # Raw → canonical type conversion
│   ├── corporate_actions.py       # Adjustment factor computation
│   ├── dedup.py                   # Cross-source deduplication
│   └── tiering.py                 # Free → paid upgrade manager
│
├── sim/                           # SENTINEL Instrument Master
│   ├── __init__.py
│   ├── master.py                  # Instrument CRUD and lifecycle
│   ├── openfigi.py                # OpenFIGI resolution client
│   ├── identifier.py              # CUSIP/ISIN/RIC/FIGI/ticker mapping
│   └── taxonomy.py                # GICS, SIC, NAICS classifications
│
├── sfe/                           # SENTINEL Filing Engine
│   ├── __init__.py
│   ├── edgar_client.py            # Raw EDGAR HTTP client
│   ├── xbrl_parser.py             # companyfacts, companyconcept, frames
│   ├── form_10k.py
│   ├── form_10q.py
│   ├── form_8k.py
│   ├── form_4.py
│   ├── form_13f.py
│   ├── form_13dg.py
│   ├── form_defproxy.py           # DEF 14A
│   ├── form_s1.py
│   ├── form_nport.py              # N-PORT mutual fund holdings
│   ├── form_adv.py
│   ├── form_d.py                  # Reg D private placements
│   ├── pit_store.py               # Point-in-time fact store
│   └── bulk_loader.py             # companyfacts.zip nightly backfill
│
├── sod/                           # SENTINEL Ownership Database
│   ├── __init__.py
│   ├── thirteenf.py               # 13F-HR parser + time series
│   ├── form4.py                   # Insider transaction parser + signal
│   ├── activist.py                # 13D/13G tracker
│   ├── congressional.py           # STOCK Act eFD disclosures
│   └── signals.py                 # Ownership change signal generators
│
├── sbe/                           # SENTINEL Backtesting Engine
│   ├── __init__.py
│   ├── strategy.py                # SentinelStrategy base class
│   ├── vectorbt_backend.py        # VectorBT research adapter
│   ├── nautilus_backend.py        # NautilusTrader production adapter
│   ├── metrics.py                 # 24-metric suite
│   ├── walk_forward.py            # Walk-forward + anchored validation
│   ├── dsr.py                     # Deflated Sharpe Ratio + PBO
│   ├── corporate_actions.py       # Backtest adjustment pipeline
│   ├── universe.py                # Universe construction + survivorship
│   └── report.py                  # Tearsheet generator
│
├── sse/                           # SENTINEL Screener Engine
│   ├── __init__.py
│   ├── screener.py                # Core screener engine
│   ├── criteria/
│   │   ├── fundamental.py         # 30 fundamental criteria
│   │   ├── technical.py           # 25 technical criteria
│   │   ├── ownership.py           # 10 ownership/insider criteria
│   │   ├── options_flow.py        # Put/call ratio, unusual activity
│   │   ├── fixed_income.py        # Bond screening criteria
│   │   ├── crypto.py              # On-chain / crypto criteria
│   │   └── custom.py              # @sentinel.factor decorator SDK
│   ├── tradingview_bridge.py      # TV Screener Python library wrapper
│   ├── universe.py                # Universe builder
│   ├── alerts.py                  # Threshold-cross alert engine
│   └── persistence.py             # Save/load/share screens
│
├── stu/                           # SENTINEL Terminal UI
│   ├── __init__.py
│   ├── app.py                     # Streamlit main entry point
│   ├── panels/
│   │   ├── watchlist.py
│   │   ├── chart.py               # TradingView Lightweight Charts bridge
│   │   ├── des_card.py            # Security description card
│   │   ├── news_stream.py
│   │   ├── filings_stream.py
│   │   ├── screener_panel.py
│   │   ├── portfolio_panel.py
│   │   ├── strategy_panel.py
│   │   ├── macro_panel.py
│   │   └── order_panel.py
│   ├── command_bar.py             # Bloomberg-style function dispatcher
│   ├── workspace.py               # Panel layout manager
│   └── websocket_client.py        # FastAPI WebSocket consumer
│
├── see/                           # SENTINEL Execution Engine
│   ├── __init__.py
│   ├── paper_simulator.py         # Paper trading sim (NautilusTrader BacktestNode)
│   ├── brokers/
│   │   ├── alpaca_broker.py       # Alpaca paper + live
│   │   ├── ib_broker.py           # Interactive Brokers (ibapi)
│   │   ├── binance_broker.py      # Binance spot + futures (CCXT + NT adapter)
│   │   ├── kraken_broker.py       # Kraken (CCXT)
│   │   └── oanda_broker.py        # OANDA FX
│   ├── risk_engine.py             # Pre-trade risk checks
│   ├── promotion.py               # Strategy promotion state machine
│   ├── kill_switch.py             # Emergency halt
│   ├── order_manager.py           # OMS: lifecycle management
│   └── execution_log.py           # Complete audit trail
│
├── spr/                           # SENTINEL Portfolio & Risk Engine
│   ├── __init__.py
│   ├── portfolio.py               # Position tracking + real-time NAV
│   ├── attribution.py             # Brinson-Hood-Beebower decomposition
│   ├── risk.py                    # VaR, CVaR, factor risk
│   ├── optimizer.py               # PyPortfolioOpt + Riskfolio-Lib wrapper
│   ├── sizing.py                  # Kelly, vol-target, risk-parity sizers
│   └── correlation.py             # Correlation monitor + alerts
│
├── sil/                           # SENTINEL Intelligence Layer
│   ├── __init__.py
│   ├── rag_pipeline.py            # LlamaIndex + pgvector RAG
│   ├── chunker.py                 # Financial document chunking strategy
│   ├── embedder.py                # Embedding model wrapper (voyage/bge/minilm)
│   ├── retriever.py               # Hybrid BM25 + vector + RRF re-rank
│   ├── analyst.py                 # Instrument analysis generator
│   ├── nl_screener.py             # Natural language → SSE criteria translator
│   ├── strategy_generator.py      # NL → SentinelStrategy spec
│   ├── backtest_explainer.py      # Performance attribution in plain English
│   ├── finbert.py                 # Sentiment classification
│   └── mcp_server.py             # 15-tool MCP server (FastMCP)
│
├── snm/                           # SENTINEL News & Media Intelligence (NEW)
│   ├── __init__.py
│   ├── feeds/
│   │   ├── rss_feed.py            # Multi-source RSS ingestion
│   │   ├── finnhub_news.py        # Finnhub news API
│   │   ├── alphavantage_news.py   # AV news + sentiment
│   │   ├── gdelt_feed.py          # GDELT global events
│   │   ├── benzinga_feed.py       # Benzinga (free tier)
│   │   └── sec_8k_feed.py         # Real-time 8-K press releases
│   ├── transcription/
│   │   ├── whisper_engine.py      # OpenAI Whisper (local) transcription
│   │   ├── earnings_call.py       # Earnings call audio → transcript
│   │   ├── podcast_pipeline.py    # Podcast RSS → download → transcribe → index
│   │   ├── fed_speech.py          # Fed/ECB speech ingestion + NLP
│   │   └── youtube_transcript.py  # YouTube financial channel transcripts
│   ├── nlp/
│   │   ├── entity_tagger.py       # spaCy NER → FIGI-linked entities
│   │   ├── sentiment.py           # FinBERT sentence-level sentiment
│   │   ├── event_extractor.py     # Earnings guidance, management changes, M&A signals
│   │   └── macro_signal.py        # Fed tone, ECB hawkish/dovish detection
│   ├── signal_bridge.py           # News event → trading signal generator
│   └── news_store.py              # PostgreSQL + pgvector news archive
│
├── sma/                           # SENTINEL Macro Analyzer
│   ├── __init__.py
│   ├── fred_client.py             # fedfred wrapper + series library
│   ├── central_banks.py           # ECB, BoE, BoC, BoJ, RBA data
│   ├── yield_curve.py             # Curve construction + spread analytics
│   ├── cot_report.py              # CFTC COT parser (NEW)
│   ├── regime.py                  # HMM + rule-based regime detector
│   ├── economic_calendar.py       # Release schedule + consensus
│   └── macro_dashboard.py         # Cross-asset macro composite
│
├── sbx/                           # SENTINEL Bond Analytics
│   ├── __init__.py
│   ├── quantlib_engine.py         # QuantLib Python wrapper
│   ├── curve_builder.py           # Yield curve from FRED data
│   ├── bond_math.py               # Duration, convexity, DV01, OAS, z-spread
│   ├── trace_client.py            # FINRA TRACE corporate bond data
│   ├── emma_client.py             # MSRB EMMA muni bond data
│   └── credit_analytics.py        # Spread analysis, rating transitions
│
├── api/                           # Internal FastAPI service layer
│   ├── __init__.py
│   ├── main.py                    # FastAPI app, CORS, lifespan
│   ├── routes/
│   │   ├── data.py                # /data/* endpoints
│   │   ├── screen.py              # /screen/* endpoints
│   │   ├── backtest.py            # /backtest/* endpoints
│   │   ├── portfolio.py           # /portfolio/* endpoints
│   │   ├── orders.py              # /orders/* endpoints (paper only by default)
│   │   └── intelligence.py        # /intel/* endpoints
│   └── websocket.py               # Live WebSocket feeds for STU
│
├── tests/
│   ├── unit/                      # Per-module unit tests
│   ├── integration/               # Cross-module integration tests
│   ├── financial_evals/           # Backtest regression tests
│   │   ├── eval_spy_buyhold.py    # SPY buy-hold benchmark must match within 2bps
│   │   ├── eval_momentum.py       # UMD factor must show positive IS Sharpe
│   │   └── eval_pit_integrity.py  # Point-in-time data integrity check
│   └── fixtures/                  # Test data: AAPL 5y daily, 10-K XML, 13F XML
│
├── scripts/
│   ├── bootstrap.py               # First-run setup: DB schema, seed data
│   ├── backfill.py                # Historical data backfill runner
│   ├── podcast_subscribe.py       # Manage podcast feed subscriptions
│   └── strategy_promote.py        # Interactive promotion wizard
│
├── infra/
│   ├── docker-compose.yml         # Full stack definition
│   ├── docker-compose.dev.yml     # Dev overrides (hot reload, debug ports)
│   ├── postgres/
│   │   ├── init.sql               # Full database schema
│   │   └── migrations/            # Alembic migration files
│   ├── timescaledb/
│   │   └── hypertables.sql        # TimescaleDB hypertable definitions
│   ├── redis/
│   │   └── redis.conf             # Redis configuration
│   └── grafana/
│       └── dashboards/            # Pre-built monitoring dashboards
│
├── config/
│   ├── settings.yaml              # Default non-secret config
│   ├── providers.yaml             # Data provider tiers and fallback chains
│   ├── strategies/                # Strategy spec YAML files
│   ├── screens/                   # Saved screen definitions
│   └── podcasts.yaml              # Podcast subscription list
│
├── .env.example                   # Template for secrets (never committed)
├── pyproject.toml                 # Poetry-managed dependencies
├── Makefile                       # Dev workflow shortcuts
└── README.md
```

**File count estimate:** ~280 Python files, averaging ~180 lines each = ~50,400 LOC. Tests add ~20,000 LOC.

---

## PART 2 — DOCKER COMPOSE STACK

```yaml
# infra/docker-compose.yml
services:

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: sentinel
      POSTGRES_USER: sentinel
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports: ["5432:5432"]
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U sentinel"]
      interval: 10s
      timeout: 5s
      retries: 5

  timescaledb:
    image: timescale/timescaledb:latest-pg16
    environment:
      POSTGRES_DB: sentinel_ts
      POSTGRES_USER: sentinel
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - timescale_data:/var/lib/postgresql/data
      - ./timescaledb/hypertables.sql:/docker-entrypoint-initdb.d/hypertables.sql
    ports: ["5433:5432"]
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server /usr/local/etc/redis/redis.conf
    volumes:
      - redis_data:/data
      - ./redis/redis.conf:/usr/local/etc/redis/redis.conf
    ports: ["6379:6379"]
    restart: unless-stopped

  sentinel_api:
    build: { context: .., dockerfile: infra/Dockerfile }
    command: uvicorn sentinel.api.main:app --host 0.0.0.0 --port 8000 --reload
    env_file: ../.env
    volumes: ["..:/app", "../data:/data"]
    ports: ["8000:8000"]
    depends_on: [postgres, timescaledb, redis]
    restart: unless-stopped

  sentinel_ingestion:
    build: { context: .., dockerfile: infra/Dockerfile }
    command: python -m sentinel.sds.daemon
    env_file: ../.env
    volumes: ["..:/app", "../data:/data"]
    depends_on: [postgres, timescaledb, redis]
    restart: unless-stopped

  sentinel_filing:
    build: { context: .., dockerfile: infra/Dockerfile }
    command: python -m sentinel.sfe.daemon
    env_file: ../.env
    volumes: ["..:/app", "../data:/data"]
    depends_on: [postgres, timescaledb, redis]
    restart: unless-stopped

  sentinel_news:
    build: { context: .., dockerfile: infra/Dockerfile }
    command: python -m sentinel.snm.daemon
    env_file: ../.env
    volumes: ["..:/app", "../data:/data", "../models:/models"]
    depends_on: [postgres, redis]
    restart: unless-stopped

  sentinel_execution:
    build: { context: .., dockerfile: infra/Dockerfile }
    command: python -m sentinel.see.daemon
    env_file: ../.env
    volumes: ["..:/app", "../data:/data"]
    depends_on: [postgres, timescaledb, redis]
    restart: unless-stopped

  sentinel_ui:
    build: { context: .., dockerfile: infra/Dockerfile }
    command: streamlit run sentinel/stu/app.py --server.port 8501
    env_file: ../.env
    volumes: ["..:/app"]
    ports: ["8501:8501"]
    depends_on: [sentinel_api]
    restart: unless-stopped

  sentinel_mcp:
    build: { context: .., dockerfile: infra/Dockerfile }
    command: python -m sentinel.sil.mcp_server
    env_file: ../.env
    ports: ["8002:8002"]
    depends_on: [sentinel_api]
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports: ["3000:3000"]
    restart: unless-stopped

volumes:
  postgres_data:
  timescale_data:
  redis_data:
  grafana_data:
```

---

## PART 3 — COMPLETE DATABASE SCHEMA

### 3.1 — PostgreSQL (postgres — port 5432)

```sql
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================
-- INSTRUMENT MASTER (SIM)
-- ============================================
CREATE TABLE instrument (
    figi            TEXT PRIMARY KEY,
    composite_figi  TEXT,
    share_class_figi TEXT,
    ticker          TEXT,
    exch_code       TEXT,
    mic             TEXT,          -- ISO 10383 Market Identifier Code
    name            TEXT,
    asset_class     TEXT,          -- Equity, Fixed Income, Commodity, Index, etc.
    sec_type        TEXT,          -- Common Stock, ETF, Mutual Fund, etc.
    sec_type2       TEXT,
    currency        TEXT,
    cusip           TEXT,
    isin            TEXT,
    sedol           TEXT,
    lei             TEXT,
    cik             TEXT,
    active          BOOLEAN DEFAULT TRUE,
    listing_date    DATE,
    delisting_date  DATE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE instrument_alias (
    id          BIGSERIAL PRIMARY KEY,
    figi        TEXT REFERENCES instrument(figi),
    source      TEXT,              -- 'ticker', 'ric', 'cusip', 'isin', etc.
    value       TEXT,
    valid_from  DATE,
    valid_to    DATE,
    UNIQUE (source, value, valid_from)
);

CREATE TABLE instrument_event (
    id          BIGSERIAL PRIMARY KEY,
    figi        TEXT REFERENCES instrument(figi),
    ts          TIMESTAMPTZ,
    event_type  TEXT,              -- IPO, DELIST, SPLIT, MERGER, RENAME, SPINOFF
    payload     JSONB,
    source      TEXT
);

CREATE INDEX idx_instrument_ticker ON instrument(ticker);
CREATE INDEX idx_instrument_cusip ON instrument(cusip);
CREATE INDEX idx_instrument_isin ON instrument(isin);
CREATE INDEX idx_alias_value ON instrument_alias(source, value);

-- ============================================
-- FUNDAMENTAL DATA (SFE)
-- ============================================
CREATE TABLE fundamental_fact (
    id              BIGSERIAL PRIMARY KEY,
    cik             TEXT,
    figi            TEXT,
    concept         TEXT,          -- us-gaap:Revenues, us-gaap:EPS, etc.
    accession       TEXT,          -- SEC accession number (authoritative source)
    filed_at        TIMESTAMPTZ,
    period_type     TEXT,          -- instant, duration
    period_start    DATE,
    period_end      DATE,
    value           NUMERIC,
    unit            TEXT,
    form            TEXT,
    frame           TEXT
);

CREATE INDEX idx_ff_cik_concept ON fundamental_fact(cik, concept, filed_at);
CREATE INDEX idx_ff_pit ON fundamental_fact(figi, concept, filed_at);

CREATE TABLE filing_index (
    accession       TEXT PRIMARY KEY,
    cik             TEXT,
    figi            TEXT,
    form_type       TEXT,
    filed_at        TIMESTAMPTZ,
    period_of_report DATE,
    document_url    TEXT,
    indexed_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- OWNERSHIP DATA (SOD)
-- ============================================
CREATE TABLE institutional_holding (
    id              BIGSERIAL PRIMARY KEY,
    filer_cik       TEXT,
    filer_name      TEXT,
    period          DATE,          -- Quarter end date
    figi            TEXT,
    cusip           TEXT,
    shares          BIGINT,
    market_value    BIGINT,        -- In thousands USD
    voting_sole     BIGINT,
    voting_shared   BIGINT,
    voting_none     BIGINT,
    accession       TEXT,
    filed_at        TIMESTAMPTZ,
    UNIQUE (filer_cik, period, cusip)
);

CREATE INDEX idx_ih_figi ON institutional_holding(figi, period);
CREATE INDEX idx_ih_filer ON institutional_holding(filer_cik, period);

CREATE TABLE insider_transaction (
    id              BIGSERIAL PRIMARY KEY,
    accession       TEXT,
    issuer_cik      TEXT,
    issuer_figi     TEXT,
    insider_cik     TEXT,
    insider_name    TEXT,
    title           TEXT,
    transaction_date DATE,
    transaction_code TEXT,         -- P=purchase, S=sale, A=award, M=option_exercise
    shares          NUMERIC,
    price_per_share NUMERIC,
    shares_after    NUMERIC,
    ownership_type  TEXT,          -- D=direct, I=indirect
    plan_10b5_1     BOOLEAN,       -- TRUE = pre-arranged plan (weaker signal)
    filed_at        TIMESTAMPTZ
);

CREATE INDEX idx_it_issuer_date ON insider_transaction(issuer_figi, transaction_date);
CREATE INDEX idx_it_code ON insider_transaction(transaction_code, transaction_date);

CREATE TABLE activist_position (
    id              BIGSERIAL PRIMARY KEY,
    filer_cik       TEXT,
    filer_name      TEXT,
    issuer_cik      TEXT,
    issuer_figi     TEXT,
    percent_owned   NUMERIC,
    form_type       TEXT,          -- SC 13D or SC 13G
    filing_type     TEXT,          -- initial, amendment
    amendment_no    INT,
    intent          TEXT,          -- active/passive
    accession       TEXT,
    filed_at        TIMESTAMPTZ
);

CREATE TABLE congressional_trade (
    id              BIGSERIAL PRIMARY KEY,
    member_name     TEXT,
    chamber         TEXT,          -- Senate, House
    transaction_date DATE,
    disclosure_date  DATE,
    ticker          TEXT,
    figi            TEXT,
    asset_type      TEXT,
    transaction_type TEXT,         -- Purchase, Sale, Exchange
    amount_range_low BIGINT,
    amount_range_high BIGINT,
    source_url      TEXT,
    filing_id       TEXT
);

CREATE INDEX idx_ct_figi ON congressional_trade(figi, transaction_date);

-- ============================================
-- NEWS & DOCUMENTS (SNM / SIL)
-- ============================================
CREATE TABLE news_item (
    id              TEXT PRIMARY KEY,   -- hash(source + headline + ts)
    ts              TIMESTAMPTZ,
    source          TEXT,
    headline        TEXT,
    body            TEXT,
    url             TEXT,
    author          TEXT,
    sentiment       FLOAT,              -- FinBERT: -1 to +1
    relevance       FLOAT,
    figis           TEXT[],
    tickers         TEXT[],
    topics          TEXT[],
    embedding       vector(1536)
);

CREATE TABLE transcript (
    id              TEXT PRIMARY KEY,
    ts              TIMESTAMPTZ,
    source_type     TEXT,              -- earnings_call, fed_speech, podcast, youtube
    company_cik     TEXT,
    company_figi    TEXT,
    title           TEXT,
    body            TEXT,              -- Full transcript text
    duration_secs   INT,
    sentiment_avg   FLOAT,
    guidance_flags  JSONB,             -- Detected guidance changes
    embedding       vector(1536),
    indexed_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE podcast_feed (
    id              BIGSERIAL PRIMARY KEY,
    name            TEXT,
    rss_url         TEXT UNIQUE,
    category        TEXT,              -- macro, earnings, sector, general
    active          BOOLEAN DEFAULT TRUE,
    last_checked    TIMESTAMPTZ
);

-- Full-text search indexes
CREATE INDEX idx_news_fts ON news_item USING GIN(to_tsvector('english', headline || ' ' || COALESCE(body, '')));
CREATE INDEX idx_news_embedding ON news_item USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_transcript_embedding ON transcript USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_news_figis ON news_item USING GIN(figis);

-- ============================================
-- PORTFOLIO & RISK (SPR)
-- ============================================
CREATE TABLE portfolio (
    id              TEXT PRIMARY KEY,
    name            TEXT,
    currency        TEXT DEFAULT 'USD',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    mode            TEXT DEFAULT 'paper'  -- paper, live
);

CREATE TABLE position (
    id              BIGSERIAL PRIMARY KEY,
    portfolio_id    TEXT REFERENCES portfolio(id),
    figi            TEXT,
    side            TEXT,              -- LONG, SHORT
    quantity        NUMERIC,
    avg_cost        NUMERIC,
    opened_at       TIMESTAMPTZ,
    closed_at       TIMESTAMPTZ,
    unrealized_pnl  NUMERIC,
    realized_pnl    NUMERIC
);

CREATE TABLE order_record (
    client_oid      TEXT PRIMARY KEY,
    portfolio_id    TEXT REFERENCES portfolio(id),
    broker_oid      TEXT,
    figi            TEXT,
    side            TEXT,
    order_type      TEXT,
    quantity        NUMERIC,
    limit_price     NUMERIC,
    status          TEXT,
    filled_qty      NUMERIC DEFAULT 0,
    avg_fill_price  NUMERIC,
    commission      NUMERIC,
    submitted_at    TIMESTAMPTZ,
    filled_at       TIMESTAMPTZ,
    cancelled_at    TIMESTAMPTZ
);

-- ============================================
-- STRATEGY PROMOTION (SEE)
-- ============================================
CREATE TABLE strategy_spec (
    id              TEXT PRIMARY KEY,
    name            TEXT,
    version         INT DEFAULT 1,
    spec_yaml       TEXT,              -- Full strategy specification
    code_hash       TEXT,              -- Hash of strategy code
    status          TEXT DEFAULT 'research',  -- research|paper|capped_live|full_live|paused|retired
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    promoted_at     TIMESTAMPTZ
);

CREATE TABLE promotion_gate (
    id              BIGSERIAL PRIMARY KEY,
    strategy_id     TEXT REFERENCES strategy_spec(id),
    gate            TEXT,              -- backtest_dsr|oos_validation|paper_soak|human_approval
    status          TEXT,              -- pending|passed|failed
    value           JSONB,
    evaluated_at    TIMESTAMPTZ
);

-- ============================================
-- MACRO DATA (SMA)
-- ============================================
CREATE TABLE cot_report (
    id              BIGSERIAL PRIMARY KEY,
    report_date     DATE,
    market_name     TEXT,
    commodity_code  TEXT,
    commercial_long BIGINT,
    commercial_short BIGINT,
    commercial_net  BIGINT,
    noncomm_long    BIGINT,
    noncomm_short   BIGINT,
    noncomm_net     BIGINT,
    nonreport_long  BIGINT,
    nonreport_short BIGINT,
    open_interest   BIGINT,
    UNIQUE (report_date, commodity_code)
);

CREATE INDEX idx_cot_date ON cot_report(report_date, market_name);
```

### 3.2 — TimescaleDB Hypertables (port 5433)

```sql
-- OHLCV bars
CREATE TABLE bar (
    ts          TIMESTAMPTZ NOT NULL,
    figi        TEXT NOT NULL,
    timeframe   TEXT NOT NULL,         -- 1m, 5m, 1h, 1d, 1w
    open        NUMERIC,
    high        NUMERIC,
    low         NUMERIC,
    close       NUMERIC,
    volume      NUMERIC,
    vwap        NUMERIC,
    source      TEXT,
    PRIMARY KEY (ts, figi, timeframe)
);
SELECT create_hypertable('bar', 'ts', chunk_time_interval => INTERVAL '1 week');
SELECT add_compression_policy('bar', INTERVAL '1 month');
CREATE INDEX idx_bar_figi ON bar(figi, ts DESC);

-- Tick data
CREATE TABLE tick (
    ts          TIMESTAMPTZ NOT NULL,
    figi        TEXT NOT NULL,
    price       NUMERIC,
    size        NUMERIC,
    side        TEXT,                  -- B, S, U (buy/sell/unknown)
    source      TEXT,
    PRIMARY KEY (ts, figi, source)
);
SELECT create_hypertable('tick', 'ts', chunk_time_interval => INTERVAL '1 day');
SELECT add_compression_policy('tick', INTERVAL '7 days');

-- Quote (L1 bid/ask)
CREATE TABLE quote (
    ts          TIMESTAMPTZ NOT NULL,
    figi        TEXT NOT NULL,
    bid         NUMERIC,
    bid_size    NUMERIC,
    ask         NUMERIC,
    ask_size    NUMERIC,
    source      TEXT,
    PRIMARY KEY (ts, figi)
);
SELECT create_hypertable('quote', 'ts', chunk_time_interval => INTERVAL '1 day');

-- Economic / macro time series
CREATE TABLE macro_series (
    ts          TIMESTAMPTZ NOT NULL,
    series_id   TEXT NOT NULL,
    value       DOUBLE PRECISION,
    vintage     DATE,
    PRIMARY KEY (ts, series_id, COALESCE(vintage, '1900-01-01'::DATE))
);
SELECT create_hypertable('macro_series', 'ts', chunk_time_interval => INTERVAL '1 year');

-- Portfolio NAV time series
CREATE TABLE nav_history (
    ts          TIMESTAMPTZ NOT NULL,
    portfolio_id TEXT NOT NULL,
    nav         NUMERIC,
    cash        NUMERIC,
    gross_exposure NUMERIC,
    net_exposure NUMERIC,
    PRIMARY KEY (ts, portfolio_id)
);
SELECT create_hypertable('nav_history', 'ts', chunk_time_interval => INTERVAL '1 month');
```

---

## PART 4 — DATA TIERING ARCHITECTURE

Every adapter implements a tiering contract. Free tier runs by default. Paid tier activates when the corresponding API key is present in the environment and `SENTINEL_DATA_TIER=paid` is set.

### 4.1 — Tiering Pattern (base_adapter.py)

```python
from abc import ABC, abstractmethod
from sentinel.core.config import settings
from sentinel.core.types import Bar, Quote, NewsItem

class BaseAdapter(ABC):
    tier: str = "free"             # "free" | "paid"
    requires_key: bool = True
    fallback: str | None = None    # adapter name to fall back to on failure

    @property
    def is_active(self) -> bool:
        if not self.requires_key:
            return True
        return self.api_key is not None

    @property
    def api_key(self) -> str | None:
        return getattr(settings, self.key_env_var, None)

    @abstractmethod
    async def get_bars(self, figi: str, tf: str, start, end) -> list[Bar]: ...

    @abstractmethod
    async def get_quote(self, figi: str) -> Quote: ...

    def health_check(self) -> DataHealthEvent: ...
```

### 4.2 — Provider Tiers & Upgrade Map

```yaml
# config/providers.yaml

equity_quotes:
  realtime_sip:                        # Full consolidated tape
    paid_providers:
      - name: alpaca_algo_trader
        cost: $99/mo
        activation: ALPACA_ALGO_KEY
        latency: <100ms
      - name: polygon_starter
        cost: $29/mo
        activation: POLYGON_API_KEY
        latency: <500ms
  realtime_iex:                        # IEX-only (free)
    free_providers:
      - name: alpaca_basic
        cost: $0
        activation: ALPACA_KEY
        latency: <500ms
  delayed_15min:                       # Free fallback
    free_providers:
      - name: finnhub
        cost: $0
        activation: FINNHUB_KEY
        latency: 15min

equity_ohlcv_intraday:
  paid_providers:
    - name: polygon_starter
      cost: $29/mo
      activation: POLYGON_API_KEY
      depth: 20+ years, 1-min
    - name: alpaca_algo_trader
      cost: $99/mo
      activation: ALPACA_ALGO_KEY
      depth: unlimited
  free_providers:
    - name: alpha_vantage
      cost: $0
      activation: AV_KEY
      depth: 2 years, 5-min (5 calls/min limit)
    - name: yfinance
      cost: $0
      activation: null
      depth: 2 months 1-min (unofficial)

news:
  paid_providers:
    - name: newsapi_pro
      cost: $449/mo
      activation: NEWSAPI_KEY
      coverage: 150K+ sources, real-time
    - name: benzinga_pro
      cost: $99/mo
      activation: BENZINGA_KEY
      coverage: financial-specific, fast
  free_providers:
    - name: finnhub_news
      cost: $0
      activation: FINNHUB_KEY
    - name: gdelt
      cost: $0
      activation: null
      latency: 15min
    - name: rss_aggregator
      cost: $0
      activation: null
      sources: [reuters_rss, ap_business, ft_rss, bloomberg_markets_rss]

fixed_income:
  paid_providers:
    - name: polygon_bonds
      cost: $199/mo
      activation: POLYGON_API_KEY_BONDS
    - name: cbonds
      cost: enterprise
  free_providers:
    - name: finra_trace
      cost: $0
      activation: null
      coverage: US corporate bonds, 15-min delayed transactions
    - name: msrb_emma
      cost: $0
      activation: null
      coverage: municipal bonds
    - name: fred
      cost: $0
      activation: FRED_KEY
      coverage: US Treasuries, agency bonds

options:
  paid_providers:
    - name: polygon_options
      cost: $79/mo (Stocks Starter)
      activation: POLYGON_API_KEY
      coverage: real-time options chain, full Greeks
    - name: cboe_livevol
      cost: enterprise
  free_providers:
    - name: yfinance_options
      cost: $0
      activation: null
      coverage: EOD options chains (unstable)
    - name: cboe_free
      cost: $0
      activation: null
      coverage: VIX, put/call ratios, settlement prices

crypto:
  free_providers:
    - name: ccxt_binance
      cost: $0
      activation: BINANCE_KEY  # optional — public endpoints work without key
      coverage: all Binance pairs, real-time
    - name: ccxt_kraken
      cost: $0
      coverage: all Kraken pairs, real-time
    - name: coingecko
      cost: $0
      activation: COINGECKO_KEY  # optional
      coverage: 14K+ coins, 1-min delay
```

### 4.3 — Tiering Decision Logic (tiering.py)

```python
class DataTierManager:
    """
    Selects the best available provider for each data category.
    Falls back gracefully from paid → free → cached.
    """
    def get_provider(self, category: str, require_tier: str = "any") -> BaseAdapter:
        providers = self._load_providers(category)
        for p in providers:
            if require_tier == "paid" and p.tier != "paid":
                continue
            if p.is_active and p.health_status == "OK":
                return p
        # fall back to last known good cache
        return self._get_cached_provider(category)

    def upgrade_suggestions(self) -> list[UpgradeSuggestion]:
        """Returns list of paid upgrades that would improve data quality, with cost/benefit."""
        ...
```

---

## PART 5 — NEWS & MEDIA INTELLIGENCE SPEC (Module SNM)

This is the most novel module — the organ the PRD underspecified.

### 5.1 — News Source Hierarchy

```
TIER 1 — Real-time (< 5 second lag)
  Paid: Benzinga Pro API, NewsAPI Pro, Bloomberg FEED (if ever affordable)
  Free: SEC EDGAR RSS (8-K press releases — same as Bloomberg KEY DEV)
        EDGAR filing alerts via EFTS streaming
        Finnhub WebSocket news (free key, delayed ~2 min)

TIER 2 — Near-real-time (5 sec – 15 min)
  Free: Alpha Vantage News API (free key)
        GDELT 2.0 (15-min refresh, 70+ languages, 100+ countries)
        RSS aggregator:
          https://feeds.reuters.com/reuters/businessNews
          https://rss.app/feeds/v1.1/tYoiMSBc0rj5KSmN.json  (AP Business)
          https://feeds.a.dj.com/rss/RSSMarketsMain.xml (WSJ markets)
          https://www.ft.com/?format=rss (FT free RSS)
          https://feeds.bloomberg.com/markets/news.rss (Bloomberg markets)

TIER 3 — Daily/scheduled
  CFTC COT reports (every Friday ~3:30pm ET) — cftc.gov
  Congressional disclosures (Senate eFD API + House PTR PDFs)
  Fed speeches (federalreserve.gov/newsevents/)
  ECB speeches (ecb.europa.eu/press/)
  SEC EDGAR bulk downloads (nightly)
```

### 5.2 — Podcast Transcription Pipeline

The most novel capability. Completely free using local Whisper.

```python
# snm/transcription/podcast_pipeline.py

SENTINEL_PODCAST_FEEDS = [
    # Macro / Rates
    {"name": "Bloomberg Odd Lots", "rss": "...", "category": "macro"},
    {"name": "Bloomberg Surveillance", "rss": "...", "category": "macro"},
    {"name": "FT Markets", "rss": "...", "category": "macro"},
    {"name": "Macro Voices", "rss": "...", "category": "macro"},
    {"name": "Forward Guidance (Blockworks)", "rss": "...", "category": "macro"},
    # Earnings / Company
    {"name": "Invest Like the Best", "rss": "...", "category": "company_research"},
    {"name": "Acquired", "rss": "...", "category": "company_research"},
    {"name": "Founder's Field Guide", "rss": "...", "category": "company_research"},
    # Trading / Quant
    {"name": "Top Traders Unplugged", "rss": "...", "category": "quant"},
    {"name": "Chat With Traders", "rss": "...", "category": "trading"},
    # Crypto
    {"name": "Bankless", "rss": "...", "category": "crypto"},
    {"name": "Unchained", "rss": "...", "category": "crypto"},
]

class PodcastPipeline:
    """
    RSS → download audio → Whisper transcription → NER → sentiment → pgvector index
    Runs on a schedule (hourly check for new episodes)
    Whisper model: large-v3 (8GB VRAM) on RTX 3090 or medium (5GB) on Mac mini
    Transcription speed: ~10x realtime on RTX 3090 (1-hour podcast in ~6 minutes)
    """
    async def process_episode(self, episode_url: str, feed_meta: dict) -> Transcript:
        audio_path = await self._download(episode_url)
        text = await self._transcribe(audio_path)          # faster-whisper
        entities = await self._extract_entities(text)      # spaCy + custom NER
        sentiment = await self._score_sentiment(text)      # FinBERT sentence-level
        guidance = await self._extract_signals(text)       # custom signal extractor
        embedding = await self._embed(text[:8000])         # voyage-finance-2 or bge
        return Transcript(
            source_type="podcast",
            title=episode_meta.title,
            body=text,
            figis=entities.figis,
            sentiment_avg=sentiment.avg,
            guidance_flags=guidance,
            embedding=embedding
        )
```

**Tools:**
- `faster-whisper` (GPU-optimized CTranslate2 backend, MIT license) — 4× faster than original Whisper on same hardware
- `spaCy` with `en_core_web_lg` + custom financial NER model for company name → FIGI resolution
- `ProsusAI/finBERT` for sentence-level positive/negative/neutral classification
- `youtube-transcript-api` (MIT) for YouTube captions without audio download

### 5.3 — Earnings Call Live Transcription

```
Sources (free):
1. SEC EDGAR EX-99 attachments on 8-K filings — text of earnings press releases
   Timing: Usually filed 1–3 hours after market close on earnings day
2. Seeking Alpha earnings call transcripts (free, available ~same day)
   Access: scrape with rate limiting; their text is public domain since it's
   transcription of public investor calls
3. Company IR websites — many post transcripts within 24 hours
4. Direct IR call audio streams — most public companies broadcast via
   Lumi, West, or Chorus Call; these are public URL streams that can
   be captured and transcribed in real-time with faster-whisper

Real-time pipeline (advanced):
1. Subscribe to SEC EDGAR 8-K RSS feed
2. When 8-K item 2.02 (Results of Operations) detected → flag company
3. Monitor company IR website for webcast URL
4. Stream audio from webcast → faster-whisper live transcription → chunk
5. Each chunk → FinBERT sentiment + entity extraction → publish on bus
6. SNM → SSE → alert if management tone changes materially mid-call
```

### 5.4 — Fed / Central Bank Intelligence

```python
# snm/nlp/macro_signal.py
# Fed hawkish/dovish scoring system

FED_HAWKISH_PHRASES = [
    "inflation remains too high", "further increases", "sustained period",
    "restrictive stance", "higher for longer", "not confident"
]
FED_DOVISH_PHRASES = [
    "inflation has eased", "balanced risks", "beginning to reduce",
    "significant progress", "confident", "appropriate to reduce"
]

class FedSentimentAnalyzer:
    """
    Scores FOMC minutes, speeches, and press conference transcripts
    on a hawkish/dovish scale. Emits MacroSignal events.
    Sources:
    - federalreserve.gov/monetarypolicy/fomccalendars.htm (minutes, statements)
    - federalreserve.gov/newsevents/speeches.htm (governor speeches)
    - fed.gov/FOMC press conference transcripts (same-day)
    All free, public domain.
    """
```

---

## PART 6 — MISSING DATA SOURCES (PRD GAPS RESOLVED)

### 6.1 — CFTC Commitment of Traders (COT)

```python
# sma/cot_report.py

COT_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_2024.zip"
# CFTC releases every Friday ~3:30pm ET. Free. No API key needed.
# Files: fut_disagg_txt_YYYY.zip (disaggregated, preferred)
#        fut_fin_txt_YYYY.zip (financial traders supplemental)
#        fut_legacy_txt_YYYY.zip (legacy format, longest history back to 1986)

class COTParser:
    """
    Parse CFTC COT report CSV → structured time series.
    Key fields per contract:
      - Market name (e.g., 'GOLD - COMMODITY EXCHANGE INC.')
      - Commercial long/short/net
      - Non-commercial (large speculator) long/short/net
      - Non-reportable (small speculator) long/short/net
      - Open interest
      - Change vs. prior week for all above
    
    Key signals:
      - Commercial net positioning extremes → contrarian signal
        (commercials are the "smart money" in physical commodities)
      - Non-commercial net extreme → momentum confirmation
      - COT Index = (current - 3yr_min) / (3yr_max - 3yr_min)
        → <20 = extreme short, >80 = extreme long
    
    Best for: Gold, Silver, Crude Oil, Natural Gas, Corn, Wheat,
              Soybeans, Copper, S&P 500, T-Notes, FX futures (EUR, GBP, JPY, CAD)
    """
    async def fetch_and_parse(self, year: int) -> list[COTReport]:
        ...
    
    def cot_index(self, market: str, lookback_weeks: int = 156) -> float:
        """Returns 0-100 index. <20 = historically net short, >80 = historically net long."""
        ...
```

### 6.2 — Congressional Trading (STOCK Act)

```python
# sod/congressional.py

SENATE_EFD_API = "https://efts.senate.gov/LATEST/search-index?q=&dateRange=custom&fromDate={}&toDate={}&senator={}"
HOUSE_PTR_URL  = "https://disclosures-clerk.house.gov/FinancialDisclosure"
# Note: House filings are PDFs; require PDF parser or third-party aggregator
# Quiver Quantitative provides parsed data free via their API (freemium)
# Capitol Trades provides CSV exports

QUIVER_CONGRESSIONAL = "https://api.quiverquant.com/beta/historical/congresstrading/{ticker}"
# Quiver Quant free tier: 100 calls/day

class CongressionalTracker:
    """
    Parse STOCK Act periodic transaction reports.
    
    Key fields:
      - Member name + chamber
      - Transaction date
      - Asset name / ticker
      - Transaction type (Purchase, Sale, Exchange)
      - Amount range ($1K-$15K, $15K-$50K, $50K-$100K, $100K-$250K, $250K-$500K, $500K-$1M, $1M+)
      - Disclosure date (transactions must be reported within 30-45 days)
    
    Alpha evidence:
      - Jochec (2020): Senators outperform market by 12% annually on purchased stocks
      - Karadas-Pettine-Strauss (2021): Congressional purchases predict positive returns
      - Pelosi-effect trades (NVDA, MSFT, Apple options) documented +15% 6-month alpha
      - Senate Intelligence/Armed Services committees have most documented edge
    
    Signal construction:
      - Buy signal: 3+ members purchase same ticker within 30 days
      - Sell signal: 2+ members sell same ticker, especially defense/tech during policy shifts
      - Committee weighting: multiply signal strength by committee relevance to sector
    """
```

### 6.3 — Options Flow Intelligence

```python
# sse/criteria/options_flow.py

class OptionsFlowCriteria:
    """
    Free data sources for options intelligence:
    
    1. CBOE Market Statistics (cboe.com/market_statistics/) — free, daily
       - Total put/call ratio (equity + index)
       - Equity-only put/call ratio (more predictive than total)
       - VIX, VVIX, SKEW — all on FRED (VIXCLS, VVIX, SKEWX series)
       - VIX term structure: VIX9D (VXST), VIX (30d), VIX3M, VIX6M — FRED
    
    2. Unusual Whales (unusualwhales.com) — freemium
       - Free tier: recent unusual options flow (delayed 15 min)
       - Key: volume/OI ratio > 5× with large notional premium = smart money signal
    
    3. yfinance options chains — free, EOD
       - All strikes × expiries for any US equity
       - Derive IV surface from Black-Scholes inversion using py_vollib
    
    4. SEC large trader reporting (Form 13H) — partially available on EDGAR
    
    Key unusual options criteria:
      - Volume/OI ratio > 5× (unusual flow vs standing interest)
      - Premium > $1M notional (institutional size)
      - Time to expiry < 30 days (short-dated = conviction)
      - OTM by 10-30% (risk-reward asymmetry)
      - Calls on depressed stocks OR puts on extended stocks
    """
    
    def cboe_put_call_ratio(self, lookback_days: int = 30) -> pd.Series:
        """Fetch from FRED: CBOE Equity Put/Call Ratio (CBOEEQUITYCALL)"""
        ...
    
    def vix_term_structure(self) -> dict[str, float]:
        """
        Returns: {
            'vix9d': float, 'vix30': float, 'vix3m': float, 'vix6m': float,
            'vvix': float, 'skew': float,
            'contango': float,  # vix30/vix9d - 1; positive = normal; negative = backwardation (stress)
            'term_slope': float  # (vix6m - vix9d) / time_delta
        }
        """
        ...
```

---

## PART 7 — BROKER INTEGRATION SPECS FOR AUTONOMOUS TRADING

### 7.1 — Supported Brokers

| Broker | Asset Classes | API Type | Paper Trading | Commission | Best For |
|---|---|---|---|---|---|
| **Alpaca** | US Equities, ETFs, Crypto | REST + WebSocket | Yes (free) | $0 commissions | US equity strategies, fast prototyping |
| **Interactive Brokers** | Stocks, Options, Futures, Bonds, FX, Crypto | ibapi (Python) + IBKR REST | Yes (paper account) | Low ($0.005/share) | Multi-asset, institutional-grade |
| **Binance** | Crypto spot + futures (USDT-M, COIN-M) | REST + WebSocket | Yes (testnet) | 0.1% spot, 0.02-0.05% futures | Crypto strategies, perps, funding rate arb |
| **Kraken** | Crypto + USD | REST | Limited | 0.26% maker | Crypto, EUR-denominated |
| **OANDA** | FX, CFDs, Indices | REST + Streaming | Yes (practice) | Spread-based | FX strategies |
| **Tradovate** | US Futures (CME) | WebSocket | Yes | $0.25/contract + exchange | Commodity trend following, CTA strategies |

### 7.2 — Alpaca Integration (Primary Day 1 Broker)

```python
# see/brokers/alpaca_broker.py
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live import StockDataStream

class AlpacaBroker(BaseBroker):
    """
    Free paper trading. $9/mo Algo Trader Plus for full SIP real-time data.
    
    Capabilities:
    - Commission-free US equities and ETFs
    - Fractional shares (min $1)
    - Extended hours trading (4am-8pm ET)
    - Options trading (requires separate approval)
    - Crypto: BTC, ETH, SOL, AVAX, DOGE, etc.
    - Short selling (requires margin account)
    
    Paper account: APCA-PAPER-API-KEY-ID + APCA-PAPER-API-SECRET-KEY
    Live account: APCA-LIVE-API-KEY-ID + APCA-LIVE-API-SECRET-KEY
    """
    
    PAPER_BASE_URL = "https://paper-api.alpaca.markets"
    LIVE_BASE_URL  = "https://api.alpaca.markets"
    
    async def submit_order(self, order: Order, mode: Literal["paper", "live"] = "paper") -> Fill:
        """Fail-Closed: mode defaults to 'paper'. 'live' requires SENTINEL_LIVE=1."""
        if mode == "live" and not settings.SENTINEL_LIVE:
            raise FailClosedError("Live orders require SENTINEL_LIVE=1 env var")
        ...
```

### 7.3 — Interactive Brokers Integration (Multi-Asset)

```python
# see/brokers/ib_broker.py
# Uses NautilusTrader's built-in IB adapter (best approach)
# Alternatively: ib_async (Python async wrapper over ibapi)

class IBBroker(BaseBroker):
    """
    NautilusTrader InteractiveBrokersInstrumentProvider +
    InteractiveBrokersDataClient +
    InteractiveBrokersExecutionClient
    
    IB TWS (Trader Workstation) or IB Gateway must be running locally.
    Paper trading: TWS paper account login
    Live trading: TWS live account login
    
    Market data subscriptions required (paid monthly via IB):
    - US Securities Snapshot + Futures Value Bundle: $10/mo
    - US Equity and Options Add-On Streaming Bundle: $4.50/mo
    - CME Group Futures: $30/mo (waived if >= $30 commissions/mo)
    
    IB Gateway (headless) recommended over TWS for production:
    docker run -d --name ibgateway ib-gateway:latest
    """
    
    IB_HOST = "127.0.0.1"
    IB_PORT_LIVE = 7496
    IB_PORT_PAPER = 7497
    
    SUPPORTED_ASSET_CLASSES = [
        "STK",   # Equities
        "OPT",   # Equity Options
        "FUT",   # Futures
        "CASH",  # FX
        "BOND",  # Corporate + Treasury bonds
        "CFD",   # CFDs
        "CMDTY", # Spot commodities
        "CRYPTO" # Crypto (limited)
    ]
```

### 7.4 — Autonomous Trading Promotion State Machine

This is the most critical safety system in SENTINEL. No strategy reaches live execution without passing all gates.

```
STATES:
  RESEARCH → PAPER → CAPPED_LIVE → FULL_LIVE → PAUSED → RETIRED

GATES (must all PASS to transition):

[RESEARCH → PAPER]
  Gate 1: backtest_dsr
    - Deflated Sharpe Ratio > 0 (strategy not explained by data snooping alone)
    - Out-of-sample Sharpe ≥ 0.5× in-sample Sharpe
    - Max drawdown ≤ configured limit (default 25%)
    - Minimum sample: 3 years of backtest data
  Gate 2: walk_forward_stability
    - Walk-forward validation across ≥ 3 out-of-sample windows
    - ≥ 2/3 windows must be profitable
  Gate 3: data_integrity_check
    - No look-ahead bias detected (timestamp audit)
    - Survivorship bias mitigation confirmed
  Gate 4: human_review
    - Human marks strategy as paper-approved via CLI

[PAPER → CAPPED_LIVE]
  Gate 5: paper_soak
    - Minimum 30 calendar days paper trading
    - Live paper Sharpe ≥ 0.5× backtest Sharpe
    - Daily loss never exceeded risk limits in paper
    - Slippage within 2× modeled assumptions
  Gate 6: execution_model_validation
    - Average fill quality score ≥ 0.8 (actual fill vs. mid at signal time)
  Gate 7: human_approval_live
    - Human explicitly types "APPROVE LIVE: {strategy_id}" in CLI
    - Two-factor confirmation code required
  Config:
    - Max position size: 2% of portfolio (capped)
    - Max total allocation: 10% of portfolio
    - Daily loss limit: 1% of portfolio

[CAPPED_LIVE → FULL_LIVE]
  Gate 8: capped_live_performance
    - 14+ days of capped live trading
    - Live Sharpe (annualized) ≥ 0.4
    - No daily loss limit breaches
    - Slippage within 1.5× modeled
  Gate 9: human_approval_full
    - Second human approval required
    - New confirmation code
  Config:
    - Max position size: configurable (default 5%)
    - Max total allocation: configurable (default 30%)
    - Daily loss limit: configurable (default 2%)

[ANY STATE → PAUSED]
  Auto-triggers:
    - Daily P&L < -daily_loss_limit
    - Kill switch file detected (~/.sentinel/KILL)
    - Rolling 3-day Sharpe < 0.3× full-sample
    - Slippage exceeds 3× model for 3+ consecutive trades
    - Any data health alert for strategy's primary data source
    - Market halt detected
  Manual: sentinel strategy pause {id}

[PAUSED → RESEARCH/PAPER/CAPPED_LIVE]
  - Must re-pass all gates for the target state
  - Exception: brief pauses due to market halt (auto-resume on market re-open)

KILL SWITCH IMPLEMENTATION:
  # File-based kill switch works even if Python process hangs
  # Checked every 100ms by a watchdog thread
  if Path("~/.sentinel/KILL").expanduser().exists():
      await flatten_all_positions()
      await cancel_all_orders()
      sys.exit(1)
```

---

## PART 8 — API CONTRACTS (Module-to-Module)

Every inter-module call has a defined contract. Error handling is explicit. No silent failures.

### 8.1 — SDS → Bus Contract

```python
# All events emitted by SDS adapters go through this publish interface
async def publish(event: SentinelEvent, channel: str) -> None:
    """
    1. Validate event schema (Pydantic)
    2. Write to Redis Stream: XADD sentinel:{channel} * {event.model_dump_json()}
    3. Write to TimescaleDB (for bars/ticks) or PostgreSQL (for news/filings)
    4. Emit DataHealthEvent: source=adapter.name, latency_ms=..., status='OK'|'STALE'|'ERROR'
    Raises: DataQualityError if event fails validation
    """
```

### 8.2 — SFE → SIL Contract

```python
# Every filing parsed by SFE triggers indexing in SIL
async def on_filing_parsed(filing: ParsedFiling) -> None:
    """
    1. Chunk document by section (Item 1, MD&A, Risk Factors, Financials, etc.)
    2. Embed each chunk (voyage-finance-2 or bge fallback)
    3. Store in PostgreSQL transcript table with pgvector embedding
    4. Update BM25 tsvector index
    5. Emit FilingIndexedEvent on bus
    Raises: EmbeddingError, ChunkingError
    """
```

### 8.3 — SSE → SBE Contract

```python
# Screener results can seed backtesting universes
async def universe_from_screen(
    screen_id: str,
    as_of: datetime,
    max_instruments: int = 500
) -> Universe:
    """
    Returns list of FIGIs passing the screen criteria as of as_of date.
    Respects point-in-time discipline (uses pit_store for fundamentals).
    Raises: UniverseEmptyError if no instruments qualify
    """
```

### 8.4 — SBE → SEE Contract (Promotion)

```python
# SBE outputs a BacktestResult that feeds the SEE promotion gates
@dataclass
class BacktestResult:
    strategy_id: str
    period_start: date
    period_end: date
    is_sharpe: float
    oos_sharpe: float
    dsr: float             # Deflated Sharpe Ratio
    max_drawdown: float
    walk_forward_results: list[WalkForwardWindow]
    look_ahead_clean: bool
    survivorship_corrected: bool
    metrics: PerformanceMetrics  # Full 24-metric suite

async def submit_for_promotion(result: BacktestResult) -> PromotionDecision:
    """
    Evaluates all Gate 1–4 criteria.
    Returns: APPROVED_FOR_PAPER | REJECTED | NEEDS_MORE_DATA
    """
```

### 8.5 — SIL MCP Server Spec

```python
# sil/mcp_server.py — using FastMCP (from MCP SDK)
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SENTINEL", version="1.0.0")

@mcp.tool()
async def search_filings(
    query: str,
    form_types: list[str] = ["10-K", "10-Q", "8-K"],
    date_from: str = None,   # YYYY-MM-DD
    date_to: str = None,
    tickers: list[str] = None,
    limit: int = 10
) -> list[FilingSearchResult]:
    """Search SEC EDGAR filings using full-text + semantic search."""

@mcp.tool()
async def get_company_facts(
    identifier: str,         # ticker, FIGI, CIK, or ISIN
    concepts: list[str] = None,  # e.g. ["Revenues", "EPS", "Assets"]
    as_of: str = None        # Point-in-time date
) -> CompanyFactsResult:
    """Get XBRL financial facts for a company."""

@mcp.tool()
async def run_screener(
    criteria: dict,          # e.g. {"pe_ratio": {"lt": 15}, "rsi": {"lt": 30}}
    universe: str = "sp500", # sp500, russell1000, all_us, all_crypto, custom
    limit: int = 50
) -> list[ScreenResult]:
    """Screen instruments by fundamental, technical, or alternative criteria."""

@mcp.tool()
async def get_ownership(
    ticker: str,
    include_13f: bool = True,
    include_form4: bool = True,
    include_congressional: bool = True
) -> OwnershipSummary:
    """Get institutional holdings, insider transactions, and congressional trading."""

@mcp.tool()
async def backtest(
    strategy_spec: str,      # YAML or Python code
    universe: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0
) -> BacktestResult:
    """Run a backtest. Returns full 24-metric report."""

@mcp.tool()
async def submit_paper_order(
    ticker: str,
    side: str,               # BUY or SELL
    quantity: float,
    order_type: str = "market",
    limit_price: float = None
) -> OrderConfirmation:
    """Submit a paper trading order. NEVER executes live without SENTINEL_LIVE=1."""

@mcp.tool()
async def get_portfolio_state(portfolio_id: str = "default") -> PortfolioState:
    """Get current positions, NAV, P&L."""

@mcp.tool()
async def fetch_macro(
    series: list[str],       # FRED series IDs e.g. ["DGS10", "T10Y2Y", "VIXCLS"]
    start_date: str = None
) -> dict[str, list[MacroPoint]]:
    """Fetch macroeconomic time series from FRED."""

@mcp.tool()
async def get_news_sentiment(
    ticker: str,
    lookback_days: int = 7,
    sources: list[str] = None
) -> NewsSentimentResult:
    """Get recent news with FinBERT sentiment scores."""

@mcp.tool()
async def price_bond(
    cusip_or_isin: str,
    as_of_date: str,
    yield_override: float = None  # If None, uses FRED market yield
) -> BondAnalyticsResult:
    """Price a bond and compute duration, convexity, DV01 using QuantLib."""

@mcp.tool()
async def detect_regime(
    asset: str = "SPY",
    model: str = "hmm",      # hmm, rule_based, hybrid
    lookback_days: int = 252
) -> RegimeResult:
    """Detect current market regime."""

@mcp.tool()
async def get_cot_positioning(
    market: str,             # e.g. "GOLD", "CRUDE OIL", "S&P 500"
    cot_index_lookback: int = 156  # weeks
) -> COTResult:
    """Get CFTC COT positioning and COT Index for a futures market."""

@mcp.tool()
async def summarize_transcript(
    company_ticker: str,
    transcript_type: str = "earnings_call",  # earnings_call, fed_speech, podcast
    most_recent: bool = True
) -> TranscriptSummary:
    """Get AI summary of most recent earnings call or relevant transcript."""

@mcp.tool()
async def promote_strategy(
    strategy_id: str,
    target_state: str        # paper, capped_live, full_live
) -> PromotionDecision:
    """Evaluate strategy promotion gates and return decision with gate details."""

@mcp.tool()
async def kill_switch(
    reason: str,
    portfolio_id: str = "all"
) -> KillSwitchResult:
    """Emergency: flatten all positions and halt trading. Irreversible until manual reset."""
```

---

## PART 9 — CONFIGURATION & SECRETS MANAGEMENT

### 9.1 — .env.example (never commit the actual .env)

```bash
# ─── DATABASE ───────────────────────────────────────────
POSTGRES_PASSWORD=changeme_strong_password
TIMESCALE_PASSWORD=changeme_strong_password

# ─── MODE ───────────────────────────────────────────────
SENTINEL_LIVE=0             # Set to 1 ONLY to enable live trading. Default: paper mode.
SENTINEL_DATA_TIER=free     # free | paid — controls which provider tiers are activated

# ─── FREE DATA APIs (all have generous free tiers) ───────
FINNHUB_KEY=your_key_here                # finnhub.io — free tier, 60 req/min
ALPHA_VANTAGE_KEY=your_key_here          # alphavantage.co — free tier, 5 req/min
FMP_KEY=your_key_here                    # financialmodelingprep.com — 250 calls/day free
FRED_KEY=your_key_here                   # fred.stlouisfed.org — free, 120 req/min
COINGECKO_KEY=your_key_here              # optional — demo key for higher limits

# ─── PAPER TRADING (free) ────────────────────────────────
ALPACA_PAPER_KEY=your_key_here
ALPACA_PAPER_SECRET=your_secret_here

# ─── PAID DATA (optional, activate by setting SENTINEL_DATA_TIER=paid) ─────
ALPACA_LIVE_KEY=                         # $9/mo Algo Trader Plus for full SIP
ALPACA_LIVE_SECRET=
POLYGON_KEY=                             # $29/mo Starter for deep intraday history
EODHD_KEY=                               # $79/mo all-in-one global data
NEWSAPI_KEY=                             # $449/mo for real-time news
BENZINGA_KEY=                            # $99/mo for fast financial news

# ─── LIVE BROKERS (only active when SENTINEL_LIVE=1) ─────
IB_GATEWAY_HOST=127.0.0.1
IB_GATEWAY_PORT=7497                     # 7497=paper, 7496=live
IB_CLIENT_ID=1
BINANCE_API_KEY=                         # Only needed for live trading; testnet works without
BINANCE_API_SECRET=

# ─── AI MODELS ──────────────────────────────────────────
ANTHROPIC_KEY=your_key_here              # Claude API
OPENAI_KEY=                              # Optional
OLLAMA_HOST=http://localhost:11434       # Local LLM fallback
LOCAL_LLM_MODEL=llama3.1:70b            # Or mistral:7b for lighter footprint
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5  # Free local embedding model

# ─── OPTIONAL INTEGRATIONS ──────────────────────────────
QUIVER_QUANT_KEY=                        # quiverquant.com — congressional trading data
```

### 9.2 — Configuration Hierarchy

```
Precedence (highest first):
1. Environment variables (from .env or actual env)
2. config/settings.yaml (non-secrets, committed to repo)
3. Module defaults (hardcoded safe defaults)

Never in code:
- API keys
- Passwords
- Connection strings with credentials
- Live trading flags
```

---

## PART 10 — TESTING STRATEGY

### 10.1 — Testing Layers

```
LAYER 1: Unit Tests (sentinel/tests/unit/)
  Per-module, per-function
  Mock all external API calls (pytest-mock + httpretty)
  Coverage target: ≥ 80% per module
  Run time: < 30 seconds

LAYER 2: Integration Tests (sentinel/tests/integration/)
  Test module-to-module contracts
  Use local test database (docker-compose.test.yml)
  Cover: SDS→TimescaleDB, SFE→PostgreSQL, SIL→pgvector, SEE→paper broker
  Run time: < 5 minutes

LAYER 3: Financial Evaluation Tests (sentinel/tests/financial_evals/)
  These are NOT traditional unit tests.
  They verify financial correctness and serve as regression guards.
  
  eval_spy_buyhold.py:
    - Fetch SPY 2015-2024 from yfinance
    - Run buy-and-hold via SBE
    - Assert CAGR within 2bps of known SPY CAGR for that period
    - Assert max drawdown within 5% of known SPY max DD (March 2020)
  
  eval_momentum_factor.py:
    - Construct 12-1 month momentum portfolio on Russell 1000
    - Assert top-decile outperforms bottom-decile (known result)
    - Assert Sharpe > 0.3 for IS period (published result)
  
  eval_pit_integrity.py:
    - Take any 100 fundamental data points
    - Assert that fundamental_fact.filed_at <= the date of any query using that fact
    - Assert no future filings visible in any backtest signal
    - This test MUST pass before any strategy can be promoted to paper
  
  eval_bond_math.py:
    - Price 5 known benchmark bonds (10yr UST, 5yr UST, 2yr UST, IG corp, HY corp)
    - Assert yield matches FRED published yield within 1bp
    - Assert duration matches textbook formula within 0.01
    - Assert DV01 matches Bloomberg-published reference value within $0.50/million

LAYER 4: Strategy Regression Tests (per strategy in config/strategies/)
  When a strategy is in PAPER or higher state:
  - Run backtest on latest data weekly
  - Assert rolling IS Sharpe hasn't degraded >30% since initial approval
  - Emit StrategyHealthAlert if degradation detected
```

---

## PART 11 — MAKEFILE (Developer Workflow)

```makefile
# Makefile — SENTINEL development workflow

.PHONY: up down dev test eval lint migrate bootstrap

# Start full stack
up:
	docker compose -f infra/docker-compose.yml up -d
	@echo "SENTINEL running: API=:8000  UI=:8501  Grafana=:3000"

# Stop stack
down:
	docker compose -f infra/docker-compose.yml down

# Development mode (hot reload, verbose logging)
dev:
	docker compose -f infra/docker-compose.yml -f infra/docker-compose.dev.yml up

# Run all tests
test:
	pytest sentinel/tests/unit/ sentinel/tests/integration/ -v --tb=short

# Run financial evaluations (slower, checks correctness)
eval:
	pytest sentinel/tests/financial_evals/ -v --tb=long

# Lint + type check
lint:
	ruff check sentinel/
	mypy sentinel/ --ignore-missing-imports

# Database migrations
migrate:
	alembic upgrade head

# First-run setup
bootstrap:
	python scripts/bootstrap.py
	@echo "Bootstrap complete. Configure .env and run 'make up'."

# Backfill historical data for a symbol
backfill:
	python scripts/backfill.py --symbol $(SYMBOL) --start $(START) --end $(END)

# Interactive strategy promotion wizard
promote:
	python scripts/strategy_promote.py --strategy-id $(STRATEGY_ID)

# Kill switch
kill:
	touch ~/.sentinel/KILL
	@echo "KILL SWITCH ACTIVATED. All strategies halted. Remove ~/.sentinel/KILL to resume."
```

---

## PART 12 — GENERATION 0 SPRINT: EXACT ACCEPTANCE CRITERIA

This is the only sprint DanteForge should generate code for in the first pass. It must be deployable and working on the Mac mini within one week.

**Deliverables:**
1. `infra/docker-compose.yml` running with PostgreSQL + TimescaleDB + Redis
2. `infra/postgres/init.sql` creating all schema tables from Part 3
3. `sentinel/core/config.py` loading from .env
4. `sentinel/core/types.py` — all Pydantic schemas for Bar, Tick, Quote, NewsItem, CorporateAction, FilingAlert
5. `sentinel/core/bus.py` — async event bus with Redis Streams bridge
6. `sentinel/sds/adapters/yfinance_adapter.py` — bars + quote + fundamentals
7. `sentinel/sds/adapters/finnhub_adapter.py` — real-time quotes + news
8. `sentinel/sds/adapters/fred_adapter.py` — macro series
9. `sentinel/sds/normalizer.py` — raw API → canonical types
10. `sentinel/sds/tiering.py` — provider selection logic
11. `sentinel/sim/openfigi.py` — FIGI resolution client
12. `sentinel/stu/app.py` — Streamlit app with: watchlist (10 instruments), chart (yfinance EOD), fundamentals card (P/E, Market Cap, Revenue)
13. `sentinel/api/main.py` — FastAPI with `/health`, `/data/quote/{ticker}`, `/data/bars/{ticker}`
14. `tests/unit/` — unit tests for all above
15. `tests/financial_evals/eval_spy_buyhold.py` — must pass before Generation 1

**Generation 0 validation test (must pass before proceeding):**
```
Given: SENTINEL stack is up (make up)
When: I open http://localhost:8501
Then: I can see a watchlist with AAPL, MSFT, SPY, BTC-USD, GC=F (Gold Futures)
And:  Each row shows real-time quote (or max 15-min delayed)
And:  Clicking AAPL shows a candlestick chart with 1-year history
And:  Clicking AAPL shows a fundamentals card with P/E, Market Cap, Revenue
And:  GET /data/quote/AAPL returns JSON with bid, ask, last, volume
And:  GET /data/bars/AAPL?timeframe=1d&start=2024-01-01 returns 252 bars
And:  eval_spy_buyhold.py passes
Total time from `make bootstrap` to all passing: ≤ 4 hours on Mac mini
```

---

## PART 13 — PYPROJECT.TOML (Dependency Declaration)

```toml
[tool.poetry]
name = "sentinel"
version = "0.1.0"
description = "Sovereign AI-Native Market Intelligence and Trading Platform"
authors = ["Ricky Porras <ricky@realempanada.com>"]
license = "Apache-2.0"

[tool.poetry.dependencies]
python = "^3.12"

# Core
pydantic = "^2.7"
pydantic-settings = "^2.2"
structlog = "^24.1"
python-dotenv = "^1.0"

# Data Infrastructure
asyncpg = "^0.29"           # PostgreSQL async driver
redis = "^5.0"              # Redis client
duckdb = "^0.10"            # In-process SQL for Parquet
pyarrow = "^16.0"           # Parquet / Arrow
alembic = "^1.13"           # Database migrations

# Financial Data — Free Tier
yfinance = "^0.2"
finnhub-python = "^2.4"
alpha-vantage = "^3.0"
fmpsdk = "^0.0.9"
fedfred = "^0.1"            # Modern FRED API client
sec-edgar-api = "^1.1"
edgartools = "^2.0"         # Best SEC EDGAR Python library
ccxt = "^4.3"               # 108+ crypto exchanges

# Financial Data — Paid Tier (installed but inactive without keys)
polygon-api-client = "^1.14"
alpaca-py = "^0.26"
eodhd = "^1.0"

# Backtesting & Quant
nautilus-trader = "^1.0"    # Execution engine (LGPL v3)
vectorbt = "^0.26"          # Research backtesting
quantlib = "^1.34"          # Bond analytics (LGPL 2.1)
pyportfolioopt = "^1.5"     # Portfolio optimization
riskfolio-lib = "^6.1"      # Risk measures
empyrical-reloaded = "^0.6" # Performance metrics
quantstats = "^0.0.62"      # Performance reporting
py-vollib = "^1.0"          # Options Greeks
hmmlearn = "^0.3"           # Regime detection

# NLP / AI
transformers = "^4.40"      # FinBERT + financial NLP models
torch = "^2.3"              # PyTorch (local inference)
faster-whisper = "^1.0"     # Podcast/earnings transcription
spacy = "^3.7"              # NER
llama-index-core = "^0.10"  # RAG framework
llama-index-vector-stores-postgres = "^0.1"
pgvector = "^0.3"           # pgvector Python client
anthropic = "^0.27"         # Claude API
mcp = "^1.0"                # Model Context Protocol SDK

# Technical Analysis
pandas-ta = "^0.3"          # 130+ TA indicators (no TA-Lib C dependency)

# Web / API
fastapi = "^0.111"
uvicorn = "^0.30"
streamlit = "^1.35"
streamlit-lightweight-charts = "^0.7"
httpx = "^0.27"
websockets = "^12.0"

# Data manipulation
pandas = "^2.2"
numpy = "^1.26"
polars = "^0.20"            # Faster than pandas for large datasets
scipy = "^1.13"

[tool.poetry.dev-dependencies]
pytest = "^8.2"
pytest-asyncio = "^0.23"
pytest-mock = "^3.14"
httpretty = "^1.1"
ruff = "^0.4"               # Linting
mypy = "^1.10"              # Type checking
```

---

## PART 14 — WHAT WE OVERLOOKED + FINAL CHECKLIST

### Things Added vs PRD

| Item | Status |
|---|---|
| CFTC COT reports (free, weekly, powerful macro signal) | ✅ Added to SMA + DBOM |
| Congressional trading (STOCK Act disclosures) | ✅ Added to SOD |
| Options flow intelligence (put/call ratio, unusual activity) | ✅ Added to SSE |
| Podcast transcription pipeline (Whisper) | ✅ Added to SNM |
| Earnings call live transcription | ✅ Added to SNM |
| Fed/ECB speech sentiment analysis | ✅ Added to SNM |
| Complete data tiering architecture (free → paid upgrade) | ✅ Added to SDS |
| Paid broker data subscription specs (IB market data, Alpaca tiers) | ✅ Added to broker specs |
| Data quality health monitoring layer | ✅ Added to core/health.py |
| Strategy promotion state machine (8-gate framework) | ✅ Added to SEE |
| Congressional trading (STOCK Act) | ✅ Added to SOD |
| Complete .env.example with all providers | ✅ Part 9 |
| Full Docker Compose with all 9 services | ✅ Part 2 |
| Complete PostgreSQL schema (DDL) | ✅ Part 3 |
| TimescaleDB hypertables | ✅ Part 3 |
| Complete pyproject.toml | ✅ Part 13 |
| Generation 0 exact acceptance criteria | ✅ Part 12 |
| Financial evaluation tests (not just unit tests) | ✅ Part 10 |
| MCP server with all 15 tools (full signatures) | ✅ Part 8 |
| DanteMind integration spec | 🔶 Placeholder — DanteMind PRD (April 2026) governs this |

### Things That Are Still Deliberately Out of Scope

| Item | Reason |
|---|---|
| Bloomberg IB Chat equivalent | Impossible — pure network effect |
| PitchBook private company depth | Data sourcing requires human analysts |
| Visible Alpha sell-side model line items | Licensed content, no free substitute at depth |
| LSEG Reuters wire real-time speed | Pay for Benzinga Pro ($99/mo) as substitute |
| CDSW / CDS pricing | Dealer-controlled OTC market, no retail access |
| Sub-millisecond HFT execution | Requires co-location; out of scope for Mac mini |

### Is This Enough to Start Building?

**Yes. Build Generation 0 now.**

The PRD answers: what to build and why.  
This TSD answers: how to build it, how every file connects, and what "done" means at each gate.

DanteForge has everything it needs to generate:
- Complete file tree (280 files)
- All database schemas
- All Pydantic type contracts
- All module API contracts
- All adapter interfaces
- Docker Compose stack
- Generation 0 acceptance criteria

The only thing not in this document is the actual implementation code — which is DanteForge's job.

**Recommended first prompt to DanteForge:**
> "Using the SENTINEL TSD, implement Generation 0. Generate the complete file tree first, then implement each file in order: infra/docker-compose.yml → infra/postgres/init.sql → sentinel/core/config.py → sentinel/core/types.py → sentinel/core/bus.py → sentinel/sds/base_adapter.py → sentinel/sds/adapters/yfinance_adapter.py → sentinel/sds/adapters/finnhub_adapter.py → sentinel/sds/adapters/fred_adapter.py → sentinel/sds/normalizer.py → sentinel/sds/tiering.py → sentinel/sim/openfigi.py → sentinel/api/main.py → sentinel/stu/app.py → tests/unit/ → tests/financial_evals/eval_spy_buyhold.py. Every file under 500 lines. No stubs. KiloCode compliance. Every module complete."

---

*SENTINEL TSD v1.0 — May 2026 — Dante Ecosystem*  
*This document is the construction blueprint. The PRD is the vision. DanteForge builds.*
