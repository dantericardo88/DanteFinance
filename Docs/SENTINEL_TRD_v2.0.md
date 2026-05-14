# SENTINEL Technical Requirements Document
**Version:** 2.1
**Date:** May 7, 2026
**Status:** Generation 1 — Build Complete (May 2026). See Gen 1 Audit below.
**Follows:** SENTINEL PRD v2.1
**Hardware Target:** Mac mini Apple Silicon (M-series), 16–32 GB RAM, 1 TB SSD
**Philosophy:** Sovereign. Self-hosted. AI-native. No mandatory cloud. Free by default.

---

## Part 1 — Architecture Overview

### 1.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STU (Terminal UI)                           │
│                    Streamlit + FastAPI WebSocket                     │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ HTTP / WebSocket
┌─────────────────────────────────▼───────────────────────────────────┐
│                        API Layer (FastAPI)                           │
│   /data  /screen  /backtest  /portfolio  /orders  /intel  /ws       │
└──┬────────┬────────┬────────┬────────┬────────┬────────┬────────────┘
   │        │        │        │        │        │        │
  SDS      SIM      SFE      SOD      SBE      SSE      SIL
  SMA      SBX      SNM      SPR      SEE     (MCP)
   │        │        │        │        │        │        │
┌──▼────────▼────────▼────────▼────────▼────────▼────────▼────────────┐
│                     Internal Event Bus                               │
│              asyncio queues + Redis Streams bridge                   │
└──┬──────────────────────────────────────────────────────────────────┘
   │
┌──▼───────────────────────────────────────────────────────┐
│                    Data Layer                             │
│  TimescaleDB (OHLCV, time series)                        │
│  PostgreSQL (filings, ownership, fundamentals, news)     │
│  pgvector (embeddings — financial documents)             │
│  Redis (cache, pub/sub, rate limiting, session state)    │
└──────────────────────────────────────────────────────────┘
```

### 1.2 Design Principles

**KiloCode Rule:** Every source file is under 500 lines. No monoliths.
**No Circular Imports:** Module dependency graph is a DAG. `core/` has no module-level imports.
**No Stubs:** Every file ships complete — no `raise NotImplementedError`.
**Event-Driven Internally:** Modules communicate via the internal event bus, not direct function calls across module boundaries.
**Secrets Never in Code:** All API keys, passwords, and tokens live in `.env` files (gitignored). `python-dotenv` + pydantic-settings for loading.

### 1.3 Repository Layout

```
sentinel/
├── core/                          # Shared primitives — imports by all modules
│   ├── __init__.py
│   ├── config.py                  # pydantic-settings Settings class
│   ├── bus.py                     # Internal event bus (asyncio + Redis Streams)
│   ├── types.py                   # All shared Pydantic v2 schemas
│   ├── health.py                  # DataHealthEvent monitor
│   ├── logging.py                 # structlog JSON structured logging
│   └── security.py                # Secret store wrapper
│
├── sds/                           # Data Spine
├── sim/                           # Instrument Master
├── sfe/                           # Filing Engine
├── sod/                           # Ownership Database
├── sbe/                           # Backtesting Engine
├── sse/                           # Screener Engine
├── stu/                           # Terminal UI
├── see/                           # Execution Engine
├── spr/                           # Portfolio & Risk
├── sil/                           # Intelligence Layer (AI/NLP/MCP)
├── sma/                           # Macro Analyzer
├── sbx/                           # Bond Analytics
├── snm/                           # News & Media Intelligence
│
├── api/                           # FastAPI service layer
│   ├── main.py
│   └── routes/
│       ├── data.py
│       ├── screen.py
│       ├── backtest.py
│       ├── portfolio.py
│       ├── orders.py
│       ├── intelligence.py
│       └── websocket.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── financial_evals/           # Backtest regression tests
│
├── infra/
│   ├── docker-compose.yml
│   ├── docker-compose.dev.yml
│   ├── postgres/init.sql
│   ├── timescaledb/hypertables.sql
│   ├── redis/redis.conf
│   └── grafana/dashboards/
│
├── config/
│   ├── settings.yaml
│   ├── providers.yaml             # Data provider tiers + fallback chains
│   ├── strategies/
│   ├── screens/
│   └── podcasts.yaml
│
├── .env.example
├── pyproject.toml                 # Poetry-managed dependencies
├── Makefile
└── README.md
```

---

## Gen 1 Build Audit — May 2026

### Files Shipped

| Module | Key Files | Status | BUILT Score |
|--------|-----------|--------|:-----------:|
| SDS | adapters/polygon_adapter.py, alpaca_adapter.py, __init__.py, normalizer.py | Complete | 1.9 avg |
| SIM | openfigi_client.py, instrument_master.py | Complete | — |
| SFE | xbrl_parser.py, form4_parser.py, form13f_parser.py | Complete | 1.8 avg |
| SOD | congressional.py | Complete | 2.0 avg |
| SMA | cot_report.py, regime.py | Complete | 3.4 avg |
| SBE | metrics.py, pbo.py, runner.py, strategies.py | Complete | 3.3 avg |
| SEE | promotion.py, broker.py | Complete | — |
| SIL | mcp_server.py, sentiment.py | Partial — rag.py missing | 1.3 avg |
| SSE | screener.py | Complete (DB empty) | 1.7 avg |
| SPR | portfolio.py | Complete | 1.3 avg |
| SBX | bond_analytics.py | Complete | 1.8 avg |
| SNM | news_feed.py | Complete | 0.7 avg |
| STU | terminal.py | Complete (24 fn handlers) | 2.6 avg |
| API | main.py + 7 routes | Complete | — |
| Infra | docker-compose.yml, init.sql, hypertables.sql, Dockerfile | Complete | — |
| Scripts | bootstrap.py, backfill.py | Complete | — |
| Tests | test_financial_evals.py (27+ tests) | Complete | — |

### Not Built in Gen 1 (Gen 2 Priorities)

| Missing Module | Target File | Impact |
|----------------|-------------|--------|
| RAG pipeline | sil/rag.py | Dims 51, 57 — zero score |
| NautilusTrader | sbe/nautilus_backend.py | Dim 62 — zero score |
| FINRA TRACE | sbx/trace_client.py | Dim 36 — zero score |
| Fama-French | spr/factor_model.py | Dim 79 — zero score |
| Social sentiment | snm/social_sentiment.py | Dim 85 — zero score |
| Economic calendar | sma/economic_calendar.py | Dim 44 — zero score |
| DCF / WACC | sfe/dcf_model.py | Dim 23 — zero score |
| Stress testing | spr/stress_test.py | Dim 83 — 1/10 |
| CB speech NLP | sma/cb_speech.py | Dim 45 — 1/10 |
| NL screener | sil/nl_screener.py | Dim 76 — 2/10 (regex only) |
| Correlation monitor | spr/correlation.py | Dim 80 — zero score |
| CCXT adapter | sds/ccxt_adapter.py | Dim 7, 106 — 2/10 |

### Known Bugs Fixed in Gen 1

| Bug | File | Fix |
|-----|------|-----|
| Double `.items()` on positions dict | spr/portfolio.py | `self._positions.items()..items()` → `self._positions.items():` |

### Operational Blockers (Must Fix Before Any Score Improves)

1. **Database is empty** — `make backfill` must be run to populate OHLCV, FRED, EDGAR, COT data
2. **sil/rag.py missing** — 30% of AI/NLP category is blocked on this single file
3. **DuckDB screener has 0 rows** — screener works but has no universe to screen

---

## Part 2 — Docker Compose Stack

### 2.1 Core Services

```yaml
# infra/docker-compose.yml
version: "3.9"
services:

  postgres:
    image: timescale/timescaledb:latest-pg16
    environment:
      POSTGRES_USER: sentinel
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: sentinel
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U sentinel"]
      interval: 10s

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
      - ./redis/redis.conf:/usr/local/etc/redis/redis.conf
    ports:
      - "6379:6379"
    command: redis-server /usr/local/etc/redis/redis.conf

  sentinel-api:
    build: .
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    environment:
      - DATABASE_URL=postgresql+asyncpg://sentinel:${POSTGRES_PASSWORD}@postgres:5432/sentinel
      - REDIS_URL=redis://redis:6379
    env_file: .env
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    volumes:
      - .:/app

  sentinel-terminal:
    build: .
    command: streamlit run stu/app.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    env_file: .env
    depends_on:
      - sentinel-api

  sentinel-workers:
    build: .
    command: python -m sentinel.workers
    env_file: .env
    depends_on:
      - postgres
      - redis

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}

volumes:
  postgres_data:
  redis_data:
  grafana_data:
```

---

## Part 3 — Database Schema

### 3.1 TimescaleDB Hypertables (Time Series)

```sql
-- infra/timescaledb/hypertables.sql

-- OHLCV data — all equity/crypto/FX/commodity bars
CREATE TABLE ohlcv (
    time        TIMESTAMPTZ     NOT NULL,
    figi        VARCHAR(12)     NOT NULL,
    open        NUMERIC(18,6)   NOT NULL,
    high        NUMERIC(18,6)   NOT NULL,
    low         NUMERIC(18,6)   NOT NULL,
    close       NUMERIC(18,6)   NOT NULL,
    volume      BIGINT          NOT NULL,
    vwap        NUMERIC(18,6),
    adj_factor  NUMERIC(12,8)   DEFAULT 1.0,  -- corporate action adjustment
    source      VARCHAR(32)     NOT NULL,      -- yfinance/polygon/alpaca/ccxt
    PRIMARY KEY (time, figi)
);
SELECT create_hypertable('ohlcv', 'time', chunk_time_interval => INTERVAL '1 month');
CREATE INDEX ON ohlcv (figi, time DESC);

-- FRED and macro time series
CREATE TABLE macro_series (
    time        TIMESTAMPTZ     NOT NULL,
    series_id   VARCHAR(64)     NOT NULL,
    value       NUMERIC(20,8),
    vintage     TIMESTAMPTZ,    -- ALFRED vintage date for point-in-time
    PRIMARY KEY (time, series_id)
);
SELECT create_hypertable('macro_series', 'time');

-- Options chain snapshots
CREATE TABLE options_chain (
    time            TIMESTAMPTZ     NOT NULL,
    underlying_figi VARCHAR(12)     NOT NULL,
    expiry          DATE            NOT NULL,
    strike          NUMERIC(12,2)   NOT NULL,
    option_type     CHAR(1)         NOT NULL,  -- C or P
    bid             NUMERIC(10,4),
    ask             NUMERIC(10,4),
    last            NUMERIC(10,4),
    volume          INTEGER,
    open_interest   INTEGER,
    iv              NUMERIC(8,6),
    delta           NUMERIC(8,6),
    gamma           NUMERIC(8,6),
    theta           NUMERIC(8,6),
    vega            NUMERIC(8,6),
    rho             NUMERIC(8,6),
    PRIMARY KEY (time, underlying_figi, expiry, strike, option_type)
);
SELECT create_hypertable('options_chain', 'time');

-- Data health events
CREATE TABLE data_health_events (
    time        TIMESTAMPTZ     NOT NULL,
    adapter     VARCHAR(64)     NOT NULL,
    event_type  VARCHAR(32)     NOT NULL,  -- staleness/gap/schema_drift/throttle
    severity    VARCHAR(16)     NOT NULL,  -- info/warning/critical
    details     JSONB,
    PRIMARY KEY (time, adapter, event_type)
);
SELECT create_hypertable('data_health_events', 'time');
```

### 3.2 PostgreSQL Structured Tables

```sql
-- infra/postgres/init.sql

-- Instrument master
CREATE TABLE instruments (
    figi            VARCHAR(12)     PRIMARY KEY,
    share_class_figi VARCHAR(12),
    composite_figi  VARCHAR(12),
    ticker          VARCHAR(16),
    name            VARCHAR(256),
    isin            VARCHAR(12),
    cusip           VARCHAR(9),
    sedol           VARCHAR(7),
    ric             VARCHAR(32),
    asset_class     VARCHAR(32),    -- equity/bond/crypto/fx/commodity
    market          VARCHAR(32),
    exchange_code   VARCHAR(8),
    currency        VARCHAR(3),
    country         VARCHAR(2),
    gics_sector     INTEGER,
    gics_group      INTEGER,
    gics_industry   INTEGER,
    gics_sub        INTEGER,
    sic_code        INTEGER,
    naics_code      INTEGER,
    ipo_date        DATE,
    delist_date     DATE,
    is_active       BOOLEAN         DEFAULT TRUE,
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     DEFAULT NOW()
);
CREATE INDEX ON instruments (ticker);
CREATE INDEX ON instruments (cusip);
CREATE INDEX ON instruments (isin);

-- XBRL financial facts (point-in-time)
CREATE TABLE financial_facts (
    id              BIGSERIAL       PRIMARY KEY,
    cik             VARCHAR(10)     NOT NULL,
    figi            VARCHAR(12),
    concept         VARCHAR(128)    NOT NULL,  -- us-gaap/Assets, etc.
    taxonomy        VARCHAR(32)     NOT NULL,  -- us-gaap / ifrs-full / dei
    label           VARCHAR(256),
    value           NUMERIC(24,4),
    unit            VARCHAR(32),
    period_start    DATE,
    period_end      DATE            NOT NULL,
    instant         DATE,
    form_type       VARCHAR(16),
    filed_date      DATE            NOT NULL,
    accession       VARCHAR(20),
    frame           VARCHAR(32),
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);
CREATE INDEX ON financial_facts (cik, concept, period_end DESC);
CREATE INDEX ON financial_facts (figi, concept, period_end DESC);
-- Partial index for common fundamental queries
CREATE INDEX ON financial_facts (concept, period_end DESC) WHERE taxonomy = 'us-gaap';

-- SEC filings index
CREATE TABLE filings (
    accession       VARCHAR(20)     PRIMARY KEY,
    cik             VARCHAR(10)     NOT NULL,
    figi            VARCHAR(12),
    form_type       VARCHAR(16)     NOT NULL,
    filed_date      TIMESTAMPTZ     NOT NULL,
    period_of_report DATE,
    filing_url      TEXT,
    document_url    TEXT,
    parsed          BOOLEAN         DEFAULT FALSE,
    parse_error     TEXT,
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);
CREATE INDEX ON filings (cik, form_type, filed_date DESC);
CREATE INDEX ON filings (figi, form_type, filed_date DESC);
CREATE INDEX ON filings (form_type, filed_date DESC);

-- Form 4 insider transactions
CREATE TABLE insider_transactions (
    id              BIGSERIAL       PRIMARY KEY,
    accession       VARCHAR(20)     NOT NULL,
    figi            VARCHAR(12),
    cik_issuer      VARCHAR(10),
    cik_owner       VARCHAR(10),
    owner_name      VARCHAR(256),
    owner_role      VARCHAR(64),    -- Officer/Director/10% Owner
    is_director     BOOLEAN,
    is_officer      BOOLEAN,
    is_ten_pct      BOOLEAN,
    transaction_date DATE,
    transaction_code CHAR(1),       -- P=Purchase S=Sale A=Award M=Exercise G=Gift F=Tax
    shares          NUMERIC(16,4),
    price_per_share NUMERIC(12,4),
    total_value     NUMERIC(18,2),
    shares_owned_after NUMERIC(16,4),
    is_10b5_1       BOOLEAN,        -- mandatory disclosure since April 2023
    filed_date      DATE,
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);
CREATE INDEX ON insider_transactions (figi, transaction_date DESC);
CREATE INDEX ON insider_transactions (transaction_code, transaction_date DESC);

-- 13F institutional holdings
CREATE TABLE institutional_holdings (
    id              BIGSERIAL       PRIMARY KEY,
    filing_id       VARCHAR(20),
    manager_cik     VARCHAR(10)     NOT NULL,
    manager_name    VARCHAR(256),
    period_of_report DATE           NOT NULL,
    figi            VARCHAR(12),
    cusip           VARCHAR(9),
    security_name   VARCHAR(256),
    value           BIGINT,         -- $ value in thousands
    shares          BIGINT,
    option_type     VARCHAR(4),     -- Put/Call/null
    voting_sole     BIGINT,
    voting_shared   BIGINT,
    voting_none     BIGINT,
    filed_date      DATE,
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);
CREATE INDEX ON institutional_holdings (figi, period_of_report DESC);
CREATE INDEX ON institutional_holdings (manager_cik, period_of_report DESC);

-- Congressional STOCK Act disclosures
CREATE TABLE congressional_trades (
    id              BIGSERIAL       PRIMARY KEY,
    member_name     VARCHAR(256)    NOT NULL,
    member_type     VARCHAR(16)     NOT NULL,  -- Senate / House
    state           VARCHAR(2),
    party           VARCHAR(16),
    transaction_date DATE,
    disclosure_date DATE            NOT NULL,
    asset_name      VARCHAR(256),
    figi            VARCHAR(12),
    ticker          VARCHAR(16),
    asset_type      VARCHAR(64),    -- Stock/Bond/Fund/Other
    transaction_type VARCHAR(32),   -- Purchase/Sale/Exchange
    amount_range    VARCHAR(32),    -- e.g., "$1,001 - $15,000"
    amount_min      INTEGER,
    amount_max      INTEGER,
    comment         TEXT,
    source_url      TEXT,
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);
CREATE INDEX ON congressional_trades (figi, transaction_date DESC);
CREATE INDEX ON congressional_trades (member_name, transaction_date DESC);
CREATE INDEX ON congressional_trades (disclosure_date DESC);

-- Macro regimes (HMM output)
CREATE TABLE macro_regimes (
    date            DATE            PRIMARY KEY,
    regime          VARCHAR(64)     NOT NULL,  -- Growth/Inflation, etc.
    regime_id       INTEGER         NOT NULL,  -- 0-3
    prob_0          NUMERIC(6,4),
    prob_1          NUMERIC(6,4),
    prob_2          NUMERIC(6,4),
    prob_3          NUMERIC(6,4),
    confidence      NUMERIC(6,4),
    updated_at      TIMESTAMPTZ     DEFAULT NOW()
);

-- Strategies
CREATE TABLE strategies (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(256)    NOT NULL,
    spec_yaml       TEXT            NOT NULL,
    status          VARCHAR(32)     NOT NULL DEFAULT 'backtest',
    dsr             NUMERIC(8,4),
    pbo             NUMERIC(8,4),
    is_sharpe       NUMERIC(8,4),
    paper_sharpe    NUMERIC(8,4),
    paper_max_dd    NUMERIC(8,4),
    paper_start_date DATE,
    live_sharpe     NUMERIC(8,4),
    promoted_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     DEFAULT NOW()
);

-- Orders and fills (complete audit trail)
CREATE TABLE orders (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id     UUID            REFERENCES strategies(id),
    broker          VARCHAR(32)     NOT NULL,
    broker_order_id VARCHAR(64),
    figi            VARCHAR(12),
    side            VARCHAR(4)      NOT NULL,  -- BUY/SELL
    order_type      VARCHAR(8)      NOT NULL,  -- MKT/LMT/STP/STPLMT
    qty             NUMERIC(16,4)   NOT NULL,
    limit_price     NUMERIC(12,4),
    stop_price      NUMERIC(12,4),
    status          VARCHAR(16)     NOT NULL DEFAULT 'PENDING',
    filled_qty      NUMERIC(16,4)   DEFAULT 0,
    avg_fill_price  NUMERIC(12,4),
    commission      NUMERIC(10,4)   DEFAULT 0,
    submitted_at    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    filled_at       TIMESTAMPTZ,
    cancelled_at    TIMESTAMPTZ,
    error_message   TEXT
);
CREATE INDEX ON orders (strategy_id, submitted_at DESC);
CREATE INDEX ON orders (figi, submitted_at DESC);

-- News archive with embeddings
CREATE TABLE news_articles (
    id              BIGSERIAL       PRIMARY KEY,
    source          VARCHAR(64)     NOT NULL,
    external_id     VARCHAR(256),
    headline        TEXT            NOT NULL,
    summary         TEXT,
    full_text       TEXT,
    url             TEXT,
    published_at    TIMESTAMPTZ     NOT NULL,
    entities        JSONB,          -- [{figi, name, mention_count}]
    sentiment_score NUMERIC(5,3),
    sentiment_label VARCHAR(16),    -- positive/negative/neutral/uncertain
    embedding       vector(1536),   -- voyage-finance-2 or text-embedding-3-small
    dedup_hash      VARCHAR(64)     UNIQUE,
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);
CREATE INDEX ON news_articles (published_at DESC);
CREATE INDEX ON news_articles USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- Entity-article junction
CREATE TABLE news_entity_links (
    article_id      BIGINT          REFERENCES news_articles(id),
    figi            VARCHAR(12),
    mention_count   INTEGER         DEFAULT 1,
    PRIMARY KEY (article_id, figi)
);
CREATE INDEX ON news_entity_links (figi, article_id DESC);
```

---

## Part 4 — Core Module Technical Specs

### 4.1 core/types.py — Canonical Pydantic v2 Schemas

```python
# core/types.py — excerpt showing key schemas
from pydantic import BaseModel, Field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Optional

class AssetClass(str, Enum):
    EQUITY = "equity"
    BOND = "bond"
    CRYPTO = "crypto"
    FX = "fx"
    COMMODITY = "commodity"
    OPTION = "option"
    FUTURE = "future"

class OHLCVBar(BaseModel):
    time: datetime
    figi: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    vwap: Optional[Decimal] = None
    adj_factor: Decimal = Decimal("1.0")
    source: str

class FinancialFact(BaseModel):
    cik: str
    figi: Optional[str]
    concept: str          # e.g. "us-gaap/Revenues"
    taxonomy: str         # "us-gaap" | "ifrs-full" | "dei"
    value: Optional[Decimal]
    unit: str
    period_start: Optional[date]
    period_end: date
    instant: Optional[date]
    form_type: str
    filed_date: date
    accession: str

class InsiderTransaction(BaseModel):
    accession: str
    figi: Optional[str]
    owner_name: str
    owner_role: str
    transaction_date: date
    transaction_code: str  # P/S/A/M/G/F
    shares: Decimal
    price_per_share: Optional[Decimal]
    total_value: Optional[Decimal]
    shares_owned_after: Optional[Decimal]
    is_10b5_1: Optional[bool]
    filed_date: date

class DataHealthEvent(BaseModel):
    time: datetime
    adapter: str
    event_type: str    # staleness/gap/schema_drift/throttle/ok
    severity: str      # info/warning/critical
    details: dict = {}

class MacroRegime(str, Enum):
    GROWTH_INFLATION = "Growth/Inflation"
    GROWTH_DEFLATION = "Growth/Deflation"
    CONTRACTION_INFLATION = "Contraction/Inflation"
    CONTRACTION_DEFLATION = "Contraction/Deflation"

class StrategyStatus(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    CAPPED_LIVE = "capped_live"
    FULL_AUTONOMOUS = "full_autonomous"
    PAUSED = "paused"
    RETIRED = "retired"
```

### 4.2 core/bus.py — Internal Event Bus

```python
# core/bus.py
import asyncio
from typing import Callable, Any
import redis.asyncio as aioredis
import json

class EventBus:
    """
    Dual-mode event bus:
    - asyncio queues for intra-process communication (low latency)
    - Redis Streams for cross-process / persistence (durability)
    """
    def __init__(self, redis_url: str):
        self._subscribers: dict[str, list[Callable]] = {}
        self._redis = aioredis.from_url(redis_url)

    async def publish(self, event_type: str, payload: Any, persist: bool = False):
        handlers = self._subscribers.get(event_type, [])
        for handler in handlers:
            asyncio.create_task(handler(payload))
        if persist:
            await self._redis.xadd(
                f"sentinel:{event_type}",
                {"payload": json.dumps(payload, default=str)}
            )

    def subscribe(self, event_type: str, handler: Callable):
        self._subscribers.setdefault(event_type, []).append(handler)
```

### 4.3 sds/base_adapter.py — Abstract Data Adapter

```python
# sds/base_adapter.py
from abc import ABC, abstractmethod
from core.types import OHLCVBar, DataHealthEvent
from datetime import datetime

class BaseAdapter(ABC):
    name: str
    rate_limit_per_min: int = 60

    @abstractmethod
    async def fetch_ohlcv(
        self,
        figi: str,
        start: datetime,
        end: datetime,
        interval: str = "1d"
    ) -> list[OHLCVBar]:
        """Fetch OHLCV bars for a given instrument and date range."""
        ...

    @abstractmethod
    async def health_check(self) -> DataHealthEvent:
        """Return current adapter health status."""
        ...

    async def fetch_with_fallback(self, primary_call, fallback_adapters: list):
        """Try primary call; on failure, iterate fallback_adapters."""
        try:
            return await primary_call()
        except Exception as e:
            for adapter in fallback_adapters:
                try:
                    return await adapter.fetch_ohlcv(...)
                except Exception:
                    continue
            raise RuntimeError(f"All adapters failed for {self.name}") from e
```

---

## Part 5 — Data Provider Stack

### 5.1 Provider Tiers and Fallback Chains

```yaml
# config/providers.yaml

equity_ohlcv:
  daily:
    tier_1:
      - name: yfinance
        cost: free
        rate_limit: "2 req/sec (unofficial)"
        coverage: "50+ years US, 20+ years international"
        risk: "unsanctioned Yahoo API, occasional throttle"
      - name: stooq
        cost: free
        rate_limit: "no documented limit"
        coverage: "US, EU, Asian markets"
        risk: "no official API"
    tier_2:
      - name: alphavantage
        cost: free (5 calls/min), $50/mo (premium)
        rate_limit: "5/min free, 75/min premium"
        coverage: "US + international"
    tier_3:
      - name: eodhd
        cost: $19.99/mo
        rate_limit: "100,000 req/day"
        coverage: "150,000+ instruments"
    fallback_order: [yfinance, stooq, alphavantage, eodhd]

  intraday_1min:
    tier_1:
      - name: alphavantage
        cost: free (5 calls/min)
        coverage: "2 years US equities"
    tier_2:
      - name: polygon
        cost: free (15-min delayed), $29/mo (real-time)
        rate_limit: "unlimited on paid"
        coverage: "all NMS securities"

real_time_quotes:
  tier_1:
    - name: finnhub_websocket
      cost: free (60 calls/min)
      latency: "<500ms"
      coverage: "US equities, some international"
  tier_2:
    - name: alpaca_basic
      cost: free (IEX data)
      note: "Not full SIP consolidated tape"
  tier_3:
    - name: alpaca_algo_trader_plus
      cost: $30/mo
      note: "Full SIP consolidated tape real-time"

edgar:
  cost: free
  rate_limit: "10 req/sec global (SEC mandate)"
  user_agent: "${EDGAR_USER_AGENT}"  # Required: "Name email@domain"
  bulk_download: "companyfacts.zip ~1.5GB, refreshed nightly"
  endpoints:
    - companyfacts: "data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    - submissions: "data.sec.gov/submissions/CIK{cik}.json"
    - company_tickers: "www.sec.gov/files/company_tickers.json"
    - fulltext_search: "efts.sec.gov/LATEST/search-index"
    - bulk_facts: "data.sec.gov/api/xbrl/companyfacts.zip"

fred:
  cost: free
  api_key: "${FRED_API_KEY}"  # Free registration at fred.stlouisfed.org
  rate_limit: "120 req/min"
  series_count: "765,000+"
  key_series:
    yield_curve: [DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS5, DGS10, DGS20, DGS30]
    breakeven: [T5YIE, T10YIE, T30YIE]
    spreads: [T10Y2Y, T10Y3M, T5Y5F]
    vix: [VIXCLS, VIXM3, VIXM6]
    inflation: [CPIAUCSL, PCEPI, CPILFESL]
    employment: [UNRATE, PAYEMS, ICSA]
    gdp: [GDP, GDPC1, GDPNow]

crypto:
  primary:
    - name: ccxt
      cost: free (open-source)
      exchanges: "100+ (Binance, Coinbase, Kraken, OKX, Bybit, BitMEX, Hyperliquid)"
      license: MIT
  defi:
    - name: defillama
      cost: free, no key
      endpoint: "https://api.llama.fi"
      coverage: "TVL, protocols, chains, yields"
  onchain:
    - name: etherscan
      cost: free (5 calls/sec with key)
      coverage: "Ethereum mainnet + L2s"

fixed_income:
  treasuries: "FRED DGS series (free, daily)"
  corporate: "FINRA TRACE (free, 15-min delay)"
  municipal: "MSRB EMMA (free, real-time trade prices)"
  
macro_gov:
  - FRED (St. Louis Fed) — primary
  - BLS API (CPI, employment) — free, 2000 req/day registered
  - BEA API (GDP, NIPA) — free with key
  - World Bank API — free, no key, 190 countries
  - IMF Data API — free, no key
  - CFTC — free, weekly COT reports
  - Treasury FiscalData — free, no key
```

### 5.2 Rate Limiting Implementation

```python
# sds/rate_limiter.py
import asyncio
import time
from collections import deque

class TokenBucketRateLimiter:
    """Token bucket rate limiter with Redis backing for cross-process sharing."""
    
    def __init__(self, redis, key: str, rate: int, per: float = 60.0):
        self.redis = redis
        self.key = f"rate_limit:{key}"
        self.rate = rate      # tokens per period
        self.per = per        # period in seconds
        self._local_queue = deque()

    async def acquire(self):
        """Block until a token is available. Max wait = 30 seconds."""
        for _ in range(300):  # 300 × 100ms = 30s timeout
            allowed = await self._try_acquire()
            if allowed:
                return
            await asyncio.sleep(0.1)
        raise TimeoutError(f"Rate limit timeout for {self.key}")

    async def _try_acquire(self) -> bool:
        now = time.time()
        async with self.redis.pipeline() as pipe:
            pipe.zremrangebyscore(self.key, 0, now - self.per)
            pipe.zcard(self.key)
            pipe.zadd(self.key, {str(now): now})
            pipe.expire(self.key, int(self.per) + 1)
            results = await pipe.execute()
        current_count = results[1]
        if current_count < self.rate:
            return True
        # Rollback the zadd
        await self.redis.zrem(self.key, str(now))
        return False
```

---

## Part 6 — AI/NLP Technical Stack

### 6.1 Embedding Strategy

```
Primary embedding:   voyage-finance-2 (1024-dim, financial domain-tuned)
                     Voyage AI API, $0.12/1M tokens
                     Best semantic search on financial text in benchmarks

Fallback embedding:  BAAI/bge-small-en-v1.5 (384-dim, local, MIT license)
                     Runs entirely on-device via sentence-transformers
                     Used for: real-time news ingestion when API budget exhausted

Query embedding:     Same model as document embedding (critical for retrieval quality)

Storage:             pgvector extension on PostgreSQL
                     IVFFLAT index (lists=100) for <100ms retrieval on 1M+ vectors
                     HNSW index (m=16, ef=200) for better recall at higher dim count
```

### 6.2 RAG Pipeline Architecture

```python
# sil/rag_pipeline.py — architecture sketch

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.retrievers import VectorIndexRetriever, BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank

class SentinelRAG:
    """Hybrid BM25 + vector retrieval with RRF fusion and re-ranking."""
    
    def __init__(self, pg_store: PGVectorStore, embed_model, llm):
        self.vector_retriever = VectorIndexRetriever(
            index=VectorStoreIndex.from_vector_store(pg_store),
            similarity_top_k=20
        )
        self.bm25_retriever = BM25Retriever(
            tokenizer="financial",  # custom tokenizer preserving $, %, bp, bps
            similarity_top_k=20
        )
        self.reranker = RankGPTRerank(top_n=5, llm=llm)

    async def query(self, question: str, filters: dict = None) -> str:
        # 1. Dual retrieval
        vector_results = await self.vector_retriever.aretrieve(question)
        bm25_results = await self.bm25_retriever.aretrieve(question)
        
        # 2. Reciprocal Rank Fusion
        fused = self._rrf_merge(vector_results, bm25_results, k=60)
        
        # 3. Re-rank top 20 with LLM
        reranked = await self.reranker.apostprocess_nodes(fused[:20], question)
        
        # 4. Generate with context
        return await self._generate(question, reranked[:5])
    
    def _rrf_merge(self, list_a, list_b, k=60):
        scores = {}
        for rank, node in enumerate(list_a):
            scores[node.node_id] = scores.get(node.node_id, 0) + 1/(k + rank + 1)
        for rank, node in enumerate(list_b):
            scores[node.node_id] = scores.get(node.node_id, 0) + 1/(k + rank + 1)
        all_nodes = {n.node_id: n for n in list_a + list_b}
        return sorted(all_nodes.values(), key=lambda n: scores[n.node_id], reverse=True)
```

### 6.3 MCP Server Implementation (FastMCP)

```python
# sil/mcp_server.py
from fastmcp import FastMCP
from core.types import MacroRegime
import anthropic

mcp = FastMCP("sentinel")

@mcp.tool()
async def search_filings(
    query: str,
    form_type: str = None,
    ticker: str = None,
    date_from: str = None,
    date_to: str = None
) -> list[dict]:
    """Search SEC EDGAR filings using full-text search. Returns matching filing excerpts."""
    ...

@mcp.tool()
async def get_financials(
    ticker: str,
    period: str = "annual",          # annual / quarterly
    metrics: list[str] = None,       # us-gaap/Revenues, etc. — None = all
    years_back: int = 5
) -> dict:
    """Return standardized financial statements for a company."""
    ...

@mcp.tool()
async def run_screen(criteria: dict) -> list[dict]:
    """Run an equity screen. criteria = {field: {operator: value}} dict."""
    ...

@mcp.tool()
async def get_congressional_trades(
    member: str = None,
    ticker: str = None,
    days_back: int = 90
) -> list[dict]:
    """Return congressional STOCK Act trade disclosures."""
    ...

@mcp.tool()
async def run_backtest(strategy_yaml: str) -> dict:
    """Parse a strategy YAML spec and run it through VectorBT. Returns 24-metric tearsheet."""
    ...

@mcp.tool()
async def get_regime(date: str = None) -> dict:
    """Return the HMM macro regime for a given date (default: today)."""
    ...

@mcp.tool()
async def get_cot_report(market: str, weeks_back: int = 52) -> dict:
    """Return CFTC COT positioning for a futures market with COT Index."""
    ...

@mcp.tool()
async def screen_natural_language(query: str) -> list[dict]:
    """Convert a plain English investment thesis into a screen and run it."""
    ...

@mcp.tool()
async def explain_strategy(strategy_id: str) -> str:
    """Generate a plain-English explanation of a strategy's backtest performance."""
    ...

# FastMCP serves via stdio for Claude Desktop or HTTP for API access
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### 6.4 NL-to-Strategy Pipeline

```python
# sil/strategy_generator.py
import anthropic

STRATEGY_SCHEMA = """
name: string
description: string
universe:
  asset_class: equity | crypto | bond | fx
  filters:
    - field: string
      operator: gt | lt | eq | gte | lte | in
      value: any
entry_signals:
  - type: fundamental | technical | ownership | macro | sentiment
    params: dict
exit_signals:
  - type: time_stop | trailing_stop | signal_reversal | fixed_stop
    params: dict
position_sizing:
  method: equal_weight | vol_target | kelly | risk_parity
  params: dict
execution:
  frequency: daily | weekly | monthly | intraday
  venue: alpaca | ib | binance | kraken
"""

async def generate_strategy(hypothesis: str) -> dict:
    """
    Convert a plain-English trading hypothesis into a validated strategy YAML spec.
    Example: "Buy small-cap stocks with insider buying when the macro regime is Growth/Inflation"
    """
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=2000,
        system=f"""You are a quantitative strategist. Convert trading hypotheses into
        structured strategy YAML specs following this schema exactly:
        {STRATEGY_SCHEMA}
        
        Return only valid YAML. Validate that all referenced fields exist in SENTINEL's
        screener criteria list.""",
        messages=[{"role": "user", "content": hypothesis}]
    )
    
    yaml_spec = response.content[0].text
    # Validate against SBE strategy schema before returning
    return validate_and_parse_strategy(yaml_spec)
```

---

## Part 7 — Backtesting Technical Spec

### 7.1 VectorBT Research Backend

```python
# sbe/vectorbt_backend.py
import vectorbt as vbt
import numpy as np
import pandas as pd
from .metrics import compute_24_metrics
from .dsr import compute_dsr, compute_pbo

class VectorBTBackend:
    """
    Vectorized research engine. 10-year / 500-symbol universe in < 60 seconds.
    Enforces look-ahead bias prevention via .shift(1) on all signals.
    """
    
    def run(self, price_data: pd.DataFrame, entries: pd.DataFrame, exits: pd.DataFrame,
            fees: float = 0.001, slippage: float = 0.0005) -> dict:
        
        # CRITICAL: shift entries by 1 bar to prevent look-ahead bias
        # Signal generated at close of bar N → executed at open of bar N+1
        entries_shifted = entries.shift(1).fillna(False)
        exits_shifted = exits.shift(1).fillna(False)
        
        portfolio = vbt.Portfolio.from_signals(
            close=price_data,
            entries=entries_shifted,
            exits=exits_shifted,
            fees=fees,
            slippage=slippage,
            freq="1D",
            init_cash=100_000
        )
        
        returns = portfolio.returns()
        metrics = compute_24_metrics(returns, portfolio)
        
        # Mandatory overfitting controls
        dsr = compute_dsr(returns)
        pbo = compute_pbo(returns, n_splits=16)
        
        return {
            "metrics": metrics,
            "dsr": dsr,
            "pbo": pbo,
            "portfolio": portfolio,
            "promoted": dsr > 0.5 and pbo < 0.5  # Promotion gate
        }
```

### 7.2 Deflated Sharpe Ratio Implementation

```python
# sbe/dsr.py
import numpy as np
from scipy import stats

def compute_dsr(returns: np.ndarray, n_trials: int = 1) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, JPM 2014).
    Adjusts Sharpe for skewness, kurtosis, and multiple-testing bias.
    
    DSR = Φ((SR̂ − E[max SR]) × √(T-1) / √(1 - γ₃·SR̂ + (γ₄-1)/4·SR̂²))
    
    where:
        SR̂  = estimated annualized Sharpe ratio
        T   = number of observations
        γ₃  = skewness of returns
        γ₄  = kurtosis of returns
        E[max SR] = expected max Sharpe under H₀ for n_trials independent strategies
    """
    T = len(returns)
    if T < 30:
        return 0.0
    
    sr = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns, fisher=True)  # excess kurtosis
    
    # Expected maximum Sharpe ratio under null (n_trials independent strategies)
    e_max_sr = _expected_max_sharpe(n_trials)
    
    numerator = (sr - e_max_sr) * np.sqrt(T - 1)
    denominator = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    
    if denominator <= 0:
        return 0.0
    
    z = numerator / denominator
    return float(stats.norm.cdf(z))

def _expected_max_sharpe(n: int) -> float:
    """E[max(SR₁,...,SRₙ)] under i.i.d. standard normal assumption."""
    if n <= 1:
        return 0.0
    euler_mascheroni = 0.5772156649
    return (1 - euler_mascheroni) * stats.norm.ppf(1 - 1/n) + euler_mascheroni * stats.norm.ppf(1 - 1/(n * np.e))
```

---

## Part 8 — Testing Strategy

### 8.1 Test Categories

| Category | Location | Purpose | Run Frequency |
|----------|---------|---------|:-------------:|
| Unit | `tests/unit/` | Single-function correctness | Every commit |
| Integration | `tests/integration/` | Cross-module data flow | Every PR |
| Financial Evals | `tests/financial_evals/` | Backtest regression: known results must match | Nightly |
| Adapter Health | `tests/adapters/` | Live API call smoke tests | Hourly (CI/CD) |

### 8.2 Financial Eval Tests

```python
# tests/financial_evals/eval_spy_buyhold.py
"""
SPY buy-hold benchmark must match CRSP total return within 2 basis points.
This is the most fundamental correctness test in the system.
"""

def test_spy_buy_hold_matches_crsp():
    """
    SPY total return from 2004-01-02 to 2023-12-31:
    CRSP benchmark: +9.87% CAGR (known from academic literature)
    SENTINEL must produce: 9.87% ± 0.02% CAGR
    """
    from sbe.vectorbt_backend import VectorBTBackend
    from sds.adapters.yfinance_adapter import YFinanceAdapter

    adapter = YFinanceAdapter()
    prices = adapter.fetch_ohlcv("BBG000BDTBL9", "2004-01-02", "2023-12-31")
    
    backend = VectorBTBackend()
    entries = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    exits = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    exits.iloc[-1] = True  # Exit at end
    
    result = backend.run(prices, entries, exits, fees=0.0, slippage=0.0)
    cagr = result["metrics"]["cagr"]
    
    assert abs(cagr - 0.0987) < 0.0002, f"CAGR {cagr:.4f} deviates from benchmark 0.0987"


# tests/financial_evals/eval_pit_integrity.py
def test_point_in_time_integrity():
    """
    Query Apple revenue as-of 2022-01-15 (before Q1 2022 report).
    Must return Q4 2021 value, NOT the restated Q4 2021 value if any restatement occurred.
    """
    from sfe.pit_store import PITStore
    pit = PITStore()
    
    result = pit.query(
        cik="0000320193",
        concept="us-gaap/RevenueFromContractWithCustomerExcludingAssessedTax",
        as_of_date=date(2022, 1, 15)
    )
    
    # Apple Q4 FY2021 (ended Sep 25, 2021) revenue: $83.36B
    assert abs(result.value - 83_360_000_000) < 1_000_000_000
```

### 8.3 Makefile Targets

```makefile
# Makefile
.PHONY: all test test-unit test-integration test-evals lint format docker-up bootstrap

all: lint test

docker-up:
	docker compose up -d postgres redis

bootstrap: docker-up
	python scripts/bootstrap.py

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-evals:
	pytest tests/financial_evals/ -v --tb=long

test: test-unit test-integration

lint:
	ruff check sentinel/ --fix
	mypy sentinel/ --ignore-missing-imports

format:
	ruff format sentinel/

backfill:
	python scripts/backfill.py --source edgar --years 5
	python scripts/backfill.py --source fred --series config/fred_series.yaml

terminal:
	streamlit run stu/app.py
```

---

## Part 9 — Security & Secrets Management

### 9.1 Secret Classification

| Secret | Storage | Rotation |
|--------|---------|---------|
| Database password | `.env` (local) / Docker secret | Quarterly |
| Redis password | `.env` | Quarterly |
| Finnhub API key | `.env` | On revocation |
| Alpha Vantage API key | `.env` | On revocation |
| Polygon API key | `.env` | On revocation |
| FRED API key | `.env` | Rarely (free) |
| Anthropic API key | `.env` | Monthly |
| Alpaca keys (paper) | `.env` | On revocation |
| Alpaca keys (live) | `.env` + OS keychain | Monthly |
| IB credentials | OS keychain only | Per IB policy |
| Voyage AI key | `.env` | Monthly |

### 9.2 Security Rules (Non-Negotiable)

1. **No secrets in Git.** `.env` is gitignored. `.env.example` contains placeholder values only.
2. **No secrets in logs.** `core/security.py` wraps all secret access; log output redacts key values.
3. **No live trading without explicit confirmation.** SEE requires `SENTINEL_LIVE_TRADING=true` environment variable AND `--live` CLI flag. Paper trading is the default.
4. **Kill switch is always reachable.** Emergency halt endpoint `/api/v1/kill` is always active, no authentication required from localhost.
5. **EDGAR rate limit is enforced in code.** Never exceed 10 req/sec. Violation risks IP ban from SEC.

---

## Part 10 — Performance Requirements

| Operation | Target Latency | Method |
|-----------|:--------------:|--------|
| Terminal load (cold) | < 5 seconds | Streamlit startup + DB connection pool warmup |
| Terminal load (warm) | < 1 second | Redis cache hit |
| Security description card (DES) | < 500ms | DB query + Redis cache |
| Fundamental screen (50+ fields, S&P 500) | < 3 seconds | DuckDB in-memory query on PostgreSQL materialized view |
| Watchlist quote update | < 1 second | Finnhub WebSocket push |
| Chart load (20yr daily, cached) | < 2 seconds | TimescaleDB hypertable range query |
| VectorBT backtest (10yr, 500 symbols) | < 60 seconds | NumPy vectorized compute |
| NautilusTrader backtest (1yr, 1-min bars, single symbol) | < 30 seconds | Rust core + Python API |
| RAG query (financial document search) | < 5 seconds | pgvector IVFFLAT + RRF rerank |
| MCP tool call (any) | < 3 seconds | Async FastAPI + connection pool |
| EDGAR 8-K detection latency | < 60 seconds | EDGAR RSS feed polling every 30s |

---

*SENTINEL TRD v2.0 — May 7, 2026*
*Next: SENTINEL_OSS_UNIVERSE.md — complete catalog of 100+ open-source projects*
