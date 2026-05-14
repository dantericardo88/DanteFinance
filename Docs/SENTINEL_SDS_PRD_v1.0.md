# SENTINEL Data Spine — Product Requirements Document
**Module:** SDS — SENTINEL Data Spine
**Version:** 1.0 (Party Build — 100% Complete)
**Date:** May 7, 2026
**Status:** Generation 0 — Specification Complete, Build Ready
**Parent:** SENTINEL PRD v2.0 / TRD v2.0
**Owner:** Ricky Porras / Dante Ecosystem
**Classification:** Foundational PRD — the data layer all other SENTINEL modules depend on
**Party Agents:** PM · Architect · Dev · UX · Scrum Master

---

## 1. Executive Summary

### Why This Module Exists

Bloomberg and CapIQ do not derive their value from exclusive access to financial data. Almost all the data they resell is public, semi-public, or licensed from sources that have free or cheap equivalents. Their true value — the thing that justifies $32K/yr per seat — is that they have already done the data engineering work that no individual or small team has time to redo: normalizing field names across 10,000 filings, adjusting for every split and dividend since 1980, tagging every delisted company so survivorship bias cannot creep into a backtest, and snapshotting fundamentals at their original as-filed values so look-ahead bias cannot corrupt a factor model.

That work, which once required a team of hundreds, can now be done by one engineer with the right pipeline and AI assistance.

**SENTINEL SDS is that pipeline.** It is the single source of truth for all financial data in the SENTINEL ecosystem. Every price, every filing value, every macro observation that any other SENTINEL module touches — screening, backtesting, portfolio risk, the AI intelligence layer — flows through SDS first, cleaned, validated, and stamped with a provenance receipt.

### The Commercial Insight

A survivorship-bias-free, point-in-time, corporate-action-adjusted, cross-validated US equity and macro dataset, built on top of free sources, is itself a product. Academic researchers pay WRDS ~$20K/yr for Compustat + CRSP access. Boutique quants pay Tiingo $100/mo or Polygon $200/mo for price data that is still not point-in-time for fundamentals. A clean, validated, documented dataset that SENTINEL can export as Parquet or host as a lightweight API has genuine standalone commercial value — estimated $49–$500/mo for researchers and boutique funds who cannot justify a WRDS subscription.

The pipeline built for SENTINEL's internal use becomes the product.

---

## 2. Scope and Non-Goals

### In Scope

| Domain | Coverage |
|--------|---------|
| US Equities | All NMS securities, current + delisted, NYSE/NASDAQ/AMEX/OTC |
| US ETFs | All exchange-listed ETFs with underlying exposure data |
| US Fundamentals | SEC EDGAR XBRL filings, 10-K/10-Q, back to XBRL inception (2009) |
| Macro / Economic | FRED 765K+ series, OECD, World Bank key indicators |
| Crypto | 100+ exchanges via CCXT, major pairs + perpetual futures |
| US Futures | Continuous contract construction, major commodity + financial futures |
| Corporate Actions | Splits, dividends, spin-offs, mergers, delistings, ticker changes |
| Ownership | 13F institutional holdings, Form 4 insider transactions |

### Deliberate Non-Goals (Gen 0)

- Real-time Level 2 / order book data — Gen 3 roadmap
- International equity fundamentals (ex-US IFRS) — Gen 2
- Licensed bond/credit data (TRACE parsing) — Gen 2
- Analyst consensus estimates aggregation — Gen 2 (LLM extraction pipeline)
- Alternative data (satellite, credit card, web traffic) — Gen 3

---

## 3. Core Problems to Solve

These are the five failure modes of free financial data. SDS must solve all five.

### Problem 1 — Survivorship Bias

**What it is:** Free data providers (yfinance, Alpha Vantage, most REST APIs) only serve currently-listed securities. A company that went bankrupt in 2015 simply does not appear. Any backtest built on this data tests only on survivors, producing returns that are systematically too high.

**Magnitude:** S&P 500 historically loses ~20 constituents per year to bankruptcy, acquisition, or delisting. Over a 10-year backtest window, ignoring delisted companies inflates simulated returns by 1–4% annually for value strategies — enough to turn a losing strategy into a perceived winner.

**SDS solution:** Maintain a `company_registry` table with every company that has ever filed with the SEC. Ingest EDGAR filing history to establish first-filing and last-filing dates as proxy for listing/delisting. Tag all securities with `listing_status`. All price queries default to `include_delisted=True`.

### Problem 2 — Corporate Action Adjustment Errors

**What it is:** Free sources apply split and dividend adjustments retroactively and inconsistently. yfinance has a documented ~2–5% error rate on adjustment factors. A single missed split creates a phantom 50% price drop.

**SDS solution:** Build a first-principles corporate action table sourced from SEC EDGAR (8-K) and Polygon. Cross-validate every adjustment factor against at least 2 independent sources. Use multiplicative adjustment factors stored separately — prices are never overwritten. Adjusted view computed on read.

### Problem 3 — Look-Ahead Bias in Fundamentals

**What it is:** Aggregated fundamental databases serve the latest *revised* version of a financial statement, not the value that was *publicly known at the time*. Using restated numbers in a backtest pretends you knew something in 2018 that was only published in 2020.

**SDS solution:** Store every EDGAR XBRL filing at the time of original ingestion with an immutable `filed_at` timestamp. Never overwrite historical snapshots — only append new filings. All fundamental queries against the backtesting engine are constrained to `filed_at <= query_date`.

### Problem 4 — Cross-Source Schema Inconsistency

**What it is:** Different providers use different field names, timezone assumptions, and missing-value representations for the same data. `open`, `Open`, `o`, `1. open` are all the same field across different APIs.

**SDS solution:** A single canonical Pydantic v2 schema for every data type (defined in `core/types.py`). Every source adapter maps to the canonical schema before data touches the database. All timestamps stored as UTC. All missing values normalized to `NULL`.

### Problem 5 — Silent Data Gaps and Staleness

**What it is:** Free APIs silently return partial data, truncate responses at undocumented limits, or simply stop updating without error.

**SDS solution:** Every ingestion run produces a `DataHealthEvent` containing expected vs actual record count, gap locations, staleness delta, schema drift flags. Automated gap-fill retries with exponential backoff. Dead-man alerts if a source misses its SLA window.

---

## 4. Architecture

### 4.1 SDS Position in SENTINEL

```
┌─────────────────────────────────────────────────────────────┐
│              External Data Sources (Free Tier)              │
│  Polygon · yfinance · EDGAR · FRED · CCXT · Alpaca · EODHD │
└───────────────────────┬─────────────────────────────────────┘
                        │  raw, heterogeneous
┌───────────────────────▼─────────────────────────────────────┐
│                  SDS — SENTINEL Data Spine                   │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Adapters   │→ │  Normalizer  │→ │  Validator        │  │
│  │  (per src)  │  │  (canonical  │  │  (cross-source,   │  │
│  └─────────────┘  │   schema)    │  │   outlier, gap)   │  │
│                   └──────────────┘  └─────────┬─────────┘  │
│  ┌─────────────────────────────────────────────▼─────────┐  │
│  │              Corporate Action Engine                   │  │
│  │  (raw price + CA table → adjusted view on read)       │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Provenance Receipt Issuer              │    │
│  │  (source, version, timestamp, validation hash)      │    │
│  └─────────────┬───────────────────────────────────────┘    │
└────────────────│────────────────────────────────────────────┘
                 │  clean, validated, stamped
┌────────────────▼────────────────────────────────────────────┐
│                      Data Layer                             │
│  TimescaleDB (OHLCV hypertables, corporate actions)         │
│  PostgreSQL (filings, fundamentals, macro, ownership)       │
│  Parquet export store (backtesting engine, data product)    │
└────────────────────────────────────────────────────────────┘
                 │  served to
┌────────────────▼────────────────────────────────────────────┐
│            All Other SENTINEL Modules                       │
│  SBE (backtest) · SSE (screener) · SPR (portfolio risk)     │
│  SIL (AI layer) · SMA (macro) · SFE (filings) · SOD (own.) │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 SDS Internal Module Structure

```
sentinel/sds/
├── __init__.py
├── orchestrator.py              # APScheduler — all ingestion jobs
│
├── adapters/
│   ├── __init__.py
│   ├── base.py                  # Abstract BaseAdapter
│   ├── polygon.py               # Primary price + corporate actions
│   ├── yfinance.py              # Secondary validation
│   ├── edgar.py                 # SEC EDGAR bulk + XBRL
│   ├── fred.py                  # FRED macro time series
│   ├── ccxt_adapter.py          # CCXT crypto OHLCV
│   ├── alpaca.py                # Real-time US equities
│   ├── eodhd.py                 # International (Gen 2)
│   └── finra.py                 # Short interest (free)
│
├── normalize/
│   ├── __init__.py
│   ├── price_normalizer.py      # OHLCV → OHLCVBar
│   ├── fundamental_normalizer.py  # XBRL → FinancialStatement
│   ├── macro_normalizer.py      # FRED → MacroObservation
│   ├── action_normalizer.py     # CA events → CorporateAction
│   └── filing_normalizer.py     # EDGAR → Filing
│
├── validate/
│   ├── __init__.py
│   ├── cross_source.py          # Compare ≥2 sources
│   ├── outlier_detector.py      # 5-sigma spike detection
│   ├── gap_detector.py          # Missing trading day detection
│   ├── schema_validator.py      # Pydantic enforcement
│   └── health_emitter.py        # DataHealthEvent publisher
│
├── corporate_actions/
│   ├── __init__.py
│   ├── action_table.py          # CA record storage
│   ├── adjustment_engine.py     # Cumulative factor computation
│   ├── action_classifier.py     # Classify action type
│   └── action_verifier.py       # Cross-source verification
│
├── survivorship/
│   ├── __init__.py
│   ├── registry.py              # company_registry builder
│   ├── delisting_tracker.py     # Delisting event ingestion
│   └── universe_builder.py      # Point-in-time universe queries
│
├── instruments/
│   ├── __init__.py
│   └── master.py                # OpenFIGI three-tier cache
│
├── futures/
│   ├── __init__.py
│   └── continuous.py            # Continuous contract construction
│
├── provenance/
│   ├── __init__.py
│   ├── receipt_issuer.py        # DataProvenance receipt
│   ├── chain_entry.py           # Immutable chain entry
│   └── chain_writer.py          # Append-only store
│
└── export/
    ├── __init__.py
    ├── parquet_exporter.py      # Export to partitioned Parquet
    ├── manifest_builder.py      # JSON sidecar manifest
    └── api_publisher.py         # Optional: lightweight REST API
```

---

## 5. Canonical Schema

All data in SENTINEL is expressed as one of these canonical Pydantic v2 types. Source adapters must map to these exactly.

### 5.1 OHLCVBar

```python
class OHLCVBar(BaseModel):
    ticker: str
    figi: str | None
    isin: str | None
    timestamp: datetime                # UTC, timezone-aware, bar close time
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal                     # Raw unadjusted — NEVER overwritten
    volume: int
    vwap: Decimal | None
    trade_count: int | None
    timeframe: str                     # "1d" | "1h" | "5m" | "1m"
    source: DataSource
    source_version: str
    ingested_at: datetime
    adjustment_factor: Decimal = Decimal("1.0")  # 1.0 = unadjusted
    is_validated: bool = False
    validation_sources: list[str] = []

    @property
    def adjusted_close(self) -> Decimal:
        return self.close * self.adjustment_factor
```

### 5.2 CorporateAction

```python
class ActionType(str, Enum):
    SPLIT = "split"
    REVERSE_SPLIT = "reverse_split"
    CASH_DIVIDEND = "cash_dividend"
    STOCK_DIVIDEND = "stock_dividend"
    SPINOFF = "spinoff"
    MERGER_CASH = "merger_cash"
    MERGER_STOCK = "merger_stock"
    DELISTING = "delisting"
    TICKER_CHANGE = "ticker_change"
    RIGHTS_ISSUE = "rights_issue"

class CorporateAction(BaseModel):
    ticker: str
    figi: str | None
    ex_date: date
    record_date: date | None
    pay_date: date | None
    action_type: ActionType
    factor: Decimal                    # Multiplicative: 2-for-1 split = 0.5
    raw_value: Decimal | None
    currency: str | None
    description: str | None
    source_primary: DataSource
    source_secondary: DataSource | None
    sources_agree: bool = False
    confidence: float = 0.0            # 0.0–1.0
```

### 5.3 FinancialStatement

```python
class FinancialStatement(BaseModel):
    ticker: str
    figi: str | None
    cik: str                           # SEC CIK — immutable company key
    period_end: date
    period_type: str                   # "annual" | "quarterly"
    filed_at: datetime                 # IMMUTABLE — set at ingestion, never updated
    amended_at: datetime | None
    accession_number: str              # SEC EDGAR unique filing ID
    statement_type: str                # "income" | "balance_sheet" | "cash_flow"
    currency: str
    scale: int                         # 1 | 1000 | 1000000
    source: DataSource
    ingested_at: datetime
    is_point_in_time: bool = True
    line_items: dict[str, Decimal | None]
```

### 5.4 MacroObservation

```python
class MacroObservation(BaseModel):
    series_id: str                     # FRED series ID (e.g., "FEDFUNDS")
    series_name: str
    observation_date: date
    value: Decimal | None
    vintage_date: datetime
    units: str
    frequency: str                     # "D" | "W" | "M" | "Q" | "A"
    seasonal_adjustment: str           # "SA" | "NSA" | "SAAR"
    source: DataSource
    ingested_at: datetime
```

### 5.5 CompanyRecord

```python
class ListingStatus(str, Enum):
    ACTIVE = "active"
    DELISTED = "delisted"
    ACQUIRED = "acquired"
    SUSPENDED = "suspended"

class CompanyRecord(BaseModel):
    cik: str                           # Primary key — immutable
    ticker_history: list[TickerAlias]  # All tickers with date ranges
    figi: str | None
    isin: str | None
    cusip: str | None
    company_name: str
    name_history: list[NameAlias]
    sic_code: str | None
    exchange: str | None
    listing_status: ListingStatus
    first_filing_date: date
    last_filing_date: date | None
    delisting_reason: str | None
    country: str = "US"
    currency: str = "USD"
```

### 5.6 DataProvenance

```python
class DataProvenance(BaseModel):
    receipt_id: str                    # UUID
    issued_at: datetime
    data_type: str
    source: DataSource
    source_version: str
    record_count: int
    date_range_start: date
    date_range_end: date
    tickers: list[str] | None
    series_ids: list[str] | None
    validation_passed: bool
    validation_sources: list[DataSource]
    anomalies_flagged: int
    adjustment_factors_applied: int
    content_hash: str                  # SHA-256 of batch before storage
    pipeline_version: str
```

---

## 6. Corporate Action Engine

### 6.1 Principles

1. **Raw prices are immutable.** `open`, `high`, `low`, `close`, `volume` are never modified after storage.
2. **Adjustment factors are multiplicative and cumulative.** Factor for bar on date `d` = product of all action factors where `ex_date > d`.
3. **Two sources must agree.** `sources_agree = True` only when two independent sources match within 0.5% tolerance.
4. **Disagreements are flagged, not resolved automatically.** Primary source factor used provisionally with `sources_agree = False`.
5. **Volume adjustment is separate.** Split: volume multiplied inversely (2-for-1 split doubles historical volume).

### 6.2 Action Types and Adjustment Logic

| Action | Price Factor | Volume Factor | Notes |
|--------|-------------|--------------|-------|
| 2-for-1 split | × 0.5 | × 2.0 | factor = 1/split_ratio |
| 1-for-5 reverse split | × 5.0 | × 0.2 | Watch for delisting precursor |
| $1.00 cash dividend | (close − div) / close on ex_date−1 | × 1.0 | Factor varies by price level |
| Stock dividend (5%) | × 1/1.05 | × 1.05 | Treated as fractional split |
| Spinoff | Complex — source new security price | × 1.0 | Separate spinoff security tracking |
| Merger (cash) | Terminal price = acquisition price | N/A | DELISTED after settlement |
| Ticker change | No price adjustment | × 1.0 | CIK stays the same |

### 6.3 Cumulative Factor Computation

```python
def compute_cumulative_factor(
    ticker: str,
    bar_date: date,
    actions: list[CorporateAction]
) -> Decimal:
    """
    For a bar on date D, multiply all action factors where ex_date > D.
    This produces the backward-adjusted price factor.
    """
    factor = Decimal("1.0")
    for action in sorted(actions, key=lambda a: a.ex_date):
        if action.ex_date > bar_date and action.ticker == ticker:
            factor *= action.factor
    return factor
```

### 6.4 Source Priority Chain

1. **Polygon.io** `corporateActionsV1` — daily snapshots, primary
2. **SEC EDGAR 8-K** — authoritative, validate against Polygon
3. **yfinance** `dividends` / `splits` — fallback validation only

---

## 7. Survivorship Bias Correction

### 7.1 Company Registry Construction

The `CompanyRegistry` covers every EDGAR registrant:
```python
# Bootstrap from SEC endpoint
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
# Returns: {index: {cik_str, ticker, title}} for all ~14,000+ active registrants

# Delisting sources (priority order):
# 1. SEC Form 15-12B/15-12G deregistration filings
# 2. Polygon corporate actions (delisting events)
# 3. EDGAR — absence of filings for >18 months = presumed inactive
```

### 7.2 Point-in-Time Universe Query

```sql
SELECT cr.*
FROM company_registry cr
WHERE cr.first_filing_date <= :query_date
  AND (cr.last_filing_date IS NULL OR cr.last_filing_date >= :query_date)
  AND cr.listing_status IN ('active', 'delisted', 'acquired', 'suspended')
  AND cr.exchange IN ('NYSE', 'NASDAQ', 'AMEX')
```

### 7.3 Delisting Price Handling

When a security is delisted during backtesting:
- Last available closing price used as exit price (worst-case)
- `FORCED_LIQUIDATION` event logged in trade ledger
- Return reflects actual loss; bankruptcy → final price → zero return for remaining period

---

## 8. Point-in-Time Fundamentals

### 8.1 Storage Model

```sql
CREATE TABLE filing_snapshots (
    accession_number  TEXT PRIMARY KEY,
    cik               TEXT NOT NULL,
    period_end        DATE NOT NULL,
    period_type       TEXT NOT NULL,
    filed_at          TIMESTAMPTZ NOT NULL,  -- SET AT INGESTION, NEVER UPDATED
    amended_at        TIMESTAMPTZ,
    statement_type    TEXT NOT NULL,
    currency          TEXT,
    scale             INT DEFAULT 1,
    line_items        JSONB NOT NULL,
    ingested_at       TIMESTAMPTZ DEFAULT NOW()
);
```

### 8.2 Point-in-Time Query Pattern

```sql
-- Get income statement publicly available on 2022-03-15
SELECT *
FROM filing_snapshots
WHERE cik = :cik
  AND statement_type = 'income'
  AND period_type = 'quarterly'
  AND filed_at <= '2022-03-15'::timestamptz
ORDER BY filed_at DESC
LIMIT 1;
```

### 8.3 Restatement Handling

When a 10-K/A (amended) is filed:
- New record inserted with new `filed_at` = amendment date
- Original record **never modified**
- Both records have same `period_end`; queries return correct as-of version automatically

---

## 9. Cross-Source Validation

### 9.1 Validation Rules

| Check | Trigger | Action |
|-------|---------|--------|
| Close price delta > 1% between sources | Every price ingest | Flag bar, use primary source |
| Adjustment factor disagrees > 0.5% | Every CA event | Flag, mark `sources_agree = False` |
| Missing trading day (market was open) | Daily gap scan | Auto-retry, escalate after 3 failures |
| Volume = 0 on non-halt day | Every bar | Flag as suspicious |
| Close > High or Close < Low | Every bar | **Hard reject** — data corruption |
| Price spike > 5 std devs from 20-day rolling | Every bar | Flag for review, do not auto-reject |
| Fundamental value change > 50% QoQ | Every filing | Flag for review |

### 9.2 Silent Throttle Detection

```python
def detect_silent_throttle(
    expected_tickers: list[str],
    actual_bars: list[OHLCVBar],
    trading_date: date
) -> bool:
    """
    yfinance returns empty DataFrames under load without raising an error.
    Compare expected count vs actual to detect silent failures.
    """
    actual_tickers = {b.ticker for b in actual_bars if b.timestamp.date() == trading_date}
    coverage_rate = len(actual_tickers) / len(expected_tickers)
    if coverage_rate < 0.95:
        raise SilentThrottleError(
            f"Only {coverage_rate:.1%} coverage — expected {len(expected_tickers)}, got {len(actual_tickers)}"
        )
    return True
```

---

## 10. Provenance and Lineage

**Pattern source:** Borrowed from DanteHarvest `harvest_core/provenance/` — chain writer + receipt issuer pattern.

Every batch of data written to the SENTINEL database is accompanied by a `DataProvenance` receipt. The provenance chain is append-only and forms an immutable audit trail.

Every batch write produces:
- `receipt_id`: UUID, unique per batch
- `content_hash`: SHA-256 of the raw payload before transformation
- `pipeline_version`: SDS module version at ingestion time
- `validation_passed`: boolean result of cross-source validation
- `anomalies_flagged`: count of anomalies detected in this batch

---

## 11. Data Sources Registry

### 11.1 Free Sources (Gen 0 Core Stack)

| Source | Data Type | Quality | Rate Limit (free) | Update Freq |
|--------|-----------|---------|-------------------|------------|
| **Polygon.io** (free) | OHLCV daily, corporate actions | High | 5 req/min | Daily |
| **SEC EDGAR Bulk** | All filings, XBRL | Authoritative | 10 req/sec | Real-time (15-min delay) |
| **FRED** | 765K+ macro series | Authoritative | 120 req/min (with key) | Varies |
| **CCXT** | Crypto OHLCV (100+ exchanges) | Medium-High | Exchange-dependent | Real-time |
| **OpenFIGI** | Identifier mapping | High | 25K req/min (free key) | Daily |
| **FINRA** | Short interest | Authoritative | Bulk download | Bi-monthly |
| **yfinance** | OHLCV, dividends, splits | Medium | ~60 req/min | Daily |
| **Alpha Vantage** (free) | OHLCV fallback | Medium | 25 req/day | Daily |

### 11.2 Source Priority Chains

```
US Equity OHLCV:
  Primary   → Polygon.io
  Fallback  → Alpaca (if brokerage connected)
  Validate  → yfinance
  Last      → Alpha Vantage

US Fundamentals:
  Primary   → SEC EDGAR XBRL (authoritative)
  Validate  → FMP (Financial Modeling Prep, free tier)

Macro:
  Primary   → FRED (no fallback needed)

Corporate Actions:
  Primary   → Polygon.io CA endpoint
  Validate  → SEC EDGAR 8-K filings
  Cross-ref → yfinance split/dividend history
```

---

## 12. Database Schema

### 12.1 TimescaleDB Hypertables

```sql
-- Primary OHLCV hypertable
CREATE TABLE ohlcv_bars (
  time            TIMESTAMPTZ NOT NULL,
  ticker          TEXT NOT NULL,
  figi            TEXT,
  open            NUMERIC(20,6) NOT NULL,
  high            NUMERIC(20,6) NOT NULL,
  low             NUMERIC(20,6) NOT NULL,
  close           NUMERIC(20,6) NOT NULL,
  volume          BIGINT NOT NULL,
  vwap            NUMERIC(20,6),
  trade_count     INT,
  timeframe       TEXT NOT NULL DEFAULT '1d',
  source          TEXT NOT NULL,
  ingested_at     TIMESTAMPTZ DEFAULT NOW(),
  is_validated    BOOLEAN DEFAULT FALSE,
  PRIMARY KEY (time, ticker, timeframe, source)
);
SELECT create_hypertable('ohlcv_bars', 'time', chunk_time_interval => INTERVAL '1 month');

-- Enable compression (saves ~80% space)
ALTER TABLE ohlcv_bars SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'ticker, source'
);
SELECT add_compression_policy('ohlcv_bars', INTERVAL '3 months');

-- Macro observations hypertable
CREATE TABLE macro_observations (
  time            TIMESTAMPTZ NOT NULL,
  series_id       TEXT NOT NULL,
  value           NUMERIC,
  vintage_date    TIMESTAMPTZ NOT NULL,
  units           TEXT,
  frequency       TEXT,
  seasonal_adj    TEXT,
  source          TEXT DEFAULT 'FRED',
  ingested_at     TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (time, series_id, vintage_date)
);
SELECT create_hypertable('macro_observations', 'time', chunk_time_interval => INTERVAL '1 year');
```

### 12.2 PostgreSQL Tables

```sql
-- Company registry
CREATE TABLE company_registry (
  cik               TEXT PRIMARY KEY,
  figi              TEXT,
  isin              TEXT,
  company_name      TEXT NOT NULL,
  sic_code          TEXT,
  exchange          TEXT,
  listing_status    TEXT DEFAULT 'active',
  first_filing_date DATE,
  last_filing_date  DATE,
  delisting_reason  TEXT,
  country           TEXT DEFAULT 'US',
  currency          TEXT DEFAULT 'USD',
  updated_at        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_cr_ticker ON company_registry(listing_status, exchange);

-- Ticker history (CIK is the stable anchor)
CREATE TABLE ticker_aliases (
  cik         TEXT REFERENCES company_registry(cik),
  ticker      TEXT NOT NULL,
  exchange    TEXT,
  valid_from  DATE NOT NULL,
  valid_to    DATE,
  PRIMARY KEY (cik, ticker, valid_from)
);
CREATE INDEX idx_ta_ticker ON ticker_aliases(ticker, valid_from, valid_to);

-- Corporate actions
CREATE TABLE corporate_actions (
  id               BIGSERIAL PRIMARY KEY,
  ticker           TEXT NOT NULL,
  figi             TEXT,
  ex_date          DATE NOT NULL,
  action_type      TEXT NOT NULL,
  factor           NUMERIC(20,10) NOT NULL,
  raw_value        NUMERIC(20,6),
  currency         TEXT,
  source_primary   TEXT NOT NULL,
  source_secondary TEXT,
  sources_agree    BOOLEAN DEFAULT FALSE,
  confidence       FLOAT DEFAULT 0.0,
  created_at       TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_ca_ticker_exdate ON corporate_actions(ticker, ex_date);

-- Point-in-time filing snapshots
CREATE TABLE filing_snapshots (
  accession_number TEXT PRIMARY KEY,
  cik              TEXT REFERENCES company_registry(cik),
  period_end       DATE NOT NULL,
  period_type      TEXT NOT NULL,
  filed_at         TIMESTAMPTZ NOT NULL,
  amended_at       TIMESTAMPTZ,
  statement_type   TEXT NOT NULL,
  currency         TEXT,
  scale            INT DEFAULT 1,
  line_items       JSONB NOT NULL,
  ingested_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_fs_cik_filed ON filing_snapshots(cik, filed_at DESC);
CREATE INDEX idx_fs_period    ON filing_snapshots(cik, period_end, statement_type);

-- Data anomalies (quality review queue)
CREATE TABLE data_anomalies (
  id               BIGSERIAL PRIMARY KEY,
  detected_at      TIMESTAMPTZ DEFAULT NOW(),
  data_type        TEXT NOT NULL,
  ticker           TEXT,
  date             DATE,
  field            TEXT,
  value_primary    NUMERIC,
  value_secondary  NUMERIC,
  delta_pct        NUMERIC,
  anomaly_type     TEXT NOT NULL,
  resolved         BOOLEAN DEFAULT FALSE,
  resolution_note  TEXT
);

-- Continuous futures
CREATE TABLE continuous_futures (
  symbol           TEXT NOT NULL,
  time             TIMESTAMPTZ NOT NULL,
  contract_month   TEXT NOT NULL,
  open             NUMERIC(20,6),
  high             NUMERIC(20,6),
  low              NUMERIC(20,6),
  close            NUMERIC(20,6),
  volume           BIGINT,
  roll_date        DATE,
  adjustment_factor NUMERIC(20,10) DEFAULT 1.0,
  PRIMARY KEY (symbol, time)
);

-- Provenance chain (append-only)
CREATE TABLE provenance_chain (
  receipt_id       TEXT PRIMARY KEY,
  issued_at        TIMESTAMPTZ NOT NULL,
  data_type        TEXT NOT NULL,
  source           TEXT NOT NULL,
  source_version   TEXT,
  record_count     INT,
  date_range_start DATE,
  date_range_end   DATE,
  validation_passed BOOLEAN,
  anomalies_flagged INT DEFAULT 0,
  content_hash     TEXT NOT NULL,
  pipeline_version TEXT NOT NULL
);
-- No UPDATE or DELETE grants on this table for the app user
```

---

## 13. Technology Stack

### 13.1 Python Version

Python 3.11+ required. Tested on 3.11 and 3.12. 3.13 not yet validated.

### 13.2 pyproject.toml

```toml
[tool.poetry]
name = "sentinel-sds"
version = "0.1.0"
description = "SENTINEL Data Spine — financial data normalization pipeline"
python = "^3.11"

[tool.poetry.dependencies]
python = "^3.11"

# Data Ingestion
polygon-api-client = "^1.14"     # Primary price + corporate actions
yfinance = "^0.2"                 # Secondary validation (known issues — validation only)
alpaca-py = "^0.26"               # Real-time US equities + paper trading
fredapi = "^0.5"                  # FRED macro (requires free API key)
ccxt = "^4.3"                     # Crypto OHLCV, 100+ exchanges
sec-edgar-downloader = "^5.0"     # EDGAR bulk filing downloads
edgartools = "^2.0"               # Form 4, 13F, 10-K structured parsing

# Storage
asyncpg = "^0.29"                 # PostgreSQL async driver
sqlalchemy = {version = "^2.0", extras = ["asyncio"]}
alembic = "^1.13"                 # Database migrations
psycopg2-binary = "^2.9"          # Sync driver for TimescaleDB admin
redis = {version = "^5.0", extras = ["asyncio"]}
duckdb = "^0.10"                  # In-process analytics + Parquet queries
pyarrow = "^16.0"                 # Parquet read/write + Arrow format
polars = "^0.20"                  # Fast batch transforms (prefer over pandas)
numpy = "^1.26"                   # Pinned below 2.0; polars/pyarrow compat

# Validation
pydantic = "^2.7"                 # v2 — all canonical schemas
pandera = "^0.18"                 # Inline DataFrame schema validation
great-expectations = "^0.18"      # Post-ingest QA suites + HTML reports

# Scheduling
apscheduler = "^3.10"             # Lightweight cron-like scheduler for Mac mini

# HTTP + Retry
httpx = "^0.27"                   # Async HTTP client
aiohttp = "^3.9"                  # Some adapters require aiohttp
tenacity = "^8.3"                 # Retry decorators with exponential backoff

# Utilities
pandas = "^2.2"                   # Compatibility with financial libs (use polars for perf)
pandas-market-calendars = "^4.3"  # Exchange calendar + gap detection
pendulum = "^3.0"                 # Timezone-aware datetime handling
python-dotenv = "^1.0"            # .env file loading
structlog = "^24.0"               # JSON structured logging
prometheus-client = "^0.20"       # Metrics exposure

# Dev / Test
[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
pytest-asyncio = "^0.23"
pytest-cov = "^5.0"
hypothesis = "^6.100"             # Property-based testing
factory-boy = "^3.3"              # Test fixture factories
pytest-recording = "^0.13"        # VCR cassette recording for HTTP replay
black = "^24.0"
ruff = "^0.4"
mypy = "^1.10"
```

### 13.3 Library Decisions

| Library | Choice | Rationale |
|---------|--------|-----------|
| DataFrame | **Polars** (batch) + Pandas (compat) | Polars is 10–100× faster for bulk transforms; Pandas required by pandas_market_calendars and some financial libs |
| HTTP | **httpx** | Native async/await; cleaner than aiohttp for most adapters |
| Retry | **tenacity** | Best decorator ergonomics; supports exponential backoff + jitter |
| Scheduling | **APScheduler** | Lightweight enough for Mac mini solo; no server required |
| Validation | **pandera** (inline) + **great-expectations** (reports) | Pandera for fast inline; GE for post-ingest HTML dashboards |
| Logging | **structlog** | JSON output; compatible with Grafana Loki; no format string interpolation |

---

## 14. Rate Limiting & Retry Architecture

### 14.1 Per-Source Limits

| Source | Free req/min | Paid req/min | Daily limit | Burst | SENTINEL strategy |
|--------|-------------|-------------|-------------|-------|------------------|
| Polygon.io (free) | 5 | unlimited ($30/mo) | — | 5 | Token bucket, 5/min; upgrade to Starter for backfill |
| yfinance | ~60 (undocumented) | N/A | — | None | Semaphore(10); treat as validation-only |
| SEC EDGAR | 600 (10/sec) | N/A | — | None | Semaphore(10) + 0.1s sleep between calls |
| FRED | 120 (with API key) | N/A | — | None | Semaphore(20) |
| CCXT (varies by exchange) | Exchange-dependent | — | — | — | Per-exchange rate limiter |
| OpenFIGI (no key) | 25,000/min | — | — | 100/req | Batch 100 per request |
| Alpha Vantage (free) | 5 | 75 ($50/mo) | 500 | None | Last fallback only |

### 14.2 Standard Retry Decorator

```python
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)
import structlog

logger = structlog.get_logger()

def sds_retry(max_attempts: int = 3):
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((
            httpx.HTTPStatusError,
            httpx.ReadTimeout,
            asyncio.TimeoutError,
            ConnectionError,
        )),
        before_sleep=before_sleep_log(logger, log_level=logging.WARNING),
    )

# Usage on every adapter method:
@sds_retry(max_attempts=3)
async def fetch_ohlcv(self, ticker: str, start: date, end: date) -> list[OHLCVBar]:
    ...
```

### 14.3 Circuit Breaker

After 5 consecutive failures on a source, that source is paused for 15 minutes. Health event emitted. Other sources continue unaffected.

```python
class CircuitBreaker:
    def __init__(self, source: str, failure_threshold: int = 5, reset_timeout: int = 900):
        self.source = source
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.opened_at: datetime | None = None

    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if (datetime.utcnow() - self.opened_at).seconds >= self.reset_timeout:
            self.reset()
            return False
        return True

    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.opened_at = datetime.utcnow()
            emit_health_event(self.source, status="circuit_open")
```

---

## 15. XBRL Parsing Pipeline

### 15.1 EDGAR API Endpoints

```
# Company metadata + filing list
GET https://data.sec.gov/submissions/CIK{cik:010d}.json

# All XBRL facts for a company (primary endpoint for fundamentals)
GET https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json

# Cross-sectional: all companies reporting a concept in a period
GET https://data.sec.gov/api/xbrl/frames/{taxonomy}/{concept}/{unit}/{period}.json
# period format: "CY2024Q1I" (instant) | "CY2024Q1" (duration)

# Full-text filing search
GET https://efts.sec.gov/LATEST/search-index?q={query}&dateRange=custom&startdt={start}&enddt={end}
```

**Rate limit compliance:** Set `User-Agent: SENTINEL/1.0 your@email.com` header. EDGAR enforces 10 req/sec per IP.

### 15.2 Taxonomy Handling

```python
SUPPORTED_TAXONOMIES = ["us-gaap", "ifrs-full", "dei", "invest"]

# XBRL fact structure from companyfacts JSON:
# facts["us-gaap"]["Revenues"]["units"]["USD"] = [
#   {"end": "2023-12-31", "val": 394328000000, "accn": "0000320193-24-000006", "filed": "2024-02-02"}
# ]

def extract_concept(facts: dict, concept: str, period_type: str = "annual") -> list[dict]:
    """Extract point-in-time values for a concept."""
    results = []
    for taxonomy in SUPPORTED_TAXONOMIES:
        if concept in facts.get(taxonomy, {}):
            for unit_type, observations in facts[taxonomy][concept]["units"].items():
                for obs in observations:
                    if period_type == "annual" and "start" in obs:
                        duration = (
                            date.fromisoformat(obs["end"]) -
                            date.fromisoformat(obs["start"])
                        ).days
                        if 340 <= duration <= 380:  # Annual filing window
                            results.append({**obs, "unit": unit_type, "taxonomy": taxonomy})
    return results
```

### 15.3 Scale Factor Detection

```python
def detect_scale_factor(value: float, concept: str) -> int:
    """
    Infer scale from magnitude. Covers ~98% of US filers.
    For micro-caps: use the 'scale' attribute from the XBRL instance .xml (Phase 2).
    """
    if "PerShare" in concept or "EPS" in concept:
        return 1
    if value > 1_000_000_000:
        return 1            # Already in raw dollars
    if value > 1_000_000:
        return 1_000        # Reported in thousands
    return 1_000_000        # Reported in millions (small companies)
```

### 15.4 15 Canonical Line Items (Gen 0 Priority)

| Canonical Name | Primary XBRL Tag | Fallback Tags |
|----------------|-----------------|--------------|
| revenue | Revenues | RevenueFromContractWithCustomerExcludingAssessedTax, SalesRevenueNet |
| net_income | NetIncomeLoss | NetIncome, ProfitLoss |
| total_assets | Assets | — |
| total_liabilities | Liabilities | LiabilitiesAndStockholdersEquity (derived) |
| total_equity | StockholdersEquity | StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest |
| operating_cash_flow | NetCashProvidedByUsedInOperatingActivities | — |
| capex | PaymentsToAcquirePropertyPlantAndEquipment | CapitalExpenditureDiscontinuedOperations |
| gross_profit | GrossProfit | — |
| operating_income | OperatingIncomeLoss | — |
| ebitda | (derived: operating_income + D&A) | — |
| eps_basic | EarningsPerShareBasic | — |
| eps_diluted | EarningsPerShareDiluted | — |
| shares_outstanding | CommonStockSharesOutstanding | — |
| total_debt | LongTermDebtAndCapitalLeaseObligations | LongTermDebt |
| free_cash_flow | (derived: operating_cash_flow − capex) | — |

---

## 16. OpenFIGI Instrument Master

### 16.1 Three-Tier Resolution Cache

```python
class InstrumentMaster:
    """
    L1: in-memory LRU cache (process lifetime)
    L2: PostgreSQL ticker_aliases + company_registry
    L3: OpenFIGI API (rate-limited, batched)
    """
    OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"

    async def get_by_ticker(self, ticker: str, exchange: str = "US") -> Instrument | None:
        cache_key = f"{ticker.upper()}:{exchange}"

        # L1
        if cached := self._mem_cache.get(cache_key):
            return cached

        # L2
        if db_result := await self._db_lookup(ticker, exchange):
            self._mem_cache[cache_key] = db_result
            return db_result

        # L3 — batch to avoid rate limit waste
        self._pending_lookups.add(cache_key)
        if len(self._pending_lookups) >= 100:
            await self._flush_openfigi_batch()
        return self._mem_cache.get(cache_key)

    async def _flush_openfigi_batch(self):
        jobs = [
            {"idType": "TICKER", "idValue": t.split(":")[0], "exchCode": t.split(":")[1]}
            for t in self._pending_lookups
        ]
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.OPENFIGI_URL,
                json=jobs,
                headers={"X-OPENFIGI-APIKEY": self._api_key},
            )
        # Parse results, upsert to L2, populate L1
        ...
```

### 16.2 Ticker Change Tracking

The `cik` column is the stable anchor. When GOOGL → GOOG, TWTR → delisted, FB → META:

```sql
-- All tickers used by Meta Platforms
SELECT ticker, valid_from, valid_to
FROM ticker_aliases
WHERE cik = (SELECT cik FROM company_registry WHERE company_name ILIKE '%meta platforms%')
ORDER BY valid_from;

-- Returns: FB (2004-05-18 → 2021-10-28), META (2021-10-28 → NULL)
```

---

## 17. Continuous Futures Contract Construction

### 17.1 Method: Ratio (Multiplicative) Backward Adjustment

The ratio method multiplies historical prices by a factor at each roll, preserving return percentages rather than absolute price levels. Correct for strategies that use returns (momentum, trend-following).

```python
def build_continuous_series(
    symbol: str,
    front_month_bars: dict[str, list[OHLCVBar]],  # contract_month → bars
    roll_schedule: list[tuple[date, str, str]]     # (roll_date, from_contract, to_contract)
) -> list[OHLCVBar]:
    """
    Ratio backward adjustment:
    For each roll date, compute: factor = price_new_front / price_old_front (on roll day)
    Multiply ALL historical bars before that roll date by that factor.
    """
    continuous = []
    cumulative_factor = Decimal("1.0")

    for roll_date, old_contract, new_contract in reversed(roll_schedule):
        old_close = get_close(front_month_bars[old_contract], roll_date)
        new_close = get_close(front_month_bars[new_contract], roll_date)
        if old_close and new_close and old_close != 0:
            roll_factor = new_close / old_close
            cumulative_factor *= roll_factor

        for bar in front_month_bars[old_contract]:
            if bar.timestamp.date() < roll_date:
                bar.adjustment_factor = cumulative_factor
                continuous.append(bar)

    return sorted(continuous, key=lambda b: b.timestamp)
```

### 17.2 Gen 0 Priority Contracts

| Symbol | Underlying | Exchange | Free Source |
|--------|-----------|----------|-------------|
| ES | S&P 500 futures | CME | Quandl CHRIS/CME_ES |
| NQ | Nasdaq 100 futures | CME | Quandl CHRIS/CME_NQ |
| CL | WTI Crude Oil | NYMEX | Quandl CHRIS/CME_CL |
| GC | Gold | COMEX | Quandl CHRIS/CME_GC |
| ZN | 10-Year T-Note | CBOT | Quandl CHRIS/CME_ZN |

---

## 18. Market Calendar Integration

```python
from pandas_market_calendars import get_calendar

def get_expected_trading_days(
    start: date,
    end: date,
    exchange: str = "NYSE"
) -> set[date]:
    cal = get_calendar(exchange)
    sessions = cal.sessions_in_range(
        pd.Timestamp(start),
        pd.Timestamp(end)
    )
    return {s.date() for s in sessions}

def detect_gaps(
    bars: list[OHLCVBar],
    exchange: str = "NYSE"
) -> dict[date, str]:
    """Return {missing_date: reason} for all missing trading days."""
    bar_dates = {b.timestamp.date() for b in bars}
    first = min(bar_dates) if bar_dates else None
    last = max(bar_dates) if bar_dates else None
    if not first:
        return {}
    expected = get_expected_trading_days(first, last, exchange)
    return {d: "data_gap" for d in (expected - bar_dates)}
```

---

## 19. Scheduler Architecture

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.redis import RedisJobStore

scheduler = AsyncIOScheduler(
    jobstores={"default": RedisJobStore(host="redis", port=6379)},
    job_defaults={
        "misfire_grace_time": 300,   # 5-min grace period for missed jobs
        "coalesce": True,            # Collapse multiple missed runs into one
        "max_instances": 1,          # Prevent concurrent duplicate runs
    }
)

# Real-time: every minute during market hours (9:30–16:00 ET, Mon–Fri)
scheduler.add_job(fetch_realtime_quotes, "cron",
    minute="*/1", hour="9-16", day_of_week="mon-fri",
    timezone="America/New_York", id="realtime_1m")

# Daily: fetch previous day bars after 6pm ET
scheduler.add_job(backfill_daily_bars, "cron",
    hour=18, minute=0, timezone="America/New_York", id="daily_bars")

# Daily: EDGAR filings check at 2am ET
scheduler.add_job(fetch_new_filings, "cron",
    hour=2, minute=0, timezone="America/New_York", id="edgar_nightly")

# Weekly: full FRED series refresh Sunday 3am ET
scheduler.add_job(refresh_macro_series, "cron",
    day_of_week="sun", hour=3, timezone="America/New_York", id="fred_weekly")

# On-demand: triggered by EDGAR webhook
async def on_edgar_filing(accession_number: str):
    scheduler.add_job(
        process_new_filing,
        args=[accession_number],
        id=f"filing_{accession_number}",
        replace_existing=True
    )
```

---

## 20. Monitoring & Observability

### 20.1 Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# All metrics prefixed sds_
INGEST_RECORDS = Counter(
    "sds_ingest_records_total",
    "Total records ingested",
    ["source", "asset_class", "status"]  # status: success|error|duplicate
)
INGEST_LATENCY = Histogram(
    "sds_ingest_latency_seconds",
    "Ingestion latency per batch",
    ["source", "operation"],
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 120]
)
GAP_COUNT = Gauge(
    "sds_data_gap_count",
    "Current number of missing trading day gaps",
    ["source", "ticker"]
)
ANOMALY_COUNT = Counter(
    "sds_anomaly_count_total",
    "Data anomalies detected",
    ["anomaly_type", "source"]
)
CA_DISAGREEMENTS = Counter(
    "sds_corporate_action_disagreements_total",
    "Corporate actions where sources disagree",
    ["action_type"]
)
VALIDATION_PASS_RATE = Gauge(
    "sds_validation_pass_rate",
    "Cross-source validation pass rate (0.0–1.0)",
    ["source"]
)
SOURCE_STALENESS = Gauge(
    "sds_source_staleness_seconds",
    "Seconds since last successful update per source",
    ["source"]
)
PROVENANCE_RECEIPTS = Counter(
    "sds_provenance_receipts_issued_total",
    "Total provenance receipts issued"
)
DATABASE_ROWS = Gauge(
    "sds_database_rows",
    "Approximate row count per table",
    ["table"]
)
```

### 20.2 Grafana Dashboard Layout

**Dashboard: SENTINEL Data Quality**

```
Row 1: Source Health
  [Source Health Grid — 3×3 badges: Polygon | yfinance | EDGAR | FRED | CCXT | Alpaca | FINRA | OpenFIGI]
  Metric: sds_source_staleness_seconds < 86400 → GREEN | < 172800 → YELLOW | else → RED

Row 2: Ingest Rate
  [Time series: sds_ingest_records_total by source, last 7 days]
  [Time series: sds_ingest_latency_seconds p95 by source]

Row 3: Quality
  [sds_anomaly_count_total by type, bar chart]
  [sds_validation_pass_rate by source, gauge]
  [sds_corporate_action_disagreements_total, stat panel]

Row 4: Gaps & Coverage
  [sds_data_gap_count top 10 tickers, table]
  [Database row counts by table, horizontal bar chart]
```

### 20.3 Alert Rules

```yaml
groups:
  - name: sds_alerts
    rules:
      - alert: SourceStalenessHigh
        expr: sds_source_staleness_seconds > 86400
        for: 30m
        labels: {severity: warning}
        annotations:
          summary: "{{ $labels.source }} has not updated in 24+ hours"

      - alert: ValidationPassRateLow
        expr: sds_validation_pass_rate < 0.99
        for: 10m
        labels: {severity: critical}
        annotations:
          summary: "Validation pass rate below 99% for {{ $labels.source }}"

      - alert: AnomalySpike
        expr: rate(sds_anomaly_count_total[1h]) > 100
        for: 5m
        labels: {severity: warning}
        annotations:
          summary: "Anomaly rate exceeding 100/hour"

      - alert: DataGapRateHigh
        expr: sds_data_gap_count > 50
        for: 1h
        labels: {severity: warning}
        annotations:
          summary: "More than 50 data gaps detected"
```

### 20.4 Structured Logging Format

```python
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

# Every log entry includes:
log = structlog.get_logger().bind(
    source="polygon",
    operation="fetch_ohlcv",
    ticker="AAPL",
)
log.info("ingestion_complete",
    record_count=252,
    duration_ms=1240,
    validation_passed=True,
    anomalies_flagged=0,
    pipeline_version="0.1.0",
)
```

---

## 21. Testing Strategy

### 21.1 Test Pyramid

| Layer | Count | Tools | What It Tests |
|-------|-------|-------|--------------|
| **Unit** | ~140 tests | pytest, hypothesis | Schema validation, adjustment math, gap detection, normalizers |
| **Integration** | ~40 tests | pytest-asyncio, vcr cassettes | Real DB + HTTP replay, full adapter-to-store cycle |
| **Financial Evals** | ~20 tests | pytest markers | Known ground-truth CA events, delistings, restatements |

### 21.2 conftest.py Setup

```python
import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

@pytest.fixture(scope="session")
def event_loop():
    return asyncio.new_event_loop()

@pytest.fixture
async def db_session(test_engine):
    """Per-test session with rollback — no test pollution."""
    async with AsyncSession(test_engine) as session:
        async with session.begin():
            yield session
            await session.rollback()

@pytest.fixture
def polygon_vcr(vcr):
    """VCR cassette for Polygon HTTP calls."""
    return vcr.use_cassette("tests/cassettes/polygon_aapl.yaml")
```

### 21.3 Financial Eval Test Cases

**Corporate Action Evals (10 known events):**

| Company | Event | Date | Expected Factor | Source |
|---------|-------|------|----------------|--------|
| AAPL | 4-for-1 split | 2020-08-31 | 0.25 | Polygon + yfinance |
| TSLA | 5-for-1 split | 2020-08-31 | 0.20 | Polygon + yfinance |
| AMZN | 20-for-1 split | 2022-06-06 | 0.05 | Polygon + yfinance |
| GOOG | 20-for-1 split | 2022-07-18 | 0.05 | Polygon + yfinance |
| NVDA | 10-for-1 split | 2024-06-10 | 0.10 | Polygon + yfinance |
| BRK.B | Cash dividend | 2023-12-29 | ~0.9997 | Polygon + yfinance |
| MSFT | 2-for-1 split | 2003-02-18 | 0.50 | EDGAR 8-K |
| GE | 1-for-8 reverse split | 2021-07-30 | 8.00 | Polygon + EDGAR |
| WM | Cash dividend | 2023-09-15 | ~0.9988 | Polygon |
| GM | Bankruptcy/delisting | 2009-07-10 | N/A → DELISTED | EDGAR |

**Survivorship Bias Evals (5 known delistings):**

| Company | CIK | Delisted | Reason |
|---------|-----|---------|--------|
| Lehman Brothers | 0000806157 | 2008-09-15 | Bankruptcy |
| Enron | 0000101830 | 2001-12-02 | Bankruptcy |
| Bear Stearns | 0000014846 | 2008-03-14 | Acquired (JPM) |
| Circuit City | 0000200406 | 2009-03-08 | Bankruptcy |
| Washington Mutual | 0000933136 | 2008-09-26 | FDIC seizure |

**Point-in-Time Eval (3 known restatements):**

1. GE (CIK 0000040987): 2019 restatement of 2016–2018 revenues — verify `filed_at <= 2018-12-31` returns pre-restatement value
2. Under Armour (CIK 0001336917): 2019 SEC investigation → Q3 2016 revenue restated — verify PIT isolation
3. General Electric Power: 2021 restatement — verify original vs amended accession numbers both stored

### 21.4 Property Tests (Hypothesis)

```python
from hypothesis import given, strategies as st
from decimal import Decimal

@given(
    factor=st.decimals(min_value=Decimal("0.001"), max_value=Decimal("100")),
    close=st.decimals(min_value=Decimal("0.01"), max_value=Decimal("100000")),
)
def test_adjustment_factor_always_positive(factor, close):
    action = CorporateAction(factor=factor, ...)
    bar = OHLCVBar(close=close, adjustment_factor=factor, ...)
    assert bar.adjusted_close > 0

@given(
    open_=st.decimals(min_value=Decimal("0.01"), max_value=Decimal("100000")),
    spread=st.decimals(min_value=Decimal("0"), max_value=Decimal("100")),
)
def test_ohlcv_invariant_high_gte_low(open_, spread):
    """high >= open >= low always."""
    high = open_ + spread
    low = max(Decimal("0.01"), open_ - spread)
    bar = OHLCVBar(open=open_, high=high, low=low, close=open_, ...)
    assert bar.high >= bar.open
    assert bar.high >= bar.low
    assert bar.open >= bar.low
```

### 21.5 CI Pipeline

```yaml
# .github/workflows/test.yml (or local equivalent)
on: [push]
jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - pytest tests/unit -m "not slow" --cov=sentinel/sds --cov-report=term-missing

  integration:
    runs-on: ubuntu-latest
    schedule: "0 2 * * *"   # Nightly at 2am
    services:
      postgres: {image: "timescale/timescaledb:latest-pg16"}
      redis:    {image: "redis:7"}
    steps:
      - pytest tests/integration -m integration

  financial_evals:
    schedule: "0 4 * * 0"   # Weekly on Sunday
    steps:
      - pytest tests/financial_evals -m financial_eval -v
```

---

## 22. Competitive Leapfrog — 10 Dimensions Where SENTINEL Beats Bloomberg

### Dimension 1: Provenance & Data Lineage
**Bloomberg:** No per-row attribution. Errors resolved by calling the helpdesk. No audit trail.
**SENTINEL:** Every row carries an append-only receipt: `source_id`, `ingested_at` (microsecond UTC), `content_hash` (SHA-256), `pipeline_version`, `is_corrected`. Immutable ledger. Researchers can reproduce any historical database state.
**Gen 0** | **SENTINEL: 9 / Bloomberg: 2**

### Dimension 2: Point-in-Time Fundamentals
**Bloomberg:** Serves revised numbers in "as-reported" mode by default. Restatements quietly overwrite historical cells.
**SENTINEL:** Every filing stored at original `filed_at`. Restatements appended, never overwrites. `filed_at <= simulation_date` enforced at API layer. Backtesters can never accidentally use future knowledge.
**Gen 0** | **SENTINEL: 9 / Bloomberg: 6**

### Dimension 3: Survivorship Bias Documentation
**Bloomberg/CapIQ:** Do not publish delisting methodology. Survivorship correction is opaque.
**SENTINEL:** Full open methodology. Source for every delisting event documented. Academic-grade: researchers can cite and reproduce the universe construction.
**Gen 0** | **SENTINEL: 9 / Bloomberg: 4**

### Dimension 4: AI-Native Query Interface
**Bloomberg:** Copilot is a bolt-on LLM wrapper around a 1980s keyboard-driven terminal. AI cannot directly access internal data structures.
**SENTINEL:** LLM IS the primary interface via MCP. Agent can directly query TimescaleDB, run screens, construct portfolios, explain data lineage — all through natural language. No UI required.
**Gen 1** | **SENTINEL: 9 / Bloomberg: 4**

### Dimension 5: Crypto Coverage
**Bloomberg:** Covers ~200 crypto tickers. Fragmented across multiple data feeds. No DeFi.
**SENTINEL:** CCXT covers 100+ exchanges, 10,000+ pairs, perpetual futures, funding rates. Single unified interface. Bloomberg cannot match this at any price point for crypto-native portfolios.
**Gen 0** | **SENTINEL: 9 / Bloomberg: 3**

### Dimension 6: Sovereign Deployment
**Bloomberg:** Vendor can suspend access instantly. All data leaves your premises. Cannot export raw data programmatically.
**SENTINEL:** Runs on hardware you own. Data stays on your machine. No vendor can revoke access. Entire pipeline is open-source inspectable.
**Gen 0** | **SENTINEL: 10 / Bloomberg: 0**

### Dimension 7: Export Portability
**Bloomberg:** Data accessible via BLPAPI in proprietary format. No Parquet, no DuckDB, no standard columnar format. Data locked in Bloomberg ecosystem.
**SENTINEL:** DuckDB-queryable Parquet with manifest. Any Python/R/Julia/SQL tool can consume the data directly. Jupyter notebook starter kit included.
**Gen 0** | **SENTINEL: 10 / Bloomberg: 1**

### Dimension 8: Adjustment Factor Auditability
**Bloomberg:** Adjustment methodology is a black box. You cannot inspect why a price changed or which corporate action produced which factor.
**SENTINEL:** Every adjustment factor stored with: source, ex_date, raw_value, cross-source confidence score, sources_agree flag. Full audit trail from raw price to adjusted price.
**Gen 0** | **SENTINEL: 10 / Bloomberg: 2**

### Dimension 9: Cost Per Analytical Insight
**Bloomberg:** $32,400/yr ÷ 250 trading days = $129.60/day per analyst. For a 3-person RIA: $388/day.
**SENTINEL:** Mac mini ($600 hardware amortized over 5 years = $0.33/day) + electricity (~$0.10/day) + optional Polygon Starter ($1/day) = **~$1.43/day for unlimited analysts**. Cost-per-insight is 99% lower.
**Gen 0** | **SENTINEL: 10 / Bloomberg: N/A**

### Dimension 10: Open Extensibility
**Bloomberg BLPAPI:** Read-only API. Cannot add custom data sources, custom analytics, custom screens. Bloomberg decides what data exists.
**SENTINEL:** Every adapter is a Python class with a `BaseAdapter` interface. Add any data source in < 100 lines. Add custom analytics in the screener. Add custom risk models in the portfolio engine. The terminal does what you tell it.
**Gen 0** | **SENTINEL: 10 / Bloomberg: 1**

---

## 23. Commercial Data Product — Full Packaging Specification

### 23.1 Dataset Catalog

| Dataset Name | Coverage | Format | Size (est.) |
|-------------|---------|--------|------------|
| `sentinel-us-equities-daily` | All NMS + delisted, 2009–present | Parquet | ~8 GB |
| `sentinel-us-fundamentals-pit` | S&P 1500 XBRL, 2009–present | Parquet | ~2 GB |
| `sentinel-macro-fred` | 500 curated FRED series, 1970–present | Parquet | ~200 MB |
| `sentinel-crypto-ohlcv` | Top 100 pairs × top 10 exchanges, 2017–present | Parquet | ~4 GB |
| `sentinel-corporate-actions` | All US NMS CAs with audit trail, 2009–present | Parquet | ~500 MB |

### 23.2 Parquet Schema — sentinel-us-equities-daily

```python
# Partitioned by: year / ticker (first letter)
# e.g., data/equities/year=2023/ticker_prefix=A/part-0001.parquet

EQUITIES_SCHEMA = pa.schema([
    pa.field("ticker",             pa.string()),
    pa.field("figi",               pa.string()),
    pa.field("cik",                pa.string()),
    pa.field("date",               pa.date32()),
    pa.field("open",               pa.decimal128(20, 6)),
    pa.field("high",               pa.decimal128(20, 6)),
    pa.field("low",                pa.decimal128(20, 6)),
    pa.field("close",              pa.decimal128(20, 6)),  # RAW unadjusted
    pa.field("adj_close",          pa.decimal128(20, 6)),  # Ratio-adjusted
    pa.field("volume",             pa.int64()),
    pa.field("vwap",               pa.decimal128(20, 6)),
    pa.field("adjustment_factor",  pa.decimal128(20, 10)),
    pa.field("listing_status",     pa.string()),           # active|delisted|acquired
    pa.field("source",             pa.string()),
    pa.field("is_validated",       pa.bool_()),
    pa.field("provenance_receipt", pa.string()),           # UUID of DataProvenance
])
```

### 23.3 Dataset Manifest (JSON Sidecar)

```json
{
  "dataset": "sentinel-us-equities-daily",
  "version": "2026-05",
  "generated_at": "2026-05-31T22:00:00Z",
  "pipeline_version": "0.1.0",
  "coverage": {
    "start_date": "2009-01-02",
    "end_date": "2026-05-30",
    "total_securities": 8421,
    "active_securities": 6203,
    "delisted_securities": 2218,
    "trading_days": 4377,
    "total_bars": 36823217
  },
  "quality_metrics": {
    "gap_rate_pct": 0.003,
    "validation_pass_rate_pct": 99.87,
    "ca_sources_agree_pct": 98.2,
    "anomalies_flagged": 1247,
    "anomalies_resolved": 1201
  },
  "provenance": {
    "primary_source": "polygon",
    "validation_source": "yfinance",
    "ca_source": "polygon+edgar",
    "content_hash_algorithm": "sha256"
  },
  "schema": { ... },
  "files": ["equities/year=*/ticker_prefix=*/*.parquet"],
  "partitioning": ["year", "ticker_prefix"]
}
```

### 23.4 DuckDB Starter Queries

```sql
-- 1. AAPL daily bars 2020–2024 with adjustment applied
SELECT date, adj_close, volume, adjustment_factor
FROM read_parquet('data/equities/year=202*/ticker_prefix=A/*.parquet')
WHERE ticker = 'AAPL' AND date BETWEEN '2020-01-01' AND '2024-12-31'
ORDER BY date;

-- 2. S&P 500 universe on 2015-01-01 (point-in-time, survivorship-complete)
SELECT ticker, company_name, listing_status
FROM read_parquet('data/equities/*.parquet')
WHERE date = '2015-01-01'
  AND listing_status IN ('active', 'delisted')  -- Include both
ORDER BY ticker;

-- 3. P/E ratio as-of-filing (no look-ahead bias)
SELECT
    e.ticker, e.date,
    e.adj_close / NULLIF(f.eps_diluted, 0) AS pe_ratio,
    f.filed_at
FROM read_parquet('data/equities/*.parquet') e
JOIN read_parquet('data/fundamentals/*.parquet') f
  ON e.ticker = f.ticker
  AND f.filed_at <= e.date          -- Point-in-time constraint
  AND f.period_type = 'annual'
  AND e.date BETWEEN '2020-01-01' AND '2023-12-31'
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY e.ticker, e.date
    ORDER BY f.filed_at DESC
) = 1;

-- 4. Average monthly return by sector 2010–2024 (survivorship-bias-free)
SELECT
    DATE_TRUNC('month', date) AS month,
    sic_sector,
    AVG(monthly_return) AS avg_return,
    COUNT(DISTINCT ticker) AS n_stocks  -- Includes delisted
FROM (
    SELECT ticker, date, sic_sector,
        (adj_close / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date)) - 1 AS monthly_return
    FROM read_parquet('data/equities/year=*/ticker_prefix=*/*.parquet')
    WHERE DATE_PART('day', date) = DATE_PART('day', DATE_TRUNC('month', date) + INTERVAL '1 month' - INTERVAL '1 day')
)
GROUP BY month, sic_sector
ORDER BY month, sic_sector;

-- 5. Companies delisted in 2020 with last trading price
SELECT ticker, company_name, MAX(date) AS last_trade_date, last_value(adj_close) AS last_price
FROM read_parquet('data/equities/year=2020/*.parquet')
WHERE listing_status = 'delisted'
GROUP BY ticker, company_name
ORDER BY last_trade_date;
```

### 23.5 Pricing Tiers

**Free** — $0/mo
> The SENTINEL Foundation dataset: 500 FRED macro series (1970–present) in clean Parquet. No registration required. Download via public Cloudflare R2 URL. Updated monthly.
> *Why free: establishes trust, builds the research community, generates inbound for paid tiers.*

**Researcher** — $49/mo
> Full US equity OHLCV (2009–present), survivorship-bias-free with 2,200+ delisted companies. S&P 1500 point-in-time fundamentals. Corporate action audit trail. Monthly vintage snapshots. DuckDB starter kit included. Commercial use permitted for personal research portfolios.
> *Target: Finance PhDs, independent researchers, solo quants who cannot get WRDS access.*

**Pro** — $149/mo
> Everything in Researcher + crypto OHLCV (100+ exchanges, 2017–present) + weekly vintage snapshots + priority gap resolution + full provenance manifests + Jupyter notebook library (10 pre-built factor research templates).
> *Target: Boutique quants, solo fund managers, small RIAs.*

**Institutional** — $500/mo
> Everything in Pro + direct TimescaleDB read replica access + custom export schedules + dedicated Slack channel + SLA (99.5% uptime for data API) + custom ticker universes on request.
> *Target: Family offices, small hedge funds, prop trading groups.*

---

## 24. Terminal UI — Data Quality Dashboard

### 24.1 Streamlit Page Layout

```
Sidebar:
  [SENTINEL logo]
  [Last refresh: 14:23:01 UTC]
  [Auto-refresh: ON (60s)]
  [Manual Refresh button]

Main Area:
  Header: "Data Quality — SDS Health Monitor"

  Row 1: Source Health Grid (full width)
  Row 2: [Ingest Rate (60%) | Validation Pass Rate (40%)]
  Row 3: [Anomaly Queue table (60%) | CA Disagreements (40%)]
  Row 4: [Coverage Map by year (50%) | Database Stats (50%)]
```

### 24.2 Panel 1 — Source Health Grid

```python
SOURCES = ["Polygon", "yfinance", "EDGAR", "FRED", "CCXT", "Alpaca", "FINRA", "OpenFIGI"]
SLA_HOURS = {"Polygon": 24, "EDGAR": 2, "FRED": 168, ...}

def source_badge(source: str, staleness_seconds: float) -> str:
    sla = SLA_HOURS[source] * 3600
    if staleness_seconds < sla * 0.5:
        return f'<span style="background:#22c55e;padding:4px 12px;border-radius:4px">{source} ✓</span>'
    elif staleness_seconds < sla:
        return f'<span style="background:#eab308;...">⚠ {source}</span>'
    else:
        return f'<span style="background:#ef4444;...">✗ {source}</span>'
```

### 24.3 Panel 3 — Anomaly Queue

```python
# Paginated table with resolve button
anomalies_df = load_open_anomalies()  # From data_anomalies WHERE resolved=FALSE

st.dataframe(
    anomalies_df[["detected_at", "ticker", "date", "anomaly_type",
                  "delta_pct", "source_primary", "source_secondary"]],
    column_config={
        "delta_pct": st.column_config.NumberColumn("Δ%", format="%.2f%%"),
        "anomaly_type": st.column_config.TextColumn("Type"),
    },
    hide_index=True,
)

# Resolve button (per row)
selected_id = st.selectbox("Select anomaly to resolve", anomalies_df["id"])
if st.button("Mark Resolved"):
    resolve_anomaly(selected_id)
    st.rerun()
```

---

## 25. Deployment Guide — Mac mini (Self-Hosted)

### 25.1 Prerequisites

```bash
# macOS 14+ (Sonoma or later)
# Docker Desktop 4.x
# Python 3.11+ (via pyenv recommended)
# Git

# Install pyenv
brew install pyenv
pyenv install 3.11.9
pyenv global 3.11.9

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
```

### 25.2 Makefile

```makefile
.PHONY: setup db-up db-down db-reset ingest-test backfill-sp500 test lint export-parquet backup

setup:
	poetry install
	cp .env.example .env
	@echo "Edit .env with your API keys, then run: make db-up"

db-up:
	docker compose -f infra/docker-compose.yml up -d
	sleep 5
	poetry run alembic upgrade head
	@echo "Database ready."

db-down:
	docker compose -f infra/docker-compose.yml down

db-reset:
	docker compose -f infra/docker-compose.yml down -v
	docker compose -f infra/docker-compose.yml up -d
	sleep 5
	poetry run alembic upgrade head
	@echo "Database reset complete."

ingest-test:
	poetry run python -m sentinel.sds.orchestrator --test-run --tickers AAPL,MSFT,GOOGL
	@echo "Test ingestion complete. Check logs for health events."

backfill-sp500:
	poetry run python -m sentinel.sds.orchestrator --backfill --universe sp500 --start 2009-01-01
	@echo "S&P 500 backfill initiated. ETA: 2-4 hours."

test:
	poetry run pytest tests/unit -m "not slow" --cov=sentinel/sds -q

test-all:
	poetry run pytest tests/ --cov=sentinel/sds

lint:
	poetry run ruff check sentinel/
	poetry run mypy sentinel/sds/ --strict

export-parquet:
	poetry run python -m sentinel.sds.export.parquet_exporter --output data/export/

backup:
	tar -czf backup/sentinel_$(date +%Y%m%d).tar.gz data/export/
	@echo "Backup complete: backup/sentinel_$(date +%Y%m%d).tar.gz"
```

### 25.3 .env.example

```bash
# ─── Database ───────────────────────────────────────────────────────────────
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=sentinel
POSTGRES_USER=sentinel_app
POSTGRES_PASSWORD=changeme_strong_password

# ─── Redis ──────────────────────────────────────────────────────────────────
REDIS_URL=redis://localhost:6379/0

# ─── Data Sources ────────────────────────────────────────────────────────────
# Polygon.io — free tier works; upgrade to Starter ($30/mo) for real-time
POLYGON_API_KEY=

# FRED — free API key at https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=

# OpenFIGI — free key at https://www.openfigi.com/api
OPENFIGI_API_KEY=

# Alpaca — free brokerage account at https://alpaca.markets
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Use paper for testing

# Alpha Vantage — free key at https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY=

# ─── Application ────────────────────────────────────────────────────────────
SENTINEL_ENV=development                  # development | production
SENTINEL_LOG_LEVEL=INFO
SENTINEL_PIPELINE_VERSION=0.1.0

# ─── Export / Backup ────────────────────────────────────────────────────────
PARQUET_EXPORT_PATH=data/export
BACKUP_PATH=backup
# Optional: Cloudflare R2 for off-site backup
R2_ACCOUNT_ID=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET_NAME=sentinel-data
```

### 25.4 First-Run Sequence

```bash
# 1. Clone + setup
git clone <repo> sentinel && cd sentinel
make setup                          # Install deps, create .env

# 2. Edit .env with API keys (FRED + Polygon minimum)
nano .env

# 3. Start database
make db-up                          # Docker Compose + migrations

# 4. Bootstrap instrument master
poetry run python -m sentinel.sds.instruments.master --bootstrap sp500

# 5. Test with 3 tickers
make ingest-test                    # AAPL, MSFT, GOOGL — verify pipeline works

# 6. Full S&P 500 backfill (run overnight)
make backfill-sp500                 # ETA 2–4 hours

# 7. Verify health
poetry run python -m sentinel.sds.validate.health_emitter --report

# 8. Export Parquet
make export-parquet
```

### 25.5 Mac mini LaunchAgent (Auto-Start on Boot)

```xml
<!-- ~/Library/LaunchAgents/com.sentinel.sds.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.sentinel.sds</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/USERNAME/.pyenv/shims/python</string>
        <string>-m</string>
        <string>sentinel.sds.orchestrator</string>
        <string>--daemon</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/USERNAME/Projects/DanteFinance/sentinel</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>SENTINEL_ENV</key>
        <string>production</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/sentinel-sds.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/sentinel-sds-error.log</string>
</dict>
</plist>
```

```bash
# Load + start
launchctl load ~/Library/LaunchAgents/com.sentinel.sds.plist
launchctl start com.sentinel.sds
```

### 25.6 Storage Requirements

| Data Type | Rows (est.) | Compressed Size |
|-----------|------------|----------------|
| OHLCV daily (S&P 500 + 2K others, 20yr) | ~40M | ~3 GB |
| OHLCV daily (full NMS 8K, 20yr, Gen 1) | ~160M | ~12 GB |
| Fundamentals (S&P 1500, 15yr, quarterly) | ~500K | ~800 MB |
| Macro (500 FRED series, 50yr) | ~9M | ~200 MB |
| Corporate actions | ~200K | ~50 MB |
| Provenance chain | ~50M | ~2 GB |
| **Total Gen 0** | | **~8 GB** |
| **Total Gen 1** | | **~20 GB** |

### 25.7 Troubleshooting

| Issue | Symptom | Fix |
|-------|---------|-----|
| Polygon rate limit | `429 Too Many Requests` in logs | Free tier = 5/min. Run backfill overnight. Or upgrade to Starter ($30/mo). |
| EDGAR 403 | `403 Forbidden` on bulk download | Add `User-Agent: SENTINEL/1.0 your@email.com` header (SEC requirement) |
| TimescaleDB hypertable error | `relation ohlcv_bars already exists` | Run `make db-reset` to reset; or check Alembic migration state |
| yfinance empty DataFrame | No data returned for known ticker | Silent throttle. Use Polygon as primary, yfinance validation-only |
| FRED API key missing | `ValueError: FRED_API_KEY not set` | Register free key at fred.stlouisfed.org; add to .env |

---

## 26. Security Model

### 26.1 API Key Management

All API keys live in `.env` (gitignored). Loaded via `pydantic-settings`:

```python
class Settings(BaseSettings):
    polygon_api_key: str = ""
    fred_api_key: str = ""
    openfigi_api_key: str = ""
    alpaca_api_key: str = ""
    alpaca_secret_key: str = SecretStr("")  # SecretStr prevents accidental logging

    model_config = SettingsConfig(env_file=".env", case_sensitive=False)
```

### 26.2 Database Privilege Model

```sql
-- App user: SELECT + INSERT only (no UPDATE/DELETE on price tables)
CREATE ROLE sentinel_app LOGIN PASSWORD 'strong_password';
GRANT CONNECT ON DATABASE sentinel TO sentinel_app;
GRANT USAGE ON SCHEMA public TO sentinel_app;
GRANT SELECT, INSERT ON ohlcv_bars, macro_observations TO sentinel_app;
GRANT SELECT, INSERT ON filing_snapshots, corporate_actions TO sentinel_app;
GRANT SELECT, INSERT ON provenance_chain TO sentinel_app;
REVOKE UPDATE, DELETE ON provenance_chain FROM sentinel_app;  -- Immutable

-- Admin user: full access (for migrations only)
CREATE ROLE sentinel_admin LOGIN PASSWORD 'admin_password';
GRANT ALL PRIVILEGES ON DATABASE sentinel TO sentinel_admin;
```

### 26.3 Network Isolation

```yaml
# infra/docker-compose.yml
networks:
  sentinel_internal:
    driver: bridge
    internal: true  # No external access

services:
  postgres:
    networks: [sentinel_internal]
    # NOT exposed on host port in production
  redis:
    networks: [sentinel_internal]
  sds:
    networks: [sentinel_internal]
  api:
    networks: [sentinel_internal]
    ports:
      - "8000:8000"  # Only the API port exposed
```

---

## 27. Functional Requirements

### FR-SDS-001 — Base Ingestion
- FR-SDS-001.1: Ingest US equity OHLCV daily bars for all configured tickers via Polygon.io
- FR-SDS-001.2: Cross-validate every bar against yfinance within 24 hours of ingestion
- FR-SDS-001.3: Ingest FRED series daily for all configured macro series (minimum 500 in Gen 0)
- FR-SDS-001.4: Ingest EDGAR XBRL 10-K and 10-Q filings within 60 seconds of SEC publication
- FR-SDS-001.5: Ingest CCXT OHLCV for top 50 crypto pairs across top 5 exchanges daily
- FR-SDS-001.6: Detect silent throttling (expected bar count vs actual via market calendar) on every ingest run

### FR-SDS-002 — Corporate Action Engine
- FR-SDS-002.1: Detect and store all split events within 24 hours of ex_date
- FR-SDS-002.2: Cross-validate every adjustment factor against ≥2 sources before marking `sources_agree = True`
- FR-SDS-002.3: Retroactively recompute cumulative adjustment factors for all historical bars on new action
- FR-SDS-002.4: Adjusted price view computable in < 10ms per bar at query time
- FR-SDS-002.5: All 10 financial eval CA test cases pass

### FR-SDS-003 — Survivorship Bias
- FR-SDS-003.1: `company_registry` covers 100% of EDGAR registrants (14K+ CIKs)
- FR-SDS-003.2: Update delisting status within 5 business days of SEC deregistration filing
- FR-SDS-003.3: Point-in-time universe queries include delisted securities by default
- FR-SDS-003.4: All 5 survivorship eval test cases pass (Lehman, Enron, Bear Stearns, Circuit City, WaMu)

### FR-SDS-004 — Point-in-Time Fundamentals
- FR-SDS-004.1: Every XBRL filing stored at original `filed_at` — never overwrite
- FR-SDS-004.2: Amended filings stored as new records with new `filed_at`
- FR-SDS-004.3: `filed_at <= query_date` enforced at API layer
- FR-SDS-004.4: All 3 restatement eval test cases return correct as-of values

### FR-SDS-005 — Data Quality
- FR-SDS-005.1: `DataHealthEvent` emitted for every ingestion run
- FR-SDS-005.2: Hard-reject any bar where Close > High or Close < Low
- FR-SDS-005.3: Flag any bar where price change exceeds 5 std devs of 20-day rolling window
- FR-SDS-005.4: Gap detection runs daily; missing bars for open market days flagged within 24 hours
- FR-SDS-005.5: Data quality dashboard shows health per source in terminal UI

### FR-SDS-006 — Provenance
- FR-SDS-006.1: Every batch write has a `DataProvenance` receipt
- FR-SDS-006.2: Provenance chain is append-only (no UPDATE/DELETE grants)
- FR-SDS-006.3: Every Parquet export includes a manifest JSON sidecar
- FR-SDS-006.4: Content hash (SHA-256) stored per batch; verifiable on export

### FR-SDS-007 — Performance
- FR-SDS-007.1: 5 years of daily OHLCV for 500 tickers in < 2 seconds on Mac mini M-series
- FR-SDS-007.2: Full universe point-in-time query (all active tickers on a given date) in < 500ms
- FR-SDS-007.3: Initial S&P 500 backfill (20 years) completeable in < 4 hours
- FR-SDS-007.4: TimescaleDB compression enabled; storage target < 15 GB for Gen 0 dataset

---

## 28. Build Roadmap

### Generation 0 — Foundation (Weeks 1–8, 40 Working Days)

| Week | Days | Deliverable | Acceptance Criteria |
|------|------|------------|---------------------|
| 1 | 1–5 | Environment + DB + Core Schema | `make db-up` → all tables created; `pytest tests/unit/test_types.py` passes |
| 2 | 6–10 | Polygon + FRED + yfinance adapters | Fetch + validate AAPL/MSFT/GOOGL; cross-source validation passes |
| 3 | 11–15 | EDGAR XBRL + company registry | S&P 500 fundamentals 2009–present loaded; point-in-time query returns correct `filed_at` |
| 4 | 16–20 | Corporate action engine | All 10 CA eval tests pass; adjustment factors cross-validated |
| 5 | 21–25 | Survivorship bias | All 5 delisting eval tests pass; 500+ delisted companies in registry |
| 6 | 26–30 | Validation + quality | DataHealthEvent green for 5 consecutive days; anomaly table populating |
| 7 | 31–35 | Provenance + Parquet export | Every write has receipt; Parquet DuckDB-queryable; manifest valid |
| 8 | 36–40 | Polish + performance + launch | All FR-SDS-007 perf targets met; Gen 0 launch checklist complete |

### Generation 1 — Expansion (Weeks 9–16)

- Full NMS universe (~8,000 securities) via Polygon Starter ($30/mo)
- Real-time quotes via Polygon WebSocket
- Options chains daily snapshots
- FINRA short interest (bi-monthly)
- Full EDGAR registrant backfill (~14K companies including delisted)

### Generation 2 — Intelligence + International (Weeks 17–24)

- LLM earnings call extraction → analyst consensus proxy (replaces Visible Alpha ~$200/mo)
- International equities via EODHD ($50/mo)
- TRACE bond trades (FINRA free)
- World Bank / OECD macro completion

### Generation 3 — Data Product Launch

- Parquet export pipeline (monthly vintages, Cloudflare R2)
- Public API (FastAPI read-only subset)
- Dataset documentation + methodology PDF
- Pricing tiers live ($49 Researcher / $149 Pro / $500 Institutional)

---

## 29. Dependencies and Integration Points

### 29.1 Internal SENTINEL Modules

| Module | What It Needs from SDS |
|--------|----------------------|
| **SBE** (Backtesting) | Point-in-time OHLCV, point-in-time fundamentals, survivorship-complete universe |
| **SSE** (Screener) | Latest fundamentals, latest prices, cross-sectional rankings |
| **SPR** (Portfolio Risk) | OHLCV history for covariance matrix, factor exposures |
| **SIL** (AI Layer) | Clean filing text + structured fundamentals for LLM context window |
| **SMA** (Macro) | All FRED series; macro regime indicators |
| **SFE** (Filing Engine) | Raw EDGAR filings + XBRL extractions |
| **SOD** (Ownership) | 13F + Form 4 data |

### 29.2 DanteHarvest Provenance Alignment

SDS's provenance model is deliberately aligned with DanteHarvest `harvest_core/provenance/`. When a shared `dante-core` package is extracted, the shared interface will be:

```python
class ProvenanceReceipt(Protocol):
    receipt_id: str
    issued_at: datetime
    content_hash: str
    pipeline_version: str

class ChainWriter(Protocol):
    def append(self, receipt: ProvenanceReceipt) -> None: ...
    def verify(self, receipt_id: str) -> bool: ...
```

This extraction happens after both projects have stabilized — not before.

---

## 30. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Polygon free tier rate limits block full NMS backfill | High | Medium | Staged backfill over 2–3 days; upgrade to Starter ($30/mo) |
| EDGAR XBRL taxonomy changes break parser | Medium | High | Schema drift detection; maintain tag mapping version table |
| yfinance breaks (API changes happen ~4×/year) | High | Low | yfinance is validation-only; Polygon primary. Breakage degrades confidence score, not functionality |
| CA errors discovered post-backfill | Medium | High | Retroactive recompute by design; detected errors trigger full recalculation |
| TimescaleDB disk usage | Low (Gen 0) | High | 20yr × 8K securities × daily ≈ 8GB compressed; well within 1TB Mac mini |
| FRED discontinues free API | Very Low | High | Download full dataset as flat files; FRED free since 1991 |
| Single machine failure (Mac mini) | Low | High | Weekly Parquet backup to Cloudflare R2; data fully reconstructible from sources |

---

## 31. Definition of Done

SDS Generation 0 is complete when ALL of the following are true:

**Data Coverage:**
- [ ] S&P 500 full daily OHLCV 2009–present loaded (< 0.01% gap rate)
- [ ] 1,000+ additional US equities including 500+ delisted
- [ ] S&P 1500 XBRL fundamentals 2009–present (15 canonical line items per company per quarter)
- [ ] 500+ FRED macro series loaded
- [ ] company_registry covers all EDGAR registrants (14K+ CIKs)

**Data Quality:**
- [ ] All 10 corporate action eval tests pass
- [ ] All 5 survivorship bias eval tests pass
- [ ] All 3 point-in-time restatement eval tests pass
- [ ] Corporate action error rate ≤ 0.1% on random 50-security spot-check
- [ ] DataHealthEvent green for ≥ 5 consecutive trading days
- [ ] Zero raw prices overwritten (adjustment factors only in `corporate_actions` table)
- [ ] No NULL stored as 0 or -1 anywhere in price tables

**Engineering:**
- [ ] All FR-SDS-001 through FR-SDS-007 pass automated tests
- [ ] All writes have provenance receipts; content hashes verify on export
- [ ] Query performance meets FR-SDS-007 targets on Mac mini M-series
- [ ] `make test` passes with > 90% coverage on `sentinel/sds/`
- [ ] `make lint` passes with no errors

**Product:**
- [ ] Terminal UI data quality dashboard shows health per source (green/yellow/red)
- [ ] Parquet export of S&P 500 dataset produces valid, DuckDB-queryable files
- [ ] Dataset manifest JSON sidecar generated and valid
- [ ] DuckDB 5 starter queries return correct results
- [ ] `make backup` creates valid archive

**Commercial:**
- [ ] Dataset manifest includes quality_metrics section
- [ ] Parquet schema matches `EQUITIES_SCHEMA` exactly
- [ ] Free tier dataset (FRED macro) exportable and shareable

---

*SENTINEL SDS PRD v1.0 — Party Build Complete.*
*The data is the terminal. Everything else is interface.*
*Five agents. One source of truth. Zero vendor lock-in.*
