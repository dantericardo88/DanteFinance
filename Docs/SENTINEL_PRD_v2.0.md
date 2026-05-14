# SENTINEL Product Requirements Document
**Version:** 2.1
**Date:** May 7, 2026
**Status:** Generation 1 — Build Complete (May 2026). Composite: 1.8/10 built vs. 4.7/10 spec. Gen 2 active.
**Owner:** Ricky Porras / Dante Ecosystem
**Classification:** Founding PRD — supersedes compass_artifact founding spec

---

## 1. Executive Summary

### The Problem

Institutional market intelligence costs $12,000–$50,000 per user per year. Bloomberg Terminal at $31,980/yr, FactSet at $28,500/yr, S&P Capital IQ Pro at $18,500/yr, AlphaSense at $50,000/yr enterprise — these platforms generate a combined $40B+ in annual revenue from the same fundamental commodity: making sense of public, semi-public, and licensed financial data.

Their pricing is maintained by three durable moats:
1. **Bloomberg IB Chat** — the OTC bond execution network, 325K+ professionals, unreplicable
2. **CUSIP licensing monopoly** — FactSet acquired CUSIP Global Services for $1.925B in 2022
3. **Historical depth** — LSEG Datastream's 60+ years back to the 1960s

**Almost everything else is replicable with free or open-source equivalents.**

SEC EDGAR's XBRL APIs deliver the same financial statement data underlying Compustat. OpenFIGI provides CUSIP/ISIN/SEDOL resolution under MIT license at no cost. FRED delivers 765,000+ macro time series free. CCXT covers 100+ crypto exchanges through one unified Python interface. NautilusTrader provides production-grade Rust-core execution. FinBERT + LlamaIndex + pgvector replace what AlphaSense charges $50K/yr for.

### The Opportunity

The AI layer has changed the definition of the deliverable. Bloomberg and AlphaSense are layering LLM copilots on top of $25K/yr seats. **SENTINEL inverts this**: a sovereign LLM agent layer sits on top of free data and open execution engines.

**SENTINEL** is the integration of the open-source financial data stack into a single sovereign, AI-native, agent-controllable terminal — built on a single Mac mini with Apple Silicon, requiring no external cloud, costing $0/yr in seat fees.

### Competitive Score (0–10 Scale, /ascend target = 9)

| Platform | Score | Annual Cost |
|----------|:-----:|------------|
| **SENTINEL (target)** | **8.5** | **$0** |
| Bloomberg | 6.4 | $31,980/yr |
| FactSet | 5.4 | $28,500/yr |
| CapIQ Pro | 5.3 | $18,500/yr |
| LSEG | 5.3 | $16,000/yr |
| Morningstar | 2.4 | $17,500/yr |
| AlphaSense | 1.0 | $50,000/yr |
| PitchBook | 0.7 | $25,000/yr |

### Gen 1 Build Audit — May 2026

| Metric | Value |
|--------|-------|
| Python files shipped | 62 |
| Lines of code | ~7,300 |
| Modules complete | 13/13 |
| BUILT composite score | **1.8 / 10** |
| Spec score (Gen 0 plan) | 4.7 / 10 |
| Reality gap | **−2.9 points** |

**Why the gap exists:** The spec awarded 6/10 to any "fully specified" dimension. Gen 1 built most planned modules, but the data ingestion pipelines have not been run (databases are empty), `sil/rag.py` was not coded (RAG pipeline entirely missing), NautilusTrader event-driven backtest was not coded, and several analytics (FINRA TRACE, Fama-French, social sentiment, economic calendar, DCF templates) were not implemented.

**What Gen 1 actually delivers well (≥ 4 built score):**
- SMA (Macro): FRED adapter + COT index + HMM regime detector — **3.4 avg**, already ahead of CapIQ
- SBE (Backtesting): VectorBT + DSR + PBO + walk-forward + promotion state machine — **3.3 avg**, leads Bloomberg
- SFE (Filings): XBRL parser (40+ GAAP concepts), Form 4, 13F parsers
- SOD (Congressional): Senate EFTS + House CSV STOCK Act tracker
- SBX (Bond analytics): QuantLib yield curve + DV01 + z-spread + convexity
- SIL (MCP): 15-tool FastMCP server (13 real, 2 stubs)
- STU (Terminal): Streamlit terminal with 24 Bloomberg function codes + Docker Compose infra

**Gen 2 critical path to 5.0/10:**
1. `make backfill` — run existing scripts, populate TimescaleDB (~2 hrs)
2. Code `sil/rag.py` — LlamaIndex + pgvector + BM25 + RRF (~400 lines)
3. Wire DuckDB screener to real data — load S&P 500 fundamentals
4. Replace NL screener regex with Claude tool-use
5. Code `sbe/nautilus_backend.py`, `spr/factor_model.py`, `snm/social_sentiment.py`

---

## 2. Product Vision

### Vision Statement

> SENTINEL is the world's first sovereign, AI-native financial terminal — a complete market intelligence and automated trading platform that any individual or institution can own outright, run on commodity hardware, and extend with code.

### Core Principles

| Principle | Meaning |
|-----------|---------|
| **Sovereign** | User owns all data, code, and infrastructure. No vendor lock-in. No seat fees. |
| **AI-native** | LLM is not a bolt-on copilot — it is the primary interface layer via MCP |
| **Self-hosted** | Runs on a Mac mini. No mandatory cloud. |
| **Free by default** | All core functionality uses free/open data sources. Paid upgrades are opt-in. |
| **Open-core** | Core is MIT/Apache licensed. Premium data connectors are add-ons. |
| **Research-first** | Every live trading feature is preceded by rigorous backtesting with statistical validity guarantees |

### Who It's For

**Primary User: The Sovereign Quant**
A self-directed investor, independent fund manager, or solo quant who needs institutional-grade tools but refuses to pay $30K/yr per seat. They can run Python, understand basic financial concepts, and want to own their workflow end-to-end.

**Secondary User: The Boutique RIA / Family Office**
A small registered investment adviser managing $50M–$500M AUM who currently uses Bloomberg + FactSet + CapIQ ($80K+/yr in seats) and wants to replace 80% of that workflow at zero marginal cost per additional analyst.

**Tertiary User: The Academic Researcher**
A finance professor or PhD student who needs CRSP-quality data for factor research but doesn't have WRDS access. SENTINEL provides the free equivalent of Compustat/CRSP for the US equity universe back to 2009 (XBRL inception).

---

## 3. User Personas

### Persona 1 — Marco, Solo Quant Fund Manager
- Manages $2M personal account + $5M friends-and-family
- Currently pays $0/yr but manually downloads CSVs from Yahoo Finance
- Needs: backtesting, screening, execution automation, macro context
- SENTINEL value: replaces $45K/yr of Bloomberg + FactSet + AlphaSense workflow

### Persona 2 — Priya, Boutique RIA Analyst
- Works at a 3-person RIA managing $150M AUM
- Current stack: Bloomberg ($32K) + CapIQ ($18K) + Excel
- Needs: comp tables, 13F tracking, earnings analysis, risk attribution
- SENTINEL value: keeps Bloomberg (for IB chat + OTC execution), eliminates CapIQ + all research tools

### Persona 3 — James, Finance PhD Student
- Researching cross-sectional momentum + earnings surprise factors
- WRDS access is granted but limited; CRSP/Compustat require faculty sponsorship
- Needs: point-in-time data, factor research pipeline, overfitting detection
- SENTINEL value: complete free research stack equivalent to WRDS for US equities

### Persona 4 — Sarah, Crypto-native Portfolio Manager
- Manages a crypto + macro multi-strategy fund
- No institutional terminal covers crypto + DeFi + macro in one interface
- Needs: CCXT execution, on-chain analytics, macro regime detection, DeFi TVL
- SENTINEL value: the only terminal on earth that covers all four of her asset classes

---

## 4. Module Requirements

### Module SDS — SENTINEL Data Spine

**Purpose:** Unified data ingestion layer — all data from all sources flows through SDS before touching any other module.

**Functional Requirements:**
- FR-SDS-1: Ingest equity OHLCV from yfinance, Stooq, Alpha Vantage, FMP, Polygon, EODHD with automatic fallback chain on provider failure
- FR-SDS-2: Ingest all EDGAR filing types within 60 seconds of SEC publication (10-K, 10-Q, 8-K, Form 4, 13F, 13D/G, DEF 14A, S-1, N-PORT, Form D, Form ADV)
- FR-SDS-3: Ingest FRED time series on configurable refresh schedules (daily for market series, weekly for release series)
- FR-SDS-4: Ingest Finnhub WebSocket real-time quotes for configured watchlist
- FR-SDS-5: Ingest CCXT unified OHLCV for 100+ crypto exchanges
- FR-SDS-6: Emit a `DataHealthEvent` for every adapter: staleness, gap detection, schema drift, silent throttle detection
- FR-SDS-7: Normalize all raw data to canonical Pydantic v2 schemas defined in `core/types.py`
- FR-SDS-8: Compute and store point-in-time corporate action adjustment factors — no retroactive restating of historical prices

**Non-Functional Requirements:**
- NFR-SDS-1: Maximum 10 requests/second against SEC EDGAR (enforced rate limit)
- NFR-SDS-2: All adapters must implement exponential backoff with jitter on 429/503 responses
- NFR-SDS-3: Data ingested from free providers must be cached in TimescaleDB hypertables — no repeated fetching for the same symbol/period

**Acceptance Criteria:**
- [ ] AAPL 20-year daily OHLCV loads in < 5 seconds from cache
- [ ] EDGAR 8-K for any US company appears in SDS within 60s of SEC publication
- [ ] Provider failure triggers automatic fallback without user intervention
- [ ] DataHealthEvent fires when a time series has a 3+ day gap unexpectedly

**/ascend Score Target:** 9/10

---

### Module SIM — SENTINEL Instrument Master

**Purpose:** Canonical identifier resolution — every instrument has one internal ID (FIGI), regardless of how it arrives.

**Functional Requirements:**
- FR-SIM-1: Resolve ticker → FIGI via OpenFIGI API (25,000 instruments/min on registered key)
- FR-SIM-2: Store bidirectional mappings: CUSIP ↔ FIGI, ISIN ↔ FIGI, SEDOL ↔ FIGI, ticker ↔ FIGI, RIC ↔ FIGI
- FR-SIM-3: Maintain GICS (11 sectors × 25 groups × 74 industries × 163 sub-industries) classification for all equities
- FR-SIM-4: Maintain SIC and NAICS codes from SEC EDGAR submissions endpoint
- FR-SIM-5: Track instrument lifecycle events: IPO date, delisting date, name changes, ticker changes, M&A merger mappings
- FR-SIM-6: Seed from `company_tickers.json` bulk file (EDGAR, refreshed nightly)

**Acceptance Criteria:**
- [ ] Given any of: AAPL, US0378331005 (ISIN), 037833100 (CUSIP) → returns same FIGI BBG000B9XRY4
- [ ] 10,000 instrument lookup completes in < 30 seconds
- [ ] Delisted instrument lookup returns lifecycle metadata, not a 404

**/ascend Score Target:** 9/10

---

### Module SFE — SENTINEL Filing Engine

**Purpose:** Parse every SEC filing type to machine-readable structured data.

**Functional Requirements:**
- FR-SFE-1: Parse 10-K/10-Q XBRL `companyfacts` to income/balance/cashflow statements with us-gaap taxonomy normalization
- FR-SFE-2: Parse Form 4 insider transactions: transaction code (P/S/A/M/G/F), shares, price, post-tx holdings, 10b5-1 flag
- FR-SFE-3: Parse 13F-HR institutional holdings: CUSIP, share count, market value, voting authority type, call/put flag
- FR-SFE-4: Parse 13D/13G activist disclosures: beneficial owner, % stake, intent classification
- FR-SFE-5: Parse DEF 14A executive compensation tables via LLM-assisted extraction (tables vary by company)
- FR-SFE-6: Parse 8-K items 1.01–9.01 with item classification and entity tagging
- FR-SFE-7: Parse N-PORT mutual fund holdings (XBRL-based, monthly with 60-day lag)
- FR-SFE-8: Maintain point-in-time fact store — no retroactive updates to historical XBRL data
- FR-SFE-9: Bulk-ingest `companyfacts.zip` (~1.5GB) nightly for full historical backfill

**Acceptance Criteria:**
- [ ] All Form 4 transactions for AAPL for 2025 parsed with 100% field completeness
- [ ] 13F filing for top 50 hedge funds parsed within 5 minutes of EDGAR publication
- [ ] Point-in-time query for AAPL revenue as-reported on 2022-01-15 returns pre-restatement value

**/ascend Score Target:** 9/10

---

### Module SOD — SENTINEL Ownership Database

**Purpose:** Track institutional, insider, activist, and congressional ownership signals.

**Functional Requirements:**
- FR-SOD-1: Time-series institutional holdings from 13F data — track position changes quarter-over-quarter
- FR-SOD-2: Generate ownership signals: new position, position increase >20%, position decrease >20%, position exit
- FR-SOD-3: Cluster institutional holders by style (growth/value/index/quant) based on portfolio characteristics
- FR-SOD-4: Congressional STOCK Act disclosures — parse Senate eFD filings and House periodic transaction reports; generate STOCK Act signal feed
- FR-SOD-5: 13D/13G activist tracker — alert on new activist positions and intent changes
- FR-SOD-6: Form 4 insider signal: cluster by insider role (CEO/CFO/Director), filter by transaction type (open-market purchase vs. option exercise)

**Research Basis:**
- Jochec (2020) and related literature document 5-15% annualized abnormal returns from following informed congressional trades
- Lakonishok & Lee (2001): insider open-market purchases (code P) have 3-6% abnormal returns over 12 months

**Acceptance Criteria:**
- [ ] Congressional trade disclosure appears in signal feed within 24h of eFD filing
- [ ] "New position" signal fires within 4 hours of 13F filing availability
- [ ] Insider cluster view shows CEO vs. Director vs. 10% owner as separate signal streams

**/ascend Score Target:** 9/10

---

### Module SBE — SENTINEL Backtesting Engine

**Purpose:** Rigorous strategy research with two complementary engines and mandatory overfitting controls.

**Functional Requirements:**
- FR-SBE-1: VectorBT backend for vectorized research — full parameter sweeps across 1,000+ variants in under 60 seconds for 10-year / 500-symbol universe
- FR-SBE-2: NautilusTrader backend for event-driven bar/tick fidelity — same code path as live trading
- FR-SBE-3: Walk-forward validation (anchored and rolling window variants)
- FR-SBE-4: Deflated Sharpe Ratio (DSR) computation — mandatory before any strategy promotion
- FR-SBE-5: Probability of Backtest Overfitting (PBO, Bailey/Borwein/Lopez de Prado/Zhu 2014) — combinatorial cross-validation
- FR-SBE-6: White's Reality Check bootstrap test
- FR-SBE-7: 24-metric performance report: Sharpe, Sortino, Calmar, CAGR, MaxDD, MaxDD duration, Omega, VaR, CVaR, Beta, Alpha, Information Ratio, Hit Rate, Profit Factor, Avg Win/Loss, Expectancy, Skewness, Kurtosis, rolling 3M Sharpe, rolling 12M Sharpe, annual return by year, monthly return heatmap
- FR-SBE-8: Survivorship-bias-free universe — historical S&P 500 / Russell 2000 membership from FRED + iShares ETF holdings history
- FR-SBE-9: Automatic regime detection integration — annotate backtest tearsheets with prevailing macro regime during each period

**The 5 Deadly Backtesting Sins (all enforced by SBE):**

| Sin | Enforcement |
|-----|-------------|
| 1. Look-ahead bias | NautilusTrader strict event ordering; VectorBT `.shift(1)` enforced |
| 2. Survivorship bias | Historical index membership snapshots from FRED + ETF history |
| 3. Overfitting | DSR + PBO mandatory gates |
| 4. Transaction cost underestimation | Explicit commission + slippage per venue in NautilusTrader fill model |
| 5. Strategy decay | Rolling Sharpe monitor + auto-pause if trailing 3M Sharpe < 0.3× full-sample |

**Acceptance Criteria:**
- [ ] SPY buy-hold benchmark matches CRSP total return within 2 basis points over 20 years
- [ ] UMD momentum factor shows positive in-sample Sharpe (replicating known academic result)
- [ ] DSR computation completes for any strategy in < 10 seconds
- [ ] Strategy with obvious overfitting (100 parameters, 3-year IS) fails PBO gate automatically

**/ascend Score Target:** 10/10 (leapfrog — no competitor has this)

---

### Module SSE — SENTINEL Screener Engine

**Purpose:** Multi-dimensional security screening across all asset classes.

**Functional Requirements:**
- FR-SSE-1: Fundamental screener — 50+ fields including P/E, P/B, P/S, EV/EBITDA, EV/Revenue, gross margin, EBITDA margin, net margin, ROIC, ROE, ROA, debt/equity, current ratio, revenue growth (1Y/3Y/5Y), EPS growth, FCF yield, dividend yield, payout ratio, market cap, enterprise value, float, shares outstanding
- FR-SSE-2: Technical screener — 25+ criteria: RSI (configurable period), MACD signal crossover, Bollinger Band position, ATR, ADX, 52-week high/low proximity, SMA/EMA crossovers, volume vs. average, consecutive up/down days
- FR-SSE-3: Ownership screener — new institutional positions, insider cluster buys, activist entry, congressional buys, short interest change
- FR-SSE-4: Options flow screener — put/call ratio threshold, unusual volume/OI ratio, large premium single trades (whales), short-dated OTM call sweeps
- FR-SSE-5: Fixed income screener — yield, duration, credit quality (TRACE), muni vs. corporate spread, maturity bucket
- FR-SSE-6: Crypto / on-chain screener — MVRV z-score, NVT ratio, exchange outflow signal, funding rate, open interest vs. volume
- FR-SSE-7: Natural language screener — parse plain English query into SSE criteria via LLM (SIL module)
- FR-SSE-8: `@sentinel.factor` decorator SDK — users define custom factors in Python; SSE integrates them as first-class screening criteria
- FR-SSE-9: Saved screens with configurable alert thresholds — email/webhook on new hits

**Acceptance Criteria:**
- [ ] Fundamental screen for "S&P 500, EV/EBITDA < 10, ROIC > 15%, debt/equity < 0.5" returns correct results matching manual calculation
- [ ] NL query "show me small-cap profitable software companies with insider buying" translates correctly to: market cap < $2B, SIC 73xx, net income > 0, Form 4 open-market purchase in last 90 days
- [ ] Custom factor from `@sentinel.factor` decorator appears in screener criteria dropdown without code restart

**/ascend Score Target:** 9/10

---

### Module STU — SENTINEL Terminal UI

**Purpose:** Bloomberg-parity terminal interface built on Streamlit with real-time panel updates.

**Functional Requirements:**
- FR-STU-1: Bloomberg-style command bar — user types `AAPL DES <Enter>` → opens security description card
- FR-STU-2: Function dispatcher maps 50+ Bloomberg function codes to SENTINEL panels (DES, GP, EQS, PORT, NEWS, FISR, 13F, COMP, WACC, etc.)
- FR-STU-3: Multi-panel workspace — up to 4 panels simultaneously, drag-to-resize, save/load layouts
- FR-STU-4: TradingView Lightweight Charts integration — multi-instrument charting, 20+ indicator overlays, comparison mode
- FR-STU-5: Real-time news stream — filtered by watchlist entities, sentiment-tagged, click to full article
- FR-STU-6: Filings stream — real-time EDGAR feed for watched companies, click to parsed summary
- FR-STU-7: Watchlist panel — real-time quote updates via Finnhub WebSocket, P/L tracking
- FR-STU-8: Order panel — paper and live order entry, OMS status, fill confirmation
- FR-STU-9: Portfolio panel — real-time NAV, position table, VaR, factor exposures, attribution
- FR-STU-10: Macro panel — FRED series browser, yield curve visualizer, regime indicator
- FR-STU-11: Screener panel — full SSE interface with results table, sort, export to CSV/JSON
- FR-STU-12: Strategy panel — backtesting launcher, tearsheet viewer, promotion status

**Acceptance Criteria:**
- [ ] Typing `AAPL DES` opens security description within 500ms
- [ ] Watchlist updates quote prices within 1 second of Finnhub WebSocket message
- [ ] Chart loads 20 years of AAPL daily data within 2 seconds from cache
- [ ] Portfolio NAV updates in real-time without page refresh

**/ascend Score Target:** 9/10

---

### Module SEE — SENTINEL Execution Engine

**Purpose:** Live and paper trading execution with pre-trade risk controls and strategy promotion governance.

**Functional Requirements:**
- FR-SEE-1: Alpaca broker adapter — paper and live; supports fractional shares, stocks, ETFs, crypto
- FR-SEE-2: Interactive Brokers adapter — ibapi; supports equities, options, futures, FX, bonds
- FR-SEE-3: Binance adapter — spot + futures via CCXT + NautilusTrader adapter
- FR-SEE-4: Kraken, OANDA, Coinbase adapters
- FR-SEE-5: Pre-trade risk checks: max position size ($ and % of portfolio), max sector concentration, correlation limit to existing holdings, drawdown circuit breaker
- FR-SEE-6: Strategy promotion state machine — codified transitions: `BACKTEST → PAPER → CAPPED_LIVE → FULL_AUTONOMOUS`; each gate requires: DSR > threshold, paper Sharpe > 0.7, paper max DD < 15%, 60-day paper run minimum
- FR-SEE-7: Kill switch — emergency halt on all open orders + positions flatten, accessible via terminal command and API endpoint
- FR-SEE-8: Complete audit trail — every order, fill, cancel, modification logged to PostgreSQL with microsecond timestamps
- FR-SEE-9: Paper trading simulator using NautilusTrader `BacktestNode` — identical code path to live trading

**Strategy Promotion Gates:**

| Gate | Backtest → Paper | Paper → Capped Live | Capped Live → Full Auto |
|------|:----------------:|:-------------------:|:-----------------------:|
| DSR | > 0.5 | — | — |
| PBO | < 0.5 | — | — |
| Paper Sharpe | — | > 0.7 (90-day) | > 0.9 (180-day) |
| Paper Max DD | — | < 15% | < 10% |
| Paper run duration | — | ≥ 60 days | ≥ 180 days |
| Live Sharpe | — | — | > 0.8 (90-day) |
| Human approval | Required | Required | Required |

**Acceptance Criteria:**
- [ ] Kill switch halts all open orders within 2 seconds
- [ ] Paper trading PnL matches backtest within 10% on same strategy over 30-day period
- [ ] Strategy promotion state machine blocks promotion when any gate fails

**/ascend Score Target:** 9/10

---

### Module SPR — SENTINEL Portfolio & Risk Engine

**Purpose:** Real-time portfolio tracking, risk attribution, and portfolio optimization.

**Functional Requirements:**
- FR-SPR-1: Real-time NAV calculation from live prices + position table
- FR-SPR-2: Brinson-Hood-Beebower attribution — allocation, selection, interaction effects vs. configurable benchmark
- FR-SPR-3: Factor risk decomposition — Fama-French 5-factor + momentum exposures, updated daily from FRED/AQR factor returns
- FR-SPR-4: Portfolio VaR (1-day, 10-day) and CVaR — parametric (normal + Student-t), historical simulation, Monte Carlo
- FR-SPR-5: Correlation monitoring — real-time pairwise correlation matrix + alert on correlation spike (potential regime change)
- FR-SPR-6: Portfolio optimizer — PyPortfolioOpt + Riskfolio-Lib: mean-variance, min-variance, risk parity, max Sharpe, Black-Litterman
- FR-SPR-7: Position sizing — Kelly criterion (full and fractional), volatility targeting (vol-target), risk-parity weighting
- FR-SPR-8: Stress testing — historical scenario replay (2008, 2020, 2022 rate shock, 1987) + user-defined factor shocks

**Acceptance Criteria:**
- [ ] Portfolio VaR matches empyrical-reloaded calculation for same portfolio within 1 basis point
- [ ] Brinson attribution effects sum to total active return (round-trip test)
- [ ] Factor exposure report loads within 3 seconds for 50-position portfolio

**/ascend Score Target:** 9/10

---

### Module SIL — SENTINEL Intelligence Layer

**Purpose:** AI-native document intelligence and agent tool surface — the layer that makes SENTINEL AI-native vs. AI-bolted-on.

**Functional Requirements:**
- FR-SIL-1: RAG pipeline over financial documents — LlamaIndex + pgvector, hybrid BM25 + vector retrieval + RRF re-ranking
- FR-SIL-2: Financial document chunking — custom chunker preserving table structure, footnote context, XBRL inline tagging
- FR-SIL-3: Embedding model — `voyage-finance-2` primary (best financial semantic search), `BAAI/bge-small-en` local fallback
- FR-SIL-4: FinBERT sentiment classification — sentence-level, entity-linked, multi-label (positive/negative/uncertainty)
- FR-SIL-5: NL → screener: parse English queries → SSE filter criteria via structured output (Claude with tool use)
- FR-SIL-6: NL → strategy: parse English trading hypothesis → SentinelStrategy YAML spec → SBE backtest launch
- FR-SIL-7: Backtest explainer — plain-English attribution of strategy performance: "Strategy underperformed in Q4 2022 because the momentum factor reversed during the Fed tightening regime"
- FR-SIL-8: MCP server (FastMCP, 15 tools):
  - `search_filings(query, form_type, date_range)`
  - `get_financials(ticker, period, metrics)`
  - `run_screen(criteria)` → returns matching securities
  - `get_ownership(ticker, holder_type)`
  - `get_insider_trades(ticker, days_back)`
  - `get_congressional_trades(member, days_back)`
  - `get_macro_series(fred_id, start_date, end_date)`
  - `run_backtest(strategy_yaml)`
  - `get_portfolio_risk(portfolio_id)`
  - `get_earnings_call_summary(ticker, quarter)`
  - `get_sentiment(text_or_ticker)`
  - `get_cot_report(market, date_range)`
  - `get_regime(date)` → current/historical macro regime
  - `screen_natural_language(query)`
  - `explain_strategy(strategy_id)`
- FR-SIL-9: Smart synonym expansion — financial term normalization: "cloud revenue" → ["IaaS", "AWS", "Azure", "GCP", "hosting", "cloud services"]
- FR-SIL-10: Industry-specific KPI extraction from earnings calls:
  - SaaS: ARR, NRR, GRR, CAC payback, LTV/CAC, magic number, RPO
  - Banks: NIM, NIE, efficiency ratio, NPL ratio, CET1, ROTCE
  - Retail: SSS, traffic/ticket, gross margin, inventory turnover
  - Pharma: drug-by-drug revenue, pipeline NPV by phase
  - Auto: deliveries, ASP, gross margin per vehicle, regulatory credits

**Acceptance Criteria:**
- [ ] RAG query "What did Apple say about services revenue in Q1 2025 earnings?" returns accurate answer with source citation within 5 seconds
- [ ] NL screener query translates correctly in 95%+ of test cases (evaluated on 100-query benchmark)
- [ ] MCP server exposes all 15 tools and passes tool schema validation
- [ ] FinBERT sentiment on held-out financial news achieves > 85% accuracy vs. human labels

**/ascend Score Target:** 10/10 (MCP surface + NL-to-strategy = full leapfrog, no competitor has either)

---

### Module SMA — SENTINEL Macro Analyzer

**Purpose:** Comprehensive macro intelligence from free government and central bank data sources.

**Functional Requirements:**
- FR-SMA-1: FRED client — 765,000+ series, configurable subscription list, ALFRED vintage data for point-in-time macro
- FR-SMA-2: Central bank adapters — ECB Statistical Data Warehouse, BoE Statistical Database, BoC, BoJ, RBA, SNB
- FR-SMA-3: Yield curve analytics — Nelson-Siegel-Svensson curve fitting, spread monitor (2s10s, 10Y-3M, 5s30s), inversion alert
- FR-SMA-4: CFTC COT report ingestion — parse CFTC legacy + disaggregated + TFF reports; 150+ futures markets; commercial vs. large speculator vs. small speculator net positioning; COT Index (percentile of net position vs. trailing 52-week range)
- FR-SMA-5: HMM macro regime detector — 4-regime model: Growth/Inflation, Growth/Deflation, Contraction/Inflation (Stagflation), Contraction/Deflation; trained on macro indicators; updates weekly; regime history back to 1970 using FRED data
- FR-SMA-6: Economic calendar — upcoming data releases, consensus estimates (Finnhub/TradingView), prior value, surprise tracker
- FR-SMA-7: Cross-country macro dashboard — GDP growth, CPI, unemployment, current account, debt/GDP for 190+ countries from World Bank / IMF free APIs
- FR-SMA-8: Inflation analytics — CPI components breakdown, PCE vs. CPI spread, real yield = nominal - breakeven (FRED T5YIE series), breakeven term structure

**Acceptance Criteria:**
- [ ] COT report for a given market loads within 5 seconds of CFTC publication Friday
- [ ] Regime detector correctly identifies 2022 as "Contraction/Inflation (Stagflation)" regime in backtest
- [ ] Yield curve inversion alert fires within 60 minutes of 2s10s spread crossing zero

**/ascend Score Target:** 9/10

---

### Module SBX — SENTINEL Bond Analytics

**Purpose:** Fixed income analytics engine using QuantLib over free public data.

**Functional Requirements:**
- FR-SBX-1: QuantLib Python wrapper — yield curve bootstrapping from FRED Treasury par yield data
- FR-SBX-2: Bond pricing — dirty price, clean price, accrued interest, YTM, YTC, YTW for any fixed-rate bond
- FR-SBX-3: Duration analytics — modified duration, Macaulay duration, effective duration (for callables via Hull-White), DV01 / basis-point value
- FR-SBX-4: Convexity — standard + effective (for callables)
- FR-SBX-5: Spread analytics — OAS computation via short-rate model (Hull-White 1-factor), z-spread, I-spread vs. swap curve
- FR-SBX-6: FINRA TRACE corporate bond data — transaction prices for all TRACE-eligible debt; 15-minute delay on free tier
- FR-SBX-7: MSRB EMMA municipal bond data — real-time trade prices, bond profile, disclosure documents, continuing disclosure filings
- FR-SBX-8: Treasury yield curve bootstrapper — build zero-coupon curve from FRED DGS series; extrapolate using Nelson-Siegel
- FR-SBX-9: Credit analytics — yield spread by credit quality bucket, spread duration, Spread01 (DV01 in spread terms)

**Acceptance Criteria:**
- [ ] QuantLib OAS computation for AAPL 2025 bond matches Bloomberg within 3 basis points
- [ ] TRACE data for top 100 corporate bond issuers loads within 10 seconds
- [ ] Yield curve builds successfully from FRED data for any date back to 1980

**/ascend Score Target:** 9/10

---

### Module SNM — SENTINEL News & Media Intelligence

**Purpose:** Multi-source financial news ingestion, transcription, and signal extraction.

**Functional Requirements:**
- FR-SNM-1: Multi-source RSS ingestion — configurable feed list (Reuters, AP, WSJ, Bloomberg free, FT, SeekingAlpha, Motley Fool, Business Wire, PR Newswire)
- FR-SNM-2: Finnhub news API — entity-tagged financial news, 60 calls/min free
- FR-SNM-3: GDELT Project feed — global 15-minute event stream, NLP-tagged, free, no key required
- FR-SNM-4: Real-time SEC 8-K press release feed — material event detection within 60 seconds
- FR-SNM-5: Whisper-based transcription pipeline — OpenAI Whisper large-v3 (local): earnings call audio → transcript → entity tagging → storage in pgvector
- FR-SNM-6: Podcast subscription pipeline — OPML-compatible feed list; download, transcribe, chunk, embed, index
- FR-SNM-7: Federal Reserve speech pipeline — FOMC minutes, Chair press conferences, governor speeches; dovish/hawkish classification; key policy sentence extraction
- FR-SNM-8: YouTube financial transcript pipeline — yt-dlp for audio extraction; Whisper transcription; financial channel whitelist
- FR-SNM-9: Entity tagging — spaCy NER model + FIGI-linking; every mentioned company linked to its FIGI in the knowledge base
- FR-SNM-10: News-to-signal bridge — configurable rules: "if sentiment < -0.7 on earnings call AND revenue guidance is negative → generate SHORT signal for SSE alert"
- FR-SNM-11: News archive — PostgreSQL + pgvector; semantic search over all ingested content; article deduplication via MinHash

**Acceptance Criteria:**
- [ ] FOMC minutes parsed and dovish/hawkish classification delivered within 5 minutes of publication
- [ ] Earnings call for any S&P 500 company transcribed within 30 minutes of call end
- [ ] News sentiment for AAPL over trailing 30 days matches directional market performance (correlation > 0.4)

**/ascend Score Target:** 9/10

---

## 5. Generation Roadmap

### Generation 0 — Specification (Current: May 2026)
- Status: PRD + TSD complete. Architecture specified. OSS stack evaluated. No code written.
- Deliverable: This document + SENTINEL_TRD_v2.0.md + SENTINEL_OSS_UNIVERSE.md

### Generation 1 — Foundation Terminal (3 months)
**Target: Replace 80% of daily equity research workflow**
- SDS: yfinance + FRED + EDGAR adapters
- SIM: OpenFIGI resolver, ticker→FIGI mapping
- SFE: 10-K/10-Q XBRL parser, Form 4, 13F parsers
- SOD: 13F ownership tracker, Form 4 signal feed, congressional disclosures
- SSE: Fundamental + technical screener (50+ fields)
- STU: Streamlit terminal, command bar, charting, watchlist
- SIL: MCP server (15 tools), RAG pipeline, FinBERT sentiment
- SMA: FRED client, COT report, yield curve, HMM regime detector
- SBX: QuantLib wrapper, TRACE client, EMMA client
- SEE: Alpaca paper trading adapter
- SBE: VectorBT backend, DSR/PBO computation
- Value delivered: Replaces ~$30K/yr Bloomberg equity research workflow

### Generation 2 — AI Research Engine (3 months)
- SIL: NL-to-strategy generator, smart synonym expansion, earnings KPI extractor
- SBE: NautilusTrader event-driven backend, walk-forward validation, Qlib/RD-Agent integration
- SPR: Full portfolio risk engine, stress testing, optimizer
- SNM: Whisper transcription pipeline, podcast ingestion, entity tagging
- SEE: IB broker adapter (live equities, options, futures)
- SDS: Polygon adapter, Binance/Kraken crypto adapters
- Value delivered: Replaces $50K/yr AlphaSense + adds AI research capabilities no terminal offers

### Generation 3 — Execution & Scale (3 months)
- SEE: Full multi-broker live trading (IB + Alpaca + Binance + Kraken + OANDA)
- SEE: Strategy promotion state machine complete
- STU: Excel plugin (Python-to-Excel via xlwings or openpyxl)
- SDS: Level 2 order book (Polygon WebSocket, paid tier)
- SBX: MBS/ABS structured product analytics
- SMA: Global macro dashboard (190 countries, IMF/World Bank)
- Value delivered: Full autonomous systematic trading platform

### Generation 4 — Institutional Grade (6 months)
- STU: Mobile companion app (React Native + FastAPI backend)
- SIL: Autonomous research agent (Qlib RD-Agent, multi-step factor mining)
- SOD: PitchBook-equivalent private markets from Form D + Crunchbase
- SDS: Satellite imagery signals (Planet Labs research API)
- SBE: Cross-asset portfolio backtesting (equities + bonds + crypto + FX + commodities)
- Value delivered: Institutional-grade platform competitive with $100K+/yr enterprise stacks

### Generation 5–8 — Open-Core Platform (12+ months)
- Multi-user SaaS layer (optional) on top of self-hosted core
- Public data marketplace — community-contributed factors and screens
- SENTINEL Exchange — share strategies (without revealing code) via performance track record
- Enterprise connectors — Bloomberg BLPAPI, FactSet API, Refinitiv bridge (for hybrid shops)
- Regulatory reporting — Form ADV, 13F auto-generation from SENTINEL portfolio records

---

## 6. Non-Goals (Deliberate Structural Gaps)

| Non-Goal | Reason | Who Owns This Moat |
|----------|--------|-------------------|
| OTC bond live bid/ask (executable quotes) | Bloomberg IB network — 325K professionals; unreplicable network effect | Bloomberg only |
| Private company valuations | Requires confidential LP data, FOIA responses, 1,800+ data analysts | PitchBook |
| Expert call transcripts (Tegus/Mosaic style) | $30K+/yr licensing from expert network platforms | AlphaSense |
| CDS pricing | Dealer-controlled data, no public feed | Bloomberg |
| Barra multi-factor model | MSCI proprietary; $100K+/yr commercial license | MSCI / Bloomberg PORT |
| CRSP 100-year historical equity data | Academic-only access; $50K+/yr commercial | WRDS/CRSP |
| Regulatory credit ratings (Moody's/S&P/Fitch) | Gated behind licensing agreements | Bloomberg/FactSet |
| Bloomberg CUSIP licensing revenue | FactSet owns CUSIP Global Services; SENTINEL uses FIGI | FactSet |

---

## 7. Success Metrics

### Generation 1 Success (3 months from build start)
- [ ] Terminal loads in < 5 seconds on Mac mini M2
- [ ] 10,000+ US equities fully described (fundamentals, ownership, filings) in SIM/SFE
- [ ] Congressional trade signal latency < 24h from eFD filing
- [ ] Fundamental screen (50+ fields) executes in < 3 seconds
- [ ] MCP server passes all 15-tool schema validation tests
- [ ] AAPL 20-year backtest completes in VectorBT in < 10 seconds

### Generation 2 Success (6 months from build start)
- [ ] NL screener query accuracy > 90% on 100-query benchmark
- [ ] Earnings call transcribed within 30 minutes for any S&P 500 company
- [ ] Strategy with DSR < 0.5 blocked from paper trading promotion (gate test)
- [ ] Factor risk decomposition matches empyrical-reloaded within 1 basis point

### Terminal Quality Bar (ongoing — /ascend target)
- All 110 competitive dimensions scored ≥ 9/10 on non-deliberate-gap items
- Dimensions with 10/10 target: #54, #59, #64, #67, #46, #50, #29, #96 (leapfrog set)

---

## 8. Technical Constraints

| Constraint | Spec |
|-----------|------|
| Primary hardware | Mac mini Apple Silicon (M-series), 16–32 GB RAM, 1 TB SSD |
| OS | macOS (primary), Linux (Docker containers) |
| Language | Python 3.11+ primary; Rust via NautilusTrader extensions |
| Data storage | TimescaleDB (time series) + PostgreSQL (structured) + pgvector (embeddings) |
| Cache | Redis (event bus + session state + rate limit tracking) |
| API layer | FastAPI + WebSocket |
| UI | Streamlit (terminal) |
| Container | Docker Compose (all services) |
| Secrets | No plaintext secrets in code or config; `python-dotenv` + `.env` (gitignored) |
| License target | MIT for core; LGPL/Apache for dependencies where acceptable |

---

*SENTINEL PRD v2.0 — May 7, 2026*
*Next document: SENTINEL_TRD_v2.0.md (full technical blueprint)*
*Previous: compass_artifact founding spec (superseded by this document)*
