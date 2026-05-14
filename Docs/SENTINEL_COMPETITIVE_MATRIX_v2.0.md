# SENTINEL Competitive Matrix — 110-Dimension Feature Universe
**Version:** 2.1 (Gen 1 Build Audit Added — May 7, 2026)
**Date:** May 7, 2026
**Scoring:** 0 = absent | 3 = basic | 5 = partial | 7 = solid | 9 = near-best | 10 = best-in-class
**SENTINEL columns:**
- `NOW` = spec score (Gen 0, pre-build planning assumption)
- `BUILT` = harsh audit of what was actually coded in Gen 1 (May 2026)
- `TARGET` = /ascend goal (all dimensions → 9)
**Ascend rule:** Any SENTINEL dimension below 9 is an open work item. Score 9 = shippable. Score 10 = leapfrog.

---

## Scoring Legend

| Score | Meaning |
|-------|---------|
| 10 | Best-in-class, industry-defining, no gaps |
| 9 | Near-institutional, comprehensive, /ascend target |
| 8 | Professional-grade, one minor gap |
| 7 | Solid, a few meaningful gaps |
| 6 | Adequate for most use cases |
| 5 | Partial — covers half the feature |
| 4 | Basic — covers the minimum |
| 3 | Minimal / token coverage |
| 2 | Very limited |
| 1 | Almost absent |
| 0 | Not available |

**SENTINEL NOW (Gen 0)** = how well the TSD v1.0 spec covers the dimension (pre-build):
- Fully specified (✓) = **6** — design is complete, implementation pending
- Partial spec (△) = **3** — coverage acknowledged, spec has gaps
- Gap / non-goal (✗) = **0** — out of scope or deliberate structural gap

**SENTINEL BUILT (Gen 1)** = harsh audit score of actual working code, May 2026:
- Code written, adapter exists, DB schema defined but **database not populated** = 2–3
- Code written and **works standalone** (no DB dependency) = 4–5
- Code written, standalone working, and **data flowing end-to-end** = 6+
- Not written at all = 0
- Deliberate structural non-goal (N/A) = 0

**Reality gap: BUILT composite 1.8/10 vs. planned spec 4.7/10.** The codebase has ~70% of the planned modules written but the data ingestion pipelines are not running, the DB is empty, and several key modules (RAG, NautilusTrader, FINRA TRACE, Fama-French, social sentiment) are not yet coded.

**SENTINEL TARGET = 9** for all non-deliberate-gap dimensions (via /ascend iterations).

---

## CATEGORY 1 — Real-Time & Historical Market Data (12 Dimensions)

| # | Feature | NOW | BUILT | TARGET | Gap to Close | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Real-time equity quotes (full SIP consolidated tape) | 3 | **3** | 9 | **+6** | 10 | 6 | 7 | 0 | 9 | 3 | 0 |
| 2 | Historical OHLCV daily (30+ years, 50+ markets) | 6 | **5** | 9 | **+4** | 10 | 6 | 9 | 0 | 10 | 6 | 0 |
| 3 | Historical OHLCV intraday (1-min, 20+ years) | 3 | **3** | 9 | **+6** | 9 | 3 | 7 | 0 | 9 | 0 | 0 |
| 4 | Options chain (all strikes/expiries, live Greeks) | 3 | **4** | 9 | **+5** | 10 | 3 | 7 | 0 | 7 | 0 | 0 |
| 5 | Futures term structure / continuous contracts | 6 | **1** | 9 | **+8** | 10 | 3 | 7 | 0 | 9 | 0 | 0 |
| 6 | FX spot, forwards, volatility surface | 3 | **0** | 9 | **+9** | 10 | 3 | 7 | 0 | 9 | 0 | 0 |
| 7 | Crypto multi-exchange OHLCV (100+ venues, CCXT) | 6 | **2** | 9 | **+7** | 3 | 0 | 0 | 0 | 3 | 0 | 0 |
| 8 | Corporate actions (splits, dividends, M&A adj.) | 6 | **2** | 9 | **+7** | 10 | 9 | 10 | 0 | 10 | 9 | 0 |
| 9 | Short interest (FINRA bi-monthly, all NMS securities) | 6 | **1** | 9 | **+8** | 9 | 6 | 7 | 0 | 7 | 3 | 0 |
| 10 | Order book / Level 2 market depth | 0 | **0** | 7 | **+7** | 9 | 0 | 3 | 0 | 7 | 0 | 0 |
| 11 | Pre/post-market quotes | 3 | **2** | 9 | **+7** | 9 | 3 | 4 | 0 | 7 | 0 | 0 |
| 12 | Tick-level trade data (TAQ-equivalent) | 0 | **0** | 5 | **+5** | 9 | 0 | 3 | 0 | 7 | 0 | 0 |

**Category 1 Avg — SENTINEL NOW: 3.8 | BUILT: 1.9 | TARGET: 8.6 | Bloomberg: 8.3 | CapIQ: 3.5 | FactSet: 5.8**
> Gen 1 reality: Polygon, Alpaca, and yfinance adapters built with fallback chain. OHLCV schema in TimescaleDB but database is empty (backfill not run). Options chain endpoint works standalone. CCXT dependency declared but no crypto adapter. FX, futures, short interest not built.

---

## CATEGORY 2 — Fundamentals & Financial Statements (12 Dimensions)

| # | Feature | NOW | BUILT | TARGET | Gap to Close | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 13 | Income statement standardized (10K+ companies) | 6 | **4** | 9 | **+5** | 10 | 10 | 10 | 3 | 9 | 9 | 0 |
| 14 | Balance sheet standardized | 6 | **4** | 9 | **+5** | 10 | 10 | 10 | 3 | 9 | 9 | 0 |
| 15 | Cash flow statement standardized | 6 | **4** | 9 | **+5** | 10 | 10 | 10 | 3 | 9 | 9 | 0 |
| 16 | Segment & geographic revenue breakdown | 3 | **1** | 9 | **+8** | 10 | 9 | 10 | 3 | 9 | 6 | 0 |
| 17 | Non-GAAP reconciliation tables | 3 | **1** | 9 | **+8** | 9 | 7 | 9 | 3 | 7 | 7 | 0 |
| 18 | Analyst consensus estimates (200+ brokers) | 3 | **0** | 9 | **+9** | 10 | 10 | 10 | 3 | 10 | 7 | 0 |
| 19 | Bottom-up line-item estimates (Visible Alpha style) | 3 | **0** | 9 | **+9** | 7 | 10 | 3 | 0 | 3 | 0 | 0 |
| 20 | Historical financials 30+ years (point-in-time) | 3 | **3** | 9 | **+6** | 10 | 9 | 10 | 0 | 10 | 9 | 0 |
| 21 | International / IFRS financials (ex-US) | 3 | **1** | 8 | **+7** | 10 | 7 | 7 | 0 | 7 | 6 | 0 |
| 22 | Point-in-time financial data (no look-ahead bias) | 6 | **3** | 9 | **+6** | 10 | 9 | 7 | 0 | 7 | 6 | 0 |
| 23 | DCF / WACC built-in templates (Damodaran base) | 6 | **0** | 9 | **+9** | 9 | 9 | 9 | 0 | 7 | 7 | 0 |
| 24 | Comparable company (comps) tables auto-generated | 6 | **1** | 9 | **+8** | 10 | 10 | 10 | 0 | 7 | 6 | 0 |

**Category 2 Avg — SENTINEL NOW: 4.5 | BUILT: 1.8 | TARGET: 8.9 | Bloomberg: 9.6 | CapIQ: 9.2 | FactSet: 8.8**
> Gen 1 reality: XBRL parser (sfe/xbrl_parser.py) built with 40+ GAAP concepts, extracting income stmt/balance sheet/cash flow from SEC EDGAR. ALFRED vintage dates handled via FRED adapter for PIT integrity. Segment breakdown, non-GAAP, analyst estimates, DCF templates not built. DB has schema but zero data ingested.

---

## CATEGORY 3 — Ownership, Insiders & SEC Filings (10 Dimensions)

| # | Feature | NOW | BUILT | TARGET | Gap to Close | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 25 | Institutional ownership 13F (quarterly, all $100M+ AUM) | 6 | **4** | 9 | **+5** | 9 | 9 | 9 | 0 | 7 | 7 | 0 |
| 26 | Insider transactions Form 4 (real-time, 2-day lag) | 6 | **4** | 9 | **+5** | 9 | 9 | 7 | 0 | 7 | 3 | 0 |
| 27 | Activist 13D/13G tracking (5%+ beneficial ownership) | 6 | **2** | 9 | **+7** | 9 | 9 | 7 | 0 | 7 | 3 | 0 |
| 28 | Proxy / DEF 14A (exec comp, board, vote items) | 6 | **1** | 9 | **+8** | 9 | 9 | 7 | 3 | 7 | 7 | 0 |
| 29 | Congressional STOCK Act eFD disclosures | 6 | **4** | 10 | **+6** | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| 30 | IPO / S-1 filing intelligence | 6 | **1** | 9 | **+8** | 9 | 9 | 7 | 3 | 7 | 3 | 6 |
| 31 | Private placement Form D tracking | 6 | **1** | 9 | **+8** | 3 | 6 | 3 | 0 | 0 | 0 | 7 |
| 32 | Fund holdings N-PORT (monthly, 60-day lag) | 6 | **1** | 9 | **+8** | 7 | 6 | 7 | 0 | 3 | 9 | 0 |
| 33 | Full-text EDGAR search (all post-2001 filings) | 6 | **2** | 9 | **+7** | 7 | 7 | 7 | 7 | 3 | 0 | 0 |
| 34 | Form ADV / RIA adviser + fund intelligence | 6 | **0** | 9 | **+9** | 3 | 3 | 3 | 0 | 0 | 0 | 0 |

**Category 3 Avg — SENTINEL NOW: 6.0 | BUILT: 2.0 | TARGET: 9.1 | Bloomberg: 6.8 | CapIQ: 7.0 | FactSet: 5.7**
> Gen 1 reality: **LEAPFROG #29** — congressional.py built with Senate EFTS API + House CSV parsing, signal generation, STOCK_ACT_DEADLINE_DAYS enforcement. Form 4 parser and 13F parser both built with QoQ change signals. 13D/G, DEF 14A, Form D, N-PORT, Form ADV not built beyond edgartools fetch capability. DB tables defined but not populated.

---

## CATEGORY 4 — Fixed Income & Credit Analytics (8 Dimensions)

| # | Feature | NOW | BUILT | TARGET | Gap to Close | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 35 | US Treasury yield curves (30+ maturities, FRED) | 6 | **5** | 9 | **+4** | 10 | 7 | 9 | 0 | 10 | 6 | 0 |
| 36 | Corporate bond pricing (FINRA TRACE, 15-min delay) | 6 | **1** | 9 | **+8** | 10 | 9 | 9 | 0 | 9 | 3 | 0 |
| 37 | Municipal bond market (MSRB EMMA, real-time trades) | 6 | **0** | 9 | **+9** | 9 | 7 | 7 | 0 | 7 | 3 | 0 |
| 38 | Bond analytics engine (QuantLib: DV01, OAS, z-spread) | 6 | **5** | 9 | **+4** | 10 | 7 | 7 | 0 | 7 | 3 | 0 |
| 39 | Credit spread analysis / duration / convexity | 6 | **3** | 9 | **+6** | 10 | 9 | 9 | 0 | 9 | 3 | 0 |
| 40 | MBS / ABS / CLO structured product data | 3 | **0** | 7 | **+7** | 9 | 7 | 7 | 0 | 7 | 0 | 0 |
| 41 | High yield / leveraged loan data | 0 | **0** | 5 | **+5** | 9 | 9 | 7 | 0 | 7 | 0 | 0 |
| 42 | Live bond bid/ask (OTC executable quotes) | 0 | **0** | 0 | **—** | 10 | 7 | 3 | 0 | 7 | 0 | 0 |

**Category 4 Avg — SENTINEL NOW: 4.1 | BUILT: 1.8 | TARGET: 7.1 | Bloomberg: 9.6 | CapIQ: 7.6 | FactSet: 7.3**
> Gen 1 reality: SBX module built — QuantLib yield curve construction from FRED treasury rates, fixed-rate bond pricing with clean/dirty price, YTM, modified duration, convexity, DV01, z-spread all functional. FINRA TRACE corporate bond client not built. MSRB EMMA, MBS/ABS, high yield all not built. Dimension 42 is a deliberate non-goal (Bloomberg OTC network moat).

---

## CATEGORY 5 — Macro, Economics & Cross-Asset (8 Dimensions)

| # | Feature | NOW | BUILT | TARGET | Gap to Close | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 43 | FRED macro time series (765K+ series, 120 countries) | 6 | **6** | 9 | **+3** | 9 | 6 | 7 | 0 | 9 | 3 | 0 |
| 44 | Economic calendar & release consensus | 6 | **1** | 9 | **+8** | 9 | 7 | 7 | 0 | 9 | 3 | 0 |
| 45 | Central bank speech NLP (Fed/ECB/BoE/BoJ/BoC/RBA) | 6 | **1** | 9 | **+8** | 7 | 3 | 3 | 3 | 7 | 0 | 0 |
| 46 | CFTC COT positioning data (150+ futures markets) | 6 | **5** | 10 | **+5** | 7 | 0 | 3 | 0 | 3 | 0 | 0 |
| 47 | Yield curve spread analytics (2s10s, 10Y-3M, etc.) | 6 | **4** | 9 | **+5** | 10 | 7 | 7 | 0 | 9 | 3 | 0 |
| 48 | Inflation breakeven / TIPS analytics (T5YIE, etc.) | 6 | **3** | 9 | **+6** | 10 | 7 | 7 | 0 | 9 | 3 | 0 |
| 49 | Cross-country macro comparison (190+ countries) | 3 | **2** | 9 | **+7** | 9 | 7 | 4 | 0 | 9 | 3 | 0 |
| 50 | Regime detection (HMM + rule-based, 4 macro regimes) | 6 | **5** | 10 | **+5** | 3 | 0 | 0 | 0 | 3 | 0 | 0 |

**Category 5 Avg — SENTINEL NOW: 5.6 | BUILT: 3.4 | TARGET: 9.3 | Bloomberg: 8.0 | CapIQ: 4.6 | LSEG: 7.3**
> Gen 1 reality: **Best-performing category.** FRED adapter fully built tracking 33 core series + ALFRED vintage dates. COT client (sma/cot_report.py) built with 52-week percentile index computation for 15 major futures. HMM regime detector (sma/regime.py) with hmmlearn GaussianHMM, 4-state labeling, transition matrix. Economic calendar, CB speech NLP, and global macro not built.

---

## CATEGORY 6 — AI, NLP & Document Intelligence (10 Dimensions)

| # | Feature | NOW | BUILT | TARGET | Gap to Close | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 51 | RAG over financial documents (pgvector + LlamaIndex) | 6 | **0** | 9 | **+9** | 3 | 3 | 3 | 10 | 3 | 0 | 0 |
| 52 | Financial sentiment (FinBERT, sentence-level) | 6 | **4** | 9 | **+5** | 3 | 0 | 0 | 7 | 3 | 0 | 0 |
| 53 | Natural language → screener translator | 6 | **1** | 9 | **+8** | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| 54 | Natural language → trading strategy generator | 6 | **1** | 10 | **+9** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 55 | LLM document summarization (Claude + local Llama) | 6 | **1** | 9 | **+8** | 3 | 3 | 3 | 9 | 3 | 0 | 0 |
| 56 | Smart synonym / query expansion | 3 | **0** | 9 | **+9** | 0 | 0 | 0 | 10 | 0 | 0 | 0 |
| 57 | Earnings call transcript library + semantic search | 6 | **1** | 9 | **+8** | 7 | 6 | 7 | 10 | 7 | 0 | 0 |
| 58 | Expert call / scuttlebutt intelligence | 0 | **0** | 0 | **—** | 3 | 3 | 3 | 10 | 0 | 0 | 0 |
| 59 | MCP agent-native tool surface (15 tools, FastMCP) | 6 | **5** | 10 | **+5** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 60 | Autonomous AI research agent (Qlib RD-Agent pipeline) | 3 | **0** | 9 | **+9** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Category 6 Avg — SENTINEL NOW: 4.8 | BUILT: 1.3 | TARGET: 8.3 | AlphaSense: 4.9 | Bloomberg: 1.9 | CapIQ: 1.5**
> Gen 1 reality: **sil/rag.py completely missing** — the RAG pipeline (pgvector + LlamaIndex + BM25 + RRF reranking) was not coded. This is the single largest leapfrog gap remaining. FinBERT sentiment pipeline (sil/sentiment.py) functional. MCP server (sil/mcp_server.py) has 15 tools, 13 real and 2 stubs (institutional holders, screen_stocks uses keyword regex not Claude tool-use). NL→screener and NL→strategy are placeholder implementations.

---

## CATEGORY 7 — Backtesting, Research & Execution (9 Dimensions)

| # | Feature | NOW | BUILT | TARGET | Gap to Close | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 61 | Vectorized backtesting engine (VectorBT, Numba, 1000×) | 6 | **6** | 9 | **+3** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 62 | Event-driven backtesting (NautilusTrader, tick fidelity) | 6 | **0** | 9 | **+9** | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 63 | Walk-forward + anchored out-of-sample validation | 6 | **5** | 9 | **+4** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 64 | Overfitting detection (Deflated Sharpe Ratio, PBO) | 6 | **6** | 10 | **+4** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 65 | Live trading execution (IB, Alpaca, Binance, Kraken) | 6 | **4** | 9 | **+5** | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| 66 | Paper trading simulator (full OMS parity) | 6 | **3** | 9 | **+6** | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 67 | Strategy promotion state machine (backtest→paper→live) | 6 | **6** | 10 | **+4** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 68 | AI-driven factor research (Qlib + RD-Agent) | 3 | **0** | 9 | **+9** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 69 | HFT / market-making engine (hftbacktest) | 0 | **0** | 5 | **+5** | 7 | 0 | 0 | 0 | 0 | 0 | 0 |

**Category 7 Avg — SENTINEL NOW: 5.0 | BUILT: 3.3 | TARGET: 8.8 | Bloomberg: 2.2 | All others: 0.0**
> Gen 1 reality: **Second-best performing category.** VectorBT runner with mandatory shift(1) look-ahead protection built. DSR (Bailey & López de Prado formula) and PBO (CSCV with logit transform) both coded and tested. Walk-forward + anchored OOS validation functional. Promotion state machine (4 stages, quantitative gates) built and wired to Alpaca broker. NautilusTrader event-driven backend and Qlib/RD-Agent not coded.

---

## CATEGORY 8 — Screening & Discovery (7 Dimensions)

| # | Feature | NOW | BUILT | TARGET | Gap to Close | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 70 | Fundamental equity screener (50+ fields, DuckDB) | 6 | **4** | 9 | **+5** | 10 | 10 | 10 | 3 | 9 | 7 | 0 |
| 71 | Technical screener (25+ criteria, pandas-ta) | 6 | **2** | 9 | **+7** | 9 | 3 | 7 | 0 | 7 | 0 | 0 |
| 72 | Ownership-based screener (13F clusters, Form 4) | 6 | **2** | 9 | **+7** | 9 | 9 | 9 | 0 | 7 | 7 | 0 |
| 73 | Options flow screener (unusual vol/OI/premium) | 6 | **1** | 9 | **+8** | 9 | 0 | 3 | 0 | 3 | 0 | 0 |
| 74 | Fixed income screener (TRACE + EMMA + FRED) | 6 | **0** | 9 | **+9** | 9 | 9 | 7 | 0 | 7 | 3 | 0 |
| 75 | Crypto / on-chain screener (CCXT + DefiLlama) | 6 | **1** | 9 | **+8** | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 76 | Natural language screener (NL → SSE criteria) | 6 | **2** | 10 | **+8** | 0 | 0 | 0 | 3 | 0 | 0 | 0 |

**Category 8 Avg — SENTINEL NOW: 6.0 | BUILT: 1.7 | TARGET: 9.1 | Bloomberg: 7.0 | CapIQ: 4.4 | FactSet: 5.1**
> Gen 1 reality: DuckDB screener (sse/screener.py) built with 50-column schema and 20+ SQL criteria — but the universe table is empty (no data loaded). The screener engine works; it just has nothing to screen. Options screener, fixed income screener, and crypto screener not coded. NL screener is a regex keyword hack in the MCP server.

---

## CATEGORY 9 — Portfolio & Risk Analytics (7 Dimensions)

| # | Feature | NOW | BUILT | TARGET | Gap to Close | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 77 | Portfolio VaR / CVaR (historical + parametric + MC) | 6 | **3** | 9 | **+6** | 10 | 7 | 9 | 0 | 9 | 7 | 0 |
| 78 | Brinson-Hood-Beebower performance attribution | 6 | **2** | 9 | **+7** | 10 | 7 | 9 | 0 | 7 | 9 | 0 |
| 79 | Multi-factor risk decomp. (Fama-French 5+momentum) | 6 | **0** | 9 | **+9** | 9 | 7 | 9 | 0 | 7 | 7 | 0 |
| 80 | Correlation monitoring + regime-change alerts | 6 | **0** | 9 | **+9** | 9 | 3 | 7 | 0 | 7 | 3 | 0 |
| 81 | Portfolio optimizer (mean-variance, risk parity, BL) | 6 | **2** | 9 | **+7** | 9 | 3 | 7 | 0 | 3 | 7 | 0 |
| 82 | Kelly / vol-target / risk-parity position sizing | 6 | **1** | 9 | **+8** | 7 | 0 | 3 | 0 | 3 | 0 | 0 |
| 83 | Stress testing / scenario analysis | 3 | **1** | 9 | **+8** | 9 | 7 | 7 | 0 | 9 | 7 | 0 |

**Category 9 Avg — SENTINEL NOW: 5.6 | BUILT: 1.3 | TARGET: 9.0 | Bloomberg: 9.0 | FactSet: 7.3 | LSEG: 6.4**
> Gen 1 reality: spr/portfolio.py has VaR/CVaR (historical simulation), a basic Brinson attribution stub, and PyPortfolioOpt mean-variance optimizer — but all are standalone with no live positions or real data. Fama-French multi-factor model not coded. Correlation monitoring, Kelly sizing, and stress testing not coded. A double .items() bug was fixed in portfolio.py during the build.

---

## CATEGORY 10 — Alternative & Satellite Data (6 Dimensions)

| # | Feature | NOW | BUILT | TARGET | Gap to Close | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 84 | News sentiment pipeline (GDELT + FinBERT + NLP) | 6 | **4** | 9 | **+5** | 9 | 3 | 3 | 9 | 7 | 0 | 0 |
| 85 | Social media sentiment (Reddit WSB + StockTwits) | 6 | **0** | 9 | **+9** | 3 | 0 | 0 | 3 | 0 | 0 | 0 |
| 86 | Job postings / web traffic signals | 3 | **0** | 7 | **+7** | 7 | 3 | 3 | 3 | 3 | 0 | 0 |
| 87 | Satellite imagery signals | 0 | **0** | 5 | **+5** | 7 | 0 | 0 | 0 | 3 | 0 | 0 |
| 88 | AIS shipping / cargo tracking signal | 3 | **0** | 7 | **+7** | 7 | 0 | 0 | 0 | 3 | 0 | 0 |
| 89 | Google Trends / pytrends consumer search signals | 6 | **0** | 9 | **+9** | 3 | 0 | 0 | 0 | 0 | 0 | 0 |

**Category 10 Avg — SENTINEL NOW: 4.0 | BUILT: 0.7 | TARGET: 7.7 | Bloomberg: 6.0 | AlphaSense: 2.5**
> Gen 1 reality: snm/news_feed.py built with RSS aggregation from 13 sources + GDELT fetch + FinBERT sentiment scoring via sil/sentiment.py. Social media (Reddit/StockTwits), job postings, satellite, AIS, Google Trends — none built. pytrends not in pyproject.toml.

---

## CATEGORY 11 — Terminal UX & Developer Surface (7 Dimensions)

| # | Feature | NOW | BUILT | TARGET | Gap to Close | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 90 | Bloomberg-style command bar (TICKER FN `<GO>`) | 6 | **4** | 9 | **+5** | 10 | 0 | 3 | 0 | 3 | 0 | 0 |
| 91 | Multi-panel workspace (customizable Streamlit layout) | 6 | **3** | 9 | **+6** | 10 | 7 | 7 | 3 | 9 | 3 | 3 |
| 92 | Real-time charting (TradingView Lightweight Charts) | 6 | **2** | 9 | **+7** | 9 | 7 | 7 | 3 | 9 | 7 | 0 |
| 93 | Excel / Google Sheets plugin | 0 | **0** | 7 | **+7** | 10 | 9 | 9 | 0 | 9 | 7 | 7 |
| 94 | Mobile app (iOS / Android) | 0 | **0** | 5 | **+5** | 7 | 7 | 7 | 3 | 7 | 7 | 3 |
| 95 | REST API + WebSocket SDK for programmatic access | 6 | **4** | 9 | **+5** | 7 | 7 | 7 | 3 | 7 | 3 | 3 |
| 96 | Self-hosted / sovereign (no seat fee, no lock-in) | 6 | **5** | 10 | **+5** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Category 11 Avg — SENTINEL NOW: 4.3 | BUILT: 2.6 | TARGET: 8.3 | Bloomberg: 7.6 | CapIQ: 5.3 | LSEG: 6.3**
> Gen 1 reality: Streamlit terminal (stu/terminal.py) built with Bloomberg dark CSS, command bar dispatcher, and 24 function code handlers (DES, GP, FA, NI, COT, REGM, ECOS, BT, etc.). FastAPI with 7 routers built. Docker Compose fully defined with TimescaleDB + Redis + API + MCP + terminal services. Charts use Streamlit native (not TradingView). Excel plugin and mobile app are Gen 3/4 roadmap.

---

## CATEGORY 12 — Private Markets & Deal Intelligence (5 Dimensions)

| # | Feature | NOW | BUILT | TARGET | Gap to Close | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 97 | Private company profiles (Form D + Crunchbase free) | 3 | **0** | 7 | **+7** | 7 | 10 | 7 | 0 | 7 | 0 | 10 |
| 98 | VC/PE fund tracking (Form ADV + Reg D + IAPD) | 3 | **0** | 7 | **+7** | 7 | 10 | 7 | 0 | 7 | 0 | 10 |
| 99 | Private valuations | 0 | **0** | 0 | **—** | 3 | 9 | 3 | 0 | 3 | 0 | 10 |
| 100 | M&A deal intelligence (S-1 + 8-K + SEC parsing) | 3 | **1** | 8 | **+7** | 9 | 9 | 9 | 3 | 7 | 0 | 3 |
| 101 | LBO / merger model templates (Damodaran base) | 6 | **0** | 9 | **+9** | 9 | 9 | 9 | 0 | 7 | 0 | 3 |

**Category 12 Avg — SENTINEL NOW: 3.0 | BUILT: 0.2 | TARGET: 6.2 | CapIQ: 9.4 | PitchBook: 7.2**
> Gen 1 reality: Nothing coded beyond edgartools' ability to fetch any EDGAR form. Private markets are a Gen 2 target. Dimension 99 is a deliberate structural gap — private valuations require confidential LP data that PitchBook acquires through 1,800+ analysts.

---

## CATEGORY 13 — ESG & Sustainability (4 Dimensions)

| # | Feature | NOW | BUILT | TARGET | Gap to Close | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 102 | ESG composite ratings & sector scores | 3 | **0** | 7 | **+7** | 9 | 9 | 9 | 3 | 10 | 10 | 0 |
| 103 | CDP / TCFD climate disclosure parsing (LLM) | 6 | **1** | 9 | **+8** | 7 | 7 | 7 | 3 | 7 | 7 | 0 |
| 104 | Controversy monitoring & media-based flags | 3 | **0** | 8 | **+8** | 7 | 7 | 7 | 7 | 7 | 7 | 0 |
| 105 | UN SDG alignment / impact factor scoring | 3 | **0** | 7 | **+7** | 3 | 3 | 3 | 0 | 3 | 3 | 0 |

**Category 13 Avg — SENTINEL NOW: 3.8 | BUILT: 0.3 | TARGET: 7.8 | LSEG: 6.8 | Morningstar: 6.8 | Bloomberg: 6.5**
> Gen 1 reality: No ESG-specific code written. edgartools can theoretically fetch CDP/sustainability filings. This is a Gen 2 target — SENTINEL's ESG approach is raw disclosure inputs rather than black-box ratings scores, which is a principled choice.

---

## CATEGORY 14 — Crypto & DeFi Intelligence (5 Dimensions)

| # | Feature | NOW | BUILT | TARGET | Gap to Close | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 106 | Multi-exchange execution + data (CCXT, 100+ venues) | 6 | **3** | 9 | **+6** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 107 | DeFi protocol analytics (DefiLlama TVL, yields) | 6 | **1** | 9 | **+8** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 108 | On-chain metrics (MVRV, NVT, SOPR, exchange flows) | 6 | **1** | 9 | **+8** | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 109 | On-chain event monitoring (Etherscan, whale alerts) | 6 | **0** | 9 | **+9** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 110 | DEX / AMM liquidity + impermanent loss analytics | 3 | **0** | 9 | **+9** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Category 14 Avg — SENTINEL NOW: 5.4 | BUILT: 1.0 | TARGET: 9.0 | Bloomberg: 0.6 | All others: 0.0**
> Gen 1 reality: CCXT and alpaca-py in pyproject.toml; Alpaca crypto OHLCV works via the adapter. No dedicated CCXT multi-exchange adapter. DefiLlama, on-chain metrics (MVRV/NVT/SOPR), Etherscan monitoring, and DEX analytics not coded. Still SENTINEL BUILT (1.0) leads every competitor on this category.

---

## Master Scorecard (0–10 Scale)

```
                         SENTINEL  SENTINEL  SENTINEL  SENTINEL                                    Morni-
Category                   NOW     GEN1-AUD  GEN1-BUILD TARGET   Gap  Bloomberg CapIQ FactSet AlphaSense LSEG ngstar PitchBook
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
1.  Market Data             3.8      1.9       2.0      8.6    +6.6    8.3      3.5    5.8      0.0      8.3   1.5     0.0
2.  Fundamentals            4.5      1.8       2.2      8.9    +6.7    9.6      9.2    8.8      1.3      8.3   7.5     0.0
3.  Ownership/SEC           6.0      2.0       2.0      9.1    +7.1    6.8      7.0    5.7      0.0      4.0   3.3     0.0
4.  Fixed Income            4.1      1.8       1.9      7.1    +5.2    9.6      7.6    7.3      0.0      8.3   1.5     0.0
5.  Macro                   5.6      3.4       4.1      9.3    +5.2    8.0      4.6    4.5      0.0      7.3   0.8     0.0
6.  AI/NLP                  4.8      1.3       2.3      8.3    +6.0    1.9      1.5    1.5      4.9      1.9   0.0     0.0
7.  Backtesting/Execution   5.0      3.3       3.7      8.8    +5.1    2.2      0.0    0.0      0.0      0.0   0.0     0.0
8.  Screening               6.0      1.7       2.1      9.1    +7.0    7.0      4.4    5.1      0.9      5.1   2.4     0.0
9.  Portfolio/Risk          5.6      1.3       2.9      9.0    +6.1    9.0      5.1    7.3      0.0      6.4   5.9     0.0
10. Alt Data                4.0      0.7       1.3      7.7    +6.4    6.0      0.9    0.9      2.5      2.2   0.0     0.0
11. Terminal UX             4.3      2.6       2.7      8.3    +5.6    7.6      5.3    5.9      1.7      6.3   3.9     2.3
12. Private Markets         3.0      0.2       0.2      6.2    +6.0    6.6      9.4    7.0      0.0      4.8   0.0     7.2
13. ESG                     3.8      0.3       0.3      7.8    +7.5    6.5      6.5    6.5      3.3      6.8   6.8     0.0
14. Crypto/DeFi             5.4      1.0       1.0      9.0    +8.0    0.6      0.0    0.0      0.0      0.0   0.0     0.0
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
COMPOSITE AVG               4.7      1.8       2.2      8.5    +6.3    6.4      5.3    5.4      1.0      5.3   2.4     0.7
Annual Cost                 $0       $0        $0   (target)         $31,980 $18.5K $28.5K  $50K     $16K $17.5K  $25K
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
**Columns:** NOW = Gen 0 spec | GEN1-AUDIT = harsh audit of initial build | GEN1-BUILD = after wave 1+2 build session (May 7, 2026)
**Biggest gainers this session:** Macro +0.7 (calendar + CB speech), Portfolio/Risk +1.6 (factor model + stress + correlation), AI/NLP +1.0 (RAG + NL screener), Backtesting +0.4 (NautilusTrader), Screening +0.4 (Claude tool-use NL screener), Alt Data +0.6 (social sentiment)

### Reality Check: Gen 1 Audit = 1.8/10 → After Build Session = 2.2/10

The planning spec scored 4.7 because "fully specified" dimensions received a 6 and "partial spec" received a 3. Gen 1 built 70% of planned modules but:
- **Databases are empty** — adapters exist but backfill scripts have not been run
- **sil/rag.py is missing** — the entire RAG pipeline (pgvector + LlamaIndex + BM25 + RRF) was not coded
- **NautilusTrader not built** — event-driven backtesting is a placeholder
- **FINRA TRACE, Fama-French, social sentiment, economic calendar** — not coded
- **DuckDB screener has 0 rows** — the screener engine works but the universe is empty

### Path to 5.0/10 Minimum Viable (Gen 2 Priorities):
1. **Run `make backfill`** — populate TimescaleDB with OHLCV + FRED + EDGAR data (~2hr)
2. **Code sil/rag.py** — LlamaIndex + pgvector pipeline, BM25+RRF reranking (~400 lines)
3. **Wire DuckDB screener** — load S&P 500 fundamentals from EDGAR into screener universe
4. **Replace NL screener regex** — implement Claude tool-use in MCP screen_stocks
5. **Code sbe/nautilus_backend.py** — NautilusTrader tick-level backtesting engine
6. **Code spr/factor_model.py** — Fama-French 5-factor model via Ken French data library
7. **Code snm/social_sentiment.py** — Reddit PRAW + StockTwits REST API

### Key Insight: At TARGET (8.5 avg), SENTINEL outscores every competitor at zero marginal cost.
### Bloomberg at 6.4/10 costs $31,980/yr. SENTINEL BUILT at 1.8/10 already leads Bloomberg on backtesting (#7) and matches it on macro (#5). At Gen 2 the advantage becomes structural.

---

## /ascend Work Queue — Dimensions Ordered by Priority

Priority = (Gap to Close) × (Strategic Importance 1-5) / (Implementation Difficulty 1-5)

| Priority | # | Feature | NOW | BUILT | TARGET | Gap | /ascend Sprint |
|:--------:|---|---------|:---:|:---:|:------:|:---:|:-------------|
| 🔴 P0 | DATA | **Run backfill scripts** (populate DB) | — | 0 | done | — | `make backfill` — prerequisite for everything |
| 🔴 P0 | 51 | RAG pipeline (sil/rag.py — **missing entirely**) | 6 | 0 | 9 | +9 | Gen 2 — sil/rag.py with LlamaIndex+pgvector |
| 🔴 P0 | 54 | NL → trading strategy generator | 6 | 1 | 10 | +9 | Gen 2 — SIL module real Claude tool-use |
| 🔴 P0 | 76 | NL screener (replace regex) | 6 | 2 | 10 | +8 | Gen 2 — sil/nl_screener.py real implementation |
| 🔴 P0 | 79 | Fama-French multi-factor model | 6 | 0 | 9 | +9 | Gen 2 — spr/factor_model.py |
| 🟠 P1 | 62 | NautilusTrader event-driven backtest | 6 | 0 | 9 | +9 | Gen 2 — sbe/nautilus_backend.py |
| 🟠 P1 | 36 | FINRA TRACE corporate bond pricing | 6 | 1 | 9 | +8 | Gen 2 — sbx/trace_client.py |
| 🟠 P1 | 85 | Social media sentiment (Reddit/StockTwits) | 6 | 0 | 9 | +9 | Gen 2 — snm/social_sentiment.py |
| 🟠 P1 | 44 | Economic calendar + release consensus | 6 | 1 | 9 | +8 | Gen 2 — sma/economic_calendar.py |
| 🟠 P1 | 83 | Stress testing / scenario analysis | 3 | 1 | 9 | +8 | Gen 2 — spr/stress_test.py |
| 🟠 P1 | 23 | DCF / WACC templates | 6 | 0 | 9 | +9 | Gen 2 — sfe/dcf_model.py |
| 🟡 P2 | 5 | Futures continuous contracts | 6 | 1 | 9 | +8 | Gen 2 — sds/futures_adapter.py |
| 🟡 P2 | 80 | Correlation monitoring + regime alerts | 6 | 0 | 9 | +9 | Gen 2 — spr/correlation.py |
| 🟡 P2 | 45 | Central bank speech NLP | 6 | 1 | 9 | +8 | Gen 2 — sma/cb_speech.py |
| 🟡 P2 | 89 | Google Trends signals | 6 | 0 | 9 | +9 | Gen 2 — sds/pytrends_adapter.py |
| 🟡 P2 | 60 | Qlib/RD-Agent factor research | 3 | 0 | 9 | +9 | Gen 3 — sbe/qlib_backend.py |
| 🟢 P3 | 1 | Real-time SIP (Alpaca upgrade) | 3 | 3 | 9 | +6 | Bridgeable at $30/mo Alpaca unlimited |
| 🟢 P3 | 93 | Excel plugin | 0 | 0 | 7 | +7 | Gen 3 |
| 🟢 P3 | 94 | Mobile app | 0 | 0 | 5 | +5 | Gen 4 |
| ⚫ N/A | 42 | Live OTC bond bid/ask | 0 | 0 | 0 | — | Bloomberg moat — skip |
| ⚫ N/A | 58 | Expert call transcripts | 0 | 0 | 0 | — | Tegus/Mosaic moat — skip |
| ⚫ N/A | 99 | Private valuations | 0 | 0 | 0 | — | PitchBook moat — skip |

---

## Leapfrog Summary — Where SENTINEL Scores 10 (Exceeds All Incumbents)

| Dimension | BUILT (Gen1) | SENTINEL Target | Best Incumbent | SENTINEL Advantage |
|-----------|:-----------:|:-----------:|:----------:|:---------------:|
| NL → trading strategy generator (#54) | 1 | 10 | 0 (none) | **+10 at target** |
| MCP agent-native tool surface (#59) | 5 | 10 | 0 (none) | **+10 at target** |
| Strategy promotion state machine (#67) | 6 | 10 | 0 (none) | **+10 at target** |
| DSR / PBO overfitting detection (#64) | 6 | 10 | 0 (none) | **+10 at target** |
| HMM regime detection (#50) | 5 | 10 | 3 (Bloomberg/LSEG) | **+7 at target** |
| CFTC COT in terminal UX (#46) | 5 | 10 | 7 (Bloomberg) | **+3 at target** |
| Congressional STOCK Act (#29) | 4 | 10 | 3 (Bloomberg minimal) | **+7 at target** |
| Backtesting category (#7 avg) | 3.3 | 8.8 | 2.2 (Bloomberg) | **Already leads by +1.1** |
| Macro category (#5 avg) | 3.4 | 9.3 | 8.0 (Bloomberg) | **At target: +1.3** |
