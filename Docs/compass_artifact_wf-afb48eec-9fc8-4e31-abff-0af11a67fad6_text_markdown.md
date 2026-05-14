# SENTINEL: Founding Specification Document
## A Sovereign, Self-Hosted, AI-Native Market Intelligence and Automated Trading Platform

**Document Type:** Founding PRD / Architecture Specification
**Date:** May 7, 2026
**Status:** Generation 0 — Pre-Build Research Complete
**Target Build Surface:** Mac mini (Apple Silicon), single-developer, DanteForge

---

## Executive Summary

The institutional market data and trading terminal stack — Bloomberg ($31,980/yr), FactSet (~$12K–$50K/yr), S&P Capital IQ Pro ($12K–$25K/yr per user, with Visible Alpha as add-on), LSEG Workspace, Morningstar Direct (~$17.5K/yr), PitchBook (~$25K/yr for 3 users), AlphaSense — generates roughly $40B+ in combined annual revenue from the same fundamental commodity: making sense of public, semi-public, and licensed financial data. The pricing reflects three durable moats: (1) the Instant Bloomberg (IB) chat network and OTC bond execution layer, (2) the CUSIP licensing monopoly (FactSet acquired CUSIP Global Services for $1.925B in March 2022), and (3) deep historical depth (Datastream's 60+ years).

Almost everything else is replicable with free or open-source equivalents. SEC EDGAR's XBRL/companyfacts/companyconcept APIs deliver the same financial statement data underlying Compustat. OpenFIGI provides free FIGI-mediated CUSIP/ISIN/SEDOL/ticker resolution under MIT license. FRED + BLS + BEA + Treasury FiscalData deliver the macro suite institutions pay LSEG for. FINRA TRACE, MSRB EMMA, and FRED cover most US fixed income transparency. yfinance + Finnhub + Alpha Vantage + FMP + Polygon + EODHD cover equity quotes/fundamentals/estimates with overlap that survives provider deprecation. CCXT covers 100+ crypto exchanges through one unified Python interface. NautilusTrader (Rust core, Python API) and VectorBT cover production execution and vectorized research; Microsoft Qlib covers AI-driven research with the new RD-Agent. FinBERT + LlamaIndex + pgvector + Claude/local LLM cover what AlphaSense charges $50K/yr for.

**SENTINEL** is the proposed integration of this open-source stack into a single sovereign, AI-native, agent-controllable terminal. Twelve modules span ingestion (SDS), instrument mastering (SIM), filing parsing (SFE), ownership intelligence (SOD), backtesting (SBE), screening (SSE), terminal UI (STU), execution (SEE), portfolio/risk (SPR), document intelligence (SIL), macro (SMA), and bond analytics (SBX). The build runs on a single Mac mini with Apple Silicon, in Python with selective Rust acceleration via NautilusTrader, with no external cloud dependencies required for the core terminal. AI is integrated via the Model Context Protocol (MCP), which Anthropic open-sourced in November 2024 and donated to the Linux Foundation's Agentic AI Foundation alongside Block, OpenAI, Google, Microsoft, AWS, Cloudflare, and Bloomberg, making MCP the de-facto agent-tool standard.

**Build Thesis:** The window to ship a sovereign, AI-native, MIT/Apache-style open-core terminal is open precisely *because* the AI layer changes the definition of the deliverable. AlphaSense, Bloomberg, and FactSet are layering LLM copilots on top of $25K/yr seats; SENTINEL inverts this: a sovereign LLM agent layer sits on top of free data and open execution engines. The 8-generation build plan ships value at every stage: by Generation 1 the terminal already replaces $30K/yr of equity research workflow; by Generation 8 it is a full live-trading sovereign quant lab.

---

## Section 1 — Competitive Platform Autopsy

### 1A — Bloomberg Terminal ($31,980/yr single seat, $28,320 multi-seat, 2025/2026 pricing after 6.5% annual increase)

Bloomberg's revenue is approximately $10–13B/yr from Terminal alone (~85% of LP revenue), serving ~355,000 professionals across 350+ exchanges, 35M+ instruments, and 2,700+ journalists. The function-code interface is keyboard-driven (TICKER `<Yellow Key>` FUNCTION `<GO>`).

| Function | Purpose | Free / Open Replication | SENTINEL Verdict |
|---|---|---|---|
| **DES** (Security Description) | 20+ field security overview: ticker, sector, market cap, shares out, key ratios, recent filings | yfinance `Ticker.info` + Finnhub profile + SEC EDGAR submissions endpoint, joined on FIGI | REPLICABLE FREE |
| **GP** (Charting) | Multi-instrument charting with overlays, indicators, comparisons, every asset class | TradingView Lightweight Charts (45KB, free, Apache 2.0); pandas-ta for indicators | REPLICABLE FREE |
| **EQS** (Equity Screening) | 50+ fundamental, technical, ownership, ESG fields; saved screens | TradingView Screener Python + custom Polars/DuckDB on aggregated free data | REPLICABLE FREE for top-30 fields; gaps in supply chain, alternative data |
| **PORT** (Portfolio Analytics) | Beta, VaR, CVaR, tracking error, Barra-style multi-factor risk decomposition | empyrical-reloaded + PyPortfolioOpt + Riskfolio-Lib + custom Brinson-Hood-Beebower | PARTIALLY REPLICABLE (no proprietary Barra factor model, but Fama-French 5+momentum suffices) |
| **SRCH / FISR** (Fixed Income) | Bond search across govt, corp, muni, MBS, ABS, CLO | FRED (Treasuries), FINRA TRACE (corp transactions), MSRB EMMA (muni), Treasury FiscalData | PARTIALLY REPLICABLE (TRACE has transaction prices but not live bid/ask) |
| **NEWS / ANR** | Bloomberg News + analyst reports | GDELT (free global news), Finnhub news, RSS, FRED news | PARTIALLY REPLICABLE (no analyst notes at depth) |
| **MSG / IB Chat** | Bloomberg's 325K+ professional chat network — OTC bond execution layer | None — pure network effect | NOT REPLICABLE |
| **COMP** (Comparables) | Cross-company financial comp tables, GAAP/IFRS normalized | SEC EDGAR XBRL `companyfacts` with us-gaap taxonomy normalization; FMP standardized financials | PARTIALLY REPLICABLE (US-only via XBRL; international companies require ifrs-full taxonomy) |
| **WACC / DCF** | Built-in DCF, WACC, comparable-multiples templates | Python DCF templates; Damodaran datasets free | REPLICABLE FREE |
| **CDSW** (CDS Pricing) | Credit default swap pricing | None at retail level — CDS data is dealer-controlled | NOT REPLICABLE |
| **MOST** (Most Active) | Real-time most-active by volume, price change, custom criteria | Polygon free tier + Finnhub WebSocket | REPLICABLE FREE (with 15-min delay) |

**BLPAPI / Bloomberg Data License (BDL):** Covers 35M+ instruments. **OpenFIGI is the open free counterpart**: MIT-licensed, no rate limit on registered keys (25,000 instruments/min), maps ID_ISIN, ID_CUSIP, ID_SEDOL, TICKER, ID_BB_GLOBAL → FIGI + share-class FIGI + composite FIGI. SENTINEL uses OpenFIGI as its instrument-master backbone (Module SIM).

**Verdict for SENTINEL:** Bloomberg's irreducible value is IB Chat (OTC liquidity network) and proprietary indices (Bloomberg Aggregate Bond Index benchmarks $100T+ in AUM). Equity research workflows, charting, screening, fundamentals, and portfolio analytics are all replicable in open source.

### 1B — FactSet Workstation (~$12,000–$45,000/yr/user; 122,000 users; 800+ data sources)

| Component | Coverage | Free Equivalent |
|---|---|---|
| **FactSet Fundamentals** | 80,000+ companies, as-reported and standardized | SEC EDGAR XBRL companyfacts (US); IFRS-full XBRL for international (limited) |
| **FactSet Estimates** | 900+ contributing brokers, consensus EPS/Revenue/CFO | Yahoo Finance (free, low broker count), Finnhub, FMP — none match 900-broker depth |
| **FactSet Ownership** | 340K+ institutions tracked | SEC EDGAR 13F-HR + Form 13D/13G + Form 4 — full coverage of US-listed via EdgarTools Python lib |
| **FactSet RBICS** | Industry classification, 14 anchor + 2 specialty industries × 6 levels = ~1,400 sub-industries (vs. GICS 11×4=158) | GICS via free vendors; SIC codes via SEC; FactSet RBICS itself not free |
| **FactSet GeoRev** | Geographic revenue mapping | Manually parsed from 10-K segment notes via LLM extraction |
| **FactSet Mercury** | AI copilot | LlamaIndex + Claude/Ollama + local financial doc corpus |
| **CUSIP Global Services** | FactSet subsidiary acquired March 2022 from S&P for **$1.925B**; ~$175M annual revenue at ~mid-to-high single-digit growth | OpenFIGI (free, MIT-licensed, the structural disruption); active class-action lawsuits (Dinosaur Financial Group v. ABA et al.) allege CUSIP licensing is a $477,750/yr-per-major-firm monopoly extraction |

**Why CUSIP is worth $1.9B:** Every regulator-facing system (clearing, settlement, custody, internal books-and-records) is hardwired to CUSIP. Migrating off CUSIP to FIGI is technically trivial but operationally infeasible because back-office systems, regulator reporting (e.g., 13F holdings tables list CUSIP), and inter-firm reconciliation all require it. SENTINEL accepts CUSIP-in (parses it from EDGAR filings) but uses FIGI as its internal canonical identifier.

### 1C — S&P Capital IQ Pro ($12K–$25K/yr; 3 tiers: Essentials/Standard/Advanced)

- **Coverage:** 66,000+ public companies (49K with current financials), 60M+ private companies (16M+ with recent financials), 110K+ PE/VC funds, 19,200 active companies estimates coverage with 140+ metrics, 29M+ fixed income securities (from Markit), 49,000+ public companies ownership, 337,000+ insiders, 12,000+ activism campaigns.
- **Visible Alpha (acquired May 2024 for ~$500M; integrated into CapIQ Pro March 2025):** 200M+ data points, 1M+ consensus line items from 200+ contributing brokers across 7,300+ companies (avg 156 line items/company), 170+ industries with KPI/segment/income/balance/cashflow click-through to source models. Refreshes within 24h average.
- **Bottom-up vs top-down consensus:** Top-down (Bloomberg-style) aggregates the *output* of analyst models — EPS, revenue, EBITDA. Bottom-up (Visible Alpha) aggregates the *input* line items — units sold by SKU, ARPU, regional revenue, gross margin assumption. Bottom-up reveals where consensus expectations actually live.
- **Capital IQ Excel Plugin:** 250+ templates, including LBO models, M&A merger models, DCF, comp tables.
- **Free substitutes:** SEC 8-K parsing (Key Developments equivalent), Damodaran NYU IB templates (free DCF/LBO), EDGAR Form 4 (insider tracking), 13F (institutional ownership). Visible Alpha line-item depth has *no* free analog — it is the single most differentiated CapIQ asset.

### 1D — LSEG Workspace / Refinitiv Eikon / Datastream (~$10K–$22K/yr)

- **Datastream:** 60+ years of time series back to the 1960s, 190+ countries. Strategies that *require* this depth: long-horizon factor research (3+ business cycles for momentum/value/quality decay), academic regime studies, institutional risk-parity backtesting through 1970s stagflation. Strategies that work fine on 10 years free: most retail factor strategies, all crypto, intraday/HFT.
- **Reuters News Wire vs free:** Reuters delivers machine-readable structured news with sub-second latency; **GDELT** is the closest free analog (15-min refresh, global, NLP-tagged, free), supplemented by Finnhub news, NewsAPI, RSS.
- **LSEG ESG:** 12,000+ companies, 630+ measures. Free: CDP (climate disclosures), UN SDG, ISS, company sustainability reports parsed via LLM. Quality gap: ESG ratings are subjective; replicating LSEG's specific scoring is impossible without their methodology, but raw inputs are largely free.
- **SDC Platinum (Deals Intelligence):** Every capital markets transaction since 1970. SEC S-1/424B4/8-K filings cover most US deals free; international deals are gated.
- **Datastream CodeBook/API:** Python wrapper, exposes time series by ticker/index/macro symbol. SENTINEL substitute: yfinance + FRED + Polygon + EODHD aggregator.

### 1E — Morningstar Direct (~$17,500/yr)

- **Fund database:** 500K+ products. **Free analog:** SEC N-PORT (mutual fund holdings, monthly with 60-day lag), N-CEN (annual fund profile), SEC Form ADV (advisor holdings via IAPD). Coverage approaches Morningstar for US-domiciled funds.
- **Star ratings:** Risk-adjusted relative ranking inside category, 3/5/10-year. Replicable: bucket funds by category (use SEC N-PORT prospectus objective), compute Sharpe/Sortino/Calmar, rank, distribute 10/22.5/35/22.5/10% across 1–5 stars.
- **Economic Moat Framework:** Wide / Narrow / None across 5 sources (network effect, intangible assets, cost advantage, switching costs, efficient scale). Replicable approach: LLM extraction from 10-K MD&A "Business" section + competitive ratios (gross margin stability, ROIC vs WACC sustained spread).
- **Style Box (9-box):** Value/Blend/Growth × Small/Mid/Large. Replicable from holdings: classify each holding by P/B and market cap, weight-average into the box.
- **Brinson-Hood-Beebower attribution:** Allocation, selection, interaction effects. Open-source implementations: `brinson_attribution` (PyPI), `pa` R package, custom pandas. Already implemented in DolphinDB and several Python repos.
- **X-Ray Portfolio Overlap:** Compute by parsing N-PORT for each held fund and intersecting holdings; trivial in pandas.

### 1F — S&P PitchBook (~$25K/yr for 3 users)

- 3.5M+ companies, 4.8M+ financing events, 110K+ funds, post-money valuation modeling.
- **Free coverage:**
  - SEC Form D (private placements over Reg D, free on EDGAR) — captures most US institutional fundings ≥ $1M
  - Form ADV via IAPD (free) — RIA/fund disclosures, AUM, principals
  - Crunchbase free tier — basic profiles, recent funding (60-day lag, capped at 200 record exports/day on free)
  - UK Companies House — free, complete, financials for all UK companies
  - OpenCorporates — 140 jurisdictions, 200M+ entity records
- **Why PitchBook's data is largely irreplaceable:** Private valuations come from confidential SAFE/term-sheet leaks, fund manager surveys, FOIA responses to public-pension LP filings, and 1,800+ data analysts cleaning the data. None of these channels can be reproduced. SENTINEL accepts this as a *deliberate gap* in private markets coverage and focuses on the 95% of data that *is* free.

### 1G — AlphaSense (~$50K/yr enterprise)

AlphaSense is structurally an LLM + RAG over 10K+ licensed sources (research notes, transcripts, expert calls, news). Smart Synonyms expand `cloud revenue` → `IaaS, AWS, Azure, GCP, hosting`. **The open-source stack**: LlamaIndex + pgvector + FinBERT (sentiment) or financial embedding model (e.g., `voyage-finance-2`, `text-embedding-3-large`) + Claude or local Ollama Llama-3.1-70B. AlphaSense's *non-replicable* asset is licensed expert call transcripts (Mosaic, Tegus); free transcripts are limited to public earnings calls (parseable from Seeking Alpha, Motley Fool, FMP).

### 1H — Visible Alpha (now in CapIQ Pro)

The single most differentiated paid dataset. SENTINEL substitute: an LLM extraction pipeline over earnings call transcripts that captures management-stated KPIs (e.g., for Apple: iPhone units, ASP, Services revenue YoY, Wearables revenue) and analyst-question implied estimates. 60–70% of Visible Alpha's line-item granularity is reconstructible this way — critically, only for the largest covered companies. Industry-specific KPI templates SENTINEL ships:
- **SaaS:** ARR, NRR, GRR, magic number, CAC payback, LTV/CAC, RPO, gross margin, S&M as % rev
- **Banks:** NIM, NIE, efficiency ratio, NPL ratio, CET1, tangible book/share, ROTCE
- **Retailers:** SSS, traffic, ticket, gross margin, inventory turnover, store count
- **Pharma:** drug-by-drug revenue forecast, R&D as % rev, pipeline NPV by phase
- **Auto:** vehicles delivered, ASP, gross margin per vehicle, regulatory credit revenue

### 1I — Additional Platforms

- **Koyfin:** Free + paid; equity-focused; closest non-Bloomberg consumer-grade terminal. Strength: international coverage via SEC + Companies House + free vendors.
- **Tikr Terminal:** International equity fundamentals via licensed S&P data; ~$15/mo.
- **WRDS (Wharton Research Data Services):** The academic gold-standard. CRSP (Center for Research in Security Prices) covers NYSE/AMEX/NASDAQ daily prices/returns/volume back to 1925; survivorship-bias-free; the CRSP/Compustat Merged (CCM) link table is the canonical academic identifier bridge. Compustat: 80,000+ companies with standardized financials back to 1950 (annual) / 1962 (quarterly). IBES: analyst estimates. TAQ: tick-and-quote NYSE. SDC Platinum: M&A deals. WRDS is academia-only access; most CRSP analyses can be approximated for live trading using CRSP-equivalent free EDGAR/yfinance data, but rigorous academic factor research **requires** CRSP.
- **Calcbench / Intrinsic / sec-api.io:** XBRL-based commercial parsers. SENTINEL builds its own via SEC EDGAR `companyfacts` endpoint + `edgartools` Python library (open-source).

---

## Section 2 — Complete Data Domain Map

### 2A — Equity Market Data

| Data Type | Best Free Source | Python Library | Coverage / Depth | Quality Gap vs Paid |
|---|---|---|---|---|
| Real-time quotes | Alpaca Basic (IEX-only), Finnhub WebSocket (60 calls/min free) | `alpaca-py`, `finnhub-python`, `websockets` | US listed, 15-min delayed for full SIP, real-time IEX | Need paid for full SIP consolidated tape; Alpaca Algo Trader Plus or Polygon |
| Historical OHLCV daily | yfinance, Stooq, EODHD free | `yfinance`, `pandas-datareader` | 50+ years US, 20+ years international | yfinance is unsanctioned, periodic Yahoo throttle |
| Historical OHLCV intraday (1-min) | Alpha Vantage (5 calls/min free), Polygon free tier | `alpha_vantage`, `polygon-api-client` | 2 years on free tiers | Limited, paid for 20+ years |
| Fundamentals (as-reported) | SEC EDGAR XBRL `companyfacts` | `sec-edgar-api`, `edgartools`, `python-edgar` | All US public, 2009–present, mandatory inline-XBRL | Pre-2009 limited; international only via IFRS-full filers |
| Fundamentals (standardized) | FMP free (250 calls/day), Finnhub | `fmpsdk`, `finnhub-python` | Global, less depth than CapIQ | Standardization rules opaque |
| Analyst estimates | Yahoo (via yfinance), Finnhub estimate, FMP | `yfinance.Ticker.analyst_price_targets` | 5–10 brokers vs 200–900 paid | Major depth gap; no Visible-Alpha-style line items free |
| Earnings (surprise, calendar) | Finnhub earnings calendar (free), FMP | `finnhub.earnings_calendar()` | Full US, international limited | Adequate for retail screening |
| Dividends, corporate actions | yfinance, FMP, polygon corporate actions | `yfinance`, `polygon` | Adequate for US | Pre-1990 international gaps |
| Short interest | FINRA bi-monthly file (free) | `requests` against FINRA short interest CSV | All NMS securities, 15-day publication lag | Real-time short interest not free |
| Insider transactions (Form 4) | SEC EDGAR | `edgartools` `Form4`, `sec-edgar-downloader` | Real-time within 2 business days | Complete |
| 13F institutional holdings | SEC EDGAR | `edgartools` `ThirteenF`, custom parser | $100M+ AUM filers, quarterly with 45-day lag | Complete |
| 13D/13G activist | SEC EDGAR | `edgartools` | Triggers at 5% beneficial ownership | Complete |
| Proxy (DEF 14A) | SEC EDGAR | `edgartools` | Executive comp, vote items | Complete |
| IPO / S-1 | SEC EDGAR | `edgartools` | Real-time | Complete |

### 2B — Fixed Income

| Data Type | Best Free Source | Notes |
|---|---|---|
| US Treasuries (yields, prices) | FRED + Treasury FiscalData | DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS5, DGS10, DGS20, DGS30 |
| Corporate bonds | FINRA TRACE (15-min delayed transactions, free via FINRA Market Data Center) | All TRACE-eligible debt; lacks live bid/ask |
| Municipal bonds | MSRB EMMA (free, real-time trade prices on 1M+ outstanding) | Most liquid muni source |
| High yield / leveraged loans | Limited; LCD (CapIQ) has the canonical data | TRACE covers 144A but loans are gated |
| ABS/MBS/CLO | EDGAR ABS Reporter (Reg AB II Form ABS-EE) | CPR, WAC, WAM exposed; CLO data more limited |
| Yield curves | FRED for 30+ countries (DGS series, ECB, BoE, BoJ) | Free, daily |
| Credit ratings | Free history limited; current rating on EMMA for munis; corp ratings gated | Major gap |
| Bond analytics | **QuantLib Python** | Dirty/clean price, YTM, modified duration, convexity, DV01 (`BondFunctions.bps`), OAS, z-spread, callable/putable bonds via Hull-White short-rate model |

**QuantLib bond pricing pattern** (canonical):
```python
import QuantLib as ql
calc_date = ql.Date(7, 5, 2026)
ql.Settings.instance().evaluationDate = calc_date
schedule = ql.Schedule(issue_date, maturity, ql.Period(ql.Semiannual),
                       ql.UnitedStates(ql.UnitedStates.GovernmentBond),
                       ql.ModifiedFollowing, ql.ModifiedFollowing,
                       ql.DateGeneration.Backward, False)
bond = ql.FixedRateBond(2, 100., schedule, [coupon],
                        ql.ActualActual(ql.ActualActual.Bond))
yield_curve = ql.YieldTermStructureHandle(ql.FlatForward(...))
bond.setPricingEngine(ql.DiscountingBondEngine(yield_curve))
ql.BondFunctions.duration(bond, rate)   # modified duration
ql.BondFunctions.convexity(bond, rate)
ql.BondFunctions.bps(bond, rate)        # DV01 / basis-point value
```

### 2C — FX

- ECB daily rates, Fed H.10 (free, daily). Real-time spot via ccxt FX-pairs on crypto exchanges (BTC/EUR proxy) or OANDA free tier (limited).
- Forwards: construct from interest-rate differentials (covered interest rate parity); free.
- FX volatility surfaces: gated. Some retail vol surfaces from CME via free delayed data.

### 2D — Commodities

- Exchange data: CME free 10-min delayed, ICE free tier, LME free tier (limited).
- EIA (energy), USDA (agriculture), USGS (metals) — full free coverage.
- Futures term structure: derivable from CME front-month chain.

### 2E — Options

- Polygon free tier, CBOE free data, yfinance options chain (delayed, sometimes broken).
- Greeks: `py_vollib`, `mibian`, `QuantLib`. SENTINEL standardizes on QuantLib + py_vollib.
- IV surface: VIX from FRED (VIXCLS), VIX9D, VIX3M, VIX6M, VVIX free; full options IV surface requires Polygon paid.

### 2F — Crypto

- **CCXT** (MIT) covers 108+ exchanges including Binance, Coinbase, Kraken, BitMEX, Bybit, OKX, Hyperliquid, Bitget, Bitfinex with unified API: `fetch_ohlcv`, `fetch_order_book`, `create_order`, `fetch_balance`. CCXT.Pro adds WebSocket.
- DefiLlama API: TVL, protocols, chains, yields. Free, no key.
- The Graph: subgraph queries for any DeFi protocol.
- On-chain: Etherscan API (free), free RPC endpoints (Alchemy/Infura free tiers).
- Key on-chain metrics: MVRV, NVT, SOPR, exchange in/outflows, active addresses (CoinMetrics community data, Glassnode free tier).

### 2G — Macro

- **FRED** (St. Louis Fed) is the spine: 765,000+ series, free with key (120 req/min limit). `fredapi`, `fedfred` Python libraries. Native pandas DataFrames; ALFRED for vintage/point-in-time.
- BLS API: 500/day unregistered, 2,000/day registered.
- BEA API: GDP, NIPA tables.
- Economic calendar: Finnhub free, TradingView (scraped), Forex Factory.
- Yield curve analytics: T10Y2Y (2s10s spread), T10Y3M (10Y-3M spread), T5YIE (5-year breakeven inflation) — all on FRED.

### 2H — ESG & Alternative

- CDP (Climate Disclosure Project), UN SDG, company sustainability PDFs → LLM extraction pipeline.
- GDELT, Reddit API, Google Trends (`pytrends`), StockTwits sentiment, Alpha Vantage news sentiment.
- AIS ship tracking (free via AIS aggregators), free satellite (Planet Labs research tier), job postings via LinkedIn/Indeed scraping.

### 2I — SEC Filing Universe (Complete Specification)

**EDGAR API ecosystem:**
- `data.sec.gov/api/xbrl/companyfacts/CIK##########.json` — every XBRL-tagged fact
- `data.sec.gov/api/xbrl/companyconcept/CIK##########/us-gaap/{TAG}.json` — single concept history
- `data.sec.gov/api/xbrl/frames/us-gaap/{TAG}/USD/CY{YEAR}Q{Q}I.json` — cross-section
- `data.sec.gov/submissions/CIK##########.json` — recent filings metadata
- `efts.sec.gov/LATEST/search-index?q=...` — full-text search across all post-2001 filings
- `www.sec.gov/files/company_tickers.json` — ticker→CIK bulk file
- **Rate limit: 10 req/sec global. Mandatory User-Agent: `Name email@domain.com`. No API key.**
- **Bulk:** `companyfacts.zip` (~1.5GB) — every fact for every filer, single download, refreshed nightly.

**Filing taxonomy:**

| Form | Purpose | Critical Fields | Parsing Difficulty |
|---|---|---|---|
| 10-K | Annual report | Income/balance/cashflow XBRL, MD&A, Risk Factors, segment notes | Medium (XBRL structured + narrative HTML) |
| 10-Q | Quarterly | Same as 10-K, less narrative | Medium |
| 8-K | Material event | Items 1.01–9.01, 8-K item 2.02 = earnings | Easy with `edgartools.EightK` |
| DEF 14A | Proxy | Exec comp tables, board, vote items | Hard (exec comp table varies; LLM-assisted) |
| S-1 | IPO registration | Use of proceeds, risk factors, financials | Hard (long, narrative-heavy) |
| 424B4 | Final prospectus | IPO price, shares, underwriters | Medium |
| Form 4 | Insider transactions | Transaction code (P/S/A/M/G/F), shares, price, post-tx holdings, 10b5-1 checkbox (mandatory after April 1, 2023 per SEC Release 33-11138) | Easy with `edgartools.Form4` |
| 13F-HR | Institutional holdings | All ≥$200K positions, CUSIP, share count, value, voting authority | Easy with `edgartools.ThirteenF` |
| SC 13D | Active 5%+ stake | Beneficial ownership, intent, contracts | Medium |
| SC 13G | Passive 5%+ stake | Beneficial ownership | Medium |
| Form ADV | Investment adviser | AUM, clients, principals | Medium (XML) |
| Form PF | Private fund | Fund-level risk metrics | Hard, partially confidential |
| N-PORT | Mutual fund holdings monthly | Full holdings (60-day lag for retail filing) | Easy (XBRL) |
| N-CEN | Annual fund census | Fund metadata | Easy |
| SD | Conflict minerals | ESG-adjacent | Easy |
| Form D | Reg D private placement | Issuer, offering size, fund managers | Easy |

---

## Section 3 — Backtesting Framework Evaluation

### Framework Comparison

| Framework | Engine | Asset Classes | Live Trading | 10y/500-symbol bench | Corp Actions | License | Status | Best At | Worst Issue |
|---|---|---|---|---|---|---|---|---|---|
| **NautilusTrader** | Event-driven, Rust core, Python API | Equities, futures, FX, crypto, perpetuals | Yes (IB, Binance, Coinbase, Bybit, BitMEX, Kraken, Deribit, Databento, Hyperliquid, dYdX, OKX, Polymarket, Betfair) | ~30s–2m | Manual via instrument provider | LGPL v3 | Production-grade, active 2026 | Backtest↔live parity, order book modeling, Rust speed | Steep learning curve; LGPLv3 contagion concerns |
| **QuantConnect LEAN** | Event-driven, C# core | Multi-asset | Yes via QC cloud | Medium | Yes built-in | Apache 2.0 | Active | Cloud research notebooks, large data lib | Cloud lock-in; local self-host possible but onerous |
| **VectorBT (open)** | Vectorized NumPy/Numba | Any tabular | No | <5s for 10y/500 | Manual | Apache 2.0 | Active | Massive parameter sweeps, robustness | No realistic execution semantics |
| **VectorBT PRO** | Vectorized | Any tabular | No | <5s | Manual | Commercial | Active | Same + more features | Paid |
| **Backtesting.py** | Event-driven simple | Single-asset bar | No | ~10s | Manual | AGPL v3 | Active | Fast prototyping, reports | Single-asset only |
| **Zipline-Reloaded** | Event-driven, daily | US equities | No | ~60s | Built-in | Apache 2.0 | Maintained | Pipeline API for cross-sectional factor research | Daily-only; no intraday |
| **bt** | Portfolio-level | Multi-asset weights | No | Fast | Manual | MIT | Maintained | Allocation strategies | No order-level fidelity |
| **PyBroker** | ML-first event-driven | Equities | Limited | Medium | Some | Apache 2.0 | Active | Walk-forward + ML wrapper | Less mature ecosystem |
| **pysystemtrade** | Event-driven futures | Futures | IB | Slow | Yes | GPLv3 | Active (R. Carver) | Systematic futures, Carver's framework | Futures-focused |
| **QSTrader** | Event-driven portfolio | Equities | No | Medium | Some | MIT | Maintained | Portfolio-level event-driven | Smaller community |
| **Qlib (Microsoft)** | AI-first | Equities (CN+US) | Limited | Fast | Some | MIT | Very active 2025–26; new RD-Agent for autonomous research | LLM-driven factor R&D, alpha-seeking | Originally CN-market focused; non-Chinese coverage requires more work |
| **FinRL** | RL | Multi | No | Medium | Manual | MIT | Active | Reinforcement learning research | Research-only |
| **hftbacktest** | Tick/L2 HFT | Crypto, futures | No | Special purpose | Yes | MIT | Active | HFT queue-position simulation | Niche |
| **Freqtrade** | Crypto bot | Crypto | Yes (CCXT) | Fast | N/A | GPLv3 | Active | Crypto live bot | Crypto-only |
| **Jesse** | Crypto | Crypto | Yes | Fast | N/A | MIT | Active | Crypto strategies | Crypto-only |
| **Hummingbot** | Market-making | Crypto, DEX | Yes | n/a | N/A | Apache 2.0 | Very active | Market-making/arb | MM-specific |
| **vnpy** | China market | China stocks/futures | Yes | Medium | Yes | MIT | Active | China market | Chinese-market focus |
| **Blankly / pfund / QuantTradingOS** | Multi-asset frameworks | Multi | Some | Varies | Manual | Various | Smaller | Modular agent-based | Less mature |

### SENTINEL Decision Matrix

| Use case | Engine | Rationale |
|---|---|---|
| Daily/weekly strategy research | **VectorBT (open)** | 1000× faster parameter sweeps; vectorized Sharpe-Sortino-Calmar across thousands of variants in seconds |
| Intraday strategy research | **NautilusTrader backtest** | Bar/tick fidelity, slippage models |
| Production live trading | **NautilusTrader live** | Same code path as backtest = no implementation gap |
| Crypto trading | **NautilusTrader** with Binance/Coinbase/Bybit adapters; **CCXT** as data fallback | Already has 10+ crypto venue integrations in production |
| Mass parameter optimization | **VectorBT** + custom optuna wrapper | NumPy/Numba beats event-driven by 100–1000× |
| AI factor research | **Qlib + RD-Agent** | Microsoft's RD-Agent automates factor mining; Apache 2.0; non-blocking integration via SDS |

### Backtesting Methodology — The 5 Deadly Sins

| Sin | Concrete Example | Detection | SENTINEL Solution |
|---|---|---|---|
| 1. Look-ahead bias | Using EOD close to enter at open | Inspect signal timestamps vs trade timestamps | NautilusTrader's strict event ordering; VectorBT's `entries.shift(1)` enforced |
| 2. Survivorship bias | Backtesting today's S&P 500 over 20 years | Compare hits-to-misses against historical index constituents | SENTINEL maintains historical SP500/Russell membership snapshots from FRED + iShares ETF holdings history |
| 3. Overfitting / data snooping | 5,000 parameter combos, picking the best | Out-of-sample degradation, **Deflated Sharpe Ratio** | DSR mandatory before any strategy promotion; Probability of Backtest Overfitting (PBO) per Bailey/Borwein/Lopez de Prado/Zhu (2014) |
| 4. Transaction cost underestimation | Assuming free fills | Compare modeled to realized slippage | NautilusTrader fill model; explicit commission + slippage per venue |
| 5. Strategy decay / regime change | Strategy stops working post-2020 | Rolling Sharpe + regime detector | HMM regime detector + auto-pause if rolling 3-month Sharpe < 0.3× full-sample Sharpe |

**Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014, JPM 40(5):94-107):**
```
DSR = Φ((SR̂ − E[max SR]) × √(T-1) / √(1 - γ₃·SR̂ + (γ₄-1)/4·SR̂²))
```
where γ₃ is skewness, γ₄ is kurtosis, T is sample length, and E[max SR] is the expected max of N independent Sharpe ratios under H₀.

**Other guardrails:**
- White's Reality Check (2000) — bootstrap-based test that the best strategy from a set is not just lucky
- Hansen's Superior Predictive Ability (SPA) test — refines White by handling poor models
- Walk-forward (anchored: train [t₀,t]; rolling: train [t-W, t]) — SENTINEL uses anchored for parameter calibration, rolling for stability checks

### 24-Metric Performance Suite (SBE)

| # | Metric | Formula |
|---|---|---|
| 1 | Total Return | ∏(1+rᵢ) − 1 |
| 2 | CAGR / Annualized | (1+TR)^(252/N) − 1 |
| 3 | Sharpe | (mean(r)−rf)/std(r) × √252 |
| 4 | Sortino | (mean(r)−rf)/std(r⁻) × √252 |
| 5 | Calmar | CAGR/MaxDD |
| 6 | Max Drawdown | min(equity/cummax(equity) − 1) |
| 7 | MaxDD Duration | longest peak-to-recovery in days |
| 8 | Win Rate | #{r>0}/#{r≠0} |
| 9 | Profit Factor | Σr⁺/|Σr⁻| |
| 10 | Avg Win | mean(r⁺) |
| 11 | Avg Loss | mean(r⁻) |
| 12 | Kelly Fraction | (W/B − (1−W)) where B = avg|win|/avg|loss| |
| 13 | Beta | cov(r, rₘ)/var(rₘ) |
| 14 | Alpha | rᵢ − [rf + β(rₘ−rf)] |
| 15 | Information Ratio | (rₚ−rᵦ)/σ(rₚ−rᵦ) × √252 |
| 16 | Tracking Error | std(rₚ−rᵦ) × √252 |
| 17 | VaR 95% | percentile(r, 5) |
| 18 | CVaR 95% | mean(r | r ≤ VaR) |
| 19 | Skewness | E[(r−μ)³]/σ³ |
| 20 | Kurtosis | E[(r−μ)⁴]/σ⁴ |
| 21 | Consec Wins/Losses | longest run of same sign |
| 22 | Avg Trade Duration | mean(close_ts − open_ts) |
| 23 | Annualized Turnover | Σ|trade_value|/avg_capital × 252/N |
| 24 | Deflated Sharpe Ratio | per Bailey & Lopez de Prado 2014 |

**Best library:** QuantStats covers 17/24 directly; empyrical-reloaded covers 12/24 (Sharpe, Sortino, Calmar, alpha/beta, VaR, CVaR, etc.); pyfolio-reloaded covers tearsheets. **SENTINEL combines empyrical-reloaded + QuantStats + custom DSR.** PyPortfolioOpt for optimization, Riskfolio-Lib for 24+ convex risk measures.

### Strategy Taxonomy (representative)

| Family | Defining Logic | Data | Holding | Asset | Alpha Decay | Key Papers |
|---|---|---|---|---|---|---|
| Value (HML) | Buy low P/B, sell high P/B | Fundamentals | Months | Equities | Decayed since 2010 | Fama-French (1992, 1993) |
| Momentum (UMD) | Buy past winners, sell past losers | Prices | 3–12mo | Multi-asset | Crash risk | Jegadeesh-Titman (1993), Carhart (1997) |
| Quality (QMJ) | High ROE/ROIC, low leverage | Fundamentals | Months | Equities | Robust | Asness-Frazzini-Pedersen (2019) |
| Low-Vol | Low historical volatility | Prices | Months | Equities | Robust | Ang-Hodrick-Xing-Zhang (2006) |
| Size (SMB) | Small minus big | Mkt cap | Months | Equities | Largely decayed | Fama-French (1992) |
| Fama-French 5 | Mkt + SMB + HML + RMW + CMA | Fundamentals | Months | Equities | Robust | Fama-French (2015) |
| Carhart 4 | FF3 + Mom | Mixed | Months | Equities | Robust | Carhart (1997) |
| MA Crossover | Fast crosses slow | Prices | Days–weeks | Any | Heavily traded | Brock-Lakonishok-LeBaron (1992) |
| RSI/MACD | Oscillator-based | Prices | Days | Any | Decayed | n/a |
| Pairs Trading | Cointegrated pair Z-score | Prices | Days | Equities | Decayed since 2003 | Gatev-Goetzmann-Rouwenhorst (2006) |
| Earnings drift (PEAD) | Buy positive surprises | Fundamentals + prices | Weeks | Equities | Robust | Bernard-Thomas (1989) |
| Insider following | Code-P open-market buys, cluster filter | Form 4 | Weeks–months | Equities | Robust for opportunistic only (Cohen-Malloy-Pomorski 2012); ~5% abnormal return / yr; opportunistic vs routine separation triples raw signal |
| 13F best ideas | Top conviction holdings of FEHF managers | 13F | Quarters | Equities | Modestly robust | Cohen-Polk-Silli (2010); Angelini-Iqbal-Jivraj (2019): 3.80% annual outperformance, Sharpe 0.75 |
| Lazy Prices | Buy non-changers, short changers in 10-K language | 10-K text | 6–12mo | Equities | Active anomaly | Cohen-Malloy-Nguyen (2020) — up to 188 bps/mo alpha |
| Risk Parity | Inverse-vol weighting | Returns + cov | Months | Multi | Robust | Bridgewater All-Weather |
| Trend Following (CTAs) | TSMOM | Prices | Months | Multi | Robust 100yr | Moskowitz-Ooi-Pedersen (2012) |
| Carry | Long high-yield, short low-yield | Yields, FX | Months | FX, fixed income | Carry crashes | Koijen-Moskowitz-Pedersen-Vrugt (2018) |
| ML / RL | Feature → ŷ classifier; agent maximizes risk-adj return | Anything | Any | Any | Overfitting risk | López de Prado, Advances in Financial ML (2018) |

---

## Section 4 — SENTINEL Module PRDs

### Module 1: SDS — SENTINEL Data Spine

**Purpose:** Universal ingestion + normalization. One door for all data.
**Inputs:** Free vendor APIs (yfinance, Finnhub, Alpha Vantage, FMP, FRED, CCXT, Polygon free, EODHD free, Treasury FiscalData, SEC EDGAR).
**Outputs:** Normalized Pydantic events on the SENTINEL bus.
**Storage tier:**
- **Redis** — real-time L1/L2/quotes cache (TTL 5–60s), pub/sub for live events
- **TimescaleDB** (PostgreSQL extension) — operational time-series (hypertables on (symbol, ts), Hypercore hybrid row→column compression for older chunks)
- **Parquet + DuckDB** — research archive, partitioned by date; queried in-process from notebooks
- **PostgreSQL + pgvector** — SEC filing documents + embeddings for SIL

**Unified event taxonomy (Pydantic v2):**
```python
class Bar(BaseModel): figi: str; ts: datetime; o,h,l,c: Decimal; v: Decimal; tf: Literal["1m","5m","1h","1d"]
class Tick(BaseModel): figi: str; ts: datetime; price: Decimal; size: Decimal; agg: Literal["B","S","U"]
class L2Update(BaseModel): figi: str; ts: datetime; side: str; level: int; price: Decimal; size: Decimal; action: str
class NewsItem(BaseModel): id: str; ts: datetime; src: str; headline: str; body: str; figis: list[str]; sentiment: float|None
class CorporateAction(BaseModel): figi: str; ex_date: date; type: Literal["DIV","SPLIT","SPINOFF","MERGER","RIGHTS"]; ratio: Decimal|None; cash: Decimal|None
class EconRelease(BaseModel): series: str; ts: datetime; value: float; vintage: date|None
class FilingAlert(BaseModel): cik: str; accession: str; form: str; filed_at: datetime
class Signal(BaseModel): strat_id: str; ts: datetime; figi: str; side: Literal["LONG","SHORT","FLAT"]; size: Decimal; reason: str
class Order(BaseModel): client_oid: str; figi: str; side: str; qty: Decimal; type: str; tif: str; px: Decimal|None
class Fill(BaseModel): client_oid: str; ts: datetime; price: Decimal; qty: Decimal; commission: Decimal
class PortfolioState(BaseModel): ts: datetime; positions: dict[str, Decimal]; nav: Decimal; pnl_1d: Decimal
```

### Module 2: SIM — SENTINEL Instrument Master

OpenFIGI as the canonical bridge. PostgreSQL schema:
```sql
CREATE TABLE instrument (
  figi TEXT PRIMARY KEY, composite_figi TEXT, share_class_figi TEXT,
  ticker TEXT, exch_code TEXT, mic TEXT, name TEXT,
  asset_class TEXT, sec_type TEXT, sec_type2 TEXT, currency TEXT,
  cusip TEXT, isin TEXT, sedol TEXT, lei TEXT, cik TEXT,
  active BOOL, listing_date DATE, delisting_date DATE);
CREATE TABLE instrument_alias (figi TEXT, source TEXT, value TEXT, valid_from DATE, valid_to DATE);
CREATE TABLE instrument_event (figi TEXT, ts TIMESTAMPTZ, type TEXT, payload JSONB);
```
Lifecycle events: IPO, delisting, ticker change, merger, split, spinoff, name change. SENTINEL writes a `delisted_in_2017_then_traded_today` test fixture to ensure no future-info leak into backtests.

### Module 3: SFE — SENTINEL Filing Engine

EDGAR parser for 14 filing types via `edgartools` + custom XBRL extractor. Endpoints used:
- `companyfacts` for financials by tag
- `companyconcept` for single-tag time series
- `frames` for cross-section
- `submissions` for filing index
- EFTS at `efts.sec.gov/LATEST/search-index` for full-text
- Bulk `companyfacts.zip` nightly for backfill

Point-in-time discipline: every fact stamped with `(filed_at, fiscal_period_end, accession)`; queries filter `WHERE filed_at <= as_of`.

### Module 4: SOD — SENTINEL Ownership Database

- 13F-HR → `(filer_cik, period_of_report, cusip, value, shares, voting_authority, change_qoq)` time series
- Form 4 → `(insider_name, role, code, shares, price, post_holdings, plan_10b5_1)`; opportunistic flag (deviation from rolling 12-mo pattern, per Cohen-Malloy-Pomorski 2012)
- 13D / 13G → activist tracker with intent and filing-trigger event
- Auto-signal generation: cluster Form 4 buys (3+ insiders in 30 days, code P, non-10b5-1 = strong); QoQ-13F-conviction-and-consensus (per Angelini-Iqbal-Jivraj 2019)

### Module 5: SBE — SENTINEL Backtesting Engine

```python
class SentinelStrategy:
    def on_start(self, ctx: Context) -> None: ...
    def on_bar(self, bar: Bar, ctx: Context) -> list[Signal]: ...
    def on_filing(self, f: FilingAlert, ctx: Context) -> list[Signal]: ...
    def on_news(self, n: NewsItem, ctx: Context) -> list[Signal]: ...
    def on_fill(self, fill: Fill, ctx: Context) -> None: ...
```
Two backends:
- **Research backend**: VectorBT — vectorized
- **Production backend**: NautilusTrader — same SentinelStrategy adapter calls into Strategy class on NT bus

Corporate actions pipeline auto-applied via SIM events. 24-metric report. Anti-overfitting: DSR + walk-forward + OOS holdout enforced via `backtest_ok = (oos_sharpe > 0.5 * is_sharpe) and (DSR > 0)`.

### Module 6: SSE — SENTINEL Screener Engine

80+ criteria across:
- **Fundamental** (P/E, P/B, EV/EBITDA, ROIC, FCF yield, gross margin, NIM for banks, ARR for SaaS …)
- **Technical** (52w high/low, RSI, MACD, MA crossover, ATR%, breakout)
- **Alternative** (insider buy cluster, 13F new position, QoQ ownership change > X%, Lazy-Prices change score)
- **Fixed income** (yield, duration, OAS, rating)
- **Crypto** (TVL change, funding rate, MVRV, NVT)

TradingView Screener Python library integration for retail-style screens. Custom factor SDK: `@sentinel.factor` decorator registers a function as a column. Persisted screens emit alerts on threshold-cross.

### Module 7: STU — SENTINEL Terminal UI

10-panel workspace: (1) Watchlist, (2) Chart, (3) DES card, (4) News stream, (5) Filings stream, (6) Screener, (7) Portfolio, (8) Strategy monitor, (9) Macro/yield curve, (10) Command bar.

Command bar — Bloomberg-inspired: `AAPL <Equity> DES <GO>` style. Implemented as fuzzy-matched function dispatcher.

**Frontend decision (per ADR-006):** Streamlit + `streamlit-lightweight-charts-pro` for v0–v1; transition to Next.js + TradingView Lightweight Charts (45KB Apache 2.0) + WebSocket from FastAPI for v2+ once panel count exceeds Streamlit reactive limits.

### Module 8: SEE — SENTINEL Execution Engine

- Paper sim (NautilusTrader BacktestNode in live-replay mode)
- Live adapters: **Alpaca** (US equities, free paper trading API, real-time IEX), **Interactive Brokers** (multi-asset, via NT's `ibapi` adapter), **Binance** (crypto via NT)
- Order types: Market, Limit, Stop, Stop-Limit, OCO, Bracket, Trailing
- **Fail-Closed safety chain**: order pre-trade → RiskEngine (notional, leverage, daily-loss, kill-switch) → ExecEngine → broker. Any layer can reject. Default mode: paper. Live mode requires explicit env var `SENTINEL_LIVE=1` + per-day human-in-the-loop confirmation chain.
- Kill switch: file-watch on `~/.sentinel/KILL` → flatten + halt within 1s.
- Daily loss limit: configurable per portfolio; auto-halt at threshold.

### Module 9: SPR — SENTINEL Portfolio & Risk Engine

Real-time mark-to-market against Redis last-price cache. Brinson-Hood-Beebower attribution (allocation, selection, interaction). VaR (historical, parametric, Monte Carlo), CVaR. Correlation monitor. Position sizing: Kelly fraction (capped at 0.25), volatility targeting, risk parity (via Riskfolio-Lib `Portfolio.rp_optimization()`).

Stack: **PyPortfolioOpt** (mean-variance, HRP, Black-Litterman), **Riskfolio-Lib** (24 risk measures: SD, MAD, CVaR, EVaR, CDaR, RLDaR, Tail Gini), **empyrical-reloaded**, **pyfolio-reloaded** for tearsheets.

### Module 10: SIL — SENTINEL Intelligence Layer

LlamaIndex + pgvector over the SEC filing corpus + earnings call transcripts + news archive. Embedding model: `voyage-finance-2` (paid) or `BAAI/bge-large-en-v1.5` (free, local), with fallback `sentence-transformers/all-MiniLM-L6-v2`. FinBERT (ProsusAI, MIT-style) for sentiment classification (positive/negative/neutral) of headlines and call sentences.

**RAG pipeline:**
1. Chunker: chunk 10-Ks by Item, 10-Qs by section, transcripts by speaker turn
2. Embed → pgvector with HNSW index
3. Hybrid retrieval: BM25 (Postgres `tsvector`) + vector cosine, RRF re-rank
4. Generate via Claude or local Ollama (Llama 3.1 70B Q4)

**MCP interface** — 15 tools exposed to DanteAgents:
1. `sentinel.search_filings(query, form_types, date_range)` — EFTS + RAG
2. `sentinel.get_company_facts(ticker_or_cik, concepts)` — XBRL
3. `sentinel.run_screener(criteria)` — SSE
4. `sentinel.get_ownership(ticker)` — SOD time series
5. `sentinel.get_insider_signal(ticker, cluster=True)` — Form 4 opportunistic filter
6. `sentinel.backtest(strategy_spec, universe, period)` — SBE
7. `sentinel.optimize_portfolio(holdings, objective)` — SPR
8. `sentinel.calculate_var(portfolio, confidence)` — SPR
9. `sentinel.fetch_macro(series_ids, freq)` — SMA/FRED
10. `sentinel.price_bond(spec, curve)` — SBX/QuantLib
11. `sentinel.get_news_sentiment(ticker, lookback)` — FinBERT over recent news
12. `sentinel.regime(asset, model="hmm")` — SMA regime detector
13. `sentinel.submit_paper_order(symbol, side, qty, type, params)` — SEE paper
14. `sentinel.get_portfolio_state()` — SPR live state
15. `sentinel.kill_switch(reason)` — SEE emergency flatten

### Module 11: SMA — SENTINEL Macro Analyzer

Full FRED integration via `fedfred` (modern client) or `fredapi`. Cross-asset macro dashboard: 2s10s, 3m10y, 5Y breakeven, Fed funds, DXY, WTI, gold, BTC. **Regime detector**: Gaussian HMM via `hmmlearn` on (log-return, realized vol) of SPY → 3 states (bull/range/bear); rule-based overlay (e.g., 200dma slope, VIX regime, yield curve inversion). Output: regime label per day, confidence, transition probability.

Economic calendar: Finnhub free + scheduled FRED release calendar.

### Module 12: SBX — SENTINEL Bond Analytics

Pure QuantLib Python wrapper. Yield curve construction from FRED Treasury constant-maturity series. Bond math: dirty/clean price, YTM, modified duration, convexity, DV01, OAS (via Hull-White or Black-Karasinski for callable bonds), z-spread (via `ZeroSpreadedTermStructure` or `SpreadedLinearZeroInterpolatedTermStructure` for non-parallel shifts and key-rate durations). TRACE corporate prices ingested daily; FRED + Treasury FiscalData for sovereigns.

---

## Section 5 — Architecture Decision Records

### ADR-001: Core Engine Language
**Decision: Hybrid — Python primary with Rust acceleration via NautilusTrader.**
Pure Rust ruled out (single dev, slow iteration). Pure Python ruled out (NautilusTrader's Rust-native MessageBus, OrderBook reconstruction, and event ordering are durable advantages). NT's Cython/Rust core gives C-level speed for the hot path while strategy code stays in Python.
*Reversibility: low — strategy code is portable; infra rewrite is the cost.*

### ADR-002: Event Bus
**Decision: NautilusTrader internal MessageBus for in-process events; Redis Streams for inter-process audit trail; ZeroMQ as escape hatch.**
Kafka rejected (ops overhead). RabbitMQ rejected (overkill). Redis Pub/Sub lacks persistence; Redis Streams gives durable log. NT's MessageBus provides nanosecond-precision ordering inside the trading process.
*Reversibility: high.*

### ADR-003: Time-Series Database
**Decision: TimescaleDB (operational hot tier) + Parquet/DuckDB (research cold tier) + Redis (real-time cache).**
QuestDB faster on read benchmarks but lacks Postgres ecosystem (pgvector for SIL = killer). DuckDB+Parquet wins for offline research. TimescaleDB hypertables + Hypercore compression handle 1M bars/day on Mac mini comfortably.
*Reversibility: medium — schema migration cost.*

### ADR-004: Backtesting Strategy
**Decision: Wrap both NautilusTrader (production) and VectorBT (research) under a unified `SentinelStrategy` abstraction.**
Custom build rejected (5+ years of NT engineering). LEAN rejected (C# core; cloud lock-in incentives). Single codebase research-to-live achieved via NT's backtest↔live parity guarantee.
*Reversibility: low.*

### ADR-005: Document Intelligence
**Decision: LlamaIndex + pgvector + hybrid retrieval + Claude (primary) / Ollama Llama-3.1-70B (sovereign fallback).**
LangChain rejected (over-abstracted, churn). Haystack/Elasticsearch rejected (ops). pgvector chosen because Postgres is already in the stack (PostgreSQL document store, SIM, SOD). PostgresML option noted for future single-network-call RAG.
*Reversibility: medium.*

### ADR-006: Frontend
**Decision: Streamlit + streamlit-lightweight-charts-pro for Generations 0–3; Next.js + TradingView Lightweight Charts + FastAPI WebSockets for Generation 4+.**
Electron rejected (multi-platform packaging cost). Pure React/Plotly possible but Streamlit's batteries-included reactivity ships Generation 1 in days. The TradingView Lightweight Charts library (Apache 2.0, 45KB) is the chart engine in either case.
*Reversibility: high — frontend is the most decoupled module.*

### ADR-007: License
**Decision: Apache 2.0 for the SENTINEL framework; LGPL v3 components (NautilusTrader) wrapped behind a stable API boundary; proprietary strategies shipped as separate non-distributed modules.**
MIT rejected (insufficient patent grant). AGPLv3 rejected (network-use copyleft incompatible with shipping a hosted variant later). BSL rejected (non-OSI). Apache 2.0 + LGPL is compatible because LGPL allows linking to non-LGPL code through the library boundary; NT's LGPLv3 obligations apply only to modifications of NT itself.
*Reversibility: low — license choice is sticky.*

### ADR-008: AI Model Integration
**Decision: DanteAgents + MCP as the routing layer. Claude via Anthropic API as primary cloud LLM; Ollama (Llama 3.1 70B Q4) as local sovereign fallback; OpenAI GPT-5 as optional alternative.**
MCP is now the de-facto agent-tool standard (Anthropic, OpenAI, Google, Microsoft, AWS, Cloudflare, Bloomberg all signatories of the Linux Foundation Agentic AI Foundation as of late 2025). Single-vendor lock-in rejected.
*Reversibility: high — MCP is model-agnostic.*

### ADR-009: Deployment
**Decision: Process-per-module via Docker Compose with shared TimescaleDB/Redis/Postgres-pgvector services. Single Mac mini host. No Kubernetes.**
Monolith rejected (live trading process must not stop for filing parser bug). K8s rejected (single-node overkill).
*Reversibility: medium — Compose → Nomad/K8s migration is straightforward.*

---

## Section 6 — SENTINEL Data Bill of Materials (Sovereign Free Stack)

| Domain | Primary Free Source | Library | Rate Limit | Depth | Quality vs Paid |
|---|---|---|---|---|---|
| US equity quotes (RT) | Alpaca Basic + Finnhub | `alpaca-py`, `finnhub-python` | 60/min Finnhub | IEX-only RT | Need Algo Trader Plus for full SIP |
| US equity OHLCV daily | yfinance + Stooq + EODHD free | `yfinance` | None official | 50+ yrs | Adequate |
| US equity OHLCV 1-min | Alpha Vantage + Polygon free | `alpha_vantage`, `polygon` | 5/min AV | 2 yrs | Limited |
| Intl equity OHLCV | yfinance + EODHD | `yfinance`, `eodhd` | — | 20+ yrs | Adequate |
| US fundamentals | SEC EDGAR XBRL | `sec-edgar-api`, `edgartools` | 10 req/s | Since 2009 | Equal-to-better than Compustat for as-reported |
| Analyst estimates | Yahoo + Finnhub + FMP | `yfinance`, `finnhub` | varies | Shallow | Major gap vs FactSet 900-broker |
| Insider Form 4 | EDGAR | `edgartools` | 10 req/s | Real-time | Complete |
| 13F holdings | EDGAR | `edgartools.ThirteenF` | 10 req/s | Q with 45-day lag | Complete |
| US Treasuries | FRED + Treasury FiscalData | `fredapi`, `fedfred` | 120/min | Decades | Complete |
| Corporate bond prices | FINRA TRACE | `requests` against FINRA | n/a | 15-min delayed | No live bid/ask |
| Credit ratings | Limited free; EMMA for muni | — | — | Spotty | Gap |
| FX spot | ECB daily, Fed H.10 | `fredapi` | — | Decades | Adequate |
| Commodity spot | EIA, USDA, USGS | `eia-python` | — | Decades | Adequate |
| Commodity futures | CME free 10-min delayed | scraping | — | Adequate | Limited |
| Crypto OHLCV | CCXT (108 exchanges) | `ccxt`, `ccxt.pro` (paid) | per-exchange | All | Equal-to-better |
| Crypto on-chain | DefiLlama + Etherscan + The Graph | `requests`, `subgrounds` | — | Live | Adequate |
| Macro | FRED + BLS + BEA + IMF SDDS+ | `fredapi`, `bls-api` | 120/min FRED | Decades | Equal-to-better than LSEG for major series |
| News + sentiment | GDELT + Finnhub + RSS + FinBERT | `gdelt-doc`, `finnhub` | varies | Live | Gap on Reuters wire |
| SEC filings (all types) | EDGAR | `edgartools` | 10 req/s | Since 1994 | Complete |
| ESG | CDP + UN SDG + LLM extraction | `requests`, custom | — | Limited | Gap |
| US options | Polygon free + CBOE + yfinance chain | `polygon`, `yfinance` | varies | Daily snapshot | Gap on full vol surface |
| ETF holdings | iShares/SSGA/Vanguard issuer files (CSV) + N-PORT | `requests` | — | Daily–monthly | Complete for major issuers |
| Mutual fund holdings | N-PORT | `edgartools` | 10 req/s | Monthly with 60-day lag | Equal to Morningstar for US |
| Private company basic | Form D + Crunchbase free + Companies House | `requests` | varies | Limited | Major gap vs PitchBook |

---

## Section 7 — Council of Minds: 20 Key Questions

**1. Hardest Bloomberg function to replace?** Instant Bloomberg (IB) Chat — a network effect with ~325K professionals where OTC bond and FX trades are negotiated. Network goods cannot be open-sourced; you'd need to migrate the network, not rebuild the technology. SENTINEL accepts this as an explicit non-goal.

**2. Why is CUSIP worth $1.9B? What breaks without it?** CUSIPs are embedded in clearing (DTCC), settlement (FICC), 13F filings (SEC mandates CUSIP), back-office reconciliation, and credit-rating workflows. Without CUSIP normalization, SENTINEL cannot correctly join 13F holdings to live prices when ticker history is fragmented (e.g., META prior to 2022 was FB; CUSIP 30303M102 unifies them). SENTINEL solves this by accepting CUSIP-in via OpenFIGI mapping then storing FIGI as canonical.

**3. Strategies needing 50+ years of data?** Long-horizon factor decay studies, multi-cycle risk parity calibration, secular trend research (1970s stagflation, 1990s Japan deflation, 2008 deleveraging). Most retail strategies (factor combos, momentum, mean reversion, crypto, intraday) work fine with 10–20 years.

**4. Information content of sell-side report in 2026 with LLMs?** Research notes are now mostly redundant for *retrospective summary* (LLMs replicate this in seconds from earnings transcripts). They retain value for: (a) channel checks / primary research, (b) industry expert long-form, (c) highly differentiated proprietary models (Visible Alpha line items), (d) buy-side relationship maintenance. The textual content alone has maybe 20% the prior value.

**5. PitchBook at $25K/yr despite unreliable private data?** Because (a) private valuations are *always* unreliable but PitchBook is the *industry-standard unreliable*, (b) the curated investor relationships and contact data are not free anywhere, (c) the data ops team of 1,800+ is irreproducible. SENTINEL's stance: skip private-co competitive parity, focus on public markets where free data is at parity.

**6. How to integrate OpenBB with NautilusTrader?** OpenBB's Open Data Platform (ODP) exposes Python providers; NT consumes Bar/QuoteTick objects via custom data clients. Integration path: implement a NautilusTrader `LiveDataClient` that wraps OpenBB providers and emits NT-native types. What breaks: (a) OpenBB's polling cadences (1–60s) don't suit NT's microsecond bus; (b) OpenBB returns vendor-specific schemas requiring per-provider FIGI mapping; (c) AGPL v3 of OpenBB is *more* viral than NT's LGPL, requiring isolation as a sidecar process.

**7. Single most important data quality control before trusting a backtest?** **Point-in-time integrity.** Specifically: (a) fundamentals filter on `filing_date <= as_of`, never on fiscal-period-end; (b) index membership uses historical SP500/Russell snapshots not today's; (c) ticker-to-issuer mapping uses time-bounded `instrument_alias` table. This is the difference between a 22%-CAGR Lazy Prices replication and an obvious overfit.

**8. Minimum viable infra for live trading US equities + BTC/ETH + WTI on Mac mini?** Mac mini M4 (16GB unified, 512GB SSD) + Docker Compose with: TimescaleDB, Postgres+pgvector, Redis, NautilusTrader live process, Streamlit UI, Ollama (optional). External: Alpaca (US equities, free paper, $9/mo Algo Trader Plus for full SIP), Binance (BTC/ETH spot+perps), IB or Tradovate (futures for WTI). Network: stable residential fiber + mobile failover. Cost: ~$1,500 hardware + $9/mo data.

**9. Strategy abstraction for both AI-generated and human-written?** A Pydantic-validated `StrategySpec` JSON schema (universe, signals, sizing, rules, params) plus a `SentinelStrategy` Python class that consumes it. Human writes class directly; AI generates spec which is materialized into the class via a code generator (or directly interpreted). Critical: spec is declarative enough to validate but not so abstract it loses signal-level nuance.

**10. Minimum 10 MCP tools?** From SIL list: `search_filings`, `get_company_facts`, `run_screener`, `get_ownership`, `get_insider_signal`, `backtest`, `fetch_macro`, `get_news_sentiment`, `submit_paper_order`, `get_portfolio_state`. These cover research → ideation → validation → paper-execution → monitoring without exposing live-trading directly.

**11. Best risk-adjusted strategy for $100K solo trader on free data in 2026?** **Trend-following on cross-asset futures and crypto perps with TSMOM rules and regime filtering** (Moskowitz-Ooi-Pedersen 2012 framework). Rationale: it survives the largest universe of regime changes, costs are minimal at retail scale, free data (yfinance + CCXT + Treasury futures via IB) is sufficient, doesn't require alpha extraction over crowded retail factors. Realistic Sharpe: 0.7–1.0; max DD: 20–25%. Alternative: insider-buy cluster following on small-caps (Cohen-Malloy-Pomorski filter), per recent arxiv evidence ~6.3% mean cumulative abnormal return for run-up clusters.

**12. Early warning of strategy decay?** (a) Rolling 3-month Sharpe < 0.3× full-sample; (b) Hit rate degrading vs. baseline; (c) Realized turnover diverging from backtested; (d) Slippage exceeding model assumption persistently; (e) Regime shift detected by HMM — auto-pause and require human reconfirm.

**13. 13F 45-day-lag alpha?** Cohen-Polk-Silli (2010) "Best Ideas" replicated by Angelini-Iqbal-Jivraj (2019): top conviction positions of fundamental hedge funds (HFU subset, Q-end + 47-day rebalance) outperform S&P 500 by **3.80% annually with Sharpe 0.75**. Alpha is real but small and concentrated in the right manager subset. Generic 13F-cloning underperforms.

**14. Most predictive Form 4 transaction codes?** **Code P (open-market purchase) only** is robust per Cohen-Malloy-Pomorski (2012): opportunistic non-routine purchases ≈ 4× the abnormal return of undifferentiated insider buys; ~5%/yr abnormal return. 10b5-1 plan checkbox (mandatory since April 1, 2023) explicitly identifies routine pre-arranged trades — *exclude these*. Cluster 3+ insiders within 30 days for stronger signal.

**15. Practical regime detection?** 3-state Gaussian HMM via `hmmlearn` on (log-return, 20-day realized vol) of SPY, plus rule-based overlays (200-DMA slope; 2s10s inversion; VIX > 25). HMM state probabilities are smooth, rule overlay is hard. SENTINEL combines: regime label = HMM state when confidence > 0.7, else rule fallback. ML-based (LSTM, transformer) regime detectors generally overfit on financial small-sample.

**16. Is Microsoft Qlib production-ready for non-Chinese markets?** Yes for research, with caveats. Qlib originally CN-focused but US data adapters exist. RD-Agent (released 2025) does **autonomous factor mining**: generates factor hypotheses, codes them, backtests, iterates. It does *not* replace a quant; it accelerates the search. Production live trading via Qlib alone is not recommended — it is research-grade. SENTINEL uses Qlib offline for factor R&D, then ports surviving factors into NautilusTrader for live execution.

**17. VectorBT vs NautilusTrader practical difference?** VectorBT computes signals across a price matrix in NumPy/Numba; NT processes events bar-by-bar through a strategy state machine. **It matters when:** (a) any strategy has order-fill-dependent state (stop-loss, position sizing on prior fill), (b) realistic slippage/latency modeling is required, (c) order-book interaction matters, (d) the strategy must run live. **It doesn't matter when:** running daily-rebalance long-only factor portfolios on EOD bars where signals are independent of fills.

**18. Why no MIT-licensed self-hosted QuantConnect competitor yet?** Three barriers: (a) **data licensing economics** — QC subsidizes data via cloud lock-in; pure self-hosted users would have to license data themselves, breaking the unit economics; (b) **algorithm framework maintenance** — LEAN is C# with ~10 years of edge cases; replicating it in MIT Python is a 5-engineer-year project; (c) **execution-model fidelity** — NautilusTrader is closest, but its LGPL license constrains downstream commercial wrappers. SENTINEL's bet is that NT under stable API boundary + Python-first strategy layer is the path that closes this gap.

**19. Open-source quant in 3 years?** Consolidation around NautilusTrader (execution), Qlib + RD-Agent (AI research), VectorBT (research speed), pgvector + LlamaIndex (document intelligence), MCP (agent integration). Bloomberg/FactSet/CapIQ don't lose institutional users but lose 80% of the long-tail individual professional / boutique fund market. AlphaSense-style RAG-over-licensed-docs becomes commoditized as embedding models specialize for finance.

**20. Top 3 unsolved problems SENTINEL can uniquely solve?**
1. **Truly point-in-time, AI-controllable financial dataset assembly** — every paid vendor offers PIT but no open stack does end-to-end. SENTINEL's SDS+SIM+SFE provides this.
2. **Backtest-to-live parity with AI agents in the loop** — agents that can't safely run live have limited value; SENTINEL's Fail-Closed safety chain + MCP exposure makes agentic strategies safe.
3. **Sovereign document intelligence over filings without sending data to cloud LLMs** — Ollama + local pgvector + FinBERT enables a local-first research workflow that institutions cannot publicly endorse but privately want.

---

## Section 8 — 8-Generation Build Plan

| Gen | Goal | Key Modules | Validation | Est LOC | Milestone |
|---|---|---|---|---|---|
| **0** | Foundation | SDS minimal (yfinance+Finnhub+FRED) + watchlist | "show me AAPL fundamentals + chart in <30s" | ~3K | Mac mini provisioned, Docker Compose up, 3 adapters live |
| **1** | Research Terminal | SDS full + SIM + SFE + SSE + STU | EDGAR XBRL parse 100 filings, 80-criterion screener returning correct results | ~15K | Replaces $30K/yr equity research workflow |
| **2** | Backtesting | SBE (VectorBT + NT) + 24 metrics | DSR + walk-forward operational; SPY buy-and-hold match within 2bps; 10-yr/500-symbol VectorBT < 10s | ~10K | First in-house factor strategy backtest |
| **3** | Ownership & Alt Data | SOD + signals | 13F new-position screen + Form 4 cluster signal alerts firing | ~6K | Insider-following micro-strategy live in paper |
| **4** | Fixed Income & Macro | SBX + SMA | QuantLib bond curve fit to FRED Treasuries within 1bp; HMM regime detector running daily | ~8K | Cross-asset macro dashboard live |
| **5** | Live Paper Trading | SEE paper + SPR real-time | Alpaca paper + Binance paper running 7 days no drift; full P&L within $0.01 of broker | ~10K | Multi-asset paper trading with SPR attribution |
| **6** | Intelligence Layer | SIL RAG + NL screener | "Find me high-quality SaaS at <8x ARR with insider buying" → correct results in <10s | ~12K | LLM analyst over 100K filings |
| **7** | DanteAgents Integration | MCP server + 15 tools | Full agent-driven research session executes end-to-end | ~5K | Sovereign agentic quant lab |
| **8** | Live Trading Unlock | SEE live (Alpaca, IB, Binance) + safety chain | 30-day live paper soak + 7-day live $1K stake completes without safety breach | ~6K | First sovereign live trade |

**Total estimated LOC: ~75,000** (excluding tests, docs, third-party). Single developer, ~2,500 LOC/week sustainable, **~30 weeks to Generation 8**.

---

## Conclusion: The SENTINEL Build Thesis

The institutional terminal industry was built on 1980s assumptions: data was scarce, network effects (IB Chat) were the moat, and proprietary identifiers (CUSIP) were rent-extractable. In 2026, three of those assumptions have flipped. SEC EDGAR's XBRL companyfacts is a public good as comprehensive as Compustat. OpenFIGI under MIT replaces CUSIP for any non-back-office system. NautilusTrader's Rust core delivers institutional-grade execution semantics for free. MCP standardizes agent-to-tool plumbing, making AI integration vendor-neutral. The remaining moats — IB Chat's network, Visible Alpha line items, PitchBook private data — are real but addressable as deliberate scope cuts.

SENTINEL is the integration. Twelve modules. ~75K LOC. 30 weeks. One Mac mini. Apache 2.0 framework with LGPL execution wrapped behind a stable boundary. Ten MCP tools that turn DanteAgents into a sovereign quant team. The first sovereign trade is the moment the ~$40B/yr institutional terminal industry stops being a ticket-of-entry and starts being a choice.

— *End of SENTINEL Founding Specification, v0.1*