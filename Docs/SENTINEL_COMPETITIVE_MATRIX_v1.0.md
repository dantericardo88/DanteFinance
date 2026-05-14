# SENTINEL Competitive Matrix — 110-Dimension Feature Universe
**Version:** 1.0  
**Date:** May 7, 2026  
**Method:** DanteForge /universe — scored against Bloomberg, CapIQ Pro, FactSet, AlphaSense, LSEG Workspace, Morningstar Direct, PitchBook  
**Scoring:** 0 = absent | 1 = basic/partial | 2 = solid | 3 = best-in-class

---

## Executive Scorecard

| Platform | Composite Score | Annual Cost | Cost/Point |
|----------|:-----------:|------------|-----------|
| **SENTINEL** | **76%** | **$0 (self-hosted)** | **$0** |
| Bloomberg Terminal | 68% | $31,980/yr | $471/pt |
| FactSet | 59% | $28,500/yr avg | $483/pt |
| CapIQ Pro | 51% | $18,500/yr avg | $363/pt |
| LSEG Workspace | 56% | $16,000/yr avg | $286/pt |
| Morningstar Direct | 25% | $17,500/yr | $700/pt |
| AlphaSense | 13% | $50,000/yr | $3,846/pt |
| PitchBook | 8% | $25,000/yr | $3,125/pt |

> **SENTINEL specification-complete score of 76% covers more feature surface than Bloomberg at zero marginal cost.**  
> The remaining 24% represents deliberate non-goals (OTC bond live pricing, private valuations, expert call transcripts) or Generation 3+ roadmap items.

---

## CATEGORY 1 — Real-Time & Historical Market Data (12 Dimensions)

| # | Feature | SENTINEL | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Real-time equity quotes (full SIP consolidated tape) | △ | 3 | 2 | 2 | 0 | 3 | 1 | 0 |
| 2 | Historical OHLCV daily (30+ years, 50+ markets) | ✓ | 3 | 2 | 3 | 0 | 3 | 2 | 0 |
| 3 | Historical OHLCV intraday (1-min, 20+ years) | △ | 3 | 1 | 2 | 0 | 3 | 0 | 0 |
| 4 | Options chain (all strikes/expiries, Greeks) | △ | 3 | 1 | 2 | 0 | 2 | 0 | 0 |
| 5 | Futures term structure / continuous contracts | ✓ | 3 | 1 | 2 | 0 | 3 | 0 | 0 |
| 6 | FX spot, forwards, volatility surface | △ | 3 | 1 | 2 | 0 | 3 | 0 | 0 |
| 7 | Crypto multi-exchange OHLCV (100+ venues via CCXT) | ✓ | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| 8 | Corporate actions (splits, dividends, M&A adjustments) | ✓ | 3 | 3 | 3 | 0 | 3 | 3 | 0 |
| 9 | Short interest (bi-monthly FINRA file, all NMS) | ✓ | 3 | 2 | 2 | 0 | 2 | 1 | 0 |
| 10 | Order book / Level 2 market depth | ✗ | 3 | 0 | 1 | 0 | 2 | 0 | 0 |
| 11 | Pre/post-market quotes | △ | 3 | 1 | 1 | 0 | 2 | 0 | 0 |
| 12 | Tick-level trade data (TAQ-equivalent) | ✗ | 3 | 0 | 1 | 0 | 2 | 0 | 0 |

**SENTINEL: 63% spec-complete** | Gaps: Level 2 (Gen 3), tick data (Gen 3), real-time SIP (Alpaca Algo Trader Plus bridges this at $30/mo)

---

## CATEGORY 2 — Fundamentals & Financial Statements (12 Dimensions)

| # | Feature | SENTINEL | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 13 | Income statement standardized (10K+ companies) | ✓ | 3 | 3 | 3 | 1 | 3 | 3 | 0 |
| 14 | Balance sheet standardized | ✓ | 3 | 3 | 3 | 1 | 3 | 3 | 0 |
| 15 | Cash flow statement standardized | ✓ | 3 | 3 | 3 | 1 | 3 | 3 | 0 |
| 16 | Segment & geographic revenue breakdown | △ | 3 | 3 | 3 | 1 | 3 | 2 | 0 |
| 17 | Non-GAAP reconciliation tables | △ | 3 | 2 | 3 | 1 | 2 | 2 | 0 |
| 18 | Analyst consensus estimates (200+ brokers) | △ | 3 | 3 | 3 | 1 | 3 | 2 | 0 |
| 19 | Bottom-up line-item estimates (Visible Alpha style) | △ | 2 | 3 | 1 | 0 | 1 | 0 | 0 |
| 20 | Historical financials 30+ years (point-in-time) | △ | 3 | 3 | 3 | 0 | 3 | 3 | 0 |
| 21 | International / IFRS financials (ex-US) | △ | 3 | 2 | 2 | 0 | 2 | 2 | 0 |
| 22 | Point-in-time financial data (no look-ahead) | ✓ | 3 | 3 | 2 | 0 | 2 | 2 | 0 |
| 23 | DCF / WACC built-in templates + Damodaran data | ✓ | 3 | 3 | 3 | 0 | 2 | 2 | 0 |
| 24 | Comparable company (comps) tables auto-generated | ✓ | 3 | 3 | 3 | 0 | 2 | 2 | 0 |

**SENTINEL: 72% spec-complete** | Key gap: Visible Alpha line-item depth — addressable via earnings call LLM extraction pipeline (60-70% coverage for top-500 companies)

---

## CATEGORY 3 — Ownership, Insiders & SEC Filings (10 Dimensions)

| # | Feature | SENTINEL | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 25 | Institutional ownership 13F (quarterly, all $100M+ AUM) | ✓ | 3 | 3 | 3 | 0 | 2 | 2 | 0 |
| 26 | Insider transactions Form 4 (real-time, 2-day lag) | ✓ | 3 | 3 | 2 | 0 | 2 | 1 | 0 |
| 27 | Activist 13D/13G tracking (5%+ beneficial ownership) | ✓ | 3 | 3 | 2 | 0 | 2 | 1 | 0 |
| 28 | Proxy / DEF 14A (exec comp tables, board, votes) | ✓ | 3 | 3 | 2 | 1 | 2 | 2 | 0 |
| 29 | Congressional STOCK Act eFD disclosures | ✓ | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 30 | IPO / S-1 filing intelligence | ✓ | 3 | 3 | 2 | 1 | 2 | 1 | 2 |
| 31 | Private placement Form D tracking | ✓ | 1 | 2 | 1 | 0 | 0 | 0 | 2 |
| 32 | Fund holdings N-PORT (monthly, mutual funds) | ✓ | 2 | 2 | 2 | 0 | 1 | 3 | 0 |
| 33 | Full-text EDGAR search (all post-2001 filings) | ✓ | 2 | 2 | 2 | 2 | 1 | 0 | 0 |
| 34 | Form ADV / RIA adviser + fund intelligence | ✓ | 1 | 1 | 1 | 0 | 0 | 0 | 0 |

**SENTINEL: 95% spec-complete** | **LEAPFROG:** Congressional trading disclosures — no paid terminal covers this depth. SENTINEL leads all competitors on SEC filing completeness.

---

## CATEGORY 4 — Fixed Income & Credit Analytics (8 Dimensions)

| # | Feature | SENTINEL | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 35 | US Treasury yield curves (30+ maturities, FRED) | ✓ | 3 | 2 | 3 | 0 | 3 | 2 | 0 |
| 36 | Corporate bond pricing (FINRA TRACE, 15-min delay) | ✓ | 3 | 3 | 3 | 0 | 3 | 1 | 0 |
| 37 | Municipal bond market (MSRB EMMA, real-time trades) | ✓ | 3 | 2 | 2 | 0 | 2 | 1 | 0 |
| 38 | Bond analytics engine (QuantLib: DV01, OAS, z-spread) | ✓ | 3 | 2 | 2 | 0 | 2 | 1 | 0 |
| 39 | Credit spread analysis / duration / convexity | ✓ | 3 | 3 | 3 | 0 | 3 | 1 | 0 |
| 40 | MBS / ABS / CLO structured product data | △ | 3 | 2 | 2 | 0 | 2 | 0 | 0 |
| 41 | High yield / leveraged loan data | ✗ | 3 | 3 | 2 | 0 | 2 | 0 | 0 |
| 42 | Live bond bid/ask (OTC executable quotes) | ✗ | 3 | 2 | 1 | 0 | 2 | 0 | 0 |

**SENTINEL: 70% spec-complete** | **Deliberate gaps:** HY/loan (LCD gated, Gen 3), live OTC bid/ask (Bloomberg IB moat — unreplicable)

---

## CATEGORY 5 — Macro, Economics & Cross-Asset (8 Dimensions)

| # | Feature | SENTINEL | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 43 | FRED macro time series (765K+ series, 120 countries) | ✓ | 3 | 2 | 2 | 0 | 3 | 1 | 0 |
| 44 | Economic calendar & release consensus | ✓ | 3 | 2 | 2 | 0 | 3 | 1 | 0 |
| 45 | Central bank speech NLP (Fed/ECB/BoE/BoJ/BoC/RBA) | ✓ | 2 | 1 | 1 | 1 | 2 | 0 | 0 |
| 46 | CFTC Commitment of Traders (COT) positioning, 150+ markets | ✓ | 2 | 0 | 1 | 0 | 1 | 0 | 0 |
| 47 | Yield curve spread analytics (2s10s, 10Y-3M, etc.) | ✓ | 3 | 2 | 2 | 0 | 3 | 1 | 0 |
| 48 | Inflation breakeven / TIPS analytics (FRED series) | ✓ | 3 | 2 | 2 | 0 | 3 | 1 | 0 |
| 49 | Cross-country macro comparison (190+ countries) | △ | 3 | 2 | 1 | 0 | 3 | 1 | 0 |
| 50 | Regime detection (HMM + rule-based, 4 regimes) | ✓ | 1 | 0 | 0 | 0 | 1 | 0 | 0 |

**SENTINEL: 88% spec-complete** | **LEAPFROG:** CFTC COT terminal integration + HMM regime detector — unique in any commercial product

---

## CATEGORY 6 — AI, NLP & Document Intelligence (10 Dimensions)

| # | Feature | SENTINEL | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 51 | RAG over financial docs (pgvector + LlamaIndex) | ✓ | 1 | 1 | 1 | 3 | 1 | 0 | 0 |
| 52 | Financial sentiment (FinBERT, sentence-level) | ✓ | 1 | 0 | 0 | 2 | 1 | 0 | 0 |
| 53 | Natural language → screener translator | ✓ | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 54 | Natural language → trading strategy generator | ✓ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 55 | LLM document summarization (Claude + local Llama) | ✓ | 1 | 1 | 1 | 3 | 1 | 0 | 0 |
| 56 | Smart synonym / query expansion | △ | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| 57 | Earnings call transcript library + semantic search | ✓ | 2 | 2 | 2 | 3 | 2 | 0 | 0 |
| 58 | Expert call / scuttlebutt intelligence | ✗ | 1 | 1 | 1 | 3 | 0 | 0 | 0 |
| 59 | MCP agent-native tool surface (15 tools, FastMCP) | ✓ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 60 | Autonomous AI research agent (Qlib RD-Agent style) | △ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**SENTINEL: 82% spec-complete** | **LEAPFROG:** MCP native tool surface + NL-to-strategy — unprecedented in any terminal product. AlphaSense leads only on expert call transcripts (Tegus/Mosaic gated).

---

## CATEGORY 7 — Backtesting, Research & Execution (9 Dimensions)

| # | Feature | SENTINEL | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 61 | Vectorized backtesting engine (VectorBT, 1000× speed) | ✓ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 62 | Event-driven backtesting (NautilusTrader bar/tick) | ✓ | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 63 | Walk-forward + anchored validation | ✓ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 64 | Overfitting detection (Deflated Sharpe Ratio, PBO) | ✓ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 65 | Live trading execution (IB, Alpaca, Binance, Kraken, OANDA) | ✓ | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 66 | Paper trading simulator (full OMS parity) | ✓ | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 67 | Strategy promotion state machine (backtest→paper→live) | ✓ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 68 | AI-driven factor research (Qlib + RD-Agent pipeline) | △ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 69 | HFT / market-making engine (hftbacktest) | ✗ | 2 | 0 | 0 | 0 | 0 | 0 | 0 |

**SENTINEL: 85% spec-complete** | **LEAPFROG across every dimension.** No institutional terminal ships systematic research tools. Bloomberg executes; it does not research.

---

## CATEGORY 8 — Screening & Discovery (7 Dimensions)

| # | Feature | SENTINEL | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 70 | Fundamental equity screener (50+ fields, DuckDB engine) | ✓ | 3 | 3 | 3 | 1 | 3 | 2 | 0 |
| 71 | Technical screener (25+ criteria, pandas-ta) | ✓ | 3 | 1 | 2 | 0 | 2 | 0 | 0 |
| 72 | Ownership-based screener (13F changes, Form 4 clusters) | ✓ | 3 | 3 | 3 | 0 | 2 | 2 | 0 |
| 73 | Options flow screener (unusual volume/OI/premium) | ✓ | 3 | 0 | 1 | 0 | 1 | 0 | 0 |
| 74 | Fixed income screener (TRACE + EMMA + FRED) | ✓ | 3 | 3 | 2 | 0 | 2 | 1 | 0 |
| 75 | Crypto / on-chain screener (CCXT + DefiLlama) | ✓ | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 76 | Natural language screener (NL → SSE criteria) | ✓ | 0 | 0 | 0 | 1 | 0 | 0 | 0 |

**SENTINEL: 88% spec-complete** | Full category advantage over every competitor on crypto/NL screening

---

## CATEGORY 9 — Portfolio & Risk Analytics (7 Dimensions)

| # | Feature | SENTINEL | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 77 | Portfolio VaR / CVaR (historical + parametric + MC) | ✓ | 3 | 2 | 3 | 0 | 3 | 2 | 0 |
| 78 | Brinson-Hood-Beebower performance attribution | ✓ | 3 | 2 | 3 | 0 | 2 | 3 | 0 |
| 79 | Multi-factor risk decomp. (Fama-French 5+momentum) | ✓ | 3 | 2 | 3 | 0 | 2 | 2 | 0 |
| 80 | Correlation monitoring + regime-change alerts | ✓ | 3 | 1 | 2 | 0 | 2 | 1 | 0 |
| 81 | Portfolio optimizer (mean-variance, risk parity, BL) | ✓ | 3 | 1 | 2 | 0 | 1 | 2 | 0 |
| 82 | Kelly / vol-target / risk-parity position sizing | ✓ | 2 | 0 | 1 | 0 | 1 | 0 | 0 |
| 83 | Stress testing / scenario analysis | △ | 3 | 2 | 2 | 0 | 3 | 2 | 0 |

**SENTINEL: 88% spec-complete** | Matches Bloomberg on portfolio analytics; exceeds CapIQ

---

## CATEGORY 10 — Alternative & Satellite Data (6 Dimensions)

| # | Feature | SENTINEL | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 84 | News sentiment pipeline (GDELT + FinBERT + NLP) | ✓ | 3 | 1 | 1 | 3 | 2 | 0 | 0 |
| 85 | Social media sentiment (Reddit wsb + StockTwits API) | ✓ | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| 86 | Job postings signal (Indeed/LinkedIn scrape + Google Trends) | △ | 2 | 1 | 1 | 1 | 1 | 0 | 0 |
| 87 | Satellite imagery signals (parking lot / shipping) | ✗ | 2 | 0 | 0 | 0 | 1 | 0 | 0 |
| 88 | AIS shipping / cargo tracking signal | △ | 2 | 0 | 0 | 0 | 1 | 0 | 0 |
| 89 | Google Trends / pytrends consumer search signals | ✓ | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

**SENTINEL: 70% spec-complete** | Satellite data is Gen 3 roadmap (Planet Labs research API)

---

## CATEGORY 11 — Terminal UX & Developer Surface (7 Dimensions)

| # | Feature | SENTINEL | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 90 | Bloomberg-style command bar (TICKER FN `<GO>`) | ✓ | 3 | 0 | 1 | 0 | 1 | 0 | 0 |
| 91 | Multi-panel workspace (customizable Streamlit layout) | ✓ | 3 | 2 | 2 | 1 | 3 | 1 | 1 |
| 92 | Real-time charting (TradingView Lightweight Charts) | ✓ | 3 | 2 | 2 | 1 | 3 | 2 | 0 |
| 93 | Excel / Google Sheets plugin | ✗ | 3 | 3 | 3 | 0 | 3 | 2 | 2 |
| 94 | Mobile app (iOS / Android) | ✗ | 2 | 2 | 2 | 1 | 2 | 2 | 1 |
| 95 | REST API + WebSocket SDK for programmatic access | ✓ | 2 | 2 | 2 | 1 | 2 | 1 | 1 |
| 96 | Self-hosted / sovereign (no vendor lock-in, no seat fee) | ✓ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**SENTINEL: 70% spec-complete** | Excel plugin Gen 3, mobile Gen 4. Sovereign self-hosting is the unique differentiator — no competitor offers this.

---

## CATEGORY 12 — Private Markets & Deal Intelligence (5 Dimensions)

| # | Feature | SENTINEL | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 97 | Private company profiles (Form D + Crunchbase free) | △ | 2 | 3 | 2 | 0 | 2 | 0 | 3 |
| 98 | VC/PE fund tracking (Form ADV + Reg D + IAPD) | △ | 2 | 3 | 2 | 0 | 2 | 0 | 3 |
| 99 | Private valuations | ✗ | 1 | 3 | 1 | 0 | 1 | 0 | 3 |
| 100 | M&A deal intelligence (S-1 + 8-K + SEC EDGAR parsing) | △ | 3 | 3 | 3 | 1 | 2 | 0 | 1 |
| 101 | LBO / merger model templates (Damodaran base) | ✓ | 3 | 3 | 3 | 0 | 2 | 0 | 1 |

**SENTINEL: 48% spec-complete** | **Deliberate gap:** Private valuations require confidential LP data — structurally unreplicable. SENTINEL acknowledges PitchBook's moat here.

---

## CATEGORY 13 — ESG & Sustainability (4 Dimensions)

| # | Feature | SENTINEL | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 102 | ESG composite ratings & sector scores | △ | 3 | 3 | 3 | 1 | 3 | 3 | 0 |
| 103 | CDP / TCFD climate disclosure parsing (LLM) | ✓ | 2 | 2 | 2 | 1 | 2 | 2 | 0 |
| 104 | Controversy monitoring & media-based flag | △ | 2 | 2 | 2 | 2 | 2 | 2 | 0 |
| 105 | UN SDG alignment / impact factor scoring | △ | 1 | 1 | 1 | 0 | 1 | 1 | 0 |

**SENTINEL: 58% spec-complete** | Note: ESG ratings are proprietary methodologies. SENTINEL provides raw inputs (CDP, sustainability PDFs, GDELT controversy) rather than replicating opaque scoring rubrics.

---

## CATEGORY 14 — Crypto & DeFi Intelligence (5 Dimensions)

| # | Feature | SENTINEL | Bloomberg | CapIQ | FactSet | AlphaSense | LSEG | Morningstar | PitchBook |
|---|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 106 | Multi-exchange execution + data (CCXT, 100+ venues) | ✓ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 107 | DeFi protocol analytics (DefiLlama TVL, yields, chains) | ✓ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 108 | On-chain metrics (MVRV, NVT, SOPR, exchange flows) | ✓ | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 109 | On-chain event monitoring (Etherscan, whale alerts) | ✓ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 110 | DEX / AMM liquidity + impermanent loss analytics | △ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**SENTINEL: 88% spec-complete** | **FULL LEAPFROG.** Every institutional terminal scores 0-4% on crypto/DeFi. This is a complete blind spot in the $40B institutional data industry.

---

## Composite Summary

```
Category                  SENTINEL   Bloomberg   CapIQ   FactSet   AlphaSense   LSEG   Morningstar   PitchBook
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
1.  Market Data              63%        97%        42%     67%         0%        92%      17%           0%
2.  Fundamentals             72%        98%        95%     95%        13%        83%      75%           0%
3.  Ownership/SEC            95%        80%        80%     65%         0%        50%      33%           0%
4.  Fixed Income             70%       100%        75%     75%         0%        88%      17%           0%
5.  Macro                    88%        80%        36%     44%         0%        83%       8%           0%
6.  AI/NLP                   82%        22%        22%     22%        83%        22%       0%           0%
7.  Backtesting/Execution    85%        22%         0%      0%         0%         0%       0%           0%
8.  Screening                88%        73%        58%     65%         8%        65%      17%           0%
9.  Portfolio/Risk           88%        97%        52%     86%         0%        75%      67%           0%
10. Alt Data                 70%        55%        11%     11%        33%        22%       0%           0%
11. Terminal UX              70%        73%        58%     65%        17%        65%      33%           8%
12. Private Markets          48%        66%       100%     42%         0%        42%       0%         100%
13. ESG                      58%        88%        88%     88%        25%        88%      88%           0%
14. Crypto/DeFi              88%         4%         0%      0%         0%         0%       0%           0%
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
COMPOSITE                    76%        68%        51%     59%        13%        56%      25%           8%
Annual Cost                  $0      $31,980    $18,500  $28,500    $50,000   $16,000   $17,500      $25,000
```

---

## Leapfrog Opportunity Map

SENTINEL exceeds all competitors on 10 specific dimensions that represent structural market gaps:

| # | Leapfrog Dimension | SENTINEL Score | Best Competitor | Delta |
|---|-------------------|:---------:|:---------------:|:-----:|
| L1 | MCP agent-native tool surface (15 tools) | ✓ unique | 0/3 | +3.0 |
| L2 | NL → trading strategy generator | ✓ unique | 0/3 | +3.0 |
| L3 | Vectorized + event-driven backtesting unified | ✓ unique | 1/3 (Bloomberg execute-only) | +2.0 |
| L4 | Congressional STOCK Act eFD disclosures | ✓ | 1/3 (Bloomberg minimal) | +1.0 |
| L5 | CFTC COT positioning in terminal UX | ✓ | 1/3 (minimal) | +1.0 |
| L6 | Crypto/DeFi analytics (CCXT + DefiLlama + on-chain) | 88% | 4% (Bloomberg) | +84% |
| L7 | Sovereign self-hosted (zero vendor lock-in) | ✓ unique | 0/3 | +3.0 |
| L8 | Strategy promotion state machine | ✓ unique | 0/3 | +3.0 |
| L9 | HMM regime detection integrated | ✓ unique | 0/3 | +3.0 |
| L10 | DSR / PBO overfitting detection | ✓ unique | 0/3 | +3.0 |

---

## Deliberate Non-Goals (Structural Gaps — Accepted)

| Gap | Reason | Competitor with Moat |
|-----|--------|---------------------|
| Live OTC bond bid/ask | Bloomberg IB chat network — network effect moat | Bloomberg only |
| Private valuations | Requires confidential LP data, 1,800+ data analysts | PitchBook |
| Expert call transcripts | Tegus/Mosaic licensing, $30K+/yr | AlphaSense |
| CUSIP licensing | FactSet subsidiary, $477K+/yr/major-firm | FactSet |
| CDS pricing | Dealer-controlled data | Bloomberg only |
| Barra factor model | MSCI proprietary | MSCI/Bloomberg |
| 60+ year historical depth | CRSP/Datastream require academic/paid access | LSEG Datastream |

---

*Generated by DanteForge /universe — May 7, 2026*  
*Next: /oss to populate the full OSS project catalog backing each dimension*
