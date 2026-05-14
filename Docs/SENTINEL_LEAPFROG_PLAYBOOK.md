# SENTINEL Leapfrog Playbook
**Version:** 1.0
**Date:** May 7, 2026
**Method:** DanteForge /adversarial-score + /competitive-leapfrog
**Purpose:** Identify the 10 structural leapfrog opportunities where SENTINEL can score 10/10 — exceeding every incumbent — and provide the exact execution plan for each.

---

## What "Leapfrog" Means

A leapfrog dimension is one where:
1. SENTINEL's target score is **10/10** (not just 9)
2. The best incumbent scores **≤ 3/10**
3. The gap is **structural** — incumbents cannot easily close it because doing so would disrupt their own revenue model

Bloomberg cannot ship a sovereign self-hosted terminal: it would destroy their $31,980/yr seat fee business.
AlphaSense cannot ship a backtesting engine: it is not their product surface.
No terminal can ship congressional trade intelligence at depth: their data teams don't prioritize it.

SENTINEL can do all of these because it has no incumbent revenue to protect.

---

## The 10 Leapfrog Dimensions

### L1 — Natural Language → Trading Strategy Generator
**Dimension:** #54 | **SENTINEL Target:** 10 | **Best Incumbent:** 0 (AlphaSense attempts NL search, not strategy generation)

**What it does:** User types "Buy small-cap biotech stocks when insider buying clusters and short interest is above 20% in a Growth regime" → SENTINEL generates a validated strategy YAML spec → launches VectorBT backtest → returns 24-metric tearsheet with DSR gate.

**Why incumbents can't match this:**
- Bloomberg is a data terminal, not a strategy engine
- AlphaSense does document search, not systematic strategy generation
- FactSet provides templates, not LLM-driven strategy synthesis
- The combination of NL understanding + structured backtesting + statistical validity gates doesn't exist anywhere

**Implementation Plan:**
```python
# sil/strategy_generator.py — core flow
hypothesis: str → claude_structured_output() → StrategySpec (Pydantic) → 
validate_criteria_exist(spec, sse_criteria_list) → sbe.run(spec) → tearsheet
```

Key components:
1. Claude `claude-opus-4-7` with structured output + tool use for criteria validation
2. Strategy YAML schema (defined in TRD Part 6.4)
3. VectorBT backend receives the parsed spec
4. DSR/PBO gates applied automatically
5. Results returned to STU strategy panel

**Sprint:** Gen 1, 3 weeks, `sil/strategy_generator.py` + `sbe/vectorbt_backend.py`
**Current Score:** 6 (spec complete) | **Target:** 10

---

### L2 — MCP Agent-Native Tool Surface (15 Tools)
**Dimension:** #59 | **SENTINEL Target:** 10 | **Best Incumbent:** 0

**What it does:** SENTINEL exposes 15 financial tools through the Model Context Protocol. Claude Code or Claude Desktop can call `search_filings(query)`, `run_screen(criteria)`, `get_congressional_trades(member)`, `run_backtest(strategy_yaml)`, `get_regime(date)` directly — no API key management, no wrapper code.

**Why incumbents can't match this:**
- MCP is an open standard donated to the Linux Foundation in November 2024
- Bloomberg, FactSet, CapIQ all have APIs, but none expose an MCP server
- MCP is the agentic-computing equivalent of what HTTP was to the web — first-mover advantage matters
- Bloomberg's API (BLPAPI) costs $10K+/yr to access and requires enterprise contracts

**Implementation Plan:**
```python
# sil/mcp_server.py (FastMCP)
@mcp.tool() search_filings(query, form_type, ticker, date_range) → list[FilingExcerpt]
@mcp.tool() get_financials(ticker, period, metrics, years_back) → FinancialSummary
@mcp.tool() run_screen(criteria) → list[SecurityMatch]
@mcp.tool() get_ownership(ticker, holder_type) → OwnershipReport
@mcp.tool() get_insider_trades(ticker, days_back) → list[InsiderTrade]
@mcp.tool() get_congressional_trades(member, ticker, days_back) → list[CongressTrade]
@mcp.tool() get_macro_series(fred_id, start_date, end_date) → MacroSeries
@mcp.tool() run_backtest(strategy_yaml) → BacktestResult
@mcp.tool() get_portfolio_risk(portfolio_id) → RiskReport
@mcp.tool() get_earnings_call_summary(ticker, quarter) → EarningsSummary
@mcp.tool() get_sentiment(text_or_ticker) → SentimentResult
@mcp.tool() get_cot_report(market, date_range) → COTReport
@mcp.tool() get_regime(date) → RegimeClassification
@mcp.tool() screen_natural_language(query) → list[SecurityMatch]
@mcp.tool() explain_strategy(strategy_id) → str
```

**Sprint:** Gen 1, 2 weeks, `sil/mcp_server.py`
**Current Score:** 6 (spec complete) | **Target:** 10

---

### L3 — Strategy Promotion State Machine
**Dimension:** #67 | **SENTINEL Target:** 10 | **Best Incumbent:** 0

**What it does:** Codified governance gates that a trading strategy must pass before going live. Backtest → Paper (requires DSR > 0.5, PBO < 0.5) → Capped Live (requires 60-day paper run, Sharpe > 0.7) → Full Autonomous (requires 180-day paper + 90-day capped live). Every transition requires human approval. Kill switch always available.

**Why this matters:** The single biggest failure mode in systematic trading is deploying a strategy that looked great in backtest but was overfit. No terminal codifies this governance. Professional quant funds spend millions on this infrastructure. SENTINEL ships it free.

**Why incumbents can't match this:**
- Bloomberg executes but doesn't govern the research-to-live pipeline
- QuantConnect LEAN has some version of this but it's cloud-locked
- The combination of DSR + PBO + HMM regime annotation + promotion gates in one open terminal doesn't exist

**Implementation Plan:**
```python
# see/promotion.py — state machine
class StrategyStateMachine:
    TRANSITIONS = {
        "backtest" → "paper": [check_dsr(>0.5), check_pbo(<0.5), require_human_approval()],
        "paper" → "capped_live": [check_paper_sharpe(>0.7), check_paper_dd(<0.15), 
                                   check_paper_duration(>=60), require_human_approval()],
        "capped_live" → "full_autonomous": [check_live_sharpe(>0.8), 
                                             check_paper_duration(>=180),
                                             check_capped_duration(>=90),
                                             require_human_approval()],
    }
    
    async def promote(self, strategy_id: str, to_status: str) -> PromotionResult:
        gates = self.TRANSITIONS[f"{current} → {to_status}"]
        results = await asyncio.gather(*[gate(strategy_id) for gate in gates])
        if all(r.passed for r in results):
            await self._apply_transition(strategy_id, to_status)
            return PromotionResult(success=True, new_status=to_status)
        return PromotionResult(success=False, failed_gates=[r for r in results if not r.passed])
```

**Sprint:** Gen 1, 2 weeks, `see/promotion.py`
**Current Score:** 6 (spec complete) | **Target:** 10

---

### L4 — Deflated Sharpe Ratio + Probability of Backtest Overfitting
**Dimension:** #64 | **SENTINEL Target:** 10 | **Best Incumbent:** 0

**What it does:** Two mandatory statistical validity tests before any strategy can be promoted from backtest:
- **DSR** (Bailey & Lopez de Prado, JPM 2014): Adjusts Sharpe for skewness, kurtosis, and multiple-testing bias. DSR < 0.5 → reject.
- **PBO** (Bailey/Borwein/Lopez de Prado/Zhu 2014): Combinatorial cross-validation to estimate probability that the selected strategy is the best by luck. PBO > 0.5 → reject.

**Why incumbents can't match this:** These are cutting-edge academic results from 2014. Even professional quant shops often skip DSR/PBO because it's complex to implement. Shipping this as a mandatory gate in a free terminal is unprecedented.

**Implementation Plan:**
```python
# sbe/dsr.py — already specified in TRD Part 7.2
# sbe/pbo.py
from itertools import combinations
import numpy as np

def compute_pbo(returns_matrix: np.ndarray, n_splits: int = 16) -> float:
    """
    Probability of Backtest Overfitting.
    returns_matrix: (T, N) — T periods, N strategy variants
    Returns: PBO ∈ [0, 1]. Higher = more likely overfit.
    """
    T, N = returns_matrix.shape
    subset_size = T // 2
    overfit_count = 0
    total_partitions = 0
    
    for is_indices in _generate_partitions(T, subset_size, n_splits):
        oos_indices = np.setdiff1d(np.arange(T), is_indices)
        is_returns = returns_matrix[is_indices]
        oos_returns = returns_matrix[oos_indices]
        
        # Best IS strategy
        is_sharpes = np.mean(is_returns, axis=0) / np.std(is_returns, axis=0) * np.sqrt(252)
        best_is_idx = np.argmax(is_sharpes)
        
        # Rank of best IS strategy in OOS
        oos_sharpes = np.mean(oos_returns, axis=0) / np.std(oos_returns, axis=0) * np.sqrt(252)
        oos_rank = np.sum(oos_sharpes >= oos_sharpes[best_is_idx]) / N
        
        if oos_rank < 0.5:  # Best IS strategy performs below median OOS
            overfit_count += 1
        total_partitions += 1
    
    return overfit_count / total_partitions
```

**Sprint:** Gen 1, 1 week, `sbe/dsr.py` + `sbe/pbo.py`
**Current Score:** 6 (spec complete) | **Target:** 10

---

### L5 — HMM Macro Regime Detection
**Dimension:** #50 | **SENTINEL Target:** 10 | **Best Incumbent:** 3 (Bloomberg has some regime views, not integrated into terminal)

**What it does:** Hidden Markov Model trained on FRED macro indicators classifies the current macro regime into 4 states: Growth/Inflation, Growth/Deflation, Contraction/Inflation (Stagflation), Contraction/Deflation. Updates weekly. Annotates all backtest tearsheets with prevailing regime during each period.

**Strategic value:** Every strategy has regime-conditional performance. A momentum strategy that returns 15% Sharpe in Growth/Inflation may return -0.5 Sharpe in Contraction/Inflation. SENTINEL makes this transparent in every backtest tearsheet. Bloomberg doesn't.

**Why incumbents score 3:** Bloomberg has economic commentary about regimes, but it's not a machine-readable classification integrated into portfolio or backtest tools.

**Implementation Plan:**
```python
# sma/regime.py
from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
from fredapi import Fred

REGIME_INDICATORS = [
    "GDP growth (GDPC1 QoQ)", "CPI YoY (CPIAUCSL)",
    "Unemployment rate (UNRATE)", "10Y-2Y spread (T10Y2Y)",
    "ISM Manufacturing PMI", "Credit spreads (BAMLC0A0CM)",
    "VIX (VIXCLS)", "USD Index (DTWEXBGS)"
]

REGIME_LABELS = {
    0: "Growth/Inflation",
    1: "Growth/Deflation", 
    2: "Contraction/Inflation",
    3: "Contraction/Deflation"
}

class MacroRegimeDetector:
    def __init__(self, n_components: int = 4):
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
    
    def train(self, fred: Fred, start_date: str = "1970-01-01"):
        features = self._build_feature_matrix(fred, start_date)
        self.model.fit(features)
        return self
    
    def predict(self, fred: Fred, as_of_date: str = None) -> dict:
        features = self._build_feature_matrix(fred, "2000-01-01", as_of_date)
        hidden_states = self.model.predict(features)
        probs = self.model.predict_proba(features)
        current_regime_id = hidden_states[-1]
        return {
            "regime": REGIME_LABELS[current_regime_id],
            "regime_id": int(current_regime_id),
            "confidence": float(probs[-1, current_regime_id]),
            "probabilities": {REGIME_LABELS[i]: float(probs[-1, i]) for i in range(4)}
        }
```

**Sprint:** Gen 1, 1 week, `sma/regime.py`
**Current Score:** 6 (spec complete) | **Target:** 10

---

### L6 — CFTC COT Positioning Intelligence in Terminal UX
**Dimension:** #46 | **SENTINEL Target:** 10 | **Best Incumbent:** 7 (Bloomberg has COT data but not a terminal-integrated COT Index)

**What it does:** Every Friday, CFTC publishes Commitment of Traders reports for 150+ futures markets. SENTINEL ingests all three report types (Legacy, Disaggregated, Traders in Financial Futures), computes:
- Net positioning by trader type (commercial hedger vs. large speculator vs. small speculator)
- **COT Index** = current net position as percentile of 52-week range (extremes signal reversals)
- Historical COT Index with overlay on price chart

**Why this gives edge:** Commercial hedgers (producers, processors) are right about commodity direction long-term. Large speculators (trend-following CTAs) are right short-term but wrong at extremes. The COT Index identifies when specs are maximally long (bearish signal) or maximally short (bullish signal) with documented historical predictive power.

**Why incumbents score 7:** Bloomberg displays COT data but doesn't integrate COT Index percentiles into a visual signal layer or screener criteria.

**Implementation Plan:**
```python
# sma/cot_report.py
import pandas as pd
import requests
from datetime import date

CFTC_URLS = {
    "legacy": "https://www.cftc.gov/dea/newcot/f_year.htm",
    "disaggregated": "https://www.cftc.gov/dea/newcot/fut_disagg_txt_hist_2006_to_present.zip",
    "tff": "https://www.cftc.gov/dea/newcot/fut_fin_txt_hist_2006_to_present.zip"
}

class COTParser:
    def fetch_and_parse(self, report_type: str = "disaggregated") -> pd.DataFrame:
        """Download and parse CFTC COT report. Returns DataFrame with 150+ markets."""
        url = CFTC_URLS[report_type]
        df = pd.read_csv(url, low_memory=False)
        return self._normalize_columns(df)
    
    def compute_cot_index(self, df: pd.DataFrame, market: str, lookback_weeks: int = 52) -> pd.Series:
        """
        COT Index = (current_net - min_net) / (max_net - min_net) × 100
        Values near 0 = spec extreme short (bullish) | Values near 100 = spec extreme long (bearish)
        """
        market_df = df[df["Market_and_Exchange_Names"] == market].sort_values("Report_Date_as_MM_DD_YYYY")
        net_spec = market_df["NonComm_Positions_Long_All"] - market_df["NonComm_Positions_Short_All"]
        
        rolling_min = net_spec.rolling(lookback_weeks).min()
        rolling_max = net_spec.rolling(lookback_weeks).max()
        cot_index = (net_spec - rolling_min) / (rolling_max - rolling_min + 1e-10) * 100
        return cot_index
```

**Sprint:** Gen 1, 1 week, `sma/cot_report.py`
**Current Score:** 6 (spec complete) | **Target:** 10

---

### L7 — Sovereign Self-Hosted (Zero Vendor Lock-in)
**Dimension:** #96 | **SENTINEL Target:** 10 | **Best Incumbent:** 0

**What it does:** SENTINEL runs entirely on a single Mac mini. No mandatory cloud subscription. No per-seat fee. No data egress charges. No vendor control over your data. You own the hardware, the database, the embeddings, the strategy code, and the audit trail.

**Why this is a leapfrog:** Every competitor is a SaaS subscription. Even "open source" platforms like QuantConnect LEAN require their cloud for full functionality. SENTINEL is the first terminal where the operator has complete sovereignty over the full stack.

**Why incumbents score 0:** Their entire business model is the recurring subscription. Bloomberg cannot give you a self-hosted Bloomberg — it would destroy their $10B/yr terminal business.

**Implementation Plan:** Already specified in TRD Part 2 (Docker Compose stack). The entire stack runs locally:
- TimescaleDB on Docker: local time-series storage
- PostgreSQL on Docker: structured data
- pgvector on Docker: embedding search
- Redis on Docker: pub/sub and caching
- FastAPI on Docker: API layer
- Streamlit on Docker: terminal UI
- Grafana on Docker: monitoring dashboards

**Sprint:** Gen 1 (it's the infrastructure — Day 1)**Current Score:** 6 (spec complete) | **Target:** 10 (achieved by definition when it runs locally)

---

### L8 — Congressional STOCK Act Intelligence
**Dimension:** #29 | **SENTINEL Target:** 10 | **Best Incumbent:** 3 (minimal coverage on any platform)

**What it does:** Members of Congress are required to disclose stock trades within 30-45 days of transaction under the STOCK Act. Research by Jochec (2020) and Eggers & Hainmueller (2014) documents 5-15% annualized abnormal returns from following informed congressional trades — particularly purchases from members on relevant committee assignments.

SENTINEL provides:
- Real-time eFD disclosure parsing (Senate + House)
- Member → committee assignment mapping (PACER/Congress.gov)
- Stock picks by committee relevance (Finance Committee + Banking = financial sector; Energy Committee = energy sector)
- Buy/sell clustering: when 5+ members buy the same stock in 30 days
- Historical performance tracking of congressional stock picks by member

**Why incumbents score 3:** This data is completely free and publicly available, but incumbents have never prioritized it. Bloomberg has a token "Congressional activity" search. No terminal has a systematic alpha signal generator on this data.

**Implementation Plan:**
```python
# sod/congressional.py
import requests
from bs4 import BeautifulSoup

SENATE_EFD_URL = "https://efts.senate.gov/LATEST/search-index?q=*&df=pd&dt={year}"
HOUSE_DISCLOSURE_URL = "https://disclosures-clerk.house.gov/FinancialDisclosure"

class CongressionalTradeParser:
    async def fetch_senate_disclosures(self, year: int = 2026) -> list[dict]:
        """Parse Senate eFD XML filings for periodic transaction reports."""
        ...
    
    async def fetch_house_disclosures(self, year: int = 2026) -> list[dict]:
        """Parse House PTR CSV/XML for periodic transaction reports."""
        ...
    
    def generate_signal(self, trades: list[dict]) -> list[dict]:
        """
        Signal generation rules:
        1. Cluster: 5+ members buy same stock in 30-day window → STRONG BUY signal
        2. Committee relevance: member on relevant committee buys sector stock → INFORMED signal
        3. Pre-announcement: trade < 14 days before material 8-K → flag for review
        """
        ...
```

**Sprint:** Gen 1, 1 week, `sod/congressional.py`
**Current Score:** 6 (spec complete) | **Target:** 10

---

### L9 — Crypto & DeFi Full Category
**Dimensions:** #106–#110 | **SENTINEL Target:** 9-10 | **Best Incumbent:** 3 (Bloomberg, token)

**What it does:** Complete crypto/DeFi intelligence layer:
- CCXT: 100+ exchange execution, OHLCV, order book, account management
- DefiLlama: TVL by protocol/chain, yield opportunities, stable coin flows
- On-chain metrics: MVRV z-score, NVT ratio, SOPR, exchange net flows (via Glassnode free + Etherscan)
- DEX analytics: Uniswap/Curve/Balancer pool TVL, fee APR, impermanent loss calculator
- Cross-asset: crypto + macro + equity in one terminal — unique positioning

**Why incumbents score 0-3:** Bloomberg added some crypto in 2021 but it's limited to price data. CapIQ, FactSet, LSEG, Morningstar all score 0. This is a complete institutional blind spot in a $3T+ asset class.

**Implementation Plan:**
- CCXT integration: `sds/adapters/ccxt_adapter.py` — already in TSD
- DefiLlama: `sds/adapters/defillama_adapter.py` — free REST API
- On-chain: `sds/adapters/etherscan_adapter.py` + `sds/adapters/glassnode_adapter.py`
- DEX analytics: `sdex/` module (new Gen 2 module)

**Sprint:** Gen 1 (CCXT + DefiLlama), Gen 2 (DEX analytics, on-chain depth)
**Current Score:** 5.4 avg | **Target:** 9.0

---

### L10 — Vectorized + Event-Driven Backtesting Unified
**Dimensions:** #61, #62, #63 | **SENTINEL Target:** 9 | **Best Incumbent:** 3 (Bloomberg execute-only)

**What it does:** Two backtesting engines in one product:
1. **VectorBT** — vectorized, Numba-accelerated: parameter sweeps of 1,000+ variants in 60 seconds
2. **NautilusTrader** — event-driven, Rust core: tick/bar fidelity, identical to live trading code path

No other terminal product ships both. Professional quants pay $50K+/yr in software licenses to have this combination.

**Why incumbents score ≤3:** Bloomberg executes but doesn't research. QuantConnect LEAN has both but is cloud-locked. Open-source tooling is available but never integrated into a single terminal with data, screening, and execution in the same product.

**Sprint:** Gen 1 (VectorBT), Gen 2 (NautilusTrader event-driven)
**Current Score:** 5.0 avg | **Target:** 9.0

---

## Adversarial Score Analysis — Where SENTINEL is Vulnerable

Running /adversarial-score against the SENTINEL spec to identify the most likely criticisms:

### Adversarial Attack 1: "yfinance is unsanctioned and will break"
**Attack:** Yahoo Finance has no official API. yfinance scrapes HTML/JSON and breaks periodically. A terminal depending on it for primary data is fragile.
**SENTINEL Response:** 
- yfinance is Tier 1 in a fallback chain: yfinance → Stooq → Alpha Vantage → EODHD
- All free data is cached in TimescaleDB — a throttle doesn't break existing data, only future fetches
- DataHealthEvent fires on staleness; operator is notified
- Score impact: not a leapfrog gap, but a reliability concern addressed by multi-adapter design
**Residual Risk:** Medium. Accept risk; mitigated by caching and fallback.

### Adversarial Attack 2: "You can't replicate Bloomberg's data depth"
**Attack:** Bloomberg covers 35M+ instruments. SENTINEL's free sources cover maybe 20,000–50,000 equities well. International coverage is thin.
**SENTINEL Response:**
- Accepted limitation. SENTINEL targets 10,000+ US equities (EDGAR universe) with deep coverage + international equities via yfinance/EODHD/Stooq.
- The 95% use case for a retail/boutique RIA is the liquid US equity + crypto universe.
- Deliberate non-goal: obscure small-cap international equities.
**Residual Risk:** Real gap. Mitigated by TargetUser definition (Persona 1-4 don't need 35M instruments).

### Adversarial Attack 3: "LGPL-3.0 on NautilusTrader creates license contagion"
**Attack:** NautilusTrader uses LGPL-3.0. Using it in a commercial product may require open-sourcing the SENTINEL code.
**SENTINEL Response:**
- LGPL-3.0 (Library GPL) specifically exempts software that *links* to the library as a shared library
- SENTINEL uses NautilusTrader as a Python dependency (dynamic linking), not as a static compilation
- Standard legal interpretation: LGPL dynamic linking does not require open-sourcing the parent application
- Mitigation: SENTINEL's open-core model plans MIT license for core anyway
- Alternative: QuantConnect LEAN (Apache 2.0) could replace NT for license-sensitive deployments
**Residual Risk:** Low. Legal opinion should be obtained before commercial deployment.

### Adversarial Attack 4: "SEC EDGAR rate limits will throttle your ingestion pipeline"
**Attack:** EDGAR enforces 10 req/sec globally. With 10K+ companies and multiple filing types, this is a real bottleneck.
**SENTINEL Response:**
- Nightly bulk download: `companyfacts.zip` (~1.5GB) covers all XBRL facts for all companies in a single download
- Real-time monitoring uses RSS feed (1 req/30 seconds), not per-company polling
- Rate limiter is enforced in code (TokenBucketRateLimiter in TRD Part 5.2)
- Historical backfill runs overnight with deliberate rate limiting
**Residual Risk:** Low. Bulk download eliminates 99% of per-filing API calls.

### Adversarial Attack 5: "FinBERT accuracy is not AlphaSense-grade"
**Attack:** ProsusAI/finbert achieves ~85% accuracy on financial phrase bank. AlphaSense's proprietary model likely outperforms. The gap matters for document intelligence quality.
**SENTINEL Response:**
- FinBERT is the baseline. SENTINEL can fine-tune on a domain-specific corpus of earnings calls.
- `voyage-finance-2` embeddings (specifically finance-tuned) compensate on retrieval tasks
- For summarization/synthesis, Claude claude-opus-4-7 outperforms any domain-specific model
- AlphaSense's "moat" is their expert call transcript corpus, not their NLP model
**Residual Risk:** Medium on sentence-level sentiment precision. Accepted tradeoff vs. $50K/yr AlphaSense.

### Adversarial Attack 6: "This can't compete with Bloomberg on fixed income"
**Attack:** Bloomberg's FISR function + live bond pricing via IB chat is categorically different from FINRA TRACE. Institutional bond desks will never replace Bloomberg with this.
**SENTINEL Response:**
- Fully accepted. Dimension #42 (live OTC bid/ask) is listed as a deliberate non-goal.
- SENTINEL competes on: yield curves, analytics (QuantLib), muni market (EMMA), corporate bond history (TRACE)
- Target users are not bond dealers. They are equity-focused investors who also need basic fixed income context.
- Score impact: Category 4 target is 7.1 (not 9) — the only category where SENTINEL doesn't aim for near-parity
**Residual Risk:** None — accepted structural limitation with clear user boundary.

### Adversarial Attack 7: "Privacy and security risk of self-hosting financial data"
**Attack:** Running a terminal with live broker credentials, API keys, and trading account access on a local Mac mini creates security risks.
**SENTINEL Response:**
- Security spec in TRD Part 9: all secrets in `.env` (gitignored), never in code/logs
- Live trading requires explicit `SENTINEL_LIVE_TRADING=true` + `--live` CLI flag
- Kill switch always accessible from localhost without authentication
- No data leaves the machine unless the operator explicitly configures external connections
- Operator controls all firewall rules — SENTINEL never calls home
**Residual Risk:** Low if operator follows security spec. Medium if operator is security-naive — document clearly in README.

---

## /ascend Sprint Plan — Path from Current State to 9+

### Sprint 1 (Weeks 1-4): Foundation Infrastructure
- Docker Compose stack up and running
- TimescaleDB + PostgreSQL + Redis + pgvector configured
- FastAPI skeleton with all route stubs
- SDS: yfinance + FRED + EDGAR adapters
- SIM: OpenFIGI resolver, ticker→FIGI mapping
- **Score movement:** Infrastructure 0→7 | Market Data 3.8→5.0

### Sprint 2 (Weeks 5-8): Filing Engine + Ownership
- SFE: 10-K/10-Q XBRL parser, Form 4, 13F parsers (edgartools)
- SOD: 13F tracker, Form 4 signal feed
- **L8 LEAPFROG: Congressional STOCK Act** — `sod/congressional.py`
- **Score movement:** Ownership/SEC 6.0→8.5 | Congressional 6→10

### Sprint 3 (Weeks 9-12): Intelligence Layer + MCP
- SIL: RAG pipeline (LlamaIndex + pgvector)
- SIL: FinBERT sentiment pipeline
- **L2 LEAPFROG: MCP server (15 tools)** — `sil/mcp_server.py`
- **L1 LEAPFROG: NL-to-strategy** — `sil/strategy_generator.py`
- **Score movement:** AI/NLP 4.8→8.5 | MCP 6→10 | NL-strategy 6→10

### Sprint 4 (Weeks 13-16): Backtesting + Overfitting Controls
- SBE: VectorBT backend
- **L4 LEAPFROG: DSR + PBO** — `sbe/dsr.py` + `sbe/pbo.py`
- SBE: Walk-forward validation
- **Score movement:** Backtesting 5.0→8.5 | DSR/PBO 6→10

### Sprint 5 (Weeks 17-20): Macro + Regime + COT
- SMA: FRED client complete, 200+ series configured
- **L5 LEAPFROG: HMM regime detector** — `sma/regime.py`
- **L6 LEAPFROG: CFTC COT** — `sma/cot_report.py`
- SBX: QuantLib wrapper, TRACE + EMMA clients
- **Score movement:** Macro 5.6→9.0 | Regime 6→10 | COT 6→10

### Sprint 6 (Weeks 21-24): Execution + Promotion + Terminal UI
- SEE: Alpaca paper + live adapter
- **L3 LEAPFROG: Strategy promotion state machine** — `see/promotion.py`
- STU: Streamlit terminal, command bar, charting, watchlist, portfolio panel
- SPR: Portfolio VaR, attribution, optimizer
- **Score movement:** Execution 5.0→8.5 | Promotion 6→10 | Terminal UX 4.3→7.0

### Sprint 7 (Weeks 25-28): Crypto + DeFi + News
- SDS: CCXT adapter (100+ exchanges)
- **L9: Crypto/DeFi** — `sds/adapters/ccxt_adapter.py` + DefiLlama + Etherscan
- SNM: Whisper transcription pipeline, GDELT, RSS ingestion, FinBERT tagging
- **Score movement:** Crypto 5.4→9.0 | Alt Data 4.0→7.5

### Sprint 8 (Weeks 29-32): Gen 1 Complete — Integration + Testing
- Financial eval tests: SPY buy-hold, momentum factor, PIT integrity
- Adapter health checks: all free data sources
- Integration tests: cross-module data flow
- Performance testing: all latency targets from TRD Part 10
- **Score movement:** All categories → Gen 1 targets | Composite ~4.7→7.5

---

## Final Score Projection by Generation

```
                         Gen 0    Gen 1    Gen 2    Gen 3    Gen 4
                         (now)   (3 mo)   (6 mo)   (9 mo)  (15 mo)
──────────────────────────────────────────────────────────────────
Market Data               3.8      5.5      7.5      8.5      9.0
Fundamentals              4.5      7.0      8.5      9.0      9.0
Ownership/SEC             6.0      9.0      9.0      9.0      9.0
Fixed Income              4.1      7.0      8.0      8.5      9.0
Macro                     5.6      9.0      9.5      9.5     10.0
AI/NLP                    4.8      7.5      9.0      9.5     10.0
Backtesting               5.0      8.0      9.0      9.0      9.5
Screening                 6.0      8.0      9.0      9.5     10.0
Portfolio/Risk            5.6      7.5      8.5      9.0      9.0
Alt Data                  4.0      6.5      7.5      8.0      9.0
Terminal UX               4.3      6.5      8.0      8.5      9.0
Private Markets           3.0      4.0      5.5      6.0      7.0
ESG                       3.8      5.5      7.0      8.0      8.5
Crypto/DeFi               5.4      9.0      9.5      9.5     10.0
──────────────────────────────────────────────────────────────────
COMPOSITE                 4.7      7.5      8.5      8.9      9.2
──────────────────────────────────────────────────────────────────
/ascend status          below   good     target   target    10 ✓
```

**At Gen 2 (6 months): SENTINEL hits the /ascend target of 8.5 composite.**
**At Gen 4 (15 months): SENTINEL achieves 9.2 composite — exceeding Bloomberg (6.4) by +2.8 points at $0/yr.**

---

## The Asymmetric Bet

Bloomberg is a $10B/yr business with 355,000 seats at $31,980/yr. Disrupting them requires winning 1% of their users to generate $32M in equivalent value.

**SENTINEL doesn't need to charge anything.** The product earns its value by:
1. Saving the operator $30K–$80K/yr in terminal subscription fees
2. Generating alpha through leapfrog features (congressional trades, COT signals, regime-conditioned strategies) that paid terminals don't offer
3. Building community through the open-core model — contributors improve the platform, improving the alpha for everyone

The incumbents cannot respond:
- Bloomberg cannot give up seat fees: that's their $10B revenue stream
- AlphaSense cannot open-source their expert call corpus: that's their core IP
- FactSet cannot give away CUSIP licensing: they paid $1.925B for it

**SENTINEL wins by being what they structurally cannot become.**

---

*SENTINEL Leapfrog Playbook v1.0 — May 7, 2026*
*DanteForge /adversarial-score + /competitive-leapfrog*
*10 leapfrog dimensions | 8 adversarial attacks analyzed | 8-sprint execution plan*
