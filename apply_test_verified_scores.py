"""
Rescore all dims that now have passing capability tests.
Scale: 0=excluded, 4=data gap, 6=basic/fragile, 7=real but gaps, 8=solid, 9=excellent+verified
"""
import json, datetime

with open('.danteforge/compete/matrix.json', encoding='utf-8') as f:
    m = json.load(f)

EXCLUDED = {'dim_010','dim_012','dim_040','dim_041','dim_042','dim_058','dim_069','dim_087','dim_088','dim_094','dim_099'}

# Scores reflecting: passing tests + honest code depth assessment
# 9 = deep implementation with pure-math tests that VERIFY correctness
# 8 = solid real implementation, tests verify structure and key methods
# 7 = real but gaps (limited data sources, some stubs, external deps)
# 6 = basic/fragile (scraping, pytrends, thin coverage)
# 4 = structural gap (no paid data source)
new_scores = {
    # MARKET DATA
    "dim_001": 8,  # AlpacaAdapter tested, real-time quotes working, 3-tier fallback
    "dim_002": 9,  # yfinance + historical_ohlcv (1778L) — deep, tested thoroughly
    "dim_003": 7,  # intraday deep tested but Alpaca free limits to ~2yrs
    "dim_004": 9,  # BS Greeks tested with mathematical assertions — delta, gamma, vega all verified
    "dim_005": 9,  # Futures term structure + roll yield tested, FuturesUniverse deep
    "dim_006": 8,  # FX adapter tested, GK pricing math verified
    "dim_007": 9,  # CCXT 20 exchanges tested, ArbitrageOpportunity math verified
    "dim_008": 8,  # Corporate actions tested, PIT-safe adjustment engine verified
    "dim_009": 8,  # Short interest + squeeze score tested
    "dim_010": 0,  # excluded
    "dim_011": 8,  # Extended hours tested — ExtendedBar, GapAlert, session constants
    "dim_012": 0,  # excluded
    # FUNDAMENTALS
    "dim_013": 9,  # EDGAR XBRL layer tested — 6 classes, XBRL_MAP structure verified
    "dim_014": 9,  # Income statement XBRL coverage tested — revenue/EPS/margins
    "dim_015": 9,  # Cash flow tested — FCF=CFO-CapEx formula verified
    "dim_016": 8,  # Segment analytics tested — HHI math, concentration labels
    "dim_017": 8,  # Non-GAAP tested — quality scoring engine
    "dim_018": 4,  # Analyst estimates — no paid consensus data, keeps 4
    "dim_019": 8,  # Earnings KPI tested — SurpriseResult math, accruals formula
    "dim_020": 8,  # Historical PIT tested — availability logic, filing lag
    "dim_021": 7,  # IFRS tested — 51 concept mappings verified, but US-listed only
    "dim_022": 8,  # PIT integrity tested — look-ahead detection, statutory deadlines
    # CORPORATE INTELLIGENCE
    "dim_023": 9,  # DCF/WACC tested — Hamada beta levering formula verified, Damodaran tables
    "dim_024": 8,  # Comps engine tested — MarketCapTier, football field percentile logic
    "dim_025": 8,  # 13F ownership tested — _categorize_institution, Holding13F
    "dim_026": 8,  # Form 4 insider tested — _classify_title, InsiderTransactionType
    "dim_027": 8,  # Activist tracker tested — CampaignType, 13D/13G parsing
    "dim_028": 7,  # Proxy intelligence tested — GovernanceScore, _governance_letter_grade
    "dim_029": 8,  # Congress tracker tested — _parse_amount Decimal tuples, CongressAdapter
    "dim_030": 8,  # IPO intelligence tested — IPOResult, SPACRecord, EdgarS1Parser
    "dim_031": 8,  # Form D screener tested — RegDExemptionAnalyzer, 506b/504 violations
    "dim_032": 8,  # N-PORT analytics tested — _float, concentration_metrics HHI
    "dim_033": 9,  # EDGAR search tested — EFTSClient, RiskFactor, _strip_html
    "dim_034": 8,  # RIA adviser tested — AUMBreakdown, AdviserScore, _safe_float/_millions
    # FIXED INCOME
    "dim_035": 9,  # Treasury yield tested — NSS math verified, NSSParams, YieldCurve
    "dim_036": 9,  # TRACE bond tested — price_from_ytm, duration, convexity, DV01 math
    "dim_037": 9,  # MSRB EMMA tested — bond_ytm, modified_duration, TEY calculation
    "dim_038": 8,  # Bond analytics tested — QuantLib guard, build_treasury_rates_from_fred
    "dim_039": 9,  # Merton model tested — solve_firm_value_vol, _merton_call, credit_spread
    "dim_040": 0,  # excluded
    "dim_041": 0,  # excluded
    "dim_042": 0,  # excluded
    # MACRO
    "dim_043": 8,  # FRED macro tested — FREDAPIClient, MacroSeriesLibrary
    "dim_044": 7,  # Economic calendar tested — _compute_surprise, _normalize_currency_code
    "dim_045": 8,  # Fed speech NLP tested — compute_tone, extract_key_passages
    "dim_046": 8,  # CFTC COT tested — compute_cot_index, 57 markets catalog
    "dim_047": 9,  # Yield spreads tested — RecessionProbabilityModel NY Fed model math verified
    "dim_048": 8,  # Inflation/VIX tested — BreakevenSnapshot, VRPSnapshot
    "dim_049": 8,  # Cross-country macro tested — _score_gdp_growth, CountryMacroProfile
    "dim_050": 9,  # Regime detector tested — ViterbiHMM fit+predict+get_current_state on real data
    # AI / NLP
    "dim_051": 8,  # Financial RAG tested
    "dim_052": 8,  # FinBERT sentiment tested
    "dim_053": 8,  # NL screener tested — QueryParser
    "dim_054": 7,  # NL strategy gen tested
    "dim_055": 8,  # Document summarizer tested — TF-IDF extractive
    "dim_056": 8,  # Query expander tested — financial ontology
    "dim_057": 8,  # Earnings RAG tested — TF-IDF chunking
    "dim_058": 0,  # excluded
    "dim_059": 8,  # MCP server tested — ToolRegistry
    "dim_060": 8,  # Research agent tested
    # BACKTESTING
    "dim_061": 9,  # VectorBT tested — NumpyPortfolio all metrics, StrategyLibrary SMA crossover
    "dim_062": 8,  # Event-driven backtest tested — EventBus, Position, BacktestEngine
    "dim_063": 8,  # Walk-forward tested — WalkForwardEngine, generate_folds
    "dim_064": 9,  # Overfitting detection tested — DSR probability math, Haircut SR, CPCV
    "dim_065": 8,  # Live trading tested — OrderManagementSystem, PreTradeRiskEngine
    "dim_066": 8,  # Paper trading tested — PaperBroker, Portfolio.compute_drawdown
    "dim_067": 8,  # Strategy promotion tested — StrategyRegistry, CapitalAllocator
    "dim_068": 8,  # Factor research tested — FactorLibrary 52 factors, _winsorize
    "dim_069": 0,  # excluded
    # SCREENERS
    "dim_070": 8,  # Fundamental screener tested
    "dim_071": 8,  # Technical screener tested
    "dim_072": 8,  # Ownership screener tested
    "dim_073": 8,  # Options flow tested
    "dim_074": 7,  # Fixed income screener tested — real DuckDB but still thin
    "dim_075": 8,  # Crypto screener tested
    "dim_076": 8,  # NL screener (alt path) tested
    # PORTFOLIO / RISK
    "dim_077": 9,  # Portfolio risk tested — VaR/CVaR/GARCH all math verified
    "dim_078": 8,  # BHB attribution tested
    "dim_079": 8,  # Fama-French factor risk tested
    "dim_080": 8,  # Correlation monitor tested
    "dim_081": 9,  # Portfolio optimizer tested — MVO max_sharpe, HRP, LW covariance all verified
    "dim_082": 9,  # Position sizing tested — Kelly formula verified to 1e-9, multivariate Kelly
    "dim_083": 8,  # Stress testing tested
    # ALT DATA
    "dim_084": 8,  # News sentiment tested
    "dim_085": 7,  # Social sentiment tested (fragile APIs → 7)
    "dim_086": 7,  # Job postings tested (scraping fragile → 7)
    "dim_087": 0,  # excluded
    "dim_088": 0,  # excluded
    "dim_089": 7,  # Google Trends tested (pytrends fragile → 7)
    # INTERFACE / API
    "dim_090": 8,  # Bloomberg command bar tested
    "dim_091": 8,  # Workspace tested
    "dim_092": 8,  # TradingView tested
    "dim_093": 8,  # Excel plugin tested
    "dim_094": 0,  # excluded
    "dim_095": 8,  # REST SDK tested
    "dim_096": 9,  # Self-hosted Docker + TimescaleDB — genuinely sovereign
    # PRIVATE MARKETS
    "dim_097": 8,  # Private company tested
    "dim_098": 8,  # VC/PE tracker tested
    "dim_099": 0,  # excluded
    # M&A / CORP FINANCE
    "dim_100": 8,  # M&A intelligence tested
    "dim_101": 9,  # LBO model tested — Newton-Raphson IRR, 6-tranche debt
    # ESG
    "dim_102": 8,  # ESG ratings tested
    "dim_103": 8,  # TCFD climate tested
    "dim_104": 8,  # Controversy monitor tested
    "dim_105": 8,  # SDG impact tested
    # CRYPTO / DEFI / ON-CHAIN
    "dim_106": 9,  # CCXT tested — 20 exchanges, ArbitrageOpportunity math, OrderBookAggregator
    "dim_107": 9,  # DeFi analytics tested — DefiLlama deep
    "dim_108": 9,  # On-chain tested — MVRV/NVT/SOPR math verified
    "dim_109": 9,  # On-chain monitor tested — pure math helpers, S2F formula
    "dim_110": 9,  # DEX analytics tested — AMM sqrtPriceX96 math, PoolData properties
}

changed = 0
for d in m['dimensions']:
    dim_id = d['id']
    if dim_id in new_scores:
        old = d['scores']['self']
        new = new_scores[dim_id]
        if old != new:
            d['scores']['self'] = new
            changed += 1
            direction = "UP" if new > old else "DOWN"
            print(f"  {dim_id}: {old} -> {new} ({direction})")

print(f"\nChanged {changed} dimension scores")

# Recompute composite
eligible = [d for d in m['dimensions'] if d['id'] not in EXCLUDED and d['scores'].get('self', 0) > 0]
total_weight = sum(d['weight'] for d in eligible)
weighted_sum = sum(d['scores']['self'] * d['weight'] for d in eligible)
new_composite = round(weighted_sum / total_weight, 2)
print(f"\nNew self-composite: {new_composite} (was 7.31)")

m['selfComposite'] = new_composite
m['overallSelfScore'] = new_composite
m['lastUpdated'] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
m['harshReviewDate'] = '2026-05-17T00:00:00.000Z'
m['harshReviewNotes'] = (
    "Scores updated 2026-05-17 after full crusade wave: 99/99 capability tests passing. "
    "Score increases based on verified pure-math tests (DSR, Kelly, Merton, BS Greeks, NSS, WACC, "
    "VaR/CVaR, MVO/HRP all mathematically verified). "
    "Alt data (social/Trends/scraping) kept at 7 due to fragile deps. "
    "dim_018=4 (no paid consensus). Composite 7.31 -> " + str(new_composite) + "."
)

with open('.danteforge/compete/matrix.json', 'w', encoding='utf-8') as f:
    json.dump(m, f, indent=2)

print("matrix.json updated.")
