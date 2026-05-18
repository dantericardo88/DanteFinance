"""
Apply harsh rescoring based on 4-agent direct code inspection (2026-05-18).
"""
import json, datetime

with open('.danteforge/compete/matrix.json', encoding='utf-8') as f:
    m = json.load(f)

EXCLUDED = {10, 12, 18, 40, 41, 42, 58, 69, 87, 88, 94, 99}

harsh_scores = {
    # MARKET DATA
    'dim_001': 9,   # realtime_quotes: multi-source fallback (Alpaca->Yahoo->Tradier) production-grade
    'dim_002': 9,   # historical_ohlcv: yfinance deep, 30yr+ verified
    'dim_003': 7,   # intraday: Alpaca free tier 2yr ceiling
    'dim_004': 9,   # options_analytics: 10 Greeks pure-Python verified
    'dim_005': 9,   # futures: 50+ contracts, roll logic solid
    'dim_006': 7,   # FX: EOD only, Frankfurter, no intraday/implied vols
    'dim_007': 9,   # CCXT: 20 exchanges, arb math verified
    'dim_008': 8,   # corporate_actions: EDGAR 8-K, M&A scoring shallow but solid
    'dim_009': 8,   # short_interest: FINRA real, squeeze score
    'dim_011': 8,   # extended_hours: ExtendedBar, GapAlert, session constants
    # FUNDAMENTALS
    'dim_013': 9,   # EDGAR XBRL: 6 classes, deep coverage verified
    'dim_014': 9,   # income_statement: XBRL revenue/EPS/margins verified
    'dim_015': 9,   # cash_flow: FCF=CFO-CapEx verified
    'dim_016': 8,   # segment_analytics: HHI, SIC peers, HTML fallback fragile
    'dim_017': 8,   # non_gaap: XBRL + 8-K parser, heuristic quality scoring
    'dim_018': 4,   # analyst_estimates: no paid consensus -- permanent ceiling
    'dim_019': 8,   # earnings_kpi: SurpriseResult math, accruals formula
    'dim_020': 8,   # historical_pit: availability logic, filing lag
    'dim_021': 7,   # IFRS: 51 concepts mapped, US-listed foreign only
    'dim_022': 8,   # pit_integrity: look-ahead detection, statutory deadlines
    # CORPORATE INTELLIGENCE
    'dim_023': 9,   # DCF/WACC: Hamada, Damodaran, Monte Carlo 10k sims verified
    'dim_024': 8,   # comps_engine: 20+ multiples, EDGAR direct, football-field
    'dim_025': 8,   # 13F_ownership: _categorize_institution verified
    'dim_026': 8,   # insider: Form 4 parsing, _classify_title verified
    'dim_027': 8,   # activist: CampaignType, 13D/13G parsing
    'dim_028': 7,   # proxy_intelligence: GovernanceScore, DEF14A inherently incomplete
    'dim_029': 8,   # congress_tracker: _parse_amount Decimal tuples
    'dim_030': 8,   # IPO_intelligence: IPOResult, SPACRecord, EdgarS1Parser
    'dim_031': 8,   # form_d_screener: RegDExemptionAnalyzer, 506b/504
    'dim_032': 8,   # nport_analytics: concentration_metrics HHI
    'dim_033': 9,   # EDGAR_search: EFTSClient, _strip_html verified
    'dim_034': 8,   # ria_adviser: AUMBreakdown, _safe_float verified
    # FIXED INCOME
    'dim_035': 9,   # treasury_yield: NSS fit+predict math verified, full curve suite
    'dim_036': 8,   # trace_bond: Newton-Raphson YTM/duration/convexity verified, no live TRACE stream
    'dim_037': 7,   # municipal_bond: 60-issuer preset, pure-Python duration, not dynamic
    'dim_038': 6,   # bond_analytics: thin QuantLib shim, no original code
    'dim_039': 9,   # credit_spread/Merton: solve_firm_value_vol verified
    # MACRO
    'dim_043': 8,   # FRED_macro: 765K+ series, composite LEI, revision tracking
    'dim_044': 6,   # econ_calendar: 13 stubs -- surprise index/intl CB not implemented
    'dim_045': 7,   # fed_speech_nlp: keyword lexicon only, no transformer
    'dim_046': 8,   # CFTC_COT: real COT index, 57 markets, 5 stubs in portfolio analytics
    'dim_047': 9,   # yield_spreads: NY Fed recession model verified
    'dim_048': 7,   # inflation_vix: 2 stubs, VIX analysis thin
    'dim_049': 7,   # cross_country_macro: G20 only, static, World Bank declared not deep
    'dim_050': 9,   # regime_detector: ViterbiHMM fit+predict+14 FRED signals verified
    # AI / NLP
    'dim_051': 7,   # financial_rag: 6 stubs, sqlite-vec promised not proven
    'dim_052': 7,   # finbert_sentiment: 5 stubs, heavy wrapper
    'dim_053': 7,   # nl_screener: 6 stubs, shallow NLU regex
    'dim_054': 7,   # nl_strategy_gen
    'dim_055': 8,   # document_summarizer: TF-IDF extractive
    'dim_056': 8,   # query_expander: financial ontology
    'dim_057': 8,   # earnings_rag: TF-IDF chunking
    'dim_059': 5,   # MCP server: 38 stubs, mostly API facades -- MAJOR downgrade
    'dim_060': 7,   # research_agent: 8 stubs, shallow tool orchestration
    # BACKTESTING
    'dim_061': 8,   # vectorbt_backtest: 2 stubs in secondary, numpy fallback solid
    'dim_062': 9,   # event_driven: ZERO stubs, 4 concrete strategies, production
    'dim_063': 8,   # walk_forward: 3 stubs in MC edge cases, core engine solid
    'dim_064': 9,   # overfitting_detection: DSR/PBO/CPCV/WhiteRC zero stubs verified
    'dim_065': 9,   # live_trading: ZERO stubs, Alpaca+TWAP+Almgren-Chriss
    'dim_066': 8,   # paper_trading: 1 stub, 5 order types, NYSE-calendar aware
    'dim_067': 8,   # strategy_promotion: StrategyRegistry, CapitalAllocator
    'dim_068': 8,   # factor_research: FactorLibrary 52 factors, _winsorize
    # SCREENERS
    'dim_070': 7,   # fundamental_screener: 1 stub, data source integration unclear
    'dim_071': 8,   # technical_screener: ZERO stubs, 32 methods
    'dim_072': 8,   # ownership_screener: 13F data
    'dim_073': 8,   # options_flow
    'dim_074': 7,   # fixed_income_screener: thin coverage, real DuckDB
    'dim_075': 8,   # crypto_screener: CCXT-backed
    'dim_076': 8,   # NL_screener_alt
    # PORTFOLIO / RISK
    'dim_077': 9,   # portfolio_risk: VaR/CVaR/GARCH all math verified
    'dim_078': 6,   # bhb_attribution: 7 stubs multi-period/FI/currency layers
    'dim_079': 8,   # factor_risk: Kenneth French integration, OLS model
    'dim_080': 7,   # correlation_monitor: 3 stubs, regime detection opaque
    'dim_081': 9,   # portfolio_optimizer: MVO max_sharpe/HRP/LW covariance verified
    'dim_082': 8,   # position_sizing: Kelly verified but 3 stubs in edge cases
    'dim_083': 7,   # stress_testing: linear shocks only, no tail risk amplification
    # ALT DATA
    'dim_084': 7,   # news_sentiment: 6 stubs in signal tuning, market reaction incomplete
    'dim_085': 8,   # social_sentiment: real Reddit/StockTwits/EDGAR, VADER built-in
    'dim_086': 8,   # job_postings: BLS JOLTS + FRED + Indeed + USAJobs
    'dim_089': 7,   # google_trends: pytrends fragile
    # INTERFACE / API
    'dim_090': 7,   # bloomberg_bar: 102 functions mapped, 3 handler stubs
    'dim_091': 7,   # workspace: multi-panel Dash/Rich, thin backend, export missing
    'dim_092': 9,   # tradingview: ZERO stubs, full UDF spec, production-ready
    'dim_093': 7,   # excel_plugin: RTD simulated (polling not push), COM missing
    'dim_095': 8,   # rest_sdk: 50+ endpoints, WebSocket, rate limiter
    'dim_096': 9,   # self-hosted Docker+TimescaleDB, sovereign
    # PRIVATE MARKETS
    'dim_097': 8,   # private_company: EDGAR EFTS, Form D, GDELT
    'dim_098': 6,   # vcpe_tracker: 80-fund hardcoded DB, 7 stubs
    # M&A / CORP FINANCE
    'dim_100': 8,   # ma_intelligence: deal FSM, merger arb, synergy NPV
    'dim_101': 9,   # lbo_model: ZERO stubs, 6-tranche waterfall, IRR verified
    # ESG
    'dim_102': 8,   # esg_ratings: CDP/EPA/OSHA multi-source, composite scoring
    'dim_103': 7,   # tcfd_climate: 4-pillar scoring, physical risk shallow
    'dim_104': 8,   # controversy_monitor: GDELT+EDGAR+EPA+OSHA zero stubs
    'dim_105': 8,   # sdg_impact: 450+ SIC mappings zero stubs
    # CRYPTO / DEFI / ON-CHAIN
    'dim_106': 8,   # CCXT: 20 exchanges tested, SOR incomplete
    'dim_107': 8,   # defi_analytics: DefiLlama deep, IL calc stub
    'dim_108': 9,   # onchain_metrics: MVRV/NVT/SOPR zero functional stubs verified
    'dim_109': 9,   # onchain_monitor: S2F formula pure math verified
    'dim_110': 8,   # dex_analytics: AMM sqrtPriceX96 verified, rugpull light
}

changes = []
for d in m['dimensions']:
    dim_id = d['id']
    if dim_id in harsh_scores:
        old = d['scores']['self']
        new = harsh_scores[dim_id]
        if old != new:
            direction = 'UP' if new > old else 'DOWN'
            changes.append((dim_id, old, new, direction))
            d['scores']['self'] = new

print(f"Changed {len(changes)} dims:")
for dim_id, old, new, direction in changes:
    print(f"  {dim_id}: {old} -> {new} ({direction})")

eligible = [d for d in m['dimensions']
            if int(d['id'].split('_')[1]) not in EXCLUDED and d['scores'].get('self', 0) > 0]
total_w = sum(d['weight'] for d in eligible)
ws = sum(d['scores']['self'] * d['weight'] for d in eligible)
new_comp = round(ws / total_w, 2)

old_comp = m['selfComposite']
print(f"\nComposite: {old_comp} -> {new_comp}")

m['selfComposite'] = new_comp
m['overallSelfScore'] = new_comp
m['lastUpdated'] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
m['harshReviewDate'] = '2026-05-18T00:00:00.000Z'
m['harshReviewNotes'] = (
    'Harsh rescore 2026-05-18: 4-agent direct code inspection of all 99 modules. '
    'Key downgrades: dim_059 MCP server (38 stubs->5), dim_078 BHB (7 stubs->6), '
    'dim_098 vcpe_tracker (hardcoded DB->6), dim_038 bond_analytics (thin shim->6), '
    'dim_044 econ_calendar (13 stubs->6), dim_037 muni_bond (preset universe->7). '
    'Key upgrades: dim_062 event_driven (zero stubs->9), dim_065 live_trading (zero stubs->9), '
    'dim_092 tradingview (UDF compliant->9), dim_001 realtime_quotes (multi-source->9). '
    f'Composite {old_comp} -> {new_comp}.'
)

with open('.danteforge/compete/matrix.json', 'w', encoding='utf-8') as f:
    json.dump(m, f, indent=2)

print("\nmatrix.json updated.")

# Summary by category
cats = {}
for d in m['dimensions']:
    if int(d['id'].split('_')[1]) not in EXCLUDED and d['scores'].get('self', 0) > 0:
        cat = d.get('category', 'other')
        cats.setdefault(cat, []).append(d['scores']['self'])

print("\nCategory averages:")
for cat, scores in sorted(cats.items()):
    print(f"  {cat}: avg={sum(scores)/len(scores):.2f} dims={len(scores)}")
