"""Apply honest self-scores based on direct code inspection + background audit (2026-05-17)."""
import json
import datetime

with open('.danteforge/compete/matrix.json', encoding='utf-8') as f:
    m = json.load(f)

# Honest scores based on direct code inspection + background agent audit
# Scale: 0=excluded, 4=data gap, 6=basic/fragile, 7=real but gaps, 8=solid, 9=excellent
EXCLUDED = {'dim_010','dim_012','dim_040','dim_041','dim_042','dim_058','dim_069','dim_087','dim_088','dim_094','dim_099'}

honest_scores = {
    # MARKET DATA
    "dim_001": 7,  # Alpaca real-time adapter thin (182L, no retry), realtime_quotes_v3 (1394L) 3-tier
    "dim_002": 8,  # yfinance (354L, tenacity, validation) + historical_ohlcv_deep (1778L)
    "dim_003": 6,  # intraday_deep.py (945L) but Alpaca free limits intraday to ~2yrs
    "dim_004": 7,  # options_analytics.py (963L) deep BS math, yfinance options chain
    "dim_005": 8,  # futures_term_structure.py (605L) + futures_v3.py (1869L)
    "dim_006": 7,  # fx_adapter.py (303L) + fx_surface_v3.py — decent but thin real-time
    "dim_007": 8,  # ccxt_multi_exchange_v3.py (2067L) — real CCXT, 100+ venues, deep
    "dim_008": 8,  # corporate_actions_v3.py (1603L, 7 cls) — solid PIT-safe adjustment engine
    "dim_009": 7,  # short_interest_v3.py (1335L) + adapter (387L). FINRA public data.
    "dim_010": 0,  # excluded
    "dim_011": 8,  # extended_hours_v3.py (1522L, 11 cls) — solid pre/post market
    "dim_012": 0,  # excluded
    # FUNDAMENTALS
    "dim_013": 8,  # fundamental_data_layer_v3.py (EDGAR XBRL, 1626L, ThreadPool, 24h cache)
    "dim_014": 8,  # same EDGAR XBRL infrastructure
    "dim_015": 8,  # same, cash flow well covered
    "dim_016": 7,  # segment_analytics_v3.py — EDGAR segment parsing is limited
    "dim_017": 7,  # non_gaap_v3.py — non-GAAP parsing inherently incomplete
    "dim_018": 4,  # analyst_estimates.py but NO paid data source. Correctly low.
    "dim_019": 7,  # earnings_kpi_tracker_v3.py + earnings_surprise.py — EDGAR-based
    "dim_020": 7,  # historical_pit_v3.py + PIT enforcement in data routes
    "dim_021": 6,  # ifrs_financials_v3.py — international coverage via EDGAR only (US-listed)
    "dim_022": 8,  # pit_integrity_v3.py + data route PIT enforcement. Well implemented.
    # CORPORATE INTELLIGENCE
    "dim_023": 8,  # dcf_wacc_v3.py (1735L, 12 cls, 40 fn) — real DCF/WACC templates
    "dim_024": 7,  # comps_engine_v3.py — peer comparison via EDGAR/yfinance
    "dim_025": 7,  # institutional_ownership_v3.py + form13f_parser.py — EDGAR 13F
    "dim_026": 7,  # form4_parser.py + insider_v3.py — EDGAR Form 4
    "dim_027": 7,  # activist_tracker_v3.py + activist_adapter.py (444L) — EDGAR 13D/13G
    "dim_028": 6,  # proxy_intelligence_v3.py — DEF 14A parsing hard, likely incomplete
    "dim_029": 7,  # congress_adapter.py + congressional.py — real eFD/PTR API (2 politicians hardcoded)
    "dim_030": 7,  # ipo_intelligence_v3.py — EDGAR S-1 based
    "dim_031": 7,  # form_d_screener_v3.py + private_company_v3.py — EDGAR Form D
    "dim_032": 7,  # nport_analytics_v3.py — EDGAR N-PORT parsing
    "dim_033": 8,  # edgar_search_v3.py (1754L) — real EFTS full-text search, deep
    "dim_034": 7,  # ria_adviser_v3.py — EDGAR Form ADV
    # FIXED INCOME
    "dim_035": 9,  # treasury_yield_v3.py (2277L) — NSS fitting, FRED, forwards, cross-currency
    "dim_036": 7,  # trace_bond_v3.py + trace_bond_pricer.py (1010L) — FINRA TRACE
    "dim_037": 7,  # msrb_emma_adapter.py (382L) + municipal_bond_v3.py — MSRB EMMA
    "dim_038": 6,  # bond_analytics.py (165L) real QuantLib math BUT no import guard, fragile
    "dim_039": 8,  # credit_spread_v3.py — full Merton 1974, KMV DD, FRED OAS
    "dim_040": 0,  # excluded
    "dim_041": 0,  # excluded
    "dim_042": 0,  # excluded
    # MACRO
    "dim_043": 8,  # fred_macro_v3.py (1668L) + fred_macro_enhanced.py (1368L) — 50+ series
    "dim_044": 6,  # economic_calendar_v3.py (1875L, 13 stubs) — calendar data fragile
    "dim_045": 7,  # fed_speech_nlp.py (1176L, 36 fn) — NLP on Fed speeches via EDGAR EFTS
    "dim_046": 8,  # cftc_cot_v3.py (1921L) — real CFTC.gov downloads, deep
    "dim_047": 8,  # yield_spread_v3.py + yield_curve_analytics.py (1538L) — deep
    "dim_048": 8,  # inflation_vix_analytics.py (1295L) — FRED TIPS, VIX series
    "dim_049": 7,  # macro_cross_country.py (1110L) — World Bank, IMF, FRED
    "dim_050": 8,  # regime_detector_v3.py (1587L) — own Baum-Welch EM implementation
    # AI / NLP
    "dim_051": 8,  # financial_rag_v3.py (1992L) — 3-tier vector store, SEC chunking
    "dim_052": 8,  # finbert_sentiment_v3.py (2130L) — LM lexicon, GDELT, graceful degrade
    "dim_053": 7,  # nl_screener_v3.py (1790L) — rule-based + Claude optional
    "dim_054": 7,  # nl_strategy_generator_v3.py (1784L, 21 cls, 0 stubs)
    "dim_055": 7,  # document_summarizer_v3.py — real summarization pipeline
    "dim_056": 7,  # query_expander_v3.py — synonym expansion
    "dim_057": 7,  # earnings_rag_v3.py (2049L, 5 stubs) — EDGAR earnings corpus RAG
    "dim_058": 0,  # excluded
    "dim_059": 8,  # mcp_server_v3.py (2141L, 101 fn) — fallback patterns, solid
    "dim_060": 7,  # research_agent_v3.py (2220L, 22 tools, ReAct) — needs ANTHROPIC_API_KEY
    # BACKTESTING
    "dim_061": 8,  # vectorbt_backtest_v3.py (2365L) — real VBT + numpy fallback
    "dim_062": 8,  # event_driven_backtest_v3.py (1818L, 0 stubs) — NautilusTrader
    "dim_063": 8,  # walk_forward_v3.py (1854L) — full WF with MC permutation tests
    "dim_064": 8,  # overfitting_detection_v3.py (1983L, 0 stubs) — DSR+PBO+CPCV, academic
    "dim_065": 8,  # live_trading_v3.py (2582L, 0 stubs) — Alpaca live execution
    "dim_066": 7,  # paper_trading_v3.py (1566L, 1 stub)
    "dim_067": 7,  # strategy_promotion_v3.py (1784L, 5 stubs)
    "dim_068": 7,  # factor_research_v3.py (2490L, 123 fn, 12 stubs)
    "dim_069": 0,  # excluded
    # SCREENERS
    "dim_070": 8,  # fundamental_screener.py (1198L, 5/5 real indicators, 0 stubs)
    "dim_071": 8,  # technical_screener.py (1277L, 0 stubs, 4/5 real)
    "dim_072": 7,  # ownership_screener.py (1041L, 0 stubs)
    "dim_073": 7,  # options_flow.py (826L, 1 stub)
    "dim_074": 6,  # fixed_income_screener in sse thin (260L) — limited depth
    "dim_075": 7,  # crypto_screener.py (1187L, 2 stubs)
    "dim_076": 7,  # nl_screener_v3.py (same engine as dim_053)
    # PORTFOLIO / RISK
    "dim_077": 8,  # portfolio_risk_v3.py (1592L) + garch_var.py — VaR/CVaR solid
    "dim_078": 7,  # bhb_attribution_v2.py (2251L, 7 stubs) — BHB attribution
    "dim_079": 8,  # factor_risk_v3.py (1785L) — real French data library
    "dim_080": 7,  # correlation_monitor_v3.py (1746L, 3 stubs)
    "dim_081": 8,  # portfolio_optimizer_v3.py (2099L, 9 optimization methods, deep)
    "dim_082": 7,  # position_sizing_v3.py (1495L, 3 stubs)
    "dim_083": 7,  # stress_testing_v3.py (1499L, 1 stub) — pre-coded scenarios
    # ALT DATA
    "dim_084": 8,  # news_sentiment_pipeline_v3.py (2199L) — GDELT + FinBERT
    "dim_085": 6,  # social_sentiment_v3.py (1500L) — Reddit/StockTwits fragile APIs
    "dim_086": 6,  # job_postings_v3.py (1774L) — BLS solid, LinkedIn/Indeed scraping fragile
    "dim_087": 0,  # excluded
    "dim_088": 0,  # excluded
    "dim_089": 6,  # google_trends_v3.py (1539L) — pytrends known to break frequently
    # INTERFACE / API
    "dim_090": 8,  # bloomberg_bar_v3.py (2531L, 135 fn) — comprehensive command bar
    "dim_091": 7,  # workspace_v3.py (1930L) — Dash-based multi-panel, requires Dash
    "dim_092": 7,  # tradingview_enhanced.py (1068L) + charting_v3.py
    "dim_093": 7,  # excel_sheets_plugin_v3.py (1971L, 7 stubs)
    "dim_094": 0,  # excluded
    "dim_095": 8,  # rest_sdk_v3.py (2187L, 28 cls, 114 fn) — comprehensive REST + WS
    "dim_096": 9,  # Self-hosted Docker + TimescaleDB + Makefile — genuinely sovereign
    # PRIVATE MARKETS
    "dim_097": 7,  # private_company_v3.py + form_d_screener_v3.py — EDGAR Form D
    "dim_098": 7,  # vcpe_tracker_v3.py — VC/PE tracking from public filings
    "dim_099": 0,  # excluded
    # M&A / CORP FINANCE
    "dim_100": 7,  # ma_intelligence_v3.py (2303L, 8 stubs) — M&A deal tracker
    "dim_101": 8,  # lbo_model_v3.py (2406L, 0 stubs) — Newton-Raphson IRR, 6-tranche debt
    # ESG
    "dim_102": 7,  # esg_ratings_v3.py + esg_composite.py — ESG from EDGAR disclosures
    "dim_103": 7,  # tcfd_climate_v3.py + climate_disclosure_parser.py
    "dim_104": 7,  # controversy_monitor_v3.py (1542L, 0 stubs) — GDELT-based
    "dim_105": 7,  # sdg_impact_v3.py — UN SDG scoring from disclosures
    # CRYPTO / DEFI / ON-CHAIN
    "dim_106": 8,  # ccxt_multi_exchange_v3.py (2067L) — deep, real CCXT execution
    "dim_107": 8,  # defi_analytics_v3.py (2260L, 4 stubs) — DefiLlama, deep
    "dim_108": 8,  # onchain_metrics_v3.py (2195L, 1 stub) — MVRV/NVT/SOPR
    "dim_109": 7,  # onchain_monitor_v3.py (1836L, 75 fn) — event monitoring
    "dim_110": 7,  # dex_analytics_v3.py + dex_amm_analytics.py — DEX/AMM analytics
}

# Apply scores
changed = 0
for d in m['dimensions']:
    dim_id = d['id']
    if dim_id in honest_scores:
        old = d['scores']['self']
        new = honest_scores[dim_id]
        if old != new:
            d['scores']['self'] = new
            changed += 1
            print(f"  {dim_id}: {old} -> {new}")

print(f"\nChanged {changed} dimension scores")

# Recompute composite (weight-average over eligible dims with self > 0)
eligible = [d for d in m['dimensions'] if d['id'] not in EXCLUDED and d['scores'].get('self', 0) > 0]
total_weight = sum(d['weight'] for d in eligible)
weighted_sum = sum(d['scores']['self'] * d['weight'] for d in eligible)
new_composite = round(weighted_sum / total_weight, 2)
print(f"\nNew honest self-composite: {new_composite}")

m['selfComposite'] = new_composite
m['overallSelfScore'] = new_composite
m['lastUpdated'] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.000Z')
m['harshReviewDate'] = '2026-05-17T00:00:00.000Z'
m['harshReviewNotes'] = (
    "Scores revised via direct code inspection + background audit agent (2026-05-17). "
    "All inflated 9s reverted to honest assessment. Key: Core SDS/quant/NLP modules are DEEP. "
    "Alt data (social/Google Trends/scraping) fragile. Bond analytics missing QuantLib guard. "
    "Test coverage thin (3 files only). dim_018 correctly stays 4 (no paid consensus data source)."
)

with open('.danteforge/compete/matrix.json', 'w', encoding='utf-8') as f:
    json.dump(m, f, indent=2)

print("matrix.json updated successfully.")
