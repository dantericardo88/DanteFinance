"""Discover actual class names in modules that exist but have wrong class references."""
import sys, os, ast
sys.path.insert(0, os.getcwd())

modules_to_check = [
    "sentinel/sfe/fundamental_screener_v3.py",
    "sentinel/sfe/charting_v3.py",
    "sentinel/sfe/historical_financials_engine.py",
    "sentinel/sfe/comps_engine_v3.py",
    "sentinel/sfe/dividend_ddm.py",
    "sentinel/sfe/cash_flow_enhanced.py",
    "sentinel/sfe/peer_comparison.py",
    "sentinel/sfe/institutional_ownership_v3.py",
    "sentinel/sma/economic_calendar_v3.py",
    "sentinel/sfe/bond_analytics_v3.py",
    "sentinel/sfe/credit_spread_analysis.py",
    "sentinel/sfe/fx_surface_v3.py",
    "sentinel/sfe/fixed_income_screener_v3.py",
    "sentinel/sfe/trace_bond_v3.py",
    "sentinel/sfe/municipal_bond_v3.py",
    "sentinel/sfe/esg_ratings_v3.py",
    "sentinel/sma/global_macro_v3.py",
    "sentinel/sma/commodity_analytics.py",
    "sentinel/sma/econ_forecasting.py",
    "sentinel/spm/portfolio_risk_v3.py",
    "sentinel/sbx/stress_testing.py",
    "sentinel/spm/correlation_monitor_v3.py",
    "sentinel/sbx/multifactor_risk_model.py",
    "sentinel/sfe/options_flow_v3.py",
    "sentinel/sfe/options_analytics.py",
    "sentinel/snm/news_feed.py",
    "sentinel/sai/document_summarizer_v3.py",
    "sentinel/sai/rag_engine_v2.py",
    "sentinel/api/realtime_quotes_v3.py",
    "sentinel/sma/social_sentiment_v3.py",
    "sentinel/sma/cftc_cot_v2.py",
    "sentinel/sfe/short_interest.py",
    "sentinel/sfe/insider_analytics.py",
    "sentinel/sfe/activist_tracker_v3.py",
    "sentinel/sfe/edgar_search_v2.py",
    "sentinel/sfe/ownership_screener_v3.py",
    "sentinel/sfe/onchain_metrics_v2.py",
    "sentinel/sfe/defi_analytics_v2.py",
    "sentinel/sfe/ccxt_multi_exchange.py",
    "sentinel/sfe/dex_amm_analytics.py",
    "sentinel/sbx/event_driven_backtest_v3.py",
    "sentinel/sbx/strategy_promotion_v3.py",
    "sentinel/sbx/portfolio_attribution.py",
    "sentinel/sbx/technical_screener_enhanced.py",
    "sentinel/sbx/walk_forward_validator.py",
    "sentinel/sbx/options_analytics.py",
    "sentinel/sfe/etf_analytics.py",
    "sentinel/sfe/yield_curve.py",
]

for f in modules_to_check:
    if not os.path.exists(f):
        print(f"  [MISSING] {f}")
        continue
    try:
        with open(f, encoding='utf-8') as fh:
            content = fh.read()
    except UnicodeDecodeError:
        with open(f, encoding='latin-1') as fh:
            content = fh.read()
    try:
        tree = ast.parse(content)
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        # Get public methods for the last class (usually the engine/main class)
        methods = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef):
                meths = [m.name for m in ast.walk(n) if isinstance(m, ast.FunctionDef) and not m.name.startswith('_')]
                methods[n.name] = meths
        print(f"  {f}: classes={classes}")
        if classes:
            last = classes[-1]
            print(f"    Main class: {last}, methods={methods.get(last, [])[:8]}")
    except Exception as e:
        print(f"  {f}: ERROR {e}")
