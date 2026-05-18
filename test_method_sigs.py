"""Check method signatures for key classes."""
import sys, os, ast
sys.path.insert(0, os.getcwd())

classes_to_inspect = [
    ("sentinel/sfe/comps_engine_v3.py", "CompsEngine"),
    ("sentinel/sfe/cash_flow_enhanced.py", "UniversalCashFlowParser"),
    ("sentinel/sfe/institutional_ownership_v3.py", "OwnershipAnalytics"),
    ("sentinel/sma/economic_calendar_v3.py", "CalendarResponse"),
    ("sentinel/sfe/bond_analytics_v3.py", "BondPricer"),
    ("sentinel/sfe/fixed_income_screener_v3.py", "FIScreenerService"),
    ("sentinel/sfe/trace_bond_v3.py", "BondPricer"),
    ("sentinel/sfe/municipal_bond_v3.py", "MuniService"),
    ("sentinel/sfe/esg_ratings_v3.py", "ESGCompositeEngine"),
    ("sentinel/sma/global_macro_v3.py", "GlobalMacroDashboard"),
    ("sentinel/sma/commodity_analytics.py", "CommodityDashboard"),
    ("sentinel/spm/portfolio_risk_v3.py", "PortfolioRiskEngine"),
    ("sentinel/sbx/stress_testing.py", "HistoricalScenarioEngine"),
    ("sentinel/spm/correlation_monitor_v3.py", "CorrelationMonitorEngine"),
    ("sentinel/sfe/options_flow_v3.py", "OptionsFlowEngine"),
    ("sentinel/sfe/options_analytics.py", "OptionsSignalEngine"),
    ("sentinel/snm/news_feed.py", "NewsFeedAggregator"),
    ("sentinel/sai/document_summarizer_v3.py", "DocumentSummarizationEngine"),
    ("sentinel/sai/rag_engine_v2.py", "FinancialRAGQueryEngine"),
    ("sentinel/api/realtime_quotes_v3.py", "QuoteFeedManager"),
    ("sentinel/sma/social_sentiment_v3.py", "SentimentAggregator"),
    ("sentinel/sma/cftc_cot_v3.py", "COTEngine"),
    ("sentinel/sfe/short_interest.py", "ShortInterestAnalyzer"),
    ("sentinel/sfe/insider_analytics.py", "InsiderSignalEngine"),
    ("sentinel/sfe/activist_tracker_v3.py", "ActivistScreener"),
    ("sentinel/sfe/edgar_search_v2.py", "AdvancedEFTSSearcher"),
    ("sentinel/sfe/onchain_metrics_v2.py", "MvrvCalculator"),
    ("sentinel/sfe/onchain_metrics_v2.py", "NvtCalculator"),
    ("sentinel/sfe/defi_analytics_v2.py", "TVLAnalytics"),
    ("sentinel/sfe/ccxt_multi_exchange.py", "CrossExchangeArbitrageDetector"),
    ("sentinel/sbx/event_driven_backtest_v3.py", "BacktestEngine"),
    ("sentinel/sbx/portfolio_attribution.py", "PortfolioAttributor"),
    ("sentinel/sbx/technical_screener_enhanced.py", "TechnicalScreener"),
    ("sentinel/spm/portfolio_optimizer_v3.py", "PortfolioOptimizerEngine"),
    ("sentinel/spm/factor_risk_v3.py", "PortfolioFactorAnalyzer"),
    ("sentinel/sfe/fundamental_screener_v3.py", "FundamentalScreener"),
    ("sentinel/sfe/treasury_yield_v3.py", "TreasuryYieldEngine"),
    ("sentinel/sfe/treasury_yield_v3.py", "FREDYieldLoader"),
]

for f, cls_name in classes_to_inspect:
    if not os.path.exists(f):
        print(f"  [MISSING] {f}")
        continue
    try:
        try:
            with open(f, encoding='utf-8') as fh:
                content = fh.read()
        except UnicodeDecodeError:
            with open(f, encoding='latin-1') as fh:
                content = fh.read()
        tree = ast.parse(content)
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef) and n.name == cls_name:
                meths = []
                for m in ast.walk(n):
                    if isinstance(m, ast.FunctionDef):
                        # Get arg names
                        args = [a.arg for a in m.args.args if a.arg != 'self']
                        meths.append(f"{m.name}({', '.join(args[:4])})")
                print(f"  {cls_name}: {meths[:10]}")
                break
    except Exception as e:
        print(f"  {f}/{cls_name}: ERROR {e}")
