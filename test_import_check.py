"""Check which imports succeed/fail."""
import sys, os
sys.path.insert(0, os.getcwd())

imports_to_check = [
    ("sentinel.sfe.fundamental_screener_v3", "FundamentalScreenerV3"),
    ("sentinel.sfe.charting_v3", "ChartingV3"),
    ("sentinel.sfe.historical_financials_engine", "HistoricalFinancialsEngine"),
    ("sentinel.sfe.comps_engine_v3", "CompsEngineV3"),
    ("sentinel.sfe.earnings_kpi_tracker_v3", "EarningsKPITrackerV3"),
    ("sentinel.sfe.dividend_ddm", "DividendDDM"),
    ("sentinel.sfe.cash_flow_enhanced", "CashFlowEnhanced"),
    ("sentinel.sfe.peer_comparison", "PeerComparison"),
    ("sentinel.sfe.proxy_intelligence_v3", "ProxyIntelligenceV3"),
    ("sentinel.sfe.institutional_ownership_v3", "InstitutionalOwnershipV3"),
    ("sentinel.sma.economic_calendar_v3", "EconomicCalendarV3"),
    ("sentinel.sfe.analyst_estimates", "AnalystEstimates"),
    ("sentinel.sfe.dcf_wacc_v3", "DCFValuationEngine"),
    ("sentinel.sfe.credit_spread_v3", "CreditRiskEngine"),
    ("sentinel.sfe.bond_analytics_v3", "BondAnalyticsV3"),
    ("sentinel.sfe.credit_spread_analysis", "CreditSpreadAnalysis"),
    ("sentinel.sfe.fx_surface_v3", "FXSurfaceV3"),
    ("sentinel.sfe.yield_curve", "YieldCurve"),
    ("sentinel.sfe.fixed_income_screener_v3", "FixedIncomeScreenerV3"),
    ("sentinel.sfe.trace_bond_v3", "TraceBondV3"),
    ("sentinel.sfe.municipal_bond_v3", "MunicipalBondV3"),
    ("sentinel.sfe.esg_ratings_v3", "ESGRatingsV3"),
    ("sentinel.sma.global_macro_v3", "GlobalMacroV3"),
    ("sentinel.sma.yield_curve_analytics", "YieldCurveAnalytics"),
    ("sentinel.sma.commodity_analytics", "CommodityAnalytics"),
    ("sentinel.sbx.futures_term_structure", "FuturesTermStructure"),
    ("sentinel.sma.econ_forecasting", "EconForecasting"),
    ("sentinel.sai.research_agent_v3", "ResearchAgentV3"),
    ("sentinel.see.portfolio_risk_v3", "PortfolioRiskV3"),
    ("sentinel.sbx.risk_analytics", "RiskAnalytics"),
    ("sentinel.sbx.stress_testing", "StressTesting"),
    ("sentinel.sbx.portfolio_optimizer", "PortfolioOptimizer"),
    ("sentinel.see.correlation_monitor_v3", "CorrelationMonitorV3"),
    ("sentinel.sbx.multifactor_risk_model", "MultifactorRiskModel"),
    ("sentinel.sfe.options_flow_v3", "OptionsFlowV3"),
    ("sentinel.sfe.options_analytics", "OptionsAnalytics"),
    ("sentinel.snm.news_feed", "NewsFeed"),
    ("sentinel.sai.document_summarizer_v3", "DocumentSummarizerV3"),
    ("sentinel.sai.rag_engine_v2", "RAGEngineV2"),
    ("sentinel.api.realtime_quotes_v3", "RealtimeQuotesV3"),
    ("sentinel.sma.social_sentiment_v3", "SocialSentimentV3"),
    ("sentinel.sma.cftc_cot_v2", "CFTCCotV2"),
    ("sentinel.sfe.short_interest", "ShortInterest"),
    ("sentinel.sfe.insider_analytics", "InsiderAnalytics"),
    ("sentinel.sfe.activist_tracker_v3", "ActivistTrackerV3"),
    ("sentinel.sfe.edgar_search_v2", "EdgarSearchV2"),
    ("sentinel.sfe.ownership_screener_v3", "OwnershipScreenerV3"),
    ("sentinel.sfe.onchain_metrics_v2", "OnchainMetricsV2"),
    ("sentinel.sfe.defi_analytics_v2", "DeFiAnalyticsV2"),
    ("sentinel.sfe.ccxt_multi_exchange", "CCXTMultiExchange"),
    ("sentinel.sfe.dex_amm_analytics", "DEXAMMAnalytics"),
    ("sentinel.sbx.event_driven_backtest_v3", "EventDrivenBacktestV3"),
    ("sentinel.sbx.strategy_promotion_v3", "StrategyPromotionV3"),
    ("sentinel.sbx.portfolio_attribution", "PortfolioAttribution"),
    ("sentinel.sbx.technical_screener_enhanced", "TechnicalScreenerEnhanced"),
    ("sentinel.sbx.walk_forward_validator", "WalkForwardValidator"),
    ("sentinel.sbx.vol_term_structure", "VolTermStructure"),
    ("sentinel.sbx.options_analytics", "OptionsAnalytics"),
    ("sentinel.sfe.etf_analytics", "ETFAnalytics"),
]

ok = []
fail = []
for mod, cls in imports_to_check:
    try:
        m = __import__(mod, fromlist=[cls])
        c = getattr(m, cls, None)
        if c:
            ok.append((mod, cls))
        else:
            fail.append((mod, cls, "class not found"))
    except Exception as e:
        fail.append((mod, cls, str(e)[:60]))

print(f"\n=== OK ({len(ok)}) ===")
for mod, cls in ok:
    print(f"  {cls} from {mod}")

print(f"\n=== FAIL ({len(fail)}) ===")
for mod, cls, err in fail:
    print(f"  {cls} from {mod}: {err}")
