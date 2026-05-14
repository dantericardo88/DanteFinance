"""
SENTINEL MCP Server — LEAPFROG #59.

54-tool FastMCP server exposing the full SENTINEL data and analysis surface
to Claude and any MCP-compatible AI client. This is the native agentic interface
that Bloomberg, CapIQ, FactSet, and AlphaSense do not have.

Score: SENTINEL 10, all incumbents 0.

Tools:
  1. get_ohlcv                — Historical price bars
  2. get_quote                — Real-time quote
  3. get_fundamentals         — Financial facts (revenue, EPS, etc.)
  4. screen_stocks            — Natural-language stock screener
  5. get_filings              — Recent SEC filings for a company
  6. get_insider_trades       — Form 4 insider transactions
  7. get_institutional_holders — 13F institutional holdings
  8. get_congressional_trades — STOCK Act disclosures
  9. run_backtest             — Execute a backtest and return metrics
 10. get_macro_series         — FRED economic time series
 11. get_cot_signals          — CFTC COT positioning signals
 12. get_macro_regime         — Current HMM macro regime
 13. get_news_sentiment       — News with FinBERT sentiment
 14. get_options_chain        — Options chain with greeks
 15. explain_strategy         — AI explanation of strategy metrics + risk
 16. get_options_analytics    — IV surface, skew, GEX, max pain, put/call ratios
 17. query_documents          — Semantic RAG search over financial documents
 18. get_social_sentiment     — Reddit + StockTwits FinBERT sentiment
 19. get_economic_calendar    — Upcoming macro data releases
 20. screen_stocks_nl         — Full NL screener via SSE engine
 21. run_stress_test          — Portfolio stress tests vs historical scenarios
 22. get_factor_exposure      — Fama-French 5-factor + momentum decomposition
 23. run_dcf_valuation        — Damodaran DCF intrinsic value with sensitivity table
 24. get_global_macro         — G7 macro dashboard: GDP, inflation, rates, yield curves
 25. get_position_sizing      — Kelly Criterion, vol-target, risk-parity position sizes
 26. get_fx_rates             — ECB official FX rates via Frankfurter API (free)
 27. get_short_interest       — FINRA RegSHO short interest + squeeze candidate screen
 28. get_corporate_bonds      — FINRA TRACE corporate bond quotes + credit curve
 29. get_segment_breakdown    — EDGAR XBRL business segment revenue breakdown
 30. screen_options_flow      — Unusual options activity: volume spikes, IV, skew, GEX
 31. generate_trading_strategy — NL→StrategySpec via Claude tool-use (dim 54)
 32. research_ticker          — Autonomous multi-step research memo (dim 60)
 33. get_google_trends        — Google Trends momentum signal for tickers (dim 89)
 34. get_defi_dashboard       — DeFi TVL, top protocols, yields, stablecoins (dim 107)
 35. get_crypto_signals       — On-chain NVT, MVRV proxy, fear/greed (dim 108)
 36. get_earnings_kpis        — EDGAR MD&A KPI extraction via Claude (dim 19)
 37. get_esg_profile          — ESG proxy signals from EDGAR DEF14A + 10-K (dims 102-103)
 38. get_alpha_signals        — Congress + COT + insider composite alpha (dim 68)
 39. expand_search_query      — Lexical + LLM + HyDE query expansion (dim 56)
 40. ingest_news_to_rag       — Cross-ingest recent news into RAG document store (dim 57)
 41. get_non_gaap_metrics     — Non-GAAP metrics + GAAP reconciliation from 8-K (dim 17)
 42. get_comps_table          — Comparable company valuation table (dim 24)
 43. get_activist_positions   — EDGAR 13D/13G activist investor tracker (dim 27)
 44. search_edgar             — Full-text EDGAR EFTS search across all form types (dim 33)
 45. get_portfolio_var        — Historical/parametric/MC VaR + CVaR (dim 77)
 46. get_portfolio_attribution — Brinson-Hood-Beebower sector attribution (dim 78)
 47. get_controversy_signals  — News-based ESG controversy monitor (dim 104)
 48. get_dex_dashboard        — DEX protocol volume analytics via DeFiLlama (dim 110)
 49. optimize_portfolio       — Mean-variance / Black-Litterman / ERC optimizer
 50. screen_bonds             — Fixed income screener (yield, duration, credit quality)
 51. get_onchain_events       — Whale tx / TVL event monitor (Etherscan + DeFiLlama)
 52. clean_data               — Multi-source consensus DataCleaner → writes to DB
 53. get_continuous_futures   — Back-adjusted continuous futures series (Panama/Ratio/Unadj)
 54. get_ta_signals           — Technical analysis signals: RSI, MACD, BB, VWAP, stochastic (dim 69-76)
 55. get_yield_curve          — Live Treasury yield curve + Nelson-Siegel fit + forward rates (dim 44)
 56. synthesize_research      — Multi-source Claude research synthesis: bull/bear/risks/catalysts (dim 89)
 57. compute_garch_var        — GARCH(1,1) conditional VaR + CVaR + backtest + vol forecast (dim 77)
 58. get_advanced_ta          — Ichimoku, Fibonacci, ADX, Parabolic SAR (dims 73-76)
 59. compute_bond_price       — Duration, convexity, DV01, scenario analysis for a bond (dims 51-52)
 60. compute_altman_z         — Altman Z-score credit risk model for equities (dim 50)
 61. screen_ma_deals          — M&A deal flow screener via EDGAR 8-K/DEFM14A/SC TO-T (dim 25)
 62. get_ma_profile           — M&A target profile: activist interest, defense mechanisms (dim 25)
 63. get_etf_profile          — ETF holdings, factor exposure, flows, expense ratio (dim 56)
 64. compare_etfs             — Side-by-side ETF comparison table (dim 56)
 65. get_commodity_dashboard  — Commodity price dashboard: energy/metals/ag + regime (dim 48)
 66. get_ifrs_fundamentals    — IFRS financials for non-US 20-F SEC filers via XBRL (dim 21)
 67. get_econ_forecast        — AR/VAR macro forecasting on FRED series + nowcast index (dim 60)
 68. get_liquidity_metrics    — Amihud, Roll, Corwin-Schultz, Kyle's lambda, ADV (dim 79)
 69. get_portfolio_liquidity  — Portfolio-level liquidity + liquidation horizon (dim 79)
 70. check_alerts             — Poll active price/RSI/MA/volume alerts, return triggered (dim 97)
 71. create_alert             — Create persistent price/RSI/MA/volume alert (dim 97)
 72. list_alerts              — List all alerts optionally filtered by ticker (dim 97)
 73. delete_alert             — Delete an alert by ID (dim 97)
 74. get_ria_profile          — Form ADV RIA adviser profile: AUM, clients, fee structure (dim 34)
 75. screen_rias              — Screen RIAs by name/AUM/state from SEC IAPD (dim 34)
 76. get_country_risk         — GDELT geopolitical risk score + Claude narrative (dim 67)
 77. get_geopolitical_dashboard — Multi-country risk dashboard + global risk index (dim 67)
 78. run_lbo_model            — LBO model: IRR/MOIC/debt schedule from assumptions (dim 101)
 79. run_merger_model         — M&A accretion/dilution model (dim 101)
 80. screen_lbo_candidate     — Auto-populate LBO model from ticker financials (dim 101)
 81. get_recent_filings       — EDGAR real-time RSS: latest 8-K/10-K/13D/Form4 (dim 99)
 82. monitor_watchlist        — Monitor specific tickers for new EDGAR filings (dim 99)
 83. get_insider_transactions — Form 4 insider transactions for a ticker (dim 26)
 84. get_fx_pair              — Deep FX analytics: forward curve, vol, carry, momentum (dim 6)
 85. get_fx_dashboard         — Multi-currency FX dashboard + DXY proxy + USD trend (dim 6)
 86. get_company_form_d       — SEC Form D private company raise history (dim 97)
 87. screen_private_market    — Screen recent Form D filings: VC/PE/HF raises (dim 97)
 88. run_scenario             — Macro scenario P&L: factor betas × macro shocks (dim 80)
 89. run_multi_scenario       — All template scenarios vs portfolio (dim 80)
 90. get_dividend_analytics   — Full dividend analytics: yield/growth/quality/DDM (dim 28)
 91. screen_dividends         — Screen multiple tickers for dividend quality (dim 28)
 92. get_options_flow         — Options flow: unusual activity, vol/OI, max pain, IV skew (dim 15)
 93. screen_options_flow      — Multi-ticker unusual options activity screener (dim 15)
 94. get_credit_analytics     — Merton structural model: PD, CDS proxy, credit score (dim 50)
 95. screen_credit            — Credit quality screen across multiple tickers (dim 50)
 96. get_peer_comparison      — Auto peer comparison: valuation/growth/profitability rank (dim 24)
 97. analyze_earnings_filing  — 8-K earnings NLP: tone/guidance/themes via Claude Haiku (dim 19)
 98. get_earnings_trend       — Multi-quarter earnings sentiment trend (dim 19)
 99. get_analyst_estimates    — Analyst consensus proxy: targets, recs, EPS/rev estimates (dim 18)
100. screen_analyst_sentiment — Multi-ticker analyst sentiment screen: upside, buys, sells (dim 18)
101. screen_convertibles      — Convertible bond screen: parity/delta/verdict for tickers (dims 38/39)
102. analyze_convertible_bond — Full CB analytics: parity, premium, greeks, bond floor (dims 38/39)
103. get_squeeze_analytics    — Short-squeeze signals: DTC, borrow proxy, gamma risk, score (dim 9)
104. screen_squeeze_candidates — Multi-ticker squeeze screen: DTC, SI float %, verdict (dim 9)
105. get_governance_profile   — EDGAR DEF 14A governance score: board/duality/say-on-pay (dim 28)
106. screen_governance        — Multi-ticker governance quality screen (dim 28)
107. get_earnings_surprise    — EPS beat/miss history, surprise %, trend, next date (dim 19)
108. screen_earnings_beats    — Multi-ticker earnings beat rate screener (dim 19)
109. get_vix_analytics        — VIX term structure, VRP, vol regime, VVIX, SKEW (dims 48/50)
110. get_vol_regime           — Market vol regime + per-ticker realized vol vs implied (dims 48/50)
111. get_insider_signal       — Cluster buy, officer sentiment, net purchase ratio, score (dim 26)
112. screen_insider_buying    — Multi-ticker insider buying screen (dim 26)
113. get_supply_chain_risk    — EDGAR XBRL customer concentration, geo HHI, risk score (dim 16)
114. screen_concentration_risk — Multi-ticker supply chain concentration screen (dim 16)
115. get_sector_rotation      — SPDR ETF relative strength, momentum, rotation signal (dim 62/69)
116. screen_sector_strength   — Top/bottom sector screener by composite momentum score (dim 62/69)
117. get_earnings_quality     — EDGAR XBRL accruals, cash conversion, operating leverage (dim 17/19)
118. screen_earnings_quality  — Multi-ticker earnings quality screener (dim 17/19)
119. get_vol_term_structure   — Options IV term structure, forward vol, skew by strike (dim 15)
120. screen_vol_surface       — Multi-ticker vol surface: skew regime, term structure shape (dim 15)
121. get_gdp_nowcast          — FRED GDP nowcast: composite leading indicator, recession prob (dim 60)
122. get_macro_nowcast_dashboard — Full macro dashboard: expansion/contraction signals, regime (dim 60)
123. get_news_flow            — yfinance + EDGAR 8-K news: event type, sentiment, volume spike (dim 92)
124. screen_news_flow         — Multi-ticker news volume spike + material event screen (dim 92)
125. get_dividend_ddm         — Gordon/H-Model/3-stage DDM + sustainability score (dim 28)
126. screen_dividend_quality  — Dividend quality screener: aristocrats, kings, at-risk (dim 28)
127. get_credit_spread_profile — FINRA TRACE Z-spread, maturity curve, migration alert (dim 50/51)
128. screen_credit_spreads    — Multi-ticker credit spread screen: wide issuers, inverted curves (dim 50)
129. get_alt_data_dashboard   — FRED PCE + GDELT + Google Trends + shipping alt data (dim 94)
130. get_alt_signal_summary   — Flat alt data signal rows for dashboard display (dim 94)
"""
from __future__ import annotations
import asyncio
from datetime import datetime, date, timedelta
from typing import Optional, Any
from fastmcp import FastMCP
from sentinel.core.config import get_settings
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

mcp = FastMCP(
    name="SENTINEL",
    version="1.0.0",
    description="Sovereign AI-native financial terminal — replaces Bloomberg at $0/yr",
)


# ─── Tool 1: OHLCV ───────────────────────────────────────────────────────────

@mcp.tool()
async def get_ohlcv(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> dict:
    """
    Fetch historical OHLCV price bars for any asset (equities, ETFs, crypto, forex).

    Args:
        ticker: Symbol (e.g. 'AAPL', 'BTC-USD', 'BTC/USDT')
        start: Start date ISO format (YYYY-MM-DD)
        end: End date ISO format (YYYY-MM-DD)
        interval: Bar interval — 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo

    Returns:
        bars: list of {time, open, high, low, close, volume, source}
    """
    from sentinel.sds.normalizer import fetch_ohlcv_with_fallback
    from sentinel.sds import build_default_adapters, get_all_adapters

    if not get_all_adapters():
        build_default_adapters()

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    bars = await fetch_ohlcv_with_fallback(ticker, start_dt, end_dt, interval)

    return {
        "ticker": ticker,
        "interval": interval,
        "count": len(bars),
        "bars": [
            {
                "time": b.time.isoformat(),
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": float(b.volume),
                "source": b.source,
            }
            for b in bars
        ],
    }


# ─── Tool 2: Quote ────────────────────────────────────────────────────────────

@mcp.tool()
async def get_quote(ticker: str) -> dict:
    """
    Fetch the latest real-time quote for a US equity or crypto asset.

    Args:
        ticker: Symbol (e.g. 'MSFT', 'ETH/USDT')

    Returns:
        bid, ask, last, volume, timestamp, source
    """
    from sentinel.sds import build_default_adapters, get_all_adapters, get_adapter

    if not get_all_adapters():
        build_default_adapters()

    # Try Finnhub → Alpaca → yfinance info
    for name in ["finnhub", "alpaca", "polygon"]:
        adapter = get_adapter(name)
        if adapter is None:
            continue
        try:
            if name == "finnhub":
                data = await adapter.fetch_ticker(ticker)
                if data:
                    return {"ticker": ticker, "last": data.get("c"), "source": name, "raw": data}
            elif name == "alpaca":
                snap = await adapter.fetch_snapshot(ticker)
                if snap:
                    return {"ticker": ticker, **snap, "source": name}
            elif name == "polygon":
                snap = await adapter.fetch_snapshot(ticker)
                if snap:
                    return {"ticker": ticker, **snap, "source": name}
        except Exception:
            continue

    return {"ticker": ticker, "error": "No real-time quote available"}


# ─── Tool 3: Fundamentals ─────────────────────────────────────────────────────

@mcp.tool()
async def get_fundamentals(ticker: str, facts: Optional[list[str]] = None) -> dict:
    """
    Fetch financial fundamentals for a company from EDGAR XBRL filings.

    Args:
        ticker: US equity ticker (e.g. 'AAPL')
        facts: Optional list of fact labels to retrieve (e.g. ['revenue', 'net_income', 'eps_diluted']).
               If omitted, returns all available facts.

    Returns:
        cik, ticker, annual_facts, quarterly_facts
    """
    from sentinel.sds.adapters.edgar_adapter import EDGARAdapter
    from sentinel.sfe.xbrl_parser import extract_facts, get_annual_facts, get_quarterly_facts

    s = get_settings()
    edgar = EDGARAdapter(user_agent=s.edgar_user_agent)
    await edgar.load_company_tickers()
    cik = edgar.ticker_to_cik(ticker)
    if not cik:
        return {"ticker": ticker, "error": f"CIK not found for {ticker}"}

    companyfacts = await edgar.fetch_companyfacts(cik)
    all_facts = extract_facts(companyfacts, cik)

    result: dict[str, Any] = {"ticker": ticker, "cik": cik, "annual": {}, "quarterly": {}}
    labels = facts or list({f.label for f in all_facts})

    for label in labels:
        annual = get_annual_facts(all_facts, label)
        quarterly = get_quarterly_facts(all_facts, label)
        if annual:
            result["annual"][label] = [
                {"period_end": f.period_end.isoformat(), "value": float(f.value), "unit": f.unit}
                for f in annual[:8]
            ]
        if quarterly:
            result["quarterly"][label] = [
                {"period_end": f.period_end.isoformat(), "value": float(f.value), "unit": f.unit}
                for f in quarterly[:12]
            ]

    return result


# ─── Tool 4: Screener ─────────────────────────────────────────────────────────

@mcp.tool()
async def screen_stocks(query: str, limit: int = 20) -> dict:
    """
    Natural-language stock screener. Translates your query to financial criteria.

    Examples:
      - 'tech stocks with PE < 20 and revenue growth > 20%'
      - 'dividend aristocrats yielding over 3%'
      - 'small cap value stocks with low debt'
      - 'insider buying in the last 30 days'

    Args:
        query: Natural language screening criteria
        limit: Maximum number of results (default 20, max 100)

    Returns:
        results: list of matching stocks with key metrics
    """
    criteria = _parse_nl_screen(query)
    criteria["limit"] = min(int(limit), 100)
    logger.info("Screen query", query=query, criteria=criteria)

    import sentinel.api.routes.screen as _screen_route

    engine = _screen_route.get_engine()
    if engine.get_universe_count() == 0:
        try:
            from sentinel.sds.db import get_session_factory
            from sentinel.sse.db_loader import populate_screener_from_db
            factory = get_session_factory()
            async with factory() as session:
                await populate_screener_from_db(session)
            engine = _screen_route.get_engine()
        except Exception as exc:
            logger.warning("Could not populate screener from DB", error=str(exc))

    results = engine.screen(criteria)
    return {
        "query": query,
        "parsed_criteria": criteria,
        "count": len(results),
        "results": [
            {
                "ticker": r.ticker,
                "name": r.name,
                "sector": r.sector,
                "market_cap": float(r.market_cap) if r.market_cap else None,
                "pe_ratio": float(r.pe_ratio) if r.pe_ratio else None,
                "dividend_yield": float(r.dividend_yield) if r.dividend_yield else None,
                "revenue_growth_yoy": float(r.revenue_growth_yoy) if r.revenue_growth_yoy else None,
                "net_margin": float(r.net_margin) if r.net_margin else None,
                "roe": float(r.roe) if r.roe else None,
            }
            for r in results
        ],
    }


# ─── Tool 5: SEC Filings ──────────────────────────────────────────────────────

@mcp.tool()
async def get_filings(
    ticker: str,
    form_type: Optional[str] = None,
    limit: int = 10,
) -> dict:
    """
    Fetch recent SEC filings for a company.

    Args:
        ticker: US equity ticker (e.g. 'TSLA')
        form_type: Optional filter — '10-K', '10-Q', '8-K', 'SC 13G', 'DEF 14A', etc.
        limit: Max number of filings to return

    Returns:
        list of {form_type, filing_date, accession, primary_doc, cik}
    """
    from sentinel.sds.adapters.edgar_adapter import EDGARAdapter

    edgar = EDGARAdapter()
    await edgar.load_company_tickers()
    cik = edgar.ticker_to_cik(ticker)
    if not cik:
        return {"ticker": ticker, "error": f"CIK not found for {ticker}"}

    filings = await edgar.fetch_recent_filings(cik, form_type=form_type, limit=limit)
    return {"ticker": ticker, "cik": cik, "filings": filings}


# ─── Tool 6: Insider Trades ───────────────────────────────────────────────────

@mcp.tool()
async def get_insider_trades(ticker: str, limit: int = 20) -> dict:
    """
    Fetch recent Form 4 insider transactions for a company.

    Args:
        ticker: US equity ticker
        limit: Max number of transactions

    Returns:
        list of {owner_name, role, tx_date, tx_code, shares, value, signal}
    """
    from sentinel.sds.adapters.edgar_adapter import EDGARAdapter
    from sentinel.sfe.form4_parser import fetch_and_parse_form4

    s = get_settings()
    edgar = EDGARAdapter(user_agent=s.edgar_user_agent)
    await edgar.load_company_tickers()
    cik = edgar.ticker_to_cik(ticker)
    if not cik:
        return {"ticker": ticker, "error": f"CIK not found for {ticker}"}

    filings = await edgar.fetch_recent_filings(cik, form_type="4", limit=limit)
    transactions = []
    for filing in filings[:10]:
        acc = filing.get("accession")
        if acc:
            txs = await fetch_and_parse_form4(cik, acc, user_agent=s.edgar_user_agent)
            transactions.extend(txs)

    return {
        "ticker": ticker,
        "count": len(transactions),
        "transactions": [
            {
                "owner": t.owner_name,
                "role": t.role.value,
                "date": t.tx_date.isoformat(),
                "type": t.tx_code.value,
                "shares": float(t.shares),
                "value": float(t.value),
                "shares_after": float(t.shares_owned_after),
            }
            for t in transactions[:limit]
        ],
    }


# ─── Tool 7: Institutional Holders ────────────────────────────────────────────

@mcp.tool()
async def get_institutional_holders(ticker: str, limit: int = 25) -> dict:
    """
    Fetch top institutional holders from the most recent 13F filings.

    Args:
        ticker: US equity ticker
        limit: Max number of institutions

    Returns:
        list of {manager_cik, issuer, shares, market_value, change_signal}
    """
    return {
        "ticker": ticker,
        "message": "13F data requires SOD bulk ingestion pipeline. Use /sod endpoint.",
        "holders": [],
    }


# ─── Tool 8: Congressional Trades ─────────────────────────────────────────────

@mcp.tool()
async def get_congressional_trades(
    ticker: Optional[str] = None,
    lookback_days: int = 90,
    chamber: Optional[str] = None,
) -> dict:
    """
    Fetch STOCK Act congressional trade disclosures. Leapfrog feature — unique to SENTINEL.

    Args:
        ticker: Optional ticker filter (e.g. 'NVDA')
        lookback_days: How many days back to search (default 90)
        chamber: 'Senate', 'House', or None for both

    Returns:
        list of {politician, chamber, party, ticker, tx_date, direction, amount_range, signal_strength}
    """
    from sentinel.sod.congressional import CongressionalTradeTracker

    tracker = CongressionalTradeTracker()
    trades = await tracker.fetch_all_recent(lookback_days=lookback_days)

    if ticker:
        trades = [t for t in trades if t.ticker.upper() == ticker.upper()]
    if chamber:
        trades = [t for t in trades if t.chamber.lower() == chamber.lower()]

    signals = tracker.generate_signals()
    if ticker:
        signals = [s for s in signals if s["ticker"].upper() == ticker.upper()]

    return {
        "ticker": ticker,
        "lookback_days": lookback_days,
        "count": len(trades),
        "signals": signals[:50],
    }


# ─── Tool 9: Backtest ─────────────────────────────────────────────────────────

@mcp.tool()
async def run_backtest(
    ticker: str,
    strategy: str,
    start: str,
    end: str,
    params: Optional[dict] = None,
) -> dict:
    """
    Run a named backtest strategy and return full 24-metric report with DSR.

    Args:
        ticker: Asset to backtest (e.g. 'SPY')
        strategy: Strategy name — 'momentum', 'mean_reversion', 'ma_crossover', 'rsi'
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        params: Optional strategy parameters (e.g. {'fast': 10, 'slow': 50})

    Returns:
        BacktestMetrics with Sharpe, DSR, PBO, CAGR, max_drawdown, win_rate, etc.
    """
    from sentinel.sds.normalizer import fetch_ohlcv_with_fallback
    from sentinel.sds import build_default_adapters, get_all_adapters
    from sentinel.sbe.runner import VectorBTRunner
    from sentinel.sbe.strategies import get_strategy_fn

    if not get_all_adapters():
        build_default_adapters()

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    bars = await fetch_ohlcv_with_fallback(ticker, start_dt, end_dt)
    if not bars:
        return {"error": f"No price data for {ticker}"}

    import pandas as pd
    prices = pd.Series(
        [float(b.close) for b in bars],
        index=pd.DatetimeIndex([b.time for b in bars]),
        name=ticker,
    )

    params = params or {}
    try:
        signal_fn = get_strategy_fn(strategy, **params)
        entries, exits = signal_fn(prices)
    except Exception as exc:
        return {"error": f"Strategy error: {exc}"}

    runner = VectorBTRunner()
    metrics = runner.run(prices, entries, exits, strategy_id=f"{strategy}_{ticker}", n_trials=1)

    return {
        "ticker": ticker,
        "strategy": strategy,
        "params": params,
        "metrics": {
            "total_return": float(metrics.total_return),
            "cagr": float(metrics.cagr),
            "sharpe_ratio": float(metrics.sharpe_ratio),
            "deflated_sharpe_ratio": float(metrics.deflated_sharpe_ratio),
            "sortino_ratio": float(metrics.sortino_ratio),
            "calmar_ratio": float(metrics.calmar_ratio),
            "max_drawdown": float(metrics.max_drawdown),
            "win_rate": float(metrics.win_rate),
            "volatility": float(metrics.volatility),
            "var_95": float(metrics.var_95),
        },
    }


# ─── Tool 10: Macro Series ────────────────────────────────────────────────────

@mcp.tool()
async def get_macro_series(
    series_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """
    Fetch a FRED macroeconomic time series (765K+ series available).

    Common series IDs:
      DGS10 (10Y Treasury), UNRATE (Unemployment), CPIAUCSL (CPI),
      FEDFUNDS (Fed Funds Rate), GDP, VIXCLS (VIX), T10Y2Y (Yield Curve),
      M2SL (M2 Money Supply), SOFR, BAMLH0A0HYM2 (HY Spread)

    Args:
        series_id: FRED series ID
        start: Optional start date (YYYY-MM-DD)
        end: Optional end date (YYYY-MM-DD)

    Returns:
        series_id, title, units, observations: [{date, value}]
    """
    from sentinel.sds.adapters.fred_adapter import FREDAdapter

    s = get_settings()
    fred = FREDAdapter(api_key=s.fred_api_key)

    start_date = date.fromisoformat(start) if start else None
    end_date = date.fromisoformat(end) if end else None
    points = await fred.fetch_series(series_id, start=start_date, end=end_date)
    info = await fred.get_series_info(series_id)

    return {
        "series_id": series_id,
        "title": info.get("title", series_id),
        "units": info.get("units", ""),
        "frequency": info.get("frequency", ""),
        "count": len(points),
        "observations": [
            {"date": p.time.date().isoformat(), "value": float(p.value)}
            for p in points[-200:]  # Return last 200 observations
        ],
    }


# ─── Tool 11: COT Signals ─────────────────────────────────────────────────────

@mcp.tool()
async def get_cot_signals(market: Optional[str] = None) -> dict:
    """
    Fetch CFTC Commitments of Traders positioning signals. Leapfrog feature.

    The COT Index measures speculator net positioning as a percentile of the
    52-week range. >80 = extreme long (contrarian bearish), <20 = extreme short
    (contrarian bullish).

    Args:
        market: Optional market filter (e.g. 'GOLD', 'CRUDE OIL', 'S&P 500').
                If omitted, returns signals for all tracked markets.

    Returns:
        list of {market, cot_index, net_position, signal, date}
    """
    from sentinel.sds.db import get_session_factory
    from sentinel.sds import repository

    session_factory = get_session_factory()
    since_dt = datetime.combine(date.today() - timedelta(days=365), datetime.min.time())
    async with session_factory() as _session:
        db_signals = await repository.get_cot_signals(
            _session, market_name=market, since=since_dt
        )

    if db_signals:
        formatted = [
            {
                "market": s["market_name"],
                "net_position": s.get("net_speculator"),
                "cot_index": float(s["cot_index"]) if s.get("cot_index") is not None else None,
                "signal": s.get("signal"),
                "date": s["report_date"].isoformat() if hasattr(s.get("report_date"), "isoformat") else str(s.get("report_date")),
            }
            for s in db_signals
        ]
        return {
            "count": len(formatted),
            "signals": formatted,
            "source": "db",
            "interpretation": "COT Index: >80=extreme long (bearish), <20=extreme short (bullish)",
        }

    # Fall back to live CFTC download
    from sentinel.sma.cot_report import COTClient

    client = COTClient()
    current_year = date.today().year
    await client.load_range(current_year - 1, current_year)

    signals = client.get_all_market_signals()
    if market:
        signals = [s for s in signals if market.upper() in s["market"].upper()]

    return {
        "count": len(signals),
        "signals": signals,
        "source": "live",
        "interpretation": "COT Index: >80=extreme long (bearish), <20=extreme short (bullish)",
    }


# ─── Tool 12: Macro Regime ────────────────────────────────────────────────────

@mcp.tool()
async def get_macro_regime() -> dict:
    """
    Get the current macroeconomic regime from the HMM 4-state detector. Leapfrog feature.

    Regimes: GROWTH_INFLATION, GROWTH_DEFLATION, CONTRACTION_INFLATION, CONTRACTION_DEFLATION.
    Based on: yield curve slope, CPI YoY, unemployment rate, VIX.

    Returns:
        current_regime, confidence, regime_probabilities, asset_class_implications
    """
    from sentinel.sma.regime import MacroRegimeDetector, build_macro_feature_df
    from sentinel.sds.db import get_session_factory
    from sentinel.sds import repository
    import pandas as pd

    s = get_settings()
    start = date.today() - timedelta(days=365 * 10)
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.utcnow()

    session_factory = get_session_factory()
    async with session_factory() as _session:
        slope_rows, cpi_rows, unrate_rows, vix_rows = await asyncio.gather(
            repository.get_macro_series("T10Y2Y", start_dt, end_dt, _session),
            repository.get_macro_series("CPIAUCSL", start_dt, end_dt, _session),
            repository.get_macro_series("UNRATE", start_dt, end_dt, _session),
            repository.get_macro_series("VIXCLS", start_dt, end_dt, _session),
        )

    def rows_to_series(rows, name):
        return pd.Series(
            {r["time"]: float(r["value"]) for r in rows}, name=name
        ).sort_index()

    if all([slope_rows, cpi_rows, unrate_rows, vix_rows]):
        slope = rows_to_series(slope_rows, "yield_slope")
        cpi = rows_to_series(cpi_rows, "cpi_yoy").pct_change(12) * 100
        unrate = rows_to_series(unrate_rows, "unrate")
        vix = rows_to_series(vix_rows, "vix")
    else:
        # Fall back to live FRED if any series is missing from DB
        from sentinel.sds.adapters.fred_adapter import FREDAdapter
        fred = FREDAdapter(api_key=s.fred_api_key)

        def to_series(pts, name):
            return pd.Series(
                {p.time: float(p.value) for p in pts}, name=name
            ).sort_index()

        slope_pts, cpi_pts, unrate_pts, vix_pts = await asyncio.gather(
            fred.fetch_series("T10Y2Y", start=start),
            fred.fetch_series("CPIAUCSL", start=start),
            fred.fetch_series("UNRATE", start=start),
            fred.fetch_series("VIXCLS", start=start),
        )
        slope = to_series(slope_pts, "yield_slope")
        cpi = to_series(cpi_pts, "cpi_yoy").pct_change(12) * 100
        unrate = to_series(unrate_pts, "unrate")
        vix = to_series(vix_pts, "vix")

    df = build_macro_feature_df(slope, cpi, unrate, vix)
    if df.empty or len(df) < 40:
        return {"error": "Insufficient macro data for regime detection"}

    detector = MacroRegimeDetector()
    detector.fit(df)
    current = detector.predict_current(df)

    if not current:
        return {"error": "Regime detection failed"}

    implications = _regime_implications(current.regime.value)

    return {
        "current_regime": current.regime.value,
        "confidence": round(current.confidence, 3),
        "date": current.date.isoformat(),
        "regime_probabilities": {k: round(v, 3) for k, v in current.regime_probabilities.items()},
        "asset_class_implications": implications,
        "features": {k: round(v, 3) for k, v in current.features.items()},
    }


# ─── Tool 13: News Sentiment ──────────────────────────────────────────────────

@mcp.tool()
async def get_news_sentiment(
    ticker: str,
    days: int = 7,
) -> dict:
    """
    Fetch recent news for a ticker with FinBERT sentiment scores.

    Args:
        ticker: US equity ticker
        days: Lookback window in days (default 7)

    Returns:
        articles: [{headline, source, date, sentiment, score, url}]
        aggregate: {positive_pct, negative_pct, neutral_pct, net_sentiment}
    """
    from sentinel.sds.adapters.finnhub_adapter import FinnhubAdapter
    from sentinel.sil.sentiment import score_sentiment_batch

    s = get_settings()
    fh = FinnhubAdapter(api_key=s.finnhub_api_key)
    end_str = date.today().isoformat()
    start_str = (date.today() - timedelta(days=days)).isoformat()

    news = await fh.fetch_news(ticker, start_str, end_str)
    if not news:
        return {"ticker": ticker, "articles": [], "aggregate": {}}

    headlines = [n.get("headline", "") for n in news]
    sentiments = await score_sentiment_batch(headlines)

    articles = []
    for n, sent in zip(news, sentiments):
        articles.append({
            "headline": n.get("headline", ""),
            "source": n.get("source", ""),
            "date": datetime.utcfromtimestamp(n.get("datetime", 0)).date().isoformat(),
            "sentiment": sent.label,
            "score": round(sent.score, 3),
            "url": n.get("url", ""),
        })

    pos = sum(1 for a in articles if a["sentiment"] == "positive")
    neg = sum(1 for a in articles if a["sentiment"] == "negative")
    total = len(articles)
    net = (pos - neg) / total if total > 0 else 0.0

    return {
        "ticker": ticker,
        "days": days,
        "articles": articles[:20],
        "aggregate": {
            "total": total,
            "positive_pct": round(pos / total * 100, 1) if total else 0,
            "negative_pct": round(neg / total * 100, 1) if total else 0,
            "neutral_pct": round((total - pos - neg) / total * 100, 1) if total else 0,
            "net_sentiment": round(net, 3),
        },
    }


# ─── Tool 14: Options Chain ───────────────────────────────────────────────────

@mcp.tool()
async def get_options_chain(
    ticker: str,
    expiry: Optional[str] = None,
    option_type: Optional[str] = None,
) -> dict:
    """
    Fetch options chain with strikes, implied volatility, and greeks.

    Args:
        ticker: Underlying ticker (e.g. 'SPY')
        expiry: Optional expiry date (YYYY-MM-DD). Defaults to nearest expiry.
        option_type: 'call', 'put', or None for both.

    Returns:
        expiry, calls: [{strike, last, bid, ask, iv, delta, gamma, theta, vega}],
        puts: [same]
    """
    from sentinel.sds.adapters.yfinance_adapter import YFinanceAdapter

    yf = YFinanceAdapter()
    chain = await yf.fetch_options_chain(ticker, expiry=expiry)
    if not chain:
        return {"ticker": ticker, "error": "No options data available"}

    result = {"ticker": ticker, "expiry": chain.get("expiry")}
    if option_type != "put":
        result["calls"] = chain.get("calls", [])[:50]
    if option_type != "call":
        result["puts"] = chain.get("puts", [])[:50]
    return result


# ─── Tool 15: Explain Strategy ────────────────────────────────────────────────

@mcp.tool()
async def explain_strategy(
    strategy_id: str,
    sharpe: float,
    dsr: float,
    pbo: Optional[float] = None,
    max_drawdown: float = 0.0,
    cagr: float = 0.0,
    n_trials: int = 1,
) -> dict:
    """
    AI-powered explanation of backtest metrics with risk assessment and recommendations.

    Args:
        strategy_id: Strategy name/identifier
        sharpe: Sharpe ratio from backtest
        dsr: Deflated Sharpe Ratio (adjusts for multiple-testing)
        pbo: Probability of Backtest Overfitting (0-1)
        max_drawdown: Maximum drawdown (negative, e.g. -0.15 for 15% drawdown)
        cagr: Compound Annual Growth Rate
        n_trials: Number of parameter combinations tested

    Returns:
        assessment, risks, recommendations, promotion_readiness
    """
    from sentinel.sbe.pbo import _interpret_pbo

    risks = []
    recommendations = []

    if sharpe < dsr:
        risks.append(f"Sharpe ({sharpe:.2f}) exceeds DSR ({dsr:.2f}) — multiple-testing penalty applied")

    if dsr < 0.95:
        risks.append(f"DSR {dsr:.3f} < 0.95 threshold — strategy may not survive multiple testing")
        recommendations.append("Reduce number of parameter combinations and re-validate with fresh data")

    if pbo is not None and pbo > 0.20:
        risks.append(f"PBO {pbo:.2%} — {_interpret_pbo(pbo)}")
        recommendations.append("Use combinatorially symmetric cross-validation on out-of-sample data")

    if max_drawdown < -0.30:
        risks.append(f"Max drawdown {max_drawdown:.1%} is severe — check position sizing")
        recommendations.append("Apply Kelly Criterion or half-Kelly sizing to reduce drawdown")

    if n_trials > 50 and dsr > 0.95:
        recommendations.append(f"With {n_trials} trials tested, confirm on truly out-of-sample period")

    promotion_ready = dsr >= 0.95 and (pbo is None or pbo <= 0.20) and sharpe > 0
    assessment = "PASS — Ready for paper trading" if promotion_ready else "FAIL — Does not meet promotion gates"

    return {
        "strategy_id": strategy_id,
        "assessment": assessment,
        "promotion_ready": promotion_ready,
        "metrics_summary": {
            "sharpe": sharpe, "dsr": dsr, "pbo": pbo,
            "max_drawdown": max_drawdown, "cagr": cagr, "n_trials": n_trials,
        },
        "risks": risks,
        "recommendations": recommendations,
        "next_step": "Proceed to paper trading via SEE promotion engine" if promotion_ready
                     else "Revisit strategy design and reduce parameter search space",
    }


# ─── Tool 16: Options Analytics ──────────────────────────────────────────────

@mcp.tool()
async def get_options_analytics(ticker: str, underlying_price: float) -> dict:
    """Get IV surface, put/call skew, gamma exposure, max pain, and put/call ratios for a ticker.
    Returns the full options market structure analysis."""
    from sentinel.sbx.options_analytics import get_options_summary
    from sentinel.sds import get_adapter
    try:
        adapter = get_adapter("polygon")
        contracts_raw = await adapter.fetch_options_chain(ticker)
        return get_options_summary(ticker, contracts_raw, underlying_price)
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


# ─── Tool 17: Document RAG Query ─────────────────────────────────────────────

@mcp.tool()
async def query_documents(
    query: str,
    ticker: str | None = None,
    doc_type: str | None = None,
    synthesize: bool = False,
) -> dict:
    """Semantic search over ingested financial documents (10-K, 10-Q, earnings calls, news).
    Uses pgvector dense + BM25 sparse retrieval with RRF reranking.
    Set synthesize=True to get a Claude-generated answer from retrieved context."""
    from sentinel.sil.rag import query as rag_query
    from sentinel.core.config import get_settings
    settings = get_settings()
    result = await rag_query(
        db_url=settings.database_url,
        query_text=query,
        ticker=ticker,
        doc_type=doc_type,
        synthesize=synthesize,
        anthropic_api_key=settings.anthropic_api_key,
    )
    return result.model_dump()


# ─── Tool 18: Social Sentiment ────────────────────────────────────────────────

@mcp.tool()
async def get_social_sentiment(ticker: str) -> dict:
    """Aggregate social media sentiment for a ticker from Reddit (WSB, r/investing) and StockTwits.
    Returns bullish/bearish/neutral percentages, FinBERT-weighted score, and top posts."""
    from sentinel.snm.social_sentiment import get_social_sentiment as _get_social
    result = await _get_social(ticker)
    return result.model_dump()


# ─── Tool 19: Economic Calendar ───────────────────────────────────────────────

@mcp.tool()
async def get_economic_calendar(days_ahead: int = 14) -> dict:
    """Get upcoming macro economic data releases with importance scoring.
    Shows FOMC, NFP, CPI, GDP, and 20+ other market-moving releases."""
    from sentinel.sma.economic_calendar import get_calendar
    from sentinel.core.config import get_settings
    settings = get_settings()
    calendar = await get_calendar(api_key=settings.fred_api_key, days_ahead=days_ahead)
    return calendar.model_dump()


# ─── Tool 20: NL Screener (Full) ──────────────────────────────────────────────

@mcp.tool()
async def screen_stocks_nl(query: str, max_results: int = 50) -> dict:
    """Screen stocks using natural language. Claude translates your query into quantitative
    criteria and runs them against the SENTINEL stock universe.
    Example: 'find cheap profitable small caps with insider buying and momentum'"""
    from sentinel.sil.nl_screener import run_nl_screen
    from sentinel.core.config import get_settings
    settings = get_settings()
    result = await run_nl_screen(
        query=query,
        db_path=":memory:",  # SSE screener uses its own DuckDB path
        anthropic_api_key=settings.anthropic_api_key,
        max_results=max_results,
    )
    return result.model_dump()


# ─── Tool 21: Stress Test ─────────────────────────────────────────────────────

@mcp.tool()
async def run_stress_test(holdings_json: str, portfolio_value: float) -> dict:
    """Run portfolio stress tests against historical scenarios (2008, COVID, 2022 rate shock, etc.)
    and parametric shocks. holdings_json: JSON string of {ticker: weight} dict."""
    import json
    from sentinel.spr.stress_test import run_stress_test as _stress
    holdings = json.loads(holdings_json)
    import pandas as pd
    # Return stress report without historical returns (parametric only if no data)
    report = _stress(holdings=holdings, portfolio_value=portfolio_value, returns_df=pd.DataFrame())
    return report.model_dump()


# ─── Tool 22: Factor Exposure ─────────────────────────────────────────────────

@mcp.tool()
async def get_factor_exposure(holdings_json: str) -> dict:
    """Decompose portfolio returns into Fama-French 5-factor + momentum exposures.
    Returns alpha, factor loadings, t-stats, and variance attribution.
    holdings_json: JSON string of {ticker: weight} dict."""
    import json
    from sentinel.spr.factor_model import decompose_portfolio
    from datetime import date, timedelta
    import pandas as pd
    holdings = json.loads(holdings_json)
    end = date.today()
    start = end - timedelta(days=365)
    result = decompose_portfolio(holdings=holdings, returns_df=pd.DataFrame(), start=start, end=end)
    return result.model_dump()


# ─── Tool 23: DCF Valuation ───────────────────────────────────────────────────

@mcp.tool()
async def run_dcf_valuation(
    ticker: str,
    current_price: float,
    wacc: float = 0.10,
    terminal_growth: float = 0.025,
) -> dict:
    """Run a DCF valuation for a stock using Damodaran methodology.
    Fetches fundamentals from EDGAR, projects FCF, computes intrinsic value vs current price.
    Returns intrinsic value, upside/downside %, and WACC × terminal growth sensitivity table."""
    from sentinel.sfe.dcf_model import DCFAssumptions, run_dcf
    # Build basic assumptions — in production would fetch from EDGAR fundamentals
    assumptions = DCFAssumptions(
        ticker=ticker,
        revenue_base=1_000_000_000,  # placeholder — wire to XBRL in Gen 2
        revenue_growth_rates=[0.10, 0.09, 0.08, 0.07, 0.06],
        terminal_growth_rate=terminal_growth,
        ebit_margin=0.15,
        tax_rate=0.21,
        capex_pct_revenue=0.05,
        da_pct_revenue=0.04,
        nwc_change_pct_revenue=0.02,
        wacc=wacc,
        net_debt=0.0,
        shares_outstanding=100.0,
    )
    result = run_dcf(assumptions, current_price)
    return result.model_dump()


# ─── Tool 24: Global Macro Dashboard ─────────────────────────────────────────

@mcp.tool()
async def get_global_macro(country: str = "all") -> dict:
    """Get global macro dashboard — GDP growth, inflation, unemployment, rates for US/EU/UK/JP/CN/AU/CA.
    Also includes G7 yield curve comparison (2Y/10Y spreads) and rate differential signals.
    country: ISO2 code (us/eu/uk/jp/cn/au/ca) or 'all' for full dashboard."""
    from sentinel.sma.global_macro import get_global_dashboard, get_yield_curve_comparison
    from sentinel.core.config import get_settings
    settings = get_settings()
    if country.lower() == "all":
        dashboard = await get_global_dashboard(fred_api_key=settings.fred_api_key)
        curves = await get_yield_curve_comparison(fred_api_key=settings.fred_api_key)
        return {
            "countries": {k: v.model_dump() for k, v in dashboard.items()},
            "yield_curves": [c.model_dump() for c in curves],
        }
    else:
        dashboard = await get_global_dashboard(fred_api_key=settings.fred_api_key)
        c = country.lower()
        if c in dashboard:
            return dashboard[c].model_dump()
        return {"error": f"Unknown country code: {country}. Use us/eu/uk/jp/cn/au/ca or 'all'"}


# ─── Tool 25: Position Sizing ─────────────────────────────────────────────────

@mcp.tool()
async def get_position_sizing(
    holdings_json: str,
    method: str = "kelly",
    portfolio_value: float = 100_000.0,
    win_rate: float = 0.55,
    avg_win: float = 0.08,
    avg_loss: float = 0.04,
    target_vol: float = 0.15,
) -> dict:
    """Compute optimal position sizes using Kelly Criterion, volatility targeting, or risk parity.
    holdings_json: JSON {ticker: weight} for risk_parity/vol_target; ignored for single Kelly.
    method: 'kelly' | 'vol_target' | 'risk_parity' | 'equal'
    Returns position sizes as fraction of portfolio and dollar amounts."""
    import json
    from sentinel.spr.kelly_sizer import kelly_from_returns, size_portfolio
    import pandas as pd

    holdings = json.loads(holdings_json)

    if method == "kelly":
        result = kelly_from_returns(
            returns=pd.Series([avg_win] * 100 + [-avg_loss] * 82),  # simulate win_rate≈55%
            risk_free_rate=0.045,
        )
        return result.model_dump()
    else:
        result = size_portfolio(
            tickers=list(holdings.keys()),
            returns_df=pd.DataFrame(),
            method=method,
            capital=portfolio_value,
            target_vol=target_vol,
        )
        return result.model_dump()


# ─── Tool 26: FX Rates ────────────────────────────────────────────────────────

@mcp.tool()
async def get_fx_rates(
    base: str = "USD",
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Get ECB official FX rates via Frankfurter API (free, no key required).
    Returns spot rates for all major G10 currency pairs.
    start/end: optional YYYY-MM-DD for historical range."""
    from sentinel.sds.adapters.fx_adapter import FXAdapter
    from datetime import date, timedelta
    fx = FXAdapter()
    if start and end:
        start_d = date.fromisoformat(start)
        end_d = date.fromisoformat(end)
        bars = await fx.fetch_base_rates(base=base.upper(), start=start_d, end=end_d)
        return {
            "base": base.upper(),
            "start": start,
            "end": end,
            "count": len(bars),
            "bars": [
                {
                    "pair": b.pair,
                    "date": b.date.isoformat(),
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                }
                for b in bars[:200]
            ],
        }
    else:
        majors = await fx.fetch_all_majors()
        return {"base": base.upper(), "rates": majors, "source": "ECB/Frankfurter"}


# ─── Tool 27: Short Interest ──────────────────────────────────────────────────

@mcp.tool()
async def get_short_interest(
    ticker: str | None = None,
    squeeze_threshold: float = 0.40,
) -> dict:
    """Get FINRA short interest data for a ticker or screen for short squeeze candidates.
    Data from FINRA daily RegSHO files — most recent trading day available.
    ticker: specific symbol, or None to return squeeze candidates above threshold."""
    from sentinel.sds.adapters.short_interest_adapter import ShortInterestAdapter
    si = ShortInterestAdapter()
    if ticker:
        record = await si.get_short_interest(ticker.upper())
        if record:
            return record.model_dump()
        return {"ticker": ticker, "error": "No short interest data found — check FINRA coverage"}
    else:
        candidates = await si.get_squeeze_candidates(min_short_pct=squeeze_threshold)
        return {
            "squeeze_threshold": squeeze_threshold,
            "candidates": [c.model_dump() for c in candidates[:25]],
        }


# ─── Tool 28: Corporate Bonds (TRACE) ────────────────────────────────────────

@mcp.tool()
async def get_corporate_bonds(
    ticker: str,
    build_curve: bool = False,
) -> dict:
    """Get FINRA TRACE corporate bond quotes and optional credit spread curve for a company.
    Data from FINRA TRACE aggregates — covers investment grade and high yield bonds.
    build_curve=True returns a full credit curve (maturity vs OAS spread)."""
    from sentinel.sbx.trace_client import TRACEClient
    client = TRACEClient()
    quotes = await client.get_bond_quotes(ticker.upper())
    result: dict = {
        "ticker": ticker,
        "bonds": [q.model_dump() for q in quotes[:20]],
        "count": len(quotes),
    }
    if build_curve and quotes:
        curve = await client.build_credit_curve(ticker.upper())
        if curve:
            result["credit_curve"] = curve.model_dump()
    return result


# ─── Tool 29: Segment Breakdown ───────────────────────────────────────────────

@mcp.tool()
async def get_segment_breakdown(ticker: str) -> dict:
    """Get business segment revenue breakdown from EDGAR XBRL dimensional facts.
    Parses 10-K/10-Q filings to extract geographic and business unit revenue splits.
    Returns segment names, revenue, and % of total for the most recent annual period."""
    from sentinel.sfe.segment_parser import get_segment_breakdown as _parse_segments
    from sentinel.sim.instrument_master import InstrumentMaster
    from sentinel.core.config import get_settings
    settings = get_settings()
    # Resolve ticker to CIK via instrument master
    im = InstrumentMaster(settings)
    instrument = await im.resolve(ticker=ticker.upper())
    cik = getattr(instrument, "cik", None) if instrument else None
    if not cik:
        return {"ticker": ticker, "error": "CIK not found — ensure instrument is in SENTINEL universe"}
    result = await _parse_segments(cik=cik, ticker=ticker.upper())
    return result.model_dump()


# ─── Tool 30: Options Flow Screener ──────────────────────────────────────────

@mcp.tool()
async def screen_options_flow(
    tickers_json: str | None = None,
    min_volume_ratio: float = 2.0,
) -> dict:
    """Screen for unusual options activity — volume spikes, IV expansions, put skew, gamma walls.
    tickers_json: JSON list of tickers, or None for S&P 500 universe.
    min_volume_ratio: minimum ratio of current to average volume to flag as unusual."""
    import json
    from sentinel.sse.options_screener import screen_universe, OptionsScreenerCriteria
    from sentinel.core.config import get_settings
    settings = get_settings()

    _default_tickers = [
        "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "JPM",
    ]
    tickers = json.loads(tickers_json) if tickers_json else _default_tickers

    # Best-effort price fetch; fall back to 100.0 so screener can still compute ratios
    prices: dict[str, float] = {}
    try:
        from sentinel.sds.adapters.yfinance_adapter import YFinanceAdapter
        yf = YFinanceAdapter()
        for t in tickers[:20]:
            try:
                info = await yf.fetch_info(t)
                prices[t] = float(info.get("regularMarketPrice") or 100.0)
            except Exception:
                prices[t] = 100.0
    except Exception:
        prices = {t: 100.0 for t in tickers}

    criteria = OptionsScreenerCriteria(min_volume_ratio=min_volume_ratio)
    alerts = await screen_universe(
        tickers=tickers,
        prices=prices,
        polygon_api_key=settings.polygon_api_key or "",
        criteria=criteria,
    )
    return {
        "alerts": [a.model_dump() for a in alerts[:50]],
        "total_alerts": len(alerts),
        "min_volume_ratio": min_volume_ratio,
    }


# ─── Tool 31: Generate Trading Strategy ────────────────────────────────────────

@mcp.tool()
async def generate_trading_strategy(
    description: str,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """
    Translate a natural-language strategy description into a structured GeneratedStrategy spec.
    Uses Claude tool-use for precise signal extraction (entry/exit signals, universe, sizing).
    Falls back to rule-based parser if no Anthropic API key.

    Args:
        description: Free-form strategy description, e.g. "Buy small-cap momentum stocks
                     with RSI > 60 and positive earnings surprise; sell after 20 days"
        model: Claude model to use (default: haiku for speed)
    """
    from sentinel.sil.strategy_generator import generate_strategy

    settings = get_settings()
    api_key = getattr(settings, "anthropic_api_key", None) or ""

    strategy = await generate_strategy(
        description=description,
        anthropic_api_key=api_key,
        model=model,
    )
    return strategy.model_dump()


# ─── Tool 32: Research Ticker ────────────────────────────────────────────────

@mcp.tool()
async def research_ticker(
    ticker: str,
    question: str = "What is the investment thesis for this company?",
) -> dict:
    """
    Autonomous multi-step research agent. Gathers fundamentals, news, insider trades,
    price history, and macro context in parallel, then synthesises a ResearchMemo via Claude.

    Args:
        ticker: Stock ticker, e.g. "NVDA"
        question: Research question to answer, e.g. "Is NVDA a buy at current valuation?"
    """
    from sentinel.sil.research_agent import research_ticker as _research

    settings = get_settings()
    api_key = getattr(settings, "anthropic_api_key", None) or ""

    memo = await _research(ticker=ticker.upper(), question=question, anthropic_api_key=api_key)
    return memo.model_dump(mode="json")


# ─── Tool 33: Google Trends Signal ───────────────────────────────────────────

@mcp.tool()
async def get_google_trends(
    ticker: str,
    keywords_json: Optional[str] = None,
    geo: str = "US",
) -> dict:
    """
    Google Trends momentum signal for a ticker. Returns 4-week vs 52-week z-score,
    trend direction (rising/falling/stable/spike), and signal strength (strong/moderate/weak/neutral/negative).

    Args:
        ticker: Ticker symbol, e.g. "AAPL" — auto-resolved to company name for search
        keywords_json: Optional JSON array of extra keywords, e.g. '["iPhone", "App Store"]'
        geo: ISO country code (default "US"); "" for worldwide
    """
    import json as _json
    from sentinel.sma.google_trends import get_trend_signal

    keywords: list[str] = []
    if keywords_json:
        try:
            keywords = _json.loads(keywords_json)
        except _json.JSONDecodeError:
            pass

    signal = get_trend_signal(ticker=ticker.upper(), keywords=keywords, geo=geo)
    return signal.model_dump()


# ─── Tool 34: DeFi Dashboard ─────────────────────────────────────────────────

@mcp.tool()
async def get_defi_dashboard(top_n: int = 20) -> dict:
    """
    DeFi ecosystem snapshot via DeFiLlama (free, no API key).
    Returns top protocols by TVL, chain TVL breakdown, top yield opportunities,
    and stablecoin market share.

    Args:
        top_n: Number of top protocols/chains to include (default 20)
    """
    from sentinel.snm.defi_analytics import DefiLlamaClient

    client = DefiLlamaClient()
    dashboard = await client.get_dashboard(top_n=top_n)
    return dashboard.model_dump()


# ─── Tool 35: On-Chain Crypto Signals ────────────────────────────────────────

@mcp.tool()
async def get_crypto_signals(
    symbol: str = "bitcoin",
) -> dict:
    """
    On-chain / market-structure signals for a crypto asset via CoinGecko (free).
    Returns NVT proxy (market cap / volume ratio — valuation), MVRV proxy (price vs 30d avg),
    fear/greed proxy, and an overall signal (strong_buy → strong_sell).

    Args:
        symbol: CoinGecko coin ID, e.g. "bitcoin", "ethereum", "solana"
    """
    from sentinel.snm.onchain_metrics import OnChainClient

    client = OnChainClient()
    signal = await client.get_signal(symbol=symbol.lower())
    return signal.model_dump()


# ─── Tool 36: Earnings KPIs ──────────────────────────────────────────────────

@mcp.tool()
async def get_earnings_kpis(
    ticker: str,
    filing_type: str = "10-Q",
) -> dict:
    """
    Extract structured KPIs and management tone from EDGAR MD&A section.
    Uses Claude for semantic extraction when available, regex fallback otherwise.
    Returns revenue, EPS, margins, guidance, management tone, and key themes.

    Args:
        ticker: Stock ticker, e.g. "MSFT"
        filing_type: "10-Q" (quarterly) or "10-K" (annual)
    """
    from sentinel.sfe.earnings_kpi import get_earnings_kpi

    settings = get_settings()
    api_key = getattr(settings, "anthropic_api_key", None)

    result = await get_earnings_kpi(
        ticker=ticker.upper(),
        anthropic_api_key=api_key,
        form_type=filing_type,
    )
    return result.model_dump(mode="json")


# ─── Tool 37: ESG Profile ────────────────────────────────────────────────────

@mcp.tool()
async def get_esg_profile(ticker: str) -> dict:
    """
    ESG proxy signals from free EDGAR filings (DEF 14A + 10-K).
    E score: environmental keywords in risk factors.
    S score: CEO pay ratio + board gender diversity from proxy.
    G score: governance policy mentions.
    Note: proxy signals only — not MSCI/Sustainalytics rated.

    Args:
        ticker: Stock ticker, e.g. "JPM"
    """
    from sentinel.sfe.esg_parser import get_esg_profile as _esg

    profile = await _esg(ticker=ticker.upper())
    return profile.model_dump(mode="json")


# ─── Tool 38: Alpha Signals ──────────────────────────────────────────────────

@mcp.tool()
async def get_alpha_signals(ticker: str) -> dict:
    """
    Composite alpha signal combining congressional STOCK Act trades,
    CFTC COT positioning, and SEC Form 4 insider transactions.
    Returns z-scored signals, cluster detection, and a composite buy/sell rating.

    Args:
        ticker: Stock ticker, e.g. "AAPL"
    """
    from sentinel.spr.signal_library import get_full_signal

    result = await get_full_signal(ticker=ticker.upper())
    return result.model_dump(mode="json")


# ─── Tool 39: Expand Search Query ────────────────────────────────────────────

@mcp.tool()
async def expand_search_query(
    query: str,
    use_hyde: bool = False,
    use_llm: bool = True,
) -> dict:
    """
    Expand a financial search query with synonyms, ticker-to-company resolution,
    and optionally LLM expansion or HyDE (Hypothetical Document Embedding).
    Used to improve RAG retrieval recall before query_documents.

    Args:
        query: Raw search query, e.g. "NVDA earnings guidance"
        use_hyde: Generate a hypothetical ideal answer and embed that (best for factual Q&A)
        use_llm: Use Claude to generate additional financial terms (best for keyword queries)
    """
    from sentinel.sil.query_expander import expand_query

    settings = get_settings()
    api_key = getattr(settings, "anthropic_api_key", None)

    result = await expand_query(
        query=query,
        anthropic_api_key=api_key if use_llm else None,
        use_hyde=use_hyde,
        use_llm=use_llm,
    )
    return result.model_dump()


# ─── Tool 40: Ingest News to RAG ─────────────────────────────────────────────

@mcp.tool()
async def ingest_news_to_rag(
    lookback_hours: int = 24,
) -> dict:
    """
    Cross-ingest recent news_articles from the structured news table into the
    document_chunks RAG vector index. Call this to refresh the RAG corpus with
    the latest news before running query_documents on news topics.

    Args:
        lookback_hours: How many hours back to pull from news_articles (default 24)
    """
    from sentinel.sil.news_rag_bridge import ingest_sentinel_news_feed

    settings = get_settings()
    db_url = settings.db_url

    stats = await ingest_sentinel_news_feed(db_url=db_url, lookback_hours=lookback_hours)
    return stats.model_dump()


# ─── Tool 41: Non-GAAP Metrics ───────────────────────────────────────────────

@mcp.tool()
async def get_non_gaap_metrics(ticker: str) -> dict:
    """
    Extract non-GAAP financial metrics from the latest EDGAR 8-K earnings release.
    Returns Adjusted EBITDA, non-GAAP EPS, Free Cash Flow, and reconciliations
    to the nearest GAAP equivalent.

    Args:
        ticker: Stock ticker, e.g. "MSFT"
    """
    from sentinel.sfe.non_gaap_parser import get_non_gaap_metrics as _ngaap

    result = await _ngaap(ticker=ticker.upper())
    return result.model_dump(mode="json")


# ─── Tool 42: Comparable Company Table ───────────────────────────────────────

@mcp.tool()
async def get_comps_table(
    ticker: str,
    peers_json: Optional[str] = None,
) -> dict:
    """
    Build a comparable company (comps) table with valuation and operating metrics
    for a ticker and its sector peers. Data from EDGAR XBRL + yfinance (free).

    Args:
        ticker: Target ticker, e.g. "NVDA"
        peers_json: Optional JSON array of peer tickers to override auto-selection,
                    e.g. '["AMD", "INTC", "AVGO"]'
    """
    import json as _json
    from sentinel.sfe.comps_table import get_comps_table as _comps

    peers: Optional[list[str]] = None
    if peers_json:
        try:
            peers = _json.loads(peers_json)
        except Exception:
            pass

    result = await _comps(ticker=ticker.upper(), include_peers=peers)
    return result.model_dump(mode="json")


# ─── Tool 43: Activist Positions ─────────────────────────────────────────────

@mcp.tool()
async def get_activist_positions(ticker: str) -> dict:
    """
    Track activist investor 13D/13G filings for a ticker from EDGAR.
    Identifies active (13D) vs passive (13G) positions, % ownership,
    and flags known activist funds (Elliott, Third Point, Icahn, etc.).

    Args:
        ticker: Stock ticker, e.g. "DIS"
    """
    from sentinel.sds.adapters.activist_adapter import get_activist_summary

    result = await get_activist_summary(ticker=ticker.upper())
    return result.model_dump(mode="json")


# ─── Tool 44: EDGAR Full-Text Search ─────────────────────────────────────────

@mcp.tool()
async def search_edgar(
    query: str,
    form_types_json: Optional[str] = None,
    ticker: Optional[str] = None,
    days_back: int = 365,
) -> dict:
    """
    Full-text search across EDGAR filings using the EFTS search API.
    Find any mention of a topic, person, product, or event across 10-Ks, 10-Qs,
    8-Ks, and proxy statements.

    Args:
        query: Search query, e.g. "artificial intelligence risk factors"
        form_types_json: Optional JSON array of form types, e.g. '["10-K", "8-K"]'
        ticker: Optional ticker to limit search to one company
        days_back: How many days back to search (default 365)
    """
    import json as _json
    from sentinel.sil.edgar_search import search_edgar as _search

    form_types: Optional[list[str]] = None
    if form_types_json:
        try:
            form_types = _json.loads(form_types_json)
        except Exception:
            pass

    result = await _search(
        query=query,
        form_types=form_types,
        days_back=days_back,
        ticker=ticker.upper() if ticker else None,
    )
    return result.model_dump(mode="json")


# ─── Tool 45: Portfolio VaR/CVaR ─────────────────────────────────────────────

@mcp.tool()
async def get_portfolio_var(
    weights_json: str,
    confidence: float = 0.95,
    horizon_days: int = 1,
    method: str = "historical",
    portfolio_value: Optional[float] = None,
) -> dict:
    """
    Compute Value-at-Risk (VaR) and Conditional VaR (CVaR/Expected Shortfall)
    for a portfolio using historical simulation, parametric, or Monte Carlo method.

    Args:
        weights_json: JSON object of ticker weights, e.g. '{"AAPL": 0.4, "MSFT": 0.6}'
        confidence: Confidence level (default 0.95 = 95% VaR)
        horizon_days: Holding period in trading days (1, 5, 10, or 21)
        method: "historical" | "parametric" | "monte_carlo"
        portfolio_value: Portfolio value in USD for dollar VaR (optional)
    """
    import json as _json
    from sentinel.spr.var_engine import compute_portfolio_var

    weights = _json.loads(weights_json)
    result = await compute_portfolio_var(
        weights=weights,
        confidence=confidence,
        horizon_days=horizon_days,
        method=method,
        portfolio_value=portfolio_value,
    )
    return result.model_dump(mode="json")


# ─── Tool 46: Brinson Attribution ────────────────────────────────────────────

@mcp.tool()
async def get_portfolio_attribution(
    holdings_json: str,
    benchmark: str = "SPY",
    days_back: int = 90,
) -> dict:
    """
    Brinson-Hood-Beebower (BHB) portfolio attribution: decomposes active return
    into allocation effect (sector bets) + selection effect (stock picking)
    + interaction effect, for each GICS sector.

    Args:
        holdings_json: JSON object of ticker weights, e.g. '{"AAPL": 0.3, "NVDA": 0.2, "JPM": 0.5}'
        benchmark: Benchmark ETF ticker (default "SPY")
        days_back: Attribution period in days (default 90 = ~1 quarter)
    """
    import json as _json
    from sentinel.spr.attribution import compute_attribution

    holdings = _json.loads(holdings_json)
    result = await compute_attribution(
        holdings=holdings,
        benchmark=benchmark.upper(),
    )
    return result.model_dump(mode="json")


# ─── Tool 47: Controversy Monitor ────────────────────────────────────────────

@mcp.tool()
async def get_controversy_signals(
    ticker: str,
    days_back: int = 90,
) -> dict:
    """
    News-based controversy monitoring for ESG risk scoring. Classifies recent
    headlines into environmental, social, governance, regulatory, litigation,
    and cybersecurity categories with severity scoring.

    Args:
        ticker: Stock ticker, e.g. "META"
        days_back: Lookback window in days (default 90)
    """
    from sentinel.snm.controversy_monitor import get_controversy_profile

    result = await get_controversy_profile(ticker=ticker.upper(), days_back=days_back)
    return result.model_dump(mode="json")


# ─── Tool 48: DEX Analytics ──────────────────────────────────────────────────

@mcp.tool()
async def get_dex_dashboard(top_n: int = 15) -> dict:
    """
    Decentralized exchange (DEX) analytics via DeFiLlama (free, no API key).
    Returns top DEX protocols by 24h volume, volume by chain, and protocol details
    including Uniswap, Curve, PancakeSwap, GMX, dYdX, etc.

    Args:
        top_n: Number of top protocols to include (default 15)
    """
    from sentinel.snm.defi_analytics import get_dex_dashboard as _dex

    result = await _dex(top_n=top_n)
    return result.model_dump(mode="json")


# ─── Tool 49: Portfolio Optimizer ────────────────────────────────────────────

@mcp.tool()
async def optimize_portfolio(
    tickers_json: str,
    method: str = "max_sharpe",
    risk_free: float = 0.05,
    lookback_days: int = 252,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    views_json: Optional[str] = None,
    view_confidence: float = 0.5,
) -> dict:
    """
    Portfolio optimizer with mean-variance (Markowitz), Black-Litterman, and
    Equal-Risk-Contribution methods. Returns optimal weights, efficient frontier,
    Sharpe ratio, and per-asset contribution statistics.

    Args:
        tickers_json: JSON array of tickers, e.g. '["AAPL","MSFT","NVDA","SPY"]'
        method: Optimization method — "min_variance" | "max_sharpe" | "black_litterman" | "erc"
        risk_free: Risk-free rate annualized (default 0.05 = 5%)
        lookback_days: Historical window for return/covariance estimation
        min_weight: Minimum weight per asset (0.0 = no short constraint)
        max_weight: Maximum weight per asset (1.0 = unconstrained)
        views_json: Black-Litterman views as JSON dict, e.g. '{"AAPL": 0.15, "MSFT": 0.12}'
        view_confidence: BL view confidence (0=no confidence, 1=absolute certainty)
    """
    import json
    from sentinel.spr.optimizer import optimize_portfolio as _opt

    tickers = json.loads(tickers_json)
    views = json.loads(views_json) if views_json else None
    result = await _opt(
        tickers=tickers, method=method, risk_free=risk_free,
        lookback_days=lookback_days, min_weight=min_weight, max_weight=max_weight,
        views=views, view_confidence=view_confidence,
    )
    return result.model_dump(mode="json")


# ─── Tool 50: Fixed Income Screener ──────────────────────────────────────────

@mcp.tool()
async def screen_bonds(
    query: Optional[str] = None,
    min_yield: Optional[float] = None,
    max_yield: Optional[float] = None,
    credit_quality_json: Optional[str] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    limit: int = 25,
) -> dict:
    """
    Fixed income screener: search bond ETF proxies and FINRA TRACE data by yield,
    duration, credit quality (IG/HY/AAA/BBB/etc.), and sector.
    Includes FRED real-time Treasury yields and credit spreads for market context.

    Args:
        query: Natural language query, e.g. "investment grade tech bonds yield > 5%"
        min_yield: Minimum yield in percent, e.g. 5.0
        max_yield: Maximum yield in percent
        credit_quality_json: JSON list of ratings, e.g. '["IG","A","BBB"]'
        min_duration: Minimum duration in years
        max_duration: Maximum duration in years
        limit: Max results to return
    """
    import json
    from sentinel.sfe.bond_screener import BondScreenCriteria, screen_bonds as _screen

    credit = json.loads(credit_quality_json) if credit_quality_json else None
    criteria = BondScreenCriteria(
        min_yield=min_yield, max_yield=max_yield,
        credit_quality=credit,
        min_duration=min_duration, max_duration=max_duration,
    )
    result = await _screen(criteria=criteria, query=query, limit=limit)
    return result.model_dump(mode="json")


# ─── Tool 51: On-Chain Event Monitor ─────────────────────────────────────────

@mcp.tool()
async def get_onchain_events(
    ticker: Optional[str] = None,
    protocol: Optional[str] = None,
    days_back: int = 7,
    min_value_usd: float = 500_000,
) -> dict:
    """
    On-chain event monitoring: whale transactions, large TVL changes, and
    protocol events via Etherscan (free) and DeFiLlama.

    Args:
        ticker: ERC-20 token symbol, e.g. "UNI", "AAVE", "LINK", "COMP"
        protocol: DeFiLlama protocol slug, e.g. "uniswap", "aave", "curve"
        days_back: Lookback window in days (default 7)
        min_value_usd: Minimum USD value for whale transaction alerts (default $500K)
    """
    from sentinel.snm.onchain_events import get_onchain_events as _events

    result = await _events(
        ticker=ticker, protocol=protocol,
        days_back=days_back, min_value_usd=min_value_usd,
    )
    return result.model_dump(mode="json")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_nl_screen(query: str) -> dict:
    """Naive keyword-based criteria parser. Replaced by Claude tool-use in full impl."""
    criteria: dict = {}
    q = query.lower()
    if "pe" in q or "p/e" in q:
        import re
        m = re.search(r"pe\s*[<>]\s*([\d.]+)", q)
        if m:
            criteria["pe_ratio_lt"] = float(m.group(1))
    if "dividend" in q or "yield" in q:
        criteria["has_dividend"] = True
    if "small cap" in q:
        criteria["market_cap_max"] = 2e9
    if "large cap" in q:
        criteria["market_cap_min"] = 10e9
    if "insider buying" in q:
        criteria["insider_buying_30d"] = True
    return criteria


def _regime_implications(regime: str) -> dict:
    implications = {
        "GROWTH_INFLATION": {
            "equities": "cautious — valuations compress with rising rates",
            "bonds": "bearish — duration risk",
            "commodities": "bullish — inflation hedge",
            "cash": "neutral",
        },
        "GROWTH_DEFLATION": {
            "equities": "bullish — goldilocks environment",
            "bonds": "neutral — low yield but stable",
            "commodities": "bearish",
            "cash": "underweight",
        },
        "CONTRACTION_INFLATION": {
            "equities": "bearish — stagflation worst for stocks",
            "bonds": "bearish — inflation erodes real return",
            "commodities": "bullish — supply constraints persist",
            "cash": "overweight",
        },
        "CONTRACTION_DEFLATION": {
            "equities": "bearish — recession risk",
            "bonds": "bullish — flight to quality, rates fall",
            "commodities": "bearish — demand collapse",
            "cash": "overweight — king in deflation",
        },
    }
    return implications.get(regime, {})


# ─── Tool 52: DataCleaner ─────────────────────────────────────────────────────

@mcp.tool()
async def clean_data(
    ticker: str,
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    write_to_db: bool = True,
) -> dict:
    """
    Run the SENTINEL multi-source DataCleaner for one ticker.

    Fetches OHLCV from ALL available adapters in parallel, computes a
    cross-source median consensus, scores each source (A/B/C/F), detects
    outlier bars and gaps, and optionally writes the consensus + quality
    scores to the database.

    Use this to populate the central data lake with audited, multi-sourced
    price history before running backtests or screeners.

    Args:
        ticker:     Asset symbol (e.g. 'AAPL', 'SPY', 'BTC-USD')
        interval:   Bar interval — '1d', '1wk', '1h', '5m', etc.
        start:      Start date YYYY-MM-DD (default: 1 year ago)
        end:        End date YYYY-MM-DD (default: today)
        write_to_db: Write consensus bars + quality scores to DB (default True)

    Returns:
        CleaningReport: sources_attempted, sources_with_data, consensus_bars,
        quality_scores (grade A/B/C/F per source), gaps_detected,
        best_source, wrote_to_db
    """
    from datetime import datetime, timedelta
    from sentinel.sds.data_cleaner import DataCleaner
    from sentinel.sds import build_default_adapters, get_all_adapters
    from sentinel.sds.db import get_session_factory

    if not get_all_adapters():
        build_default_adapters()

    end_dt = datetime.fromisoformat(end) if end else datetime.utcnow()
    start_dt = datetime.fromisoformat(start) if start else (end_dt - timedelta(days=365))

    session_factory = get_session_factory()
    async with session_factory() as session:
        cleaner = DataCleaner(session)
        report = await cleaner.clean_ticker(
            ticker=ticker.upper(),
            interval=interval,
            start=start_dt,
            end=end_dt,
            write_to_db=write_to_db,
        )

    result = report.to_dict()
    result["best_source"] = report.best_source()
    return result


@mcp.tool()
async def clean_universe(
    tickers_json: str,
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_concurrent: int = 5,
    write_to_db: bool = True,
) -> dict:
    """
    Run the SENTINEL DataCleaner across a universe of tickers.

    Identical to clean_data but processes a list of tickers with bounded
    concurrency. Ideal for populating the full data lake in one call.

    Args:
        tickers_json: JSON array of tickers, e.g. '["AAPL","MSFT","NVDA"]'
        interval:     Bar interval (default '1d')
        start:        Start date YYYY-MM-DD
        end:          End date YYYY-MM-DD
        max_concurrent: Max parallel fetches (default 5)
        write_to_db:  Write consensus + quality scores to DB (default True)

    Returns:
        summary: tickers_processed, grade_distribution, total_consensus_bars,
                 reports: per-ticker CleaningReport dicts
    """
    import json as _json
    from datetime import datetime, timedelta
    from sentinel.sds.data_cleaner import DataCleaner
    from sentinel.sds import build_default_adapters, get_all_adapters
    from sentinel.sds.db import get_session_factory

    if not get_all_adapters():
        build_default_adapters()

    tickers = _json.loads(tickers_json)
    end_dt = datetime.fromisoformat(end) if end else datetime.utcnow()
    start_dt = datetime.fromisoformat(start) if start else (end_dt - timedelta(days=365))

    session_factory = get_session_factory()
    async with session_factory() as session:
        reports = await DataCleaner(session).clean_universe(
            tickers=[t.upper() for t in tickers],
            interval=interval,
            start=start_dt,
            end=end_dt,
            max_concurrent=max_concurrent,
            write_to_db=write_to_db,
        )

    grade_dist: dict[str, int] = {"A": 0, "B": 0, "C": 0, "F": 0}
    total_bars = 0
    for r in reports:
        total_bars += r.consensus_bars
        for qs in r.quality_scores:
            grade_dist[qs.grade] = grade_dist.get(qs.grade, 0) + 1

    return {
        "tickers_processed": len(reports),
        "total_consensus_bars": total_bars,
        "grade_distribution": grade_dist,
        "reports": [r.to_dict() for r in reports],
    }


# ─── Tool 53: Continuous Futures ──────────────────────────────────────────────

@mcp.tool()
async def get_continuous_futures(
    root: str,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    method: str = "panama",
) -> dict:
    """
    Build a back-adjusted continuous futures price series from individual contracts.

    Stitches raw CME/NYMEX/CBOT contract bars stored in the SENTINEL ohlcv table
    into a single gapless series. Three adjustment methods:

    - **panama**  — additive back-adjustment (preserves $ differences; standard
                    for spread/basis strategies)
    - **ratio**   — multiplicative back-adjustment (preserves % returns; standard
                    for trend/momentum strategies)
    - **unadj**   — no adjustment; raw chain concatenation (best for volume/OI)

    Well-known roots: ES (S&P 500), NQ (Nasdaq), CL (Crude Oil),
                      GC (Gold), ZB (30Y Treasury)

    Args:
        root:   CME root symbol (e.g. 'ES', 'NQ', 'CL', 'GC', 'ZB')
        start:  Start date YYYY-MM-DD (default '2010-01-01')
        end:    End date YYYY-MM-DD (default today)
        method: 'panama' | 'ratio' | 'unadj'

    Returns:
        root, method, count, bars: [{time, open, high, low, close, volume}]
    """
    from datetime import date
    from sentinel.sds.continuous_futures import build_continuous_series, ContractSpec
    from sentinel.sds.db import get_session_factory

    if method not in ("panama", "ratio", "unadj"):
        return {"error": "method must be panama | ratio | unadj"}

    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end) if end else date.today()
    except ValueError as exc:
        return {"error": f"Invalid date: {exc}"}

    _WELL_KNOWN = {
        "ES": ContractSpec.es, "NQ": ContractSpec.nq,
        "CL": ContractSpec.cl, "GC": ContractSpec.gc, "ZB": ContractSpec.zb,
    }
    root_up = root.upper()
    spec_fn = _WELL_KNOWN.get(root_up)
    spec = spec_fn() if spec_fn else ContractSpec(
        root=root_up, exchange="CME", months=[3, 6, 9, 12]
    )

    session_factory = get_session_factory()
    async with session_factory() as session:
        bars = await build_continuous_series(session, spec, start_date, end_date, method)

    return {
        "root": root_up,
        "ticker": f"CONT:{root_up}1",
        "method": method,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "count": len(bars),
        "bars": [
            {
                "time": b.time.isoformat(),
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": b.volume,
            }
            for b in bars
        ],
    }


# ─── Tool 63: ETF Profile ────────────────────────────────────────────────────

@mcp.tool()
async def get_etf_profile(
    ticker: str,
) -> dict:
    """
    Full ETF profile: AUM, expense ratio, NAV premium/discount, top-10 holdings,
    sector weights, factor exposure (beta/size/value/momentum), performance, and
    estimated 30-day flows. Bloomberg ETF analytics equivalent at $0.

    Args:
        ticker: ETF ticker symbol (e.g. "SPY", "QQQ", "ARKK", "VXUS")
    """
    from sentinel.sfe.etf_analytics import get_etf_profile as _etf

    result = await _etf(ticker=ticker)
    return result.model_dump(mode="json")


# ─── Tool 64: ETF Comparison ──────────────────────────────────────────────────

@mcp.tool()
async def compare_etfs(
    tickers_json: str,
) -> dict:
    """
    Side-by-side ETF comparison: AUM, expense ratio, YTD/1Y return, Sharpe,
    top sector, and holdings count for a list of ETFs.

    Args:
        tickers_json: JSON list of ETF tickers, e.g. '["SPY","QQQ","IWM","DIA"]'
    """
    import json
    from sentinel.sfe.etf_analytics import compare_etfs as _cmp

    tickers = json.loads(tickers_json)
    result = await _cmp(tickers=tickers)
    return result.model_dump(mode="json")


# ─── Tool 65: Commodity Dashboard ────────────────────────────────────────────

@mcp.tool()
async def get_commodity_dashboard() -> dict:
    """
    Real-time commodity price dashboard: energy (WTI, nat gas, RBOB),
    metals (gold, silver, copper, platinum), agriculture (corn, wheat, soybeans).
    Includes futures term structure (contango/backwardation) and commodity
    regime detection (inflationary/deflationary/neutral) via FRED PPI vs CPI.
    """
    from sentinel.sma.commodity_analytics import get_commodity_dashboard as _cmd

    result = await _cmd()
    return result.model_dump(mode="json")


# ─── Tool 61: M&A Deal Screener ──────────────────────────────────────────────

@mcp.tool()
async def screen_ma_deals(
    days_back: int = 30,
    min_value_billions: Optional[float] = None,
    sector: Optional[str] = None,
    deal_type: Optional[str] = None,
    limit: int = 25,
) -> dict:
    """
    Screen recent M&A deals from EDGAR 8-K, DEFM14A, and SC TO-T filings.
    Extracts deal type, value, acquirer/target, and current status.

    Args:
        days_back:           Days of history to scan (default 30)
        min_value_billions:  Minimum deal size in $B (default None = all)
        sector:              GICS sector filter (default None = all)
        deal_type:           "merger" | "acquisition" | "spinoff" | "divestiture" | None
        limit:               Max results (default 25)
    """
    from sentinel.sfe.ma_screener import screen_ma_deals as _screen

    result = await _screen(
        days_back=days_back, min_value_billions=min_value_billions,
        sector=sector, deal_type=deal_type, limit=limit,
    )
    return result.model_dump(mode="json")


# ─── Tool 62: M&A Target Profile ─────────────────────────────────────────────

@mcp.tool()
async def get_ma_profile(
    ticker: str,
) -> dict:
    """
    M&A target profile for a stock: recent deal activity, activist investor
    interest (13D/13G filings), and corporate defense mechanisms from DEF14A.

    Args:
        ticker: Stock ticker symbol (e.g. "ATVI", "MSFT", "VMW")
    """
    from sentinel.sfe.ma_screener import get_ma_profile as _profile

    result = await _profile(ticker=ticker)
    return result.model_dump(mode="json")


# ─── Tool 59: Bond Price Analytics ──────────────────────────────────────────

@mcp.tool()
async def compute_bond_price(
    face_value: float = 1000.0,
    coupon_rate: float = 0.05,
    years_to_maturity: float = 10.0,
    yield_to_maturity: Optional[float] = None,
    frequency: int = 2,
    credit_spread_bps: float = 0.0,
) -> dict:
    """
    Full fixed income analytics for a bond: price, Macaulay/modified/effective
    duration, convexity, DV01, and ±100/200bps rate shock scenarios.
    If YTM is omitted, interpolates from live FRED Treasury curve.

    Args:
        face_value:         Face value in USD (default 1000)
        coupon_rate:        Annual coupon rate, e.g. 0.05 = 5% (default 0.05)
        years_to_maturity:  Years to maturity (default 10)
        yield_to_maturity:  Override YTM; None = use FRED curve interpolation
        frequency:          Coupon payments per year — 1 annual, 2 semi-annual (default 2)
        credit_spread_bps:  Credit spread added to Treasury yield (default 0)
    """
    from sentinel.sfe.bond_analytics import compute_bond_price_analytics

    result = await compute_bond_price_analytics(
        face_value=face_value, coupon_rate=coupon_rate,
        years_to_maturity=years_to_maturity, yield_to_maturity=yield_to_maturity,
        frequency=frequency, credit_spread_bps=credit_spread_bps,
    )
    return result.model_dump(mode="json")


# ─── Tool 60: Altman Z-Score ──────────────────────────────────────────────────

@mcp.tool()
async def compute_altman_z(
    ticker: str,
) -> dict:
    """
    Altman Z-score credit risk model: classifies public companies as safe,
    grey-zone, or financial distress based on five balance sheet ratios.
    Z > 2.99 = safe, 1.81-2.99 = grey, < 1.81 = distress.

    Args:
        ticker: Stock ticker symbol (e.g. "F", "NFLX", "AAPL")
    """
    from sentinel.sfe.bond_analytics import compute_altman_z as _az

    result = await _az(ticker=ticker)
    return result.model_dump(mode="json")


# ─── Tool 57: GARCH-VaR ──────────────────────────────────────────────────────

@mcp.tool()
async def compute_garch_var(
    tickers_json: str,
    weights_json: Optional[str] = None,
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float = 1_000_000,
    backtest_days: int = 252,
) -> dict:
    """
    GARCH(1,1) conditional VaR: accounts for volatility clustering, provides
    time-varying risk estimates, stress VaR, and Basel III backtesting.
    More accurate than static historical/parametric VaR under market stress.

    Args:
        tickers_json:      JSON list of tickers, e.g. '["AAPL","MSFT"]'
        weights_json:      JSON list of weights (equal-weight if omitted)
        confidence:        VaR confidence level (default 0.95)
        horizon_days:      VaR horizon in days (default 1)
        portfolio_value:   Portfolio value in USD (default $1M)
        backtest_days:     Days to backtest VaR model (default 252)
    """
    import json
    from sentinel.spr.garch_var import compute_garch_var as _gvar

    tickers = json.loads(tickers_json)
    weights = json.loads(weights_json) if weights_json else None
    result = await _gvar(
        tickers=tickers, weights=weights, confidence=confidence,
        horizon_days=horizon_days, portfolio_value=portfolio_value,
        backtest_days=backtest_days,
    )
    return result.model_dump(mode="json")


# ─── Tool 58: Advanced TA ────────────────────────────────────────────────────

@mcp.tool()
async def get_advanced_ta(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> dict:
    """
    Advanced technical analysis: Ichimoku Cloud, Fibonacci retracements,
    ADX (trend strength), and Parabolic SAR — composite bull/bear score.

    Args:
        ticker:   Stock/ETF ticker symbol
        period:   History period — "6mo", "1y", "2y" (default "1y")
        interval: Bar interval — "1d", "1h" (default "1d")
    """
    from sentinel.spr.ta_advanced import get_advanced_ta as _ata

    result = await _ata(ticker=ticker, period=period, interval=interval)
    return result.model_dump(mode="json")


# ─── Tool 54: Technical Analysis Signals ─────────────────────────────────────

@mcp.tool()
async def get_ta_signals(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> dict:
    """
    Technical analysis signals: RSI, MACD, Bollinger Bands, VWAP, Stochastic,
    Williams %R, OBV, ATR, moving averages. Returns composite bullish/bearish score.

    Args:
        ticker:   Stock/ETF ticker symbol
        period:   History period — "1mo", "3mo", "6mo", "1y", "2y" (default "1y")
        interval: Bar interval — "1d", "1h", "15m" (default "1d")
    """
    from sentinel.spr.ta_engine import get_ta_signals as _ta

    result = await _ta(ticker=ticker, period=period, interval=interval)
    return result.model_dump(mode="json")


# ─── Tool 55: Yield Curve Analytics ──────────────────────────────────────────

@mcp.tool()
async def get_yield_curve(
    as_of: Optional[str] = None,
    include_history_days: int = 90,
) -> dict:
    """
    Live US Treasury yield curve with Nelson-Siegel fit, forward rates, and
    credit spreads (IG/HY OAS). Flags curve inversion and recession signals.

    Args:
        as_of:                ISO date string (default = today)
        include_history_days: Days of slope history to return (default 90)
    """
    from sentinel.sfe.yield_curve import get_yield_curve as _yc

    result = await _yc(as_of=as_of, include_history_days=include_history_days)
    return result.model_dump(mode="json")


# ─── Tool 56: Research Synthesis ─────────────────────────────────────────────

@mcp.tool()
async def synthesize_research(
    ticker: str,
    query: Optional[str] = None,
    include_filings: bool = True,
    include_news: bool = True,
    include_fundamentals: bool = True,
    include_insider: bool = True,
    max_tokens: int = 2048,
) -> dict:
    """
    Multi-source AI research synthesis: gathers SEC filings, news, fundamentals,
    and insider data then uses Claude to produce bull/bear analysis, key risks,
    and catalyst watch. Bloomberg Terminal has no equivalent.

    Args:
        ticker:               Stock ticker to research
        query:                Optional focus question (e.g. "What are the margin risks?")
        include_filings:      Include SEC filings in context (default True)
        include_news:         Include recent news in context (default True)
        include_fundamentals: Include comps/fundamental data (default True)
        include_insider:      Include insider transaction context (default True)
        max_tokens:           Max tokens for synthesis response (default 2048)
    """
    from sentinel.sil.research_synthesis import synthesize_research as _synth

    result = await _synth(
        ticker=ticker, query=query,
        include_filings=include_filings, include_news=include_news,
        include_fundamentals=include_fundamentals, include_insider=include_insider,
        max_tokens=max_tokens,
    )
    return result.model_dump(mode="json")


# ─── Tool 66: IFRS Fundamentals ──────────────────────────────────────────────

@mcp.tool()
async def get_ifrs_fundamentals(
    ticker: str,
    periods: int = 4,
) -> dict:
    """
    IFRS financial statements for non-US companies via SEC 20-F XBRL filings.
    Returns income statement, balance sheet, and cash flow for up to 4 annual periods.
    Covers revenue, gross profit, net income, assets, equity, and operating cash flow.
    Computes revenue growth, gross margin, net margin, ROE, debt/equity ratios.

    Args:
        ticker:  Ticker of non-US ADR or foreign private issuer (e.g. "ASML", "BABA", "NVO")
        periods: Annual periods to return (default 4)
    """
    from sentinel.sfe.ifrs_fundamentals import get_ifrs_fundamentals as _ifrs

    result = await _ifrs(ticker=ticker, periods=periods)
    return result.model_dump(mode="json")


# ─── Tool 67: Economic Forecasting ───────────────────────────────────────────

@mcp.tool()
async def get_econ_forecast(
    series_ids_json: Optional[str] = None,
    horizon: int = 6,
    ar_order: int = 3,
    include_var: bool = True,
) -> dict:
    """
    Macro economic forecasting via AR/VAR models on FRED data.
    Forecasts unemployment, CPI, Fed Funds, Treasury spread, industrial production.
    Includes a nowcast expansion/contraction index from current indicators.

    Args:
        series_ids_json: JSON list of FRED series IDs, e.g. '["UNRATE","CPIAUCSL"]'
                         Default: UNRATE, CPIAUCSL, FEDFUNDS, T10Y2Y, INDPRO
        horizon:         Forecast periods ahead (default 6)
        ar_order:        AR/VAR lag order (default 3)
        include_var:     Also run multivariate VAR jointly (default True)
    """
    import json
    from sentinel.sma.econ_forecasting import get_econ_forecast as _econ

    series_ids = json.loads(series_ids_json) if series_ids_json else None
    result = await _econ(
        series_ids=series_ids, horizon=horizon,
        ar_order=ar_order, include_var=include_var,
    )
    return result.model_dump(mode="json")


# ─── Tool 68: Liquidity Metrics ──────────────────────────────────────────────

@mcp.tool()
async def get_liquidity_metrics(
    ticker: str,
    period_days: int = 63,
) -> dict:
    """
    Market microstructure liquidity metrics: Amihud illiquidity ratio, Roll's spread,
    Corwin-Schultz bid-ask spread proxy, Kyle's lambda (price impact), turnover ratio,
    and average daily dollar volume. Classifies liquidity as high/medium/low/illiquid.

    Args:
        ticker:      Stock ticker (e.g. "AAPL", "GME", "TSLA")
        period_days: Trading days to analyze (default 63 ≈ 3 months)
    """
    from sentinel.spr.liquidity_analytics import get_liquidity_metrics as _liq

    result = await _liq(ticker=ticker, period_days=period_days)
    return result.model_dump(mode="json")


# ─── Tool 69: Portfolio Liquidity ────────────────────────────────────────────

@mcp.tool()
async def get_portfolio_liquidity(
    tickers_json: str,
    weights_json: Optional[str] = None,
    period_days: int = 63,
) -> dict:
    """
    Portfolio-level liquidity analysis: weighted aggregate of Amihud, bid-ask spread,
    Kyle's lambda, and ADV across all holdings. Estimates days-to-liquidate 90% of a
    $10M portfolio assuming 20% of ADV per day.

    Args:
        tickers_json: JSON list of tickers, e.g. '["AAPL","MSFT","GME"]'
        weights_json: JSON list of weights summing to 1 (default equal-weight)
        period_days:  Trading days to analyze (default 63)
    """
    import json
    from sentinel.spr.liquidity_analytics import get_portfolio_liquidity as _pliq

    tickers = json.loads(tickers_json)
    weights = json.loads(weights_json) if weights_json else None
    result = await _pliq(tickers=tickers, weights=weights, period_days=period_days)
    return result.model_dump(mode="json")


# ─── Tool 70: Check Price Alerts ─────────────────────────────────────────────

@mcp.tool()
async def check_alerts(
    tickers_json: Optional[str] = None,
) -> dict:
    """
    Poll all active price alerts against current market data. Returns triggered alerts
    with current price and trigger reason. Supports price thresholds, percent change,
    RSI overbought/oversold, MA crossovers, and volume spikes.

    Args:
        tickers_json: JSON list of tickers to check (default: all active tickers)
    """
    import json
    from sentinel.sil.price_alerts import check_alerts as _check

    tickers = json.loads(tickers_json) if tickers_json else None
    result = await _check(tickers=tickers)
    return result.model_dump(mode="json")


# ─── Tool 71: Create Price Alert ─────────────────────────────────────────────

@mcp.tool()
async def create_alert(
    ticker: str,
    alert_type: str,
    threshold: Optional[float] = None,
    note: str = "",
    params_json: Optional[str] = None,
) -> dict:
    """
    Create a persistent price alert. Alert types: price_above, price_below,
    pct_change_up, pct_change_down, volume_spike, rsi_overbought, rsi_oversold,
    ma_crossover, ma_crossunder.

    Args:
        ticker:      Stock ticker (e.g. "AAPL")
        alert_type:  One of the AlertType enum values above
        threshold:   Price level or % threshold (required for price/pct alerts)
        note:        Optional user note
        params_json: JSON dict of extra params, e.g. '{"multiplier": 2.5}' for volume_spike
    """
    import json
    from sentinel.sil.price_alerts import create_alert as _create

    params = json.loads(params_json) if params_json else None
    result = await _create(
        ticker=ticker, alert_type=alert_type,
        threshold=threshold, note=note, params=params,
    )
    return result.model_dump(mode="json")


# ─── Tool 72: List Alerts ─────────────────────────────────────────────────────

@mcp.tool()
async def list_alerts(
    ticker: Optional[str] = None,
    active_only: bool = True,
) -> list:
    """
    List all persisted price alerts, optionally filtered by ticker.

    Args:
        ticker:      Filter to specific ticker (default None = all tickers)
        active_only: Only return untripped alerts (default True)
    """
    from sentinel.sil.price_alerts import list_alerts as _list

    results = await _list(ticker=ticker, active_only=active_only)
    return [r.model_dump(mode="json") for r in results]


# ─── Tool 73: Delete Alert ────────────────────────────────────────────────────

@mcp.tool()
async def delete_alert(
    alert_id: str,
) -> dict:
    """
    Delete a price alert by its UUID.

    Args:
        alert_id: UUID of the alert to delete (from create_alert or list_alerts)
    """
    from sentinel.sil.price_alerts import delete_alert as _del

    deleted = await _del(alert_id=alert_id)
    return {"deleted": deleted, "alert_id": alert_id}


# ─── Tool 74: Form ADV RIA Profile ───────────────────────────────────────────

@mcp.tool()
async def get_ria_profile(firm_name: str) -> dict:
    """
    Look up a Registered Investment Adviser (RIA) by name via SEC IAPD.
    Returns AUM, client count, fee structure, investment styles, and registration info.
    Equivalent to CapIQ RIA intelligence module at $0.

    Args:
        firm_name: RIA firm name (e.g. "Bridgewater", "Vanguard", "BlackRock")
    """
    from sentinel.sfe.form_adv import get_ria_profile as _ria

    result = await _ria(firm_name=firm_name)
    return result.model_dump(mode="json")


# ─── Tool 75: RIA Screener ────────────────────────────────────────────────────

@mcp.tool()
async def screen_rias(
    query: str,
    min_aum_billions: Optional[float] = None,
    max_aum_billions: Optional[float] = None,
    state: Optional[str] = None,
    limit: int = 10,
) -> dict:
    """
    Screen Registered Investment Advisers by name, AUM range, and state.
    Data from SEC IAPD (Investment Adviser Public Disclosure) — 100% free.

    Args:
        query:             Search term (firm name or partial name)
        min_aum_billions:  Minimum AUM in $B (default None = no filter)
        max_aum_billions:  Maximum AUM in $B (default None = no filter)
        state:             2-letter state code (e.g. "NY", "CA")
        limit:             Max results (default 10)
    """
    from sentinel.sfe.form_adv import screen_rias as _screen

    result = await _screen(
        query=query, min_aum_billions=min_aum_billions,
        max_aum_billions=max_aum_billions, state=state, limit=limit,
    )
    return result.model_dump(mode="json")


# ─── Tool 76: Country Geopolitical Risk ───────────────────────────────────────

@mcp.tool()
async def get_country_risk(
    country: str,
    lookback_days: int = 30,
) -> dict:
    """
    Score geopolitical risk for a country using GDELT news event analysis
    + Claude Haiku narrative. Returns conflict/political/economic sub-scores,
    30-day event timeline, and investment risk narrative.

    Args:
        country:       Country name (e.g. "Russia", "Iran", "Taiwan", "Ukraine")
        lookback_days: Days of GDELT history to analyze (default 30)
    """
    from sentinel.sma.geopolitical_risk import get_country_risk as _risk

    result = await _risk(country=country, lookback_days=lookback_days)
    return result.model_dump(mode="json")


# ─── Tool 77: Geopolitical Dashboard ─────────────────────────────────────────

@mcp.tool()
async def get_geopolitical_dashboard(
    countries_json: Optional[str] = None,
) -> dict:
    """
    Multi-country geopolitical risk dashboard via GDELT.
    Scores 8 default countries (US, CN, RU, IR, SA, UA, IL, TW) and
    computes global risk index, flashpoints, and highest-risk regions.

    Args:
        countries_json: JSON list of country names (default: 8 major risk regions)
    """
    import json
    from sentinel.sma.geopolitical_risk import get_geopolitical_dashboard as _dash

    countries = json.loads(countries_json) if countries_json else None
    result = await _dash(countries=countries)
    return result.model_dump(mode="json")


# ─── Tool 78: LBO Model ───────────────────────────────────────────────────────

@mcp.tool()
async def run_lbo_model(
    purchase_price: float,
    ebitda: float,
    ebitda_growth_rate: float = 0.05,
    leverage_multiple: float = 5.0,
    interest_rate: float = 0.085,
    hold_years: int = 5,
    exit_multiple: float = 8.0,
    tax_rate: float = 0.25,
) -> dict:
    """
    Run a full LBO model: debt schedule, interest waterfall, FCF, equity exit,
    IRR and MOIC. Pure financial model — no external data needed.
    Equivalent to CapIQ LBO template at $0.

    Args:
        purchase_price:      Total enterprise value paid ($M)
        ebitda:              Entry year EBITDA ($M)
        ebitda_growth_rate:  Annual EBITDA growth (e.g. 0.05 = 5%)
        leverage_multiple:   Debt/EBITDA at entry (e.g. 5.0)
        interest_rate:       Annual interest rate on debt (e.g. 0.085 = 8.5%)
        hold_years:          Investment horizon in years (e.g. 5)
        exit_multiple:       EV/EBITDA at exit (e.g. 8.0)
        tax_rate:            Corporate tax rate (default 0.25)
    """
    from sentinel.sfe.lbo_model import LBOAssumptions, run_lbo_model as _lbo

    assumptions = LBOAssumptions(
        purchase_price=purchase_price, ebitda=ebitda,
        ebitda_growth_rate=ebitda_growth_rate, leverage_multiple=leverage_multiple,
        interest_rate=interest_rate, hold_years=hold_years,
        exit_multiple=exit_multiple, tax_rate=tax_rate,
    )
    result = _lbo(assumptions=assumptions)
    return result.model_dump(mode="json")


# ─── Tool 79: Merger Model ────────────────────────────────────────────────────

@mcp.tool()
async def run_merger_model(
    acquirer_eps: float,
    acquirer_shares_mm: float,
    acquirer_price: float,
    target_eps: float,
    target_shares_mm: float,
    acquisition_price_per_share: float,
    pct_stock: float = 0.0,
    synergies_after_tax_mm: float = 0.0,
    cost_of_debt: float = 0.08,
    tax_rate: float = 0.25,
) -> dict:
    """
    M&A accretion/dilution analysis: tests whether an acquisition is EPS accretive
    or dilutive to the acquirer. Computes combined EPS, premium paid, and deal economics.

    Args:
        acquirer_eps:                  Acquirer EPS (trailing 12m)
        acquirer_shares_mm:            Acquirer diluted shares outstanding (millions)
        acquirer_price:                Acquirer stock price
        target_eps:                    Target EPS (trailing 12m)
        target_shares_mm:              Target diluted shares outstanding (millions)
        acquisition_price_per_share:   Offer price per target share
        pct_stock:                     Fraction of deal paid in acquirer stock (0.0-1.0)
        synergies_after_tax_mm:        After-tax annual synergies ($M)
        cost_of_debt:                  Interest rate on cash consideration debt
        tax_rate:                      Combined tax rate (default 0.25)
    """
    from sentinel.sfe.lbo_model import MergerAssumptions, run_merger_model as _merger

    assumptions = MergerAssumptions(
        acquirer_eps=acquirer_eps, acquirer_shares_mm=acquirer_shares_mm,
        acquirer_price=acquirer_price, target_eps=target_eps,
        target_shares_mm=target_shares_mm,
        acquisition_price_per_share=acquisition_price_per_share,
        pct_stock=pct_stock, synergies_after_tax_mm=synergies_after_tax_mm,
        cost_of_debt=cost_of_debt, tax_rate=tax_rate,
    )
    result = _merger(assumptions=assumptions)
    return result.model_dump(mode="json")


# ─── Tool 80: LBO Candidate Screener ─────────────────────────────────────────

@mcp.tool()
async def screen_lbo_candidate(ticker: str) -> dict:
    """
    Auto-populate and run an LBO model using live financial data for a public company.
    Fetches EV, EBITDA, and multiples from yfinance. Returns IRR, MOIC, and debt schedule.

    Args:
        ticker: Stock ticker (e.g. "DELL", "CCL", "HCA", "PVH")
    """
    from sentinel.sfe.lbo_model import screen_lbo_candidate as _screen

    result = await _screen(ticker=ticker)
    return result.model_dump(mode="json")


# ─── Tool 81: EDGAR Recent Filings ───────────────────────────────────────────

@mcp.tool()
async def get_recent_filings(
    form_types_json: Optional[str] = None,
    days_back: int = 1,
    limit: int = 50,
) -> dict:
    """
    Real-time EDGAR filing monitor: fetches latest SEC filings across all companies.
    Covers 8-K (material events), 10-K/10-Q, SC 13D (activist), S-1 (IPO), Form 4.
    Equivalent to Bloomberg filing alerts at $0.

    Args:
        form_types_json: JSON list of form types, e.g. '["8-K","SC 13D","S-1"]'
                         Default: all high-priority form types
        days_back:       Days of history to fetch (default 1 = today's filings)
        limit:           Max filings to return (default 50)
    """
    import json
    from sentinel.sil.edgar_monitor import get_recent_filings as _filings

    form_types = json.loads(form_types_json) if form_types_json else None
    result = await _filings(form_types=form_types, days_back=days_back, limit=limit)
    return result.model_dump(mode="json")


# ─── Tool 82: EDGAR Watchlist Monitor ────────────────────────────────────────

@mcp.tool()
async def monitor_watchlist(
    tickers_json: str,
    form_types_json: Optional[str] = None,
    days_back: int = 7,
) -> list:
    """
    Monitor specific tickers for new EDGAR filings. Returns one result per ticker
    with all new filings and high-priority alerts (activist, M&A, insider, IPO).

    Args:
        tickers_json:    JSON list of tickers, e.g. '["AAPL","MSFT","TSLA"]'
        form_types_json: JSON list of form types to watch (default: all types)
        days_back:       Days of history (default 7)
    """
    import json
    from sentinel.sil.edgar_monitor import monitor_watchlist as _watch

    tickers = json.loads(tickers_json)
    form_types = json.loads(form_types_json) if form_types_json else None
    results = await _watch(tickers=tickers, form_types=form_types, days_back=days_back)
    return [r.model_dump(mode="json") for r in results]


# ─── Tool 83: Insider Transactions ───────────────────────────────────────────

@mcp.tool()
async def get_insider_transactions(
    ticker: str,
    days_back: int = 30,
    limit: int = 20,
) -> list:
    """
    Form 4 insider transactions for a specific ticker from EDGAR.
    Returns purchases, sales, and grants by officers and directors.

    Args:
        ticker:    Stock ticker (e.g. "AAPL", "TSLA")
        days_back: Days of history (default 30)
        limit:     Max transactions (default 20)
    """
    from sentinel.sil.edgar_monitor import get_insider_transactions as _insider

    results = await _insider(ticker=ticker, days_back=days_back, limit=limit)
    return [r.model_dump(mode="json") for r in results]


# ─── Tool 84: FX Pair Analytics ──────────────────────────────────────────────

@mcp.tool()
async def get_fx_pair(
    pair: str,
    history_days: int = 90,
) -> dict:
    """
    Deep FX analytics for a currency pair: spot rate, forward curve (1M/3M/6M/1Y),
    realized vol, carry signal, and momentum z-score. Uses Frankfurter ECB rates
    + FRED interest rate differentials + yfinance vol estimation.

    Args:
        pair:         Currency pair, e.g. "EURUSD", "GBPUSD", "USDJPY"
        history_days: Days of history for vol/momentum calculation (default 90)
    """
    from sentinel.sfe.fx_analytics import get_fx_pair as _fx

    result = await _fx(pair=pair, history_days=history_days)
    return result.model_dump(mode="json")


# ─── Tool 85: FX Dashboard ────────────────────────────────────────────────────

@mcp.tool()
async def get_fx_dashboard(
    pairs_json: Optional[str] = None,
    history_days: int = 90,
) -> dict:
    """
    Multi-currency FX dashboard: spot rates, forward curves, vol surface, carry and
    momentum signals for major USD pairs. Includes DXY proxy and USD trend signal.

    Args:
        pairs_json:   JSON list of pairs (default: EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, USDCNY, USDMXN)
        history_days: History for vol/momentum (default 90)
    """
    import json
    from sentinel.sfe.fx_analytics import get_fx_dashboard as _dash

    pairs = json.loads(pairs_json) if pairs_json else None
    result = await _dash(pairs=pairs, history_days=history_days)
    return result.model_dump(mode="json")


# ─── Tool 86: Form D Private Company ─────────────────────────────────────────

@mcp.tool()
async def get_company_form_d(
    company_name: str,
    limit: int = 5,
) -> list:
    """
    SEC Form D (Regulation D) filings for a private company: raise amount, offering type,
    exemption (506b/506c), key persons, and state. Covers VC-backed startups, hedge funds,
    PE funds, and private placements. Free PitchBook equivalent.

    Args:
        company_name: Company name to search (e.g. "OpenAI", "Anthropic", "Stripe")
        limit:        Max filings to return (default 5)
    """
    from sentinel.sfe.form_d import get_company_form_d as _formd

    results = await _formd(company_name=company_name, limit=limit)
    return [r.model_dump(mode="json") for r in results]


# ─── Tool 87: Private Market Screener ────────────────────────────────────────

@mcp.tool()
async def screen_private_market(
    query: Optional[str] = None,
    state: Optional[str] = None,
    min_amount_mm: Optional[float] = None,
    fund_type: Optional[str] = None,
    days_back: int = 30,
    limit: int = 25,
) -> dict:
    """
    Screen recent SEC Form D filings to monitor new VC raises, hedge fund launches,
    PE deals, and private placements. Aggregates by state and offering type.

    Args:
        query:         Search term (company name or keyword)
        state:         2-letter state code (e.g. "CA", "NY")
        min_amount_mm: Minimum raise size in $M
        fund_type:     "Hedge Fund" | "Venture Capital Fund" | "Private Equity Fund"
        days_back:     Days of recent filings to scan (default 30)
        limit:         Max results (default 25)
    """
    from sentinel.sfe.form_d import screen_private_market as _screen

    result = await _screen(
        query=query, state=state, min_amount_mm=min_amount_mm,
        fund_type=fund_type, days_back=days_back, limit=limit,
    )
    return result.model_dump(mode="json")


# ─── Tool 88: Scenario Analysis ───────────────────────────────────────────────

@mcp.tool()
async def run_scenario(
    tickers_json: str,
    weights_json: Optional[str] = None,
    scenario_name: Optional[str] = None,
    portfolio_value: float = 1_000_000,
    equity_shock_pct: float = 0.0,
    rate_shock_bps: float = 0.0,
    credit_spread_bps: float = 0.0,
    usd_shock_pct: float = 0.0,
    oil_shock_pct: float = 0.0,
    gold_shock_pct: float = 0.0,
) -> dict:
    """
    Macro scenario analysis: apply factor shocks to a portfolio using OLS betas.
    Templates: 2008_crisis, covid_crash, rate_hike_200bps, soft_landing,
    stagflation, china_taiwan, usd_crash. Or specify custom shocks directly.

    Args:
        tickers_json:    JSON list of tickers, e.g. '["AAPL","TLT","GLD"]'
        weights_json:    JSON list of weights summing to 1 (default equal-weight)
        scenario_name:   Template name (e.g. "2008_crisis") — overrides custom shocks
        portfolio_value: Portfolio size in USD (default $1M)
        equity_shock_pct: S&P 500 % return shock
        rate_shock_bps:   10Y Treasury rate change in bps
        credit_spread_bps: IG spread widening in bps
        usd_shock_pct:    USD index % change
        oil_shock_pct:    WTI % change
        gold_shock_pct:   Gold % change
    """
    import json
    from sentinel.spr.scenario_analysis import MacroShock, run_scenario as _scenario

    tickers = json.loads(tickers_json)
    weights = json.loads(weights_json) if weights_json else None
    shock = None if scenario_name else MacroShock(
        equity_shock_pct=equity_shock_pct, rate_shock_bps=rate_shock_bps,
        credit_spread_bps=credit_spread_bps, usd_shock_pct=usd_shock_pct,
        oil_shock_pct=oil_shock_pct, gold_shock_pct=gold_shock_pct,
        scenario_name="Custom",
    )
    result = await _scenario(
        tickers=tickers, weights=weights, shock=shock,
        scenario_name=scenario_name, portfolio_value=portfolio_value,
    )
    return result.model_dump(mode="json")


# ─── Tool 89: Multi-Scenario Comparison ──────────────────────────────────────

@mcp.tool()
async def run_multi_scenario(
    tickers_json: str,
    weights_json: Optional[str] = None,
    scenario_names_json: Optional[str] = None,
    portfolio_value: float = 1_000_000,
) -> dict:
    """
    Run all macro scenario templates against a portfolio simultaneously.
    Returns P&L for each scenario, best/worst outcomes, and most resilient scenario.

    Args:
        tickers_json:       JSON list of tickers
        weights_json:       JSON list of weights (default equal-weight)
        scenario_names_json: JSON list of template names (default: all 7 templates)
        portfolio_value:    Portfolio size in USD (default $1M)
    """
    import json
    from sentinel.spr.scenario_analysis import run_multi_scenario as _multi

    tickers = json.loads(tickers_json)
    weights = json.loads(weights_json) if weights_json else None
    scenario_names = json.loads(scenario_names_json) if scenario_names_json else None
    result = await _multi(
        tickers=tickers, weights=weights,
        scenario_names=scenario_names, portfolio_value=portfolio_value,
    )
    return result.model_dump(mode="json")


# ─── Tool 90: Dividend Analytics ─────────────────────────────────────────────

@mcp.tool()
async def get_dividend_analytics(ticker: str) -> dict:
    """
    Full dividend and corporate action analytics: yield, growth rate (5Y CAGR),
    consistency score, payout ratio, dividend quality score (0-10), DDM intrinsic
    value, and recent corporate actions (splits, special dividends, cuts).

    Args:
        ticker: Stock ticker (e.g. "AAPL", "JNJ", "KO", "T")
    """
    from sentinel.sfe.corporate_actions import get_dividend_analytics as _div

    result = await _div(ticker=ticker)
    return result.model_dump(mode="json")


# ─── Tool 91: Dividend Screener ───────────────────────────────────────────────

@mcp.tool()
async def screen_dividends(
    tickers_json: str,
    min_yield_pct: Optional[float] = None,
    min_quality_score: Optional[float] = None,
    exclude_no_dividend: bool = True,
) -> dict:
    """
    Screen multiple tickers for dividend quality: yield, 5Y growth rate, payout ratio,
    quality score, and DDM verdict. Identify high-quality dividend compounders.

    Args:
        tickers_json:       JSON list of tickers, e.g. '["JNJ","KO","PEP","MCD","PG"]'
        min_yield_pct:      Minimum dividend yield % filter (default None)
        min_quality_score:  Minimum quality score 0-10 (default None)
        exclude_no_dividend: Exclude tickers with no dividend (default True)
    """
    import json
    from sentinel.sfe.corporate_actions import screen_dividends as _screen

    tickers = json.loads(tickers_json)
    result = await _screen(
        tickers=tickers, min_yield_pct=min_yield_pct,
        min_quality_score=min_quality_score, exclude_no_dividend=exclude_no_dividend,
    )
    return result.model_dump(mode="json")


# ─── Tool 92: Options Flow Analysis ─────────────────────────────────────────

@mcp.tool()
async def get_options_flow(
    ticker: str,
    min_unusual_score: float = 3.0,
    max_expirations: int = 3,
) -> dict:
    """
    Deep options flow analysis: unusual activity scoring (0-10), max pain, IV skew,
    put/call ratios, and unusual contract screener. Free Bloomberg OVDV equivalent.

    Args:
        ticker:            Stock ticker (e.g. "AAPL", "SPY", "NVDA")
        min_unusual_score: Minimum unusual activity score to include (default 3.0)
        max_expirations:   Number of nearest expiration dates to scan (default 3)
    """
    from sentinel.sbx.options_flow import get_options_flow as _flow

    result = await _flow(ticker=ticker, min_unusual_score=min_unusual_score,
                         max_expirations=max_expirations)
    return result.model_dump(mode="json")


# ─── Tool 93: Options Flow Screen ────────────────────────────────────────────

@mcp.tool()
async def screen_options_flow_unusual(
    tickers_json: str,
    min_unusual_score: float = 5.0,
) -> dict:
    """
    Multi-ticker unusual options activity screener: surface tickers with abnormal
    vol/OI ratios, large dollar premiums, or extreme IV. Returns most bullish,
    most bearish, and highest dollar-premium contracts.

    Args:
        tickers_json:      JSON list of tickers, e.g. '["SPY","AAPL","NVDA","TSLA"]'
        min_unusual_score: Minimum unusual score threshold (default 5.0)
    """
    import json
    from sentinel.sbx.options_flow import screen_options_flow as _screen

    tickers = json.loads(tickers_json)
    result = await _screen(tickers=tickers, min_unusual_score=min_unusual_score)
    return result.model_dump(mode="json")


# ─── Tool 94: Credit Analytics (Merton Model) ────────────────────────────────

@mcp.tool()
async def get_credit_analytics(ticker: str) -> dict:
    """
    Structural credit analysis using Merton's model: asset value, distance-to-default,
    probability of default, CDS spread proxy, Altman Z-score, and credit tier
    (AAA–D). Free Bloomberg CRPR equivalent.

    Args:
        ticker: Stock ticker (e.g. "AAPL", "GE", "HCA", "F")
    """
    from sentinel.spr.credit_analytics import get_credit_analytics as _credit

    result = await _credit(ticker=ticker)
    return result.model_dump(mode="json")


# ─── Tool 95: Credit Quality Screener ────────────────────────────────────────

@mcp.tool()
async def screen_credit(
    tickers_json: str,
    max_credit_score: Optional[float] = None,
    min_credit_score: Optional[float] = None,
) -> dict:
    """
    Screen multiple tickers for credit quality: Merton PD, CDS proxy, Altman Z,
    debt/equity, interest coverage. Identify distressed credits or investment-grade
    compounders.

    Args:
        tickers_json:    JSON list of tickers, e.g. '["AAPL","GE","F","BA","DAL"]'
        max_credit_score: Max credit score filter (lower = more distressed)
        min_credit_score: Min credit score filter (higher = investment grade)
    """
    import json
    from sentinel.spr.credit_analytics import screen_credit as _screen

    tickers = json.loads(tickers_json)
    result = await _screen(tickers=tickers, max_credit_score=max_credit_score,
                           min_credit_score=min_credit_score)
    return result.model_dump(mode="json")


# ─── Tool 96: Auto Peer Comparison ───────────────────────────────────────────

@mcp.tool()
async def get_peer_comparison(
    ticker: str,
    custom_peers_json: Optional[str] = None,
    max_peers: int = 7,
) -> dict:
    """
    Automatic peer comparison: industry-based peer discovery, 17-metric comparison
    table (valuation, growth, profitability, health), percentile rankings, and
    overall verdict. Free CapIQ comps equivalent.

    Args:
        ticker:           Subject ticker (e.g. "NVDA", "AAPL", "JPM")
        custom_peers_json: JSON list of custom peer tickers (overrides auto-discovery)
        max_peers:        Max number of peers to include (default 7)
    """
    import json
    from sentinel.sfe.peer_comparison import get_peer_comparison as _peers

    custom_peers = json.loads(custom_peers_json) if custom_peers_json else None
    result = await _peers(ticker=ticker, custom_peers=custom_peers, max_peers=max_peers)
    return result.model_dump(mode="json")


# ─── Tool 109: VIX Term Structure Analytics ──────────────────────────────────

@mcp.tool()
async def get_vix_analytics(history_days: int = 252) -> dict:
    """
    VIX term structure: spot VIX, VIX3M, VIX6M, VIX1Y, VVIX, CBOE Skew Index.
    Volatility risk premium (VIX minus SPX 30d realized vol), VIX percentile,
    contango/backwardation flag, vol regime (5 buckets), and 60-day history.

    Args:
        history_days: Days of history for percentile calculation (default 252 = 1 year)
    """
    from sentinel.sma.vix_analytics import get_vix_analytics as _vix

    result = await _vix(history_days=history_days)
    return result.model_dump(mode="json")


@mcp.tool()
async def get_vol_regime(
    tickers_json: Optional[str] = None,
    history_days: int = 252,
) -> dict:
    """
    Market volatility regime dashboard: VIX-derived regime + per-ticker 20d/60d
    realized vol vs implied vol premium. Fear/greed proxy (100 - VIX percentile).

    Args:
        tickers_json: JSON list of tickers for realized vol (default: SPY only)
        history_days: History for vol calculation (default 252)
    """
    import json
    from sentinel.sma.vix_analytics import get_vol_regime as _regime

    tickers = json.loads(tickers_json) if tickers_json else None
    result = await _regime(tickers=tickers, history_days=history_days)
    return result.model_dump(mode="json")


# ─── Tool 111: Insider Cluster Signal ────────────────────────────────────────

@mcp.tool()
async def get_insider_signal(
    ticker: str,
    days_back: int = 180,
) -> dict:
    """
    Aggregate insider trading signal: cluster buy detection (≥2 distinct insiders),
    officer vs director sentiment, net purchase ratio, unusual transaction size,
    recent momentum, and composite signal score (0-10). Free SmartInsider equivalent.

    Args:
        ticker:    Stock ticker (e.g. "AAPL", "NVDA", "TSLA")
        days_back: Days of Form 4 history to analyze (default 180)
    """
    from sentinel.sfe.insider_signal import get_insider_signal as _insider

    result = await _insider(ticker=ticker, days_back=days_back)
    return result.model_dump(mode="json")


@mcp.tool()
async def screen_insider_buying(
    tickers_json: str,
    min_signal_score: float = 5.0,
) -> dict:
    """
    Multi-ticker insider buying screen: identify cluster buys, officer buyers,
    and strong buy signals. Sort by composite signal score.

    Args:
        tickers_json:     JSON list of tickers, e.g. '["AAPL","NVDA","MSFT"]'
        min_signal_score: Minimum composite score 0-10 (default 5.0)
    """
    import json
    from sentinel.sfe.insider_signal import screen_insider_buying as _screen

    tickers = json.loads(tickers_json)
    result = await _screen(tickers=tickers, min_signal_score=min_signal_score)
    return result.model_dump(mode="json")


# ─── Tool 107: Earnings Surprise Tracker ─────────────────────────────────────

@mcp.tool()
async def get_earnings_surprise(
    ticker: str,
    quarters: int = 8,
) -> dict:
    """
    EPS beat/miss history from yfinance: per-quarter actual vs estimate, surprise %,
    beat rate, consistency score (0-10), trend (improving/stable/deteriorating),
    consecutive beats streak, and next earnings date + EPS estimate.

    Args:
        ticker:   Stock ticker (e.g. "AAPL", "NVDA", "MSFT")
        quarters: Number of quarters to analyze (default 8 = trailing 2 years)
    """
    from sentinel.sfe.earnings_surprise import get_earnings_surprise as _surp

    result = await _surp(ticker=ticker, quarters=quarters)
    return result.model_dump(mode="json")


@mcp.tool()
async def screen_earnings_beats(
    tickers_json: str,
    min_beat_rate: float = 0.60,
) -> dict:
    """
    Multi-ticker earnings beat rate screener: identify consistent EPS beaters,
    biggest average surprise, and recent misses. Requires ≥4 quarters of data.

    Args:
        tickers_json:  JSON list of tickers, e.g. '["AAPL","NVDA","MSFT","AMZN"]'
        min_beat_rate: Minimum beat rate fraction 0-1 (default 0.60 = 60%)
    """
    import json
    from sentinel.sfe.earnings_surprise import screen_earnings_beats as _screen

    tickers = json.loads(tickers_json)
    result = await _screen(tickers=tickers, min_beat_rate=min_beat_rate)
    return result.model_dump(mode="json")


# ─── Tool 105: Corporate Governance Profile ──────────────────────────────────

@mcp.tool()
async def get_governance_profile(ticker: str) -> dict:
    """
    Corporate governance scoring from EDGAR DEF 14A proxy: board size and
    independence %, CEO/Chairman duality, audit committee independence, say-on-pay
    vote %, board gender diversity, classified board, poison pill, pay-for-performance.
    Composite governance score 0-10. Free ISS proxy equivalent.

    Args:
        ticker: Stock ticker (e.g. "AAPL", "GS", "JPM", "TSLA")
    """
    from sentinel.sfe.governance import get_governance_profile as _gov

    result = await _gov(ticker=ticker)
    return result.model_dump(mode="json")


@mcp.tool()
async def screen_governance(
    tickers_json: str,
    min_score: float = 5.0,
) -> dict:
    """
    Screen multiple companies for governance quality: composite score, strong (≥7)
    and weak (≤4) governance lists, average score, best and worst.

    Args:
        tickers_json: JSON list of tickers, e.g. '["AAPL","GS","TSLA","META"]'
        min_score:    Minimum governance score to include (default 5.0)
    """
    import json
    from sentinel.sfe.governance import screen_governance as _screen

    tickers = json.loads(tickers_json)
    result = await _screen(tickers=tickers, min_score=min_score)
    return result.model_dump(mode="json")


# ─── Tool 103: Short-Squeeze Analytics ──────────────────────────────────────

@mcp.tool()
async def get_squeeze_analytics(ticker: str) -> dict:
    """
    Advanced short-squeeze signal suite: days-to-cover, SI % of float, borrow cost
    proxy (easy/moderate/hard/special), month-over-month SI change, price momentum,
    gamma squeeze risk flag, and composite squeeze score (0-10).

    Args:
        ticker: Stock ticker (e.g. "GME", "AMC", "BBBY", "TSLA")
    """
    from sentinel.sbx.squeeze_analytics import get_squeeze_analytics as _squeeze

    result = await _squeeze(ticker=ticker)
    return result.model_dump(mode="json")


@mcp.tool()
async def screen_squeeze_candidates(
    tickers_json: str,
    min_squeeze_score: float = 5.0,
) -> dict:
    """
    Multi-ticker short-squeeze screen: DTC, SI % float, borrow difficulty, gamma
    squeeze risk, and composite squeeze score. Identify high-risk squeeze setups.

    Args:
        tickers_json:      JSON list of tickers, e.g. '["GME","AMC","TSLA","BBBY"]'
        min_squeeze_score: Minimum composite squeeze score 0-10 (default 5.0)
    """
    import json
    from sentinel.sbx.squeeze_analytics import screen_squeeze_candidates as _screen

    tickers = json.loads(tickers_json)
    result = await _screen(tickers=tickers, min_squeeze_score=min_squeeze_score)
    return result.model_dump(mode="json")


# ─── Tool 99: Analyst Estimates Proxy ────────────────────────────────────────

@mcp.tool()
async def get_analyst_estimates(ticker: str) -> dict:
    """
    Analyst consensus proxy via yfinance: price targets (mean/high/low/median),
    upside %, recommendation mean (Strong Buy→Strong Sell), recent upgrades/
    downgrades, and EPS/revenue quarterly estimates. Free FactSet Estimates lite.

    Args:
        ticker: Stock ticker (e.g. "AAPL", "NVDA", "MSFT")
    """
    from sentinel.sfe.analyst_estimates import get_analyst_estimates as _est

    result = await _est(ticker=ticker)
    return result.model_dump(mode="json")


@mcp.tool()
async def screen_analyst_sentiment(
    tickers_json: str,
    min_upside_pct: float = 10.0,
) -> dict:
    """
    Multi-ticker analyst sentiment screen: filter by minimum price target upside,
    identify strong-buy and strong-sell consensus tickers. Free sell-side screener.

    Args:
        tickers_json:    JSON list of tickers, e.g. '["AAPL","NVDA","MSFT","TSLA"]'
        min_upside_pct:  Minimum mean target upside % (default 10.0)
    """
    import json
    from sentinel.sfe.analyst_estimates import screen_analyst_sentiment as _screen

    tickers = json.loads(tickers_json)
    result = await _screen(tickers=tickers, min_upside_pct=min_upside_pct)
    return result.model_dump(mode="json")


# ─── Tool 101: Convertible Bond Screen ───────────────────────────────────────

@mcp.tool()
async def screen_convertibles(tickers_json: str) -> dict:
    """
    Convertible bond screen using live equity prices: parity, conversion premium,
    delta/gamma, bond floor, and equity-like/balanced/bond-like classification.
    Uses default CB terms (2.5% coupon, 3Y maturity, 20% premium to spot).

    Args:
        tickers_json: JSON list of tickers, e.g. '["TSLA","NVDA","MSTR","COIN"]'
    """
    import json
    from sentinel.sbx.convertible_bonds import screen_convertibles as _screen

    tickers = json.loads(tickers_json)
    result = await _screen(tickers=tickers)
    return result.model_dump(mode="json")


# ─── Tool 102: Convertible Bond Full Analytics ───────────────────────────────

@mcp.tool()
async def analyze_convertible_bond(
    ticker: str,
    face_value: float = 1000.0,
    coupon_rate: float = 0.025,
    maturity_years: float = 3.0,
    conversion_ratio: float = 10.0,
    straight_bond_yield: float = 0.07,
    market_price: Optional[float] = None,
    implied_vol: float = 0.30,
) -> dict:
    """
    Full convertible bond analytics: parity, conversion premium, investment premium,
    breakeven (payback years), Greeks (delta/gamma/theta/rho), bond floor.
    Uses live equity price from yfinance.

    Args:
        ticker:             Stock ticker for live price fetch
        face_value:         Bond face value (default $1,000)
        coupon_rate:        Annual coupon rate (default 0.025 = 2.5%)
        maturity_years:     Remaining years to maturity (default 3.0)
        conversion_ratio:   Shares received per bond (default 10)
        straight_bond_yield: Risk-free + credit spread (default 0.07)
        market_price:       Actual CB market price if known (default: estimated)
        implied_vol:        Equity vol for greeks (default 0.30)
    """
    import asyncio
    from sentinel.sbx.convertible_bonds import ConvertibleTerms, analyze_convertible

    import yfinance as _yf
    info = await asyncio.to_thread(lambda: _yf.Ticker(ticker).fast_info)
    current_price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None) or 100.0

    terms = ConvertibleTerms(
        ticker=ticker,
        face_value=face_value,
        coupon_rate=coupon_rate,
        maturity_years=maturity_years,
        conversion_ratio=conversion_ratio,
        current_stock_price=current_price,
        straight_bond_yield=straight_bond_yield,
        market_price=market_price,
        implied_vol=implied_vol,
    )
    result = analyze_convertible(terms=terms)
    return result.model_dump(mode="json")


# ─── Tool 97: Earnings Filing NLP ────────────────────────────────────────────

@mcp.tool()
async def analyze_earnings_filing(
    ticker: str,
    filing_date: Optional[str] = None,
) -> dict:
    """
    Earnings 8-K NLP analysis via Claude Haiku: tone (bullish/neutral/bearish),
    guidance signal, revenue/profit signals, key themes, risks, catalysts,
    and sentiment score (-1 to +1). Free AlphaSense equivalent.

    Args:
        ticker:       Stock ticker (e.g. "AAPL", "MSFT", "NVDA")
        filing_date:  Target date YYYY-MM-DD (default: most recent 8-K earnings)
    """
    from sentinel.sil.earnings_nlp import analyze_earnings_filing as _earn

    result = await _earn(ticker=ticker, filing_date=filing_date)
    return result.model_dump(mode="json")


# ─── Tool 98: Earnings Sentiment Trend ───────────────────────────────────────

@mcp.tool()
async def get_earnings_trend(
    ticker: str,
    quarters: int = 4,
) -> dict:
    """
    Multi-quarter earnings sentiment trend: per-quarter tone, guidance changes,
    sentiment score arc, and improving/deteriorating/stable trend verdict.

    Args:
        ticker:   Stock ticker (e.g. "AAPL", "MSFT", "NVDA")
        quarters: Number of quarters to analyze (default 4 = trailing 1 year)
    """
    from sentinel.sil.earnings_nlp import get_earnings_trend as _trend

    result = await _trend(ticker=ticker, quarters=quarters)
    return result.model_dump(mode="json")


# ─── Tools 123-130: Wave-17 Pre-wired ────────────────────────────────────────

@mcp.tool()
async def get_news_flow(ticker: str, max_items: int = 20) -> dict:
    """
    Real-time news flow monitor: yfinance news + EDGAR 8-K RSS aggregated and
    classified by event type (earnings/M&A/regulatory/credit/guidance/restructuring),
    sentiment (positive/negative/neutral), materiality, and volume spike detection.

    Args:
        ticker:    Stock ticker (e.g. "AAPL", "MSFT", "NVDA")
        max_items: Max news items to return (default 20)
    """
    from sentinel.sma.news_flow import get_news_flow as _nf

    result = await _nf(ticker=ticker, max_items=max_items)
    return result.model_dump(mode="json")


@mcp.tool()
async def screen_news_flow(
    tickers_json: str,
    min_spike_ratio: float = 1.5,
) -> dict:
    """
    Multi-ticker news volume spike and material event screen: identify tickers
    with abnormal news volume (>1.5× trailing average) or material events
    (earnings/M&A/guidance) in the last 24 hours.

    Args:
        tickers_json:    JSON list of tickers, e.g. '["AAPL","NVDA","MSFT"]'
        min_spike_ratio: Minimum volume spike ratio to flag (default 1.5)
    """
    import json
    from sentinel.sma.news_flow import screen_news_flow as _screen

    tickers = json.loads(tickers_json)
    result = await _screen(tickers=tickers, min_spike_ratio=min_spike_ratio)
    return result.model_dump(mode="json")


@mcp.tool()
async def get_dividend_ddm(ticker: str) -> dict:
    """
    Multi-stage Dividend Discount Model: Gordon Growth, H-Model (Fuller & Hsia),
    and 3-stage DDM intrinsic value estimates. Dividend sustainability score
    (payout ratio, FCF coverage, consecutive growth years), Aristocrat/King
    classification, consensus intrinsic value. Free Bloomberg DDIS equivalent.

    Args:
        ticker: Stock ticker (e.g. "JNJ", "KO", "PG", "AAPL")
    """
    from sentinel.sfe.dividend_ddm import get_dividend_ddm as _ddm

    result = await _ddm(ticker=ticker)
    return result.model_dump(mode="json")


@mcp.tool()
async def screen_dividend_quality(
    tickers_json: str,
    min_yield_pct: float = 1.0,
) -> dict:
    """
    Dividend quality screener: sustainability scores, Aristocrats (25+ years),
    Kings (50+ years), at-risk flags, consensus DDM intrinsic value, upside %.

    Args:
        tickers_json:  JSON list of tickers, e.g. '["JNJ","KO","PG","T","VZ"]'
        min_yield_pct: Minimum dividend yield % filter (default 1.0)
    """
    import json
    from sentinel.sfe.dividend_ddm import screen_dividend_quality as _screen

    tickers = json.loads(tickers_json)
    result = await _screen(tickers=tickers, min_yield_pct=min_yield_pct)
    return result.model_dump(mode="json")


@mcp.tool()
async def get_credit_spread_profile(ticker: str) -> dict:
    """
    Credit spread term structure from FINRA TRACE: Z-spread per bond, maturity-
    bucketed credit curve (short/medium/long), curve slope, migration alert
    (avg spread >300bps spike), credit tier (IG_HG/IG/HY_BB/HY_B/HY_CCC/NR).
    Free Bloomberg YAS equivalent.

    Args:
        ticker: Issuer ticker (e.g. "AAPL", "GE", "F", "T")
    """
    from sentinel.spr.credit_spread_term import get_credit_spread_profile as _csp

    result = await _csp(ticker=ticker)
    return result.model_dump(mode="json")


@mcp.tool()
async def screen_credit_spreads(tickers_json: str) -> dict:
    """
    Multi-ticker credit spread screen: identify wide-spread issuers (>300bps),
    inverted credit curves (long < short spread), migration alerts, and IG vs
    HY average spreads across the universe.

    Args:
        tickers_json: JSON list of tickers, e.g. '["AAPL","GE","F","T","NFLX"]'
    """
    import json
    from sentinel.spr.credit_spread_term import screen_credit_spreads as _screen

    tickers = json.loads(tickers_json)
    result = await _screen(tickers=tickers)
    return result.model_dump(mode="json")


@mcp.tool()
async def get_alt_data_dashboard(
    tickers_json: Optional[str] = None,
) -> dict:
    """
    Alternative data proxy dashboard: FRED PCE spending decomposition (consumer
    strength score, spending rotation), GDELT news volume signals per ticker,
    Google Trends interest (breakout detection), shipping/manufacturing proxies
    from FRED. Composite alt data score and regime (bullish/bearish/neutral).

    Args:
        tickers_json: JSON list of tickers for trends/GDELT (default: AMZN/WMT/TSLA)
    """
    import json
    from sentinel.sma.alt_data_proxies import get_alt_data_dashboard as _alt

    tickers = json.loads(tickers_json) if tickers_json else None
    result = await _alt(tickers=tickers)
    return result.model_dump(mode="json")


@mcp.tool()
async def get_alt_signal_summary(
    tickers_json: Optional[str] = None,
) -> dict:
    """
    Flat alternative data signal summary: per-category signal rows (PCE, GDELT,
    trends, shipping) for dashboard display. Lighter weight than full dashboard.

    Args:
        tickers_json: JSON list of tickers (default: AMZN/WMT/TSLA)
    """
    import json
    from sentinel.sma.alt_data_proxies import get_alt_signal_summary as _summary

    tickers = json.loads(tickers_json) if tickers_json else None
    result = await _summary(tickers=tickers)
    return [r.model_dump(mode="json") for r in result]


# ─── Tools 115-122: Wave-16 Pre-wired ────────────────────────────────────────

@mcp.tool()
async def get_sector_rotation(history_days: int = 252, top_n: int = 3) -> dict:
    """
    SPDR sector ETF rotation heatmap: relative strength vs SPY over 1M/3M/6M/12M,
    composite momentum score, market regime (risk-on/off/neutral/defensive), and
    rotation signal (overweight/underweight sectors).

    Args:
        history_days: Days of price history for RS calculation (default 252)
        top_n:        Number of top/bottom sectors to highlight (default 3)
    """
    from sentinel.sma.sector_rotation import get_sector_rotation as _sr

    result = await _sr(history_days=history_days, top_n=top_n)
    return result.model_dump(mode="json")


@mcp.tool()
async def screen_sector_strength(top_n: int = 5, history_days: int = 63) -> dict:
    """
    Screen all 11 SPDR sectors by composite momentum score: relative strength
    vs SPY, market breadth (% sectors positive 1M), regime, top/bottom sectors.

    Args:
        top_n:        Number of top sectors to return (default 5)
        history_days: History for RS calculation (default 63 = 3 months)
    """
    from sentinel.sma.sector_rotation import screen_sector_strength as _screen

    result = await _screen(top_n=top_n, history_days=history_days)
    return result.model_dump(mode="json")


@mcp.tool()
async def get_earnings_quality(ticker: str, years: int = 5) -> dict:
    """
    Earnings quality analytics from EDGAR XBRL: Sloan accruals ratio, cash
    conversion ratio (CFO/NI), operating leverage, quality score 0-10 and
    tier (AAA Quality → Very Low Quality). Free FactSet Earnings Quality equivalent.

    Args:
        ticker: Stock ticker (e.g. "AAPL", "MSFT", "GE")
        years:  Years of annual data to analyze (default 5)
    """
    from sentinel.sfe.earnings_quality import get_earnings_quality as _eq

    result = await _eq(ticker=ticker, years=years)
    return result.model_dump(mode="json")


@mcp.tool()
async def screen_earnings_quality(
    tickers_json: str,
    min_quality_score: float = 5.0,
) -> dict:
    """
    Screen multiple companies for earnings quality: accruals, cash conversion,
    quality tier, trend. Identify high-quality earnings vs earnings management risk.

    Args:
        tickers_json:      JSON list of tickers, e.g. '["AAPL","GE","MSFT","NFLX"]'
        min_quality_score: Minimum quality score 0-10 (default 5.0)
    """
    import json
    from sentinel.sfe.earnings_quality import screen_earnings_quality as _screen

    tickers = json.loads(tickers_json)
    result = await _screen(tickers=tickers, min_quality_score=min_quality_score)
    return result.model_dump(mode="json")


@mcp.tool()
async def get_vol_term_structure(ticker: str) -> dict:
    """
    Options volatility term structure: ATM IV per expiration (up to 6), forward
    volatility between tenors, 25-delta and 10-delta downside skew, put/call skew,
    term slope (contango/backwardation), and vol-of-vol per slice.

    Args:
        ticker: Stock or ETF ticker (e.g. "SPY", "AAPL", "QQQ")
    """
    from sentinel.sbx.vol_term_structure import get_vol_term_structure as _vts

    result = await _vts(ticker=ticker)
    return result.model_dump(mode="json")


@mcp.tool()
async def screen_vol_surface(tickers_json: str) -> dict:
    """
    Multi-ticker volatility surface screen: front ATM IV, term slope, skew regime,
    inverted term structure flags, elevated downside skew flags.

    Args:
        tickers_json: JSON list of tickers, e.g. '["SPY","QQQ","IWM","GLD"]'
    """
    import json
    from sentinel.sbx.vol_term_structure import screen_vol_surface as _screen

    tickers = json.loads(tickers_json)
    result = await _screen(tickers=tickers)
    return result.model_dump(mode="json")


@mcp.tool()
async def get_gdp_nowcast(history_months: int = 24) -> dict:
    """
    GDP nowcast from FRED leading indicators: 10-component weighted composite
    (industrial production, payrolls, retail sales, jobless claims, yield curve,
    consumer sentiment, building permits, housing starts, PCE). Recession probability
    and estimated annualized quarterly GDP growth rate.

    Args:
        history_months: Months of FRED history for standardization window (default 24)
    """
    from sentinel.spr.macro_nowcast import get_gdp_nowcast as _nowcast

    result = await _nowcast(history_months=history_months)
    return result.model_dump(mode="json")


@mcp.tool()
async def get_macro_nowcast_dashboard(history_months: int = 24) -> dict:
    """
    Full macro nowcast dashboard: GDP estimate, per-series readings (MoM/YoY change,
    trend), expansion/contraction signal lists, macro regime (expansion/late_cycle/
    contraction/recovery), key risks narrative.

    Args:
        history_months: Months of FRED history (default 24)
    """
    from sentinel.spr.macro_nowcast import get_macro_nowcast_dashboard as _dashboard

    result = await _dashboard(history_months=history_months)
    return result.model_dump(mode="json")


# ─── Tool 113: Supply Chain Concentration Risk ───────────────────────────────

@mcp.tool()
async def get_supply_chain_risk(ticker: str) -> dict:
    """
    Supply chain concentration analysis from EDGAR XBRL: customer concentration %,
    major customer count, geographic diversification (HHI), risk flags, and composite
    supply chain risk score 0-10. Free CapIQ supply chain risk equivalent.

    Args:
        ticker: Stock ticker (e.g. "AAPL", "TSLA", "NVDA")
    """
    from sentinel.sfe.supply_chain import get_supply_chain_risk as _sc

    result = await _sc(ticker=ticker)
    return result.model_dump(mode="json")


@mcp.tool()
async def screen_concentration_risk(
    tickers_json: str,
    max_customer_concentration: float = 0.30,
) -> dict:
    """
    Screen multiple companies for supply chain concentration risk: customer concentration
    %, geographic HHI, risk flags, composite risk score. Identify single-customer-dependent
    companies vs geographically diversified peers.

    Args:
        tickers_json:               JSON list of tickers, e.g. '["AAPL","TSLA","NVDA"]'
        max_customer_concentration: Max top-customer share filter 0-1 (default 0.30 = 30%)
    """
    import json
    from sentinel.sfe.supply_chain import screen_concentration_risk as _screen

    tickers = json.loads(tickers_json)
    result = await _screen(tickers=tickers, max_customer_concentration=max_customer_concentration)
    return result.model_dump(mode="json")


def run_server(host: str = "0.0.0.0", port: int = 8001) -> None:
    """Start the MCP server."""
    logger.info("Starting SENTINEL MCP server", host=host, port=port)
    mcp.run(transport="sse", host=host, port=port)


if __name__ == "__main__":
    run_server()
