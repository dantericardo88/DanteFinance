"""
SENTINEL Terminal — Bloomberg-style Streamlit UI with command bar dispatcher.

Supported Bloomberg function codes (46+):
  DES — Security description & fundamentals
  GP  — Price chart
  GPC — Comparative price chart
  HP  — Historical prices
  FA  — Financial analysis (XBRL facts)
  RV  — Relative value comparison
  DVD — Dividend history
  OPT — Options chain
  OMON— Options monitor
  NI  — News for ticker
  CN  — Congressional trades
  IN  — Insider transactions
  HDS — Institutional holders
  SECF— SEC filings
  CMAP— Competitor map
  WEI — World economic indicators (FRED)
  ECOS— Economic data screen (FRED core series)
  COT — CFTC COT signals
  REGM— Macro regime detector
  YC  — Yield curve
  SRCH— Screener (fundamental + technical)
  BT  — Backtest runner
  PORT— Portfolio overview
  RISK— Portfolio risk analytics
  MSG — Orders / execution
  ACT — Account summary
  MCP — AI assistant (MCP tool interface)
  FX  — ECB foreign exchange rates (Frankfurter API)
  GLOBAL — G7 global macro dashboard (FRED multi-country)
  KELLY — Kelly criterion + risk parity position sizer
  SHORT — FINRA RegSHO short interest + squeeze screen
  BONDS — FINRA TRACE corporate bond quotes + credit curve
  SEG — EDGAR XBRL business segment revenue breakdown
  OFLOW — Unusual options flow screener
  STRAT — NL→strategy generator (Claude tool-use)
  RESEARCH — Autonomous research agent (multi-step Claude)
  TRENDS — Google Trends momentum signal
  DEFI — DeFi TVL, protocols, yields (DeFiLlama)
  ONCHAIN — On-chain NVT/MVRV/fear-greed (CoinGecko)
  EKP — Earnings KPI extractor (EDGAR MD&A + Claude)
  ESG — ESG proxy profile (DEF14A + 10-K)
  ALPHA — Congress + COT + insider composite alpha
  XLS — Bloomberg-style Excel export (xlsxwriter)
  OPTFLOW — Options flow: unusual activity, max pain, IV skew
  EARNLP — Earnings 8-K NLP: tone/guidance/themes via Claude Haiku
  CREDIT — Credit analytics: Merton PD, CDS proxy, Altman Z, tier
  PEERS — Auto peer comparison: industry discovery, 17-metric rank
  SQUEEZE — Short-squeeze: FINRA DTC, borrow proxy, gamma risk score
  ANLEST — Analyst estimates proxy: targets, recs, EPS/rev estimates
  CONV — Convertible bond: parity, premium, delta/gamma/theta/rho
  GOV — Corporate governance: EDGAR DEF 14A board/duality/say-on-pay score
  ESURP — Earnings surprise: EPS beat/miss history, trend, consistency score
  VIXTS — VIX term structure: spot/3M/6M/1Y, VRP, regime, VVIX, SKEW
  INSIG — Insider cluster signal: cluster buy, officer sentiment, score
  SUPCHAIN — Supply chain concentration: EDGAR XBRL customer/geo risk
  SECROT — Sector rotation: SPDR ETF relative strength, momentum heatmap
  EQSCORE — Earnings quality: accruals, cash conversion, operating leverage
  IVTERM — IV term structure: multi-expiration ATM IV, forward vol, skew
  NOWCAST — GDP nowcast: FRED leading indicators, recession probability
"""
from __future__ import annotations
import asyncio
import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Any, Optional

st.set_page_config(
    page_title="SENTINEL Terminal",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Bloomberg-dark palette
SENTINEL_CSS = """
<style>
    .stApp { background-color: #0a0a0a; color: #e0e0e0; }
    .cmd-bar input { background: #1a1a1a; color: #00ff88; font-family: monospace;
                     font-size: 16px; border: 1px solid #00ff88; }
    .metric-box { background: #1a1a1a; border: 1px solid #333; border-radius: 4px;
                  padding: 8px 12px; margin: 4px 0; }
    .positive { color: #00ff88; }
    .negative { color: #ff4444; }
    .neutral  { color: #ffbb00; }
    h1, h2, h3 { color: #00ff88; font-family: 'Courier New', monospace; }
    .stDataFrame { background: #0f0f0f; }
</style>
"""
st.markdown(SENTINEL_CSS, unsafe_allow_html=True)


def run_async(coro):
    """Run an async coroutine from Streamlit's sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _fmt_large(n: Any) -> str:
    """Format large numbers as $1.2T, $450B, $3.2M, $850K."""
    try:
        v = float(n)
    except (TypeError, ValueError):
        return "N/A"
    if v >= 1e12:
        return f"${v/1e12:.2f}T"
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    if v >= 1e6:
        return f"${v/1e6:.2f}M"
    if v >= 1e3:
        return f"${v/1e3:.2f}K"
    return f"${v:.2f}"


# ─── Dashboard ────────────────────────────────────────────────────────────────

def _render_dashboard():
    """Default dashboard: watchlist + macro snapshot + recent news."""
    st.markdown("## SENTINEL Dashboard")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Watchlist")
        watchlist = st.session_state.get("watchlist", ["SPY", "QQQ", "AAPL", "NVDA", "BTC-USD"])
        for ticker in watchlist:
            with st.container():
                st.markdown(f"**{ticker}** — `GP {ticker}` to chart | `DES {ticker}` for fundamentals")

    with col2:
        st.markdown("### 🌐 Macro Snapshot")
        st.markdown("""
        | Series | Value | Signal |
        |--------|-------|--------|
        | 10Y Treasury | — | FRED → `WEI DGS10` |
        | VIX | — | `WEI VIXCLS` |
        | Yield Curve | — | `YC` |
        | Fed Funds | — | `WEI FEDFUNDS` |
        """)
        st.info("Type `ECOS` for full FRED core series | `REGM` for macro regime")

    with col3:
        st.markdown("### 🤖 AI Interface")
        st.info("Type `MCP <query>` to ask SENTINEL's AI assistant\n\n"
                "Example: `MCP What is the macro regime and what should I buy?`")
        st.markdown("### 🧭 Quick Commands")
        st.code("DES AAPL    — Fundamentals\n"
                "GP SPY      — Price chart\n"
                "BT SPY momentum — Backtest\n"
                "SRCH        — Screener\n"
                "CN          — Congressional trades\n"
                "COT         — CFTC positioning\n"
                "REGM        — Macro regime")


# ─── DES: Security Description ────────────────────────────────────────────────

def _render_des(args: list[str]):
    ticker = args[0] if args else st.text_input("Ticker:", "AAPL")
    if not ticker:
        return
    st.markdown(f"## DES — {ticker} Security Description")
    with st.spinner("Loading..."):
        from sentinel.sds.adapters.yfinance_adapter import YFinanceAdapter
        yf = YFinanceAdapter()
        info = run_async(yf.fetch_info(ticker))

    if not info:
        st.error(f"No data for {ticker}")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Last Price", f"${info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}")
        st.metric("Market Cap", _fmt_large(info.get("marketCap", 0)))
    with col2:
        st.metric("PE Ratio", info.get("trailingPE", "N/A"))
        st.metric("EPS (TTM)", info.get("trailingEps", "N/A"))
    with col3:
        st.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
        st.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")
    with col4:
        st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get("dividendYield") else "N/A")
        st.metric("Beta", info.get("beta", "N/A"))

    st.markdown("### Company Info")
    st.write(f"**Name:** {info.get('longName', ticker)}")
    st.write(f"**Sector:** {info.get('sector', '—')} | **Industry:** {info.get('industry', '—')}")
    st.write(f"**Exchange:** {info.get('exchange', '—')} | **Currency:** {info.get('currency', 'USD')}")
    if info.get("longBusinessSummary"):
        with st.expander("Business Summary"):
            st.write(info["longBusinessSummary"])


# ─── GP: Price Chart ──────────────────────────────────────────────────────────

def _render_gp(args: list[str]):
    ticker = args[0] if args else st.text_input("Ticker:", "SPY")
    if not ticker:
        return
    period = args[1] if len(args) > 1 else "1Y"

    period_map = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "3Y": 365*3, "5Y": 365*5, "MAX": 365*20}
    days = period_map.get(period.upper(), 365)

    st.markdown(f"## GP — {ticker} Price Chart ({period})")
    start = date.today() - timedelta(days=days)
    with st.spinner("Loading price data..."):
        from sentinel.sds.adapters.yfinance_adapter import YFinanceAdapter
        yf = YFinanceAdapter()
        bars = run_async(yf.fetch_ohlcv(ticker, datetime.combine(start, datetime.min.time()),
                                        datetime.combine(date.today(), datetime.min.time())))
    if not bars:
        st.error(f"No price data for {ticker}")
        return

    df = pd.DataFrame([{"date": b.time, "close": float(b.close), "volume": float(b.volume)} for b in bars])
    df = df.set_index("date")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.line_chart(df["close"])
    with col2:
        latest = df["close"].iloc[-1]
        first = df["close"].iloc[0]
        ret = (latest - first) / first * 100
        st.metric("Current", f"${latest:.2f}")
        st.metric(f"{period} Return", f"{ret:+.2f}%", delta=f"{ret:+.2f}%")
        st.metric("Bars", len(df))


# ─── GPC: Comparative Chart ───────────────────────────────────────────────────

def _render_gpc(args: list[str]):
    tickers = args if args else ["SPY", "QQQ", "IWM"]
    st.markdown(f"## GPC — Comparative: {' vs '.join(tickers)}")
    start = date.today() - timedelta(days=365)
    dfs = {}
    with st.spinner("Loading..."):
        from sentinel.sds.adapters.yfinance_adapter import YFinanceAdapter
        yf = YFinanceAdapter()
        for ticker in tickers:
            bars = run_async(yf.fetch_ohlcv(ticker, datetime.combine(start, datetime.min.time()),
                                            datetime.now()))
            if bars:
                series = pd.Series({b.time: float(b.close) for b in bars}, name=ticker)
                dfs[ticker] = series / series.iloc[0] * 100  # Rebase to 100

    if dfs:
        chart_df = pd.DataFrame(dfs)
        st.line_chart(chart_df)
        st.caption("Rebased to 100 at start date")


# ─── HP: Historical Prices ────────────────────────────────────────────────────

def _render_hp(args: list[str]):
    ticker = args[0] if args else st.text_input("Ticker:", "AAPL")
    if not ticker:
        return
    st.markdown(f"## HP — {ticker} Historical Prices")
    start = date.today() - timedelta(days=365)
    with st.spinner("Loading..."):
        from sentinel.sds.adapters.yfinance_adapter import YFinanceAdapter
        yf = YFinanceAdapter()
        bars = run_async(yf.fetch_ohlcv(ticker, datetime.combine(start, datetime.min.time()), datetime.now()))
    if bars:
        df = pd.DataFrame([{
            "Date": b.time.date(), "Open": float(b.open), "High": float(b.high),
            "Low": float(b.low), "Close": float(b.close), "Volume": int(b.volume),
        } for b in bars]).sort_values("Date", ascending=False)
        st.dataframe(df, use_container_width=True)


# ─── FA: Financial Analysis ───────────────────────────────────────────────────

def _render_fa(args: list[str]):
    ticker = args[0] if args else st.text_input("Ticker:", "AAPL")
    if not ticker:
        return
    st.markdown(f"## FA — {ticker} Financial Analysis (EDGAR XBRL)")
    with st.spinner("Loading XBRL data..."):
        from sentinel.sds.adapters.edgar_adapter import EDGARAdapter
        from sentinel.sfe.xbrl_parser import extract_facts, get_annual_facts
        edgar = EDGARAdapter()
        run_async(edgar.load_company_tickers())
        cik = edgar.ticker_to_cik(ticker)
        if not cik:
            st.error(f"CIK not found for {ticker}")
            return
        companyfacts = run_async(edgar.fetch_companyfacts(cik))
        all_facts = extract_facts(companyfacts, cik)

    key_labels = ["revenue", "net_income", "gross_profit", "cfo", "eps_diluted",
                  "total_assets", "long_term_debt", "stockholders_equity"]
    for label in key_labels:
        ann = get_annual_facts(all_facts, label)
        if ann:
            rows = [{"Year": f.period_end.year, "Value": float(f.value), "Unit": f.unit} for f in ann[:8]]
            st.markdown(f"**{label.replace('_', ' ').title()}**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ─── NI: News ─────────────────────────────────────────────────────────────────

def _render_ni(args: list[str]):
    ticker = args[0] if args else st.text_input("Ticker:", "AAPL")
    if not ticker:
        return
    st.markdown(f"## NI — {ticker} News & Sentiment")
    with st.spinner("Loading news..."):
        from sentinel.sds.adapters.finnhub_adapter import FinnhubAdapter
        from sentinel.sil.sentiment import score_sentiment_batch
        from sentinel.core.config import get_settings
        s = get_settings()
        fh = FinnhubAdapter(api_key=s.finnhub_api_key)
        end_str = date.today().isoformat()
        start_str = (date.today() - timedelta(days=7)).isoformat()
        news = run_async(fh.fetch_news(ticker, start_str, end_str))
        headlines = [n.get("headline", "") for n in news[:20]]
        sentiments = run_async(score_sentiment_batch(headlines)) if headlines else []

    for article, sent in zip(news[:20], sentiments):
        color = "🟢" if sent.label == "positive" else "🔴" if sent.label == "negative" else "⚪"
        st.markdown(f"{color} **{article.get('headline', '')}**  \n"
                    f"*{article.get('source', '')}* — {article.get('url', '')} "
                    f"(sentiment: {sent.label}, {sent.score:.2f})")


# ─── CN: Congressional Trades ─────────────────────────────────────────────────

def _render_cn(args: list[str]):
    ticker = args[0] if args else None
    st.markdown(f"## CN — Congressional STOCK Act Trades{f': {ticker}' if ticker else ''}")
    st.caption("🔱 Leapfrog #29 — Unique to SENTINEL. No Bloomberg, CapIQ, or FactSet equivalent.")
    with st.spinner("Fetching congressional disclosures..."):
        from sentinel.sod.congressional import CongressionalTradeTracker
        tracker = CongressionalTradeTracker()
        run_async(tracker.fetch_all_recent(lookback_days=90))
        signals = tracker.generate_signals()
        if ticker:
            signals = [s for s in signals if s["ticker"].upper() == ticker.upper()]

    if signals:
        df = pd.DataFrame(signals)
        st.dataframe(df[["politician", "chamber", "party", "ticker",
                          "direction", "amount_range", "tx_date",
                          "filing_lag_days", "signal_strength"]],
                     use_container_width=True)
    else:
        st.info("No congressional trades found for the selected criteria")


# ─── COT: CFTC Positioning ────────────────────────────────────────────────────

def _render_cot(args: list[str]):
    market = " ".join(args) if args else None
    st.markdown("## COT — CFTC Commitments of Traders Signals")
    st.caption("🔱 Leapfrog #46 — COT Index: >80=extreme long (bearish), <20=extreme short (bullish)")
    with st.spinner("Loading COT data (may take a moment)..."):
        from sentinel.sma.cot_report import COTClient
        client = COTClient()
        run_async(client.load_range(date.today().year - 1, date.today().year))
        signals = client.get_all_market_signals()
        if market:
            signals = [s for s in signals if market.upper() in s["market"].upper()]

    if signals:
        df = pd.DataFrame(signals)
        df["cot_bar"] = df["cot_index"].apply(lambda x: "🟢" if x <= 20 else "🔴" if x >= 80 else "⚪")
        st.dataframe(df[["code", "market", "cot_index", "net_position", "signal", "date", "cot_bar"]],
                     use_container_width=True)


# ─── REGM: Macro Regime ───────────────────────────────────────────────────────

def _render_regm(args: list[str]):
    st.markdown("## REGM — HMM Macro Regime Detector")
    st.caption("🔱 Leapfrog #50 — 4-state hidden Markov model. Unique to SENTINEL.")
    with st.spinner("Running HMM regime detection..."):
        from sentinel.sil.mcp_server import get_macro_regime
        result = run_async(get_macro_regime())

    if "error" in result:
        st.error(result["error"])
        return

    regime = result["current_regime"]
    confidence = result["confidence"]
    color = {"GROWTH_DEFLATION": "🟢", "GROWTH_INFLATION": "🟡",
              "CONTRACTION_DEFLATION": "🔵", "CONTRACTION_INFLATION": "🔴"}.get(regime, "⚪")

    st.markdown(f"## {color} Current Regime: **{regime}**")
    st.metric("Confidence", f"{confidence:.1%}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Regime Probabilities")
        prob_df = pd.DataFrame([
            {"Regime": k, "Probability": f"{v:.1%}"}
            for k, v in result["regime_probabilities"].items()
        ])
        st.dataframe(prob_df, use_container_width=True)
    with col2:
        st.markdown("### Asset Class Implications")
        impl = result.get("asset_class_implications", {})
        for asset, signal in impl.items():
            icon = "🟢" if "bull" in signal else "🔴" if "bear" in signal else "⚪"
            st.markdown(f"**{asset.title()}:** {icon} {signal}")


# ─── ECOS: Economic Screen ────────────────────────────────────────────────────

def _render_ecos(args: list[str]):
    st.markdown("## ECOS — FRED Economic Data Screen (33 Core Series)")
    with st.spinner("Fetching FRED data..."):
        from sentinel.sds.adapters.fred_adapter import FREDAdapter, CORE_FRED_SERIES
        from sentinel.core.config import get_settings
        s = get_settings()
        fred = FREDAdapter(api_key=s.fred_api_key)
        results = run_async(fred.fetch_core_series())

    rows = []
    for sid, points in results.items():
        if points:
            latest = sorted(points, key=lambda p: p.time)[-1]
            rows.append({
                "Series": sid, "Label": CORE_FRED_SERIES.get(sid, sid),
                "Latest Value": float(latest.value),
                "Date": latest.time.date().isoformat(),
            })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ─── YC: Yield Curve ──────────────────────────────────────────────────────────

def _render_yc(args: list[str]):
    st.markdown("## YC — US Treasury Yield Curve")
    history_days = st.slider("Slope history (days):", 30, 365, 90, key="yc_hist")

    with st.spinner("Fetching yield curve from FRED..."):
        try:
            from sentinel.sfe.yield_curve import get_yield_curve
            result = run_async(get_yield_curve(include_history_days=history_days))

            # Inversion alert
            inv_label = "🔴 INVERTED" if result.inversion_flag else "🟢 Normal"
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("10Y", f"{result.curve[-3].yield_pct:.2f}%" if len(result.curve) >= 3 else "N/A")
            c2.metric("2Y", f"{[t for t in result.curve if t.tenor == '2Y'][0].yield_pct:.2f}%" if any(t.tenor == "2Y" for t in result.curve) else "N/A")
            c3.metric("10Y-2Y Slope", f"{result.slope_10y2y:.0f}bps")
            c4.metric("Curve", f"{inv_label} ({result.recession_signal})")

            # Spot curve chart
            curve_df = pd.DataFrame([{"Tenor": t.tenor, "Yield (%)": t.yield_pct} for t in result.curve])
            st.line_chart(curve_df.set_index("Tenor"))

            # Nelson-Siegel fitted curve
            if result.fitted_curve:
                fit_df = pd.DataFrame(result.fitted_curve).rename(
                    columns={"tenor_years": "Maturity (yr)", "fitted_yield": "Fitted Yield (%)"})
                with st.expander("Nelson-Siegel Fitted Curve"):
                    st.line_chart(fit_df.set_index("Maturity (yr)"))
                    ns = result.ns_params
                    st.caption(f"β₀={ns.get('beta0', 0):.3f}  β₁={ns.get('beta1', 0):.3f}  β₂={ns.get('beta2', 0):.3f}  λ={ns.get('lambda_', 0):.3f}")

            # Forward rates
            if result.forward_rates:
                st.markdown("**Implied Forward Rates**")
                fwd_cols = st.columns(len(result.forward_rates))
                for col, (tenor, rate) in zip(fwd_cols, result.forward_rates.items()):
                    col.metric(f"{tenor} fwd", f"{rate:.2f}%")

            # Credit spreads
            if result.oas_ig_bps or result.oas_hy_bps:
                st.markdown("**Credit Spreads (OAS)**")
                s1, s2 = st.columns(2)
                s1.metric("IG OAS", f"{result.oas_ig_bps:.0f}bps" if result.oas_ig_bps else "N/A")
                s2.metric("HY OAS", f"{result.oas_hy_bps:.0f}bps" if result.oas_hy_bps else "N/A")

            # Slope history
            if result.slope_history:
                hist_df = pd.DataFrame(result.slope_history).set_index("date")
                with st.expander(f"10Y-2Y Slope History ({history_days}d)"):
                    st.line_chart(hist_df)

            st.dataframe(curve_df, use_container_width=True)

            if result.warnings:
                for w in result.warnings:
                    st.warning(w)

        except ImportError:
            # Fallback: basic FREDAdapter curve
            from sentinel.sds.adapters.fred_adapter import FREDAdapter
            from sentinel.core.config import get_settings
            s = get_settings()
            fred = FREDAdapter(api_key=s.fred_api_key)
            tenor_map = {"1M": "DGS1MO", "3M": "DGS3MO", "6M": "DGS6MO", "1Y": "DGS1",
                         "2Y": "DGS2", "5Y": "DGS5", "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30"}

            async def _fetch_all():
                tasks = {t: fred.fetch_series(sid, start=date(2020, 1, 1)) for t, sid in tenor_map.items()}
                return {t: await coro for t, coro in tasks.items()}

            rates = run_async(_fetch_all())
            curve_data = {}
            for tenor, points in rates.items():
                if points:
                    latest = sorted(points, key=lambda p: p.time)[-1]
                    curve_data[tenor] = float(latest.value)
            if curve_data:
                df = pd.DataFrame(list(curve_data.items()), columns=["Tenor", "Rate(%)"])
                st.line_chart(df.set_index("Tenor"))
                st.dataframe(df, use_container_width=True)


# ─── WEI: World Economic Indicator ────────────────────────────────────────────

def _render_wei(args: list[str]):
    series_id = args[0] if args else "DGS10"
    st.markdown(f"## WEI — {series_id} Economic Indicator")
    with st.spinner("Loading FRED data..."):
        from sentinel.sds.adapters.fred_adapter import FREDAdapter
        from sentinel.core.config import get_settings
        s = get_settings()
        fred = FREDAdapter(api_key=s.fred_api_key)
        points = run_async(fred.fetch_series(series_id, start=date(2000, 1, 1)))
        info = run_async(fred.get_series_info(series_id))

    if points:
        df = pd.Series(
            {p.time: float(p.value) for p in points}, name=series_id
        )
        st.markdown(f"**{info.get('title', series_id)}** | Units: {info.get('units', '')}")
        st.line_chart(df)


# ─── SRCH: Screener ───────────────────────────────────────────────────────────

def _render_srch(args: list[str]):
    st.markdown("## SRCH — Stock Screener")
    query = " ".join(args) if args else st.text_input(
        "Natural language query:",
        "tech stocks with PE < 25 and revenue growth > 15%",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        use_nl = st.checkbox("Claude NL mode (requires Anthropic key)", value=True)
    with col2:
        limit = st.number_input("Max results", min_value=5, max_value=100, value=25, step=5)

    if st.button("▶ Screen") or args:
        with st.spinner("Running screen..."):
            if use_nl:
                try:
                    from sentinel.sil.mcp_server import screen_stocks_nl
                    result = run_async(screen_stocks_nl(query=query, max_results=int(limit)))
                    # nl_screener returns NLScreenResult.model_dump()
                    rows = result.get("results", [])
                    criteria_display = result.get("criteria", {})
                except Exception:
                    from sentinel.sil.mcp_server import screen_stocks
                    result = run_async(screen_stocks(query=query, limit=int(limit)))
                    rows = result.get("results", [])
                    criteria_display = result.get("parsed_criteria", {})
            else:
                from sentinel.sil.mcp_server import screen_stocks
                result = run_async(screen_stocks(query=query, limit=int(limit)))
                rows = result.get("results", [])
                criteria_display = result.get("parsed_criteria", {})

        if criteria_display:
            st.caption(f"Parsed criteria: {criteria_display}")

        count = len(rows)
        st.markdown(f"**{count} result{'s' if count != 1 else ''} found**")

        if rows:
            import pandas as pd

            def _fmt_cap(v):
                if v is None:
                    return "—"
                if v >= 1e12:
                    return f"${v/1e12:.1f}T"
                if v >= 1e9:
                    return f"${v/1e9:.1f}B"
                return f"${v/1e6:.0f}M"

            def _fmt_pct(v):
                return f"{v*100:.1f}%" if v is not None else "—"

            def _fmt_float(v, decimals=2):
                return f"{v:.{decimals}f}" if v is not None else "—"

            display_rows = []
            for r in rows:
                display_rows.append({
                    "Ticker": r.get("ticker", ""),
                    "Name": (r.get("name") or "")[:28],
                    "Sector": (r.get("sector") or "")[:18],
                    "Mkt Cap": _fmt_cap(r.get("market_cap")),
                    "P/E": _fmt_float(r.get("pe_ratio"), 1),
                    "Rev Gr": _fmt_pct(r.get("revenue_growth_yoy")),
                    "Net Mgn": _fmt_pct(r.get("net_margin")),
                    "ROE": _fmt_pct(r.get("roe")),
                    "Div Yld": _fmt_pct(r.get("dividend_yield")),
                })

            df = pd.DataFrame(display_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No stocks matched your criteria. Try relaxing the filters.")

        with st.expander("Raw response"):
            st.json(result)


# ─── BT: Backtest ─────────────────────────────────────────────────────────────

def _render_bt(args: list[str]):
    ticker = args[0] if args else "SPY"
    strategy = args[1].lower() if len(args) > 1 else "momentum"
    st.markdown(f"## BT — Backtest: {strategy.upper()} on {ticker}")

    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Start", date(2018, 1, 1))
    with col2:
        end_date = st.date_input("End", date.today())
    with col3:
        st.write("")
        run_btn = st.button("▶ Run Backtest")

    if run_btn:
        from sentinel.sil.mcp_server import run_backtest as mcp_backtest
        with st.spinner("Running backtest..."):
            result = run_async(mcp_backtest(
                ticker=ticker, strategy=strategy,
                start=start_date.isoformat(), end=end_date.isoformat(),
            ))

        if "error" in result:
            st.error(result["error"])
        else:
            m = result.get("metrics", {})
            cols = st.columns(5)
            metrics = [
                ("Total Return", f"{m.get('total_return', 0)*100:.1f}%"),
                ("CAGR", f"{m.get('cagr', 0)*100:.1f}%"),
                ("Sharpe", f"{m.get('sharpe_ratio', 0):.2f}"),
                ("DSR", f"{m.get('deflated_sharpe_ratio', 0):.3f}"),
                ("Max DD", f"{m.get('max_drawdown', 0)*100:.1f}%"),
            ]
            for col, (label, val) in zip(cols, metrics):
                col.metric(label, val)

            st.markdown("### Full Metrics")
            st.json(m)

            dsr = m.get("deflated_sharpe_ratio", 0)
            if dsr >= 0.95:
                st.success(f"✅ DSR {dsr:.3f} ≥ 0.95 — Strategy passes promotion gate to PAPER trading")
            else:
                st.warning(f"⚠️ DSR {dsr:.3f} < 0.95 — Strategy does not meet promotion threshold")


# ─── PORT: Portfolio ──────────────────────────────────────────────────────────

def _render_port(args: list[str]):
    st.markdown("## PORT — Portfolio Overview")
    st.info("Connect Alpaca credentials in .env to see live positions. Running in paper mode.")
    from sentinel.see.broker import AlpacaBroker
    from sentinel.core.config import get_settings
    s = get_settings()
    if s.alpaca_api_key:
        with st.spinner("Loading portfolio..."):
            broker = AlpacaBroker(api_key=s.alpaca_api_key, secret_key=s.alpaca_secret_key, paper=True)
            positions = run_async(broker.get_positions())
            account = run_async(broker.get_account())
        if account:
            col1, col2, col3 = st.columns(3)
            col1.metric("Portfolio Value", f"${account.get('portfolio_value', 0):,.2f}")
            col2.metric("Cash", f"${account.get('cash', 0):,.2f}")
            col3.metric("Buying Power", f"${account.get('buying_power', 0):,.2f}")
        if positions:
            df = pd.DataFrame(positions)
            st.dataframe(df, use_container_width=True)
    else:
        st.warning("Configure ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")


# ─── IN: Insider Trades ───────────────────────────────────────────────────────

def _render_in(args: list[str]):
    ticker = args[0] if args else st.text_input("Ticker:", "AAPL")
    if not ticker:
        return
    st.markdown(f"## IN — {ticker} Insider Transactions (Form 4)")
    with st.spinner("Loading Form 4 data..."):
        from sentinel.sil.mcp_server import get_insider_trades
        result = run_async(get_insider_trades(ticker=ticker))
    transactions = result.get("transactions", [])
    if transactions:
        df = pd.DataFrame(transactions)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No insider transactions found")


# ─── SECF: SEC Filings ────────────────────────────────────────────────────────

def _render_secf(args: list[str]):
    ticker = args[0] if args else st.text_input("Ticker:", "AAPL")
    form_type = args[1] if len(args) > 1 else None
    if not ticker:
        return
    st.markdown(f"## SECF — {ticker} SEC Filings{f' ({form_type})' if form_type else ''}")
    with st.spinner("Loading filings..."):
        from sentinel.sil.mcp_server import get_filings
        result = run_async(get_filings(ticker=ticker, form_type=form_type))
    filings = result.get("filings", [])
    if filings:
        st.dataframe(pd.DataFrame(filings), use_container_width=True)
    else:
        st.info("No filings found")


# ─── HDS: Institutional Holders ───────────────────────────────────────────────

def _render_hds(args: list[str]):
    ticker = args[0] if args else st.text_input("Ticker:", "AAPL")
    if not ticker:
        return
    st.markdown(f"## HDS — {ticker} Institutional Holders (13F)")

    with st.spinner("Loading institutional holders..."):
        try:
            import yfinance as yf
            info = yf.Ticker(ticker)
            holders_df = info.institutional_holders
            major_df = info.major_holders
        except Exception as exc:
            st.error(f"yfinance error: {exc}")
            return

    if major_df is not None and not major_df.empty:
        st.markdown("### Ownership Summary")
        st.dataframe(major_df, use_container_width=True)

    if holders_df is not None and not holders_df.empty:
        st.markdown(f"### Top Institutional Holders ({len(holders_df)} reported)")
        display_cols = [c for c in ["Holder", "Shares", "% Out", "Value", "Date Reported"] if c in holders_df.columns]
        if display_cols:
            st.dataframe(holders_df[display_cols].sort_values("Shares", ascending=False) if "Shares" in holders_df.columns else holders_df, use_container_width=True)
            if "Shares" in holders_df.columns and "Holder" in holders_df.columns:
                top10 = holders_df.nlargest(10, "Shares").set_index("Holder")["Shares"]
                st.bar_chart(top10)
    else:
        st.info(f"No institutional holder data available for {ticker}")


# ─── DVD: Dividends ───────────────────────────────────────────────────────────

def _render_dvd(args: list[str]):
    ticker = args[0] if args else st.text_input("Ticker:", "AAPL")
    if not ticker:
        return
    st.markdown(f"## DVD — {ticker} Dividend History")
    from sentinel.sds.adapters.yfinance_adapter import YFinanceAdapter
    yf = YFinanceAdapter()
    with st.spinner("Loading dividends..."):
        divs = run_async(yf.fetch_dividends(ticker))
    if divs is not None and not divs.empty:
        st.line_chart(divs)
        st.dataframe(divs.reset_index().rename(columns={0: "Dividend"}).sort_values("Date", ascending=False),
                     use_container_width=True)
    else:
        st.info(f"{ticker} does not pay dividends or no history available")


# ─── OPT: Options ─────────────────────────────────────────────────────────────

def _render_opt(args: list[str]):
    ticker = args[0] if args else st.text_input("Ticker:", "SPY")
    if not ticker:
        return
    st.markdown(f"## OPT — {ticker} Options Chain")
    from sentinel.sds.adapters.yfinance_adapter import YFinanceAdapter
    yf = YFinanceAdapter()
    with st.spinner("Loading options..."):
        chain = run_async(yf.fetch_options_chain(ticker))
    if chain:
        st.markdown(f"**Expiry:** {chain.get('expiry')}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Calls**")
            if chain.get("calls"):
                df = pd.DataFrame(chain["calls"])
                cols = [c for c in ["strike", "lastPrice", "bid", "ask", "impliedVolatility", "openInterest"] if c in df.columns]
                st.dataframe(df[cols].head(20), use_container_width=True)
        with col2:
            st.markdown("**Puts**")
            if chain.get("puts"):
                df = pd.DataFrame(chain["puts"])
                cols = [c for c in ["strike", "lastPrice", "bid", "ask", "impliedVolatility", "openInterest"] if c in df.columns]
                st.dataframe(df[cols].head(20), use_container_width=True)


# ─── RISK: Risk Analytics ─────────────────────────────────────────────────────

def _render_risk(args: list[str]):
    ticker = args[0] if args else st.text_input("Ticker:", "SPY")
    if not ticker:
        return
    st.markdown(f"## RISK — {ticker} Risk Analytics (Historical + Monte Carlo)")

    lookback = st.slider("Lookback days", min_value=63, max_value=504, value=252, step=63)

    from sentinel.sds.adapters.yfinance_adapter import YFinanceAdapter
    import numpy as np
    from datetime import timedelta

    end_dt = datetime.combine(date.today(), datetime.min.time())
    start_dt = end_dt - timedelta(days=lookback + 10)

    with st.spinner(f"Computing risk metrics for {ticker}..."):
        yf = YFinanceAdapter()
        bars = run_async(yf.fetch_ohlcv(ticker, start_dt, end_dt, "1d"))

    if not bars or len(bars) < 20:
        st.warning(f"Insufficient data for {ticker} — need at least 20 bars")
        return

    closes = [float(b.close) for b in sorted(bars, key=lambda x: x.time)]
    returns = np.diff(closes) / closes[:-1]
    dates = [b.time for b in sorted(bars, key=lambda x: x.time)[1:]]

    # Core metrics
    ann_vol = float(np.std(returns) * np.sqrt(252))
    var_95 = float(np.percentile(returns, 5))
    var_99 = float(np.percentile(returns, 1))
    cvar_95 = float(np.mean(returns[returns <= var_95])) if any(returns <= var_95) else var_95
    cvar_99 = float(np.mean(returns[returns <= var_99])) if any(returns <= var_99) else var_99
    mean_ret = float(np.mean(returns))
    sharpe = float(mean_ret / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0

    # Max drawdown
    cum_rets = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cum_rets)
    drawdowns = (cum_rets - rolling_max) / rolling_max
    max_dd = float(np.min(drawdowns))

    # Skewness and kurtosis (moment-based)
    mu, sigma = np.mean(returns), np.std(returns)
    skewness = float(np.mean(((returns - mu) / sigma) ** 3)) if sigma > 0 else 0.0
    excess_kurt = float(np.mean(((returns - mu) / sigma) ** 4) - 3) if sigma > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Ann. Volatility", f"{ann_vol*100:.1f}%")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with col2:
        st.metric("VaR 95% (1-day)", f"{var_95*100:.2f}%", delta_color="inverse")
        st.metric("VaR 99% (1-day)", f"{var_99*100:.2f}%", delta_color="inverse")
    with col3:
        st.metric("CVaR 95%", f"{cvar_95*100:.2f}%", delta_color="inverse")
        st.metric("CVaR 99%", f"{cvar_99*100:.2f}%", delta_color="inverse")
    with col4:
        st.metric("Max Drawdown", f"{max_dd*100:.1f}%", delta_color="inverse")
        st.metric("Skewness", f"{skewness:.2f}")

    # Returns bar chart
    st.markdown("### Daily Returns")
    returns_series = pd.Series(returns, index=pd.to_datetime(dates))
    st.bar_chart(returns_series)

    # Cumulative returns
    st.markdown("### Cumulative Returns")
    cum_series = pd.Series(cum_rets - 1, index=pd.to_datetime(dates))
    st.line_chart(cum_series)

    # Monte Carlo VaR
    with st.expander("📐 Monte Carlo VaR (10,000 paths, 21-day horizon)"):
        n_sims, horizon = 10_000, 21
        sim_returns = np.random.normal(mean_ret, float(np.std(returns)), (n_sims, horizon))
        path_returns = np.prod(1 + sim_returns, axis=1) - 1
        mc_var_95 = float(np.percentile(path_returns, 5))
        mc_var_99 = float(np.percentile(path_returns, 1))
        mc_col1, mc_col2 = st.columns(2)
        with mc_col1:
            st.metric("MC VaR 95% (21-day)", f"{mc_var_95*100:.1f}%", delta_color="inverse")
        with mc_col2:
            st.metric("MC VaR 99% (21-day)", f"{mc_var_99*100:.1f}%", delta_color="inverse")
        mc_series = pd.Series(sorted(path_returns))
        st.line_chart(mc_series.rename("21-day path return distribution"))


# ─── MSG: Orders ──────────────────────────────────────────────────────────────

def _render_msg(args: list[str]):
    st.markdown("## MSG — Order Management")
    st.warning("⚠️ Live trading requires SENTINEL_LIVE_TRADING=true in .env")
    col1, col2, col3 = st.columns(3)
    with col1:
        order_ticker = st.text_input("Ticker")
    with col2:
        order_side = st.selectbox("Side", ["buy", "sell"])
    with col3:
        order_qty = st.number_input("Quantity", min_value=0.0, step=1.0)
    if st.button("Submit Order (Paper)"):
        st.info("Paper order submission — connect Alpaca via ACT to execute")


# ─── ACT: Account ─────────────────────────────────────────────────────────────

def _render_act(args: list[str]):
    st.markdown("## ACT — Account Summary")
    _render_port(args)


# ─── MCP: AI Assistant ────────────────────────────────────────────────────────

def _render_mcp(args: list[str]):
    query = " ".join(args) if args else st.text_input("Ask SENTINEL AI:", "What is the current macro regime and what should I be long?")
    st.markdown("## MCP — SENTINEL AI Assistant")
    st.caption("🔱 Leapfrog #59 — Unique to SENTINEL. No Bloomberg equivalent.")
    if query:
        st.info(f"Query: {query}")
        st.markdown("""
        The full MCP interface connects via `make mcp` to the FastMCP server.
        Claude can then call any of the 15 SENTINEL tools natively:
        - `get_ohlcv`, `get_quote`, `get_fundamentals`, `screen_stocks`
        - `get_congressional_trades`, `run_backtest`, `get_macro_regime`
        - `get_cot_signals`, `get_news_sentiment`, `explain_strategy`

        Start the MCP server: `make mcp`
        Then connect Claude to: `http://localhost:8001/sse`
        """)


# ─── OPT_ANALYTICS: Options Market Analytics ─────────────────────────────────

def _render_opt_analytics(args: list[str]) -> None:
    """OPT — Options market analytics: IV surface, skew, gamma exposure, max pain."""
    ticker = args[0] if args else st.text_input("Ticker:", "SPY")
    if not ticker:
        return
    st.subheader(f"OPT — Options Analytics: {ticker}")
    underlying_price = st.number_input("Underlying price", value=100.0, min_value=0.01)
    if st.button("Load Options Analytics"):
        with st.spinner("Fetching options chain..."):
            try:
                from sentinel.sds.adapters.polygon_adapter import PolygonAdapter
                from sentinel.sbx.options_analytics import get_options_summary
                from sentinel.core.config import get_settings
                s = get_settings()
                adapter = PolygonAdapter(api_key=s.polygon_api_key)
                contracts_raw = run_async(adapter.fetch_options_chain(ticker))
                summary = get_options_summary(ticker, contracts_raw, underlying_price)

                col1, col2, col3 = st.columns(3)
                pc = summary.get("pc_ratio", {})
                gex = summary.get("gex", {})
                mp = summary.get("max_pain", {})
                with col1:
                    st.metric("Put/Call OI Ratio", f"{pc.get('oi_ratio', 0):.2f}")
                with col2:
                    st.metric("Net GEX ($M)", f"{gex.get('net_gex', 0)/1e6:.1f}")
                with col3:
                    st.metric("Max Pain Strike", f"${mp.get('max_pain_strike', 0):.0f}")

                # Term structure
                ts = summary.get("term_structure", [])
                if ts:
                    ts_df = pd.DataFrame(ts)
                    st.write("**IV Term Structure**")
                    st.line_chart(ts_df.set_index("dte")["atm_iv"] if "dte" in ts_df.columns else ts_df)

                # Skew table
                skew = summary.get("skew", [])
                if skew:
                    st.write("**25-Delta Risk Reversal Skew by Expiry**")
                    skew_df = pd.DataFrame(skew)
                    st.dataframe(skew_df[["expiry_date", "risk_reversal_25d", "atm_iv"]].head(8) if "expiry_date" in skew_df.columns else skew_df)
            except Exception as e:
                st.error(f"Options error: {e}")


# ─── DCF: Discounted Cash Flow Valuation ──────────────────────────────────────

def _render_dcf(args: list[str]) -> None:
    """DCF — Discounted cash flow valuation using Damodaran methodology."""
    ticker = args[0] if args else st.text_input("Ticker:", "AAPL")
    if not ticker:
        return
    st.subheader(f"DCF — Intrinsic Value: {ticker}")
    col1, col2, col3 = st.columns(3)
    with col1:
        current_price = st.number_input("Current Price ($)", value=100.0, min_value=0.01)
    with col2:
        wacc = st.slider("WACC (%)", 6.0, 20.0, 10.0, 0.5) / 100
    with col3:
        tgr = st.slider("Terminal Growth (%)", 1.0, 4.0, 2.5, 0.25) / 100

    if st.button("Run DCF"):
        with st.spinner("Running valuation..."):
            try:
                from sentinel.sfe.dcf_model import DCFAssumptions, run_dcf
                assumptions = DCFAssumptions(
                    ticker=ticker,
                    revenue_base=1_000_000_000,
                    revenue_growth_rates=[0.10, 0.09, 0.08, 0.07, 0.06],
                    terminal_growth_rate=tgr,
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
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Intrinsic Value", f"${result.intrinsic_value_per_share:.2f}")
                with col2:
                    upside = result.upside_pct * 100
                    st.metric("Upside / Downside", f"{upside:+.1f}%", delta_color="normal")
                with col3:
                    st.metric("Enterprise Value", f"${result.enterprise_value/1e9:.2f}B")

                st.write("**WACC × Terminal Growth Sensitivity**")
                sensitivity_data = []
                for wacc_str, tg_dict in result.sensitivity.items():
                    for tg_str, iv in tg_dict.items():
                        sensitivity_data.append({"WACC": wacc_str, "Terminal Growth": tg_str, "Intrinsic Value": f"${iv:.2f}"})
                if sensitivity_data:
                    sens_df = pd.DataFrame(sensitivity_data).pivot(index="WACC", columns="Terminal Growth", values="Intrinsic Value")
                    st.dataframe(sens_df)
            except Exception as e:
                st.error(f"DCF error: {e}")


# ─── SOC: Social Media Sentiment ──────────────────────────────────────────────

def _render_social(args: list[str]) -> None:
    """SOC — Social media sentiment from Reddit and StockTwits."""
    ticker = args[0] if args else st.text_input("Ticker:", "TSLA")
    if not ticker:
        return
    st.subheader(f"SOC — Social Sentiment: {ticker}")
    if st.button("Load Social Sentiment"):
        with st.spinner("Scanning Reddit & StockTwits..."):
            try:
                from sentinel.snm.social_sentiment import get_social_sentiment
                result = run_async(get_social_sentiment(ticker))

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Posts", result.total_posts)
                with col2:
                    st.metric("Bullish %", f"{result.bullish_pct:.0f}%")
                with col3:
                    st.metric("Bearish %", f"{result.bearish_pct:.0f}%")
                with col4:
                    st.metric("Signal", result.sentiment_signal.replace("_", " ").title())

                st.metric("Avg FinBERT Score", f"{result.avg_finbert_score:+.3f}", help="-1 = very negative, +1 = very positive")

                if result.top_posts:
                    st.write("**Top Posts**")
                    for p in result.top_posts[:5]:
                        sentiment_icon = "🟢" if p.finbert_sentiment == "positive" else ("🔴" if p.finbert_sentiment == "negative" else "⚪")
                        st.write(f"{sentiment_icon} [{p.platform.upper()}] {p.title[:120]}")
            except Exception as e:
                st.error(f"Social sentiment error: {e}")


# ─── STRESS: Portfolio Stress Test ────────────────────────────────────────────

def _render_stress(args: list[str]) -> None:
    """STRESS — Portfolio stress test across historical scenarios."""
    st.subheader("STRESS — Portfolio Stress Test")
    st.info("Enter your portfolio holdings as ticker:weight pairs (e.g. AAPL:0.3, MSFT:0.3, SPY:0.4)")
    holdings_input = st.text_area("Holdings (ticker:weight, one per line)", "SPY:0.6\nTLT:0.3\nGLD:0.1")
    portfolio_value = st.number_input("Portfolio Value ($)", value=100_000.0, min_value=1.0)

    if st.button("Run Stress Test"):
        with st.spinner("Running stress scenarios..."):
            try:
                from sentinel.spr.stress_test import run_stress_test
                holdings = {}
                for line in holdings_input.strip().split("\n"):
                    if ":" in line:
                        t, w = line.strip().split(":")
                        holdings[t.strip()] = float(w.strip())

                report = run_stress_test(holdings=holdings, portfolio_value=portfolio_value, returns_df=pd.DataFrame())

                st.error(f"Worst Scenario: **{report.worst_scenario}** ({report.worst_loss_pct*100:.1f}% loss)")

                scenario_rows = [{"Scenario": s.scenario, "Description": s.description,
                                  "Return": f"{s.portfolio_return*100:.1f}%", "Max DD": f"{s.max_drawdown*100:.1f}%"}
                                 for s in report.historical_scenarios]
                st.dataframe(pd.DataFrame(scenario_rows))

                shock_rows = [{"Shock": s.shock_name, "P&L": f"${s.portfolio_pnl:,.0f}", "Return": f"{s.portfolio_pnl_pct*100:.1f}%"}
                             for s in report.parametric_shocks]
                st.write("**Parametric Shocks**")
                st.dataframe(pd.DataFrame(shock_rows))
            except Exception as e:
                st.error(f"Stress test error: {e}")


# ─── FACTOR: Fama-French Factor Model ─────────────────────────────────────────

def _render_factor(args: list[str]) -> None:
    """FACTOR — Fama-French 5-factor + momentum exposure analysis."""
    st.subheader("FACTOR — Multi-Factor Risk Decomposition")
    holdings_input = st.text_area("Portfolio Holdings (ticker:weight)", "SPY:1.0")
    if st.button("Compute Factor Exposures"):
        with st.spinner("Fetching FF5 factors..."):
            try:
                from sentinel.spr.factor_model import fetch_ff5_factors, decompose_portfolio
                holdings = {}
                for line in holdings_input.strip().split("\n"):
                    if ":" in line:
                        t, w = line.strip().split(":")
                        holdings[t.strip()] = float(w.strip())

                end = date.today()
                start = end - timedelta(days=365)
                result = decompose_portfolio(holdings=holdings, returns_df=pd.DataFrame(), start=start, end=end)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Portfolio Alpha (ann.)", f"{result.portfolio_alpha*100:.2f}%")
                with col2:
                    st.metric("R²", f"{result.r_squared:.3f}")

                exp_df = pd.DataFrame([{"Factor": k, "Contribution %": f"{v*100:.1f}%"}
                                       for k, v in result.factor_contributions.items()])
                st.write("**Factor Variance Attribution**")
                st.dataframe(exp_df)
            except Exception as e:
                st.error(f"Factor model error: {e}")


# ─── CAL: Economic Calendar ───────────────────────────────────────────────────

def _render_cal(args: list[str]) -> None:
    """CAL — Economic calendar: upcoming macro data releases."""
    st.subheader("CAL — Economic Calendar")
    days_ahead = st.slider("Days ahead", 7, 60, 30)
    if st.button("Load Calendar"):
        with st.spinner("Fetching FRED release schedule..."):
            try:
                from sentinel.sma.economic_calendar import get_calendar
                from sentinel.core.config import get_settings
                s = get_settings()
                calendar = run_async(get_calendar(api_key=s.fred_api_key, days_ahead=days_ahead))

                if calendar.next_high_impact:
                    nhi = calendar.next_high_impact
                    st.success(f"Next high-impact: **{nhi.name}** on {nhi.release_date} ({nhi.release_time or 'time TBD'})")

                rows = [{"Date": r.release_date, "Release": r.name, "Importance": r.importance.upper(),
                         "Frequency": r.frequency, "Time": r.release_time or ""}
                        for r in sorted(calendar.releases, key=lambda x: x.release_date)]
                df = pd.DataFrame(rows)

                def highlight_importance(row):
                    if row["Importance"] == "HIGH":
                        return ["background-color: #3d1a1a"] * len(row)
                    elif row["Importance"] == "MEDIUM":
                        return ["background-color: #2a2a1a"] * len(row)
                    return [""] * len(row)

                st.dataframe(df.style.apply(highlight_importance, axis=1))
            except Exception as e:
                st.error(f"Calendar error: {e}")


# ─── FX: Foreign Exchange Rates ───────────────────────────────────────────────

def _render_fx(args: list[str]) -> None:
    """FX — ECB official FX rates via Frankfurter API."""
    base = args[0] if args else "USD"
    st.subheader(f"FX — Foreign Exchange: {base} Base Rates (ECB)")
    st.caption("Source: European Central Bank via Frankfurter API (free)")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", date.today() - timedelta(days=90))
    with col2:
        end_date = st.date_input("End", date.today())
    if st.button("Load FX Rates"):
        with st.spinner("Fetching ECB rates..."):
            try:
                from sentinel.sds.adapters.fx_adapter import FXAdapter
                fx = FXAdapter()
                bars = run_async(fx.fetch_base_rates(
                    base=base.upper(),
                    start=start_date,
                    end=end_date,
                ))
                if bars:
                    df = pd.DataFrame([
                        {"Date": b.date, "Symbol": b.pair, "Rate": float(b.close)}
                        for b in bars
                    ])
                    pivoted = df.pivot(index="Date", columns="Symbol", values="Rate")
                    st.line_chart(pivoted)
                    st.dataframe(pivoted.tail(10))
                else:
                    st.info("No FX data available for selected period")
            except Exception as e:
                st.error(f"FX error: {e}")


# ─── GLOBAL: Global Macro Dashboard ──────────────────────────────────────────

def _render_global(args: list[str]) -> None:
    """GLOBAL — Cross-country macro dashboard: GDP, CPI, unemployment, rates."""
    st.subheader("GLOBAL — Global Macro Dashboard (FRED Multi-Country)")
    if st.button("Load Global Macro") or args:
        with st.spinner("Fetching FRED macro data for G7+..."):
            try:
                from sentinel.sma.global_macro import get_global_dashboard, get_yield_curve_comparison
                from sentinel.core.config import get_settings
                s = get_settings()
                dashboard = run_async(get_global_dashboard(fred_api_key=s.fred_api_key))
                curves = run_async(get_yield_curve_comparison(fred_api_key=s.fred_api_key))

                st.write("**Country Macro Scores (0-10)**")
                score_rows = []
                for code, data in dashboard.items():
                    score_rows.append({
                        "Country": code.upper(),
                        "GDP Growth": f"{data.gdp_growth:.1f}%" if data.gdp_growth is not None else "N/A",
                        "Inflation": f"{data.inflation:.1f}%" if data.inflation is not None else "N/A",
                        "Unemployment": f"{data.unemployment:.1f}%" if data.unemployment is not None else "N/A",
                        "Policy Rate": f"{data.policy_rate:.2f}%" if data.policy_rate is not None else "N/A",
                        "Score": f"{data.macro_score:.1f}/10" if data.macro_score is not None else "N/A",
                    })
                st.dataframe(pd.DataFrame(score_rows), use_container_width=True)

                if curves:
                    st.write("**G7 Yield Curve Spreads (10Y-2Y)**")
                    curve_rows = [
                        {"Country": c.country.upper(), "2Y": f"{c.rate_2y:.2f}%",
                         "10Y": f"{c.rate_10y:.2f}%",
                         "Spread": f"{c.spread_10y_2y:+.2f}%",
                         "Inverted": "YES" if c.is_inverted else "No"}
                        for c in curves if c.rate_2y is not None and c.rate_10y is not None
                    ]
                    st.dataframe(pd.DataFrame(curve_rows), use_container_width=True)
            except Exception as e:
                st.error(f"Global macro error: {e}")


# ─── KELLY: Kelly / Position Sizer ────────────────────────────────────────────

def _render_kelly(args: list[str]) -> None:
    """KELLY — Kelly Criterion, volatility targeting, and risk parity position sizer."""
    st.subheader("KELLY — Optimal Position Sizer")
    method = st.selectbox("Method", ["kelly", "vol_target", "risk_parity", "equal"])
    col1, col2, col3 = st.columns(3)
    with col1:
        portfolio_value = st.number_input("Portfolio ($)", value=100_000.0, min_value=1.0)
    with col2:
        avg_win = st.number_input("Avg Win %", value=8.0, min_value=0.1) / 100
        target_vol = st.slider("Target Vol", 0.05, 0.30, 0.15, 0.01)
    with col3:
        avg_loss = st.number_input("Avg Loss %", value=4.0, min_value=0.1) / 100
        win_rate = st.slider("Win Rate", 0.4, 0.8, 0.55, 0.01)

    holdings_input = st.text_area("Holdings (ticker:weight)", "SPY:0.6\nTLT:0.3\nGLD:0.1")

    if st.button("Compute Sizes"):
        with st.spinner("Computing optimal position sizes..."):
            try:
                import pandas as _pd
                from sentinel.spr.kelly_sizer import size_portfolio, kelly_from_returns

                holdings = {}
                for line in holdings_input.strip().split("\n"):
                    if ":" in line:
                        t, w = line.strip().split(":")
                        holdings[t.strip()] = float(w.strip())

                if method == "kelly":
                    n = 1000
                    wins = [avg_win] * round(win_rate * n)
                    losses = [-avg_loss] * (n - round(win_rate * n))
                    result = kelly_from_returns(
                        returns=_pd.Series(wins + losses),
                        risk_free_rate=0.045,
                    )
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Kelly Fraction", f"{result.kelly_fraction:.3f}")
                    col2.metric("Half-Kelly", f"{result.half_kelly:.3f}")
                    col3.metric("Dollar Position", f"${result.kelly_fraction * portfolio_value:,.0f}")
                    st.info(f"Recommendation: {result.recommendation}")
                else:
                    # size_portfolio needs returns data for risk_parity/vol_target;
                    # without live data, use user-specified weights scaled to capital
                    tickers_list = list(holdings.keys())
                    user_total = sum(holdings.values()) or 1.0
                    if method == "equal" or not tickers_list:
                        n_t = len(tickers_list) or 1
                        weights_map = {t: 1.0 / n_t for t in tickers_list}
                    else:
                        weights_map = {t: v / user_total for t, v in holdings.items()}
                    rows = [{"Ticker": t, "Weight": f"{w:.3f}", "Dollar Amount": f"${w * portfolio_value:,.0f}"}
                            for t, w in weights_map.items()]
                    st.dataframe(_pd.DataFrame(rows), use_container_width=True)
                    if method != "equal":
                        st.caption("risk_parity/vol_target require live price history — using specified weights")
            except Exception as e:
                st.error(f"Kelly error: {e}")


# ─── SHORT: Short Interest Screen ─────────────────────────────────────────────

def _render_short(args: list[str]) -> None:
    """SHORT — FINRA short interest data and short squeeze candidates."""
    ticker = args[0].upper() if args else None
    st.subheader("SHORT — Short Interest & Squeeze Screen (FINRA)")
    st.caption("Source: FINRA Daily RegSHO Short Volume files")

    if ticker:
        st.write(f"**Single ticker: {ticker}**")
        with st.spinner(f"Fetching short interest for {ticker}..."):
            try:
                from sentinel.sds.adapters.short_interest_adapter import ShortInterestAdapter
                si = ShortInterestAdapter()
                record = run_async(si.get_short_interest(ticker))
                if record:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Short Volume", f"{record.short_volume:,}")
                    col2.metric("Total Volume", f"{record.total_volume:,}")
                    col3.metric("Short %", f"{record.short_pct * 100:.1f}%")
                    if record.short_pct > 0.40:
                        st.warning(f"High short interest ({record.short_pct*100:.0f}%) — potential squeeze candidate")
                else:
                    st.info(f"No FINRA short interest data for {ticker}")
            except Exception as e:
                st.error(f"Short interest error: {e}")
    else:
        threshold = st.slider("Min Short %", 20, 60, 40) / 100
        if st.button("Scan for Squeeze Candidates"):
            with st.spinner("Scanning FINRA short data..."):
                try:
                    from sentinel.sds.adapters.short_interest_adapter import ShortInterestAdapter
                    si = ShortInterestAdapter()
                    candidates = run_async(si.get_squeeze_candidates(min_short_pct=threshold))
                    if candidates:
                        rows = [{"Ticker": c.ticker, "Short %": f"{c.short_pct*100:.1f}%",
                                 "Short Volume": f"{c.short_volume:,}", "Date": str(c.date)}
                                for c in candidates[:30]]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    else:
                        st.info("No squeeze candidates found above threshold")
                except Exception as e:
                    st.error(f"Short squeeze scan error: {e}")


# ─── BONDS: Corporate Bond Quotes (FINRA TRACE) ───────────────────────────────

def _render_bonds(args: list[str]) -> None:
    """BONDS — FINRA TRACE corporate bond quotes and credit spread curve."""
    ticker = args[0].upper() if args else st.text_input("Ticker:", "AAPL")
    if not ticker:
        return
    st.subheader(f"BONDS — Corporate Bond Market: {ticker} (FINRA TRACE)")
    st.caption("Source: FINRA TRACE Aggregate data — investment grade & high yield")
    build_curve = st.checkbox("Build credit spread curve", value=True)

    if st.button("Load Bond Data") or args:
        with st.spinner("Fetching FINRA TRACE data..."):
            try:
                from sentinel.sbx.trace_client import TRACEClient
                client = TRACEClient()
                quotes = run_async(client.get_bond_quotes(ticker))

                if not quotes:
                    st.info(f"No TRACE bond data found for {ticker}")
                    return

                st.success(f"Found {len(quotes)} bond(s)")
                rows = [{"Maturity": str(q.maturity_date),
                         "Coupon": f"{q.coupon_rate:.2f}%" if q.coupon_rate else "N/A",
                         "Last Price": f"{q.last_price:.2f}" if q.last_price else "N/A",
                         "OAS Spread": f"{q.oas_spread:.0f}bps" if q.oas_spread else "N/A",
                         "Volume": f"{q.volume:,}" if q.volume else "N/A"}
                        for q in quotes[:20]]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                if build_curve:
                    curve = run_async(client.build_credit_curve(ticker))
                    if curve and curve.points:
                        st.write("**Credit Spread Curve**")
                        curve_df = pd.DataFrame([
                            {"Maturity (yrs)": p.maturity_years, "OAS Spread (bps)": p.spread_bps}
                            for p in curve.points
                        ])
                        st.line_chart(curve_df.set_index("Maturity (yrs)"))
            except Exception as e:
                st.error(f"TRACE error: {e}")


# ─── SEG: Business Segment Breakdown ─────────────────────────────────────────

def _render_seg(args: list[str]) -> None:
    """SEG — Business segment revenue breakdown from EDGAR XBRL."""
    ticker = args[0].upper() if args else st.text_input("Ticker:", "AAPL")
    if not ticker:
        return
    st.subheader(f"SEG — Business Segments: {ticker} (EDGAR XBRL)")
    st.caption("Parses 10-K dimensional XBRL facts to extract segment revenue")

    if st.button("Load Segments") or args:
        with st.spinner("Parsing EDGAR XBRL segments..."):
            try:
                from sentinel.sfe.segment_parser import get_segment_breakdown
                from sentinel.sim.instrument_master import InstrumentMaster
                from sentinel.core.config import get_settings
                s = get_settings()
                im = InstrumentMaster(s)
                instrument = run_async(im.resolve(ticker=ticker))
                cik = getattr(instrument, "cik", None) if instrument else None

                if not cik:
                    st.warning(f"CIK not found for {ticker}. Enter manually below.")
                    cik = st.text_input("CIK (from SEC EDGAR):", "")
                    if not cik:
                        return

                result = run_async(get_segment_breakdown(cik=cik, ticker=ticker))

                if result.segments:
                    st.write(f"**Filing Period: {result.period_end}** (Form: {result.form_type})")
                    rows = [{"Segment": seg.name,
                             "Revenue ($M)": f"{float(seg.revenue)/1e6:.1f}",
                             "% of Total": f"{seg.pct_of_total:.1f}%"}
                            for seg in result.segments]
                    df_seg = pd.DataFrame(rows)
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.dataframe(df_seg, use_container_width=True)
                    with col2:
                        chart_data = {seg.name: float(seg.pct_of_total) for seg in result.segments}
                        st.bar_chart(chart_data)
                else:
                    st.info("No segment data found — company may not report segments separately")
            except Exception as e:
                st.error(f"Segment parser error: {e}")


# ─── OFLOW: Options Flow Screener ────────────────────────────────────────────

def _render_oflow(args: list[str]) -> None:
    """OFLOW — Options flow screener: unusual volume, IV spikes, put skew, gamma walls."""
    tickers_input = " ".join(args) if args else ""
    st.subheader("OFLOW — Options Flow Screener")
    st.caption("Screens for unusual options activity: volume spikes, IV expansion, directional flows")

    col1, col2 = st.columns(2)
    with col1:
        tickers_text = st.text_area(
            "Tickers (space or comma separated)",
            tickers_input or "SPY AAPL TSLA NVDA MSFT"
        )
    with col2:
        vol_ratio = st.slider("Min Volume Ratio (vs avg)", 1.5, 5.0, 2.0, 0.5)
        st.caption("Current volume must exceed historical avg x this multiple")

    if st.button("Scan Options Flow"):
        tickers = [t.strip().upper() for t in tickers_text.replace(",", " ").split() if t.strip()]
        if not tickers:
            st.warning("Enter at least one ticker")
            return

        with st.spinner(f"Scanning {len(tickers)} tickers for unusual options activity..."):
            try:
                from sentinel.sse.options_screener import (
                    screen_universe, OptionsScreenerCriteria,
                )
                from sentinel.core.config import get_settings
                s = get_settings()
                criteria = OptionsScreenerCriteria(min_volume_ratio=vol_ratio)

                # Fetch spot prices; fall back to $100 placeholder if unavailable
                prices: dict[str, float] = {}
                try:
                    import yfinance as _yf
                    for _t in tickers:
                        try:
                            info = _yf.Ticker(_t).fast_info
                            prices[_t] = float(getattr(info, "last_price", 100.0) or 100.0)
                        except Exception:
                            prices[_t] = 100.0
                except Exception:
                    prices = {_t: 100.0 for _t in tickers}

                results = run_async(screen_universe(
                    tickers=tickers,
                    prices=prices,
                    polygon_api_key=s.polygon_api_key,
                    criteria=criteria,
                ))

                if not results:
                    st.info("No unusual options activity detected above threshold")
                    return

                # Flatten: one row per OptionsScreenResult, showing top alert
                rows = []
                all_alert_types: list[str] = []
                for r in results[:50]:
                    top = r.alerts[0] if r.alerts else None
                    all_alert_types.extend(a.alert_type for a in r.alerts)
                    rows.append({
                        "Ticker": r.ticker,
                        "Signal": r.signal.replace("_", " ").title(),
                        "P/C Ratio": f"{r.put_call_ratio:.2f}" if r.put_call_ratio is not None else "—",
                        "IV Pct": f"{r.iv_percentile:.0f}%" if r.iv_percentile is not None else "—",
                        "Skew 25d": f"{r.skew_25d:+.3f}" if r.skew_25d is not None else "—",
                        "Top Alert": top.alert_type.replace("_", " ").title() if top else "—",
                        "Severity": top.severity.upper() if top else "—",
                        "# Alerts": len(r.alerts),
                    })

                df_alerts = pd.DataFrame(rows)
                st.success(f"Found {len(results)} ticker(s) with options activity")

                alert_counts = pd.Series(all_alert_types).value_counts() if all_alert_types else pd.Series(dtype=int)
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.dataframe(df_alerts, use_container_width=True)
                with col2:
                    st.write("**Alert Types**")
                    if not alert_counts.empty:
                        st.bar_chart(alert_counts)
            except Exception as e:
                st.error(f"Options flow error: {e}")


# ─── STRAT — NL→Strategy Generator ──────────────────────────────────────────

def _render_strat(args: list[str]):
    st.markdown("## STRAT — NL→Strategy Generator")
    st.caption("Translate a natural-language description into a structured trading strategy spec via Claude tool-use")

    default_desc = " ".join(args) if args else ""
    description = st.text_area(
        "Strategy description:",
        value=default_desc or "Buy large-cap tech stocks with RSI < 35 (oversold) and positive revenue growth. "
                               "Sell when RSI > 65 or 15% stop-loss hit. Position size using Kelly criterion.",
        height=120,
    )
    model = st.selectbox("Claude model:", ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"], index=0)

    if st.button("⚡ Generate Strategy Spec", key="strat_gen"):
        with st.spinner("Claude is parsing your strategy..."):
            try:
                from sentinel.sil.strategy_generator import generate_strategy
                from sentinel.core.config import get_settings
                s = get_settings()
                api_key = getattr(s, "anthropic_api_key", None) or ""
                strategy = run_async(generate_strategy(description=description, anthropic_api_key=api_key, model=model))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Strategy Name", strategy.name)
                    st.metric("Confidence", f"{strategy.confidence:.0%}")
                with col2:
                    st.metric("Asset Class", strategy.universe.asset_class.title())
                    st.metric("Stop Loss", f"{strategy.stop_loss_pct:.0%}")
                with col3:
                    st.metric("Sizing Method", strategy.position_sizing.method.title())
                    st.metric("Rebalance", strategy.rebalance_frequency.title())

                st.markdown("**Entry Signals**")
                for sig in strategy.entry_signals:
                    st.markdown(f"- `{sig.indicator}` {sig.operator} {sig.threshold or ''} (lookback {sig.lookback})")

                st.markdown("**Exit Signals**")
                for sig in strategy.exit_signals:
                    st.markdown(f"- `{sig.indicator}` {sig.operator} {sig.threshold or ''}")

                if strategy.universe.tickers:
                    st.markdown(f"**Universe:** {', '.join(strategy.universe.tickers[:10])}")
                if strategy.universe.index:
                    st.markdown(f"**Index:** {strategy.universe.index}")
                if strategy.warnings:
                    for w in strategy.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"Strategy generation error: {e}")


# ─── RESEARCH — Autonomous Research Agent ────────────────────────────────────

def _render_research(args: list[str]):
    ticker = args[0].upper() if args else st.text_input("Ticker:", "NVDA")
    if not ticker:
        return
    st.markdown(f"## RESEARCH — AI Research Memo: {ticker}")
    st.caption("Autonomous multi-step research: fundamentals + news + insider trades + macro → Claude synthesis")

    question = st.text_input(
        "Research question:",
        value="What is the investment thesis and key risks for this company?",
    )

    if st.button("🔬 Run Research Agent", key="research_run"):
        with st.spinner(f"Researching {ticker}... (10-15 sec)"):
            try:
                from sentinel.sil.research_agent import research_ticker as _research
                from sentinel.core.config import get_settings
                s = get_settings()
                api_key = getattr(s, "anthropic_api_key", None) or ""
                memo = run_async(_research(ticker=ticker, question=question, anthropic_api_key=api_key))

                rec_color = {"Buy": "positive", "Sell": "negative", "Hold": "neutral"}.get(memo.recommendation, "neutral")
                st.markdown(f"### Recommendation: <span class='{rec_color}'>{memo.recommendation}</span> "
                            f"(confidence {memo.confidence:.0%})", unsafe_allow_html=True)
                st.markdown(f"**Data quality:** {memo.data_quality} | **Sources:** {', '.join(memo.sources_used)}")
                st.divider()

                st.markdown("**Executive Summary**")
                st.write(memo.executive_summary)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Bull Case**")
                    for b in memo.bull_case:
                        st.markdown(f"✅ {b}")
                with col2:
                    st.markdown("**Bear Case**")
                    for b in memo.bear_case:
                        st.markdown(f"⚠️ {b}")

                if memo.key_facts:
                    st.markdown("**Key Facts**")
                    for f in memo.key_facts[:8]:
                        st.markdown(f"- {f.fact} *(source: {f.source}, confidence: {f.confidence})*")

                if memo.valuation:
                    st.markdown("**Valuation**")
                    for v in memo.valuation:
                        upside = f" | upside {v.upside_pct:+.0f}%" if v.upside_pct is not None else ""
                        st.markdown(f"- {v.method}: implied ${v.implied_value:.2f}{upside} | {v.assumptions}")

            except Exception as e:
                st.error(f"Research agent error: {e}")


# ─── TRENDS — Google Trends Signal ───────────────────────────────────────────

def _render_trends(args: list[str]):
    ticker = args[0].upper() if args else st.text_input("Ticker:", "AAPL")
    if not ticker:
        return
    st.markdown(f"## TRENDS — Google Search Momentum: {ticker}")
    st.caption("5-year weekly Google Trends data — 4-week vs 52-week momentum z-score")

    col1, col2 = st.columns(2)
    with col1:
        geo = st.selectbox("Geography:", ["US", "GB", "DE", "JP", ""], index=0,
                           format_func=lambda x: x or "Worldwide")
    with col2:
        extra_kw = st.text_input("Extra keywords (comma-sep):", "")

    if st.button("📈 Get Trends Signal", key="trends_run"):
        with st.spinner("Fetching Google Trends..."):
            try:
                from sentinel.sma.google_trends import get_trend_signal
                keywords = [k.strip() for k in extra_kw.split(",") if k.strip()] if extra_kw else []
                signal = get_trend_signal(ticker=ticker, keywords=keywords, geo=geo)

                dir_color = "positive" if signal.direction in ("rising", "spike") else (
                    "negative" if signal.direction == "falling" else "neutral")
                st.markdown(f"### Trend Direction: <span class='{dir_color}'>{signal.direction.title()}</span>",
                            unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Signal Strength", signal.signal_strength.title())
                c2.metric("Momentum Z-Score", f"{signal.momentum_zscore:.2f}")
                c3.metric("4-Week Avg", f"{signal.avg_interest_4w:.1f}")
                c4.metric("52-Week Avg", f"{signal.avg_interest_52w:.1f}")

                if signal.warning:
                    st.warning(signal.warning)
                if signal.related_queries:
                    st.markdown("**Related Queries:**")
                    st.write(", ".join(signal.related_queries[:10]))
            except Exception as e:
                st.error(f"Google Trends error: {e}")


# ─── DEFI — DeFi Dashboard ───────────────────────────────────────────────────

def _render_defi(args: list[str]):
    st.markdown("## DEFI — DeFi Ecosystem Dashboard")
    st.caption("Live TVL, protocols, yields, and stablecoin data via DeFiLlama (free, no API key)")

    top_n = st.slider("Top N protocols:", 5, 50, 20)

    if st.button("🔗 Load DeFi Dashboard", key="defi_load"):
        with st.spinner("Fetching DeFiLlama data..."):
            try:
                from sentinel.snm.defi_analytics import DefiLlamaClient
                client = DefiLlamaClient()
                dash = run_async(client.get_dashboard(top_n=top_n))

                c1, c2, c3 = st.columns(3)
                c1.metric("Total DeFi TVL", f"${dash.total_tvl / 1e9:.1f}B")
                c2.metric("Protocols Tracked", dash.protocol_count)
                c3.metric("Chains Tracked", dash.chain_count)

                st.markdown("**Top Protocols by TVL**")
                if dash.top_protocols:
                    rows = [{"Protocol": p.name, "Chain": p.chain, "TVL ($B)": f"${p.tvl / 1e9:.2f}",
                             "Category": p.category, "7d Change": f"{p.change_7d:+.1f}%" if p.change_7d else "N/A"}
                            for p in dash.top_protocols[:top_n]]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                if dash.top_yields:
                    st.markdown("**Top Yield Opportunities**")
                    rows = [{"Pool": y.pool, "Chain": y.chain, "Symbol": y.symbol,
                             "APY": f"{y.apy:.1f}%", "TVL ($M)": f"${y.tvl_usd / 1e6:.1f}",
                             "IL Risk": y.il_risk}
                            for y in dash.top_yields[:10]]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                if dash.stablecoins:
                    st.markdown("**Stablecoin Market**")
                    rows = [{"Name": s.name, "Symbol": s.symbol, "Peg": s.peg_type,
                             "Mkt Cap ($B)": f"${s.circulating / 1e9:.2f}",
                             "Price": f"${s.price:.4f}"}
                            for s in dash.stablecoins[:8]]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

            except Exception as e:
                st.error(f"DeFi dashboard error: {e}")


# ─── ONCHAIN — On-Chain Crypto Signals ───────────────────────────────────────

def _render_onchain(args: list[str]):
    symbol = args[0].lower() if args else ""
    if not symbol:
        symbol = st.text_input("CoinGecko ID:", "bitcoin",
                               help="e.g. bitcoin, ethereum, solana, cardano")
    if not symbol:
        return

    st.markdown(f"## ONCHAIN — On-Chain Signals: {symbol.title()}")
    st.caption("NVT proxy (mcap/volume), MVRV proxy (price vs 30d avg), fear/greed via CoinGecko (free)")

    if st.button("⛓ Get On-Chain Signal", key="onchain_run"):
        with st.spinner(f"Fetching on-chain data for {symbol}..."):
            try:
                from sentinel.snm.onchain_metrics import OnChainClient
                client = OnChainClient()
                sig = run_async(client.get_signal(symbol=symbol))

                sig_color = "positive" if "buy" in sig.overall_signal else (
                    "negative" if "sell" in sig.overall_signal else "neutral")
                st.markdown(f"### Signal: <span class='{sig_color}'>{sig.overall_signal.replace('_', ' ').title()}</span> "
                            f"(strength {sig.signal_strength:.0%})", unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Price (USD)", f"${sig.price_usd:,.2f}" if sig.price_usd else "N/A")
                c2.metric("Market Cap ($B)", f"${sig.market_cap / 1e9:.1f}" if sig.market_cap else "N/A")
                c3.metric("NVT Proxy", f"{sig.nvt_proxy:.1f}" if sig.nvt_proxy else "N/A")
                c4.metric("Fear/Greed", f"{sig.fear_greed_proxy:.0f}/100" if sig.fear_greed_proxy else "N/A")

                c5, c6, c7 = st.columns(3)
                c5.metric("NVT Signal", sig.nvt_signal)
                c6.metric("MVRV Signal", sig.mvrv_signal)
                c7.metric("30d Return", f"{sig.price_change_30d:+.1f}%" if sig.price_change_30d else "N/A")

                if sig.warnings:
                    for w in sig.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"On-chain signals error: {e}")


# ─── EKP — Earnings KPI Extractor ────────────────────────────────────────────

def _render_ekp(args: list[str]):
    ticker = args[0].upper() if args else st.text_input("Ticker:", "MSFT")
    if not ticker:
        return
    st.markdown(f"## EKP — Earnings KPI Extraction: {ticker}")
    st.caption("EDGAR MD&A → Claude extracts structured KPIs, management tone, and forward guidance")

    col1, col2 = st.columns(2)
    with col1:
        form_type = st.selectbox("Filing type:", ["10-Q", "10-K"], index=0)

    if st.button("📋 Extract Earnings KPIs", key="ekp_run"):
        with st.spinner(f"Fetching EDGAR {form_type} and extracting KPIs for {ticker}..."):
            try:
                from sentinel.sfe.earnings_kpi import get_earnings_kpi
                from sentinel.core.config import get_settings
                s = get_settings()
                api_key = getattr(s, "anthropic_api_key", None)
                result = run_async(get_earnings_kpi(ticker=ticker, anthropic_api_key=api_key, form_type=form_type))

                c1, c2, c3 = st.columns(3)
                c1.metric("Period", result.period)
                c2.metric("Filing Type", result.filing_type)
                c3.metric("Filed Date", str(result.filed_date) if result.filed_date else "N/A")

                if result.management_tone:
                    st.markdown("**Management Tone**")
                    tone = result.management_tone
                    tc1, tc2, tc3 = st.columns(3)
                    tc1.metric("Overall", tone.overall.title())
                    tc2.metric("Confidence Score", f"{tone.confidence_score:.0%}")
                    tc3.metric("Guidance Provided", "Yes" if tone.guidance_provided else "No")
                    if tone.key_themes:
                        st.markdown(f"*Key themes: {', '.join(tone.key_themes[:5])}*")

                if result.kpis:
                    st.markdown("**Extracted KPIs**")
                    rows = [{"KPI": k.name, "Value": k.value, "Unit": k.unit or "",
                             "Period": k.period or "", "Trend": k.trend or "",
                             "Guidance": "✅" if k.is_guidance else ""}
                            for k in result.kpis]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"Earnings KPI error: {e}")


# ─── ESG — ESG Profile ───────────────────────────────────────────────────────

def _render_esg(args: list[str]):
    ticker = args[0].upper() if args else st.text_input("Ticker:", "JPM")
    if not ticker:
        return
    st.markdown(f"## ESG — ESG Proxy Profile: {ticker}")
    st.caption("Derived from EDGAR DEF 14A (proxy) + 10-K. Proxy signals only — not MSCI/Sustainalytics rated.")

    if st.button("🌿 Get ESG Profile", key="esg_run"):
        with st.spinner(f"Fetching EDGAR filings for {ticker}..."):
            try:
                from sentinel.sfe.esg_parser import get_esg_profile as _esg
                profile = run_async(_esg(ticker=ticker))

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("E Score", f"{profile.environmental_score:.0f}/100" if profile.environmental_score else "N/A")
                c2.metric("S Score", f"{profile.social_score:.0f}/100" if profile.social_score else "N/A")
                c3.metric("G Score", f"{profile.governance_score:.0f}/100" if profile.governance_score else "N/A")
                c4.metric("Composite", f"{profile.composite_score:.0f}/100" if profile.composite_score else "N/A")

                if profile.pay_ratio and profile.pay_ratio.pay_ratio:
                    st.metric("CEO Pay Ratio", f"{profile.pay_ratio.pay_ratio:.0f}x")

                if profile.board_composition and profile.board_composition.gender_diversity_pct is not None:
                    bc = profile.board_composition
                    bc1, bc2, bc3 = st.columns(3)
                    bc1.metric("Total Directors", bc.total_directors)
                    bc2.metric("Gender Diversity", f"{bc.gender_diversity_pct:.0f}%")
                    bc3.metric("Independence", f"{bc.independence_pct:.0f}%" if bc.independence_pct else "N/A")

                if profile.environmental and profile.environmental.climate_mention_count:
                    env = profile.environmental
                    st.markdown("**Environmental Signals**")
                    st.markdown(f"- Climate mentions: {env.climate_mention_count} | "
                                f"Carbon mentions: {env.carbon_mention_count} | "
                                f"Net zero: {'✅' if env.net_zero_mentioned else '❌'}")

                if profile.data_completeness:
                    st.caption(f"Data completeness: {profile.data_completeness}")
            except Exception as e:
                st.error(f"ESG profile error: {e}")


# ─── ALPHA — Alpha Signal Library ────────────────────────────────────────────

def _render_alpha(args: list[str]):
    ticker = args[0].upper() if args else st.text_input("Ticker:", "AAPL")
    if not ticker:
        return
    st.markdown(f"## ALPHA — Composite Alpha Signals: {ticker}")
    st.caption("Congressional STOCK Act trades + CFTC COT + SEC Form 4 insiders → z-scored alpha composite")

    if st.button("📡 Get Alpha Signals", key="alpha_run"):
        with st.spinner(f"Fetching alpha signals for {ticker}..."):
            try:
                from sentinel.spr.signal_library import get_full_signal
                comp = run_async(get_full_signal(ticker=ticker))

                sig_color = "positive" if "buy" in comp.composite.signal else (
                    "negative" if "sell" in comp.composite.signal else "neutral")
                st.markdown(f"### Composite: <span class='{sig_color}'>"
                            f"{comp.composite.signal.replace('_', ' ').title()}</span> "
                            f"(strength {comp.composite.strength:.0%})", unsafe_allow_html=True)
                st.markdown(f"*{comp.composite.explanation}*")

                cols = st.columns(3)
                for col, (label, src) in zip(cols, [
                    ("Congressional", comp.congress),
                    ("COT", comp.cot),
                    ("Insider", comp.insider),
                ]):
                    if src:
                        sig = src.signal if hasattr(src, "signal") else None
                        if sig:
                            col.metric(label, sig.signal.replace("_", " ").title(),
                                       f"z={sig.z_score:.2f}" if sig.z_score else "")
                    else:
                        col.metric(label, "No data")

                if comp.congress and hasattr(comp.congress, "cluster_detected") and comp.congress.cluster_detected:
                    st.info(f"🔔 Congressional cluster detected — {comp.congress.unique_members_buying} members buying")
            except Exception as e:
                st.error(f"Alpha signals error: {e}")


# ─── XLS — Excel Export ──────────────────────────────────────────────────────

def _render_xls(args: list[str]):
    ticker = args[0].upper() if args else st.text_input("Ticker:", "AAPL")
    if not ticker:
        return
    st.markdown(f"## XLS — Bloomberg-Style Excel Export: {ticker}")
    st.caption("Bloomberg dark-theme workbook with HP (price), FA (fundamentals), PORT, ECOS sheets via xlsxwriter")

    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("History period:", ["1y", "2y", "5y"], index=0)
    with col2:
        include_macro = st.checkbox("Include ECOS macro sheet", value=True)

    if st.button("📊 Export to Excel", key="xls_run"):
        with st.spinner(f"Building Excel workbook for {ticker}..."):
            try:
                import yfinance as yf
                from sentinel.stu.excel_export import export_to_excel
                hist = yf.Ticker(ticker).history(period=period)
                if hist.empty:
                    st.error(f"No price data available for {ticker}")
                    return

                fundamentals: dict = {}
                try:
                    import requests as _req
                    resp = _req.get(f"http://localhost:8000/api/v1/data/fundamentals/{ticker}", timeout=5)
                    if resp.ok:
                        fundamentals = resp.json()
                except Exception:
                    pass

                buf = export_to_excel(
                    ticker=ticker,
                    ohlcv_df=hist,
                    fundamentals=fundamentals if fundamentals else None,
                    macro_series={"include": include_macro} if include_macro else None,
                )
                st.download_button(
                    label=f"⬇️ Download {ticker}.xlsx",
                    data=buf.getvalue(),
                    file_name=f"SENTINEL_{ticker}_{date.today()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                st.success(f"Excel workbook ready — {ticker} ({period})")
            except Exception as e:
                st.error(f"Excel export error: {e}")


# ─── NGAAP — Non-GAAP Metrics ────────────────────────────────────────────────

def _render_ngaap(args: list[str]):
    ticker = args[0].upper() if args else st.text_input("Ticker:", "MSFT")
    if not ticker:
        return
    st.markdown(f"## NGAAP — Non-GAAP Earnings Parser: {ticker}")
    st.caption("Parses latest EDGAR 8-K earnings release for Adjusted EBITDA, Non-GAAP EPS, FCF via regex")

    if st.button("Parse Non-GAAP Metrics", key="ngaap_run"):
        with st.spinner(f"Fetching EDGAR 8-K for {ticker}..."):
            try:
                from sentinel.sfe.non_gaap_parser import get_non_gaap_metrics
                result = run_async(get_non_gaap_metrics(ticker=ticker))

                c1, c2, c3 = st.columns(3)
                c1.metric("Ticker", result.ticker)
                c2.metric("Accession", result.accession_number or "N/A")
                c3.metric("Filed Date", str(result.filed_date) if result.filed_date else "N/A")

                if result.metrics:
                    st.markdown("**Non-GAAP Metrics Extracted**")
                    rows = [{"Metric": m.name, "Value": m.value, "Unit": m.unit or "",
                             "Period": m.period or "", "YoY": m.yoy_change or ""}
                            for m in result.metrics]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("No non-GAAP metrics found in latest 8-K.")

                if result.reconciliations:
                    st.markdown("**GAAP to Non-GAAP Reconciliation**")
                    for rec in result.reconciliations[:3]:
                        st.markdown(f"**{rec.metric_name}**: GAAP {rec.gaap_value} to Non-GAAP {rec.non_gaap_value}")
                        if rec.adjustments:
                            for adj in rec.adjustments[:5]:
                                st.markdown(f"  - {adj.description}: {adj.amount}")

                if result.guidance:
                    st.markdown("**Forward Guidance**")
                    for g in result.guidance[:5]:
                        st.markdown(f"- {g.name}: {g.value} ({g.period or 'N/A'})")

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"Non-GAAP parser error: {e}")


# ─── COMPS — Comparable Company Analysis ─────────────────────────────────────

def _render_comps(args: list[str]):
    ticker = args[0].upper() if args else st.text_input("Ticker:", "NVDA")
    if not ticker:
        return
    st.markdown(f"## COMPS — Comparable Company Analysis: {ticker}")
    st.caption("Builds peer comps table from EDGAR XBRL + yfinance: EV/EBITDA, P/E, EV/Revenue, margins, growth")

    custom_peers = st.text_input("Custom peer tickers (comma-separated, optional):", "")

    if st.button("Build Comps Table", key="comps_run"):
        with st.spinner(f"Building comps table for {ticker} and peers..."):
            try:
                from sentinel.sfe.comps_table import get_comps_table
                peers = [p.strip().upper() for p in custom_peers.split(",") if p.strip()] or None
                result = run_async(get_comps_table(ticker=ticker, include_peers=peers))

                st.markdown(f"**Sector:** {result.sector or 'N/A'} | **Peers:** {len(result.peers)}")

                if result.subject:
                    sub = result.subject
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("EV/EBITDA", f"{sub.ev_ebitda:.1f}x" if sub.ev_ebitda else "N/A")
                    c2.metric("P/E (TTM)", f"{sub.pe_ratio:.1f}x" if sub.pe_ratio else "N/A")
                    c3.metric("EV/Revenue", f"{sub.ev_revenue:.1f}x" if sub.ev_revenue else "N/A")
                    c4.metric("EBITDA Margin", f"{sub.ebitda_margin:.1%}" if sub.ebitda_margin else "N/A")

                if result.peers:
                    st.markdown("**Peer Comparison**")
                    rows = []
                    for p in result.peers:
                        rows.append({
                            "Ticker": p.ticker,
                            "EV/EBITDA": f"{p.ev_ebitda:.1f}x" if p.ev_ebitda else "N/A",
                            "P/E": f"{p.pe_ratio:.1f}x" if p.pe_ratio else "N/A",
                            "EV/Rev": f"{p.ev_revenue:.1f}x" if p.ev_revenue else "N/A",
                            "EBITDA Mgn": f"{p.ebitda_margin:.1%}" if p.ebitda_margin else "N/A",
                            "Rev Growth": f"{p.revenue_growth:.1%}" if p.revenue_growth else "N/A",
                            "Mkt Cap ($B)": f"${p.market_cap / 1e9:.1f}" if p.market_cap else "N/A",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"Comps table error: {e}")


# ─── ACT13D — Activist Investor Monitor ──────────────────────────────────────

def _render_act13d(args: list[str]):
    ticker = args[0].upper() if args else st.text_input("Ticker:", "TSLA")
    if not ticker:
        return
    st.markdown(f"## ACT13D — Activist Investor Monitor: {ticker}")
    st.caption("Scans EDGAR 13D/13G filings for activist positions. Flags known activists (Elliott, Icahn, etc.)")

    if st.button("Scan Activist Filings", key="act13d_run"):
        with st.spinner(f"Scanning EDGAR 13D/13G filings for {ticker}..."):
            try:
                from sentinel.sds.adapters.activist_adapter import get_activist_summary
                result = run_async(get_activist_summary(ticker=ticker))

                if result.is_under_activist_pressure:
                    st.error(f"ACTIVIST ALERT: {ticker} is under activist pressure")
                else:
                    st.success(f"No active campaigns detected for {ticker}")

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Filings (13D/G)", result.total_filings)
                c2.metric("Active Campaigns", result.active_campaigns)
                c3.metric("Known Activists", result.known_activist_count)

                if result.filings:
                    st.markdown("**Activist Filings**")
                    rows = [{"Filer": f.filer_name, "Type": f.form_type,
                             "Ownership %": f"{f.ownership_pct:.1f}%" if f.ownership_pct else "N/A",
                             "Filed": str(f.filed_date) if f.filed_date else "N/A",
                             "Known Activist": "Yes" if f.is_known_activist else "No",
                             "Activist": f.activist_name or ""}
                            for f in result.filings[:15]]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                if result.campaign_summary:
                    st.markdown("**Campaign Summary**")
                    st.markdown(result.campaign_summary)

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"Activist adapter error: {e}")


# ─── ESRCH — EDGAR Full-Text Search ──────────────────────────────────────────

def _render_esrch(args: list[str]):
    query_default = " ".join(args) if args else ""
    st.markdown("## ESRCH — EDGAR Full-Text Search")
    st.caption("EDGAR EFTS full-text search across all SEC filings. Find disclosures, risk factors, any keyword.")

    query = st.text_input("Search query:", query_default or "material weakness internal controls")
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker_filter = st.text_input("Filter by ticker (optional):", "")
    with col2:
        form_types_input = st.text_input("Form types (comma-sep, optional):", "10-K,10-Q")
    with col3:
        days_back = st.slider("Days back:", 30, 1825, 365)

    if st.button("Search EDGAR", key="esrch_run") and query:
        with st.spinner(f"Searching EDGAR for: {query}..."):
            try:
                from sentinel.sil.edgar_search import search_edgar
                form_types = [f.strip() for f in form_types_input.split(",") if f.strip()] or None
                ticker_val = ticker_filter.upper() if ticker_filter.strip() else None
                result = run_async(search_edgar(query=query, form_types=form_types,
                                                days_back=days_back, ticker=ticker_val))

                st.markdown(f"**{result.total_hits:,} results** | Query: *{result.query}* | "
                            f"Time: {result.search_time_ms:.0f}ms")

                if result.results:
                    rows = [{"Entity": r.entity_name, "Ticker": r.ticker or "N/A",
                             "Form": r.form_type, "Filed": str(r.filed_at)[:10],
                             "Excerpt": r.excerpt[:120] + "..." if r.excerpt and len(r.excerpt) > 120 else r.excerpt or ""}
                            for r in result.results]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("No results found. Try broader search terms.")

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"EDGAR search error: {e}")


# ─── VAR — Portfolio VaR Engine ───────────────────────────────────────────────

def _render_var(args: list[str]):
    st.markdown("## VAR — Portfolio Value-at-Risk Engine")
    st.caption("Historical simulation, parametric (normal), Monte Carlo VaR/CVaR with per-component decomposition")

    col1, col2 = st.columns(2)
    with col1:
        weights_input = st.text_area("Portfolio weights (ticker: weight):",
                                     "AAPL: 0.30\nMSFT: 0.25\nNVDA: 0.20\nSPY: 0.25", height=120)
        confidence = st.slider("Confidence level:", 0.90, 0.99, 0.95, step=0.01)
    with col2:
        method = st.selectbox("VaR method:", ["historical", "parametric", "monte_carlo"])
        horizon_days = st.number_input("Horizon (days):", 1, 252, 1)
        portfolio_value = st.number_input("Portfolio value ($, optional):", 0, 10_000_000, 0)

    if st.button("Compute VaR", key="var_run"):
        try:
            weights: dict[str, float] = {}
            for line in weights_input.strip().splitlines():
                if ":" in line:
                    t, w = line.split(":", 1)
                    weights[t.strip().upper()] = float(w.strip())
        except ValueError as e:
            st.error(f"Invalid weights format: {e}")
            return

        if not weights:
            st.error("Enter at least one ticker: weight pair.")
            return

        with st.spinner(f"Computing {method} VaR ({confidence:.0%} confidence, {horizon_days}d horizon)..."):
            try:
                from sentinel.spr.var_engine import compute_portfolio_var
                result = run_async(compute_portfolio_var(
                    weights=weights, confidence=confidence, horizon_days=horizon_days,
                    method=method, portfolio_value=portfolio_value or None,
                ))

                pvar = result.portfolio_var
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("VaR (%)", f"{pvar.var_pct:.2%}")
                c2.metric("CVaR (%)", f"{pvar.cvar_pct:.2%}")
                c3.metric("VaR ($)", f"${pvar.var_dollar:,.0f}" if pvar.var_dollar else "N/A")
                c4.metric("CVaR ($)", f"${pvar.cvar_dollar:,.0f}" if pvar.cvar_dollar else "N/A")

                st.metric("Diversification Benefit", f"{result.diversification_benefit:.1%}")

                if result.component_vars:
                    st.markdown("**Per-Component VaR**")
                    rows = [{"Ticker": t, "Weight": f"{w:.1%}",
                             "VaR (%)": f"{v.var_pct:.2%}", "CVaR (%)": f"{v.cvar_pct:.2%}"}
                            for (t, w), v in zip(weights.items(), result.component_vars.values())]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                if result.correlation_matrix:
                    st.markdown("**Correlation Matrix**")
                    corr_df = pd.DataFrame(result.correlation_matrix,
                                           index=list(weights.keys()), columns=list(weights.keys()))
                    st.dataframe(corr_df.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
                                 use_container_width=True)

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"VaR engine error: {e}")


# ─── ATTRIB — Brinson Attribution ────────────────────────────────────────────

def _render_attrib(args: list[str]):
    st.markdown("## ATTRIB — Brinson-Hood-Beebower Attribution")
    st.caption("Allocation, selection, interaction effects by GICS sector vs benchmark (SPY/QQQ/IWM/DIA)")

    col1, col2 = st.columns(2)
    with col1:
        holdings_input = st.text_area("Holdings (ticker: weight):",
                                      "AAPL: 0.20\nMSFT: 0.20\nNVDA: 0.15\nJPM: 0.15\nXOM: 0.10\nSPY: 0.20",
                                      height=150)
        benchmark = st.selectbox("Benchmark:", ["SPY", "QQQ", "IWM", "DIA"])
    with col2:
        days_back = st.slider("Analysis window (days):", 21, 365, 90)

    if st.button("Run Attribution", key="attrib_run"):
        try:
            holdings: dict[str, float] = {}
            for line in holdings_input.strip().splitlines():
                if ":" in line:
                    t, w = line.split(":", 1)
                    holdings[t.strip().upper()] = float(w.strip())
        except ValueError as e:
            st.error(f"Invalid holdings format: {e}")
            return

        if not holdings:
            st.error("Enter at least one ticker: weight pair.")
            return

        with st.spinner(f"Running BHB attribution vs {benchmark} ({days_back}d)..."):
            try:
                from sentinel.spr.attribution import compute_attribution
                _end = date.today()
                _start = _end - timedelta(days=days_back)
                result = run_async(compute_attribution(holdings=holdings, benchmark=benchmark,
                                                        start_date=_start, end_date=_end))

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Portfolio Return", f"{result.portfolio_return:.2%}")
                c2.metric("Benchmark Return", f"{result.benchmark_return:.2%}")
                c3.metric("Active Return", f"{result.active_return:.2%}")
                c4.metric("R2", f"{result.r_squared:.2f}" if result.r_squared else "N/A")

                ae = result.total_allocation_effect
                se = result.total_selection_effect
                ie = result.total_interaction_effect
                ac1, ac2, ac3 = st.columns(3)
                ac1.metric("Allocation Effect", f"{ae:.2%}" if ae else "N/A")
                ac2.metric("Selection Effect", f"{se:.2%}" if se else "N/A")
                ac3.metric("Interaction Effect", f"{ie:.2%}" if ie else "N/A")

                if result.sector_attributions:
                    st.markdown("**Sector Attribution**")
                    rows = [{"Sector": s.sector,
                             "Port Wt": f"{s.portfolio_weight:.1%}",
                             "Bench Wt": f"{s.benchmark_weight:.1%}",
                             "Port Ret": f"{s.portfolio_return:.2%}" if s.portfolio_return else "N/A",
                             "Bench Ret": f"{s.benchmark_return:.2%}" if s.benchmark_return else "N/A",
                             "Alloc": f"{s.allocation_effect:.2%}" if s.allocation_effect else "N/A",
                             "Select": f"{s.selection_effect:.2%}" if s.selection_effect else "N/A",
                             "Total": f"{s.total_effect:.2%}" if s.total_effect else "N/A"}
                            for s in result.sector_attributions]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"Attribution error: {e}")


# ─── CONTRA — Controversy Monitor ────────────────────────────────────────────

def _render_contra(args: list[str]):
    ticker = args[0].upper() if args else st.text_input("Ticker:", "META")
    if not ticker:
        return
    st.markdown(f"## CONTRA — Controversy and ESG Risk Monitor: {ticker}")
    st.caption("GDELT news API controversy classification: environmental, social, governance, regulatory, litigation, cyber")

    days_back = st.slider("Days to scan:", 14, 365, 90, key="contra_days")

    if st.button("Scan Controversies", key="contra_run"):
        with st.spinner(f"Scanning GDELT news for {ticker} controversies ({days_back}d)..."):
            try:
                from sentinel.snm.controversy_monitor import get_controversy_profile
                result = run_async(get_controversy_profile(ticker=ticker, days_back=days_back))

                risk_color = {"low": "positive", "medium": "neutral",
                              "high": "negative", "very_high": "negative"}.get(result.risk_level, "neutral")
                st.markdown(f"### Risk Level: <span class='{risk_color}'>"
                            f"{result.risk_level.replace('_', ' ').title()}</span> "
                            f"(score {result.controversy_score:.0f}/100)", unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Signals", result.total_signals)
                c2.metric("High Severity", result.high_severity_count)
                c3.metric("Recent (7d)", result.recent_count)

                if result.category_breakdown:
                    st.markdown("**By Category**")
                    cat_df = pd.DataFrame([{"Category": k.title(), "Count": v}
                                           for k, v in result.category_breakdown.items() if v > 0])
                    if not cat_df.empty:
                        st.bar_chart(cat_df.set_index("Category"))

                if result.signals:
                    st.markdown("**Recent Controversy Signals**")
                    rows = [{"Date": str(s.date)[:10], "Category": s.category.title(),
                             "Severity": s.severity, "Headline": s.headline[:100],
                             "Source": s.source or "N/A"}
                            for s in result.signals[:20]]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"Controversy monitor error: {e}")


# ─── DEX — DEX Analytics Dashboard ───────────────────────────────────────────

def _render_dex(args: list[str]):
    st.markdown("## DEX — Decentralized Exchange Analytics")
    st.caption("Top DEX protocols by 24h/7d volume, market share, chain breakdown via DeFiLlama (free, no API key)")

    top_n = st.slider("Top N DEX protocols:", 5, 30, 15, key="dex_topn")

    if st.button("Load DEX Dashboard", key="dex_load"):
        with st.spinner("Fetching DeFiLlama DEX data..."):
            try:
                from sentinel.snm.defi_analytics import get_dex_dashboard
                dash = run_async(get_dex_dashboard(top_n=top_n))

                c1, c2, c3 = st.columns(3)
                c1.metric("Total 24h Volume", f"${dash.total_volume_24h / 1e9:.2f}B")
                c2.metric("Total 7d Volume", f"${dash.total_volume_7d / 1e9:.2f}B" if dash.total_volume_7d else "N/A")
                c3.metric("Protocols Tracked", dash.total_protocols)

                if dash.protocols:
                    st.markdown("**Top DEX Protocols by Volume**")
                    rows = [{"Protocol": p.name, "Chain": p.chain or "Multi",
                             "24h Vol ($M)": f"${p.volume_24h / 1e6:.1f}" if p.volume_24h else "N/A",
                             "7d Vol ($M)": f"${p.volume_7d / 1e6:.1f}" if p.volume_7d else "N/A",
                             "Market Share": f"{p.market_share:.1%}" if p.market_share else "N/A",
                             "Category": p.category or "DEX"}
                            for p in dash.protocols[:top_n]]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                if dash.chain_breakdown:
                    st.markdown("**Volume by Chain**")
                    chain_df = pd.DataFrame([{"Chain": k, "24h Vol ($M)": v / 1e6}
                                             for k, v in list(dash.chain_breakdown.items())[:10]])
                    if not chain_df.empty:
                        st.bar_chart(chain_df.set_index("Chain"))

                if dash.warnings:
                    for w in dash.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"DEX dashboard error: {e}")


# ─── PORTOPT — Portfolio Optimizer ───────────────────────────────────────────

def _render_portopt(args: list[str]):
    st.markdown("## PORTOPT — Portfolio Optimizer")
    st.caption("Mean-variance (Markowitz), Black-Litterman, Equal-Risk-Contribution. Efficient frontier + per-asset stats.")

    col1, col2 = st.columns(2)
    with col1:
        tickers_input = st.text_input("Tickers (comma-separated):", "AAPL,MSFT,NVDA,JPM,XOM,SPY")
        method = st.selectbox("Method:", ["max_sharpe", "min_variance", "black_litterman", "erc"])
        risk_free = st.number_input("Risk-free rate (annualized):", 0.0, 0.15, 0.05, step=0.005, format="%.3f")
    with col2:
        lookback_days = st.slider("Lookback (days):", 63, 756, 252)
        min_weight = st.number_input("Min weight per asset:", 0.0, 0.5, 0.0, step=0.01)
        max_weight = st.number_input("Max weight per asset:", 0.1, 1.0, 1.0, step=0.05)

    bl_views = ""
    if method == "black_litterman":
        bl_views = st.text_area("BL Views (ticker: expected_return):",
                                "AAPL: 0.15\nMSFT: 0.12", height=80)

    if st.button("Run Optimization", key="portopt_run"):
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if not tickers:
            st.error("Enter at least 2 tickers.")
            return

        views = None
        if method == "black_litterman" and bl_views.strip():
            try:
                views = {k.strip().upper(): float(v.strip())
                         for line in bl_views.splitlines() if ":" in line
                         for k, v in [line.split(":", 1)]}
            except ValueError:
                st.error("Invalid BL views format. Use 'TICKER: 0.15' per line.")
                return

        with st.spinner(f"Optimizing {len(tickers)}-asset portfolio ({method}, {lookback_days}d)..."):
            try:
                from sentinel.spr.optimizer import optimize_portfolio
                result = run_async(optimize_portfolio(
                    tickers=tickers, method=method, risk_free=risk_free,
                    lookback_days=lookback_days, min_weight=min_weight,
                    max_weight=max_weight, views=views,
                ))

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Portfolio Return", f"{result.portfolio_return:.2%}")
                c2.metric("Portfolio Vol", f"{result.portfolio_volatility:.2%}")
                c3.metric("Sharpe Ratio", f"{result.portfolio_sharpe:.2f}")
                c4.metric("Diversification", f"{result.diversification_ratio:.2f}")

                st.markdown("**Optimal Weights**")
                wt_rows = [{"Ticker": t, "Weight": f"{w:.1%}",
                            "Return": f"{a.expected_return:.2%}",
                            "Vol": f"{a.volatility:.2%}",
                            "Sharpe": f"{a.sharpe:.2f}" if a.sharpe else "N/A"}
                           for t, w, a in zip(result.weights.keys(),
                                               result.weights.values(), result.assets)]
                st.dataframe(pd.DataFrame(wt_rows), use_container_width=True)

                if result.efficient_frontier:
                    st.markdown("**Efficient Frontier**")
                    ef_df = pd.DataFrame([
                        {"Volatility": p.volatility, "Return": p.expected_return, "Sharpe": p.sharpe}
                        for p in result.efficient_frontier
                    ])
                    st.line_chart(ef_df.set_index("Volatility")["Return"])

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"Optimizer error: {e}")


# ─── FISCRN — Fixed Income Screener ──────────────────────────────────────────

def _render_fiscrn(args: list[str]):
    query_default = " ".join(args) if args else ""
    st.markdown("## FISCRN — Fixed Income Screener")
    st.caption("Screen bond ETF proxies + FINRA TRACE by yield, duration, credit quality, sector. FRED spreads for context.")

    query = st.text_input("Natural language (optional):", query_default or "investment grade bonds yield > 5%")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        min_yield = st.number_input("Min yield (%):", 0.0, 20.0, 0.0, step=0.25)
        max_yield = st.number_input("Max yield (%):", 0.0, 20.0, 0.0, step=0.25)
    with col2:
        min_dur = st.number_input("Min duration (yrs):", 0.0, 30.0, 0.0, step=0.5)
        max_dur = st.number_input("Max duration (yrs):", 0.0, 30.0, 0.0, step=0.5)
    with col3:
        credit_opts = st.multiselect("Credit quality:", ["IG", "HY", "AAA", "AA", "A", "BBB", "BB", "B", "CCC"])
    with col4:
        limit = st.slider("Max results:", 5, 50, 25)

    if st.button("Screen Bonds", key="fiscrn_run"):
        with st.spinner("Screening bonds..."):
            try:
                from sentinel.sfe.bond_screener import BondScreenCriteria, screen_bonds
                criteria = BondScreenCriteria(
                    min_yield=min_yield or None,
                    max_yield=max_yield or None,
                    min_duration=min_dur or None,
                    max_duration=max_dur or None,
                    credit_quality=credit_opts or None,
                )
                result = run_async(screen_bonds(criteria=criteria, query=query or None, limit=limit))

                if result.market_context:
                    st.markdown("**Market Context**")
                    ctx = result.market_context
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("10Y Treasury", f"{ctx.treasury_10y:.2f}%" if ctx.treasury_10y else "N/A")
                    mc2.metric("IG Spread", f"{ctx.oas_ig_bps:.0f}bps" if ctx.oas_ig_bps else "N/A")
                    mc3.metric("HY Spread", f"{ctx.oas_hy_bps:.0f}bps" if ctx.oas_hy_bps else "N/A")

                st.markdown(f"**{result.total_matched} results**")
                if result.results:
                    rows = [{"Symbol": r.symbol, "Name": r.name,
                             "Yield": f"{r.yield_pct:.2f}%",
                             "Duration": f"{r.duration_years:.1f}y",
                             "Spread": f"{r.spread_vs_treasury:.0f}bps" if r.spread_vs_treasury else "N/A",
                             "Quality": r.credit_quality,
                             "Category": r.category,
                             "AUM ($B)": f"{r.aum_billions:.1f}" if r.aum_billions else "N/A",
                             "YTD": f"{r.ytd_return_pct:.1f}%" if r.ytd_return_pct else "N/A"}
                            for r in result.results]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"Bond screener error: {e}")


# ─── CEVT — On-Chain Event Monitor ───────────────────────────────────────────

def _render_cevt(args: list[str]):
    ticker_default = args[0].upper() if args else ""
    st.markdown("## CEVT — On-Chain Event Monitor")
    st.caption("Whale transactions, large TVL changes, protocol events via Etherscan (free) + DeFiLlama.")

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("ERC-20 token (e.g. UNI, AAVE, LINK):", ticker_default)
        protocol = st.text_input("DeFiLlama protocol slug (e.g. uniswap, aave):", "")
    with col2:
        days_back = st.slider("Days back:", 1, 30, 7, key="cevt_days")
        min_value = st.number_input("Min transaction value ($):", 0, 10_000_000, 500_000, step=100_000)

    if st.button("Scan On-Chain Events", key="cevt_run"):
        with st.spinner("Scanning on-chain events..."):
            try:
                from sentinel.snm.onchain_events import get_onchain_events
                result = run_async(get_onchain_events(
                    ticker=ticker.upper() if ticker else None,
                    protocol=protocol.lower() if protocol else None,
                    days_back=days_back, min_value_usd=float(min_value),
                ))

                alert_color = "negative" if result.alert_score > 60 else (
                    "neutral" if result.alert_score > 30 else "positive")
                st.markdown(f"### Alert Score: <span class='{alert_color}'>"
                            f"{result.alert_score:.0f}/100</span>", unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Events", result.total_events)
                c2.metric("High Severity", result.high_severity_count)
                c3.metric("Total Value", f"${result.total_value_usd / 1e6:.1f}M" if result.total_value_usd else "N/A")

                if result.dominant_event_type:
                    st.markdown(f"**Dominant event type:** {result.dominant_event_type.replace('_', ' ').title()}")

                if result.events:
                    st.markdown("**Recent Events**")
                    rows = [{"Date": str(e.timestamp)[:16] if e.timestamp else "N/A",
                             "Type": e.event_type.replace("_", " ").title(),
                             "Chain": e.chain,
                             "Value ($M)": f"${e.value_usd / 1e6:.2f}" if e.value_usd else "N/A",
                             "Severity": e.severity.upper(),
                             "Description": e.description[:80]}
                            for e in result.events[:20]]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"On-chain events error: {e}")


# ─── ETFPROF — ETF Profile ───────────────────────────────────────────────────

def _render_etfprof(args: list[str]):
    ticker = args[0].upper() if args else st.text_input("ETF Ticker:", "SPY").upper()
    if not ticker:
        return
    st.markdown(f"## ETFPROF — ETF Profile: {ticker}")
    st.caption("AUM, expense ratio, NAV premium/discount, top holdings, sector weights, factor exposure, flows")

    if st.button("Load Profile", key="etfprof_run"):
        with st.spinner(f"Loading ETF data for {ticker}..."):
            try:
                from sentinel.sfe.etf_analytics import get_etf_profile
                result = run_async(get_etf_profile(ticker=ticker))

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("AUM", f"${result.aum_billions:.1f}B" if result.aum_billions else "N/A")
                c2.metric("Expense Ratio", f"{result.expense_ratio:.3%}" if result.expense_ratio else "N/A")
                c3.metric("Holdings", str(result.holdings_count))
                pd_str = f"{result.premium_discount_pct:+.2f}%" if result.premium_discount_pct is not None else "N/A"
                c4.metric("NAV Premium/Discount", pd_str)

                p1, p2, p3 = st.columns(3)
                p1.metric("YTD Return", f"{result.ytd_return:+.2f}%" if result.ytd_return is not None else "N/A")
                p2.metric("1Y Return", f"{result.one_year_return:+.2f}%" if result.one_year_return is not None else "N/A")
                p3.metric("Sharpe (1Y)", f"{result.sharpe_ratio:.2f}" if result.sharpe_ratio is not None else "N/A")

                col_top, col_sec = st.columns(2)
                with col_top:
                    st.markdown("**Top Holdings**")
                    if result.top_holdings:
                        h_rows = [{"Symbol": h.symbol, "Name": h.name or "", "Weight": f"{h.weight:.2%}"}
                                  for h in result.top_holdings]
                        st.dataframe(pd.DataFrame(h_rows), use_container_width=True)

                with col_sec:
                    st.markdown("**Sector Weights**")
                    if result.sector_weights:
                        s_rows = [{"Sector": k, "Weight": f"{v:.2%}"}
                                  for k, v in sorted(result.sector_weights.items(), key=lambda x: -x[1])]
                        st.dataframe(pd.DataFrame(s_rows), use_container_width=True)

                st.markdown("**Factor Exposure**")
                fe = result.factor_exposure
                f1, f2, f3, f4, f5, f6 = st.columns(6)
                f1.metric("Beta", f"{fe.market_beta:.2f}")
                f2.metric("Size", f"{fe.size_factor:+.2f}")
                f3.metric("Value", f"{fe.value_factor:+.2f}")
                f4.metric("Momentum", f"{fe.momentum_factor:+.2f}")
                f5.metric("Quality", f"{fe.quality_factor:+.2f}")
                f6.metric("Low-Vol", f"{fe.volatility_factor:+.2f}")

                if result.estimated_flow_30d is not None:
                    flow_str = f"${result.estimated_flow_30d:+,.0f}M"
                    flow_label = "📈 Inflow" if result.estimated_flow_30d > 0 else "📉 Outflow"
                    st.metric(f"Est. 30d Flow", f"{flow_label} {flow_str}")

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"ETF profile error: {e}")


# ─── ETFCMP — ETF Comparison ──────────────────────────────────────────────────

def _render_etfcmp(args: list[str]):
    default = " ".join(args) if args else "SPY QQQ IWM DIA"
    st.markdown("## ETFCMP — ETF Comparison")
    st.caption("Side-by-side: AUM, expense ratio, returns, Sharpe, top sector")

    tickers_raw = st.text_input("ETF tickers (space or comma separated):", default)
    tickers = [t.strip().upper() for t in tickers_raw.replace(",", " ").split() if t.strip()]

    if st.button("Compare", key="etfcmp_run") and len(tickers) >= 2:
        with st.spinner("Fetching ETF data..."):
            try:
                import json
                from sentinel.sfe.etf_analytics import compare_etfs
                result = run_async(compare_etfs(tickers=tickers))

                rows = [{
                    "Ticker": r.ticker, "Name": r.name or "",
                    "AUM ($B)": f"{r.aum_billions:.1f}" if r.aum_billions else "N/A",
                    "ER (%)": f"{r.expense_ratio:.3%}" if r.expense_ratio else "N/A",
                    "YTD": f"{r.ytd_return:+.2f}%" if r.ytd_return is not None else "N/A",
                    "1Y": f"{r.one_year_return:+.2f}%" if r.one_year_return is not None else "N/A",
                    "Sharpe": f"{r.sharpe_ratio:.2f}" if r.sharpe_ratio is not None else "N/A",
                    "Holdings": r.holdings_count,
                    "Top Sector": r.top_sector or "N/A",
                } for r in result.rows]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                h1, h2, h3 = st.columns(3)
                h1.metric("Best YTD", result.best_ytd or "N/A")
                h2.metric("Best Sharpe", result.best_sharpe or "N/A")
                h3.metric("Lowest Cost", result.lowest_cost or "N/A")

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"ETF comparison error: {e}")


# ─── COMMOD — Commodity Dashboard ────────────────────────────────────────────

def _render_commod(args: list[str]):
    st.markdown("## COMMOD — Commodity Dashboard")
    st.caption("Energy, metals, agriculture prices via FRED + yfinance. Futures structure. Inflation regime detection.")

    if st.button("Load Dashboard", key="commod_run"):
        with st.spinner("Fetching commodity prices..."):
            try:
                from sentinel.sma.commodity_analytics import get_commodity_dashboard
                result = run_async(get_commodity_dashboard())

                regime_color = {"inflationary": "🔴", "deflationary": "🟢", "neutral": "🟡"}.get(result.commodity_regime, "⚪")
                r1, r2 = st.columns(2)
                r1.metric("Commodity Regime", f"{regime_color} {result.commodity_regime.upper()}")
                r2.metric("PPI-CPI Gap", f"{result.cpi_vs_ppi_gap:+.2f}%" if result.cpi_vs_ppi_gap else "N/A",
                          help="Positive = producer squeeze (PPI rising faster than CPI)")

                for sector_data in result.sectors:
                    st.markdown(f"**{sector_data.sector}**")
                    rows = [{
                        "Commodity": c.name,
                        "Price": f"{c.price:.2f} {c.price_unit}" if c.price else "N/A",
                        "1d Δ": f"{c.change_1d:+.2f}%" if c.change_1d is not None else "N/A",
                        "1m Δ": f"{c.change_1m:+.2f}%" if c.change_1m is not None else "N/A",
                        "YTD": f"{c.change_ytd:+.2f}%" if c.change_ytd is not None else "N/A",
                        "Source": c.data_source,
                    } for c in sector_data.commodities]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                if result.futures_spreads:
                    st.markdown("**Futures Term Structure**")
                    spread_rows = [{
                        "Contract": s.name,
                        "Front": f"{s.front_month:.2f}" if s.front_month else "N/A",
                        "Next": f"{s.next_month:.2f}" if s.next_month else "N/A",
                        "Spread": f"{s.spread:+.2f}" if s.spread else "N/A",
                        "Structure": s.structure.upper(),
                    } for s in result.futures_spreads]
                    st.dataframe(pd.DataFrame(spread_rows), use_container_width=True)

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"Commodity dashboard error: {e}")


# ─── IFRS — International IFRS Fundamentals ───────────────────────────────────

def _render_ifrs(args: list[str]):
    st.markdown("## IFRS — International Fundamentals (20-F XBRL)")
    st.caption("IFRS financial statements for non-US ADRs and foreign private issuers via SEC 20-F filings.")

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Ticker:", value=args[0] if args else "", placeholder="e.g. ASML, BABA, NVO, SAP")
    with col2:
        periods = st.slider("Annual periods:", 1, 8, 4)

    if st.button("Load IFRS Financials", key="ifrs_run") and ticker:
        with st.spinner(f"Fetching XBRL data for {ticker.upper()}..."):
            try:
                from sentinel.sfe.ifrs_fundamentals import get_ifrs_fundamentals
                result = run_async(get_ifrs_fundamentals(ticker=ticker.upper(), periods=periods))

                st.markdown(f"**{result.entity_name}** | CIK: {result.cik} | Standard: {result.filing_standard} | Latest: {result.latest_period}")
                cols = st.columns(4)
                r = result.ratios
                cols[0].metric("Rev Growth", f"{r.revenue_growth_pct:+.1f}%" if r.revenue_growth_pct is not None else "N/A")
                cols[1].metric("Gross Margin", f"{r.gross_margin_pct:.1f}%" if r.gross_margin_pct is not None else "N/A")
                cols[2].metric("Net Margin", f"{r.net_margin_pct:.1f}%" if r.net_margin_pct is not None else "N/A")
                cols[3].metric("ROE", f"{r.roe_pct:.1f}%" if r.roe_pct is not None else "N/A")

                for section_name, items in [
                    ("Income Statement", result.statements.income_statement),
                    ("Balance Sheet", result.statements.balance_sheet),
                    ("Cash Flow", result.statements.cash_flow),
                ]:
                    if items:
                        st.markdown(f"**{section_name}**")
                        rows = []
                        for item in items:
                            row = {"Line Item": item.label, "Unit": item.unit}
                            for v in item.values[:periods]:
                                row[v.get("period_end", "—")] = _fmt_large(v.get("value"))
                            rows.append(row)
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"IFRS error: {e}")


# ─── ECONFC — Economic Forecasting ───────────────────────────────────────────

def _render_econfc(args: list[str]):
    st.markdown("## ECONFC — Economic Forecasting")
    st.caption("AR/VAR macro forecasting on FRED series. Unemployment, CPI, Fed Funds, yield spread, industrial production.")

    col1, col2, col3 = st.columns(3)
    with col1:
        horizon = st.slider("Forecast horizon (months):", 1, 12, 6)
        ar_order = st.slider("AR/VAR lag order:", 1, 6, 3)
    with col2:
        include_var = st.checkbox("Include VAR (multivariate)", value=True)
        series_input = st.text_input("FRED series (comma-sep, blank=default):", placeholder="UNRATE,CPIAUCSL,FEDFUNDS")
    with col3:
        st.caption("Default series: UNRATE, CPIAUCSL, FEDFUNDS, T10Y2Y, INDPRO")

    if st.button("Run Forecast", key="econfc_run"):
        with st.spinner("Fetching FRED data and fitting models..."):
            try:
                from sentinel.sma.econ_forecasting import get_econ_forecast
                series_ids = [s.strip() for s in series_input.split(",") if s.strip()] or None
                result = run_async(get_econ_forecast(
                    series_ids=series_ids, horizon=horizon,
                    ar_order=ar_order, include_var=include_var,
                ))

                nc = result.nowcast
                nc_color = "🟢" if nc.signal == "expansion" else "🔴" if nc.signal == "contraction" else "🟡"
                st.metric("Nowcast Index", f"{nc_color} {nc.score:+.2f}", help="−10=deep contraction, +10=strong expansion")

                for fc in result.forecasts:
                    with st.expander(f"{fc.series_name} ({fc.series_id}) — {fc.model_order}", expanded=False):
                        st.caption(f"Last actual: {fc.last_actual:.3f} on {fc.last_actual_date} | RMSE: {fc.in_sample_rmse:.4f}" if fc.in_sample_rmse else f"Last actual: {fc.last_actual:.3f} on {fc.last_actual_date}")
                        rows = [{"Period": p.period, "Forecast": f"{p.value:.4f}",
                                 "Lower": f"{p.confidence_lower:.4f}" if p.confidence_lower else "—",
                                 "Upper": f"{p.confidence_upper:.4f}" if p.confidence_upper else "—"}
                                for p in fc.forecasts]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"Economic forecasting error: {e}")


# ─── LIQ — Liquidity Analytics ───────────────────────────────────────────────

def _render_liq(args: list[str]):
    st.markdown("## LIQ — Liquidity Analytics")
    st.caption("Amihud illiquidity, Roll spread, Corwin-Schultz bid-ask proxy, Kyle's lambda, ADV, turnover. Market microstructure.")

    col1, col2 = st.columns(2)
    with col1:
        ticker_input = st.text_input("Ticker(s) (comma-sep for portfolio):", value=args[0] if args else "", placeholder="e.g. AAPL or AAPL,MSFT,GME")
    with col2:
        period_days = st.slider("Analysis window (trading days):", 20, 252, 63)

    if st.button("Compute Liquidity", key="liq_run") and ticker_input:
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        with st.spinner(f"Computing liquidity metrics for {', '.join(tickers)}..."):
            try:
                from sentinel.spr.liquidity_analytics import get_liquidity_metrics, get_portfolio_liquidity

                if len(tickers) == 1:
                    result = run_async(get_liquidity_metrics(ticker=tickers[0], period_days=period_days))
                    cols = st.columns(4)
                    cols[0].metric("Liquidity Score", f"{result.liquidity_score:.1f}/10", help=result.liquidity_label.upper())
                    cols[1].metric("ADV (20d)", _fmt_large(result.adv_20d_usd) if result.adv_20d_usd else "N/A")
                    cols[2].metric("Amihud (ann.)", f"{result.amihud_annualized:.4f}" if result.amihud_annualized else "N/A")
                    cols[3].metric("B/A Spread Est.", f"{result.corwin_schultz_spread_pct:.3f}%" if result.corwin_schultz_spread_pct else "N/A")
                    rows = [{
                        "Metric": "Roll Spread (%)", "Value": f"{result.roll_spread_pct:.4f}%" if result.roll_spread_pct else "N/A",
                    }, {
                        "Metric": "Kyle's Lambda", "Value": f"{result.kyle_lambda:.6f}" if result.kyle_lambda else "N/A",
                    }, {
                        "Metric": "Turnover (ann.)", "Value": f"{result.turnover_ratio_annualized:.2f}%" if result.turnover_ratio_annualized else "N/A",
                    }, {
                        "Metric": "Volume Spike Days", "Value": str(result.volume_spike_days),
                    }, {
                        "Metric": "ADV 60d", "Value": _fmt_large(result.adv_60d_usd) if result.adv_60d_usd else "N/A",
                    }]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    result = run_async(get_portfolio_liquidity(tickers=tickers, period_days=period_days))
                    st.metric("Portfolio Liquidity Score", f"{result.portfolio_liquidity_score:.1f}/10")
                    if result.portfolio_adv_usd:
                        st.metric("Portfolio ADV", _fmt_large(result.portfolio_adv_usd))
                    if result.days_to_liquidate_90pct:
                        st.metric("Days to Liquidate 90% ($10M)", f"{result.days_to_liquidate_90pct:.1f}")
                    for m in result.metrics:
                        st.markdown(f"**{m.ticker}** — Score: {m.liquidity_score:.1f}/10 ({m.liquidity_label}) | ADV: {_fmt_large(m.adv_20d_usd) if m.adv_20d_usd else 'N/A'} | B/A: {m.corwin_schultz_spread_pct:.3f}%" if m.corwin_schultz_spread_pct else f"**{m.ticker}** — Score: {m.liquidity_score:.1f}/10")

                for w in (result.warnings if hasattr(result, 'warnings') else []):
                    st.warning(w)
            except Exception as e:
                st.error(f"Liquidity analytics error: {e}")


# ─── ALRT — Price Alerting System ────────────────────────────────────────────

def _render_alrt(args: list[str]):
    st.markdown("## ALRT — Price Alerts")
    st.caption("Persistent price/RSI/MA/volume alerts. Stored in ~/.sentinel/alerts.json.")

    tab_check, tab_create, tab_manage = st.tabs(["Check Alerts", "Create Alert", "Manage Alerts"])

    with tab_check:
        ticker_filter = st.text_input("Filter by ticker (blank=all):", key="alrt_check_ticker", placeholder="e.g. AAPL")
        if st.button("Check Now", key="alrt_check_run"):
            with st.spinner("Polling prices..."):
                try:
                    from sentinel.sil.price_alerts import check_alerts
                    tickers = [ticker_filter.upper()] if ticker_filter else None
                    summary = run_async(check_alerts(tickers=tickers))
                    c1, c2 = st.columns(2)
                    c1.metric("Active Alerts", summary.total_active)
                    c2.metric("Triggered Today", summary.total_triggered_today)
                    for cr in summary.triggered:
                        st.success(f"**{cr.ticker}** @ {cr.current_price:.2f} — {len(cr.triggered_alerts)} alert(s) triggered")
                        for a in cr.triggered_alerts:
                            st.write(f"  • {a.alert_type} | threshold: {a.threshold} | triggered: {a.triggered_price:.2f} | note: {a.note or '—'}")
                except Exception as e:
                    st.error(f"Alert check error: {e}")

    with tab_create:
        c1, c2 = st.columns(2)
        with c1:
            new_ticker = st.text_input("Ticker:", key="alrt_new_ticker", placeholder="AAPL")
            alert_type = st.selectbox("Alert type:", [
                "price_above", "price_below", "pct_change_up", "pct_change_down",
                "volume_spike", "rsi_overbought", "rsi_oversold", "ma_crossover", "ma_crossunder",
            ])
            threshold = st.number_input("Threshold (price / % / RSI level):", value=0.0, format="%.4f")
        with c2:
            note = st.text_input("Note (optional):", placeholder="Breakout above resistance")
            extra_params = st.text_input("Extra params JSON (optional):", placeholder='{"multiplier": 2.5}')
        if st.button("Create Alert", key="alrt_create_run") and new_ticker:
            try:
                import json as _json
                from sentinel.sil.price_alerts import create_alert
                params = _json.loads(extra_params) if extra_params else None
                alert = run_async(create_alert(
                    ticker=new_ticker.upper(), alert_type=alert_type,
                    threshold=threshold or None, note=note, params=params,
                ))
                st.success(f"Alert created: {alert.id} | {alert.ticker} {alert.alert_type} @ {alert.threshold}")
            except Exception as e:
                st.error(f"Create alert error: {e}")

    with tab_manage:
        mg_ticker = st.text_input("Filter by ticker:", key="alrt_mg_ticker", placeholder="blank=all")
        show_all = st.checkbox("Show triggered too", value=False)
        if st.button("Load Alerts", key="alrt_mg_run"):
            try:
                from sentinel.sil.price_alerts import list_alerts, delete_alert
                alerts = run_async(list_alerts(ticker=mg_ticker.upper() if mg_ticker else None, active_only=not show_all))
                if not alerts:
                    st.info("No alerts found.")
                else:
                    rows = [{"ID": a.id[:8], "Ticker": a.ticker, "Type": a.alert_type,
                             "Threshold": a.threshold, "Active": a.active, "Note": a.note,
                             "Created": a.created_at[:10]} for a in alerts]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                del_id = st.text_input("Delete alert ID (8-char prefix or full UUID):", key="alrt_del_id")
                if st.button("Delete", key="alrt_del_run") and del_id:
                    matched = [a for a in alerts if a.id.startswith(del_id)]
                    if matched:
                        deleted = run_async(delete_alert(matched[0].id))
                        st.success(f"Deleted: {matched[0].id}") if deleted else st.error("Delete failed")
                    else:
                        st.warning("No matching alert ID found")
            except Exception as e:
                st.error(f"Manage alerts error: {e}")


# ─── FXDASH — FX Analytics Dashboard ────────────────────────────────────────

def _render_fxdash(args: list[str]):
    st.markdown("## FXDASH — FX Analytics Dashboard")
    st.caption("Forward curves, realized vol, carry signal, and momentum for major USD pairs. ECB Frankfurter rates + FRED differentials.")

    tab_pair, tab_dash = st.tabs(["Single Pair", "Dashboard"])

    with tab_pair:
        c1, c2 = st.columns(2)
        with c1:
            pair = st.text_input("Pair:", value=args[0].upper() if args else "EURUSD", placeholder="e.g. EURUSD, GBPUSD, USDJPY")
            history_days = st.slider("History (days):", 30, 252, 90)
        if st.button("Analyze Pair", key="fx_pair_run") and pair:
            with st.spinner(f"Fetching {pair.upper()}..."):
                try:
                    from sentinel.sfe.fx_analytics import get_fx_pair
                    result = run_async(get_fx_pair(pair=pair.upper(), history_days=history_days))
                    cols = st.columns(4)
                    cols[0].metric("Spot", f"{result.spot:.4f}", delta=f"{result.change_1d:+.2f}%" if result.change_1d else None)
                    cols[1].metric("30d Vol", f"{result.vol_surface.realized_vol_30d:.1f}%" if result.vol_surface.realized_vol_30d else "N/A")
                    cols[2].metric("Carry", ["🔴 Negative", "⚪ Neutral", "🟢 Positive"][result.carry_score + 1])
                    cols[3].metric("Momentum Z", f"{result.momentum_zscore:+.2f}" if result.momentum_zscore else "N/A")
                    if result.forward_curve:
                        st.markdown("**Forward Curve**")
                        rows = [{"Tenor": f.tenor, "Forward Rate": f"{f.forward_rate:.4f}",
                                 "Fwd Points": f"{f.forward_points:+.1f}", "Rate Diff (ann.)": f"{f.implied_yield_diff:+.2f}%"} for f in result.forward_curve]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    for w in result.warnings:
                        st.warning(w)
                except Exception as e:
                    st.error(f"FX pair error: {e}")

    with tab_dash:
        pairs_input = st.text_input("Pairs (comma-sep, blank=default 8):", placeholder="EURUSD, GBPUSD, USDJPY")
        if st.button("Load FX Dashboard", key="fx_dash_run"):
            with st.spinner("Fetching all FX pairs..."):
                try:
                    from sentinel.sfe.fx_analytics import get_fx_dashboard
                    pairs = [p.strip().upper() for p in pairs_input.split(",") if p.strip()] or None
                    result = run_async(get_fx_dashboard(pairs=pairs))
                    trend_icon = "📈" if result.usd_trend == "strengthening" else "📉" if result.usd_trend == "weakening" else "➡️"
                    st.metric("USD Trend", f"{trend_icon} {result.usd_trend.upper()}", help="DXY proxy from equal-weighted basket")
                    rows = [{"Pair": p.pair, "Spot": f"{p.spot:.4f}",
                             "1d": f"{p.change_1d:+.2f}%" if p.change_1d else "—",
                             "1m": f"{p.change_1m:+.2f}%" if p.change_1m else "—",
                             "Vol 30d": f"{p.vol_surface.realized_vol_30d:.1f}%" if p.vol_surface.realized_vol_30d else "—",
                             "Carry": ["−", "0", "+"][p.carry_score + 1],
                             "Mom Z": f"{p.momentum_zscore:+.2f}" if p.momentum_zscore else "—"} for p in result.pairs]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    for w in result.warnings:
                        st.warning(w)
                except Exception as e:
                    st.error(f"FX dashboard error: {e}")


# ─── FORMD — Private Market / Form D Intelligence ────────────────────────────

def _render_formd(args: list[str]):
    st.markdown("## FORMD — Private Market Intelligence (Form D)")
    st.caption("SEC Regulation D filings: VC raises, hedge fund launches, PE deals, private placements. Free PitchBook equivalent.")

    tab_co, tab_screen = st.tabs(["Company Lookup", "Screen Market"])

    with tab_co:
        company = st.text_input("Company name:", value=" ".join(args) if args else "", placeholder="e.g. OpenAI, Anthropic, Stripe")
        limit = st.slider("Max filings:", 1, 10, 5)
        if st.button("Look Up", key="formd_co_run") and company:
            with st.spinner(f"Searching EDGAR Form D for {company}..."):
                try:
                    from sentinel.sfe.form_d import get_company_form_d
                    filings = run_async(get_company_form_d(company_name=company, limit=limit))
                    if not filings:
                        st.info("No Form D filings found.")
                    else:
                        rows = [{"Date": f.file_date, "Company": f.company_name,
                                 "Type": f.offering_type, "Fund": f.fund_type or "—",
                                 "Raised": _fmt_large(f.amount_sold) if f.amount_sold else "—",
                                 "Total": _fmt_large(f.total_offering_amount) if f.total_offering_amount else "—",
                                 "State": f.state or "—", "Exemption": f.exemption_type or "—"} for f in filings]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                except Exception as e:
                    st.error(f"Form D lookup error: {e}")

    with tab_screen:
        c1, c2, c3 = st.columns(3)
        with c1:
            query = st.text_input("Search term:", placeholder="keyword or blank for all")
            days_back = st.slider("Days back:", 7, 90, 30)
        with c2:
            state = st.text_input("State (2-letter):", placeholder="CA")
            min_amount = st.number_input("Min raise ($M, 0=any):", 0.0, 10000.0, 0.0)
        with c3:
            fund_type = st.selectbox("Fund type:", ["All", "Hedge Fund", "Venture Capital Fund", "Private Equity Fund"])
            screen_limit = st.slider("Max results:", 10, 50, 25)
        if st.button("Screen", key="formd_screen_run"):
            with st.spinner("Scanning recent Form D filings..."):
                try:
                    from sentinel.sfe.form_d import screen_private_market
                    result = run_async(screen_private_market(
                        query=query or None, state=state.upper() if state else None,
                        min_amount_mm=min_amount if min_amount > 0 else None,
                        fund_type=fund_type if fund_type != "All" else None,
                        days_back=days_back, limit=screen_limit,
                    ))
                    st.markdown(f"**{result.total_found} filings** | Total raised: {_fmt_large(result.total_capital_raised) if result.total_capital_raised else 'N/A'}")
                    rows = [{"Date": f.file_date, "Company": f.company_name, "Type": f.offering_type,
                             "Raised": _fmt_large(f.amount_sold) if f.amount_sold else "—",
                             "State": f.state or "—"} for f in result.filings]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    c1, c2 = st.columns(2)
                    if result.by_state:
                        c1.markdown("**By State**")
                        c1.dataframe(pd.DataFrame(list(result.by_state.items()), columns=["State", "Count"]).sort_values("Count", ascending=False), use_container_width=True)
                    if result.by_offering_type:
                        c2.markdown("**By Type**")
                        c2.dataframe(pd.DataFrame(list(result.by_offering_type.items()), columns=["Type", "Count"]), use_container_width=True)
                except Exception as e:
                    st.error(f"Form D screen error: {e}")


# ─── SCEN — Macro Scenario Analysis ──────────────────────────────────────────

def _render_scen(args: list[str]):
    st.markdown("## SCEN — Macro Scenario Analysis")
    st.caption("Apply macro shocks to a portfolio via factor betas. Templates: 2008 crisis, COVID, rate hike, stagflation, and more.")

    tab_single, tab_multi = st.tabs(["Single Scenario", "All Scenarios"])

    TEMPLATES = ["2008_crisis", "covid_crash", "rate_hike_200bps", "soft_landing", "stagflation", "china_taiwan", "usd_crash"]

    with tab_single:
        c1, c2 = st.columns(2)
        with c1:
            tickers_input = st.text_input("Portfolio tickers (comma-sep):", placeholder="AAPL, TLT, GLD, SPY")
            portfolio_value = st.number_input("Portfolio value ($):", 10000.0, 1e9, 1_000_000.0, step=50000.0)
            scenario_choice = st.selectbox("Template:", ["Custom"] + TEMPLATES)
        with c2:
            eq_shock = st.slider("Equity shock (%):", -60.0, 60.0, 0.0, step=1.0) if scenario_choice == "Custom" else 0.0
            rate_shock = st.slider("Rate shock (bps):", -300.0, 300.0, 0.0, step=10.0) if scenario_choice == "Custom" else 0.0
            usd_shock = st.slider("USD shock (%):", -20.0, 20.0, 0.0, step=1.0) if scenario_choice == "Custom" else 0.0
            oil_shock = st.slider("Oil shock (%):", -70.0, 70.0, 0.0, step=5.0) if scenario_choice == "Custom" else 0.0

        if st.button("Run Scenario", key="scen_single_run") and tickers_input:
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            with st.spinner(f"Computing scenario impact on {len(tickers)} assets..."):
                try:
                    from sentinel.spr.scenario_analysis import MacroShock, run_scenario
                    shock = None if scenario_choice != "Custom" else MacroShock(
                        equity_shock_pct=eq_shock, rate_shock_bps=rate_shock,
                        usd_shock_pct=usd_shock, oil_shock_pct=oil_shock, scenario_name="Custom",
                    )
                    result = run_async(run_scenario(
                        tickers=tickers, shock=shock,
                        scenario_name=None if scenario_choice == "Custom" else scenario_choice,
                        portfolio_value=portfolio_value,
                    ))
                    pnl_color = "🟢" if result.total_pnl_usd >= 0 else "🔴"
                    cols = st.columns(3)
                    cols[0].metric("Portfolio P&L", f"{pnl_color} {_fmt_large(abs(result.total_pnl_usd))}", delta=f"{result.total_return_pct:+.2f}%")
                    cols[1].metric("Best Asset", result.best_asset)
                    cols[2].metric("Worst Asset", result.worst_asset)
                    rows = [{"Ticker": a.ticker, "Weight": f"{a.weight*100:.1f}%",
                             "Est. Return": f"{a.estimated_return_pct:+.2f}%",
                             "P&L": _fmt_large(a.pnl_usd),
                             "Driven by": a.dominant_factor} for a in result.asset_results]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    for w in result.warnings:
                        st.warning(w)
                except Exception as e:
                    st.error(f"Scenario analysis error: {e}")

    with tab_multi:
        multi_tickers = st.text_input("Portfolio tickers:", placeholder="AAPL, TLT, GLD, SPY", key="scen_multi_tickers")
        multi_value = st.number_input("Portfolio value ($):", 10000.0, 1e9, 1_000_000.0, step=50000.0, key="scen_multi_value")
        if st.button("Run All Scenarios", key="scen_multi_run") and multi_tickers:
            tickers = [t.strip().upper() for t in multi_tickers.split(",") if t.strip()]
            with st.spinner(f"Running {len(TEMPLATES)} scenarios..."):
                try:
                    from sentinel.spr.scenario_analysis import run_multi_scenario
                    result = run_async(run_multi_scenario(tickers=tickers, portfolio_value=multi_value))
                    st.markdown(f"**Most resilient:** {result.most_resilient_scenario} | **Worst:** {result.worst_scenario}")
                    rows = [{"Scenario": s.scenario.scenario_name,
                             "Return": f"{s.total_return_pct:+.2f}%",
                             "P&L": _fmt_large(s.total_pnl_usd),
                             "Best": s.best_asset, "Worst": s.worst_asset} for s in result.scenarios]
                    st.dataframe(pd.DataFrame(rows).sort_values("Return"), use_container_width=True)
                except Exception as e:
                    st.error(f"Multi-scenario error: {e}")


# ─── DVDS — Dividend Analytics ────────────────────────────────────────────────

def _render_dvds(args: list[str]):
    st.markdown("## DVDS — Dividend & Corporate Action Analytics")
    st.caption("Dividend yield, 5Y growth rate, quality score (0-10), DDM intrinsic value, payout ratio, splits, and special dividends.")

    tab_single, tab_screen = st.tabs(["Single Ticker", "Screen Dividends"])

    with tab_single:
        ticker = st.text_input("Ticker:", value=args[0] if args else "", placeholder="e.g. JNJ, KO, PEP, AAPL")
        if st.button("Analyze Dividends", key="dvds_single_run") and ticker:
            with st.spinner(f"Fetching dividend data for {ticker.upper()}..."):
                try:
                    from sentinel.sfe.corporate_actions import get_dividend_analytics
                    result = run_async(get_dividend_analytics(ticker=ticker.upper()))
                    score_color = "🟢" if result.dividend_quality_score >= 7 else "🟡" if result.dividend_quality_score >= 4 else "🔴"
                    cols = st.columns(4)
                    cols[0].metric("Yield", f"{result.current_yield_pct:.2f}%" if result.current_yield_pct else "N/A")
                    cols[1].metric("5Y DGR", f"{result.dividend_growth_rate_5y:+.1f}%" if result.dividend_growth_rate_5y else "N/A")
                    cols[2].metric("Quality", f"{score_color} {result.dividend_quality_score:.1f}/10")
                    cols[3].metric("Payout Ratio", f"{result.payout_ratio_pct:.1f}%" if result.payout_ratio_pct else "N/A")
                    if result.ddm_intrinsic_value:
                        st.metric("DDM Intrinsic Value", f"${result.ddm_intrinsic_value:.2f}", help=result.ddm_verdict)
                    if result.ex_dividend_date:
                        st.info(f"Next ex-dividend: **{result.ex_dividend_date}** | Pay date: {result.pay_date or '—'}")
                    if result.recent_history:
                        st.markdown("**Recent Dividend History**")
                        rows = [{"Date": d.date, "Amount": f"${d.amount:.4f}", "Frequency": d.frequency} for d in result.recent_history]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    if result.corporate_actions:
                        st.markdown("**Corporate Actions**")
                        rows = [{"Date": a.date, "Action": a.action_type, "Description": a.description} for a in result.corporate_actions]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    for w in result.warnings:
                        st.warning(w)
                except Exception as e:
                    st.error(f"Dividend analytics error: {e}")

    with tab_screen:
        tickers_input = st.text_input("Tickers (comma-sep):", placeholder="JNJ, KO, PEP, MCD, PG, ABBV, T, VZ")
        c1, c2, c3 = st.columns(3)
        with c1:
            min_yield = st.number_input("Min yield (%, 0=any):", 0.0, 20.0, 0.0, step=0.5)
        with c2:
            min_quality = st.number_input("Min quality score (0=any):", 0.0, 10.0, 0.0, step=0.5)
        with c3:
            excl_no_div = st.checkbox("Exclude non-dividend payers", value=True)
        if st.button("Screen", key="dvds_screen_run") and tickers_input:
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            with st.spinner(f"Analyzing dividends for {len(tickers)} tickers..."):
                try:
                    from sentinel.sfe.corporate_actions import screen_dividends
                    result = run_async(screen_dividends(
                        tickers=tickers,
                        min_yield_pct=min_yield if min_yield > 0 else None,
                        min_quality_score=min_quality if min_quality > 0 else None,
                        exclude_no_dividend=excl_no_div,
                    ))
                    cols = st.columns(3)
                    cols[0].metric("Avg Yield", f"{result.avg_yield_pct:.2f}%" if result.avg_yield_pct else "N/A")
                    cols[1].metric("Top Yielder", result.top_yielder or "N/A")
                    cols[2].metric("Highest Quality", result.highest_quality or "N/A")
                    rows = [{"Ticker": a.ticker, "Yield": f"{a.current_yield_pct:.2f}%" if a.current_yield_pct else "—",
                             "5Y DGR": f"{a.dividend_growth_rate_5y:+.1f}%" if a.dividend_growth_rate_5y else "—",
                             "Quality": f"{a.dividend_quality_score:.1f}/10",
                             "Payout": f"{a.payout_ratio_pct:.1f}%" if a.payout_ratio_pct else "—",
                             "DDM": a.ddm_verdict} for a in result.analytics]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                except Exception as e:
                    st.error(f"Dividend screen error: {e}")


# ─── RIAPROF — Form ADV / RIA Intelligence ───────────────────────────────────

def _render_riaprof(args: list[str]):
    st.markdown("## RIAPROF — RIA Adviser Intelligence (Form ADV)")
    st.caption("SEC IAPD: AUM, client types, fee structure, investment styles for Registered Investment Advisers.")

    tab_search, tab_screen = st.tabs(["Firm Lookup", "Screen RIAs"])

    with tab_search:
        firm_name = st.text_input("Firm name:", value=" ".join(args) if args else "", placeholder="e.g. Bridgewater, Vanguard, Two Sigma")
        if st.button("Look Up RIA", key="ria_lookup_run") and firm_name:
            with st.spinner(f"Fetching Form ADV for {firm_name}..."):
                try:
                    from sentinel.sfe.form_adv import get_ria_profile
                    result = run_async(get_ria_profile(firm_name=firm_name))
                    cols = st.columns(3)
                    cols[0].metric("AUM", _fmt_large(result.aum_usd) if result.aum_usd else "N/A")
                    cols[1].metric("Clients", f"{result.num_clients:,}" if result.num_clients else "N/A")
                    cols[2].metric("Advisers", str(result.num_advisers) if result.num_advisers else "N/A")
                    st.markdown(f"**{result.name}** | CRD: {result.crd_number or '—'} | State: {result.primary_state or '—'} | Latest ADV: {result.latest_adv_date or '—'}")
                    if result.fee_structure:
                        fees = []
                        if result.fee_structure.pct_of_aum: fees.append("% of AUM")
                        if result.fee_structure.hourly: fees.append("Hourly")
                        if result.fee_structure.fixed_fee: fees.append("Fixed")
                        if result.fee_structure.performance_based: fees.append("Performance")
                        st.markdown(f"**Fee structure:** {', '.join(fees) or '—'}")
                    if result.investment_styles:
                        st.markdown(f"**Investment styles:** {', '.join(result.investment_styles)}")
                    if result.client_types:
                        rows = [{"Category": c.category, "Count": c.count or "—", "% Clients": f"{c.pct_of_clients:.1f}%" if c.pct_of_clients else "—"} for c in result.client_types]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    for w in result.warnings:
                        st.warning(w)
                except Exception as e:
                    st.error(f"RIA lookup error: {e}")

    with tab_screen:
        c1, c2, c3 = st.columns(3)
        with c1:
            screen_query = st.text_input("Search:", placeholder="firm name or keyword")
            limit = st.slider("Results:", 5, 25, 10)
        with c2:
            min_aum = st.number_input("Min AUM ($B, 0=any):", 0.0, 10000.0, 0.0)
            max_aum = st.number_input("Max AUM ($B, 0=any):", 0.0, 10000.0, 0.0)
        with c3:
            state = st.text_input("State (2-letter, blank=all):", placeholder="NY")
        if st.button("Screen", key="ria_screen_run") and screen_query:
            with st.spinner("Screening RIAs..."):
                try:
                    from sentinel.sfe.form_adv import screen_rias
                    result = run_async(screen_rias(
                        query=screen_query,
                        min_aum_billions=min_aum if min_aum > 0 else None,
                        max_aum_billions=max_aum if max_aum > 0 else None,
                        state=state.upper() if state else None,
                        limit=limit,
                    ))
                    st.markdown(f"**{result.total_found} RIAs found**")
                    rows = [{"Name": p.name, "AUM": _fmt_large(p.aum_usd) if p.aum_usd else "—",
                             "Clients": p.num_clients or "—", "State": p.primary_state or "—",
                             "CRD": p.crd_number or "—"} for p in result.profiles]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                except Exception as e:
                    st.error(f"RIA screen error: {e}")


# ─── GEORISK — Geopolitical Risk Dashboard ────────────────────────────────────

def _render_georisk(args: list[str]):
    st.markdown("## GEORISK — Geopolitical Risk Dashboard")
    st.caption("GDELT event analysis + Claude Haiku synthesis. Conflict, political instability, and sanctions risk by country.")

    tab_country, tab_dashboard = st.tabs(["Single Country", "Global Dashboard"])

    with tab_country:
        country = st.text_input("Country:", value=args[0] if args else "", placeholder="e.g. Russia, Iran, Taiwan, Ukraine")
        lookback = st.slider("Lookback (days):", 7, 90, 30)
        if st.button("Score Risk", key="georisk_single_run") and country:
            with st.spinner(f"Analyzing GDELT signals for {country}..."):
                try:
                    from sentinel.sma.geopolitical_risk import get_country_risk
                    result = run_async(get_country_risk(country=country, lookback_days=lookback))
                    score_color = "🔴" if result.overall_score >= 7 else "🟡" if result.overall_score >= 4 else "🟢"
                    cols = st.columns(4)
                    cols[0].metric("Overall Risk", f"{score_color} {result.overall_score:.1f}/10")
                    cols[1].metric("Conflict", f"{result.conflict_score:.1f}/10")
                    cols[2].metric("Political", f"{result.political_score:.1f}/10")
                    cols[3].metric("Economic", f"{result.economic_score:.1f}/10")
                    st.markdown(f"**Trend:** {result.trend.upper()} | **Top themes:** {', '.join(result.top_themes[:5])}")
                    st.info(result.narrative)
                    if result.recent_events:
                        rows = [{"Date": e.date[:10], "Headline": e.title[:80], "Theme": e.theme, "Source": e.source} for e in result.recent_events[:10]]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    for w in result.warnings:
                        st.warning(w)
                except Exception as e:
                    st.error(f"Geopolitical risk error: {e}")

    with tab_dashboard:
        custom_countries = st.text_input("Countries (comma-sep, blank=default 8):", placeholder="Russia, Iran, Taiwan, Ukraine")
        if st.button("Load Dashboard", key="georisk_dash_run"):
            with st.spinner("Fetching GDELT data for all regions..."):
                try:
                    import json
                    from sentinel.sma.geopolitical_risk import get_geopolitical_dashboard
                    countries = [c.strip() for c in custom_countries.split(",") if c.strip()] or None
                    result = run_async(get_geopolitical_dashboard(countries=countries))
                    st.metric("Global Risk Index", f"{result.global_risk_index:.1f}/10",
                              help=f"Highest: {result.highest_risk} | Lowest: {result.lowest_risk}")
                    if result.key_flashpoints:
                        st.markdown("**Key Flashpoints:** " + " | ".join(result.key_flashpoints))
                    rows = [{"Country": r.country,
                             "Risk": f"{r.overall_score:.1f}/10",
                             "Conflict": f"{r.conflict_score:.1f}",
                             "Political": f"{r.political_score:.1f}",
                             "Trend": r.trend.upper()} for r in sorted(result.regions, key=lambda x: x.overall_score, reverse=True)]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    for w in result.warnings:
                        st.warning(w)
                except Exception as e:
                    st.error(f"Geopolitical dashboard error: {e}")


# ─── LBO — LBO / Merger Model ────────────────────────────────────────────────

def _render_lbo(args: list[str]):
    st.markdown("## LBO — Leveraged Buyout & Merger Model")
    st.caption("LBO returns (IRR/MOIC) and M&A accretion/dilution analysis. Equivalent to CapIQ LBO templates at $0.")

    tab_lbo, tab_merger, tab_screen = st.tabs(["LBO Model", "Merger A/D", "LBO Screener"])

    with tab_lbo:
        c1, c2 = st.columns(2)
        with c1:
            purchase_price = st.number_input("Purchase Price / EV ($M):", 100.0, 1e6, 1000.0, step=50.0)
            ebitda = st.number_input("Entry EBITDA ($M):", 10.0, 1e5, 120.0, step=5.0)
            ebitda_growth = st.slider("EBITDA Growth Rate (% p.a.):", 0.0, 30.0, 5.0) / 100
            leverage = st.slider("Leverage (Debt/EBITDA):", 1.0, 9.0, 5.0, step=0.5)
        with c2:
            interest_rate = st.slider("Interest Rate (%):", 4.0, 15.0, 8.5, step=0.25) / 100
            hold_years = st.slider("Hold Period (years):", 2, 10, 5)
            exit_multiple = st.slider("Exit EV/EBITDA Multiple:", 4.0, 20.0, 8.0, step=0.5)
            tax_rate = st.slider("Tax Rate (%):", 10.0, 40.0, 25.0, step=1.0) / 100

        if st.button("Run LBO Model", key="lbo_run"):
            try:
                from sentinel.sfe.lbo_model import LBOAssumptions, run_lbo_model
                assumptions = LBOAssumptions(
                    purchase_price=purchase_price, ebitda=ebitda,
                    ebitda_growth_rate=ebitda_growth, leverage_multiple=leverage,
                    interest_rate=interest_rate, hold_years=hold_years,
                    exit_multiple=exit_multiple, tax_rate=tax_rate,
                )
                result = run_lbo_model(assumptions=assumptions)
                verdict_color = "🟢" if "strong" in result.verdict else "🟡" if "acceptable" in result.verdict else "🔴"
                cols = st.columns(4)
                cols[0].metric("IRR", f"{result.irr*100:.1f}%")
                cols[1].metric("MOIC", f"{result.moic:.2f}×")
                cols[2].metric("Equity Invested", _fmt_large(result.equity_invested * 1e6))
                cols[3].metric("Verdict", f"{verdict_color} {result.verdict.title()}")
                rows = [{"Year": y.year, "EBITDA": f"${y.ebitda:.1f}M", "Interest": f"${y.interest_expense:.1f}M",
                         "Debt Balance": f"${y.debt_balance:.1f}M", "FCF": f"${y.free_cash_flow:.1f}M"} for y in result.years]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            except Exception as e:
                st.error(f"LBO model error: {e}")

    with tab_merger:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Acquirer**")
            acq_eps = st.number_input("EPS ($):", 0.01, 1000.0, 5.0, key="acq_eps")
            acq_shares = st.number_input("Shares (M):", 1.0, 100000.0, 1000.0, key="acq_shares")
            acq_price = st.number_input("Stock Price ($):", 1.0, 10000.0, 100.0, key="acq_price")
        with c2:
            st.markdown("**Target**")
            tgt_eps = st.number_input("EPS ($):", 0.01, 1000.0, 3.0, key="tgt_eps")
            tgt_shares = st.number_input("Shares (M):", 1.0, 100000.0, 500.0, key="tgt_shares")
            tgt_acq_price = st.number_input("Offer Price ($):", 1.0, 10000.0, 60.0, key="tgt_acq_price")
        c3, c4 = st.columns(2)
        with c3:
            pct_stock = st.slider("% Paid in Stock:", 0.0, 100.0, 0.0) / 100
            synergies = st.number_input("After-tax Synergies ($M):", 0.0, 10000.0, 0.0)
        with c4:
            cost_debt = st.slider("Cost of Debt (%):", 3.0, 15.0, 8.0) / 100

        if st.button("Run Merger Model", key="merger_run"):
            try:
                from sentinel.sfe.lbo_model import MergerAssumptions, run_merger_model
                assumptions = MergerAssumptions(
                    acquirer_eps=acq_eps, acquirer_shares_mm=acq_shares, acquirer_price=acq_price,
                    target_eps=tgt_eps, target_shares_mm=tgt_shares,
                    acquisition_price_per_share=tgt_acq_price, pct_stock=pct_stock,
                    synergies_after_tax_mm=synergies, cost_of_debt=cost_debt,
                )
                result = run_merger_model(assumptions=assumptions)
                verdict_color = "🟢" if result.accretion_pct > 0 else "🔴"
                cols = st.columns(4)
                cols[0].metric("Combined EPS", f"${result.combined_eps:.2f}")
                cols[1].metric("Accretion", f"{result.accretion_pct:+.2f}%", delta=f"{result.verdict}")
                cols[2].metric("Premium Paid", f"{result.premium_paid_pct:.1f}%")
                cols[3].metric("Deal Value", _fmt_large(result.deal_value_mm * 1e6))
                st.metric("New Shares Issued", f"{result.new_shares_issued_mm:.1f}M", help="Dilution from stock component")
            except Exception as e:
                st.error(f"Merger model error: {e}")

    with tab_screen:
        ticker = st.text_input("Ticker:", value=args[0] if args else "", placeholder="e.g. DELL, CCL, HCA")
        if st.button("Screen LBO", key="lbo_screen_run") and ticker:
            with st.spinner(f"Fetching financials for {ticker.upper()}..."):
                try:
                    from sentinel.sfe.lbo_model import screen_lbo_candidate
                    result = run_async(screen_lbo_candidate(ticker=ticker.upper()))
                    lbo = result.lbo_result
                    cols = st.columns(3)
                    cols[0].metric("IRR", f"{lbo.irr*100:.1f}%")
                    cols[1].metric("MOIC", f"{lbo.moic:.2f}×")
                    cols[2].metric("Entry EV/EBITDA", f"{lbo.assumptions.purchase_price / lbo.assumptions.ebitda:.1f}×")
                    st.caption(f"Market data used: {result.market_data_used}")
                    for w in result.warnings:
                        st.warning(w)
                except Exception as e:
                    st.error(f"LBO screen error: {e}")


# ─── EDGMON — EDGAR Filing Monitor ───────────────────────────────────────────

def _render_edgmon(args: list[str]):
    st.markdown("## EDGMON — EDGAR Filing Monitor")
    st.caption("Real-time SEC EDGAR filings: 8-K, 10-K, SC 13D (activist), S-1 (IPO), Form 4 (insider). Free Bloomberg filing alerts equivalent.")

    tab_recent, tab_watch, tab_insider = st.tabs(["Recent Filings", "Watchlist", "Insider Transactions"])

    with tab_recent:
        c1, c2 = st.columns(2)
        with c1:
            days_back = st.slider("Days back:", 1, 30, 1)
            limit = st.slider("Max filings:", 10, 100, 50)
        with c2:
            form_options = ["8-K", "10-K", "10-Q", "SC 13D", "SC 13G", "S-1", "4", "DEF 14A", "20-F", "6-K"]
            selected_forms = st.multiselect("Form types (blank=all high-priority):", form_options)

        if st.button("Fetch Filings", key="edgmon_recent_run"):
            with st.spinner("Scanning EDGAR..."):
                try:
                    from sentinel.sil.edgar_monitor import get_recent_filings
                    result = run_async(get_recent_filings(
                        form_types=selected_forms or None, days_back=days_back, limit=limit,
                    ))
                    st.markdown(f"**{result.total_found} filings found** ({result.new_count} new) | " +
                                " | ".join(f"{k}: {v}" for k, v in result.form_type_counts.items()))
                    if result.alerts:
                        st.markdown("**🚨 High-Priority Alerts:**")
                        for a in result.alerts:
                            st.warning(f"[{a.filing.form_type}] **{a.filing.entity_name}** — {a.alert_reason}")
                    rows = [{"Date": f.file_date, "Company": f.entity_name, "Form": f.form_type,
                             "Description": f.description, "New": "✓" if f.is_new else ""} for f in result.filings]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    for w in result.warnings:
                        st.warning(w)
                except Exception as e:
                    st.error(f"EDGAR monitor error: {e}")

    with tab_watch:
        ticker_input = st.text_input("Tickers (comma-sep):", placeholder="AAPL, MSFT, TSLA")
        watch_days = st.slider("Days back:", 1, 30, 7, key="edgmon_watch_days")
        if st.button("Monitor", key="edgmon_watch_run") and ticker_input:
            tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
            with st.spinner(f"Monitoring {len(tickers)} tickers..."):
                try:
                    from sentinel.sil.edgar_monitor import monitor_watchlist
                    results = run_async(monitor_watchlist(tickers=tickers, days_back=watch_days))
                    for r in results:
                        with st.expander(f"{r.ticker} — {len(r.filings)} filings", expanded=bool(r.alerts)):
                            for a in r.alerts:
                                st.warning(f"🚨 {a.filing.form_type}: {a.alert_reason}")
                            if r.filings:
                                rows = [{"Date": f.file_date, "Form": f.form_type, "Description": f.description} for f in r.filings]
                                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                except Exception as e:
                    st.error(f"Watchlist monitor error: {e}")

    with tab_insider:
        ins_ticker = st.text_input("Ticker:", value=args[0] if args else "", placeholder="AAPL", key="ins_ticker")
        ins_days = st.slider("Days back:", 7, 180, 30, key="ins_days")
        if st.button("Fetch Insider Transactions", key="ins_run") and ins_ticker:
            with st.spinner(f"Fetching Form 4 filings for {ins_ticker.upper()}..."):
                try:
                    from sentinel.sil.edgar_monitor import get_insider_transactions
                    filings = run_async(get_insider_transactions(
                        ticker=ins_ticker.upper(), days_back=ins_days,
                    ))
                    if not filings:
                        st.info("No Form 4 filings found in this period.")
                    else:
                        rows = [{"Date": f.file_date, "Company": f.entity_name, "Form": f.form_type,
                                 "Description": f.description} for f in filings]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                except Exception as e:
                    st.error(f"Insider transactions error: {e}")


# ─── MASCRN — M&A Deal Screener ──────────────────────────────────────────────

def _render_mascrn(args: list[str]):
    st.markdown("## MASCRN — M&A Deal Screener")
    st.caption("Recent M&A deals from EDGAR 8-K, DEFM14A, and SC TO-T filings. Extracts deal type, value, and status.")

    col1, col2, col3 = st.columns(3)
    with col1:
        days_back = st.slider("Days back:", 7, 180, 30)
        limit = st.slider("Max results:", 5, 50, 25)
    with col2:
        deal_type = st.selectbox("Deal type:", ["All", "merger", "acquisition", "spinoff", "divestiture"], index=0)
        min_val = st.number_input("Min deal value ($B, 0=any):", 0.0, 500.0, 0.0, step=1.0)
    with col3:
        sector = st.text_input("Sector filter (optional):", placeholder="e.g. Technology")

    if st.button("Screen Deals", key="mascrn_run"):
        with st.spinner("Scanning EDGAR for M&A activity..."):
            try:
                from sentinel.sfe.ma_screener import screen_ma_deals
                result = run_async(screen_ma_deals(
                    days_back=days_back,
                    min_value_billions=min_val if min_val > 0 else None,
                    sector=sector or None,
                    deal_type=deal_type if deal_type != "All" else None,
                    limit=limit,
                ))

                st.markdown(f"**{result.total_found} deals found** (last {days_back} days)")

                if result.deals:
                    rows = [{
                        "Date": d.filing_date,
                        "Type": d.deal_type,
                        "Acquirer": d.acquirer or "N/A",
                        "Target": d.target or "N/A",
                        "Value": f"${d.deal_value_billions:.1f}B" if d.deal_value_billions else "N/A",
                        "Status": d.status,
                        "Form": d.form_type,
                    } for d in result.deals]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                    if st.checkbox("Show descriptions"):
                        for d in result.deals:
                            with st.expander(f"{d.filing_date} — {d.acquirer or '?'} / {d.target or '?'}"):
                                st.write(d.description)
                                st.caption(f"[EDGAR]({d.edgar_url})")

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"M&A screener error: {e}")


# ─── MAPROF — M&A Target Profile ──────────────────────────────────────────────

def _render_maprof(args: list[str]):
    ticker = args[0].upper() if args else st.text_input("Ticker:", "AAPL").upper()
    if not ticker:
        return
    st.markdown(f"## MAPROF — M&A Target Profile: {ticker}")
    st.caption("Activist interest, defense mechanisms, recent deal activity from EDGAR")

    if st.button("Get M&A Profile", key="maprof_run"):
        with st.spinner(f"Analyzing {ticker} for M&A activity..."):
            try:
                from sentinel.sfe.ma_screener import get_ma_profile
                result = run_async(get_ma_profile(ticker=ticker))

                a1, a2 = st.columns(2)
                a1.metric("Acquisition Target?", "🎯 YES" if result.is_acquisition_target else "No")
                a2.metric("Activist Interest?", "⚡ YES" if result.activist_interest else "No")

                if result.defense_mechanisms:
                    st.markdown("**Defense Mechanisms**")
                    for m in result.defense_mechanisms:
                        st.write(f"  🛡️ {m}")

                if result.recent_deals:
                    st.markdown(f"**Recent Deal Activity ({len(result.recent_deals)})**")
                    rows = [{
                        "Date": d.filing_date,
                        "Type": d.deal_type,
                        "Description": d.description[:120] + "..." if len(d.description) > 120 else d.description,
                        "Form": d.form_type,
                    } for d in result.recent_deals]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("No recent M&A activity found in EDGAR filings.")

                st.caption(f"As of: {result.as_of}")
                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"M&A profile error: {e}")


# ─── BONDA — Bond Price Analytics ────────────────────────────────────────────

def _render_bonda(args: list[str]):
    st.markdown("## BONDA — Bond Price Analytics")
    st.caption("Duration, convexity, DV01, rate shock scenarios. YTM interpolated from live FRED curve if omitted.")

    col1, col2, col3 = st.columns(3)
    with col1:
        face = st.number_input("Face value ($):", 100.0, 10_000_000.0, 1000.0, step=1000.0)
        coupon = st.number_input("Annual coupon rate (%):", 0.0, 20.0, 5.0, step=0.25) / 100
        maturity = st.number_input("Years to maturity:", 0.25, 30.0, 10.0, step=0.5)
    with col2:
        ytm_input = st.number_input("YTM override (%, 0=auto):", 0.0, 20.0, 0.0, step=0.25)
        freq = st.selectbox("Coupon frequency:", [1, 2, 4, 12], index=1,
                            format_func=lambda x: {1: "Annual", 2: "Semi-annual", 4: "Quarterly", 12: "Monthly"}[x])
    with col3:
        spread = st.number_input("Credit spread (bps):", 0.0, 1000.0, 0.0, step=5.0)

    if st.button("Compute Analytics", key="bonda_run"):
        with st.spinner("Computing bond analytics..."):
            try:
                from sentinel.sfe.bond_analytics import compute_bond_price_analytics
                result = run_async(compute_bond_price_analytics(
                    face_value=face, coupon_rate=coupon,
                    years_to_maturity=maturity,
                    yield_to_maturity=ytm_input / 100 if ytm_input > 0 else None,
                    frequency=freq, credit_spread_bps=spread,
                ))

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Clean Price", f"${result.clean_price:.4f}")
                c2.metric("YTM", f"{result.yield_to_maturity:.3%}")
                c3.metric("DV01 ($1M)", f"${result.dv01:,.0f}")
                c4.metric("Accrued Int.", f"${result.accrued_interest:.4f}")

                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Macaulay Dur.", f"{result.macaulay_duration:.3f}y")
                d2.metric("Modified Dur.", f"{result.modified_duration:.3f}y")
                d3.metric("Eff. Duration", f"{result.effective_duration:.3f}y")
                d4.metric("Convexity", f"{result.convexity:.2f}")

                st.markdown("**Rate Shock Scenarios (price change %)**")
                scen_df = pd.DataFrame([
                    {"Shock": k, "Price Δ (%)": f"{v:+.2f}%"}
                    for k, v in result.scenarios.items()
                ])
                st.dataframe(scen_df, use_container_width=True)

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"Bond analytics error: {e}")


# ─── ZSCORE — Altman Z-Score Credit Risk ──────────────────────────────────────

def _render_zscore(args: list[str]):
    ticker = args[0].upper() if args else st.text_input("Ticker:", "F").upper()
    if not ticker:
        return
    st.markdown(f"## ZSCORE — Altman Z-Score: {ticker}")
    st.caption("Credit risk model: Z > 2.99 safe | 1.81-2.99 grey zone | < 1.81 distress")

    if st.button("Compute Z-Score", key="zscore_run"):
        with st.spinner(f"Computing Altman Z-Score for {ticker}..."):
            try:
                from sentinel.sfe.bond_analytics import compute_altman_z
                result = run_async(compute_altman_z(ticker=ticker))

                zone_color = {"safe": "🟢", "grey": "🟡", "distress": "🔴"}.get(result.classification, "⚪")
                st.markdown(f"### {zone_color} Z = {result.z_score:.2f} — **{result.classification.upper()}**")
                st.metric("Distress Probability", f"{result.probability_of_distress:.1%}")

                st.markdown("**Component Ratios**")
                comp_df = pd.DataFrame([
                    {"Ratio": "X1 — Working Capital / Assets", "Value": f"{result.working_capital_to_assets:.4f}", "Weight": "1.2"},
                    {"Ratio": "X2 — Retained Earnings / Assets", "Value": f"{result.retained_earnings_to_assets:.4f}", "Weight": "1.4"},
                    {"Ratio": "X3 — EBIT / Assets", "Value": f"{result.ebit_to_assets:.4f}", "Weight": "3.3"},
                    {"Ratio": "X4 — Mkt Cap / Total Liabilities", "Value": f"{result.market_cap_to_book_liabilities:.4f}", "Weight": "0.6"},
                    {"Ratio": "X5 — Sales / Assets", "Value": f"{result.sales_to_assets:.4f}", "Weight": "1.0"},
                ])
                st.dataframe(comp_df, use_container_width=True)
                st.caption(f"Data source: {result.data_source} | As of: {result.as_of}")

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"Altman Z-Score error: {e}")


# ─── GARCHVAR — GARCH Conditional VaR ───────────────────────────────────────

def _render_garchvar(args: list[str]):
    st.markdown("## GARCHVAR — GARCH(1,1) Conditional VaR")
    st.caption("Time-varying VaR + CVaR using GARCH volatility clustering model. Basel III backtest. Stress VaR.")

    tickers_raw = st.text_input("Tickers (comma-separated):", "SPY,QQQ,AAPL")
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    col1, col2, col3 = st.columns(3)
    with col1:
        confidence = st.selectbox("Confidence:", [0.95, 0.99, 0.999], index=0)
        horizon = st.number_input("Horizon (days):", 1, 21, 1)
    with col2:
        pv = st.number_input("Portfolio value ($):", 100_000, 100_000_000, 1_000_000, step=100_000)
    with col3:
        backtest_days = st.slider("Backtest window (days):", 63, 504, 252)

    if st.button("Compute GARCH-VaR", key="garchvar_run") and tickers:
        with st.spinner("Fitting GARCH(1,1) and computing conditional VaR..."):
            try:
                import json
                from sentinel.spr.garch_var import compute_garch_var
                result = run_async(compute_garch_var(
                    tickers=tickers, confidence=confidence,
                    horizon_days=horizon, portfolio_value=pv,
                    backtest_days=backtest_days,
                ))

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("GARCH VaR (1d)", f"${result.garch_var_1d:,.0f}")
                c2.metric("GARCH CVaR (1d)", f"${result.garch_cvar_1d:,.0f}")
                c3.metric(f"VaR ({horizon}d)", f"${result.garch_var_Nd:,.0f}")
                c4.metric("Stress VaR", f"${result.stress_var:,.0f}")

                st.caption(f"Stress scenario: {result.stress_scenario}")

                st.markdown("**GARCH Parameters**")
                gp = result.garch_params
                g1, g2, g3, g4 = st.columns(4)
                g1.metric("ω (omega)", f"{gp.omega:.2e}")
                g2.metric("α (shock)", f"{gp.alpha:.4f}")
                g3.metric("β (persistence)", f"{gp.beta:.4f}")
                g4.metric("Long-run vol", f"{gp.long_run_vol:.1%}")

                st.markdown("**Volatility Forecast (annualised)**")
                vf = result.vol_forecast
                v1, v2, v3 = st.columns(3)
                v1.metric("5-day fwd", f"{vf.get('5d', 0):.1%}")
                v2.metric("10-day fwd", f"{vf.get('10d', 0):.1%}")
                v3.metric("21-day fwd", f"{vf.get('21d', 0):.1%}")

                bt = result.backtest
                tl_color = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(bt.traffic_light, "⚪")
                st.markdown(f"**Backtest — Basel III {tl_color} {bt.traffic_light.upper()}**")
                b1, b2, b3 = st.columns(3)
                b1.metric("Exceedances", f"{bt.exceedances} / {backtest_days}")
                b2.metric("Exc. rate", f"{bt.exceedance_rate:.1%}")
                b3.metric("Kupiec p-val", f"{bt.kupiec_pvalue:.3f}")

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"GARCH-VaR error: {e}")


# ─── ADVTA — Advanced Technical Analysis ──────────────────────────────────────

def _render_advta(args: list[str]):
    ticker = args[0].upper() if args else st.text_input("Ticker:", "AAPL").upper()
    if not ticker:
        return
    st.markdown(f"## ADVTA — Advanced Technical Analysis: {ticker}")
    st.caption("Ichimoku Cloud, Fibonacci retracements, ADX trend strength, Parabolic SAR")

    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("Period:", ["6mo", "1y", "2y"], index=1, key="advta_period")
    with col2:
        interval = st.selectbox("Interval:", ["1d", "1h"], index=0, key="advta_interval")

    if st.button("Run Advanced TA", key="advta_run"):
        with st.spinner(f"Computing advanced signals for {ticker}..."):
            try:
                from sentinel.spr.ta_advanced import get_advanced_ta
                result = run_async(get_advanced_ta(ticker=ticker, period=period, interval=interval))

                score = result.composite_score
                label = result.composite_label
                color = "🟢" if score > 0.2 else ("🔴" if score < -0.2 else "🟡")
                st.markdown(f"### {color} {label} (score: {score:+.2f})")
                st.caption(f"Price: **${result.price:.2f}** | As of: {result.as_of}")

                col_ich, col_fib = st.columns(2)
                with col_ich:
                    st.markdown("**Ichimoku Cloud**")
                    ich = result.ichimoku
                    st.write(f"Signal: **{ich.signal}** | Price: **{ich.price_vs_cloud}** cloud")
                    st.write(f"Tenkan: {ich.tenkan_sen:.2f} | Kijun: {ich.kijun_sen:.2f}")
                    st.write(f"Span A: {ich.senkou_span_a:.2f} | Span B: {ich.senkou_span_b:.2f}")

                with col_fib:
                    st.markdown("**Fibonacci Retracements**")
                    fib = result.fibonacci
                    st.write(f"Swing High: ${fib.swing_high:.2f} | Low: ${fib.swing_low:.2f}")
                    st.write(f"Nearest Support: ${fib.nearest_support:.2f}")
                    st.write(f"Nearest Resistance: ${fib.nearest_resistance:.2f}")
                    fib_df = pd.DataFrame(
                        [{"Level": k, "Price": f"${v:.2f}"} for k, v in fib.levels.items()]
                    )
                    st.dataframe(fib_df, use_container_width=True)

                col_adx, col_sar = st.columns(2)
                with col_adx:
                    st.markdown("**ADX (Trend Strength)**")
                    adx = result.adx
                    st.metric("ADX", f"{adx.adx:.1f}", help=">25 = trending, <20 = ranging")
                    st.write(f"+DI: {adx.plus_di:.1f} | -DI: {adx.minus_di:.1f}")
                    st.write(f"Signal: **{adx.signal}**")

                with col_sar:
                    st.markdown("**Parabolic SAR**")
                    sar = result.parabolic_sar
                    sar_color = "🟢" if sar.trend == "bullish" else "🔴"
                    st.write(f"{sar_color} Trend: **{sar.trend}**")
                    st.metric("SAR Level", f"${sar.sar:.2f}")
                    st.write(f"Reversal at: ${sar.reversal_price:.2f}")
                    st.write(f"AF: {sar.acceleration:.3f}")

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"Advanced TA error: {e}")


# ─── TASIG — Technical Analysis Signals ──────────────────────────────────────

def _render_tasig(args: list[str]):
    ticker = args[0].upper() if args else st.text_input("Ticker:", "AAPL").upper()
    if not ticker:
        return
    st.markdown(f"## TASIG — Technical Analysis: {ticker}")
    st.caption("RSI, MACD, Bollinger Bands, VWAP, Stochastic, Williams %R, OBV, ATR. Composite bull/bear score.")

    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("History period:", ["3mo", "6mo", "1y", "2y"], index=2)
    with col2:
        interval = st.selectbox("Interval:", ["1d", "1h"], index=0)

    if st.button("Run TA", key="tasig_run"):
        with st.spinner(f"Computing signals for {ticker}..."):
            try:
                from sentinel.spr.ta_engine import get_ta_signals
                result = run_async(get_ta_signals(ticker=ticker, period=period, interval=interval))

                score = result.composite_score
                label = result.composite_label
                color = "🟢" if score > 0.2 else ("🔴" if score < -0.2 else "🟡")
                st.markdown(f"### {color} {label} (score: {score:+.2f})")
                st.caption(f"Trend: **{result.trend}** | Price: **${result.price:.2f}** | As of: {result.as_of}")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Support Levels**")
                    for s in result.support_levels:
                        st.write(f"  ${s:.2f}")
                with c2:
                    st.markdown("**Resistance Levels**")
                    for r in result.resistance_levels:
                        st.write(f"  ${r:.2f}")

                st.markdown("**Signals**")
                rows = [{"Indicator": sig.name, "Value": f"{sig.value:.4f}",
                         "Signal": sig.signal, "Note": sig.description}
                        for sig in result.signals]
                df = pd.DataFrame(rows)
                def _color_signal(val):
                    if val == "bullish":
                        return "background-color: #0a2a0a; color: #00ff41"
                    if val == "bearish":
                        return "background-color: #2a0a0a; color: #ff4141"
                    return ""
                st.dataframe(df.style.applymap(_color_signal, subset=["Signal"]),
                             use_container_width=True)

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"TA engine error: {e}")


# ─── SYNTH — Research Synthesis ───────────────────────────────────────────────

def _render_synth(args: list[str]):
    ticker = args[0].upper() if args else st.text_input("Ticker:", "NVDA").upper()
    if not ticker:
        return
    st.markdown(f"## SYNTH — AI Research Synthesis: {ticker}")
    st.caption("Multi-source Claude synthesis: SEC filings + news + fundamentals + insider → bull/bear/risks/catalysts")

    query = st.text_input("Optional focus question:", placeholder="e.g. What are the margin expansion risks?")
    col1, col2 = st.columns(2)
    with col1:
        inc_filings = st.checkbox("Include SEC filings", value=True)
        inc_news = st.checkbox("Include news", value=True)
    with col2:
        inc_fundamentals = st.checkbox("Include fundamentals", value=True)
        inc_insider = st.checkbox("Include insider data", value=True)

    if st.button("Synthesize", key="synth_run"):
        with st.spinner(f"Gathering data and synthesizing research for {ticker}..."):
            try:
                from sentinel.sil.research_synthesis import synthesize_research
                result = run_async(synthesize_research(
                    ticker=ticker,
                    query=query or None,
                    include_filings=inc_filings,
                    include_news=inc_news,
                    include_fundamentals=inc_fundamentals,
                    include_insider=inc_insider,
                ))

                conf_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(result.confidence, "⚪")
                st.caption(f"{conf_color} Confidence: **{result.confidence}** | As of: {result.as_of}")

                st.markdown("### Synthesis")
                st.markdown(result.synthesis)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Bull Case**")
                    st.markdown(result.bull_case)
                with c2:
                    st.markdown("**Bear Case**")
                    st.markdown(result.bear_case)

                if result.key_risks:
                    st.markdown("**Key Risks**")
                    for risk in result.key_risks:
                        st.markdown(f"- {risk}")

                if result.catalyst_watch:
                    st.markdown("**Catalyst Watch**")
                    for cat in result.catalyst_watch:
                        st.markdown(f"- {cat}")

                if result.sources_used:
                    with st.expander("Sources"):
                        src_rows = [{"Type": s.source_type, "Title": s.title,
                                     "Sentiment": s.sentiment, "Relevance": f"{s.relevance:.0%}"}
                                    for s in result.sources_used]
                        st.dataframe(pd.DataFrame(src_rows), use_container_width=True)

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"Research synthesis error: {e}")


# ─── ESURP — Earnings Surprise Tracker ───────────────────────────────────────

def _render_esurp(args: list[str]):
    tickers_in = " ".join(args) if args else ""
    ticker_input = st.text_input("Ticker(s) (space-separated for screen):", tickers_in or "AAPL")
    tickers = [t.strip().upper() for t in ticker_input.split() if t.strip()]
    if not tickers:
        return
    st.markdown("## ESURP — Earnings Surprise Tracker")
    st.caption("EPS beat/miss history vs estimates: surprise %, beat rate, consistency score, trend. Free FactSet earnings quality.")

    mode = st.radio("Mode", ["Single ticker", "Beat screener"], horizontal=True)
    col1, col2 = st.columns(2)
    with col1:
        quarters = st.number_input("Quarters to analyze", 2, 16, 8)
    with col2:
        min_beat = st.slider("Min beat rate % (screen)", 0.0, 100.0, 60.0, 5.0) / 100

    if st.button("Analyze Earnings Surprise", key="esurp_run"):
        with st.spinner("Fetching earnings history..."):
            try:
                if mode == "Single ticker":
                    from sentinel.sfe.earnings_surprise import get_earnings_surprise
                    result = run_async(get_earnings_surprise(ticker=tickers[0], quarters=quarters))

                    trend_color = {"improving": "🟢", "stable": "🟡", "deteriorating": "🔴",
                                   "insufficient data": "⚪"}.get(result.trend, "⚪")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Beat Rate", f"{result.beat_rate:.0%}" if result.beat_rate is not None else "—")
                    c2.metric("Avg Surprise", f"{result.avg_surprise_pct:+.1f}%" if result.avg_surprise_pct else "—")
                    c3.metric("Consistency", f"{result.consistency_score:.1f}/10")
                    c4.metric("Trend", f"{trend_color} {result.trend.title()}")

                    col_s, col_n = st.columns(2)
                    with col_s:
                        st.metric("Consecutive Beats", result.consecutive_beats)
                        st.metric("Quarters Analyzed", result.quarters_analyzed)
                    with col_n:
                        if result.next_earnings_date:
                            st.metric("Next Earnings", result.next_earnings_date)
                        if result.estimated_next_eps is not None:
                            st.metric("Est. Next EPS", f"${result.estimated_next_eps:.2f}")

                    if result.surprises:
                        st.markdown("**Quarter-by-Quarter EPS**")
                        rows = []
                        for s in result.surprises[:quarters]:
                            beat_icon = "✓" if s.beat else ("✗" if s.beat is False else "—")
                            rows.append({
                                "Period": s.period,
                                "Actual": f"${s.actual_eps:.2f}" if s.actual_eps is not None else "—",
                                "Estimate": f"${s.estimate_eps:.2f}" if s.estimate_eps is not None else "—",
                                "Surprise": f"{s.surprise_pct:+.1f}%" if s.surprise_pct is not None else "—",
                                "Beat": beat_icon,
                                "Label": s.magnitude_label.title(),
                            })
                        df = pd.DataFrame(rows)
                        def _beat_color(val):
                            if val == "✓": return "background-color: #0a2a0a; color: #00ff41"
                            if val == "✗": return "background-color: #2a0a0a; color: #ff4141"
                            return ""
                        st.dataframe(df.style.applymap(_beat_color, subset=["Beat"]), use_container_width=True)
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
                else:
                    from sentinel.sfe.earnings_surprise import screen_earnings_beats
                    result = run_async(screen_earnings_beats(tickers=tickers, min_beat_rate=min_beat))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tickers Screened", result.tickers_screened)
                    c2.metric("Avg Beat Rate", f"{result.avg_beat_rate:.0%}" if result.avg_beat_rate else "—")
                    c3.metric("Passing Filter", len(result.results))
                    if result.most_consistent:
                        st.success(f"Most consistent: **{result.most_consistent}**")
                    if result.biggest_avg_beat:
                        st.info(f"Biggest avg beat: **{result.biggest_avg_beat}**")
                    if result.recent_misses:
                        st.warning(f"Recent misses: {' | '.join(result.recent_misses)}")
                    if result.results:
                        rows = [{"Ticker": r.ticker,
                                 "Beat Rate": f"{r.beat_rate:.0%}",
                                 "Avg Surprise": f"{r.avg_surprise_pct:+.1f}%" if r.avg_surprise_pct else "—",
                                 "Score": f"{r.consistency_score:.1f}",
                                 "Quarters": r.quarters_analyzed,
                                 "Last Q": "Beat" if r.last_quarter_beat else ("Miss" if r.last_quarter_beat is False else "—")}
                                for r in result.results]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
            except Exception as e:
                st.error(f"Earnings surprise error: {e}")


# ─── GOV — Corporate Governance Scoring ──────────────────────────────────────

def _render_gov(args: list[str]):
    tickers_in = " ".join(args) if args else ""
    ticker_input = st.text_input("Ticker(s) (space-separated for screen):", tickers_in or "AAPL")
    tickers = [t.strip().upper() for t in ticker_input.split() if t.strip()]
    if not tickers:
        return
    st.markdown("## GOV — Corporate Governance Analysis")
    st.caption("EDGAR DEF 14A: board independence, CEO duality, say-on-pay, diversity, poison pill. Free ISS proxy equivalent.")

    mode = st.radio("Mode", ["Single ticker", "Screen"], horizontal=True)
    min_gov_score = st.slider("Min governance score (screen)", 0.0, 10.0, 5.0, 0.5)

    if st.button("Analyze Governance", key="gov_run"):
        with st.spinner("Fetching DEF 14A proxy filing..."):
            try:
                if mode == "Single ticker":
                    from sentinel.sfe.governance import get_governance_profile
                    result = run_async(get_governance_profile(ticker=tickers[0]))

                    score = result.governance_score
                    verdict_color = "🟢" if score >= 7 else ("🔴" if score <= 4 else "🟡")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Gov Score", f"{verdict_color} {score:.1f}/10")
                    c2.metric("Verdict", result.verdict.title())
                    c3.metric("Filing Date", result.filing_date or "—")
                    c4.metric("CIK", result.cik or "—")

                    col_board, col_checks = st.columns(2)
                    with col_board:
                        st.markdown("**Board Composition**")
                        if result.board_size is not None:
                            st.metric("Board Size", result.board_size)
                        if result.board_independence_pct is not None:
                            ind_color = "🟢" if result.board_independence_pct >= 70 else "🔴"
                            st.metric("Independence", f"{ind_color} {result.board_independence_pct:.0f}%")
                        duality_icon = "🔴 YES (negative)" if result.ceo_duality else "🟢 No"
                        st.markdown(f"**CEO=Chairman:** {duality_icon}")
                        div_icon = "🟢 Yes" if result.board_has_female_director else "🔴 No"
                        st.markdown(f"**Female Director:** {div_icon}")
                    with col_checks:
                        st.markdown("**Governance Checks**")
                        audit_icon = "🟢 Independent" if result.audit_committee_independent else "🔴 Not confirmed"
                        st.markdown(f"**Audit Committee:** {audit_icon}")
                        if result.say_on_pay_pct is not None:
                            sop_color = "🟢" if result.say_on_pay_pct > 90 else ("🟡" if result.say_on_pay_pct > 80 else "🔴")
                            st.metric("Say-on-Pay", f"{sop_color} {result.say_on_pay_pct:.1f}%")
                        stag_icon = "🔴 YES (negative)" if result.classified_board else "🟢 No"
                        st.markdown(f"**Staggered Board:** {stag_icon}")
                        pill_icon = "🔴 YES (negative)" if result.poison_pill else "🟢 No"
                        st.markdown(f"**Poison Pill:** {pill_icon}")
                        p4p_icon = "🟢 Yes" if result.pay_for_performance else "🟡 Not confirmed"
                        st.markdown(f"**Pay-for-Performance:** {p4p_icon}")

                    if result.score_components:
                        st.markdown("**Score Components**")
                        comp_rows = [{"Factor": k, "Points": f"{v:+.1f}"} for k, v in result.score_components.items()]
                        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

                    if result.filing_url:
                        st.caption(f"Source: {result.filing_url}")
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
                else:
                    from sentinel.sfe.governance import screen_governance
                    result = run_async(screen_governance(tickers=tickers, min_score=min_gov_score))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tickers", result.tickers_screened)
                    c2.metric("Avg Score", f"{result.avg_score:.1f}/10" if result.avg_score else "—")
                    c3.metric("Strong Governance", len(result.strong_governance))
                    if result.best_governance:
                        st.success(f"Best: **{result.best_governance}**")
                    if result.worst_governance:
                        st.error(f"Worst: **{result.worst_governance}**")
                    if result.results:
                        rows = [{"Ticker": r.ticker, "Score": f"{r.governance_score:.1f}",
                                 "Verdict": r.verdict.title(),
                                 "CEO=Chair": "Yes" if r.ceo_duality else "No",
                                 "Board Indep": f"{r.board_independence_pct:.0f}%" if r.board_independence_pct else "—",
                                 "Say-on-Pay": f"{r.say_on_pay_pct:.1f}%" if r.say_on_pay_pct else "—"}
                                for r in result.results]
                        df = pd.DataFrame(rows)
                        def _gov_score_color(val):
                            try:
                                v = float(str(val))
                                if v >= 7: return "background-color: #0a2a0a; color: #00ff41"
                                if v <= 4: return "background-color: #2a0a0a; color: #ff4141"
                                return ""
                            except Exception:
                                return ""
                        st.dataframe(df.style.applymap(_gov_score_color, subset=["Score"]), use_container_width=True)
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
            except Exception as e:
                st.error(f"Governance analysis error: {e}")


# ─── SQUEEZE — Short-Squeeze Extended Signals ────────────────────────────────

def _render_squeeze(args: list[str]):
    tickers_in = " ".join(args) if args else ""
    ticker_input = st.text_input("Ticker(s) (space-separated for screen):", tickers_in or "GME AMC TSLA")
    tickers = [t.strip().upper() for t in ticker_input.split() if t.strip()]
    if not tickers:
        return
    st.markdown("## SQUEEZE — Short-Squeeze Signal Analytics")
    st.caption("FINRA DTC, SI % float, borrow cost proxy (easy/moderate/hard/special), gamma squeeze risk, composite score 0-10.")

    mode = st.radio("Mode", ["Single ticker", "Screen"], horizontal=True)
    min_score = st.slider("Min squeeze score (screen)", 0.0, 10.0, 5.0, 0.5)

    if st.button("Analyze Squeeze Risk", key="squeeze_run"):
        with st.spinner("Fetching short interest and equity data..."):
            try:
                if mode == "Single ticker":
                    from sentinel.sbx.squeeze_analytics import get_squeeze_analytics
                    result = run_async(get_squeeze_analytics(ticker=tickers[0]))

                    verdict_color = {
                        "high squeeze risk": "🔴", "elevated squeeze risk": "🟠",
                        "moderate squeeze risk": "🟡", "low squeeze risk": "🟢"
                    }.get(result.squeeze_verdict, "⚪")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Squeeze Score", f"{result.squeeze_score:.1f}/10")
                    c2.metric("Verdict", f"{verdict_color} {result.squeeze_verdict.title()}")
                    c3.metric("Gamma Risk", "YES 🔴" if result.gamma_squeeze_risk else "no 🟢")
                    c4.metric("As of", result.as_of[:10])

                    col_s, col_p = st.columns(2)
                    with col_s:
                        st.markdown("**Short Interest Signals**")
                        ss = result.short_signals
                        if ss.si_pct_float: st.metric("SI % Float", f"{ss.si_pct_float:.1f}%")
                        if ss.days_to_cover: st.metric("Days-to-Cover", f"{ss.days_to_cover:.1f}d")
                        if ss.si_change_mom_pct is not None:
                            st.metric("SI MoM Change", f"{ss.si_change_mom_pct:+.1f}%")
                        if ss.borrow_cost_proxy_pct:
                            st.metric("Borrow Cost Proxy", f"{ss.borrow_cost_proxy_pct:.1f}%/yr")
                        borrow_color = {"easy": "🟢", "moderate": "🟡", "hard": "🟠", "special": "🔴"}.get(ss.borrow_difficulty, "⚪")
                        st.markdown(f"**Borrow:** {borrow_color} {ss.borrow_difficulty.title()}")
                    with col_p:
                        st.markdown("**Price Momentum**")
                        pm = result.price_momentum
                        if pm.current_price: st.metric("Price", f"${pm.current_price:.2f}")
                        if pm.return_5d_pct is not None: st.metric("5d Return", f"{pm.return_5d_pct:+.1f}%")
                        if pm.return_20d_pct is not None: st.metric("20d Return", f"{pm.return_20d_pct:+.1f}%")
                        if pm.rsi_14 is not None: st.metric("RSI-14", f"{pm.rsi_14:.1f}")
                        if result.call_put_oi_ratio is not None: st.metric("C/P OI Ratio", f"{result.call_put_oi_ratio:.2f}x")

                    if result.score_components:
                        st.markdown("**Score Breakdown**")
                        comp_rows = [{"Component": k, "Points": f"{v:+.1f}"} for k, v in result.score_components.items()]
                        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
                else:
                    from sentinel.sbx.squeeze_analytics import screen_squeeze_candidates
                    result = run_async(screen_squeeze_candidates(tickers=tickers, min_squeeze_score=min_score))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Scanned", result.tickers_screened)
                    c2.metric("Avg SI Float", f"{result.avg_si_pct_float:.1f}%" if result.avg_si_pct_float else "—")
                    c3.metric("Gamma Candidates", len(result.gamma_squeeze_candidates))
                    if result.highest_squeeze_risk:
                        st.error(f"Highest risk: {' | '.join(result.highest_squeeze_risk)}")
                    if result.gamma_squeeze_candidates:
                        st.warning(f"Gamma squeeze: {' | '.join(result.gamma_squeeze_candidates)}")
                    if result.results:
                        rows = [{"Ticker": r.ticker, "Score": f"{r.squeeze_score:.1f}",
                                 "Verdict": r.squeeze_verdict.title(),
                                 "SI Float": f"{r.si_pct_float:.1f}%" if r.si_pct_float else "—",
                                 "DTC": f"{r.days_to_cover:.1f}d" if r.days_to_cover else "—",
                                 "Borrow": r.borrow_difficulty.title()}
                                for r in result.results]
                        df = pd.DataFrame(rows)
                        def _verdict_color(val):
                            if "high" in val.lower(): return "background-color: #2a0a0a; color: #ff4141"
                            if "elevated" in val.lower(): return "background-color: #2a1500; color: #ff8800"
                            if "moderate" in val.lower(): return "background-color: #2a2a00; color: #ffff00"
                            return ""
                        st.dataframe(df.style.applymap(_verdict_color, subset=["Verdict"]), use_container_width=True)
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
            except Exception as e:
                st.error(f"Squeeze analytics error: {e}")


# ─── ANLEST — Analyst Estimates Proxy ────────────────────────────────────────

def _render_anlest(args: list[str]):
    tickers_in = " ".join(args) if args else ""
    ticker_input = st.text_input("Ticker(s) (space-separated for screen):", tickers_in or "AAPL")
    tickers = [t.strip().upper() for t in ticker_input.split() if t.strip()]
    if not tickers:
        return
    st.markdown("## ANLEST — Analyst Estimates")
    st.caption("Consensus targets, recs, EPS/revenue estimates via yfinance. Free FactSet Estimates proxy.")

    mode = st.radio("Mode", ["Single ticker", "Screen"], horizontal=True)
    min_upside = st.slider("Min upside % (screen mode)", 0.0, 50.0, 10.0, 2.5) if len(tickers) > 1 else 10.0

    if st.button("Fetch Analyst Data", key="anlest_run"):
        with st.spinner("Fetching analyst consensus..."):
            try:
                if mode == "Single ticker":
                    from sentinel.sfe.analyst_estimates import get_analyst_estimates
                    result = run_async(get_analyst_estimates(ticker=tickers[0]))
                    cons_color = {"Strong Buy": "🟢", "Buy": "🟢", "Hold": "🟡",
                                  "Sell": "🔴", "Strong Sell": "🔴"}.get(result.consensus_label, "⚪")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Consensus", f"{cons_color} {result.consensus_label}")
                    upside = result.price_upside_pct
                    upside_str = f"{upside:+.1f}%" if upside is not None else "—"
                    c2.metric("Mean Target", f"${result.target_mean:.2f}" if result.target_mean else "—",
                              delta=upside_str)
                    c3.metric("Analysts", result.num_analysts or "—")
                    c4.metric("Dispersion", f"{result.analyst_dispersion:.1%}" if result.analyst_dispersion else "—")

                    col_t, col_g = st.columns(2)
                    with col_t:
                        st.markdown("**Price Targets**")
                        st.write(f"High: **${result.target_high:.2f}**" if result.target_high else "High: —")
                        st.write(f"Median: **${result.target_median:.2f}**" if result.target_median else "Median: —")
                        st.write(f"Low: **${result.target_low:.2f}**" if result.target_low else "Low: —")
                    with col_g:
                        st.markdown("**Grade Activity (90d)**")
                        st.metric("Upgrades", result.upgrades_90d)
                        st.metric("Downgrades", result.downgrades_90d)

                    if result.recent_grades:
                        st.markdown(f"**Recent Grade Changes ({len(result.recent_grades)})**")
                        grade_rows = [{"Firm": g.firm, "To": g.to_grade, "From": g.from_grade,
                                       "Action": g.action.title(), "Date": g.date}
                                      for g in result.recent_grades[:10]]
                        df = pd.DataFrame(grade_rows)
                        def _action_color(val):
                            if val == "Upgrade": return "background-color: #0a2a0a; color: #00ff41"
                            if val == "Downgrade": return "background-color: #2a0a0a; color: #ff4141"
                            return ""
                        st.dataframe(df.style.applymap(_action_color, subset=["Action"]), use_container_width=True)

                    col_eps, col_rev = st.columns(2)
                    with col_eps:
                        if result.eps_estimates:
                            st.markdown("**EPS Estimates**")
                            eps_rows = [{"Period": e.period, "Avg": f"${e.avg_estimate:.2f}" if e.avg_estimate else "—",
                                         "Low": f"${e.low_estimate:.2f}" if e.low_estimate else "—",
                                         "High": f"${e.high_estimate:.2f}" if e.high_estimate else "—",
                                         "# Analysts": e.number_of_analysts}
                                        for e in result.eps_estimates]
                            st.dataframe(pd.DataFrame(eps_rows), use_container_width=True)
                    with col_rev:
                        if result.revenue_estimates:
                            st.markdown("**Revenue Estimates**")
                            rev_rows = [{"Period": e.period,
                                         "Avg": f"${e.avg_estimate/1e9:.2f}B" if e.avg_estimate else "—",
                                         "Low": f"${e.low_estimate/1e9:.2f}B" if e.low_estimate else "—",
                                         "High": f"${e.high_estimate/1e9:.2f}B" if e.high_estimate else "—"}
                                        for e in result.revenue_estimates]
                            st.dataframe(pd.DataFrame(rev_rows), use_container_width=True)

                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
                else:
                    from sentinel.sfe.analyst_estimates import screen_analyst_sentiment
                    result = run_async(screen_analyst_sentiment(tickers=tickers, min_upside_pct=min_upside))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tickers", result.tickers_screened)
                    c2.metric("Avg Upside", f"{result.avg_upside_pct:+.1f}%" if result.avg_upside_pct else "—")
                    c3.metric("Strong Buys", len(result.strong_buys))
                    if result.most_upside:
                        st.success(f"Most upside: **{result.most_upside}**")
                    if result.most_downside:
                        st.error(f"Most downside: **{result.most_downside}**")
                    if result.results:
                        rows = [{"Ticker": r.ticker, "Consensus": r.consensus,
                                 "Target": f"${r.target_mean:.2f}" if r.target_mean else "—",
                                 "Upside": f"{r.upside_pct:+.1f}%" if r.upside_pct else "—",
                                 "Analysts": r.num_analysts}
                                for r in result.results]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
            except Exception as e:
                st.error(f"Analyst estimates error: {e}")


# ─── CONV — Convertible Bond Analytics ───────────────────────────────────────

def _render_conv(args: list[str]):
    tickers_in = " ".join(args) if args else ""
    ticker_input = st.text_input("Ticker(s) for screen (or single for detail):", tickers_in or "TSLA NVDA MSTR")
    tickers = [t.strip().upper() for t in ticker_input.split() if t.strip()]
    if not tickers:
        return
    st.markdown("## CONV — Convertible Bond Analytics")
    st.caption("Parity, conversion premium, bond floor, delta/gamma/theta/rho. Free Bloomberg CBOV equivalent.")

    mode = st.radio("Mode", ["Screen (default terms)", "Single detailed"], horizontal=True)

    if mode == "Single detailed" and tickers:
        st.markdown("**CB Terms**")
        col1, col2, col3 = st.columns(3)
        with col1:
            face = st.number_input("Face Value ($)", 100.0, 10000.0, 1000.0, 100.0)
            coupon = st.number_input("Coupon Rate (%)", 0.0, 15.0, 2.5, 0.25) / 100
        with col2:
            maturity = st.number_input("Maturity (years)", 0.25, 30.0, 3.0, 0.25)
            conv_ratio = st.number_input("Conversion Ratio", 1.0, 100.0, 10.0, 1.0)
        with col3:
            yield_ = st.number_input("Straight Bond Yield (%)", 1.0, 20.0, 7.0, 0.25) / 100
            imp_vol = st.number_input("Implied Vol (%)", 5.0, 150.0, 30.0, 5.0) / 100

    if st.button("Analyze Convertibles", key="conv_run"):
        with st.spinner("Computing convertible analytics..."):
            try:
                if mode == "Screen (default terms)":
                    from sentinel.sbx.convertible_bonds import screen_convertibles
                    result = run_async(screen_convertibles(tickers=tickers))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tickers", result.tickers_screened)
                    c2.metric("Avg Delta", f"{result.avg_delta:.2f}" if result.avg_delta else "—")
                    c3.metric("Balanced", len(result.balanced))

                    col_eq, col_bond = st.columns(2)
                    with col_eq:
                        if result.equity_like:
                            st.success(f"Equity-like (δ>0.7): {', '.join(result.equity_like)}")
                    with col_bond:
                        if result.bond_like:
                            st.info(f"Bond-like (δ<0.3): {', '.join(result.bond_like)}")

                    if result.results:
                        rows = [{"Ticker": r.ticker, "Parity": f"${r.parity:.2f}",
                                 "Premium": f"{r.premium_pct:+.1f}%",
                                 "Delta": f"{r.delta:.2f}", "Bond Floor": f"${r.bond_floor:.2f}",
                                 "Verdict": r.verdict.title()}
                                for r in result.results]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
                else:
                    import asyncio
                    from sentinel.sbx.convertible_bonds import ConvertibleTerms, analyze_convertible
                    import yfinance as _yf
                    info = run_async(asyncio.to_thread(lambda: _yf.Ticker(tickers[0]).fast_info))
                    price = getattr(info, "last_price", None) or 100.0
                    terms = ConvertibleTerms(
                        ticker=tickers[0], face_value=face, coupon_rate=coupon,
                        maturity_years=maturity, conversion_ratio=conv_ratio,
                        current_stock_price=price, straight_bond_yield=yield_,
                        implied_vol=imp_vol,
                    )
                    result = analyze_convertible(terms=terms)

                    verdict_color = {"equity-like": "🟢", "balanced": "🟡", "bond-like": "🔵"}.get(result.verdict, "⚪")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Verdict", f"{verdict_color} {result.verdict.title()}")
                    c2.metric("Parity", f"${result.parity:.2f}")
                    c3.metric("Conv. Premium", f"{result.premium_pct:+.1f}%")
                    c4.metric("Bond Floor", f"${result.bond_floor:.2f}")

                    col_v, col_g = st.columns(2)
                    with col_v:
                        st.markdown("**Valuation**")
                        st.metric("Market Price (est.)", f"${result.market_price:.2f}")
                        st.metric("Investment Premium", f"{result.investment_premium_pct:+.1f}%")
                        st.metric("Conversion Price", f"${result.conversion_price:.2f}")
                        st.metric("Breakeven", f"{result.breakeven_years:.1f}y" if result.breakeven_years else "—")
                        st.metric("Annual Coupon", f"${result.coupon_income_annual:.2f}")
                    with col_g:
                        st.markdown("**Greeks**")
                        g = result.greeks
                        st.metric("Delta (δ)", f"{g.delta:.4f}")
                        st.metric("Gamma (γ)", f"{g.gamma:.6f}")
                        st.metric("Theta (θ/yr)", f"{g.theta:.4f}")
                        st.metric("Rho (ρ)", f"{g.rho:.4f}")

                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
            except Exception as e:
                st.error(f"Convertible bond error: {e}")


# ─── OPTFLOW — Options Flow Screener ─────────────────────────────────────────

def _render_optflow(args: list[str]):
    tickers_in = " ".join(args) if args else ""
    ticker_input = st.text_input("Ticker(s) (space-separated for screen):", tickers_in or "SPY AAPL NVDA")
    tickers = [t.strip().upper() for t in ticker_input.split() if t.strip()]
    if not tickers:
        return
    st.markdown("## OPTFLOW — Unusual Options Activity")
    st.caption("Vol/OI spikes, dollar premium, IV surface, max pain, put/call ratios. Free Bloomberg OVDV equivalent.")

    col1, col2, col3 = st.columns(3)
    with col1:
        mode = st.radio("Mode", ["Single ticker", "Screen"], horizontal=True)
    with col2:
        min_score = st.slider("Min unusual score", 0.0, 10.0, 5.0, 0.5)
    with col3:
        max_exp = st.number_input("Max expirations", 1, 6, 3)

    if st.button("Scan Options Flow", key="optflow_run"):
        subject = tickers[0] if mode == "Single ticker" else None
        with st.spinner("Scanning options chains..."):
            try:
                if mode == "Single ticker":
                    from sentinel.sbx.options_flow import get_options_flow
                    result = run_async(get_options_flow(
                        ticker=tickers[0], min_unusual_score=min_score, max_expirations=max_exp
                    ))
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Price", f"${result.current_price:.2f}")
                    sent_color = "🟢" if result.sentiment == "bullish" else ("🔴" if result.sentiment == "bearish" else "🟡")
                    c2.metric("Sentiment", f"{sent_color} {result.sentiment.title()}")
                    c3.metric("P/C Volume", f"{result.pc_volume_ratio:.2f}")
                    c4.metric("P/C OI", f"{result.pc_oi_ratio:.2f}")

                    col_v, col_iv = st.columns(2)
                    with col_v:
                        st.markdown("**Volume**")
                        st.metric("Call Vol", f"{result.total_call_volume:,.0f}")
                        st.metric("Put Vol", f"{result.total_put_volume:,.0f}")
                    with col_iv:
                        st.markdown("**IV Skew**")
                        skew = result.iv_skew
                        st.metric("ATM IV", f"{skew.get('atm', 0):.1%}" if skew else "—")
                        st.metric("25Δ Skew", f"{(skew.get('otm_put',0)-skew.get('otm_call',0)):.1%}" if skew else "—")

                    if result.max_pain:
                        st.metric("Max Pain", f"${result.max_pain:.2f}")

                    if result.unusual_contracts:
                        st.markdown(f"**Unusual Contracts ({len(result.unusual_contracts)})**")
                        rows = [{"Exp": c.expiration, "Strike": f"${c.strike:.0f}",
                                 "Type": c.option_type.upper(), "Score": f"{c.unusual_score:.1f}",
                                 "Vol/OI": f"{c.volume_oi_ratio:.1f}",
                                 "$Premium": f"${c.dollar_premium/1e6:.2f}M" if c.dollar_premium > 1e6 else f"${c.dollar_premium:,.0f}",
                                 "IV": f"{c.implied_volatility:.1%}", "DTE": c.days_to_expiry}
                                for c in result.unusual_contracts[:15]]
                        df = pd.DataFrame(rows)
                        def _score_color(val):
                            try:
                                v = float(val)
                                if v >= 7: return "background-color: #1a2a0a; color: #00ff41"
                                if v >= 5: return "background-color: #2a2a00; color: #ffff00"
                                return ""
                            except Exception:
                                return ""
                        st.dataframe(df.style.applymap(_score_color, subset=["Score"]), use_container_width=True)
                    else:
                        st.info("No contracts above the unusual score threshold.")

                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)

                else:
                    from sentinel.sbx.options_flow import screen_options_flow
                    result = run_async(screen_options_flow(tickers=tickers, min_unusual_score=min_score))
                    st.caption(f"Tickers scanned: {result.tickers_screened} | As of: {result.as_of}")
                    if result.unusual_activity:
                        rows = [{"Ticker": c.ticker if hasattr(c, 'ticker') else "—",
                                 "Exp": c.expiration, "Strike": f"${c.strike:.0f}",
                                 "Type": c.option_type.upper(), "Score": f"{c.unusual_score:.1f}",
                                 "$Premium": f"${c.dollar_premium/1e6:.2f}M" if c.dollar_premium > 1e6 else f"${c.dollar_premium:,.0f}",
                                 "IV": f"{c.implied_volatility:.1%}"}
                                for c in result.unusual_activity[:20]]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    else:
                        st.info("No unusual activity detected above threshold.")
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
            except Exception as e:
                st.error(f"Options flow error: {e}")


# ─── EARNLP — Earnings 8-K NLP ────────────────────────────────────────────────

def _render_earnlp(args: list[str]):
    ticker = args[0].upper() if args else st.text_input("Ticker:", "AAPL").upper()
    if not ticker:
        return
    st.markdown(f"## EARNLP — Earnings Transcript NLP: {ticker}")
    st.caption("EDGAR 8-K text extraction + Claude Haiku structured analysis: tone, guidance, themes, sentiment score.")

    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio("Mode", ["Latest filing", "Trend (multi-quarter)"], horizontal=True)
    with col2:
        quarters = st.number_input("Quarters (trend mode)", 2, 8, 4)

    if st.button("Analyze Earnings", key="earnlp_run"):
        with st.spinner(f"Fetching and analyzing {ticker} earnings filings..."):
            try:
                if mode == "Latest filing":
                    from sentinel.sil.earnings_nlp import analyze_earnings_filing
                    result = run_async(analyze_earnings_filing(ticker=ticker))
                    tone_color = {"bullish": "🟢", "neutral": "🟡", "bearish": "🔴"}.get(result.tone, "⚪")
                    guide_color = {"raised": "🟢", "maintained": "🟡", "lowered": "🔴", "withdrawn": "🔴"}.get(result.guidance_signal, "⚪")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Tone", f"{tone_color} {result.tone.title()}")
                    c2.metric("Guidance", f"{guide_color} {result.guidance_signal.title()}")
                    c3.metric("Sentiment", f"{result.sentiment_score:+.2f}")
                    c4.metric("Period", result.period_of_report or "—")

                    col_rev, col_prof = st.columns(2)
                    with col_rev:
                        st.markdown(f"**Revenue Signal:** {result.revenue_signal or '—'}")
                    with col_prof:
                        st.markdown(f"**Profit Signal:** {result.profit_signal or '—'}")

                    if result.key_themes:
                        st.markdown("**Key Themes**")
                        for t in result.key_themes:
                            st.markdown(f"- {t}")
                    col_risk, col_cat = st.columns(2)
                    with col_risk:
                        if result.risks:
                            st.markdown("**Risks**")
                            for r in result.risks:
                                st.markdown(f"- {r}")
                    with col_cat:
                        if result.catalysts:
                            st.markdown("**Catalysts**")
                            for c in result.catalysts:
                                st.markdown(f"- {c}")
                    if result.text_excerpt:
                        with st.expander("Text excerpt"):
                            st.text(result.text_excerpt[:800])
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
                else:
                    from sentinel.sil.earnings_nlp import get_earnings_trend
                    result = run_async(get_earnings_trend(ticker=ticker, quarters=quarters))
                    trend_color = {"improving": "🟢", "stable": "🟡", "deteriorating": "🔴"}.get(result.sentiment_trend, "⚪")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Trend", f"{trend_color} {result.sentiment_trend.title()}")
                    c2.metric("Avg Sentiment", f"{result.avg_sentiment:+.2f}")
                    c3.metric("Latest Tone", result.latest_tone.title() if result.latest_tone else "—")

                    if result.analyses:
                        rows = [{"Quarter": a.period_of_report or a.file_date,
                                 "Tone": a.tone.title(), "Guidance": a.guidance_signal.title(),
                                 "Score": f"{a.sentiment_score:+.2f}",
                                 "Revenue": a.revenue_signal or "—", "Profit": a.profit_signal or "—"}
                                for a in result.analyses]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    if result.guidance_changes:
                        st.markdown("**Guidance Changes**")
                        for gc in result.guidance_changes:
                            st.markdown(f"- {gc}")
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
            except Exception as e:
                st.error(f"Earnings NLP error: {e}")


# ─── CREDIT — Credit Analytics (Merton Model) ────────────────────────────────

def _render_credit(args: list[str]):
    tickers_in = " ".join(args) if args else ""
    ticker_input = st.text_input("Ticker(s) (space-separated for screen):", tickers_in or "AAPL")
    tickers = [t.strip().upper() for t in ticker_input.split() if t.strip()]
    if not tickers:
        return
    st.markdown("## CREDIT — Credit Analytics")
    st.caption("Merton structural model: asset value, distance-to-default, PD, CDS spread proxy, Altman Z. Free Bloomberg CRPR.")

    mode = st.radio("Mode", ["Single ticker", "Screen"], horizontal=True)

    if st.button("Run Credit Analysis", key="credit_run"):
        with st.spinner("Running Merton model..."):
            try:
                if mode == "Single ticker":
                    from sentinel.spr.credit_analytics import get_credit_analytics
                    result = run_async(get_credit_analytics(ticker=tickers[0]))
                    tier_color = {"AAA/AA": "🟢", "A/BBB": "🟢", "BB/B": "🟡", "CCC/CC": "🔴", "D": "🔴"}.get(result.credit_tier, "⚪")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Credit Score", f"{result.credit_score:.1f}/10")
                    c2.metric("Tier", f"{tier_color} {result.credit_tier}")
                    c3.metric("Verdict", result.verdict.title())
                    c4.metric("As of", result.as_of)

                    col_m, col_f = st.columns(2)
                    with col_m:
                        st.markdown("**Merton Model**")
                        m = result.merton
                        if m:
                            st.metric("Distance-to-Default", f"{m.distance_to_default:.2f}σ")
                            st.metric("Probability of Default", f"{m.probability_of_default:.2%}")
                            st.metric("CDS Proxy", f"{m.cds_spread_bps:.0f} bps")
                            st.metric("Asset Vol", f"{m.asset_vol:.1%}")
                            conv_icon = "✓" if m.converged else "⚠"
                            st.caption(f"{conv_icon} Merton convergence: {'yes' if m.converged else 'no'}")
                    with col_f:
                        st.markdown("**Credit Factors**")
                        f = result.factors
                        if f:
                            st.metric("Debt/Equity", f"{f.debt_to_equity:.2f}x")
                            st.metric("Interest Coverage", f"{f.interest_coverage:.1f}x")
                            st.metric("Current Ratio", f"{f.current_ratio:.2f}x")
                            st.metric("Altman Z-Score", f"{f.altman_z:.2f}" if f.altman_z else "—")
                            fcf_icon = "✓" if f.fcf_positive else "✗"
                            st.caption(f"{fcf_icon} FCF: {'positive' if f.fcf_positive else 'negative'}")

                    col3, col4 = st.columns(2)
                    with col3:
                        if result.market_cap_mm:
                            st.metric("Market Cap", f"${result.market_cap_mm:,.0f}M")
                        if result.total_debt_mm:
                            st.metric("Total Debt", f"${result.total_debt_mm:,.0f}M")
                    with col4:
                        if result.ev_mm:
                            st.metric("EV", f"${result.ev_mm:,.0f}M")
                        if result.equity_vol:
                            st.metric("Equity Vol", f"{result.equity_vol:.1%}")

                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)

                else:
                    from sentinel.spr.credit_analytics import screen_credit
                    result = run_async(screen_credit(tickers=tickers))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg Score", f"{result.avg_credit_score:.1f}/10")
                    c2.metric("Strongest", result.strongest_credit or "—")
                    c3.metric("Distressed", result.distressed_count)

                    if result.results:
                        rows = [{"Ticker": r.ticker, "Score": f"{r.credit_score:.1f}",
                                 "Tier": r.credit_tier, "Verdict": r.verdict.title(),
                                 "PD": f"{r.merton.probability_of_default:.2%}" if r.merton else "—",
                                 "CDS (bps)": f"{r.merton.cds_spread_bps:.0f}" if r.merton else "—",
                                 "D/E": f"{r.factors.debt_to_equity:.2f}" if r.factors else "—",
                                 "Z-Score": f"{r.factors.altman_z:.2f}" if (r.factors and r.factors.altman_z) else "—"}
                                for r in result.results]
                        df = pd.DataFrame(rows)
                        def _tier_color(val):
                            if val in ("AAA/AA", "A/BBB"): return "background-color: #0a2a0a; color: #00ff41"
                            if val in ("BB/B"): return "background-color: #2a2a00; color: #ffff00"
                            if val in ("CCC/CC", "D"): return "background-color: #2a0a0a; color: #ff4141"
                            return ""
                        st.dataframe(df.style.applymap(_tier_color, subset=["Tier"]), use_container_width=True)

                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
            except Exception as e:
                st.error(f"Credit analytics error: {e}")


# ─── PEERS — Auto Peer Comparison ─────────────────────────────────────────────

def _render_peers(args: list[str]):
    ticker = args[0].upper() if args else st.text_input("Subject ticker:", "NVDA").upper()
    if not ticker:
        return
    custom_peers_raw = " ".join(args[1:]) if len(args) > 1 else ""
    st.markdown(f"## PEERS — Peer Comparison: {ticker}")
    st.caption("Industry-based auto-discovery, 17 metrics (valuation/growth/profitability/health), percentile ranking. Free CapIQ comps.")

    col1, col2 = st.columns(2)
    with col1:
        custom_input = st.text_input("Custom peers (optional, space-separated):", custom_peers_raw)
    with col2:
        max_peers = st.number_input("Max peers", 3, 12, 7)

    if st.button("Compare Peers", key="peers_run"):
        with st.spinner(f"Fetching peer data for {ticker}..."):
            try:
                from sentinel.sfe.peer_comparison import get_peer_comparison
                custom_peers = [p.strip().upper() for p in custom_input.split() if p.strip()] or None
                result = run_async(get_peer_comparison(ticker=ticker, custom_peers=custom_peers, max_peers=max_peers))

                pct = result.overall_percentile
                pct_color = "🟢" if pct >= 60 else ("🔴" if pct <= 40 else "🟡")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Sector", result.sector or "—")
                c2.metric("Industry", result.industry or "—")
                c3.metric("Peers", len(result.peers))
                c4.metric("Overall Percentile", f"{pct_color} {pct:.0f}th")

                st.markdown(f"**Verdict:** {result.verdict}")

                if result.metrics_table:
                    st.markdown("**Metrics Comparison**")
                    rows = []
                    for pm in result.metrics_table:
                        row = {"Ticker": pm.ticker, "Name": (pm.company_name or "")[:25]}
                        if pm.pe_ratio: row["P/E"] = f"{pm.pe_ratio:.1f}x"
                        if pm.forward_pe: row["Fwd P/E"] = f"{pm.forward_pe:.1f}x"
                        if pm.ev_ebitda: row["EV/EBITDA"] = f"{pm.ev_ebitda:.1f}x"
                        if pm.revenue_growth_pct: row["Rev Growth"] = f"{pm.revenue_growth_pct:.1f}%"
                        if pm.gross_margin_pct: row["Gross Mgn"] = f"{pm.gross_margin_pct:.1f}%"
                        if pm.operating_margin_pct: row["Op Mgn"] = f"{pm.operating_margin_pct:.1f}%"
                        if pm.roe_pct: row["ROE"] = f"{pm.roe_pct:.1f}%"
                        if pm.debt_to_equity: row["D/E"] = f"{pm.debt_to_equity:.2f}x"
                        rows.append(row)
                    df = pd.DataFrame(rows)
                    def _highlight_subject(row):
                        return ["background-color: #0d1f2d; font-weight: bold"] * len(row) if row.get("Ticker") == ticker else [""] * len(row)
                    st.dataframe(df.style.apply(_highlight_subject, axis=1), use_container_width=True)

                if result.rankings:
                    st.markdown("**Rankings (1 = best)**")
                    rank_rows = [{"Metric": r.metric, "Rank": f"{r.subject_rank}/{r.total_peers}",
                                  "Percentile": f"{r.percentile:.0f}th",
                                  "Best": r.best_ticker, "Worst": r.worst_ticker}
                                 for r in sorted(result.rankings, key=lambda x: x.percentile, reverse=True)]
                    rdf = pd.DataFrame(rank_rows)
                    def _rank_pct_color(val):
                        try:
                            v = float(str(val).replace("th", ""))
                            if v >= 70: return "background-color: #0a2a0a; color: #00ff41"
                            if v <= 30: return "background-color: #2a0a0a; color: #ff4141"
                            return ""
                        except Exception:
                            return ""
                    st.dataframe(rdf.style.applymap(_rank_pct_color, subset=["Percentile"]), use_container_width=True)

                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
            except Exception as e:
                st.error(f"Peer comparison error: {e}")


# ─── SECROT — Sector Rotation Heatmap ────────────────────────────────────────

def _render_secrot(args: list[str]):
    st.markdown("## SECROT — Sector Rotation Heatmap")
    st.caption("11 SPDR sector ETFs vs SPY: relative strength 1M/3M/6M/12M, composite momentum score, regime, rotation signal.")

    col1, col2 = st.columns(2)
    with col1:
        history_days = st.number_input("History days", 63, 504, 252)
    with col2:
        top_n = st.number_input("Top/bottom N sectors", 1, 5, 3)

    mode = st.radio("Mode", ["Heatmap", "Strength Screen"], horizontal=True)

    if st.button("Run Sector Rotation", key="secrot_run"):
        with st.spinner("Computing sector relative strength vs SPY..."):
            try:
                if mode == "Heatmap":
                    from sentinel.sma.sector_rotation import get_sector_rotation
                    result = run_async(get_sector_rotation(history_days=history_days, top_n=top_n))

                    regime_color = {"risk_on": "🟢", "defensive": "🔴", "risk_off": "🔴",
                                    "neutral": "🟡", "late_cycle": "🟡"}.get(result.market_regime, "⚪")
                    c1, c2 = st.columns(2)
                    c1.metric("Market Regime", f"{regime_color} {result.market_regime.replace('_', ' ').title()}")
                    c2.markdown(f"**Rotation Signal:** {result.rotation_signal}")

                    if result.top_sectors:
                        st.success(f"Overweight: {' | '.join(result.top_sectors)}")
                    if result.bottom_sectors:
                        st.error(f"Underweight: {' | '.join(result.bottom_sectors)}")

                    if result.sectors:
                        st.markdown("**Sector Momentum Heatmap**")
                        rows = []
                        for s in sorted(result.sectors, key=lambda x: x.momentum_score, reverse=True):
                            rows.append({
                                "Sector": s.name,
                                "Ticker": s.ticker,
                                "Score": f"{s.momentum_score:.1f}",
                                "RS 1M": f"{s.rs_1m:+.1%}" if s.rs_1m is not None else "—",
                                "RS 3M": f"{s.rs_3m:+.1%}" if s.rs_3m is not None else "—",
                                "RS 6M": f"{s.rs_6m:+.1%}" if s.rs_6m is not None else "—",
                                "Trend": s.trend.title() if s.trend else "—",
                                "Rank": s.rank_composite if s.rank_composite is not None else "—",
                            })
                        def _score_color(val):
                            try:
                                v = float(str(val))
                                if v >= 7: return "background-color: #0a2a0a; color: #00ff41"
                                if v <= 3: return "background-color: #2a0a0a; color: #ff4141"
                                return ""
                            except Exception:
                                return ""
                        df = pd.DataFrame(rows)
                        st.dataframe(df.style.applymap(_score_color, subset=["Score"]), use_container_width=True)
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
                else:
                    from sentinel.sma.sector_rotation import screen_sector_strength
                    result = run_async(screen_sector_strength(top_n=top_n, history_days=history_days))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Market Regime", result.market_regime.replace("_", " ").title() if hasattr(result, "market_regime") else "—")
                    c2.metric("Market Breadth", f"{result.market_breadth:.0%}" if result.market_breadth is not None else "—")
                    c3.metric("Avg Momentum", f"{result.avg_momentum:.1f}/10" if result.avg_momentum is not None else "—")
                    if result.results:
                        rows = [{"Ticker": r.ticker, "Sector": r.name,
                                 "Score": f"{r.momentum_score:.1f}",
                                 "Rank": r.rank_composite,
                                 "Trend": r.trend.title() if r.trend else "—"}
                                for r in result.results]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
            except Exception as e:
                st.error(f"Sector rotation error: {e}")


# ─── EQSCORE — Earnings Quality Score ─────────────────────────────────────────

def _render_eqscore(args: list[str]):
    tickers_in = " ".join(args) if args else ""
    ticker_input = st.text_input("Ticker(s) (space-separated for screen):", tickers_in or "AAPL")
    tickers = [t.strip().upper() for t in ticker_input.split() if t.strip()]
    if not tickers:
        return
    st.markdown("## EQSCORE — Earnings Quality Score")
    st.caption("EDGAR XBRL: Sloan accruals ratio, cash conversion (CFO/NI), quality score 0-10 and tier. Free FactSet Earnings Quality equivalent.")

    mode = st.radio("Mode", ["Single ticker", "Screen"], horizontal=True)
    col1, col2 = st.columns(2)
    with col1:
        years = st.number_input("Years of history", 2, 10, 5)
    with col2:
        min_score = st.slider("Min quality score (screen)", 0.0, 10.0, 5.0, 0.5)

    if st.button("Analyze Earnings Quality", key="eqscore_run"):
        with st.spinner("Fetching EDGAR XBRL accruals data..."):
            try:
                if mode == "Single ticker":
                    from sentinel.sfe.earnings_quality import get_earnings_quality
                    result = run_async(get_earnings_quality(ticker=tickers[0], years=years))

                    tier_color = {"aaa quality": "🟢", "high quality": "🟢", "moderate quality": "🟡",
                                  "low quality": "🔴", "very low quality": "🔴"}.get(result.quality_tier.lower(), "⚪")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Quality Score", f"{result.quality_score:.1f}/10")
                    c2.metric("Quality Tier", f"{tier_color} {result.quality_tier}")
                    c3.metric("Avg Accruals", f"{result.avg_accruals_ratio:+.3f}" if result.avg_accruals_ratio is not None else "—")
                    c4.metric("Cash Conversion", f"{result.avg_cash_conversion:.2f}x" if result.avg_cash_conversion is not None else "—")

                    trend_color = {"improving": "🟢", "stable": "🟡", "deteriorating": "🔴"}.get(result.trend, "⚪")
                    st.metric("Trend", f"{trend_color} {result.trend.title()}")
                    st.metric("Data Quality", result.data_quality.title())
                    st.metric("Periods Analyzed", len(result.periods))

                    if result.periods:
                        st.markdown("**Annual Earnings Quality History**")
                        rows = [{"Year": p.fiscal_year,
                                 "Accruals Ratio": f"{p.accruals_ratio:+.3f}" if p.accruals_ratio is not None else "—",
                                 "Cash Conversion": f"{p.cash_conversion:.2f}x" if p.cash_conversion is not None else "—",
                                 "Sloan Ratio": f"{p.sloan_ratio:+.3f}" if p.sloan_ratio is not None else "—",
                                 "Op Leverage": f"{p.operating_leverage:.2f}x" if p.operating_leverage is not None else "—",
                                 "Flag": p.quality_flag.title() if p.quality_flag else "—"}
                                for p in result.periods]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

                    if result.risk_flags:
                        st.markdown("**Risk Flags**")
                        for flag in result.risk_flags:
                            st.warning(f"⚠ {flag}")
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
                else:
                    from sentinel.sfe.earnings_quality import screen_earnings_quality
                    result = run_async(screen_earnings_quality(tickers=tickers, min_quality_score=min_score))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tickers Screened", result.tickers_screened)
                    c2.metric("High Quality", result.high_quality_count)
                    c3.metric("Low Quality", result.low_quality_count)
                    if result.best_quality:
                        st.success(f"Best quality: **{result.best_quality}**")
                    if result.worst_quality:
                        st.error(f"Worst quality: **{result.worst_quality}**")
                    if result.results:
                        rows = [{"Ticker": r.ticker,
                                 "Score": f"{r.quality_score:.1f}",
                                 "Tier": r.quality_tier,
                                 "Accruals": f"{r.avg_accruals_ratio:+.3f}" if r.avg_accruals_ratio is not None else "—",
                                 "Cash Conv.": f"{r.avg_cash_conversion:.2f}x" if r.avg_cash_conversion is not None else "—",
                                 "Trend": r.trend.title()}
                                for r in result.results]
                        def _eq_color(val):
                            try:
                                v = float(str(val))
                                if v >= 7: return "background-color: #0a2a0a; color: #00ff41"
                                if v <= 3: return "background-color: #2a0a0a; color: #ff4141"
                                return ""
                            except Exception:
                                return ""
                        df = pd.DataFrame(rows)
                        st.dataframe(df.style.applymap(_eq_color, subset=["Score"]), use_container_width=True)
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
            except Exception as e:
                st.error(f"Earnings quality error: {e}")


# ─── IVTERM — IV Term Structure ────────────────────────────────────────────────

def _render_ivterm(args: list[str]):
    tickers_in = " ".join(args) if args else ""
    ticker_input = st.text_input("Ticker(s) (space-separated for screen):", tickers_in or "SPY")
    tickers = [t.strip().upper() for t in ticker_input.split() if t.strip()]
    if not tickers:
        return
    st.markdown("## IVTERM — Implied Volatility Term Structure")
    st.caption("Multi-expiration ATM IV, forward vol, 25-delta skew, put/call skew, contango/backwardation flag.")

    mode = st.radio("Mode", ["Single ticker", "Vol surface screen"], horizontal=True)

    if st.button("Run IV Term Structure", key="ivterm_run"):
        with st.spinner("Fetching options chain..."):
            try:
                if mode == "Single ticker":
                    from sentinel.sbx.vol_term_structure import get_vol_term_structure
                    result = run_async(get_vol_term_structure(ticker=tickers[0]))

                    contango_icon = "📈 Contango (normal)" if result.contango else "📉 Backwardation (stress)"
                    slope_str = f"{result.term_slope:+.4f}" if result.term_slope is not None else "—"
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Spot", f"${result.spot:.2f}" if result.spot else "—")
                    c2.markdown(f"**Structure:** {contango_icon}")
                    c3.metric("Term Slope", slope_str)

                    if result.skew_summary:
                        skew = result.skew_summary
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            avg25 = skew.get("avg_25d_skew")
                            st.metric("Avg 25-Delta Skew", f"{avg25:+.1%}" if avg25 is not None else "—")
                        with col_s2:
                            st.metric("Skew Regime", skew.get("skew_regime", "—").replace("_", " ").title())

                    if result.slices:
                        st.markdown("**IV Term Structure by Expiration**")
                        rows = [{"Expiration": s.expiration, "DTE": s.dte,
                                 "ATM IV": f"{s.atm_iv:.1%}" if s.atm_iv is not None else "—",
                                 "25D Skew": f"{s.iv_skew_25d:+.1%}" if s.iv_skew_25d is not None else "—",
                                 "10D Skew": f"{s.iv_skew_10d:+.1%}" if s.iv_skew_10d is not None else "—",
                                 "P/C Skew": f"{s.put_call_skew:+.1%}" if s.put_call_skew is not None else "—",
                                 "Vol-of-Vol": f"{s.vol_of_vol:.1%}" if s.vol_of_vol is not None else "—"}
                                for s in result.slices]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

                    if result.forward_vols:
                        st.markdown("**Forward Volatility**")
                        fwd_rows = [{"From": f.from_tenor, "To": f.to_tenor,
                                     "Forward Vol": f"{f.forward_vol:.1%}" if f.forward_vol is not None else "—"}
                                    for f in result.forward_vols]
                        st.dataframe(pd.DataFrame(fwd_rows), use_container_width=True)
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
                else:
                    from sentinel.sbx.vol_term_structure import screen_vol_surface
                    result = run_async(screen_vol_surface(tickers=tickers))
                    c1, c2 = st.columns(2)
                    c1.metric("Tickers Screened", len(result.tickers_screened))
                    c2.metric("As Of", result.as_of)
                    if result.elevated_skew:
                        st.warning(f"Elevated downside skew: {' | '.join(result.elevated_skew)}")
                    if result.inverted_term_structure:
                        st.error(f"Inverted term structure (stress): {' | '.join(result.inverted_term_structure)}")
                    if result.results:
                        rows = [{"Ticker": tkr,
                                 "Spot": f"${r.spot:.2f}" if r.spot else "—",
                                 "Front ATM IV": f"{r.atm_iv_front:.1%}" if r.atm_iv_front is not None else "—",
                                 "Back ATM IV": f"{r.atm_iv_back:.1%}" if r.atm_iv_back is not None else "—",
                                 "Term Slope": f"{r.term_slope:+.4f}" if r.term_slope is not None else "—",
                                 "Skew Regime": r.skew_regime.replace("_", " ").title() if r.skew_regime else "—",
                                 "IV Pct": f"{r.iv_percentile:.0f}%" if r.iv_percentile is not None else "—"}
                                for tkr, r in result.results.items()]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
            except Exception as e:
                st.error(f"IV term structure error: {e}")


# ─── NOWCAST — GDP Nowcast Dashboard ──────────────────────────────────────────

def _render_nowcast(args: list[str]):
    st.markdown("## NOWCAST — GDP Nowcast & Macro Leading Indicators")
    st.caption("10-component FRED nowcast: industrial production, payrolls, retail sales, yield curve, jobless claims, consumer sentiment. Recession probability estimate.")

    history_months = st.number_input("History months for standardization", 12, 60, 24)
    mode = st.radio("Mode", ["Full Dashboard", "Quick Nowcast"], horizontal=True)

    if st.button("Run Macro Nowcast", key="nowcast_run"):
        with st.spinner("Fetching FRED leading indicators..."):
            try:
                if mode == "Full Dashboard":
                    from sentinel.spr.macro_nowcast import get_macro_nowcast_dashboard
                    result = run_async(get_macro_nowcast_dashboard(history_months=history_months))

                    regime_color = {"expansion": "🟢", "recovery": "🟢", "late_cycle": "🟡",
                                    "contraction": "🔴"}.get(result.regime, "⚪")
                    c1, c2, c3, c4 = st.columns(4)
                    nowcast = result.nowcast
                    c1.metric("GDP Nowcast", f"{nowcast.nowcast_gdp_growth:+.1f}%" if nowcast.nowcast_gdp_growth is not None else "—")
                    c2.metric("Composite Index", f"{nowcast.composite_index:.1f}/10" if nowcast.composite_index is not None else "—")
                    c3.metric("Recession Prob.", f"{nowcast.recession_probability:.0%}" if nowcast.recession_probability is not None else "—")
                    c4.metric("Regime", f"{regime_color} {result.regime.replace('_', ' ').title()}")

                    if result.expansion_signals:
                        st.success(f"Expansion signals: {' | '.join(result.expansion_signals)}")
                    if result.contraction_signals:
                        st.error(f"Contraction signals: {' | '.join(result.contraction_signals)}")
                    if result.key_risks:
                        st.markdown("**Key Macro Risks**")
                        for r in result.key_risks:
                            st.warning(f"⚠ {r}")

                    if result.series_readings:
                        st.markdown("**Leading Indicator Readings**")
                        rows = [{"Series": s.series_id, "Name": s.name,
                                 "Latest": f"{s.latest_value:.2f}" if s.latest_value is not None else "—",
                                 "MoM Δ": f"{s.mom_change:+.2%}" if s.mom_change is not None else "—",
                                 "YoY Δ": f"{s.yoy_change:+.2%}" if s.yoy_change is not None else "—",
                                 "Trend": s.trend.title()}
                                for s in result.series_readings]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    if result.nowcast.warnings:
                        for w in result.nowcast.warnings:
                            st.warning(w)
                else:
                    from sentinel.spr.macro_nowcast import get_gdp_nowcast
                    result = run_async(get_gdp_nowcast(history_months=history_months))

                    c1, c2, c3 = st.columns(3)
                    c1.metric("GDP Nowcast", f"{result.nowcast_gdp_growth:+.1f}%" if result.nowcast_gdp_growth is not None else "—")
                    c2.metric("Composite Index", f"{result.composite_index:.1f}/10" if result.composite_index is not None else "—")
                    c3.metric("Recession Prob.", f"{result.recession_probability:.0%}" if result.recession_probability is not None else "—")
                    st.metric("Confidence", result.nowcast_confidence.title() if result.nowcast_confidence else "—")

                    if result.components:
                        st.markdown("**Nowcast Components**")
                        rows = [{"Series": c.series_id, "Name": c.name,
                                 "Weight": f"{c.weight:+.2f}",
                                 "Z-Score": f"{c.standardized_value:+.2f}" if c.standardized_value is not None else "—",
                                 "Contribution": f"{c.contribution:+.2f}" if c.contribution is not None else "—"}
                                for c in result.components]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
            except Exception as e:
                st.error(f"Macro nowcast error: {e}")


# ─── VIXTS — VIX Term Structure & Volatility Regime ─────────────────────────

def _render_vixts(args: list[str]):
    st.markdown("## VIXTS — VIX Term Structure & Volatility Regime")
    st.caption("VIX spot/3M/6M/1Y term structure, VVIX, CBOE SKEW, volatility risk premium (VRP), vol regime, contango/backwardation flag.")

    col1, col2 = st.columns(2)
    with col1:
        history_days = st.number_input("History days", 63, 756, 252)
    with col2:
        tickers_vol = st.text_input("Tickers for realized vol (optional):", "SPY QQQ IWM")

    mode = st.radio("Mode", ["VIX Analytics", "Vol Regime Dashboard"], horizontal=True)

    if st.button("Run VIX Analysis", key="vixts_run"):
        with st.spinner("Fetching VIX term structure from yfinance..."):
            try:
                if mode == "VIX Analytics":
                    from sentinel.sma.vix_analytics import get_vix_analytics
                    result = run_async(get_vix_analytics(history_days=history_days))

                    regime_color = {"very_low": "🟢", "low": "🟢", "moderate": "🟡",
                                    "elevated": "🔴", "extreme": "🔴"}.get(result.vol_regime, "⚪")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("VIX Spot", f"{result.spot_vix:.2f}" if result.spot_vix else "—")
                    c2.metric("Regime", f"{regime_color} {result.vol_regime.replace('_', ' ').title()}")
                    c3.metric("VIX Percentile", f"{result.vix_percentile:.0f}%" if result.vix_percentile else "—")
                    c4.metric("VRP", f"{result.vrp:+.2f}" if result.vrp is not None else "—")

                    col_term, col_misc = st.columns(2)
                    with col_term:
                        st.markdown("**Term Structure**")
                        if result.vix3m is not None:
                            st.metric("VIX3M", f"{result.vix3m:.2f}")
                        if result.vix6m is not None:
                            st.metric("VIX6M", f"{result.vix6m:.2f}")
                        if result.vix1y is not None:
                            st.metric("VIX1Y", f"{result.vix1y:.2f}")
                        if result.contango is not None:
                            contango_icon = "📈 Contango (normal)" if result.contango else "📉 Backwardation (stress)"
                            st.markdown(f"**Structure:** {contango_icon}")
                        if result.term_slope_pct is not None:
                            st.metric("Term Slope %", f"{result.term_slope_pct:+.1f}%")
                    with col_misc:
                        st.markdown("**Sentiment Proxies**")
                        if result.vvix is not None:
                            st.metric("VVIX (Vol-of-Vol)", f"{result.vvix:.2f}")
                        if result.skew_index is not None:
                            st.metric("CBOE SKEW", f"{result.skew_index:.2f}")
                        st.metric("Mean Reversion", result.mean_reversion_signal.title())
                        st.metric("VVIX Regime", result.vvix_regime.title())
                        if result.realized_vol_30d_spy is not None:
                            st.metric("SPY RVol 30d", f"{result.realized_vol_30d_spy:.1%}")

                    if result.history_60d:
                        st.markdown("**60-Day VIX History**")
                        hist_rows = [{"Date": h.date, "VIX": h.vix,
                                      "VIX3M": h.vix3m, "Term Slope": h.term_slope}
                                     for h in result.history_60d[-30:]]
                        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True)
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
                else:
                    from sentinel.sma.vix_analytics import get_vol_regime
                    tickers_list = [t.strip().upper() for t in tickers_vol.split() if t.strip()] or None
                    result = run_async(get_vol_regime(tickers=tickers_list, history_days=history_days))

                    regime_color = {"very_low": "🟢", "low": "🟢", "moderate": "🟡",
                                    "elevated": "🔴", "extreme": "🔴"}.get(result.market_regime, "⚪")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Market Regime", f"{regime_color} {result.market_regime.replace('_', ' ').title()}")
                    c2.metric("VIX", f"{result.spot_vix:.2f}" if result.spot_vix else "—")
                    c3.metric("Fear/Greed Proxy", f"{result.fear_greed_proxy:.0f}/100" if result.fear_greed_proxy is not None else "—")

                    if result.ticker_vols:
                        st.markdown("**Per-Ticker Realized Volatility**")
                        rows = [{"Ticker": v.ticker,
                                 "RVol 20d": f"{v.realized_vol_20d:.1%}" if v.realized_vol_20d else "—",
                                 "RVol 60d": f"{v.realized_vol_60d:.1%}" if v.realized_vol_60d else "—",
                                 "IV/RV Premium": f"{v.iv_rv_premium:+.1%}" if v.iv_rv_premium is not None else "—"}
                                for v in result.ticker_vols]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
            except Exception as e:
                st.error(f"VIX analytics error: {e}")


# ─── INSIG — Insider Cluster Signal ──────────────────────────────────────────

def _render_insig(args: list[str]):
    tickers_in = " ".join(args) if args else ""
    ticker_input = st.text_input("Ticker(s) (space-separated for screen):", tickers_in or "AAPL")
    tickers = [t.strip().upper() for t in ticker_input.split() if t.strip()]
    if not tickers:
        return
    st.markdown("## INSIG — Insider Cluster Signal")
    st.caption("Aggregate Form 4 insider trading: cluster buy detection (≥2 distinct insiders), officer/director sentiment, net purchase ratio, score 0-10.")

    mode = st.radio("Mode", ["Single ticker", "Screen"], horizontal=True)
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.number_input("Days of history", 30, 365, 180)
    with col2:
        min_score = st.slider("Min signal score (screen)", 0.0, 10.0, 5.0, 0.5)

    if st.button("Analyze Insider Signal", key="insig_run"):
        with st.spinner("Fetching insider transactions..."):
            try:
                if mode == "Single ticker":
                    from sentinel.sfe.insider_signal import get_insider_signal
                    result = run_async(get_insider_signal(ticker=tickers[0], days_back=days_back))

                    verdict_color = {"strong buy signal": "🟢", "moderate buy signal": "🟢",
                                     "neutral": "🟡", "sell signal": "🔴"}.get(result.signal_verdict.lower(), "⚪")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Signal Score", f"{result.signal_score:.1f}/10")
                    c2.metric("Verdict", f"{verdict_color} {result.signal_verdict.title()}")
                    c3.metric("Transactions", result.total_transactions)
                    c4.metric("Net Purchase Ratio", f"{result.net_purchase_ratio:.0%}" if result.net_purchase_ratio is not None else "—")

                    col_detail, col_meta = st.columns(2)
                    with col_detail:
                        st.markdown("**Signal Components**")
                        cluster_icon = "🟢 YES" if result.cluster_buy else "🔴 No"
                        st.markdown(f"**Cluster Buy (≥2 insiders):** {cluster_icon}")
                        officer_icon = {"buying": "🟢 Buying", "selling": "🔴 Selling"}.get(result.officer_sentiment.lower(), "🟡 Neutral")
                        st.markdown(f"**Officer Sentiment:** {officer_icon}")
                        if result.unusual_size_buy:
                            st.info("⚡ Unusual transaction size detected")
                        if result.recent_momentum_positive is not None:
                            mom_icon = "🟢 Positive" if result.recent_momentum_positive else "🔴 Negative"
                            st.markdown(f"**Recent Momentum:** {mom_icon}")
                        if result.score_components:
                            st.markdown("**Score Components**")
                            comp_rows = [{"Component": k, "Points": f"{v:+.1f}"} for k, v in result.score_components.items()]
                            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)
                    with col_meta:
                        st.markdown("**Activity Summary**")
                        st.metric("Buy Transactions", result.buy_transactions)
                        st.metric("Sell Transactions", result.sell_transactions)
                        st.metric("Total Buy Value", f"${result.total_buy_value_usd:,.0f}")
                        st.metric("Total Sell Value", f"${result.total_sell_value_usd:,.0f}")

                    if result.recent_transactions:
                        st.markdown("**Recent Insider Transactions**")
                        rows = [{"Date": t.date, "Insider": t.insider_name,
                                 "Role": t.position, "Type": t.transaction_type,
                                 "Shares": f"{t.shares:,}" if t.shares else "—",
                                 "Value": f"${t.value_usd:,.0f}" if t.value_usd else "—"}
                                for t in result.recent_transactions[:15]]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
                else:
                    from sentinel.sfe.insider_signal import screen_insider_buying
                    result = run_async(screen_insider_buying(tickers=tickers, min_signal_score=min_score))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tickers Screened", result.tickers_screened)
                    c2.metric("Avg Score", f"{result.avg_signal_score:.1f}/10" if result.avg_signal_score is not None else "—")
                    c3.metric("Cluster Buys", len(result.cluster_buys))
                    if result.top_signals:
                        st.success(f"Top signals: {' | '.join(result.top_signals)}")
                    if result.cluster_buys:
                        st.info(f"Cluster buys: {' | '.join(result.cluster_buys)}")
                    if result.officer_buyers:
                        st.info(f"Officer buyers: {' | '.join(result.officer_buyers)}")
                    if result.results:
                        rows = [{"Ticker": r.ticker, "Score": f"{r.signal_score:.1f}",
                                 "Verdict": r.signal_verdict.title(),
                                 "Cluster": "Yes" if r.cluster_buy else "No",
                                 "Officer": r.officer_sentiment.title(),
                                 "Net Ratio": f"{r.net_purchase_ratio:.0%}" if r.net_purchase_ratio is not None else "—"}
                                for r in result.results]
                        def _insig_color(val):
                            try:
                                v = float(str(val))
                                if v >= 7: return "background-color: #0a2a0a; color: #00ff41"
                                if v <= 3: return "background-color: #2a0a0a; color: #ff4141"
                                return ""
                            except Exception:
                                return ""
                        df = pd.DataFrame(rows)
                        st.dataframe(df.style.applymap(_insig_color, subset=["Score"]), use_container_width=True)
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
            except Exception as e:
                st.error(f"Insider signal error: {e}")


# ─── SUPCHAIN — Supply Chain Concentration Risk ───────────────────────────────

def _render_supchain(args: list[str]):
    tickers_in = " ".join(args) if args else ""
    ticker_input = st.text_input("Ticker(s) (space-separated for screen):", tickers_in or "AAPL TSLA NVDA")
    tickers = [t.strip().upper() for t in ticker_input.split() if t.strip()]
    if not tickers:
        return
    st.markdown("## SUPCHAIN — Supply Chain Concentration Risk")
    st.caption("EDGAR XBRL customer concentration %, major customers, geographic HHI, risk flags, composite risk score 0-10.")

    mode = st.radio("Mode", ["Single ticker", "Concentration screen"], horizontal=True)
    max_conc = st.slider("Max customer concentration % (screen)", 0.0, 100.0, 30.0, 5.0) / 100

    if st.button("Analyze Supply Chain Risk", key="supchain_run"):
        with st.spinner("Parsing EDGAR XBRL concentration data..."):
            try:
                if mode == "Single ticker":
                    from sentinel.sfe.supply_chain import get_supply_chain_risk
                    result = run_async(get_supply_chain_risk(ticker=tickers[0]))

                    risk_color = "🟢" if result.supply_chain_risk_score <= 3 else ("🔴" if result.supply_chain_risk_score >= 7 else "🟡")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Risk Score", f"{risk_color} {result.supply_chain_risk_score:.1f}/10")
                    c2.metric("Top Customer %", f"{result.top_customer_pct:.0%}" if result.top_customer_pct else "—")
                    c3.metric("Major Customers", result.num_major_customers if result.num_major_customers is not None else "—")
                    c4.metric("Data Quality", result.data_quality.title())

                    col_conc, col_geo = st.columns(2)
                    with col_conc:
                        st.markdown("**Customer Concentration**")
                        if result.major_customers:
                            cust_rows = [{"Customer": c.description,
                                          "Revenue %": f"{c.revenue_pct:.0%}" if c.revenue_pct else "—",
                                          "Major": "Yes" if c.is_major else "No"}
                                         for c in result.major_customers]
                            st.dataframe(pd.DataFrame(cust_rows), use_container_width=True)
                        conc_color = "🟢" if result.concentration_score <= 3 else ("🔴" if result.concentration_score >= 7 else "🟡")
                        st.metric("Concentration Score", f"{conc_color} {result.concentration_score:.1f}/10")
                    with col_geo:
                        st.markdown("**Geographic Diversification**")
                        if result.geographic_segments:
                            geo_rows = [{"Region": g.region,
                                         "Revenue %": f"{g.revenue_pct:.0%}" if g.revenue_pct else "—"}
                                        for g in result.geographic_segments]
                            st.dataframe(pd.DataFrame(geo_rows), use_container_width=True)
                        if result.hhi_geographic is not None:
                            hhi_color = "🟢" if result.hhi_geographic < 1500 else ("🔴" if result.hhi_geographic > 2500 else "🟡")
                            st.metric("Geographic HHI", f"{hhi_color} {result.hhi_geographic:.0f}")
                        if result.international_revenue_pct is not None:
                            st.metric("International Revenue", f"{result.international_revenue_pct:.0%}")
                        div_color = "🟢" if result.diversification_score >= 7 else ("🔴" if result.diversification_score <= 3 else "🟡")
                        st.metric("Diversification Score", f"{div_color} {result.diversification_score:.1f}/10")

                    if result.risk_flags:
                        st.markdown("**Risk Flags**")
                        for flag in result.risk_flags:
                            st.warning(f"⚠ {flag}")
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
                else:
                    from sentinel.sfe.supply_chain import screen_concentration_risk
                    result = run_async(screen_concentration_risk(tickers=tickers, max_customer_concentration=max_conc))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tickers Screened", result.tickers_screened)
                    c2.metric("Avg Risk Score", f"{result.avg_risk_score:.1f}/10" if result.avg_risk_score is not None else "—")
                    c3.metric("Matching Filter", len(result.results))
                    if result.most_concentrated:
                        st.error(f"Highest concentration: **{result.most_concentrated}**")
                    if result.most_diversified:
                        st.success(f"Most diversified: **{result.most_diversified}**")
                    if result.highest_risk:
                        st.warning(f"Highest risk: {' | '.join(result.highest_risk)}")
                    if result.results:
                        rows = [{"Ticker": r.ticker,
                                 "Risk Score": f"{r.supply_chain_risk_score:.1f}",
                                 "Top Customer %": f"{r.top_customer_pct:.0%}" if r.top_customer_pct else "—",
                                 "Major Customers": r.num_major_customers if r.num_major_customers is not None else "—",
                                 "Intl Revenue": f"{r.international_revenue_pct:.0%}" if r.international_revenue_pct is not None else "—"}
                                for r in result.results]
                        def _risk_color(val):
                            try:
                                v = float(str(val))
                                if v <= 3: return "background-color: #0a2a0a; color: #00ff41"
                                if v >= 7: return "background-color: #2a0a0a; color: #ff4141"
                                return ""
                            except Exception:
                                return ""
                        df = pd.DataFrame(rows)
                        st.dataframe(df.style.applymap(_risk_color, subset=["Risk Score"]), use_container_width=True)
                    if result.warnings:
                        for w in result.warnings:
                            st.warning(w)
            except Exception as e:
                st.error(f"Supply chain risk error: {e}")


# ─── HELP ─────────────────────────────────────────────────────────────────────

def _render_help(args: list[str]):
    st.markdown("## SENTINEL Command Reference")
    st.markdown("""
    | Code | Description | Example |
    |------|-------------|---------|
    | DES | Security description and fundamentals | DES AAPL |
    | GP | Price chart | GP SPY 1Y |
    | GPC | Comparative chart | GPC SPY QQQ IWM |
    | HP | Historical prices table | HP MSFT |
    | FA | Financial analysis (XBRL) | FA NVDA |
    | DVD | Dividend history | DVD KO |
    | OPT | Options analytics: IV surface, GEX, max pain | OPT SPY |
    | OMON | Options chain (raw) | OMON SPY |
    | NI | News and sentiment | NI TSLA |
    | CN | Congressional STOCK Act trades | CN NVDA |
    | IN | Insider trades (Form 4) | IN AAPL |
    | HDS | Institutional holders (13F) | HDS MSFT |
    | SECF | SEC filings | SECF AAPL 10-K |
    | WEI | FRED economic series | WEI DGS10 |
    | ECOS | FRED core 33-series screen | ECOS |
    | YC | Treasury yield curve | YC |
    | COT | CFTC COT positioning signals | COT GOLD |
    | REGM | HMM macro regime detector | REGM |
    | SRCH | Natural-language screener | SRCH tech PE<20 growth>20% |
    | BT | Backtest runner | BT SPY momentum |
    | PORT | Portfolio overview | PORT |
    | RISK | Risk analytics | RISK |
    | MSG | Order management | MSG |
    | ACT | Account summary | ACT |
    | MCP | AI assistant (Claude) | MCP What should I buy? |
    | DCF | Discounted cash flow intrinsic value | DCF AAPL |
    | SOC | Social sentiment (Reddit + StockTwits) | SOC TSLA |
    | STRESS | Portfolio stress test (historical scenarios) | STRESS |
    | FACTOR | Fama-French 5-factor decomposition | FACTOR |
    | CAL | Economic calendar (FRED releases) | CAL |
    | FX | ECB FX rates via Frankfurter API | FX EURUSD |
    | GLOBAL | G7 macro dashboard | GLOBAL |
    | KELLY | Kelly criterion position sizer | KELLY |
    | SHORT | FINRA short interest + squeeze scan | SHORT TSLA |
    | BONDS | FINRA TRACE corporate bond quotes | BONDS AAPL |
    | SEG | EDGAR XBRL segment revenue | SEG AAPL |
    | OFLOW | Options flow screener | OFLOW SPY AAPL NVDA |
    | CLN | DataCleaner consensus pipeline | CLN AAPL |
    | STRAT | NL to strategy generator (Claude tool-use) | STRAT buy oversold tech |
    | RESEARCH | Autonomous AI research memo | RESEARCH NVDA |
    | TRENDS | Google Trends momentum signal | TRENDS AAPL |
    | DEFI | DeFi TVL, protocols, yields | DEFI |
    | ONCHAIN | On-chain NVT/MVRV/fear-greed | ONCHAIN bitcoin |
    | EKP | Earnings KPI extractor (MD&A + Claude) | EKP MSFT 10-Q |
    | ESG | ESG proxy profile (EDGAR DEF14A + 10-K) | ESG JPM |
    | ALPHA | Congress + COT + insider alpha composite | ALPHA NVDA |
    | XLS | Bloomberg-style Excel export | XLS AAPL |
    | NGAAP | Non-GAAP earnings parser (8-K) | NGAAP MSFT |
    | COMPS | Comparable company analysis | COMPS NVDA |
    | ACT13D | Activist investor monitor (13D/13G) | ACT13D TSLA |
    | ESRCH | EDGAR full-text search | ESRCH material weakness |
    | VAR | Portfolio Value-at-Risk (hist/param/MC) | VAR |
    | ATTRIB | Brinson attribution vs benchmark | ATTRIB |
    | CONTRA | Controversy and ESG risk monitor | CONTRA META |
    | DEX | Decentralized exchange analytics | DEX |
    | PORTOPT | Portfolio optimizer (mean-var / BL / ERC) | PORTOPT |
    | FISCRN | Fixed income screener (yield, duration, credit) | FISCRN |
    | CEVT | On-chain event monitor (whale tx, TVL moves) | CEVT UNI |
    | TASIG | Technical analysis signals (RSI/MACD/BB/VWAP) | TASIG AAPL |
    | ADVTA | Advanced TA: Ichimoku, Fibonacci, ADX, SAR | ADVTA NVDA |
    | SYNTH | AI research synthesis (bull/bear/risks/catalysts) | SYNTH NVDA |
    | GARCHVAR | GARCH(1,1) conditional VaR + Basel backtest | GARCHVAR |
    | BONDA | Bond price, duration, convexity, DV01, scenarios | BONDA |
    | ZSCORE | Altman Z-Score credit risk model | ZSCORE NFLX |
    | MASCRN | M&A deal screener (EDGAR 8-K/DEFM14A/SC TO-T) | MASCRN |
    | MAPROF | M&A target profile (activist, defense mech.) | MAPROF AAPL |
    | ETFPROF | ETF profile: holdings, factors, flows, NAV | ETFPROF SPY |
    | ETFCMP | ETF side-by-side comparison | ETFCMP SPY QQQ IWM |
    | COMMOD | Commodity dashboard: energy/metals/ag + regime | COMMOD |
    | IFRS | International IFRS financials (non-US 20-F filers) | IFRS ASML |
    | ECONFC | Economic forecasting: AR/VAR on FRED macro series | ECONFC |
    | LIQ | Liquidity analytics: Amihud, bid-ask, Kyle's lambda | LIQ AAPL |
    | ALRT | Price alert management: create/check/manage | ALRT |
    | RIAPROF | Form ADV RIA intelligence: AUM, clients, fees | RIAPROF Bridgewater |
    | GEORISK | Geopolitical risk dashboard (GDELT + Claude) | GEORISK Russia |
    | LBO | LBO model + M&A accretion/dilution + live screen | LBO DELL |
    | EDGMON | EDGAR filing monitor: 8-K, 13D, S-1, Form 4 | EDGMON |
    | FXDASH | FX analytics: forward curve, vol, carry, momentum | FXDASH EURUSD |
    | FORMD | Private market intelligence: Form D SEC filings | FORMD OpenAI |
    | SCEN | Macro scenario analysis: shock → portfolio P&L | SCEN |
    | DVDS | Dividend analytics: yield/growth/quality/DDM | DVDS JNJ |
    | OPTFLOW | Options flow: unusual activity, max pain, IV skew | OPTFLOW SPY AAPL NVDA |
    | EARNLP | Earnings 8-K NLP: tone/guidance/themes (Claude) | EARNLP MSFT |
    | CREDIT | Credit analytics: Merton PD, CDS proxy, Altman Z | CREDIT GE F |
    | PEERS | Auto peer comparison: valuation/growth/rank | PEERS NVDA |
    | SQUEEZE | Short-squeeze: DTC, borrow proxy, gamma risk | SQUEEZE GME AMC |
    | ANLEST | Analyst estimates: targets/recs/EPS/revenue | ANLEST AAPL |
    | CONV | Convertible bond: parity/premium/delta/greeks | CONV TSLA NVDA |
    | GOV | Corporate governance: board/duality/say-on-pay | GOV AAPL TSLA |
    | ESURP | Earnings surprise: EPS beat/miss history, trend | ESURP AAPL NVDA |
    | VIXTS | VIX term structure: spot/3M/6M/1Y, VRP, regime | VIXTS |
    | INSIG | Insider cluster signal: officer buy, score 0-10 | INSIG AAPL NVDA |
    | SUPCHAIN | Supply chain concentration: customer/geo risk | SUPCHAIN AAPL TSLA |
    | SECROT | Sector rotation: SPDR ETF RS heatmap, momentum | SECROT |
    | EQSCORE | Earnings quality: accruals, cash conversion, tier | EQSCORE AAPL GE |
    | IVTERM | IV term structure: ATM IV, forward vol, skew | IVTERM SPY AAPL |
    | NOWCAST | GDP nowcast: FRED indicators, recession prob. | NOWCAST |
    | HELP | This help screen | HELP |

    Leapfrog features (unavailable in Bloomberg, CapIQ, FactSet):
    CN, COT, REGM, BT, MCP, SOC, STRESS, FACTOR, CAL, SHORT, BONDS, GLOBAL,
    KELLY, SEG, OFLOW, STRAT, RESEARCH, TRENDS, DEFI, ONCHAIN, EKP, ESG,
    ALPHA, XLS, NGAAP, COMPS, ACT13D, ESRCH, VAR, ATTRIB, CONTRA, DEX,
    PORTOPT, FISCRN, CEVT, TASIG, SYNTH, ADVTA, GARCHVAR, BONDA, ZSCORE,
    MASCRN, MAPROF, ETFPROF, ETFCMP, COMMOD, IFRS, ECONFC, LIQ, ALRT,
    RIAPROF, GEORISK, LBO, EDGMON, FXDASH, FORMD, SCEN, DVDS,
    OPTFLOW, EARNLP, CREDIT, PEERS, SQUEEZE, ANLEST, CONV, GOV,
    ESURP, VIXTS, INSIG, SUPCHAIN, SECROT, EQSCORE, IVTERM, NOWCAST
    """)


def _render_cln(args: list[str]):
    ticker = args[0] if args else st.text_input("Ticker:", "AAPL")
    if not ticker:
        return
    st.markdown(f"## CLN — DataCleaner: {ticker}")
    st.caption("Multi-source consensus pipeline: parallel fetch, validation, median consensus, quality scoring, DB write")

    col1, col2 = st.columns(2)
    with col1:
        interval = st.selectbox("Interval", ["1d", "1h", "4h"], index=0)
        write_to_db = st.checkbox("Write consensus to DB", value=True)
    with col2:
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        st.markdown(f"**Period:** {start_date} to {end_date}")

    col_run, col_rank = st.columns(2)

    with col_run:
        if st.button("Run DataCleaner", key="cln_run"):
            with st.spinner(f"Cleaning {ticker} across all adapters..."):
                try:
                    import requests as _req
                    resp = _req.post(
                        "http://localhost:8000/api/v1/cleaner/run",
                        json={
                            "ticker": ticker.upper(),
                            "interval": interval,
                            "start": str(start_date),
                            "end": str(end_date),
                            "write_to_db": write_to_db,
                        },
                        timeout=60,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    st.success(f"Consensus built — quality score: {result.get('quality_score', 'N/A'):.2f}")
                    if result.get("rows_written"):
                        st.metric("Rows written", result["rows_written"])
                    if result.get("source_agreement"):
                        st.metric("Source agreement", f"{result['source_agreement']:.1%}")
                    if result.get("warnings"):
                        for w in result["warnings"]:
                            st.warning(w)
                except Exception as e:
                    st.error(f"DataCleaner error: {e}")

    with col_rank:
        if st.button("Source Rankings", key="cln_rank"):
            with st.spinner("Fetching source quality rankings..."):
                try:
                    import requests as _req
                    resp = _req.get("http://localhost:8000/api/v1/cleaner/rankings", timeout=10)
                    resp.raise_for_status()
                    rankings = resp.json()
                    if rankings:
                        st.dataframe(pd.DataFrame(rankings), use_container_width=True)
                except Exception as e:
                    st.error(f"Rankings fetch error: {e}")


# ─── Main dispatch ─────────────────────────────────────────────────────────────

def main():
    st.markdown('<div class="terminal-header">SENTINEL — Sovereign Financial Terminal</div>',
                unsafe_allow_html=True)

    col_cmd, col_go = st.columns([5, 1])
    with col_cmd:
        raw = st.text_input(
            "",
            placeholder="Enter function code: DES AAPL  |  GP SPY 1Y  |  MCP explain yield curve  |  HELP",
            label_visibility="collapsed",
            key="terminal_input",
        )
    with col_go:
        go = st.button("GO", use_container_width=True)

    if not raw and not go:
        _render_help([])
        return

    parts = raw.strip().split()
    if not parts:
        _render_help([])
        return

    fn_code = parts[0].upper()
    args = parts[1:]

    dispatch: dict[str, Any] = {
        "DES": _render_des,
        "GP": _render_gp,
        "GPC": _render_gpc,
        "HP": _render_hp,
        "FA": _render_fa,
        "DVD": _render_dvd,
        "OPT": _render_opt_analytics,
        "OMON": _render_opt,
        "NI": _render_ni,
        "CN": _render_cn,
        "IN": _render_in,
        "HDS": _render_hds,
        "SECF": _render_secf,
        "WEI": _render_wei,
        "ECOS": _render_ecos,
        "YC": _render_yc,
        "COT": _render_cot,
        "REGM": _render_regm,
        "SRCH": _render_srch,
        "BT": _render_bt,
        "PORT": _render_port,
        "RISK": _render_risk,
        "MSG": _render_msg,
        "ACT": _render_act,
        "MCP": _render_mcp,
        "HELP": _render_help,
        "?": _render_help,
        "DCF": _render_dcf,
        "SOC": _render_social,
        "STRESS": _render_stress,
        "FACTOR": _render_factor,
        "CAL": _render_cal,
        "FX": _render_fx,
        "GLOBAL": _render_global,
        "KELLY": _render_kelly,
        "SHORT": _render_short,
        "BONDS": _render_bonds,
        "SEG": _render_seg,
        "OFLOW": _render_oflow,
        "CLN": _render_cln,
        "STRAT": _render_strat,
        "RESEARCH": _render_research,
        "TRENDS": _render_trends,
        "DEFI": _render_defi,
        "ONCHAIN": _render_onchain,
        "EKP": _render_ekp,
        "ESG": _render_esg,
        "ALPHA": _render_alpha,
        "XLS": _render_xls,
        "NGAAP": _render_ngaap,
        "COMPS": _render_comps,
        "ACT13D": _render_act13d,
        "ESRCH": _render_esrch,
        "VAR": _render_var,
        "ATTRIB": _render_attrib,
        "CONTRA": _render_contra,
        "DEX": _render_dex,
        "PORTOPT": _render_portopt,
        "FISCRN": _render_fiscrn,
        "CEVT": _render_cevt,
        "TASIG": _render_tasig,
        "SYNTH": _render_synth,
        "GARCHVAR": _render_garchvar,
        "ADVTA": _render_advta,
        "BONDA": _render_bonda,
        "ZSCORE": _render_zscore,
        "MASCRN": _render_mascrn,
        "MAPROF": _render_maprof,
        "ETFPROF": _render_etfprof,
        "ETFCMP": _render_etfcmp,
        "COMMOD": _render_commod,
        "IFRS": _render_ifrs,
        "ECONFC": _render_econfc,
        "LIQ": _render_liq,
        "ALRT": _render_alrt,
        "RIAPROF": _render_riaprof,
        "GEORISK": _render_georisk,
        "LBO": _render_lbo,
        "EDGMON": _render_edgmon,
        "FXDASH": _render_fxdash,
        "FORMD": _render_formd,
        "SCEN": _render_scen,
        "DVDS": _render_dvds,
        "OPTFLOW": _render_optflow,
        "EARNLP": _render_earnlp,
        "CREDIT": _render_credit,
        "PEERS": _render_peers,
        "SQUEEZE": _render_squeeze,
        "ANLEST": _render_anlest,
        "CONV": _render_conv,
        "GOV": _render_gov,
        "ESURP": _render_esurp,
        "VIXTS": _render_vixts,
        "INSIG": _render_insig,
        "SUPCHAIN": _render_supchain,
        "SECROT": _render_secrot,
        "EQSCORE": _render_eqscore,
        "IVTERM": _render_ivterm,
        "NOWCAST": _render_nowcast,
    }

    handler = dispatch.get(fn_code)
    if handler:
        handler(args)
    else:
        st.error(f"Unknown function: {fn_code}. Type HELP for a list of commands.")


if __name__ == "__main__":
    main()
