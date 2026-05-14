"""analyst_estimates.py — Analyst consensus proxy via yfinance (Dim 18)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RecommendationGrade(BaseModel):
    model_config = ConfigDict(frozen=True)

    firm: str
    to_grade: str
    from_grade: str
    action: str  # upgrade / downgrade / initiated / reiterated
    date: str


class QuarterlyEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    period: str  # "0q" (current), "1q" (next), "0y", "1y"
    avg_estimate: Optional[float] = None
    low_estimate: Optional[float] = None
    high_estimate: Optional[float] = None
    number_of_analysts: Optional[int] = None


class AnalystEstimates(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    current_price: Optional[float] = None
    target_mean: Optional[float] = None
    target_median: Optional[float] = None
    target_high: Optional[float] = None
    target_low: Optional[float] = None
    price_upside_pct: Optional[float] = None
    num_analysts: Optional[int] = None
    recommendation_mean: Optional[float] = None
    consensus_label: str  # Strong Buy / Buy / Hold / Sell / Strong Sell / N/A
    analyst_dispersion: Optional[float] = None
    recent_grades: list[RecommendationGrade]
    upgrades_90d: int
    downgrades_90d: int
    eps_estimates: list[QuarterlyEstimate]
    revenue_estimates: list[QuarterlyEstimate]
    as_of: str
    warnings: list[str]


class AnalystTickerSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    upside_pct: Optional[float] = None
    consensus: str
    num_analysts: Optional[int] = None
    target_mean: Optional[float] = None
    current_price: Optional[float] = None


class AnalystSentimentScreen(BaseModel):
    model_config = ConfigDict(frozen=True)

    tickers_screened: int
    min_upside_filter: float
    results: list[AnalystTickerSummary]
    most_upside: Optional[str] = None
    most_downside: Optional[str] = None
    strong_buys: list[str]
    strong_sells: list[str]
    avg_upside_pct: Optional[float] = None
    as_of: str
    warnings: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _consensus_label(rec_mean: Optional[float]) -> str:
    """Map yfinance recommendationMean (1=Strong Buy … 5=Strong Sell) to label."""
    if rec_mean is None:
        return "N/A"
    if rec_mean <= 1.5:
        return "Strong Buy"
    if rec_mean <= 2.5:
        return "Buy"
    if rec_mean <= 3.5:
        return "Hold"
    if rec_mean <= 4.5:
        return "Sell"
    return "Strong Sell"


def _classify_action(action_raw: str, to_grade: str, from_grade: str) -> str:
    """Normalise recommendation action to upgrade/downgrade/initiated/reiterated."""
    action = str(action_raw).lower().strip()
    if "init" in action or "coverage" in action:
        return "initiated"
    if "up" in action:
        return "upgrade"
    if "down" in action:
        return "downgrade"
    # Fallback: infer from grade change
    buy_grades = {"strong buy", "buy", "outperform", "overweight", "positive"}
    sell_grades = {"sell", "strong sell", "underperform", "underweight", "negative"}
    if from_grade and to_grade:
        fg = from_grade.lower()
        tg = to_grade.lower()
        if fg in sell_grades and tg in buy_grades:
            return "upgrade"
        if fg in buy_grades and tg in sell_grades:
            return "downgrade"
    return "reiterated"


def _parse_quarterly_estimates(df, periods: list[str]) -> list[QuarterlyEstimate]:
    """Convert a yfinance estimates DataFrame to QuarterlyEstimate list."""
    result: list[QuarterlyEstimate] = []
    if df is None:
        return result
    try:
        import pandas as pd  # noqa: PLC0415 — lazy import by design
        if not isinstance(df, pd.DataFrame) or df.empty:
            return result
        for period in periods:
            if period not in df.columns:
                continue
            col = df[period]

            def _get(row_label: str) -> Optional[float]:
                try:
                    val = col.get(row_label)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return None
                    return float(val)
                except Exception:
                    return None

            def _get_int(row_label: str) -> Optional[int]:
                try:
                    val = col.get(row_label)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return None
                    return int(val)
                except Exception:
                    return None

            result.append(
                QuarterlyEstimate(
                    period=period,
                    avg_estimate=_get("avg"),
                    low_estimate=_get("low"),
                    high_estimate=_get("high"),
                    number_of_analysts=_get_int("numberOfAnalysts"),
                )
            )
    except Exception as exc:
        logger.warning("Error parsing quarterly estimates", error=str(exc))
    return result


# ---------------------------------------------------------------------------
# Sync yfinance fetch — runs inside asyncio.to_thread
# ---------------------------------------------------------------------------


def _fetch_analyst_data_sync(ticker: str) -> dict:  # noqa: C901
    """Sync yfinance fetch — called via asyncio.to_thread. Lazy-imports yfinance/pandas."""
    import yfinance as yf  # noqa: PLC0415 — lazy import by design
    import pandas as pd    # noqa: PLC0415 — lazy import by design

    result: dict = {
        "info": {},
        "recommendations": None,
        "eps_estimates": None,
        "revenue_estimates": None,
        "errors": [],
    }

    try:
        yf_ticker = yf.Ticker(ticker)
    except Exception as exc:
        result["errors"].append(f"Ticker init failed: {exc}")
        return result

    # --- info dict -----------------------------------------------------------
    try:
        result["info"] = yf_ticker.info or {}
    except Exception as exc:
        result["errors"].append(f"info fetch failed: {exc}")

    # --- recommendations table -----------------------------------------------
    try:
        recs = yf_ticker.recommendations
        if isinstance(recs, pd.DataFrame) and not recs.empty:
            result["recommendations"] = recs
    except Exception as exc:
        result["errors"].append(f"recommendations fetch failed: {exc}")

    # --- EPS estimates -------------------------------------------------------
    try:
        eps_df = yf_ticker.earnings_estimate
        if isinstance(eps_df, pd.DataFrame) and not eps_df.empty:
            result["eps_estimates"] = eps_df
    except Exception as exc:
        result["errors"].append(f"earnings_estimate fetch failed: {exc}")

    # --- Revenue estimates ---------------------------------------------------
    try:
        rev_df = yf_ticker.revenue_estimate
        if isinstance(rev_df, pd.DataFrame) and not rev_df.empty:
            result["revenue_estimates"] = rev_df
    except Exception as exc:
        result["errors"].append(f"revenue_estimate fetch failed: {exc}")

    return result


# ---------------------------------------------------------------------------
# Entry point: get_analyst_estimates
# ---------------------------------------------------------------------------


async def get_analyst_estimates(ticker: str) -> AnalystEstimates:
    """Fetch analyst consensus estimates for a single ticker via yfinance."""
    ticker_upper = ticker.upper()
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    warnings: list[str] = []

    logger.info("Fetching analyst estimates", ticker=ticker_upper)

    try:
        raw = await asyncio.to_thread(_fetch_analyst_data_sync, ticker_upper)
    except Exception as exc:
        logger.error("Analyst data fetch failed", ticker=ticker_upper, error=str(exc))
        warnings.append(f"Data fetch failed: {exc}")
        raw = {"info": {}, "recommendations": None, "eps_estimates": None,
               "revenue_estimates": None, "errors": [str(exc)]}

    warnings.extend(raw.get("errors", []))

    info: dict = raw.get("info") or {}

    # --- Price targets -------------------------------------------------------
    target_mean: Optional[float] = info.get("targetMeanPrice")
    target_median: Optional[float] = info.get("targetMedianPrice")
    target_high: Optional[float] = info.get("targetHighPrice")
    target_low: Optional[float] = info.get("targetLowPrice")
    num_analysts: Optional[int] = info.get("numberOfAnalystOpinions")
    rec_mean: Optional[float] = info.get("recommendationMean")

    # Current price: prefer currentPrice, fallback to regularMarketPrice
    current_price: Optional[float] = (
        info.get("currentPrice") or info.get("regularMarketPrice")
    )

    # --- Derived metrics -----------------------------------------------------
    price_upside_pct: Optional[float] = None
    if target_mean is not None and current_price and current_price > 0:
        price_upside_pct = round((target_mean - current_price) / current_price * 100, 4)

    analyst_dispersion: Optional[float] = None
    if (
        target_high is not None
        and target_low is not None
        and target_mean is not None
        and target_mean > 0
    ):
        analyst_dispersion = round((target_high - target_low) / target_mean, 4)

    label = _consensus_label(rec_mean)

    # --- Recent recommendation grades (last 90 days) -------------------------
    recent_grades: list[RecommendationGrade] = []
    upgrades_90d = 0
    downgrades_90d = 0
    cutoff = datetime.utcnow() - timedelta(days=90)

    recs_df = raw.get("recommendations")
    if recs_df is not None:
        try:
            import pandas as pd  # noqa: PLC0415 — lazy import by design

            df = recs_df.copy()
            # Normalise index: some versions expose date as index, some as column
            if "Date" in df.columns:
                df["_date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
            elif df.index.name in ("Date", "date") or hasattr(df.index, "tz"):
                df = df.reset_index()
                date_col = [c for c in df.columns if c.lower() == "date"]
                df["_date"] = pd.to_datetime(
                    df[date_col[0]] if date_col else df.index,
                    errors="coerce", utc=True,
                )
            else:
                df["_date"] = pd.NaT

            # yfinance columns: Firm, To Grade, From Grade, Action
            col_map = {c.lower().replace(" ", "_"): c for c in df.columns}
            firm_col = col_map.get("firm", "Firm")
            to_col = col_map.get("to_grade", "To Grade")
            from_col = col_map.get("from_grade", "From Grade")
            action_col = col_map.get("action", "Action")

            for _, row in df.iterrows():
                row_date = row.get("_date")
                date_str = (
                    row_date.strftime("%Y-%m-%d")
                    if row_date is not pd.NaT and row_date is not None
                    else "unknown"
                )

                firm = str(row.get(firm_col, "")) or "Unknown"
                to_grade = str(row.get(to_col, "")) or ""
                from_grade = str(row.get(from_col, "")) or ""
                action_raw = str(row.get(action_col, "")) or ""
                action = _classify_action(action_raw, to_grade, from_grade)

                grade = RecommendationGrade(
                    firm=firm,
                    to_grade=to_grade,
                    from_grade=from_grade,
                    action=action,
                    date=date_str,
                )

                # Filter to last 90 days
                if row_date is not pd.NaT and row_date is not None:
                    try:
                        row_naive = row_date.replace(tzinfo=None)
                        if row_naive >= cutoff:
                            recent_grades.append(grade)
                            if action == "upgrade":
                                upgrades_90d += 1
                            elif action == "downgrade":
                                downgrades_90d += 1
                    except Exception:
                        pass  # skip rows with unparseable dates

        except Exception as exc:
            warnings.append(f"recommendations parsing error: {exc}")
            logger.warning("Recommendations parse error", ticker=ticker_upper, error=str(exc))

    # --- EPS / Revenue estimates ---------------------------------------------
    periods = ["0q", "1q", "0y", "1y"]
    eps_estimates = _parse_quarterly_estimates(raw.get("eps_estimates"), periods)
    revenue_estimates = _parse_quarterly_estimates(raw.get("revenue_estimates"), periods)

    if not info and not raw.get("errors"):
        warnings.append("No analyst data returned — ticker may lack coverage")

    estimates = AnalystEstimates(
        ticker=ticker_upper,
        current_price=current_price,
        target_mean=target_mean,
        target_median=target_median,
        target_high=target_high,
        target_low=target_low,
        price_upside_pct=price_upside_pct,
        num_analysts=num_analysts,
        recommendation_mean=rec_mean,
        consensus_label=label,
        analyst_dispersion=analyst_dispersion,
        recent_grades=recent_grades,
        upgrades_90d=upgrades_90d,
        downgrades_90d=downgrades_90d,
        eps_estimates=eps_estimates,
        revenue_estimates=revenue_estimates,
        as_of=as_of,
        warnings=warnings,
    )

    logger.info(
        "Analyst estimates complete",
        ticker=ticker_upper,
        consensus=label,
        upside_pct=price_upside_pct,
        num_analysts=num_analysts,
        recent_grades=len(recent_grades),
    )

    return estimates


# ---------------------------------------------------------------------------
# Entry point: screen_analyst_sentiment
# ---------------------------------------------------------------------------


async def screen_analyst_sentiment(
    tickers: list[str],
    min_upside_pct: float = 10.0,
) -> AnalystSentimentScreen:
    """Parallel-fetch tickers, filter by min_upside_pct, return ranked screen."""
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    warnings: list[str] = []

    logger.info(
        "Screening analyst sentiment",
        tickers=len(tickers),
        min_upside_pct=min_upside_pct,
    )

    # --- Parallel fetch -------------------------------------------------------
    fetch_tasks = [get_analyst_estimates(t) for t in tickers]
    raw_results: list[AnalystEstimates | BaseException] = await asyncio.gather(
        *fetch_tasks, return_exceptions=True
    )

    summaries: list[AnalystTickerSummary] = []
    for ticker, res in zip(tickers, raw_results):
        if isinstance(res, BaseException):
            warnings.append(f"Fetch error for {ticker}: {res}")
            continue
        warnings.extend(res.warnings)
        summaries.append(
            AnalystTickerSummary(
                ticker=res.ticker,
                upside_pct=res.price_upside_pct,
                consensus=res.consensus_label,
                num_analysts=res.num_analysts,
                target_mean=res.target_mean,
                current_price=res.current_price,
            )
        )

    # --- Filter by min_upside_pct -------------------------------------------
    filtered = [
        s for s in summaries
        if s.upside_pct is not None and s.upside_pct >= min_upside_pct
    ]

    # --- Sort by upside descending -------------------------------------------
    filtered.sort(key=lambda s: (s.upside_pct is None, -(s.upside_pct or 0)))

    # --- Summary statistics --------------------------------------------------
    all_upsides = [s.upside_pct for s in summaries if s.upside_pct is not None]
    avg_upside: Optional[float] = (
        round(float(np.mean(all_upsides)), 4) if all_upsides else None
    )

    most_upside: Optional[str] = None
    most_downside: Optional[str] = None
    if all_upsides:
        ranked = sorted(
            [s for s in summaries if s.upside_pct is not None],
            key=lambda s: s.upside_pct,  # type: ignore[arg-type]
            reverse=True,
        )
        most_upside = ranked[0].ticker if ranked else None
        most_downside = ranked[-1].ticker if ranked else None

    strong_buys = [s.ticker for s in summaries if s.consensus == "Strong Buy"]
    strong_sells = [s.ticker for s in summaries if s.consensus == "Strong Sell"]

    screen = AnalystSentimentScreen(
        tickers_screened=len(tickers),
        min_upside_filter=min_upside_pct,
        results=filtered,
        most_upside=most_upside,
        most_downside=most_downside,
        strong_buys=strong_buys,
        strong_sells=strong_sells,
        avg_upside_pct=avg_upside,
        as_of=as_of,
        warnings=warnings,
    )

    logger.info(
        "Analyst sentiment screen complete",
        tickers_screened=screen.tickers_screened,
        results_returned=len(filtered),
        strong_buys=len(strong_buys),
        strong_sells=len(strong_sells),
        avg_upside_pct=avg_upside,
    )

    return screen
