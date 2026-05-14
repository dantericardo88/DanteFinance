"""Data routes — OHLCV, quotes, fundamentals, options.

DB-first pattern: check TimescaleDB before calling live adapters.
Live fallback triggers ingest so subsequent calls hit DB.
"""
from __future__ import annotations
from datetime import datetime, date
from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.sds import get_all_adapters, build_default_adapters
from sentinel.sds.db import get_session

router = APIRouter()


@router.get("/ohlcv/{ticker}")
async def get_ohlcv(
    ticker: str,
    start: date = Query(..., description="Start date YYYY-MM-DD"),
    end: date = Query(..., description="End date YYYY-MM-DD"),
    interval: str = Query("1d", description="Bar interval: 1m,5m,15m,30m,1h,1d,1wk,1mo"),
    adjusted: bool = Query(True, description="Apply split/dividend backward adjustment"),
    session: AsyncSession = Depends(get_session),
):
    """Fetch OHLCV bars. DB-first — falls back to live adapters and persists to DB."""
    from sentinel.sds import repository
    from sentinel.sds.ingest import ingest_ticker_ohlcv
    from sentinel.sds.corporate_actions import get_corporate_actions as _ca_fallback

    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.max.time())

    # DB-first: check TimescaleDB
    rows = await repository.get_ohlcv_bars_by_ticker(
        ticker=ticker, interval=interval, start=start_dt, end=end_dt, session=session,
    )

    source = "db"
    if not rows:
        # Live fallback — ingest and return fresh
        if not get_all_adapters():
            build_default_adapters()
        result = await ingest_ticker_ohlcv(
            ticker=ticker, interval=interval,
            start=start_dt, end=end_dt,
            session=session,
        )
        if result.ok:
            rows = await repository.get_ohlcv_bars_by_ticker(
                ticker=ticker, interval=interval,
                start=start_dt, end=end_dt,
                session=session,
            )
            source = result.source or "live"

    # Apply corporate action adjustments from DB
    ca_actions = []
    if adjusted and rows:
        figi = rows[0]["figi"]
        ca_actions = await repository.get_corporate_actions(figi=figi, session=session)

    from sentinel.sds.corporate_actions import compute_cumulative_factor

    def _adj_factor(row_time) -> float:
        if not ca_actions:
            return 1.0
        d = row_time.date() if hasattr(row_time, "date") else row_time
        return float(compute_cumulative_factor(d, ca_actions))

    return {
        "ticker": ticker,
        "interval": interval,
        "adjusted": adjusted,
        "source": source,
        "count": len(rows),
        "bars": [
            {
                "time": r["time"].isoformat() if hasattr(r["time"], "isoformat") else str(r["time"]),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "adj_close": float(r["close"]) * _adj_factor(r["time"]) if adjusted else float(r["close"]),
                "volume": float(r["volume"]),
                "adj_factor": _adj_factor(r["time"]) if adjusted else 1.0,
                "source": r["source"],
            }
            for r in rows
        ],
    }


@router.get("/quote/{ticker}")
async def get_quote(ticker: str):
    """Latest real-time quote."""
    from sentinel.sds import get_adapter
    for name in ["alpaca", "polygon", "finnhub"]:
        adapter = get_adapter(name)
        if adapter is None:
            continue
        try:
            if name == "alpaca":
                snap = await adapter.fetch_snapshot(ticker)
                if snap:
                    return {"ticker": ticker, "source": name, **snap}
            elif name == "polygon":
                snap = await adapter.fetch_snapshot(ticker)
                if snap:
                    return {"ticker": ticker, "source": name, **snap}
            elif name == "finnhub":
                data = await adapter.fetch_ticker(ticker)
                if data:
                    return {"ticker": ticker, "source": name, **data}
        except Exception:
            continue
    raise HTTPException(404, f"No quote available for {ticker}")


@router.get("/fundamentals/{ticker}")
async def get_fundamentals(
    ticker: str,
    facts: Optional[str] = Query(None, description="Comma-separated GAAP concept labels"),
    as_of: Optional[date] = Query(
        None,
        description=(
            "Point-in-time date (YYYY-MM-DD). Only facts filed ON OR BEFORE this date "
            "are returned. Omit for latest. Required for bias-free backtesting."
        ),
    ),
    session: AsyncSession = Depends(get_session),
):
    """EDGAR XBRL fundamentals with point-in-time (PIT) enforcement.

    DB-first: checks financial_facts table before calling live EDGAR API.
    When as_of is provided, only facts with filed_date <= as_of are returned,
    preventing look-ahead bias in backtests.
    """
    from sentinel.sds.adapters.edgar_adapter import EDGARAdapter
    from sentinel.sds.ingest import ingest_edgar_facts
    from sentinel.sds import repository

    edgar = EDGARAdapter()
    await edgar.load_company_tickers()
    cik = edgar.ticker_to_cik(ticker)
    if not cik:
        raise HTTPException(404, f"CIK not found for {ticker}")

    as_of_dt = datetime.combine(as_of, datetime.max.time()) if as_of else None

    # DB-first: pull from financial_facts table
    db_rows = await repository.get_financial_facts(
        cik=cik, session=session, as_of=as_of_dt
    )

    source = "db"
    if not db_rows:
        # Live fallback — ingest and re-query
        ingest_result = await ingest_edgar_facts(cik=cik, ticker=ticker, session=session)
        if ingest_result.ok:
            db_rows = await repository.get_financial_facts(
                cik=cik, session=session, as_of=as_of_dt
            )
            source = "edgar_live"

    # Group by label then annual/quarterly
    from collections import defaultdict
    by_label: dict[str, list] = defaultdict(list)
    for row in db_rows:
        label = row.get("label") or row.get("concept", "")
        by_label[label].append(row)

    label_filter = set(facts.split(",")) if facts else None
    annual: dict = {}
    quarterly: dict = {}

    for label, rows in by_label.items():
        if label_filter and label not in label_filter:
            continue
        sorted_rows = sorted(rows, key=lambda r: r["period_end"] or date.min, reverse=True)
        # Annual: period spans roughly 12 months
        ann = [r for r in sorted_rows if r.get("period_start") and
               r["period_end"] and
               (r["period_end"] - r["period_start"]).days >= 300]
        # Quarterly: period spans 60-120 days
        qrt = [r for r in sorted_rows if r.get("period_start") and
               r["period_end"] and
               60 <= (r["period_end"] - r["period_start"]).days < 300]

        def _fmt(r: dict) -> dict:
            fd = r.get("filed")
            return {
                "period_end": r["period_end"].isoformat() if r.get("period_end") else None,
                "filed_date": fd.isoformat() if fd else None,
                "value": float(r["value"]) if r.get("value") is not None else None,
                "form": r.get("form"),
                "accession": r.get("accession"),
            }

        if ann:
            annual[label] = [_fmt(r) for r in ann[:8]]
        if qrt:
            quarterly[label] = [_fmt(r) for r in qrt[:12]]

    return {
        "ticker": ticker,
        "cik": cik,
        "as_of": as_of.isoformat() if as_of else None,
        "pit_enforced": as_of is not None,
        "source": source,
        "total_facts": len(db_rows),
        "annual": annual,
        "quarterly": quarterly,
    }


@router.get("/options/{ticker}")
async def get_options(ticker: str, expiry: Optional[str] = None):
    """Options chain from yfinance."""
    from sentinel.sds import get_adapter
    adapter = get_adapter("yfinance")
    if adapter is None:
        raise HTTPException(503, "yfinance adapter not available")
    chain = await adapter.fetch_options_chain(ticker, expiry=expiry)
    if not chain:
        raise HTTPException(404, f"No options data for {ticker}")
    return chain


@router.get("/health")
async def adapter_health():
    """Run health checks on all registered adapters."""
    from sentinel.sds.normalizer import run_all_health_checks
    events = await run_all_health_checks()
    return [
        {"adapter": e.adapter, "event": e.event_type, "severity": e.severity.value,
         "details": e.details}
        for e in events
    ]
