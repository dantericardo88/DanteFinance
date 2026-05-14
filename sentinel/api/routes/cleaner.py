"""DataCleaner API — multi-source consensus cleaning endpoints."""
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.sds.db import get_session

router = APIRouter()


@router.post("/run")
async def run_cleaner(
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Run the DataCleaner pipeline for a single ticker."""
    ticker = body.get("ticker", "")
    interval = body.get("interval", "1d")
    write_to_db = body.get("write_to_db", True)

    raw_start = body.get("start")
    raw_end = body.get("end")

    try:
        start_dt = datetime.combine(
            date.fromisoformat(raw_start) if raw_start else date(1900, 1, 1),
            datetime.min.time(),
        )
        end_dt = datetime.combine(
            date.fromisoformat(raw_end) if raw_end else date.today(),
            datetime.max.time(),
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date: {exc}") from exc

    try:
        from sentinel.sds.data_cleaner import DataCleaner
        cleaner = DataCleaner(session)
        report = await cleaner.clean_ticker(ticker, interval, start_dt, end_dt, write_to_db)
        return report.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/quality-report")
async def quality_report(
    ticker: str,
    interval: str = "1d",
    limit: int = Query(50, ge=1, le=500),
    source: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    """Return per-source data quality scores for a ticker."""
    from sentinel.sds import repository
    scores = await repository.get_quality_scores(ticker, interval, session, source, limit)
    return {
        "ticker": ticker,
        "interval": interval,
        "scores": scores,
        "count": len(scores),
    }


@router.post("/run-universe")
async def run_universe(
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Run the DataCleaner pipeline across a universe of tickers."""
    _default_universe = [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL",
        "META", "TSLA", "JPM", "V", "UNH",
        "SPY", "QQQ", "GLD", "TLT",
    ]
    tickers = body.get("tickers") or _default_universe
    interval = body.get("interval", "1d")
    max_concurrent = body.get("max_concurrent", 5)
    write_to_db = body.get("write_to_db", True)

    try:
        from sentinel.sds.data_cleaner import DataCleaner
        reports = await DataCleaner(session).clean_universe(
            tickers,
            interval,
            max_concurrent=max_concurrent,
            write_to_db=write_to_db,
        )
        return {
            "status": "ok",
            "tickers_processed": len(reports),
            "reports": [r.to_dict() for r in reports],
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/source-rankings")
async def source_rankings(
    interval: str = "1d",
    session: AsyncSession = Depends(get_session),
):
    """Rank data sources by availability and deviation for a given interval."""
    try:
        result = await session.execute(
            text("""
                SELECT
                    source,
                    AVG(availability)        AS avg_availability,
                    AVG(mean_abs_deviation)  AS avg_deviation,
                    COUNT(*) FILTER (WHERE grade = 'A') AS grade_a,
                    COUNT(*) FILTER (WHERE grade = 'B') AS grade_b,
                    COUNT(*) FILTER (WHERE grade = 'C') AS grade_c,
                    COUNT(*) FILTER (WHERE grade = 'F') AS grade_f,
                    COUNT(*) AS total_observations
                FROM data_quality_scores
                WHERE interval = :interval
                GROUP BY source
                ORDER BY avg_availability DESC, avg_deviation ASC
            """),
            {"interval": interval},
        )
        return {
            "interval": interval,
            "sources": [dict(row) for row in result.mappings()],
        }
    except Exception:
        return {"interval": interval, "sources": []}
