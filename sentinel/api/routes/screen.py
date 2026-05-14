"""Screener routes."""
from fastapi import APIRouter, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.sds.db import get_session
from sentinel.sse.screener import ScreenerEngine

router = APIRouter()
_engine: ScreenerEngine | None = None


def get_engine() -> ScreenerEngine:
    global _engine
    if _engine is None:
        _engine = ScreenerEngine()
        _engine.initialize_tables()
    return _engine


@router.post("/")
async def screen(
    criteria: dict = Body(...),
    session: AsyncSession = Depends(get_session),
):
    """Run a stock screen with the given criteria dict.

    Auto-populates the universe from PostgreSQL on first call if empty.
    """
    engine = get_engine()
    if engine.get_universe_count() == 0:
        from sentinel.sse.db_loader import populate_screener_from_db
        await populate_screener_from_db(session)
    results = engine.screen(criteria)
    return {
        "count": len(results),
        "results": [
            {
                "ticker": r.ticker, "name": r.name, "sector": r.sector,
                "market_cap": float(r.market_cap) if r.market_cap else None,
                "pe_ratio": float(r.pe_ratio) if r.pe_ratio else None,
                "dividend_yield": float(r.dividend_yield) if r.dividend_yield else None,
            }
            for r in results
        ],
    }


@router.post("/refresh")
async def refresh_screener(session: AsyncSession = Depends(get_session)):
    """Reload the screener universe from PostgreSQL."""
    from sentinel.sse.db_loader import populate_screener_from_db
    n = await populate_screener_from_db(session)
    return {"status": "ok", "tickers_loaded": n}


@router.get("/universe/stats")
async def universe_stats():
    engine = get_engine()
    return {
        "total": engine.get_universe_count(),
        "sectors": engine.get_sector_breakdown(),
    }
