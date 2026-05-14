"""Macro routes — FRED series, yield curve, COT, regime.

DB-first pattern: check macro_data table before calling FRED live.
Reduces API calls and serves historical data at DB speed.
Pass as_of for ALFRED point-in-time queries (prevents look-ahead in backtests).
"""
from datetime import date, datetime
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.sds.db import get_session

router = APIRouter()


@router.get("/series/{series_id}")
async def get_fred_series(
    series_id: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
    as_of: Optional[date] = Query(
        None,
        description="Point-in-time date — only return vintages released on or before this date",
    ),
    limit: int = Query(500, ge=1, le=10000),
    session: AsyncSession = Depends(get_session),
):
    """Serve a FRED series. Checks DB first; falls back to live FRED API if empty."""
    from sentinel.sds import repository
    from sentinel.sds.ingest import ingest_macro_series

    start_dt = datetime.combine(start, datetime.min.time()) if start else datetime(1900, 1, 1)
    end_dt = datetime.combine(end, datetime.max.time()) if end else datetime.utcnow()
    as_of_dt = datetime.combine(as_of, datetime.max.time()) if as_of else None

    # DB-first: check if we already have data for this series
    rows = await repository.get_macro_series(
        series_id=series_id,
        start=start_dt,
        end=end_dt,
        session=session,
        as_of=as_of_dt,
    )

    source = "db"
    if not rows:
        # Live fallback — ingest and return fresh
        result = await ingest_macro_series(series_id=series_id, session=session, start=start)
        if result.ok:
            rows = await repository.get_macro_series(
                series_id=series_id,
                start=start_dt,
                end=end_dt,
                session=session,
                as_of=as_of_dt,
            )
            source = "fred_live"

    observations = [
        {
            "date": r["time"].date().isoformat() if hasattr(r["time"], "date") else str(r["time"]),
            "value": float(r["value"]),
            "vintage": r["vintage"].isoformat() if r.get("vintage") else None,
        }
        for r in rows[-limit:]
    ]

    return {
        "series_id": series_id,
        "count": len(observations),
        "source": source,
        "as_of": as_of.isoformat() if as_of else None,
        "pit_enforced": as_of is not None,
        "observations": observations,
    }


@router.get("/yield-curve")
async def yield_curve(
    as_of: Optional[date] = Query(None, description="Point-in-time date for historical curve"),
    session: AsyncSession = Depends(get_session),
):
    """Current (or historical) Treasury yield curve. Served from DB when populated."""
    import asyncio
    from sentinel.sds import repository

    tenor_series = {
        "1M": "DGS1MO", "3M": "DGS3MO", "6M": "DGS6MO", "1Y": "DGS1",
        "2Y": "DGS2", "5Y": "DGS5", "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30",
    }
    as_of_dt = datetime.combine(as_of, datetime.max.time()) if as_of else None
    cutoff = date(2000, 1, 1)
    start_dt = datetime.combine(cutoff, datetime.min.time())
    end_dt = as_of_dt or datetime.utcnow()

    rows_map = await asyncio.gather(*[
        repository.get_macro_series(
            series_id=sid, start=start_dt, end=end_dt,
            session=session, as_of=as_of_dt,
        )
        for sid in tenor_series.values()
    ])

    curve = {}
    for tenor, rows in zip(tenor_series.keys(), rows_map):
        if rows:
            latest = rows[-1]
            t = latest["time"]
            curve[tenor] = {
                "rate": float(latest["value"]),
                "date": t.date().isoformat() if hasattr(t, "date") else str(t),
            }

    return {
        "yield_curve": curve,
        "as_of": as_of.isoformat() if as_of else None,
        "pit_enforced": as_of is not None,
    }


@router.get("/core")
async def core_series(session: AsyncSession = Depends(get_session)):
    """Latest value for all core macro series. Triggers live ingest for any missing series."""
    from sqlalchemy import text
    from sentinel.sds.ingest import ingest_macro_series

    # Pull most recent observation per series in one query
    result = await session.execute(text("""
        SELECT DISTINCT ON (series_id)
               series_id,
               time AS latest_date,
               value AS latest_value,
               vintage
        FROM macro_data
        ORDER BY series_id, time DESC
    """))
    rows = {r["series_id"]: r for r in result.mappings()}

    # For any core series missing from DB, ingest live
    CORE = [
        "GDP", "CPIAUCSL", "UNRATE", "FEDFUNDS", "T10Y2Y", "VIXCLS",
        "DGS10", "DGS2", "SOFR", "T10YIE", "PAYEMS", "M2SL",
        "BAMLH0A0HYM2", "DEXUSEU", "DEXJPUS",
    ]
    missing = [s for s in CORE if s not in rows]
    for series_id in missing:
        res = await ingest_macro_series(series_id=series_id, session=session)
        if res.ok:
            # Re-query the single series we just ingested
            fresh = await session.execute(
                text("""
                    SELECT DISTINCT ON (series_id)
                           series_id, time AS latest_date, value AS latest_value, vintage
                    FROM macro_data
                    WHERE series_id = :sid
                    ORDER BY series_id, time DESC
                """),
                {"sid": series_id},
            )
            row = fresh.mappings().fetchone()
            if row:
                rows[series_id] = dict(row)

    series_out = {}
    for sid, r in rows.items():
        t = r["latest_date"]
        series_out[sid] = {
            "latest_value": float(r["latest_value"]) if r.get("latest_value") is not None else None,
            "latest_date": t.date().isoformat() if hasattr(t, "date") else str(t),
            "vintage": r["vintage"].isoformat() if r.get("vintage") else None,
        }

    return {
        "series": series_out,
        "total": len(series_out),
        "missing_from_db": missing,
    }
