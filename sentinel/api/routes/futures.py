"""Continuous futures API endpoints."""
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.sds.db import get_session

router = APIRouter()


@router.get("/continuous/{root}")
async def get_continuous_series(
    root: str,
    start: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    method: str = Query("panama", description="panama | ratio | unadj"),
    session: AsyncSession = Depends(get_session),
):
    """Return a back-adjusted continuous futures series for the given root symbol.

    Reads raw contract bars from the ohlcv table and stitches them using the
    chosen adjustment method.

    - **root**: CME root symbol, e.g. ``ES``, ``NQ``, ``CL``, ``GC``
    - **method**: ``panama`` (additive), ``ratio`` (multiplicative), ``unadj``
    """
    try:
        start_date = date.fromisoformat(start) if start else date(2010, 1, 1)
        end_date   = date.fromisoformat(end)   if end   else date.today()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date: {exc}") from exc

    if method not in ("panama", "ratio", "unadj"):
        raise HTTPException(status_code=422, detail="method must be panama | ratio | unadj")

    try:
        from sentinel.sds.continuous_futures import build_continuous_series, ContractSpec

        # Use well-known spec if available, else build a generic quarterly spec
        _WELL_KNOWN = {
            "ES": ContractSpec.es,
            "NQ": ContractSpec.nq,
            "CL": ContractSpec.cl,
            "GC": ContractSpec.gc,
            "ZB": ContractSpec.zb,
        }
        spec_fn = _WELL_KNOWN.get(root.upper())
        spec = spec_fn() if spec_fn else ContractSpec(
            root=root.upper(), exchange="CME", months=[3, 6, 9, 12]
        )

        bars = await build_continuous_series(session, spec, start_date, end_date, method)

        return {
            "root": root.upper(),
            "method": method,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "count": len(bars),
            "bars": [
                {
                    "time": b.time.isoformat(),
                    "open": float(b.open),
                    "high": float(b.high),
                    "low":  float(b.low),
                    "close": float(b.close),
                    "volume": b.volume,
                }
                for b in bars
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/specs")
async def list_contract_specs():
    """Return the list of well-known continuous futures specs built into SENTINEL."""
    from sentinel.sds.continuous_futures import ContractSpec
    specs = {
        "ES": ContractSpec.es(),
        "NQ": ContractSpec.nq(),
        "CL": ContractSpec.cl(),
        "GC": ContractSpec.gc(),
        "ZB": ContractSpec.zb(),
    }
    return {
        root: {
            "exchange": s.exchange,
            "months": s.months,
            "roll_days_before": s.roll_days_before,
            "yfinance_ticker": f"{root}=F",
        }
        for root, s in specs.items()
    }


@router.post("/build")
async def build_and_persist(
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Build a continuous series and write it back to the ohlcv table.

    Body: ``{"root": "ES", "start": "2010-01-01", "end": "2024-12-31",
             "method": "panama"}``
    """
    root   = body.get("root", "ES").upper()
    method = body.get("method", "panama")
    try:
        start_date = date.fromisoformat(body["start"]) if "start" in body else date(2010, 1, 1)
        end_date   = date.fromisoformat(body["end"])   if "end"   in body else date.today()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date: {exc}") from exc

    try:
        from sentinel.sds.continuous_futures import (
            build_and_persist_continuous, ContractSpec,
        )
        _WELL_KNOWN = {
            "ES": ContractSpec.es, "NQ": ContractSpec.nq,
            "CL": ContractSpec.cl, "GC": ContractSpec.gc, "ZB": ContractSpec.zb,
        }
        spec_fn = _WELL_KNOWN.get(root)
        spec = spec_fn() if spec_fn else ContractSpec(
            root=root, exchange="CME", months=[3, 6, 9, 12]
        )
        bars_written = await build_and_persist_continuous(session, spec, start_date, end_date, method)
        return {"status": "ok", "root": root, "method": method, "bars_written": bars_written}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
