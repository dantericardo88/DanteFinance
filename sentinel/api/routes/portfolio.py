"""Portfolio routes — positions, performance, allocation, risk analytics.

Data sources (in priority order):
  1. strategy_registry + backtest_results tables — strategy performance from DB
  2. Alpaca paper/live account — real positions (requires ALPACA_API_KEY)

No live trading key required to use portfolio analytics — all backtest-derived
metrics work with zero broker configuration.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.sds.db import get_session

router = APIRouter()


# ── Overview ──────────────────────────────────────────────────────────────────

@router.get("/")
async def portfolio_overview(session: AsyncSession = Depends(get_session)):
    """Portfolio dashboard — active strategies, best performers, live equity."""
    # Strategies from registry
    strats_result = await session.execute(text("""
        SELECT strategy_id, name, status, status_since, kill_switch, tags, created_at
        FROM strategy_registry
        ORDER BY created_at DESC
        LIMIT 50
    """))
    strategies = [dict(r) for r in strats_result.mappings()]

    # Best backtest results
    best_result = await session.execute(text("""
        SELECT strategy_id, ticker, total_return, cagr, sharpe_ratio,
               max_drawdown, win_rate, start_date, end_date
        FROM backtest_results
        ORDER BY sharpe_ratio DESC NULLS LAST
        LIMIT 20
    """))
    best_backtests = [dict(r) for r in best_result.mappings()]

    # Aggregate stats
    agg = await session.execute(text("""
        SELECT
            COUNT(DISTINCT strategy_id)         AS total_strategies,
            COUNT(*)                             AS total_backtests,
            AVG(sharpe_ratio)                    AS avg_sharpe,
            AVG(cagr)                            AS avg_cagr,
            MAX(cagr)                            AS best_cagr,
            MIN(max_drawdown)                    AS worst_drawdown
        FROM backtest_results
    """))
    stats = dict(agg.mappings().fetchone() or {})

    # Try live positions (non-blocking)
    live_equity = None
    live_positions = []
    try:
        from sentinel.see.broker import AlpacaBroker
        from sentinel.core.config import get_settings
        s = get_settings()
        if s.has_alpaca:
            broker = AlpacaBroker(
                api_key=s.alpaca_api_key,
                secret_key=s.alpaca_secret_key,
                paper=True,
            )
            acct = await broker.get_account()
            live_equity = float(acct.get("equity", 0))
            live_positions = await broker.get_positions()
    except Exception:
        pass

    return {
        "as_of": datetime.utcnow().isoformat(),
        "live": {
            "equity": live_equity,
            "positions": live_positions,
            "broker": "alpaca_paper" if live_equity else None,
        },
        "strategies": {
            "total": len(strategies),
            "active": [s for s in strategies if s.get("status") not in ("RETIRED", "PAUSED")],
        },
        "backtests": {
            "stats": {k: round(float(v), 4) if v is not None else None for k, v in stats.items()},
            "top_by_sharpe": best_backtests[:10],
        },
    }


# ── Strategies ────────────────────────────────────────────────────────────────

@router.get("/strategies")
async def list_strategies(
    status: Optional[str] = Query(None, description="Filter by status: BACKTEST, PAPER, CAPPED_LIVE"),
    session: AsyncSession = Depends(get_session),
):
    """All strategies in the registry with their promotion history."""
    filters = []
    params: dict = {}
    if status:
        filters.append("status = :status")
        params["status"] = status.upper()

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    result = await session.execute(
        text(f"""
            SELECT strategy_id, name, description, status, status_since,
                   kill_switch, params, tags, created_at, updated_at
            FROM strategy_registry
            {where}
            ORDER BY updated_at DESC
        """),
        params,
    )
    strategies = [dict(r) for r in result.mappings()]

    # Attach best backtest for each strategy
    for s in strategies:
        bt = await session.execute(
            text("""
                SELECT sharpe_ratio, cagr, max_drawdown, total_return,
                       start_date, end_date, n_trials
                FROM backtest_results
                WHERE strategy_id = :sid
                ORDER BY sharpe_ratio DESC NULLS LAST
                LIMIT 1
            """),
            {"sid": s["strategy_id"]},
        )
        row = bt.mappings().fetchone()
        s["best_backtest"] = dict(row) if row else None

    return {"count": len(strategies), "strategies": strategies}


# ── Performance ───────────────────────────────────────────────────────────────

@router.get("/performance")
async def portfolio_performance(
    strategy_id: Optional[str] = None,
    ticker: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
):
    """Backtest performance records — sortable by any metric."""
    filters = []
    params: dict = {"limit": limit}
    if strategy_id:
        filters.append("strategy_id = :strategy_id")
        params["strategy_id"] = strategy_id
    if ticker:
        filters.append("ticker = :ticker")
        params["ticker"] = ticker.upper()

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    result = await session.execute(
        text(f"""
            SELECT strategy_id, ticker, start_date, end_date, interval,
                   total_return, cagr, sharpe_ratio, sortino_ratio, calmar_ratio,
                   deflated_sharpe, max_drawdown, win_rate, volatility,
                   var_95, cvar_95, beta, alpha, n_trials, params, created_at
            FROM backtest_results
            {where}
            ORDER BY sharpe_ratio DESC NULLS LAST
            LIMIT :limit
        """),
        params,
    )
    records = [dict(r) for r in result.mappings()]

    # Summary stats
    if records:
        returns = [r["total_return"] for r in records if r.get("total_return")]
        sharpes = [r["sharpe_ratio"] for r in records if r.get("sharpe_ratio")]
        summary = {
            "count": len(records),
            "avg_return": round(sum(returns) / len(returns), 4) if returns else None,
            "avg_sharpe": round(sum(sharpes) / len(sharpes), 4) if sharpes else None,
            "best_sharpe": round(max(sharpes), 4) if sharpes else None,
            "best_return": round(max(returns), 4) if returns else None,
        }
    else:
        summary = {"count": 0}

    return {"summary": summary, "records": records}


# ── Positions (live) ───────────────────────────────────────────────────────────

@router.get("/positions")
async def live_positions():
    """Current positions from Alpaca paper/live account."""
    from sentinel.core.config import get_settings
    s = get_settings()
    if not s.has_alpaca:
        return {
            "positions": [],
            "note": "No Alpaca credentials — set ALPACA_API_KEY + ALPACA_SECRET_KEY in .env",
        }
    from sentinel.see.broker import AlpacaBroker
    broker = AlpacaBroker(
        api_key=s.alpaca_api_key,
        secret_key=s.alpaca_secret_key,
        paper=not s.is_live_trading_enabled,
    )
    positions = await broker.get_positions()
    acct = await broker.get_account()
    return {
        "account": {
            "equity": float(acct.get("equity", 0)),
            "cash": float(acct.get("cash", 0)),
            "buying_power": float(acct.get("buying_power", 0)),
            "paper": not s.is_live_trading_enabled,
        },
        "positions": positions,
        "count": len(positions),
    }


# ── Allocation ────────────────────────────────────────────────────────────────

@router.get("/allocation")
async def portfolio_allocation(
    strategy_id: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    """Allocation breakdown — by ticker, by interval, by strategy tier."""
    filters = []
    params: dict = {}
    if strategy_id:
        filters.append("strategy_id = :strategy_id")
        params["strategy_id"] = strategy_id
    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    by_ticker = await session.execute(
        text(f"""
            SELECT ticker,
                   COUNT(*)            AS backtest_count,
                   AVG(sharpe_ratio)   AS avg_sharpe,
                   AVG(cagr)           AS avg_cagr,
                   MAX(cagr)           AS best_cagr
            FROM backtest_results
            {where}
            GROUP BY ticker
            ORDER BY avg_sharpe DESC NULLS LAST
            LIMIT 30
        """),
        params,
    )

    by_status = await session.execute(text("""
        SELECT status, COUNT(*) AS count
        FROM strategy_registry
        GROUP BY status
        ORDER BY count DESC
    """))

    return {
        "by_ticker": [dict(r) for r in by_ticker.mappings()],
        "by_strategy_status": [dict(r) for r in by_status.mappings()],
    }


# ── Risk ──────────────────────────────────────────────────────────────────────

@router.get("/risk")
async def portfolio_risk(
    strategy_id: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    """Aggregate risk metrics across all or one strategy's backtests."""
    filters = []
    params: dict = {}
    if strategy_id:
        filters.append("strategy_id = :strategy_id")
        params["strategy_id"] = strategy_id
    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    result = await session.execute(
        text(f"""
            SELECT
                AVG(volatility)         AS avg_volatility,
                MAX(ABS(max_drawdown))  AS worst_drawdown,
                AVG(var_95)             AS avg_var_95,
                AVG(cvar_95)            AS avg_cvar_95,
                AVG(beta)               AS avg_beta,
                AVG(deflated_sharpe)    AS avg_dsr,
                COUNT(*)                AS sample_size
            FROM backtest_results
            {where}
        """),
        params,
    )
    risk = dict(result.mappings().fetchone() or {})
    return {
        "risk_metrics": {k: round(float(v), 4) if v is not None else None for k, v in risk.items()},
        "strategy_id": strategy_id,
        "note": "Metrics derived from backtest history. Live risk requires active positions.",
    }


# ─── Factor Model ─────────────────────────────────────────────────────────────
@router.post("/factor-exposure")
async def get_factor_exposure(holdings: dict[str, float]):
    """Fama-French 5-factor + momentum decomposition for a portfolio."""
    from sentinel.spr.factor_model import decompose_portfolio
    from datetime import date, timedelta
    import pandas as pd
    end = date.today()
    start = end - timedelta(days=365)
    result = decompose_portfolio(holdings=holdings, returns_df=pd.DataFrame(), start=start, end=end)
    return result.model_dump()

# ─── Stress Test ──────────────────────────────────────────────────────────────
@router.post("/stress-test")
async def run_stress_test_endpoint(holdings: dict[str, float], portfolio_value: float = 100_000.0):
    """Run portfolio through historical crisis scenarios and parametric shocks."""
    from sentinel.spr.stress_test import run_stress_test
    import pandas as pd
    report = run_stress_test(holdings=holdings, portfolio_value=portfolio_value, returns_df=pd.DataFrame())
    return report.model_dump()

# ─── DCF Valuation ────────────────────────────────────────────────────────────
@router.get("/dcf/{ticker}")
async def run_dcf_endpoint(
    ticker: str,
    current_price: float = Query(...),
    wacc: float = Query(0.10),
    terminal_growth: float = Query(0.025),
):
    """DCF intrinsic value via Damodaran methodology."""
    from sentinel.sfe.dcf_model import DCFAssumptions, run_dcf
    assumptions = DCFAssumptions(
        ticker=ticker, revenue_base=1_000_000_000,
        revenue_growth_rates=[0.10, 0.09, 0.08, 0.07, 0.06],
        terminal_growth_rate=terminal_growth, ebit_margin=0.15, tax_rate=0.21,
        capex_pct_revenue=0.05, da_pct_revenue=0.04, nwc_change_pct_revenue=0.02,
        wacc=wacc, net_debt=0.0, shares_outstanding=100.0,
    )
    result = run_dcf(assumptions, current_price)
    return result.model_dump()
