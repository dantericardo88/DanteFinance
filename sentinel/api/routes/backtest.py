"""Backtest routes."""
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sentinel.sds.normalizer import fetch_ohlcv_with_fallback
from sentinel.sds import get_all_adapters, build_default_adapters
from sentinel.sbe.runner import VectorBTRunner
from sentinel.sbe.strategies import get_strategy_fn

router = APIRouter()


class BacktestRequest(BaseModel):
    ticker: str
    strategy: str
    start: str
    end: str
    params: Optional[dict] = None
    n_trials: int = 1


@router.post("/run")
async def run_backtest(req: BacktestRequest):
    """Run a named backtest strategy."""
    if not get_all_adapters():
        build_default_adapters()
    bars = await fetch_ohlcv_with_fallback(
        req.ticker,
        datetime.fromisoformat(req.start),
        datetime.fromisoformat(req.end),
    )
    if not bars:
        raise HTTPException(404, f"No price data for {req.ticker}")

    import pandas as pd
    prices = pd.Series(
        [float(b.close) for b in bars],
        index=pd.DatetimeIndex([b.time for b in bars]),
    )

    try:
        signal_fn = get_strategy_fn(req.strategy, **(req.params or {}))
        entries, exits = signal_fn(prices)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    runner = VectorBTRunner()
    metrics = runner.run(prices, entries, exits,
                         strategy_id=f"{req.strategy}_{req.ticker}",
                         n_trials=req.n_trials)

    return {
        "ticker": req.ticker,
        "strategy": req.strategy,
        "params": req.params,
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
            "cvar_95": float(metrics.cvar_95),
            "beta": float(metrics.beta),
            "alpha": float(metrics.alpha),
            "skewness": float(metrics.skewness),
            "kurtosis": float(metrics.kurtosis),
        },
    }
