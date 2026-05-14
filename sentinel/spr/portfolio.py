"""Portfolio NAV tracking, Brinson-Hood-Beebower attribution, and optimizer integration."""
from __future__ import annotations
from datetime import date, datetime
from decimal import Decimal
from typing import Optional
import numpy as np
import pandas as pd
from sentinel.core.logging import get_logger

logger = get_logger(__name__)


class Portfolio:
    """
    Tracks positions, computes NAV, and produces performance attribution.
    Positions stored as {ticker: {figi, shares, avg_cost}}.
    """

    def __init__(self, name: str, cash: float = 100_000.0) -> None:
        self.name = name
        self._cash = cash
        self._initial_cash = cash
        self._positions: dict[str, dict] = {}
        self._transactions: list[dict] = []
        self._nav_history: list[tuple[date, float]] = []

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict:
        return dict(self._positions)

    def buy(self, ticker: str, shares: float, price: float, figi: str = "") -> bool:
        cost = shares * price
        if cost > self._cash:
            logger.warning("Insufficient cash", ticker=ticker, needed=cost, available=self._cash)
            return False
        self._cash -= cost
        if ticker in self._positions:
            pos = self._positions[ticker]
            total_shares = pos["shares"] + shares
            total_cost = pos["shares"] * pos["avg_cost"] + cost
            self._positions[ticker] = {
                "figi": figi or pos["figi"],
                "shares": total_shares,
                "avg_cost": total_cost / total_shares,
            }
        else:
            self._positions[ticker] = {"figi": figi, "shares": shares, "avg_cost": price}
        self._transactions.append({
            "date": date.today().isoformat(), "type": "buy",
            "ticker": ticker, "shares": shares, "price": price, "value": cost,
        })
        return True

    def sell(self, ticker: str, shares: float, price: float) -> bool:
        if ticker not in self._positions or self._positions[ticker]["shares"] < shares:
            logger.warning("Insufficient shares", ticker=ticker)
            return False
        proceeds = shares * price
        self._cash += proceeds
        self._positions[ticker]["shares"] -= shares
        if self._positions[ticker]["shares"] < 1e-6:
            del self._positions[ticker]
        self._transactions.append({
            "date": date.today().isoformat(), "type": "sell",
            "ticker": ticker, "shares": shares, "price": price, "value": proceeds,
        })
        return True

    def compute_nav(self, current_prices: dict[str, float]) -> float:
        equity_value = sum(
            pos["shares"] * current_prices.get(ticker, pos["avg_cost"])
            for ticker, pos in self._positions.items()
        )
        nav = self._cash + equity_value
        self._nav_history.append((date.today(), nav))
        return nav

    def get_weights(self, current_prices: dict[str, float]) -> dict[str, float]:
        nav = self.compute_nav(current_prices)
        if nav <= 0:
            return {}
        return {
            ticker: pos["shares"] * current_prices.get(ticker, pos["avg_cost"]) / nav
            for ticker, pos in self._positions.items()
        }

    def compute_returns(self) -> pd.Series:
        if len(self._nav_history) < 2:
            return pd.Series(dtype=float)
        dates, navs = zip(*self._nav_history)
        s = pd.Series(navs, index=pd.DatetimeIndex(dates))
        return s.pct_change().dropna()

    def total_return(self, current_prices: dict[str, float]) -> float:
        nav = self.compute_nav(current_prices)
        return (nav - self._initial_cash) / self._initial_cash

    def unrealized_pnl(self, current_prices: dict[str, float]) -> dict[str, float]:
        pnl = {}
        for ticker, pos in self._positions.items():
            price = current_prices.get(ticker, pos["avg_cost"])
            pnl[ticker] = pos["shares"] * (price - pos["avg_cost"])
        return pnl


def compute_var_cvar(returns: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """
    Historical simulation VaR and CVaR at given confidence level.
    Returns (VaR, CVaR) as positive numbers (loss magnitude).
    """
    if returns.empty:
        return 0.0, 0.0
    sorted_returns = returns.sort_values()
    cutoff_idx = int(len(sorted_returns) * (1 - confidence))
    var = abs(float(sorted_returns.iloc[cutoff_idx]))
    cvar = abs(float(sorted_returns.iloc[:cutoff_idx].mean()))
    return var, cvar


def brinson_attribution(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
    portfolio_returns: dict[str, float],
    benchmark_returns: dict[str, float],
    sector_map: dict[str, str],
) -> dict:
    """
    Brinson-Hood-Beebower performance attribution.
    Returns allocation effect, selection effect, interaction effect per sector.
    """
    sectors = set(sector_map.values())
    results = {}

    for sector in sectors:
        sector_tickers_p = {t for t, s in sector_map.items() if s == sector and t in portfolio_weights}
        sector_tickers_b = {t for t, s in sector_map.items() if s == sector and t in benchmark_weights}

        # Sector weights
        wp = sum(portfolio_weights.get(t, 0) for t in sector_tickers_p)
        wb = sum(benchmark_weights.get(t, 0) for t in sector_tickers_b)

        # Sector returns (value-weighted)
        rp = (sum(portfolio_weights.get(t, 0) * portfolio_returns.get(t, 0) for t in sector_tickers_p) / wp
              if wp > 0 else 0.0)
        rb = (sum(benchmark_weights.get(t, 0) * benchmark_returns.get(t, 0) for t in sector_tickers_b) / wb
              if wb > 0 else 0.0)

        # Total benchmark return (for allocation effect)
        rb_total = sum(w * benchmark_returns.get(t, 0) for t, w in benchmark_weights.items())

        allocation = (wp - wb) * (rb - rb_total)
        selection = wb * (rp - rb)
        interaction = (wp - wb) * (rp - rb)

        results[sector] = {
            "portfolio_weight": round(wp, 4),
            "benchmark_weight": round(wb, 4),
            "portfolio_return": round(rp, 4),
            "benchmark_return": round(rb, 4),
            "allocation_effect": round(allocation, 4),
            "selection_effect": round(selection, 4),
            "interaction_effect": round(interaction, 4),
            "total_effect": round(allocation + selection + interaction, 4),
        }

    return results


def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    method: str = "max_sharpe",
    risk_free_rate: float = 0.05,
    weight_bounds: tuple = (0.0, 0.20),
) -> dict[str, float]:
    """
    Optimize portfolio weights using PyPortfolioOpt.
    Methods: 'max_sharpe', 'min_volatility', 'max_quadratic_utility', 'efficient_risk'
    """
    from pypfopt import EfficientFrontier, expected_returns as er, risk_models
    from pypfopt.discrete_allocation import DiscreteAllocation

    ef = EfficientFrontier(expected_returns, cov_matrix, weight_bounds=weight_bounds)

    if method == "max_sharpe":
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    elif method == "min_volatility":
        ef.min_volatility()
    elif method == "max_quadratic_utility":
        ef.max_quadratic_utility(risk_aversion=1)
    else:
        ef.max_sharpe(risk_free_rate=risk_free_rate)

    weights = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
    logger.info("Portfolio optimized", method=method, expected_return=round(perf[0], 4),
                volatility=round(perf[1], 4), sharpe=round(perf[2], 4))

    return {ticker: round(w, 4) for ticker, w in weights.items() if w > 0.001}
