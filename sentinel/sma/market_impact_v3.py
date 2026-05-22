"""
Almgren-Chriss (2001) optimal execution framework and transaction cost analysis.
Pure numpy/scipy — zero network calls, zero external data deps.

dim_121 — Market impact modeling / Almgren-Chriss optimal execution (target: 9)

Classes
-------
ACParams
    Almgren-Chriss market impact parameters (sigma, gamma, eta, epsilon, tau).

ExecutionPlan
    Optimal or benchmark trajectory with holdings, trades, expected cost, variance.

AlmgrenChriss
    Full A-C model:
    .optimal_trajectory()  → closed-form optimal VWAP/risk-averse trajectory
    .twap_trajectory()     → uniform (TWAP) liquidation
    .efficient_frontier()  → (E[C], Var[C]) curve across lambda values
    .expected_cost()       → model E[C] for any plan
    .cost_variance()       → model Var[C] for any plan

ImplementationShortfall
    .compute()             → IS decomposition (bps)
    .vwap_benchmark()      → fill vs VWAP in bps

TCAAnalytics
    .participation_cost()  → sqrt-model market impact cost
    .kyle_lambda()         → OLS price impact coefficient
    .market_impact_report()→ formatted dict of cost metrics

Convenience functions
---------------------
optimal_execution    → ExecutionPlan
twap_execution       → ExecutionPlan
compute_is           → IS dict
kyle_lambda          → float
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ACParams:
    """Almgren-Chriss market impact parameters.

    Parameters
    ----------
    sigma : float
        Daily price volatility ($/share/day^0.5).
    gamma : float
        Permanent impact coefficient (linear permanent impact g(v) = gamma * v).
    eta : float
        Temporary impact coefficient (linear temporary impact h(v) = eta * v).
    epsilon : float
        Fixed transaction cost — half bid-ask spread ($/share).
    tau : float
        Trading interval length in days (default 1.0).
    """
    sigma: float
    gamma: float
    eta: float
    epsilon: float
    tau: float = 1.0


@dataclass
class ExecutionPlan:
    """Execution trajectory produced by AlmgrenChriss.

    Attributes
    ----------
    holdings : np.ndarray
        x_j — shares held at each time step j (length n+1, starts at X, ends at 0).
    trades : np.ndarray
        v_j — shares traded at each step j (length n).  Positive = selling.
    times : np.ndarray
        Time points t_j in days (length n+1).
    expected_cost : float
        E[C] from the A-C model in dollars.
    cost_variance : float
        Var[C] from the A-C model in dollars^2.
    risk_aversion : float
        Lambda (risk aversion) used to generate this plan.
    """
    holdings: np.ndarray
    trades: np.ndarray
    times: np.ndarray
    expected_cost: float
    cost_variance: float
    risk_aversion: float

    @property
    def trajectory_type(self) -> str:
        """Classify trajectory shape relative to TWAP."""
        n = len(self.trades)
        if n < 2:
            return "trivial"
        twap_rate = self.holdings[0] / n
        first_half_avg = np.mean(self.trades[: n // 2])
        second_half_avg = np.mean(self.trades[n // 2 :])
        tol = 0.05 * twap_rate
        if abs(first_half_avg - twap_rate) <= tol and abs(second_half_avg - twap_rate) <= tol:
            return "TWAP"
        if first_half_avg > second_half_avg + tol:
            return "front-loaded"
        if second_half_avg > first_half_avg + tol:
            return "back-loaded"
        return "optimal"


# ──────────────────────────────────────────────────────────────────────────────
# Core Almgren-Chriss model
# ──────────────────────────────────────────────────────────────────────────────

class AlmgrenChriss:
    """Almgren-Chriss (2001) optimal execution model.

    Parameters
    ----------
    params : ACParams
        Market impact parameters.
    """

    def __init__(self, params: ACParams) -> None:
        self.p = params

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _kappa(self, lam: float) -> float:
        """Compute kappa from risk aversion lambda.

        kappa^2 = lambda * sigma^2 / (eta - gamma * tau / 2)

        Returns kappa >= 0.  When denominator <= 0 (very small eta), clamps
        the result to a small positive value to preserve monotone decreasing
        liquidation.
        """
        p = self.p
        denom = p.eta - p.gamma * p.tau / 2.0
        if denom <= 0:
            # Fall back: treat as if gamma=0
            denom = p.eta
        kappa2 = lam * (p.sigma ** 2) / denom
        return float(np.sqrt(max(kappa2, 0.0)))

    def _discrete_holdings(self, X: float, T: float, n: int, kappa: float) -> np.ndarray:
        """Discrete optimal holdings x_j for j = 0 .. n.

        x_j = X * sinh(kappa*(T - j*tau)) / sinh(kappa*T)

        When kappa is near 0 (risk-neutral limit) sinh expansion reduces to
        the linear TWAP trajectory.
        """
        tau = T / n
        times = np.arange(n + 1) * tau  # [0, tau, 2tau, ..., T]
        remaining = T - times            # [T, T-tau, ..., 0]
        kT = kappa * T
        if kT < 1e-8:
            # Linear limit: x_j = X * (T - j*tau) / T
            holdings = X * remaining / T
        else:
            holdings = X * np.sinh(kappa * remaining) / np.sinh(kT)
        # Enforce boundary conditions exactly
        holdings[0] = X
        holdings[-1] = 0.0
        return holdings

    # ------------------------------------------------------------------
    # Public trajectory methods
    # ------------------------------------------------------------------

    def optimal_trajectory(
        self, X: float, T: float, n: int, lam: float
    ) -> ExecutionPlan:
        """Compute the Almgren-Chriss optimal liquidation trajectory.

        Parameters
        ----------
        X : float
            Shares to liquidate (positive = sell).
        T : float
            Total trading horizon in days.
        n : int
            Number of trading intervals.
        lam : float
            Risk aversion coefficient lambda.

        Returns
        -------
        ExecutionPlan
        """
        tau = T / n
        kappa = self._kappa(lam)
        holdings = self._discrete_holdings(X, T, n, kappa)
        trades = (holdings[:-1] - holdings[1:]) / tau  # v_j = (x_{j-1} - x_j)/tau
        times = np.arange(n + 1) * tau

        ec = self._expected_cost(X, holdings, trades, tau)
        vc = self._cost_variance(holdings, tau)
        return ExecutionPlan(
            holdings=holdings,
            trades=trades,
            times=times,
            expected_cost=ec,
            cost_variance=vc,
            risk_aversion=lam,
        )

    def twap_trajectory(self, X: float, T: float, n: int) -> ExecutionPlan:
        """Uniform (TWAP) liquidation — equal shares per interval.

        Parameters
        ----------
        X : float
            Shares to liquidate.
        T : float
            Total horizon in days.
        n : int
            Number of trading intervals.
        """
        tau = T / n
        holdings = X * np.linspace(1.0, 0.0, n + 1)
        trades = np.full(n, X / (n * tau))  # constant rate v = X / T
        times = np.arange(n + 1) * tau

        ec = self._expected_cost(X, holdings, trades, tau)
        vc = self._cost_variance(holdings, tau)
        return ExecutionPlan(
            holdings=holdings,
            trades=trades,
            times=times,
            expected_cost=ec,
            cost_variance=vc,
            risk_aversion=0.0,
        )

    def efficient_frontier(
        self,
        X: float,
        T: float,
        n: int,
        lambdas: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the efficient frontier across a range of risk-aversion values.

        Parameters
        ----------
        X, T, n : float, float, int
            Liquidation parameters (same as optimal_trajectory).
        lambdas : np.ndarray
            Array of lambda values to sweep.

        Returns
        -------
        expected_costs : np.ndarray
        variances : np.ndarray
        """
        costs = np.empty(len(lambdas))
        variances = np.empty(len(lambdas))
        for i, lam in enumerate(lambdas):
            plan = self.optimal_trajectory(X, T, n, lam)
            costs[i] = plan.expected_cost
            variances[i] = plan.cost_variance
        return costs, variances

    # ------------------------------------------------------------------
    # Cost formulae
    # ------------------------------------------------------------------

    def _expected_cost(
        self,
        X: float,
        holdings: np.ndarray,
        trades: np.ndarray,
        tau: float,
    ) -> float:
        """Almgren-Chriss E[C]:

        E[C] = epsilon * sum(|v_j|*tau) + (gamma/2)*X^2 + eta*tau*sum(v_j^2)

        where v_j are trading rates (shares/day).
        """
        p = self.p
        # Spread cost: epsilon per share (flat)
        spread_cost = p.epsilon * np.sum(np.abs(trades) * tau)
        # Permanent impact: gamma/2 * X^2 (market-wide, independent of schedule)
        perm_cost = 0.5 * p.gamma * X ** 2
        # Temporary impact: eta * tau * sum(v_j^2)
        temp_cost = p.eta * tau * np.sum(trades ** 2)
        return float(spread_cost + perm_cost + temp_cost)

    def _cost_variance(self, holdings: np.ndarray, tau: float) -> float:
        """Almgren-Chriss Var[C] = sigma^2 * tau * sum(x_j^2) over j=0..n-1."""
        return float(self.p.sigma ** 2 * tau * np.sum(holdings[:-1] ** 2))

    def expected_cost(self, plan: ExecutionPlan) -> float:
        """Re-compute E[C] from a plan (uses stored trades/holdings)."""
        tau = plan.times[1] - plan.times[0] if len(plan.times) > 1 else self.p.tau
        return self._expected_cost(
            plan.holdings[0], plan.holdings, plan.trades, tau
        )

    def cost_variance(self, plan: ExecutionPlan) -> float:
        """Re-compute Var[C] from a plan."""
        tau = plan.times[1] - plan.times[0] if len(plan.times) > 1 else self.p.tau
        return self._cost_variance(plan.holdings, tau)


# ──────────────────────────────────────────────────────────────────────────────
# Implementation Shortfall
# ──────────────────────────────────────────────────────────────────────────────

class ImplementationShortfall:
    """Compute implementation shortfall and VWAP benchmarks."""

    def compute(
        self,
        decision_price: float,
        arrival_price: float,
        fill_prices: np.ndarray,
        fill_sizes: np.ndarray,
        target_size: float,
    ) -> dict:
        """Decompose implementation shortfall into components.

        IS = (arrival_price - avg_fill_price) * shares_executed
           + (arrival_price - close_price) * shares_not_executed

        Here we approximate close_price with the last fill price, consistent
        with a same-day execution model.

        Parameters
        ----------
        decision_price : float
            Price when order was decided (pre-market).
        arrival_price : float
            Price at order arrival (market open / first quote).
        fill_prices : np.ndarray
            Array of actual fill prices.
        fill_sizes : np.ndarray
            Array of fill sizes (shares) matching fill_prices.
        target_size : float
            Total shares intended to trade.

        Returns
        -------
        dict with keys:
            IS_bps             — total IS in basis points
            paper_profit       — gain/loss from decision to arrival (dollars)
            execution_cost     — cost vs arrival price (dollars)
            opportunity_cost   — cost from unexecuted shares (dollars)
            avg_fill_price     — volume-weighted fill price
            shares_executed    — total shares executed
            shares_unexecuted  — shares not executed
        """
        fill_prices = np.asarray(fill_prices, dtype=float)
        fill_sizes = np.asarray(fill_sizes, dtype=float)

        shares_executed = float(np.sum(fill_sizes))
        shares_unexecuted = float(target_size - shares_executed)

        if shares_executed <= 0:
            avg_fill_price = arrival_price
        else:
            avg_fill_price = float(np.dot(fill_prices, fill_sizes) / shares_executed)

        # Paper profit: gain from holding from decision to arrival (positive = good)
        paper_profit = (arrival_price - decision_price) * target_size

        # Execution cost: additional cost vs arrival price.
        # Positive = buyer paid more than arrival (cost to investor).
        # = (avg_fill - arrival) * shares for a buy order.
        execution_cost = (avg_fill_price - arrival_price) * shares_executed

        # Opportunity cost: unexecuted shares — estimated as cost of missing
        # the move from arrival to close (last fill approximates close).
        # Positive = close moved away from investor (missed opportunity).
        close_price = float(fill_prices[-1]) if len(fill_prices) > 0 else arrival_price
        opportunity_cost = (close_price - arrival_price) * shares_unexecuted

        # Total IS = sum of execution slippage + opportunity cost (positive = cost)
        total_is = execution_cost + opportunity_cost  # dollars
        # Convert to bps vs decision-price notional
        notional = decision_price * target_size
        is_bps = (total_is / notional * 10_000) if notional != 0 else 0.0

        return {
            "IS_bps": float(is_bps),
            "paper_profit": float(paper_profit),
            "execution_cost": float(execution_cost),
            "opportunity_cost": float(opportunity_cost),
            "avg_fill_price": float(avg_fill_price),
            "shares_executed": float(shares_executed),
            "shares_unexecuted": float(shares_unexecuted),
        }

    def vwap_benchmark(
        self,
        fill_prices: np.ndarray,
        fill_sizes: np.ndarray,
        vwap: float,
    ) -> float:
        """Compute fill performance vs VWAP benchmark in bps.

        Returns (fill_price - vwap) / vwap * 10_000 (positive = worse for buyer).

        Parameters
        ----------
        fill_prices : np.ndarray
        fill_sizes : np.ndarray
        vwap : float
            Market VWAP to benchmark against.
        """
        fill_prices = np.asarray(fill_prices, dtype=float)
        fill_sizes = np.asarray(fill_sizes, dtype=float)
        total_shares = np.sum(fill_sizes)
        if total_shares == 0 or vwap == 0:
            return 0.0
        avg_fill = float(np.dot(fill_prices, fill_sizes) / total_shares)
        return float((avg_fill - vwap) / vwap * 10_000)


# ──────────────────────────────────────────────────────────────────────────────
# TCA Analytics
# ──────────────────────────────────────────────────────────────────────────────

class TCAAnalytics:
    """Transaction cost analysis utilities."""

    def participation_cost(
        self,
        adv_fraction: float,
        sigma: float,
        participation_rate: float,
    ) -> float:
        """Square-root market impact model.

        cost = sigma * sqrt(participation_rate) * adv_fraction

        Parameters
        ----------
        adv_fraction : float
            Order size as fraction of ADV (average daily volume).
        sigma : float
            Daily return volatility (fraction, e.g. 0.01 for 1%).
        participation_rate : float
            Participation rate as fraction of daily volume (e.g. 0.05 for 5%).

        Returns
        -------
        float
            Expected cost in same units as sigma (fraction of price).
        """
        return float(sigma * np.sqrt(participation_rate) * adv_fraction)

    def kyle_lambda(
        self,
        price_changes: np.ndarray,
        order_flow: np.ndarray,
    ) -> float:
        """OLS estimate of Kyle (1985) lambda (price impact coefficient).

        lambda = Cov(delta_p, order_flow) / Var(order_flow)

        Estimated via OLS regression: delta_p = lambda * order_flow + noise.

        Parameters
        ----------
        price_changes : np.ndarray
            Observed price changes (delta_p).
        order_flow : np.ndarray
            Signed order flow (positive = buy pressure).

        Returns
        -------
        float
            Kyle lambda estimate.
        """
        price_changes = np.asarray(price_changes, dtype=float)
        order_flow = np.asarray(order_flow, dtype=float)
        var_of = float(np.var(order_flow))
        if var_of < 1e-30:
            return 0.0
        cov = float(np.cov(price_changes, order_flow, bias=False)[0, 1])
        return float(cov / var_of)

    def market_impact_report(
        self,
        plan: ExecutionPlan,
        params: ACParams,
    ) -> dict:
        """Generate a summary market impact report for an execution plan.

        Returns
        -------
        dict with keys:
            expected_cost_bps  — E[C] in bps of notional (assumes $1 per share)
            cost_std_bps       — sqrt(Var[C]) in bps
            sharpe_of_cost     — E[C] / sqrt(Var[C]) (higher = more consistent cost)
            twap_vs_optimal_bps— cost saving of optimal over TWAP in bps
            trajectory_type    — plan.trajectory_type
            n_intervals        — number of trading intervals
            risk_aversion      — lambda used
        """
        X = float(plan.holdings[0])
        T = float(plan.times[-1]) if len(plan.times) > 1 else 1.0
        n = len(plan.trades)
        notional = X  # $1/share normalisation

        ec_bps = (plan.expected_cost / notional * 10_000) if notional > 0 else 0.0
        std_bps = (np.sqrt(plan.cost_variance) / notional * 10_000) if notional > 0 else 0.0
        sharpe = float(plan.expected_cost / np.sqrt(plan.cost_variance)) if plan.cost_variance > 0 else 0.0

        # Compare with TWAP
        ac = AlmgrenChriss(params)
        twap = ac.twap_trajectory(X, T, n)
        twap_vs_opt_bps = ((twap.expected_cost - plan.expected_cost) / notional * 10_000) if notional > 0 else 0.0

        return {
            "expected_cost_bps": float(ec_bps),
            "cost_std_bps": float(std_bps),
            "sharpe_of_cost": float(sharpe),
            "twap_vs_optimal_bps": float(twap_vs_opt_bps),
            "trajectory_type": plan.trajectory_type,
            "n_intervals": n,
            "risk_aversion": plan.risk_aversion,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ──────────────────────────────────────────────────────────────────────────────

def optimal_execution(
    X: float,
    T: float,
    n: int,
    params: ACParams,
    lam: float,
) -> ExecutionPlan:
    """Compute Almgren-Chriss optimal execution trajectory.

    Parameters
    ----------
    X : float
        Shares to liquidate.
    T : float
        Horizon in days.
    n : int
        Number of trading intervals.
    params : ACParams
        Market impact parameters.
    lam : float
        Risk aversion coefficient.
    """
    return AlmgrenChriss(params).optimal_trajectory(X, T, n, lam)


def twap_execution(
    X: float,
    T: float,
    n: int,
    params: ACParams,
) -> ExecutionPlan:
    """Compute uniform TWAP execution trajectory."""
    return AlmgrenChriss(params).twap_trajectory(X, T, n)


def compute_is(
    decision_price: float,
    arrival_price: float,
    fill_prices,
    fill_sizes,
    target: float,
) -> dict:
    """Compute implementation shortfall decomposition.

    Parameters
    ----------
    decision_price : float
    arrival_price : float
    fill_prices : array-like
    fill_sizes : array-like
    target : float
        Total intended order size (shares).
    """
    return ImplementationShortfall().compute(
        decision_price,
        arrival_price,
        np.asarray(fill_prices, dtype=float),
        np.asarray(fill_sizes, dtype=float),
        target,
    )


def kyle_lambda(
    price_changes,
    order_flow,
) -> float:
    """OLS estimate of Kyle lambda."""
    return TCAAnalytics().kyle_lambda(
        np.asarray(price_changes, dtype=float),
        np.asarray(order_flow, dtype=float),
    )
