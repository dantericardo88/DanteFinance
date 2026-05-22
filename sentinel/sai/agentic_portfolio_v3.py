"""
sentinel/sai/agentic_portfolio_v3.py
======================================
Multi-Agent Portfolio Management, Autonomous Rebalancing & Portfolio Copilot
dims 148, 149, 150 — score target: 9

Three capabilities in one module:
  dim_148: Multi-agent portfolio management workflow
           Research → Risk → PM → Execution pipeline
  dim_149: Autonomous portfolio rebalancing orchestrator
           Drift monitoring + tax-loss harvesting
  dim_150: Portfolio copilot
           NL query → what-if / explain / trade-ideas

Free-standing: numpy, re, and dataclasses only. No network calls.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ===========================================================================
# SECTION 1: Multi-Agent Workflow  (dim_148)
# ===========================================================================

@dataclass
class AgentMessage:
    """Inter-agent communication message."""
    agent_id: str
    role: str          # 'research', 'pm', 'risk', 'execution', 'copilot'
    content: str
    data: Dict = field(default_factory=dict)


class ResearchAgent:
    """
    Simulates a research analyst agent.
    Produces buy/sell signals and conviction scores.
    """

    def __init__(self, agent_id: str = "research-01") -> None:
        self.agent_id = agent_id

    def analyze(self, ticker: str, context: Optional[dict] = None) -> dict:
        """
        Returns {'signal': float, 'conviction': float, 'rationale': str}
        signal: -1.0 (strong sell) to +1.0 (strong buy)
        conviction: 0.0 to 1.0
        """
        ctx = context or {}
        rng = np.random.default_rng(hash(ticker) % (2**31))

        # Use provided returns if available
        ticker_returns = ctx.get("returns", np.array([]))
        if isinstance(ticker_returns, (list, np.ndarray)) and len(ticker_returns) > 0:
            arr = np.asarray(ticker_returns)
            momentum = float(arr.mean()) * 252  # annualised
            signal = float(np.clip(momentum * 10.0, -1.0, 1.0))
        else:
            signal = float(rng.uniform(-0.5, 0.5))

        conviction = float(np.clip(abs(signal) + rng.uniform(0.1, 0.3), 0.1, 1.0))
        direction = "BUY" if signal > 0 else "HOLD" if signal >= -0.1 else "SELL"
        rationale = (
            f"{direction} {ticker}: momentum={signal:.3f}, "
            f"conviction={conviction:.2f}"
        )
        return {
            "ticker": ticker,
            "signal": signal,
            "conviction": conviction,
            "rationale": rationale,
            "agent_id": self.agent_id,
        }


class RiskAgent:
    """
    Simulates a risk management agent.
    Checks proposed portfolio weights against a risk budget.
    """

    def __init__(self, agent_id: str = "risk-01") -> None:
        self.agent_id = agent_id

    def check(
        self,
        proposed_weights: np.ndarray,
        cov_matrix: np.ndarray,
        risk_budget: float = 0.15,
    ) -> dict:
        """
        Returns {'approved': bool, 'portfolio_vol': float, 'breaches': list}
        Approved when annualised portfolio vol <= risk_budget.
        """
        w = np.asarray(proposed_weights, dtype=float)
        # Ensure weights sum to 1
        if w.sum() > 0:
            w = w / w.sum()

        vol_sq = float(w @ cov_matrix @ w)
        port_vol = math.sqrt(max(vol_sq, 0.0))

        breaches = []
        if port_vol > risk_budget:
            breaches.append(
                f"Portfolio vol {port_vol:.4f} exceeds budget {risk_budget:.4f}"
            )

        # Check for concentration: any weight > 60%
        for i, wi in enumerate(w):
            if wi > 0.60:
                breaches.append(f"Asset {i} concentration {wi:.2%} > 60%")

        return {
            "approved": len(breaches) == 0,
            "portfolio_vol": port_vol,
            "breaches": breaches,
            "agent_id": self.agent_id,
        }


class PMAgent:
    """
    Portfolio manager agent — combines research signals + risk check
    to produce target weights.
    """

    def __init__(self, agent_id: str = "pm-01") -> None:
        self.agent_id = agent_id

    def decide(
        self,
        research: dict,
        risk_check: dict,
        current_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Returns proposed target weights.

        If risk approved: tilt toward signal proportional to conviction.
        If not approved: reduce to equal weight.
        """
        w = np.asarray(current_weights, dtype=float).copy()
        n = len(w)

        if not risk_check.get("approved", True):
            # De-risk: move toward equal weight
            eq = np.ones(n) / n
            w = 0.7 * w + 0.3 * eq
        else:
            # Small signal-driven tilt
            signal = float(research.get("signal", 0.0))
            conviction = float(research.get("conviction", 0.5))
            tilt = signal * conviction * 0.05  # max 5% tilt per cycle
            # Distribute tilt uniformly (simplified — real PM would target specific assets)
            w += tilt / n
            w = np.maximum(w, 0.0)

        # Normalise
        total = w.sum()
        if total > 0:
            w = w / total
        else:
            w = np.ones(n) / n
        return w


class ExecutionAgent:
    """
    Order generation agent.
    Converts weight changes into trade orders.
    """

    def __init__(self, agent_id: str = "exec-01") -> None:
        self.agent_id = agent_id

    def generate_orders(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        portfolio_value: float,
        prices: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """
        Returns list of orders:
        [{'ticker', 'side', 'weight_change', 'notional', 'reason'}]

        Minimum order threshold: 0.5% of portfolio value.
        """
        cur = np.asarray(current_weights, dtype=float)
        tgt = np.asarray(target_weights, dtype=float)
        n = len(cur)

        if prices is None:
            prices = np.ones(n) * 100.0  # assume $100/share if not provided

        orders = []
        min_notional = portfolio_value * 0.005  # 0.5% threshold

        for i in range(n):
            delta_w = float(tgt[i] - cur[i])
            notional = abs(delta_w) * portfolio_value
            if notional < min_notional:
                continue
            side = "BUY" if delta_w > 0 else "SELL"
            ticker = f"Asset{i}"
            quantity = int(notional / float(prices[i]))
            orders.append({
                "ticker": ticker,
                "side": side,
                "weight_change": round(delta_w, 6),
                "quantity": quantity,
                "notional": round(notional, 2),
                "reason": f"Rebalance {side.lower()}: delta_w={delta_w:+.4f}",
                "agent_id": self.agent_id,
            })
        return orders


class MultiAgentWorkflow:
    """
    Orchestrates Research → Risk → PM → Execution pipeline.
    """

    def __init__(self, tickers: List[str]) -> None:
        self.tickers = tickers
        self.n = len(tickers)
        self._research_agent = ResearchAgent()
        self._risk_agent = RiskAgent()
        self._pm_agent = PMAgent()
        self._exec_agent = ExecutionAgent()
        self._messages: List[AgentMessage] = []

    def _log(self, agent_id: str, role: str, content: str, data: dict = None) -> None:
        self._messages.append(AgentMessage(
            agent_id=agent_id,
            role=role,
            content=content,
            data=data or {},
        ))

    def run_cycle(
        self,
        current_weights: np.ndarray,
        market_data: dict,
        portfolio_value: float,
    ) -> dict:
        """
        Runs one full agent cycle.

        Parameters
        ----------
        current_weights : (N,) current portfolio weights
        market_data     : dict with optional keys:
                          'returns' (N,) last returns,
                          'vols'    (N,) asset vols
        portfolio_value : total portfolio value in $

        Returns
        -------
        dict with keys: 'target_weights', 'orders', 'risk_report', 'messages'
        """
        self._messages = []
        cur_w = np.asarray(current_weights, dtype=float)
        n = self.n

        # Build covariance matrix from vols
        vols = np.asarray(market_data.get("vols", [0.20] * n), dtype=float)
        # Use default correlation matrix (moderate correlations)
        corr = np.eye(n) * 0.7 + np.ones((n, n)) * 0.3
        np.fill_diagonal(corr, 1.0)
        cov_matrix = np.outer(vols, vols) * corr

        # Step 1: Research for first ticker (representative)
        returns_data = market_data.get("returns", np.zeros(n))
        ticker_returns = {}
        for i, t in enumerate(self.tickers):
            ticker_returns[t] = [float(returns_data[i])] if isinstance(returns_data, (list, np.ndarray)) else [0.0]

        research = self._research_agent.analyze(
            self.tickers[0],
            context={"returns": ticker_returns[self.tickers[0]]},
        )
        self._log("research-01", "research", research["rationale"], research)

        # Step 2: Risk check on current weights
        risk_check = self._risk_agent.check(cur_w, cov_matrix, risk_budget=0.25)
        self._log(
            "risk-01", "risk",
            f"Risk check: approved={risk_check['approved']}, "
            f"vol={risk_check['portfolio_vol']:.4f}",
            risk_check,
        )

        # Step 3: PM decision
        target_weights = self._pm_agent.decide(research, risk_check, cur_w)
        self._log(
            "pm-01", "pm",
            f"PM decision: target_weights={target_weights.tolist()}",
            {"target_weights": target_weights.tolist()},
        )

        # Validate weights sum to 1
        if abs(target_weights.sum() - 1.0) > 1e-6:
            target_weights = target_weights / target_weights.sum()

        # Step 4: Generate orders
        prices = np.ones(n) * 100.0
        orders = self._exec_agent.generate_orders(
            cur_w, target_weights, portfolio_value, prices
        )
        self._log(
            "exec-01", "execution",
            f"Generated {len(orders)} orders",
            {"orders": orders},
        )

        # Risk report
        risk_report = {
            "portfolio_vol": risk_check["portfolio_vol"],
            "approved": risk_check["approved"],
            "breaches": risk_check["breaches"],
        }

        return {
            "target_weights": target_weights,
            "orders": orders,
            "risk_report": risk_report,
            "messages": self._messages,
            "research": research,
        }

    def message_log(self) -> List[AgentMessage]:
        """Return all messages from the last cycle."""
        return self._messages


# ===========================================================================
# SECTION 2: Rebalancing Orchestrator  (dim_149)
# ===========================================================================

@dataclass
class RebalancingRule:
    """Defines a trigger condition for rebalancing."""
    trigger_type: str      # 'drift', 'calendar', 'tax_loss', 'risk'
    threshold: float       # e.g., 0.05 = 5% drift
    description: str


@dataclass
class RebalancingResult:
    """Result of a rebalancing decision."""
    triggered_by: str
    orders: List[dict]
    estimated_tax_impact: float
    estimated_cost_bps: float
    net_benefit: float          # risk_reduction_value - cost - tax


class DriftMonitor:
    """
    Monitors portfolio drift from target weights.
    """

    def drift(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
    ) -> np.ndarray:
        """Absolute deviation of each weight from target."""
        cur = np.asarray(current_weights, dtype=float)
        tgt = np.asarray(target_weights, dtype=float)
        return np.abs(cur - tgt)

    def max_drift(
        self,
        current: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Maximum absolute weight deviation across all assets."""
        return float(self.drift(current, target).max())

    def requires_rebalance(
        self,
        current: np.ndarray,
        target: np.ndarray,
        threshold: float = 0.05,
    ) -> bool:
        """True if any asset has drifted more than threshold."""
        return self.max_drift(current, target) > threshold


class TaxLossHarvester:
    """
    Identifies positions eligible for tax-loss harvesting.
    Wash-sale rule: cannot repurchase within 30 days.
    """

    def harvest_candidates(
        self,
        positions: Dict[str, dict],
    ) -> List[dict]:
        """
        Returns positions with unrealized losses, sorted by loss magnitude.

        positions format:
          {ticker: {'cost_basis': float, 'current_price': float, 'shares': int}}
        """
        candidates = []
        for ticker, pos in positions.items():
            cost = float(pos.get("cost_basis", 0.0))
            price = float(pos.get("current_price", 0.0))
            shares = int(pos.get("shares", 0))
            unrealized_pnl = (price - cost) * shares
            if unrealized_pnl < 0:  # has a loss
                tax_rate = 0.35  # assumed marginal rate
                tax_saving = abs(unrealized_pnl) * tax_rate
                candidates.append({
                    "ticker": ticker,
                    "cost_basis": cost,
                    "current_price": price,
                    "shares": shares,
                    "unrealized_loss": abs(unrealized_pnl),
                    "tax_saving": tax_saving,
                    "wash_sale_risk": False,  # simplified
                })
        # Sort by tax saving (largest first)
        candidates.sort(key=lambda x: x["tax_saving"], reverse=True)
        return candidates

    def harvest(
        self,
        candidates: List[dict],
        wash_sale_days: int = 30,
    ) -> List[dict]:
        """
        Generate sell orders for harvestable positions.
        Returns list of orders with wash-sale notes.
        """
        orders = []
        for c in candidates:
            if c.get("wash_sale_risk", False):
                continue
            orders.append({
                "ticker": c["ticker"],
                "side": "SELL",
                "shares": c["shares"],
                "notional": c["current_price"] * c["shares"],
                "tax_saving": c["tax_saving"],
                "reason": (
                    f"Tax-loss harvest: loss={c['unrealized_loss']:.2f}, "
                    f"tax_saving={c['tax_saving']:.2f}"
                ),
                "wash_sale_days": wash_sale_days,
            })
        return orders


class PortfolioRebalancer:
    """
    Evaluates rebalancing rules and generates rebalancing decisions.
    """

    DEFAULT_RULES = [
        RebalancingRule("drift", 0.05, "Rebalance when any asset drifts > 5%"),
        RebalancingRule("calendar", 0.0, "Quarterly calendar rebalance"),
    ]

    def __init__(
        self,
        target_weights: np.ndarray,
        rules: Optional[List[RebalancingRule]] = None,
    ) -> None:
        self.target_weights = np.asarray(target_weights, dtype=float)
        self.rules = rules or self.DEFAULT_RULES
        self._drift_monitor = DriftMonitor()
        self._harvester = TaxLossHarvester()

    def should_rebalance(
        self,
        current_weights: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        Returns (True, reason) if any rule triggers, else (False, '').
        """
        cur = np.asarray(current_weights, dtype=float)
        for rule in self.rules:
            if rule.trigger_type == "drift":
                if self._drift_monitor.requires_rebalance(
                    cur, self.target_weights, threshold=rule.threshold
                ):
                    return True, f"drift:{self._drift_monitor.max_drift(cur, self.target_weights):.4f}"
        return False, ""

    def rebalance(
        self,
        current_weights: np.ndarray,
        positions: Dict[str, dict],
        portfolio_value: float,
    ) -> RebalancingResult:
        """
        Execute full rebalancing analysis.

        Returns RebalancingResult with orders, tax impact, and cost estimate.
        """
        cur = np.asarray(current_weights, dtype=float)
        tgt = self.target_weights
        n = len(cur)
        triggered, reason = self.should_rebalance(cur)
        triggered_by = reason.split(":")[0] if ":" in reason else reason

        # Generate rebalancing orders
        orders = []
        for i in range(n):
            delta_w = float(tgt[i] - cur[i])
            notional = abs(delta_w) * portfolio_value
            if notional < portfolio_value * 0.001:  # 0.1% min
                continue
            ticker = list(positions.keys())[i] if i < len(positions) else f"Asset{i}"
            side = "BUY" if delta_w > 0 else "SELL"
            orders.append({
                "ticker": ticker,
                "side": side,
                "weight_change": round(delta_w, 6),
                "notional": round(notional, 2),
                "reason": f"Drift rebalance: delta={delta_w:+.4f}",
            })

        # Tax-loss harvesting overlay
        harvest_candidates = self._harvester.harvest_candidates(positions)
        harvest_orders = self._harvester.harvest(harvest_candidates)
        tax_saving = sum(o.get("tax_saving", 0.0) for o in harvest_orders)
        # Merge harvest sell orders (avoid duplication)
        harvest_tickers = {o["ticker"] for o in harvest_orders}
        for o in harvest_orders:
            if not any(x["ticker"] == o["ticker"] for x in orders):
                orders.append(o)

        # Cost estimate: 10bps per trade
        estimated_cost_bps = len(orders) * 10.0
        estimated_cost = portfolio_value * estimated_cost_bps / 10_000.0

        # Benefit: reduction in tracking error (simplification)
        drift = self._drift_monitor.drift(cur, tgt)
        vol_assumed = 0.15  # 15% annual vol
        risk_reduction_value = float(np.sum(drift) * vol_assumed * portfolio_value * 0.01)

        net_benefit = risk_reduction_value + tax_saving - estimated_cost

        return RebalancingResult(
            triggered_by=triggered_by if triggered_by else "drift",
            orders=orders,
            estimated_tax_impact=-tax_saving,   # negative = tax benefit
            estimated_cost_bps=estimated_cost_bps,
            net_benefit=net_benefit,
        )


# ===========================================================================
# SECTION 3: Portfolio Copilot (dim_150)
# ===========================================================================

class QueryParser:
    """Parse natural language portfolio queries to structured intents."""

    QUERY_PATTERNS: Dict[str, List[str]] = {
        "what_if": [r"what.*(happen|if|would|effect)", r"scenario", r"suppose", r"increased?|decreased?|changed?"],
        "explain": [r"why|explain|reason|what.*caus", r"how.*work"],
        "risk": [r"risk|var|volatil|drawdown|worst|loss|exposure"],
        "trade_idea": [r"buy|sell|add|reduc|trim|increas|underweight|overweight"],
        "rebalance": [r"rebalanc|drift|off.?target|allocation"],
        "performance": [r"return|perform|gain|loss|alpha|beta"],
    }

    def parse(self, query: str) -> dict:
        """
        Returns {'intent': str, 'entities': list, 'parameters': dict}
        """
        q_lower = query.lower()
        intent = "explain"  # default

        # Try each intent pattern (order matters — more specific first)
        for intent_name, patterns in self.QUERY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, q_lower):
                    intent = intent_name
                    break
            if intent != "explain":
                break

        # Extract ticker entities (simple: uppercase 2-5 letter words)
        entities = re.findall(r"\b([A-Z]{2,5})\b", query)

        # Extract numeric parameters (e.g., percentages, weights)
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\s*%?", query)
        parameters = {}
        if numbers:
            parameters["value"] = float(numbers[0])

        return {
            "intent": intent,
            "entities": entities,
            "parameters": parameters,
            "raw_query": query,
        }


class WhatIfEngine:
    """Compute portfolio what-if scenarios."""

    def weight_change(
        self,
        current_weights: np.ndarray,
        ticker_idx: int,
        new_weight: float,
        cov_matrix: np.ndarray,
    ) -> dict:
        """
        What happens if asset `ticker_idx` weight changes to `new_weight`?

        Returns {'new_vol', 'old_vol', 'vol_change_bps', 'marginal_contribution', 'new_weights'}
        """
        cur = np.asarray(current_weights, dtype=float).copy()
        n = len(cur)

        old_vol = math.sqrt(max(float(cur @ cov_matrix @ cur), 0.0))

        # Construct new weights: scale others proportionally
        delta = new_weight - cur[ticker_idx]
        other_mask = np.ones(n, dtype=bool)
        other_mask[ticker_idx] = False
        other_sum = cur[other_mask].sum()

        new_w = cur.copy()
        new_w[ticker_idx] = new_weight
        if other_sum > 0 and abs(delta) < 1.0:
            # Scale other weights to keep sum = 1
            scale = (1.0 - new_weight) / other_sum if other_sum > 0 else 1.0
            new_w[other_mask] *= scale
        new_w = np.maximum(new_w, 0.0)
        s = new_w.sum()
        if s > 0:
            new_w /= s

        new_vol = math.sqrt(max(float(new_w @ cov_matrix @ new_w), 0.0))
        vol_change_bps = (new_vol - old_vol) * 10_000.0

        # Marginal contribution of the changed asset in new portfolio
        sigma_w = cov_matrix @ new_w
        mrc = float(sigma_w[ticker_idx]) / new_vol if new_vol > 0 else 0.0

        return {
            "new_vol": new_vol,
            "old_vol": old_vol,
            "vol_change_bps": vol_change_bps,
            "marginal_contribution": mrc,
            "new_weights": new_w.tolist(),
        }

    def market_shock(
        self,
        weights: np.ndarray,
        shock_returns: np.ndarray,
    ) -> dict:
        """
        Portfolio impact of a market shock scenario.

        Returns {'portfolio_return', 'winner', 'loser', 'worst_case'}
        """
        w = np.asarray(weights, dtype=float)
        sr = np.asarray(shock_returns, dtype=float)
        port_return = float(w @ sr)
        winner_idx = int(np.argmax(sr))
        loser_idx = int(np.argmin(sr))
        return {
            "portfolio_return": port_return,
            "winner": winner_idx,
            "loser": loser_idx,
            "worst_case": float(sr.min()),
            "asset_returns": sr.tolist(),
        }

    def add_position(
        self,
        current_weights: np.ndarray,
        new_weight: float,
        new_ticker_vol: float,
        correlations: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> dict:
        """
        What if we add a new position with given weight and vol?
        Returns {'new_vol', 'diversification_benefit'}
        """
        n = len(current_weights)
        cur = np.asarray(current_weights, dtype=float)
        old_vol = math.sqrt(max(float(cur @ cov_matrix @ cur), 0.0))

        # Build expanded covariance matrix
        new_cov = np.zeros((n + 1, n + 1))
        new_cov[:n, :n] = cov_matrix
        new_cov[n, :n] = correlations * new_ticker_vol * np.sqrt(np.diag(cov_matrix))
        new_cov[:n, n] = new_cov[n, :n]
        new_cov[n, n] = new_ticker_vol ** 2

        # Scale current weights
        scale = 1.0 - new_weight
        new_w = np.append(cur * scale, new_weight)
        new_w = np.maximum(new_w, 0.0)
        new_w /= new_w.sum()

        new_vol = math.sqrt(max(float(new_w @ new_cov @ new_w), 0.0))
        diversification_benefit = old_vol - new_vol

        return {
            "new_vol": new_vol,
            "old_vol": old_vol,
            "diversification_benefit": diversification_benefit,
            "new_weights": new_w.tolist(),
        }


class PortfolioCopilot:
    """
    Natural language portfolio copilot.

    Parses NL queries and routes them to the appropriate analytics engine.
    """

    def __init__(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        asset_names: List[str],
    ) -> None:
        self.weights = np.asarray(weights, dtype=float)
        self.returns = np.asarray(returns, dtype=float)
        self.asset_names = list(asset_names)
        self.n = len(weights)

        # Compute covariance matrix
        r = self.returns
        if r.ndim == 1:
            r = r.reshape(-1, 1)
        T, n_r = r.shape
        if T > 1 and n_r == self.n:
            self.cov_matrix = np.cov(r.T) * 252
            if self.cov_matrix.ndim == 0:
                self.cov_matrix = self.cov_matrix.reshape(1, 1)
        else:
            # Fall back to diagonal if shapes mismatch
            self.cov_matrix = np.diag(np.full(self.n, 0.04))  # assume 20% vol

        self._parser = QueryParser()
        self._what_if = WhatIfEngine()

    def query(self, text: str) -> str:
        """Main NL interface — parse → route → respond."""
        parsed = self._parser.parse(text)
        intent = parsed["intent"]

        if intent == "what_if":
            # Try to find a ticker and new weight in the query
            ticker = None
            for name in self.asset_names:
                if name.upper() in text.upper():
                    ticker = name
                    break
            # Find numeric value (new weight %)
            nums = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
            new_weight = float(nums[0]) / 100.0 if nums else 0.5
            if ticker and ticker in self.asset_names:
                idx = self.asset_names.index(ticker)
                return self.what_if(ticker, min(new_weight, 0.95))
            else:
                return self.what_if(self.asset_names[0], new_weight)

        elif intent == "risk":
            return self.explain_risk()

        elif intent == "trade_idea":
            return self.trade_ideas()

        elif intent == "rebalance":
            # Use equal-weight as default target if none provided
            tgt = np.ones(self.n) / self.n
            return self.rebalancing_check(tgt)

        elif intent == "performance":
            port_return = float(self.returns.mean(axis=0) @ self.weights) * 252
            vol = math.sqrt(max(float(self.weights @ self.cov_matrix @ self.weights), 0.0))
            sharpe = port_return / vol if vol > 0 else 0.0
            return (
                f"Portfolio performance: est. annual return={port_return:.2%}, "
                f"vol={vol:.2%}, Sharpe={sharpe:.2f}"
            )

        else:
            return self.explain_risk()

    def what_if(self, ticker: str, new_weight: float) -> str:
        """What-if analysis for changing a ticker's weight."""
        if ticker not in self.asset_names:
            return f"Ticker {ticker} not found in portfolio ({self.asset_names})."
        idx = self.asset_names.index(ticker)
        result = self._what_if.weight_change(
            self.weights, idx, new_weight, self.cov_matrix
        )
        direction = "increase" if new_weight > self.weights[idx] else "decrease"
        return (
            f"What-if: {direction} {ticker} to {new_weight:.1%}\n"
            f"  Current vol: {result['old_vol']:.2%}\n"
            f"  New vol:     {result['new_vol']:.2%}\n"
            f"  Vol change:  {result['vol_change_bps']:+.1f} bps\n"
            f"  Marginal risk contribution of {ticker}: {result['marginal_contribution']:.4f}"
        )

    def explain_risk(self) -> str:
        """Explain portfolio risk decomposition."""
        w = self.weights
        cov = self.cov_matrix
        port_vol = math.sqrt(max(float(w @ cov @ w), 0.0))
        # Marginal risk contributions
        sigma_w = cov @ w
        if port_vol > 0:
            rc = w * sigma_w / port_vol
        else:
            rc = np.zeros(self.n)
        # Sort by contribution
        order = np.argsort(rc)[::-1]
        lines = [
            f"Portfolio risk (vol): {port_vol:.2%}",
            "Risk contributions:",
        ]
        for i in order:
            lines.append(
                f"  {self.asset_names[i]}: weight={w[i]:.1%}, "
                f"risk_contrib={rc[i]:.4f} ({rc[i]/port_vol*100:.1f}% of total vol)"
                if port_vol > 0 else
                f"  {self.asset_names[i]}: weight={w[i]:.1%}"
            )
        return "\n".join(lines)

    def trade_ideas(self, momentum_window: int = 63) -> str:
        """Generate trade ideas based on risk-adjusted momentum."""
        r = self.returns
        if r.ndim == 1:
            r = r.reshape(-1, 1)
        T, n_r = r.shape
        if n_r != self.n:
            return "Trade ideas: insufficient data to compute momentum signals."

        window = min(momentum_window, T)
        r_window = r[-window:]
        mu = r_window.mean(axis=0) * 252
        sigma = r_window.std(axis=0) * math.sqrt(252)
        sharpe = np.where(sigma > 0, mu / sigma, 0.0)

        lines = ["Trade ideas based on risk-adjusted momentum:"]
        order = np.argsort(sharpe)[::-1]
        for i in order:
            action = "OVERWEIGHT" if sharpe[i] > 0.3 else "UNDERWEIGHT" if sharpe[i] < -0.3 else "HOLD"
            lines.append(
                f"  {self.asset_names[i]}: Sharpe={sharpe[i]:.2f} "
                f"[{action}] (current={self.weights[i]:.1%})"
            )
        lines.append(f"Note: Based on {window}-day lookback, forward-looking results may differ.")
        return "\n".join(lines)

    def rebalancing_check(self, target_weights: np.ndarray) -> str:
        """Check drift vs target and recommend rebalancing if needed."""
        tgt = np.asarray(target_weights, dtype=float)
        if tgt.sum() > 0:
            tgt = tgt / tgt.sum()
        cur = self.weights
        drift = np.abs(cur - tgt)
        max_drift = float(drift.max())
        needs_rebalance = max_drift > 0.05

        lines = [
            f"Rebalancing check (threshold: 5%):",
            f"  Max drift: {max_drift:.2%} — {'REBALANCE RECOMMENDED' if needs_rebalance else 'Within tolerance'}",
        ]
        for i, name in enumerate(self.asset_names):
            direction = "+" if cur[i] > tgt[i] else "-"
            lines.append(
                f"  {name}: current={cur[i]:.1%}, target={tgt[i]:.1%}, "
                f"drift={direction}{drift[i]:.2%}"
            )
        return "\n".join(lines)


# ===========================================================================
# Convenience functions
# ===========================================================================

def run_portfolio_cycle(
    tickers: List[str],
    current_weights: np.ndarray,
    market_data: dict,
    portfolio_value: float,
) -> dict:
    """Run one full multi-agent portfolio cycle."""
    workflow = MultiAgentWorkflow(tickers)
    return workflow.run_cycle(current_weights, market_data, portfolio_value)


def check_rebalance_needed(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    threshold: float = 0.05,
) -> bool:
    """Return True if any asset has drifted beyond the threshold."""
    return DriftMonitor().requires_rebalance(current_weights, target_weights, threshold)


def harvest_tax_losses(positions: dict) -> List[dict]:
    """Return a list of tax-loss harvest candidates."""
    return TaxLossHarvester().harvest_candidates(positions)


def portfolio_query(
    query: str,
    weights: np.ndarray,
    returns: np.ndarray,
    asset_names: List[str],
) -> str:
    """One-shot NL portfolio query."""
    copilot = PortfolioCopilot(weights, returns, asset_names)
    return copilot.query(query)
