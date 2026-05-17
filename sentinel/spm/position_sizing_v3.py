"""
SENTINEL SPM — Position Sizing v3
dim_082: Kelly / Vol-Target / Risk-Parity sizing (score 7 → 9)

Comprehensive position sizing framework covering:
  - Kelly Criterion (discrete, continuous, multivariate, fractional)
  - Volatility targeting (EWMA, dynamic leverage, portfolio rebalancing)
  - Risk parity / ERC / HRP (Hierarchical Risk Parity)
  - Risk budgeting system
  - Sizing backtester

Free data only: yfinance for prices.
Math: numpy / scipy (scipy guarded).
"""

from __future__ import annotations

import warnings
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class KellyResult:
    win_rate: float
    avg_win: float
    avg_loss: float
    full_kelly: float
    half_kelly: float
    quarter_kelly: float
    expected_cagr_full: float
    expected_cagr_half: float
    expected_cagr_quarter: float
    ruin_probability_full: float
    ruin_probability_half: float
    ruin_probability_quarter: float
    n_trades: int
    notes: str = ""

    def summary(self) -> str:
        lines = [
            f"Kelly Analysis ({self.n_trades} trades)",
            f"  Win rate        : {self.win_rate:.1%}",
            f"  Avg win         : {self.avg_win:.2%}",
            f"  Avg loss        : {self.avg_loss:.2%}",
            f"  Full Kelly      : {self.full_kelly:.2%}  (CAGR {self.expected_cagr_full:.1%}, ruin {self.ruin_probability_full:.1%})",
            f"  Half Kelly      : {self.half_kelly:.2%}  (CAGR {self.expected_cagr_half:.1%}, ruin {self.ruin_probability_half:.1%})",
            f"  Quarter Kelly   : {self.quarter_kelly:.2%}  (CAGR {self.expected_cagr_quarter:.1%}, ruin {self.ruin_probability_quarter:.1%})",
        ]
        if self.notes:
            lines.append(f"  Notes: {self.notes}")
        return "\n".join(lines)


@dataclass
class PositionSize:
    method: str
    shares: float
    notional: float
    portfolio_pct: float
    risk_per_trade: float
    notes: str = ""

    def summary(self) -> str:
        return (
            f"[{self.method}] shares={self.shares:.1f}  notional=${self.notional:,.0f}  "
            f"pct={self.portfolio_pct:.2%}  risk/trade=${self.risk_per_trade:,.0f}"
        )


@dataclass
class RiskContribution:
    tickers: list
    weights: np.ndarray
    marginal_risk: np.ndarray
    risk_contributions: np.ndarray
    portfolio_vol: float

    def summary(self) -> str:
        lines = [f"Risk Contributions (portfolio vol {self.portfolio_vol:.2%}):"]
        for i, t in enumerate(self.tickers):
            lines.append(
                f"  {t:10s}  w={self.weights[i]:.2%}  MRC={self.marginal_risk[i]:.4f}  "
                f"RC={self.risk_contributions[i]:.2%}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Kelly Criterion
# ---------------------------------------------------------------------------

class KellyCriterion:
    """
    Full Kelly, fractional Kelly, continuous Kelly, multivariate Kelly.
    Includes Monte Carlo ruin probability and growth simulation.
    """

    # ------------------------------------------------------------------
    # Core Kelly formulas
    # ------------------------------------------------------------------

    @staticmethod
    def compute_full_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Discrete Kelly: f* = (p*b - q) / b
        where b = avg_win / avg_loss, q = 1 - p.
        Returns fraction of capital to risk per trade.
        """
        if avg_loss <= 0:
            raise ValueError("avg_loss must be > 0")
        if not (0 < win_rate < 1):
            raise ValueError("win_rate must be in (0, 1)")

        b = avg_win / avg_loss
        q = 1.0 - win_rate
        kelly = (win_rate * b - q) / b
        return float(np.clip(kelly, 0.0, 1.0))

    @staticmethod
    def compute_continuous_kelly(mu: float, sigma: float, rf: float = 0.0) -> float:
        """
        Continuous Kelly for log-normal returns.
        f* = (mu - rf) / sigma^2
        mu, sigma are annual; rf is risk-free rate.
        """
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        kelly = (mu - rf) / (sigma ** 2)
        return float(np.clip(kelly, 0.0, 2.0))  # cap at 2x

    @staticmethod
    def compute_multivariate_kelly(
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        rf: float = 0.0,
    ) -> np.ndarray:
        """
        Multi-asset Kelly: f* = Sigma^{-1} * (mu - rf).
        Equivalent to maximum Sharpe weights (scaled).
        """
        n = len(expected_returns)
        excess = expected_returns - rf

        try:
            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # Pseudoinverse fallback
            cov_inv = np.linalg.pinv(cov_matrix)

        weights = cov_inv @ excess

        # Normalize to unit sum if all positive
        if weights.sum() > 0:
            weights = weights / weights.sum()

        return np.clip(weights, 0.0, 1.0)

    @staticmethod
    def compute_fractional_kelly(full_kelly: float, fraction: float = 0.25) -> float:
        """
        Fractional Kelly: most practitioners use 1/4 to 1/2 Kelly
        to reduce volatility of wealth while preserving most of the
        long-run growth advantage.
        """
        if not (0 < fraction <= 1):
            raise ValueError("fraction must be in (0, 1]")
        return float(np.clip(full_kelly * fraction, 0.0, 1.0))

    # ------------------------------------------------------------------
    # From trade history
    # ------------------------------------------------------------------

    def compute_kelly_from_trades(self, trades: pd.DataFrame) -> KellyResult:
        """
        Compute Kelly metrics from a trade history DataFrame.

        Parameters
        ----------
        trades : pd.DataFrame
            Columns: 'win' (bool), 'return_pct' (float, e.g. 0.05 = 5%)

        Returns
        -------
        KellyResult
        """
        required = {"win", "return_pct"}
        missing = required - set(trades.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        trades = trades.dropna(subset=["win", "return_pct"]).copy()
        if len(trades) < 10:
            raise ValueError("Need at least 10 trades for meaningful Kelly estimate")

        wins = trades[trades["win"] == True]["return_pct"]
        losses = trades[trades["win"] == False]["return_pct"]

        win_rate = len(wins) / len(trades)
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 1e-6

        # Clamp degenerate cases
        avg_win = max(avg_win, 1e-6)
        avg_loss = max(avg_loss, 1e-6)
        win_rate = np.clip(win_rate, 0.01, 0.99)

        full_kelly = self.compute_full_kelly(win_rate, avg_win, avg_loss)
        half_kelly = self.compute_fractional_kelly(full_kelly, 0.50)
        quarter_kelly = self.compute_fractional_kelly(full_kelly, 0.25)

        # Expected CAGR per period = p*ln(1+f*b) + q*ln(1-f)
        def expected_log_growth(f: float) -> float:
            b = avg_win / avg_loss
            q = 1.0 - win_rate
            gain_term = win_rate * np.log(max(1.0 + f * b, 1e-9))
            loss_term = q * np.log(max(1.0 - f, 1e-9))
            return gain_term + loss_term

        cagr_full = np.exp(expected_log_growth(full_kelly)) - 1.0
        cagr_half = np.exp(expected_log_growth(half_kelly)) - 1.0
        cagr_quarter = np.exp(expected_log_growth(quarter_kelly)) - 1.0

        # Ruin probability via MC
        ruin_full = self.compute_ruin_probability(full_kelly, 0.50, 5000)
        ruin_half = self.compute_ruin_probability(half_kelly, 0.50, 5000)
        ruin_quarter = self.compute_ruin_probability(quarter_kelly, 0.50, 5000)

        notes = ""
        if full_kelly > 0.5:
            notes = "Full Kelly >50% — very aggressive; use fractional Kelly in practice."
        elif full_kelly < 0.0:
            notes = "Negative Kelly — edge is unfavorable; do not trade this system."

        return KellyResult(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            full_kelly=full_kelly,
            half_kelly=half_kelly,
            quarter_kelly=quarter_kelly,
            expected_cagr_full=float(cagr_full),
            expected_cagr_half=float(cagr_half),
            expected_cagr_quarter=float(cagr_quarter),
            ruin_probability_full=ruin_full,
            ruin_probability_half=ruin_half,
            ruin_probability_quarter=ruin_quarter,
            n_trades=len(trades),
            notes=notes,
        )

    def compute_ruin_probability(
        self,
        kelly_fraction: float,
        max_drawdown_tolerance: float = 0.50,
        n_simulations: int = 10_000,
        win_rate: float = 0.55,
        avg_win: float = 0.10,
        avg_loss: float = 0.07,
        n_periods: int = 252,
    ) -> float:
        """
        Monte Carlo: fraction of paths that hit max_drawdown_tolerance
        before doubling equity over n_periods.

        Uses the supplied win_rate / avg_win / avg_loss as the bet
        distribution (can be overridden; defaults assume a mild edge).
        """
        if kelly_fraction <= 0:
            return 1.0

        b = avg_win / avg_loss
        rng = np.random.default_rng(42)

        ruin_count = 0
        for _ in range(n_simulations):
            equity = 1.0
            peak = 1.0
            ruined = False
            for _ in range(n_periods):
                outcome = rng.random()
                if outcome < win_rate:
                    equity *= 1.0 + kelly_fraction * b * avg_loss
                else:
                    equity *= 1.0 - kelly_fraction * avg_loss
                equity = max(equity, 1e-10)
                peak = max(peak, equity)
                drawdown = (peak - equity) / peak
                if drawdown >= max_drawdown_tolerance:
                    ruined = True
                    break
            if ruined:
                ruin_count += 1

        return ruin_count / n_simulations

    def simulate_kelly_growth(
        self,
        kelly_fraction: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        n_periods: int = 252,
        n_paths: int = 1_000,
    ) -> pd.DataFrame:
        """
        Simulate distribution of terminal wealth at given kelly_fraction.
        Returns DataFrame with columns: path_id, terminal_wealth, max_drawdown.
        """
        b = avg_win / avg_loss
        rng = np.random.default_rng(0)
        records = []

        for path_id in range(n_paths):
            equity = 1.0
            peak = 1.0
            for _ in range(n_periods):
                if rng.random() < win_rate:
                    equity *= 1.0 + kelly_fraction * b * avg_loss
                else:
                    equity *= max(1.0 - kelly_fraction * avg_loss, 1e-10)
                peak = max(peak, equity)
            max_dd = (peak - equity) / max(peak, 1e-10)
            records.append({"path_id": path_id, "terminal_wealth": equity, "max_drawdown": max_dd})

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Volatility Targeting
# ---------------------------------------------------------------------------

class VolatilityTargeting:
    """
    Position sizing to hit a specific portfolio volatility level.
    Implements EWMA vol, dynamic leverage, and full portfolio rebalancing.
    """

    # ------------------------------------------------------------------
    # Core sizing
    # ------------------------------------------------------------------

    @staticmethod
    def compute_position_size(
        signal_strength: float,
        target_vol: float,
        asset_vol: float,
        portfolio_equity: float,
        price: float,
    ) -> float:
        """
        Shares = (target_vol * equity * signal_strength) / (asset_vol * price)

        Parameters
        ----------
        signal_strength : 0-1 scaling (1 = full, 0.5 = half)
        target_vol      : annual target volatility (e.g. 0.10 = 10%)
        asset_vol       : asset's annual realized volatility
        portfolio_equity: total equity in dollars
        price           : current asset price
        """
        if asset_vol <= 0 or price <= 0 or portfolio_equity <= 0:
            return 0.0
        signal_strength = float(np.clip(signal_strength, 0.0, 1.0))
        notional = (target_vol / asset_vol) * portfolio_equity * signal_strength
        shares = notional / price
        return max(0.0, shares)

    @staticmethod
    def compute_portfolio_vol_target(
        holdings: dict[str, float],
        returns_history: dict[str, pd.Series],
        target_annual_vol: float = 0.10,
    ) -> dict[str, float]:
        """
        Scale all position weights so that portfolio vol ~= target_annual_vol.
        Returns new position weights (fractions of equity).
        """
        tickers = [t for t in holdings if t in returns_history]
        if not tickers:
            return {}

        # Align returns
        df = pd.concat([returns_history[t].rename(t) for t in tickers], axis=1).dropna()
        if len(df) < 21:
            return {t: holdings[t] for t in tickers}

        # Current weights
        total_val = sum(abs(v) for v in holdings.values())
        w = np.array([holdings[t] / max(total_val, 1e-10) for t in tickers])

        # Portfolio covariance (annualized)
        cov = df.cov().values * 252
        port_var = float(w @ cov @ w)
        port_vol = np.sqrt(max(port_var, 1e-12))

        if port_vol < 1e-9:
            return {t: holdings[t] for t in tickers}

        scale = target_annual_vol / port_vol
        new_weights = w * scale

        return {t: float(new_weights[i]) for i, t in enumerate(tickers)}

    @staticmethod
    def compute_dynamic_leverage(
        portfolio_vol: float,
        target_vol: float,
        max_leverage: float = 2.0,
    ) -> float:
        """
        Bridgewater-style dynamic leverage:
          Leverage = min(target_vol / portfolio_vol, max_leverage)
        When actual vol < target: increase exposure (up to max).
        When actual vol > target: reduce exposure.
        """
        if portfolio_vol <= 0:
            return 1.0
        leverage = target_vol / portfolio_vol
        return float(np.clip(leverage, 0.0, max_leverage))

    @staticmethod
    def compute_ewma_vol(
        returns: pd.Series,
        lambda_: float = 0.94,
        annualize: bool = True,
    ) -> float:
        """
        RiskMetrics EWMA volatility:
          sigma^2_t = lambda * sigma^2_{t-1} + (1-lambda) * r^2_t

        Parameters
        ----------
        lambda_ : decay factor (0.94 = RiskMetrics daily default)
        annualize: multiply by sqrt(252) if True
        """
        r = returns.dropna().values.astype(float)
        if len(r) < 2:
            return float(returns.std() * (np.sqrt(252) if annualize else 1.0))

        var = float(np.var(r[:20]) if len(r) >= 20 else np.var(r))
        for ret in r:
            var = lambda_ * var + (1.0 - lambda_) * ret ** 2

        vol = np.sqrt(max(var, 0.0))
        if annualize:
            vol *= np.sqrt(252)
        return float(vol)

    @staticmethod
    def compute_realized_vol(
        returns: pd.Series,
        window: int = 21,
    ) -> pd.Series:
        """
        Rolling realized volatility, annualized.
        """
        return returns.rolling(window).std() * np.sqrt(252)

    @staticmethod
    def compute_vol_regime(current_vol: float, long_run_vol: float) -> str:
        """
        Classify current vol relative to long-run average.
        LOW  : < 0.8 × long_run_vol
        NORMAL: 0.8–1.2 × long_run_vol
        HIGH : > 1.2 × long_run_vol
        """
        if long_run_vol <= 0:
            return "UNKNOWN"
        ratio = current_vol / long_run_vol
        if ratio < 0.8:
            return "LOW"
        elif ratio <= 1.2:
            return "NORMAL"
        else:
            return "HIGH"

    @staticmethod
    def compute_vol_of_vol(vols: pd.Series) -> float:
        """
        Volatility of volatility — measures vol regime uncertainty.
        """
        v = vols.dropna()
        if len(v) < 2:
            return 0.0
        return float(v.std() / max(v.mean(), 1e-9))

    def rebalance_to_vol_target(
        self,
        portfolio: dict[str, float],
        returns_history: dict[str, pd.Series],
        target_vol: float = 0.10,
    ) -> dict[str, float]:
        """
        Full rebalancing: compute new weights for entire portfolio
        so that portfolio vol ≈ target_vol.
        Returns dict: ticker -> new weight (fraction of equity).
        """
        new_weights = self.compute_portfolio_vol_target(
            portfolio, returns_history, target_annual_vol=target_vol
        )
        # Normalize so weights sum to 1 (or less for safety margin)
        total_w = sum(abs(v) for v in new_weights.values())
        if total_w > 1.0:
            new_weights = {t: v / total_w for t, v in new_weights.items()}
        return new_weights


# ---------------------------------------------------------------------------
# Risk Parity Engine
# ---------------------------------------------------------------------------

class RiskParityEngine:
    """
    Equal Risk Contribution (ERC) and Hierarchical Risk Parity (HRP).
    """

    # ------------------------------------------------------------------
    # Core ERC
    # ------------------------------------------------------------------

    def compute_risk_parity_weights(
        self,
        cov_matrix: np.ndarray,
        risk_budget: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Equal Risk Contribution (ERC) weights.
        Minimizes sum of squared pairwise risk contribution differences.

        Parameters
        ----------
        cov_matrix  : (n, n) covariance matrix
        risk_budget : (n,) target risk contributions; None = equal (ERC)

        Solves via scipy.minimize (L-BFGS-B) or iterative SPBA fallback.
        """
        n = cov_matrix.shape[0]
        if risk_budget is None:
            risk_budget = np.ones(n) / n
        else:
            risk_budget = np.asarray(risk_budget, dtype=float)
            risk_budget = risk_budget / risk_budget.sum()

        w0 = np.ones(n) / n  # equal weight starting point

        if _HAS_SCIPY:
            def objective(w: np.ndarray) -> float:
                w = np.maximum(w, 1e-9)
                port_var = float(w @ cov_matrix @ w)
                port_vol = np.sqrt(max(port_var, 1e-12))
                mrc = cov_matrix @ w / port_vol
                rc = w * mrc
                rc_pct = rc / (rc.sum() + 1e-12)
                diff = rc_pct - risk_budget
                return float(diff @ diff)

            constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
            bounds = [(0.0, 1.0)] * n
            result = minimize(
                objective,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if result.success:
                w_opt = np.maximum(result.x, 0.0)
                return w_opt / w_opt.sum()

        # Fallback: Simple Iterative Algorithm (Maillard et al. 2010)
        return self._spba_erc(cov_matrix, risk_budget)

    @staticmethod
    def _spba_erc(
        cov_matrix: np.ndarray,
        risk_budget: np.ndarray,
        max_iter: int = 500,
        tol: float = 1e-8,
    ) -> np.ndarray:
        """
        Simple Parabolic Bisection Algorithm (Maillard et al. 2010).
        Pure numpy iterative solver for ERC.
        """
        n = cov_matrix.shape[0]
        w = np.ones(n) / n

        for _ in range(max_iter):
            w_prev = w.copy()
            port_var = float(w @ cov_matrix @ w)
            port_vol = np.sqrt(max(port_var, 1e-12))
            mrc = cov_matrix @ w / port_vol
            rc = w * mrc
            rc_total = rc.sum()

            for i in range(n):
                # Target: RC_i = budget_i * total_rc
                target_rc = risk_budget[i] * rc_total
                # Gradient: d(RC_i)/d(w_i) = mrc_i + w_i * sigma_ii / port_vol
                grad = mrc[i] + w[i] * cov_matrix[i, i] / max(port_vol, 1e-12)
                if abs(grad) > 1e-12:
                    w[i] = max(w[i] * target_rc / (rc[i] + 1e-12), 1e-9)

            w = np.maximum(w, 1e-9)
            w /= w.sum()

            if np.max(np.abs(w - w_prev)) < tol:
                break

        return w

    def compute_risk_contributions(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Fractional risk contributions RC_i = w_i * (Sigma*w)_i / sqrt(w'*Sigma*w).
        Returns array of RC_i (fraction, sums to 1.0).
        """
        w = np.asarray(weights, dtype=float)
        port_var = float(w @ cov_matrix @ w)
        port_vol = np.sqrt(max(port_var, 1e-12))
        mrc = cov_matrix @ w / port_vol
        rc = w * mrc
        return rc / (rc.sum() + 1e-12)

    @staticmethod
    def compute_naive_risk_parity(vols: np.ndarray) -> np.ndarray:
        """
        Inverse-vol weighting: weight_i = (1/vol_i) / sum(1/vol_j).
        Fast approximation ignoring correlations.
        """
        vols = np.maximum(np.asarray(vols, dtype=float), 1e-9)
        inv_vols = 1.0 / vols
        return inv_vols / inv_vols.sum()

    @staticmethod
    def compute_leveraged_risk_parity(
        weights: np.ndarray,
        target_vol: float,
        portfolio_vol: float,
    ) -> np.ndarray:
        """
        Bridgewater All-Weather style: scale weights by (target_vol / portfolio_vol).
        Weights may exceed 1.0 (leveraged).
        """
        if portfolio_vol <= 0:
            return weights.copy()
        leverage = target_vol / portfolio_vol
        return np.asarray(weights, dtype=float) * leverage

    def compute_hierarchical_risk_parity(
        self,
        cov_matrix: np.ndarray,
        tickers: list[str],
    ) -> np.ndarray:
        """
        Lopez de Prado HRP (2016):
          1. Hierarchical clustering on correlation matrix
          2. Quasi-diagonalization (reorder)
          3. Recursive bisection to allocate weights

        No matrix inversion — robust to estimation error.
        Pure numpy implementation.
        """
        n = len(tickers)
        corr = self._cov_to_corr(cov_matrix)

        # Step 1: Distance matrix and linkage
        dist = np.sqrt((1.0 - corr) / 2.0)
        np.fill_diagonal(dist, 0.0)
        link = self._single_linkage(dist)

        # Step 2: Quasi-diagonalization (sort leaf order)
        sorted_idx = self._get_sorted_leaves(link, n)

        # Step 3: Recursive bisection
        weights = np.ones(n)
        cluster_items = [sorted_idx]

        while cluster_items:
            cluster_items = [
                item[j:k]
                for item in cluster_items
                for j, k in ((0, len(item) // 2), (len(item) // 2, len(item)))
                if len(item) > 1
            ]
            for subcluster in cluster_items:
                if len(subcluster) <= 1:
                    continue
                left = subcluster[: len(subcluster) // 2]
                right = subcluster[len(subcluster) // 2 :]

                left_var = self._cluster_var(weights, cov_matrix, left)
                right_var = self._cluster_var(weights, cov_matrix, right)

                alpha = 1.0 - left_var / (left_var + right_var + 1e-12)
                weights[left] *= alpha
                weights[right] *= 1.0 - alpha

        weights = np.maximum(weights, 0.0)
        weights /= weights.sum()
        return weights

    # ------------------------------------------------------------------
    # HRP helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.diag(cov))
        std = np.maximum(std, 1e-12)
        corr = cov / np.outer(std, std)
        return np.clip(corr, -1.0, 1.0)

    @staticmethod
    def _single_linkage(dist: np.ndarray) -> np.ndarray:
        """Agglomerative single-linkage clustering. Returns (n-1, 4) linkage matrix."""
        n = dist.shape[0]
        link = []
        active = list(range(n))
        cluster_id = list(range(n))
        members = {i: [i] for i in range(n)}
        next_id = n

        for _ in range(n - 1):
            best_dist = np.inf
            best_pair = (0, 1)

            for ii in range(len(active)):
                for jj in range(ii + 1, len(active)):
                    ci, cj = active[ii], active[jj]
                    # Single linkage: min distance between members
                    d = np.min(dist[np.ix_(members[ci], members[cj])])
                    if d < best_dist:
                        best_dist = d
                        best_pair = (ii, jj)

            ii, jj = best_pair
            ci, cj = active[ii], active[jj]
            new_size = len(members[ci]) + len(members[cj])
            link.append([float(ci), float(cj), best_dist, float(new_size)])
            members[next_id] = members[ci] + members[cj]
            active.remove(ci)
            active.remove(cj)
            active.append(next_id)
            next_id += 1

        return np.array(link)

    @staticmethod
    def _get_sorted_leaves(link: np.ndarray, n: int) -> list[int]:
        """Compute sorted leaf indices from linkage matrix (quasi-diagonalization)."""
        members = {i: [i] for i in range(n)}
        for k, row in enumerate(link):
            ci, cj = int(row[0]), int(row[1])
            new_id = n + k
            members[new_id] = members[ci] + members[cj]

        root_id = n + len(link) - 1
        return members[root_id]

    @staticmethod
    def _cluster_var(
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        idx: list[int],
    ) -> float:
        """Variance of a subset of assets, weighted by current weights."""
        w_sub = weights[idx]
        cov_sub = cov_matrix[np.ix_(idx, idx)]
        w_sub_norm = w_sub / max(w_sub.sum(), 1e-12)
        return float(w_sub_norm @ cov_sub @ w_sub_norm)


# ---------------------------------------------------------------------------
# Risk Budgeting System
# ---------------------------------------------------------------------------

class RiskBudgetingSystem:
    """
    Allocate risk budget across strategies/asset classes.
    """

    def __init__(self, total_vol_target: float = 0.10):
        self.total_vol_target = total_vol_target
        self._budgets: dict[str, float] = {}

    def set_risk_budget(self, total_var_target: float = 0.10) -> None:
        """Set annual vol target for total portfolio."""
        self.total_vol_target = total_var_target

    def allocate_risk_budget(
        self,
        strategies: list[str],
        risk_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Distribute risk budget proportionally to risk_weights.
        Risk_budget_i = total_vol * risk_weight_i / sum(risk_weights).
        """
        total_w = sum(risk_weights.get(s, 1.0) for s in strategies)
        budgets = {}
        for s in strategies:
            w = risk_weights.get(s, 1.0)
            budgets[s] = self.total_vol_target * w / max(total_w, 1e-12)
        self._budgets = budgets
        return budgets

    @staticmethod
    def compute_marginal_risk_contribution(
        holdings: dict[str, float],
        cov_matrix: np.ndarray,
    ) -> dict[str, float]:
        """
        MRC_i = (Sigma * w)_i / sqrt(w' Sigma w)
        How much adding one unit of asset i changes portfolio vol.
        """
        tickers = list(holdings.keys())
        w = np.array(list(holdings.values()), dtype=float)
        if w.sum() > 0:
            w = w / w.sum()

        port_var = float(w @ cov_matrix @ w)
        port_vol = np.sqrt(max(port_var, 1e-12))
        mrc = (cov_matrix @ w) / port_vol

        return {t: float(mrc[i]) for i, t in enumerate(tickers)}

    def optimize_risk_budget_allocation(
        self,
        strategies: list[str],
        target_contributions: dict[str, float],
        cov_matrix: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """
        Find weights such that each strategy hits its target risk contribution.
        Uses RiskParityEngine with custom risk budget.
        """
        n = len(strategies)
        target_arr = np.array([target_contributions.get(s, 1.0 / n) for s in strategies])
        target_arr /= target_arr.sum()

        if cov_matrix is None:
            # Return proportional weights as approximation
            return {s: float(target_arr[i]) for i, s in enumerate(strategies)}

        engine = RiskParityEngine()
        weights = engine.compute_risk_parity_weights(cov_matrix, risk_budget=target_arr)
        return {s: float(weights[i]) for i, s in enumerate(strategies)}


# ---------------------------------------------------------------------------
# Position Sizing Orchestrator
# ---------------------------------------------------------------------------

class PositionSizingOrchestrator:
    """
    Unified interface for all position sizing methods.
    """

    def __init__(self):
        self._kelly = KellyCriterion()
        self._vol = VolatilityTargeting()
        self._rp = RiskParityEngine()

    def size_position(self, method: str, **kwargs) -> PositionSize:
        """
        Route to correct sizing method.

        Parameters
        ----------
        method : one of "kelly", "vol_target", "risk_parity",
                 "fixed_fraction", "max_drawdown_limit", "cvar_budget"
        **kwargs: method-specific parameters
        """
        dispatch = {
            "kelly": self._size_kelly,
            "vol_target": self._size_vol_target,
            "risk_parity": self._size_risk_parity,
            "fixed_fraction": self._size_fixed_fraction,
            "max_drawdown_limit": self._size_max_dd,
            "cvar_budget": self._size_cvar,
        }
        if method not in dispatch:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(dispatch)}")
        return dispatch[method](**kwargs)

    # ------------------------------------------------------------------
    # Internal sizing methods
    # ------------------------------------------------------------------

    def _size_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        portfolio_equity: float,
        price: float,
        fraction: float = 0.25,
        **_,
    ) -> PositionSize:
        full_k = self._kelly.compute_full_kelly(win_rate, avg_win, avg_loss)
        frac_k = self._kelly.compute_fractional_kelly(full_k, fraction)
        notional = portfolio_equity * frac_k
        shares = notional / max(price, 1e-9)
        return PositionSize(
            method=f"kelly_{fraction}",
            shares=shares,
            notional=notional,
            portfolio_pct=frac_k,
            risk_per_trade=notional * avg_loss,
            notes=f"full_kelly={full_k:.2%}  fractional({fraction})={frac_k:.2%}",
        )

    def _size_vol_target(
        self,
        signal_strength: float,
        target_vol: float,
        asset_vol: float,
        portfolio_equity: float,
        price: float,
        **_,
    ) -> PositionSize:
        shares = self._vol.compute_position_size(
            signal_strength, target_vol, asset_vol, portfolio_equity, price
        )
        notional = shares * price
        pct = notional / max(portfolio_equity, 1e-9)
        return PositionSize(
            method="vol_target",
            shares=shares,
            notional=notional,
            portfolio_pct=pct,
            risk_per_trade=notional * asset_vol / np.sqrt(252),
            notes=f"target_vol={target_vol:.1%}  asset_vol={asset_vol:.1%}",
        )

    def _size_risk_parity(
        self,
        cov_matrix: np.ndarray,
        tickers: list[str],
        portfolio_equity: float,
        prices: dict[str, float],
        **_,
    ) -> PositionSize:
        weights = self._rp.compute_risk_parity_weights(cov_matrix)
        results = {}
        for i, t in enumerate(tickers):
            price = prices.get(t, 1.0)
            notional = portfolio_equity * weights[i]
            results[t] = {"weight": weights[i], "notional": notional, "shares": notional / price}
        # Return aggregate PositionSize for the first ticker as representative
        t0 = tickers[0]
        return PositionSize(
            method="risk_parity",
            shares=results[t0]["shares"],
            notional=results[t0]["notional"],
            portfolio_pct=float(weights[0]),
            risk_per_trade=0.0,
            notes="ERC weights: " + ", ".join(f"{t}={weights[i]:.2%}" for i, t in enumerate(tickers)),
        )

    def _size_fixed_fraction(
        self,
        portfolio_equity: float,
        entry_price: float,
        stop_loss_price: float,
        risk_per_trade: float = 0.01,
        **_,
    ) -> PositionSize:
        risk_dollars = portfolio_equity * risk_per_trade
        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance <= 0:
            shares = 0.0
        else:
            shares = risk_dollars / stop_distance
        notional = shares * entry_price
        pct = notional / max(portfolio_equity, 1e-9)
        return PositionSize(
            method="fixed_fraction",
            shares=shares,
            notional=notional,
            portfolio_pct=pct,
            risk_per_trade=risk_dollars,
            notes=f"risk_per_trade={risk_per_trade:.1%}  stop_dist=${stop_distance:.2f}",
        )

    def _size_max_dd(
        self,
        portfolio_equity: float,
        entry_price: float,
        stop_loss_price: float,
        max_portfolio_dd_tolerance: float = 0.10,
        **_,
    ) -> PositionSize:
        """
        Size so that hitting the stop does not exceed max_portfolio_dd_tolerance.
        """
        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance <= 0:
            shares = 0.0
        else:
            max_loss_dollars = portfolio_equity * max_portfolio_dd_tolerance
            shares = max_loss_dollars / stop_distance
        notional = shares * entry_price
        pct = notional / max(portfolio_equity, 1e-9)
        return PositionSize(
            method="max_drawdown_limit",
            shares=shares,
            notional=notional,
            portfolio_pct=pct,
            risk_per_trade=shares * stop_distance,
            notes=f"max_dd_tol={max_portfolio_dd_tolerance:.1%}",
        )

    def _size_cvar(
        self,
        signal: float,
        cvar_target: float,
        cvar_per_unit: float,
        portfolio_equity: float,
        price: float = 1.0,
        **_,
    ) -> PositionSize:
        return self.compute_cvar_budget_size(signal, cvar_target, cvar_per_unit, portfolio_equity, price)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def compute_cvar_budget_size(
        self,
        signal: float,
        cvar_target: float,
        cvar_per_unit: float,
        portfolio_equity: float,
        price: float = 1.0,
    ) -> PositionSize:
        """
        Size so that CVaR contribution ≤ cvar_target (fraction of equity).
        units = (cvar_target * equity * |signal|) / cvar_per_unit
        """
        if cvar_per_unit <= 0:
            return PositionSize("cvar_budget", 0, 0, 0, 0, "cvar_per_unit=0")
        max_loss = portfolio_equity * cvar_target * abs(signal)
        notional = max_loss / max(cvar_per_unit, 1e-9)
        shares = notional / max(price, 1e-9)
        return PositionSize(
            method="cvar_budget",
            shares=shares,
            notional=notional,
            portfolio_pct=notional / max(portfolio_equity, 1e-9),
            risk_per_trade=max_loss,
            notes=f"cvar_target={cvar_target:.1%}  cvar_per_unit={cvar_per_unit:.4f}",
        )

    @staticmethod
    def compute_fixed_fraction_size(
        portfolio_equity: float,
        entry_price: float,
        stop_loss_price: float,
        risk_per_trade: float = 0.01,
    ) -> float:
        """
        Classic: risk risk_per_trade% of equity per trade.
        Returns number of shares.
        """
        risk_dollars = portfolio_equity * risk_per_trade
        stop_dist = abs(entry_price - stop_loss_price)
        if stop_dist <= 0:
            return 0.0
        return risk_dollars / stop_dist

    def compute_optimal_f(self, returns: pd.Series) -> float:
        """
        Ralph Vince's Optimal f — maximize geometric growth rate.
        TWR(f) = product(1 + f * r_t / max_loss) over all trades.
        Maximize via grid search.
        """
        r = returns.dropna().values.astype(float)
        if len(r) < 5:
            return 0.0

        max_loss = abs(r.min())
        if max_loss < 1e-9:
            return 0.0

        best_f, best_twr = 0.01, -np.inf
        for f_candidate in np.linspace(0.01, 0.99, 100):
            terms = 1.0 + f_candidate * r / max_loss
            if np.any(terms <= 0):
                break
            twr = np.prod(terms)
            if twr > best_twr:
                best_twr = twr
                best_f = f_candidate

        return float(best_f)

    def blend_methods(
        self,
        methods: list[str],
        weights: list[float],
        **kwargs,
    ) -> PositionSize:
        """
        Ensemble: weighted average of multiple sizing methods.
        Returns blended PositionSize.
        """
        if len(methods) != len(weights):
            raise ValueError("methods and weights must have same length")

        weights_arr = np.array(weights, dtype=float)
        weights_arr /= weights_arr.sum()

        results = [self.size_position(m, **kwargs) for m in methods]

        blended_shares = sum(w * r.shares for w, r in zip(weights_arr, results))
        blended_notional = sum(w * r.notional for w, r in zip(weights_arr, results))
        blended_pct = sum(w * r.portfolio_pct for w, r in zip(weights_arr, results))
        blended_risk = sum(w * r.risk_per_trade for w, r in zip(weights_arr, results))

        method_str = "+".join(f"{m}({w:.0%})" for m, w in zip(methods, weights_arr))
        return PositionSize(
            method=f"blend[{method_str}]",
            shares=blended_shares,
            notional=blended_notional,
            portfolio_pct=blended_pct,
            risk_per_trade=blended_risk,
        )


# ---------------------------------------------------------------------------
# Sizing Backtester
# ---------------------------------------------------------------------------

class SizingBacktester:
    """
    Historically test sizing methods using real return data from yfinance.
    """

    def __init__(self):
        self._vol_engine = VolatilityTargeting()
        self._rp_engine = RiskParityEngine()

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_returns(
        tickers: list[str],
        start: str = "2010-01-01",
        end: str = "2024-12-31",
    ) -> dict[str, pd.Series]:
        if not _HAS_YF:
            logger.warning("yfinance not available — generating synthetic returns")
            rng = np.random.default_rng(42)
            dates = pd.bdate_range(start, end)
            return {
                t: pd.Series(rng.normal(0.0005, 0.01, len(dates)), index=dates)
                for t in tickers
            }

        try:
            raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw[["Close"]] if "Close" in raw.columns else raw
            returns = prices.pct_change().dropna()
            return {t: returns[t].dropna() for t in tickers if t in returns.columns}
        except Exception as exc:
            logger.warning(f"yfinance fetch failed: {exc} — using synthetic data")
            rng = np.random.default_rng(42)
            dates = pd.bdate_range(start, end)
            return {
                t: pd.Series(rng.normal(0.0005, 0.01, len(dates)), index=dates)
                for t in tickers
            }

    @staticmethod
    def _compute_stats(portfolio_returns: pd.Series) -> dict:
        r = portfolio_returns.dropna()
        if len(r) < 2:
            return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDrawdown": 0.0, "Volatility": 0.0}

        ann_ret = float((1 + r).prod() ** (252 / len(r)) - 1)
        ann_vol = float(r.std() * np.sqrt(252))
        sharpe = ann_ret / max(ann_vol, 1e-9)

        cum = (1 + r).cumprod()
        roll_max = cum.cummax()
        dd = (cum - roll_max) / roll_max
        max_dd = float(dd.min())

        return {
            "CAGR": ann_ret,
            "Sharpe": sharpe,
            "MaxDrawdown": max_dd,
            "Volatility": ann_vol,
        }

    # ------------------------------------------------------------------
    # Backtests
    # ------------------------------------------------------------------

    def backtest_vol_target(
        self,
        returns: dict[str, pd.Series],
        target_vol: float = 0.10,
        rebalance_freq: str = "M",
    ) -> pd.Series:
        """
        Simulate a volatility-targeted portfolio over history.
        Monthly rebalancing: scale weights so portfolio vol ≈ target_vol.
        """
        tickers = list(returns.keys())
        df = pd.concat([returns[t].rename(t) for t in tickers], axis=1).dropna()

        if len(df) < 42:
            return pd.Series(dtype=float)

        # Equal-weight baseline for first period
        weights = {t: 1.0 / len(tickers) for t in tickers}
        port_returns = []
        dates = []

        rebal_dates = df.resample(rebalance_freq).last().index

        for i, date in enumerate(df.index):
            if date in rebal_dates and i >= 21:
                hist = df.iloc[max(0, i - 63) : i]
                cov_ann = hist.cov() * 252
                try:
                    w = {t: weights.get(t, 1.0 / len(tickers)) for t in tickers}
                    new_w = self._vol_engine.compute_portfolio_vol_target(
                        w, {t: hist[t] for t in tickers}, target_annual_vol=target_vol
                    )
                    total_w = sum(new_w.values())
                    if 0.01 < total_w < 5.0:
                        weights = new_w
                except Exception:
                    pass

            row = df.loc[date]
            r = sum(weights.get(t, 0.0) * row[t] for t in tickers)
            port_returns.append(r)
            dates.append(date)

        return pd.Series(port_returns, index=dates, name="vol_target")

    def backtest_risk_parity(
        self,
        returns: dict[str, pd.Series],
        lookback: int = 63,
    ) -> pd.Series:
        """
        Monthly-rebalanced ERC portfolio.
        """
        tickers = list(returns.keys())
        df = pd.concat([returns[t].rename(t) for t in tickers], axis=1).dropna()

        if len(df) < lookback + 5:
            return pd.Series(dtype=float)

        rebal_dates = df.resample("M").last().index
        weights = np.ones(len(tickers)) / len(tickers)
        port_returns = []
        dates = []

        for i, date in enumerate(df.index):
            if date in rebal_dates and i >= lookback:
                hist = df.iloc[i - lookback : i]
                cov = hist.cov().values * 252
                try:
                    weights = self._rp_engine.compute_risk_parity_weights(cov)
                except Exception:
                    pass

            row = df.loc[date].values
            r = float(weights @ row)
            port_returns.append(r)
            dates.append(date)

        return pd.Series(port_returns, index=dates, name="risk_parity")

    def compare_methods(self, returns: dict[str, pd.Series]) -> pd.DataFrame:
        """
        Compare sizing methods: equal weight, vol-target, ERC, HRP, Kelly.
        Returns DataFrame with Sharpe, MaxDrawdown, CAGR, Volatility per method.
        """
        tickers = list(returns.keys())
        df = pd.concat([returns[t].rename(t) for t in tickers], axis=1).dropna()

        results = {}

        # 1. Equal weight
        ew_ret = df.mean(axis=1)
        results["equal_weight"] = self._compute_stats(ew_ret)

        # 2. Vol target
        vt_ret = self.backtest_vol_target(returns, target_vol=0.10)
        if len(vt_ret) > 10:
            results["vol_target_10pct"] = self._compute_stats(vt_ret)

        # 3. ERC risk parity
        rp_ret = self.backtest_risk_parity(returns)
        if len(rp_ret) > 10:
            results["erc_risk_parity"] = self._compute_stats(rp_ret)

        # 4. HRP (one-shot on full history cov)
        cov_full = df.cov().values * 252
        try:
            hrp_w = self._rp_engine.compute_hierarchical_risk_parity(cov_full, tickers)
            hrp_ret = df @ hrp_w
            results["hrp"] = self._compute_stats(hrp_ret)
        except Exception:
            pass

        # 5. Naive inverse-vol
        vols = df.std().values * np.sqrt(252)
        inv_vol_w = RiskParityEngine.compute_naive_risk_parity(vols)
        iv_ret = df @ inv_vol_w
        results["inverse_vol"] = self._compute_stats(iv_ret)

        out = pd.DataFrame(results).T
        out.index.name = "method"
        return out.round(4)


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    print("=" * 70)
    print("SENTINEL SPM — Position Sizing v3 — dim_082")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Kelly from sample trade history
    # ---------------------------------------------------------------
    print("\n[1] Kelly Criterion from synthetic trade history")
    rng = np.random.default_rng(99)
    n_trades = 200
    win_mask = rng.random(n_trades) < 0.55
    trades_df = pd.DataFrame({
        "win": win_mask,
        "return_pct": np.where(win_mask, rng.uniform(0.03, 0.12, n_trades), -rng.uniform(0.02, 0.08, n_trades)),
    })

    kelly = KellyCriterion()
    kr = kelly.compute_kelly_from_trades(trades_df)
    print(kr.summary())

    # Simulate Kelly paths
    sim = kelly.simulate_kelly_growth(
        kelly_fraction=kr.quarter_kelly,
        win_rate=kr.win_rate,
        avg_win=kr.avg_win,
        avg_loss=kr.avg_loss,
        n_periods=252,
        n_paths=500,
    )
    print(f"\n  Quarter-Kelly 252-day terminal wealth (500 paths):")
    print(f"    Median: {sim['terminal_wealth'].median():.2f}x")
    print(f"    5th pct: {sim['terminal_wealth'].quantile(0.05):.2f}x")
    print(f"    95th pct: {sim['terminal_wealth'].quantile(0.95):.2f}x")
    print(f"    Avg max drawdown: {sim['max_drawdown'].mean():.2%}")

    # ---------------------------------------------------------------
    # 2. Volatility targeting: SPY / QQQ / GLD / TLT / AGG
    # ---------------------------------------------------------------
    print("\n[2] Volatility Targeting — SPY/QQQ/GLD/TLT/AGG")
    backtester = SizingBacktester()
    tickers_5 = ["SPY", "QQQ", "GLD", "TLT", "AGG"]
    ret_dict = backtester._fetch_returns(tickers_5, start="2015-01-01", end="2024-12-31")
    print(f"  Fetched {len(ret_dict)} tickers, {min(len(v) for v in ret_dict.values())} bars each")

    vol_engine = VolatilityTargeting()
    sample_returns = ret_dict.get("SPY", pd.Series(rng.normal(0.0004, 0.01, 500)))
    ewma_v = vol_engine.compute_ewma_vol(sample_returns)
    realized_v = vol_engine.compute_realized_vol(sample_returns)
    long_run = float(realized_v.mean())
    regime = vol_engine.compute_vol_regime(ewma_v, long_run)
    vov = vol_engine.compute_vol_of_vol(realized_v.dropna())

    print(f"  SPY EWMA vol (annual): {ewma_v:.2%}")
    print(f"  SPY long-run avg vol : {long_run:.2%}")
    print(f"  Vol regime           : {regime}")
    print(f"  Vol-of-vol           : {vov:.4f}")

    spy_price = 480.0
    pos = vol_engine.compute_position_size(1.0, 0.10, ewma_v, 1_000_000, spy_price)
    print(f"\n  Vol-target position (10% target, $1M equity):")
    print(f"    SPY @ ${spy_price}: {pos:.0f} shares  (${pos*spy_price:,.0f} notional)")

    lev = vol_engine.compute_dynamic_leverage(ewma_v, 0.10, max_leverage=2.0)
    print(f"  Dynamic leverage: {lev:.2f}x")

    # ---------------------------------------------------------------
    # 3. Risk Parity weights for 5-asset portfolio
    # ---------------------------------------------------------------
    print("\n[3] Risk Parity Weights — 5-asset portfolio")
    rp_engine = RiskParityEngine()

    df_all = pd.concat([ret_dict[t].rename(t) for t in tickers_5 if t in ret_dict], axis=1).dropna()
    cov_5 = df_all.cov().values * 252
    vols_5 = np.sqrt(np.diag(cov_5))

    erc_w = rp_engine.compute_risk_parity_weights(cov_5)
    naive_w = rp_engine.compute_naive_risk_parity(vols_5)
    hrp_w = rp_engine.compute_hierarchical_risk_parity(cov_5, tickers_5)
    rc = rp_engine.compute_risk_contributions(erc_w, cov_5)

    print(f"  {'Ticker':8s} {'ERC':>8s} {'Naive':>8s} {'HRP':>8s} {'RiskContrib':>12s}")
    for i, t in enumerate(tickers_5):
        print(f"  {t:8s} {erc_w[i]:8.2%} {naive_w[i]:8.2%} {hrp_w[i]:8.2%} {rc[i]:12.2%}")

    # ---------------------------------------------------------------
    # 4. Compare sizing methods historically
    # ---------------------------------------------------------------
    print("\n[4] Compare Sizing Methods (historical backtest 2015-2024)")
    comparison = backtester.compare_methods(ret_dict)
    print(comparison.to_string())

    # ---------------------------------------------------------------
    # 5. Orchestrator demo
    # ---------------------------------------------------------------
    print("\n[5] Position Sizing Orchestrator")
    orch = PositionSizingOrchestrator()

    ps_kelly = orch.size_position(
        "kelly",
        win_rate=kr.win_rate,
        avg_win=kr.avg_win,
        avg_loss=kr.avg_loss,
        portfolio_equity=1_000_000,
        price=480.0,
        fraction=0.25,
    )
    print(f"  {ps_kelly.summary()}")

    ps_vol = orch.size_position(
        "vol_target",
        signal_strength=0.8,
        target_vol=0.10,
        asset_vol=ewma_v,
        portfolio_equity=1_000_000,
        price=480.0,
    )
    print(f"  {ps_vol.summary()}")

    ps_ff = orch.size_position(
        "fixed_fraction",
        portfolio_equity=1_000_000,
        entry_price=480.0,
        stop_loss_price=460.0,
        risk_per_trade=0.01,
    )
    print(f"  {ps_ff.summary()}")

    opt_f = orch.compute_optimal_f(sample_returns)
    print(f"\n  Optimal-f (Ralph Vince) for SPY: {opt_f:.4f}")

    print("\nDone.")
