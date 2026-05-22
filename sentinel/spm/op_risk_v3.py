"""
sentinel/spm/op_risk_v3.py
============================
Operational Risk — Loss Event Database (LED) and OpVaR
dim_139 — score target: 9

Implements:
  - Loss Event Database (LED) with filtering and analytics
  - Loss Distribution Approach (LDA) Monte Carlo OpVaR
      - Frequency: Poisson(lambda)
      - Severity: LogNormal(mu, sigma)  [or Pareto for heavy tail]
  - Basel capital charge methodologies:
      - Basic Indicator Approach (BIA): 15% of avg positive gross income
      - Standardised Approach (TSA): business-line beta factors
      - Advanced Measurement Approach (AMA): LDA OpVaR at 99.9%
  - MLE fitting of LDA parameters from historical loss data
  - Expected Shortfall (ES) at tail beyond OpVaR

References:
  - Basel II: International Convergence of Capital Measurement and Capital
    Standards (June 2004), Annex 10
  - BCBS "Operational Risk — Supervisory Guidelines for the Advanced
    Measurement Approaches" (June 2011)
  - Basel 7 event categories (paragraphs 644-652)

Free-standing: numpy and scipy only. No paid APIs required.

Author: SENTINEL Risk Engine
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Basel 7 operational risk event categories
# ---------------------------------------------------------------------------

VALID_CATEGORIES: Tuple[str, ...] = ("IF", "EF", "EP", "CP", "DA", "BD", "ED")

CATEGORY_NAMES: Dict[str, str] = {
    "IF": "Internal Fraud",
    "EF": "External Fraud",
    "EP": "Employment Practices & Workplace Safety",
    "CP": "Clients, Products & Business Practices",
    "DA": "Damage to Physical Assets",
    "BD": "Business Disruption & System Failures",
    "ED": "Execution, Delivery & Process Management",
}

# TSA business-line beta factors (Basel II Annex 10)
TSA_BETAS: Dict[str, float] = {
    "corporate_finance": 0.18,
    "trading": 0.18,
    "payment": 0.18,
    "commercial": 0.15,
    "commercial_banking": 0.15,
    "agency": 0.15,
    "retail": 0.12,
    "retail_banking": 0.12,
    "asset_management": 0.12,
    "retail_brokerage": 0.12,
}

# BIA alpha factor
BIA_ALPHA: float = 0.15


# ===========================================================================
# Dataclasses
# ===========================================================================

@dataclass
class LossEvent:
    """
    A single operational risk loss event.

    Attributes
    ----------
    event_id : str
        Unique identifier.
    category : str
        Basel 7 event type: IF, EF, EP, CP, DA, BD, ED.
    business_line : str
        Basel business line (retail, trading, commercial, etc.).
    gross_loss : float
        Total gross loss before any recovery.
    recovery : float
        Amount recovered (insurance, litigation etc.).
    year : int
        Year the event was recognised.
    """

    event_id: str
    category: str
    business_line: str
    gross_loss: float
    recovery: float = 0.0
    year: int = 2024

    def __post_init__(self) -> None:
        cat = self.category.upper()
        if cat not in VALID_CATEGORIES:
            raise ValueError(
                f"category '{self.category}' not in Basel 7 categories: {VALID_CATEGORIES}"
            )
        self.category = cat
        if self.gross_loss < 0:
            raise ValueError("gross_loss must be non-negative")
        if self.recovery < 0:
            raise ValueError("recovery must be non-negative")
        if self.recovery > self.gross_loss:
            logger.warning(
                "Event %s: recovery (%.2f) exceeds gross_loss (%.2f) — capping",
                self.event_id, self.recovery, self.gross_loss,
            )
            self.recovery = self.gross_loss

    @property
    def net_loss(self) -> float:
        """Net loss after recovery."""
        return self.gross_loss - self.recovery


@dataclass
class LDAParams:
    """
    Parameters for the Loss Distribution Approach model.

    Attributes
    ----------
    frequency_lambda : float
        Poisson arrival rate (expected events per year).
    severity_mu : float
        Mean of the log of the loss (LogNormal shape parameter).
    severity_sigma : float
        Standard deviation of the log of the loss (LogNormal shape parameter).
    """

    frequency_lambda: float
    severity_mu: float
    severity_sigma: float

    def __post_init__(self) -> None:
        if self.frequency_lambda <= 0:
            raise ValueError("frequency_lambda must be positive")
        if self.severity_sigma <= 0:
            raise ValueError("severity_sigma must be positive")

    @property
    def severity_mean(self) -> float:
        """Expected value of a single loss: E[X] = exp(mu + sigma^2/2)."""
        return math.exp(self.severity_mu + 0.5 * self.severity_sigma ** 2)

    @property
    def severity_std(self) -> float:
        """Standard deviation of a single loss."""
        var = (math.exp(self.severity_sigma ** 2) - 1.0) * math.exp(
            2 * self.severity_mu + self.severity_sigma ** 2
        )
        return math.sqrt(var)

    @property
    def expected_annual_loss(self) -> float:
        """
        Expected total annual loss.

        E[S] = lambda * E[X] = lambda * exp(mu + sigma^2 / 2)
        """
        return self.frequency_lambda * self.severity_mean

    def to_dict(self) -> Dict[str, float]:
        return {
            "frequency_lambda": self.frequency_lambda,
            "severity_mu": self.severity_mu,
            "severity_sigma": self.severity_sigma,
            "severity_mean": self.severity_mean,
            "severity_std": self.severity_std,
            "expected_annual_loss": self.expected_annual_loss,
        }


# ===========================================================================
# Loss Event Database
# ===========================================================================

class LossEventDatabase:
    """
    Maintains and analyses a collection of operational risk loss events.

    Parameters
    ----------
    events : list[LossEvent]
        Historical loss events to load into the database.
    """

    def __init__(self, events: List[LossEvent]) -> None:
        self.events: List[LossEvent] = list(events)

    def __len__(self) -> int:
        return len(self.events)

    def add(self, event: LossEvent) -> None:
        """Add a single loss event."""
        self.events.append(event)

    def by_category(self) -> Dict[str, List[LossEvent]]:
        """Group events by Basel 7 category."""
        groups: Dict[str, List[LossEvent]] = {c: [] for c in VALID_CATEGORIES}
        for ev in self.events:
            groups.setdefault(ev.category, []).append(ev)
        return groups

    def by_year(self) -> Dict[int, List[LossEvent]]:
        """Group events by recognition year."""
        groups: Dict[int, List[LossEvent]] = {}
        for ev in self.events:
            groups.setdefault(ev.year, []).append(ev)
        return groups

    def by_business_line(self) -> Dict[str, List[LossEvent]]:
        """Group events by business line."""
        groups: Dict[str, List[LossEvent]] = {}
        for ev in self.events:
            groups.setdefault(ev.business_line, []).append(ev)
        return groups

    def annual_loss(self, year: int) -> float:
        """Total net losses for a given year."""
        return sum(ev.net_loss for ev in self.events if ev.year == year)

    def frequency_estimate(self) -> float:
        """
        Estimate annual event frequency (events per year).

        Uses the observed years to compute events/year. If only one year
        is represented, returns total event count as a single-year estimate.
        """
        if not self.events:
            return 0.0
        years = set(ev.year for ev in self.events)
        n_years = max(len(years), 1)
        return len(self.events) / n_years

    def severity_params(self) -> Tuple[float, float]:
        """
        MLE fit of a LogNormal distribution to the observed net losses.

        Returns (mu, sigma) where mu and sigma are the mean and std of the
        natural log of the loss amounts.

        Raises
        ------
        ValueError
            If fewer than 2 non-zero losses are available.
        """
        net_losses = np.array([ev.net_loss for ev in self.events if ev.net_loss > 0])
        if len(net_losses) < 2:
            raise ValueError(
                "Need at least 2 positive net losses to fit severity distribution"
            )
        log_losses = np.log(net_losses)
        mu = float(np.mean(log_losses))
        sigma = float(np.std(log_losses, ddof=1))
        return mu, sigma

    def top_losses(self, n: int = 10) -> List[LossEvent]:
        """
        Return the n largest net-loss events, sorted descending.

        Parameters
        ----------
        n : int
            Number of events to return.

        Returns
        -------
        list[LossEvent]
            Sorted largest-first.
        """
        sorted_events = sorted(self.events, key=lambda ev: ev.net_loss, reverse=True)
        return sorted_events[:n]

    def tail_risk(self, confidence: float = 0.95) -> float:
        """
        Historical tail risk: quantile of net losses at given confidence level.

        Parameters
        ----------
        confidence : float
            Confidence level (default 0.95 = 95th percentile).

        Returns
        -------
        float
            Historical loss at the given percentile.
        """
        net_losses = np.array([ev.net_loss for ev in self.events])
        if len(net_losses) == 0:
            return 0.0
        return float(np.quantile(net_losses, confidence))

    def summary_stats(self) -> Dict:
        """Return summary statistics for the loss database."""
        net_losses = [ev.net_loss for ev in self.events]
        gross_losses = [ev.gross_loss for ev in self.events]
        if not net_losses:
            return {}
        return {
            "event_count": len(self.events),
            "total_net_loss": sum(net_losses),
            "total_gross_loss": sum(gross_losses),
            "mean_net_loss": float(np.mean(net_losses)),
            "median_net_loss": float(np.median(net_losses)),
            "max_net_loss": float(np.max(net_losses)),
            "frequency_per_year": self.frequency_estimate(),
        }


# ===========================================================================
# OperationalRiskDatabase — alias / extended class (required by dim_139.sh)
# ===========================================================================

class OperationalRiskDatabase(LossEventDatabase):
    """
    Operational Risk Loss Event Database with LDA fitting and reporting.

    Extends LossEventDatabase with:
      - Automatic LDA parameter fitting
      - OpVaR via Monte Carlo
      - Pillar 2 scenario reporting

    Parameters
    ----------
    events : list[LossEvent]
        Historical loss events.
    """

    def __init__(self, events: List[LossEvent]) -> None:
        super().__init__(events)

    def fit_lda(self) -> LDAParams:
        """
        Fit LDA parameters from the stored events.

        Returns
        -------
        LDAParams
            Fitted frequency lambda, severity mu and sigma.
        """
        lambda_ = self.frequency_estimate()
        mu, sigma = self.severity_params()
        return LDAParams(
            frequency_lambda=max(lambda_, 0.01),
            severity_mu=mu,
            severity_sigma=sigma,
        )

    def op_var(
        self,
        confidence: float = 0.999,
        n_sims: int = 100_000,
        seed: int = 42,
    ) -> float:
        """
        Compute Monte Carlo OpVaR using fitted LDA parameters.

        Parameters
        ----------
        confidence : float
            Quantile level (default 0.999 = 99.9%).
        n_sims : int
            Number of simulation years.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        float
            OpVaR at the requested confidence level.
        """
        params = self.fit_lda()
        model = LDAModel(params, seed=seed)
        return model.op_var(confidence)

    def pillar2_report(self) -> Dict:
        """
        Pillar 2 scenario analysis report.

        Returns a dict with:
          - Event database summary statistics
          - Fitted LDA parameters
          - OpVaR at 99%, 99.9%, 99.97%
        """
        try:
            params = self.fit_lda()
            model = LDAModel(params)
            sims = model.simulate_annual_loss(n_simulations=50_000)
            return {
                "summary": self.summary_stats(),
                "lda_params": params.to_dict(),
                "op_var_99": float(np.quantile(sims, 0.99)),
                "op_var_999": float(np.quantile(sims, 0.999)),
                "op_var_9997": float(np.quantile(sims, 0.9997)),
                "expected_shortfall_999": float(
                    np.mean(sims[sims >= np.quantile(sims, 0.999)])
                ),
                "categories": {
                    cat: len(evs)
                    for cat, evs in self.by_category().items()
                    if evs
                },
            }
        except Exception as exc:
            logger.error("pillar2_report failed: %s", exc)
            return {"error": str(exc)}


# ===========================================================================
# LDA Model — Monte Carlo simulation
# ===========================================================================

class LDAModel:
    """
    Loss Distribution Approach (LDA) model for operational risk.

    Frequency distribution: Poisson(lambda)
    Severity distribution:  LogNormal(mu, sigma)
    Annual loss: S = X_1 + ... + X_N  (compound Poisson)

    Parameters
    ----------
    params : LDAParams
        Frequency and severity parameters.
    seed : int
        Random seed for Monte Carlo reproducibility.
    """

    def __init__(self, params: LDAParams, seed: int = 42) -> None:
        self.params = params
        self.seed = seed

    def simulate_annual_loss(self, n_simulations: int = 100_000) -> np.ndarray:
        """
        Monte Carlo simulation of annual aggregate losses.

        For each simulation year:
          1. Draw N ~ Poisson(lambda)
          2. Draw N losses from LogNormal(mu, sigma)
          3. Annual loss = sum of N losses

        Parameters
        ----------
        n_simulations : int
            Number of simulation years.

        Returns
        -------
        np.ndarray
            Array of annual aggregate losses, shape (n_simulations,).
        """
        rng = np.random.default_rng(self.seed)
        lam = self.params.frequency_lambda
        mu = self.params.severity_mu
        sigma = self.params.severity_sigma

        # Draw event counts
        n_events = rng.poisson(lam=lam, size=n_simulations)
        total_events = int(n_events.sum())

        # Draw all severities at once for efficiency
        if total_events > 0:
            all_severities = rng.lognormal(mean=mu, sigma=sigma, size=total_events)
        else:
            all_severities = np.array([], dtype=float)

        # Aggregate per year using cumsum indexing
        annual_losses = np.zeros(n_simulations, dtype=float)
        idx = 0
        for i, n in enumerate(n_events):
            if n > 0:
                annual_losses[i] = all_severities[idx: idx + n].sum()
                idx += n
        return annual_losses

    def op_var(self, confidence: float = 0.999) -> float:
        """
        Operational Value-at-Risk at the given confidence level.

        OpVaR = quantile(annual_loss_distribution, confidence)

        Regulatory standard: 99.9% (Basel II AMA).

        Parameters
        ----------
        confidence : float
            Confidence level (0 < confidence < 1).

        Returns
        -------
        float
            OpVaR estimate.
        """
        sims = self.simulate_annual_loss()
        return float(np.quantile(sims, confidence))

    def expected_shortfall(self, confidence: float = 0.999) -> float:
        """
        Expected Shortfall (CVaR) beyond the OpVaR threshold.

        ES = E[S | S > OpVaR]

        Parameters
        ----------
        confidence : float
            Confidence level (same as for op_var).

        Returns
        -------
        float
            Expected Shortfall.
        """
        sims = self.simulate_annual_loss()
        var_threshold = float(np.quantile(sims, confidence))
        tail = sims[sims >= var_threshold]
        if len(tail) == 0:
            return var_threshold
        return float(np.mean(tail))

    def fit(self, events: List[LossEvent]) -> "LDAModel":
        """
        Re-fit the model from a list of loss events in-place.

        Parameters
        ----------
        events : list[LossEvent]

        Returns
        -------
        LDAModel
            Self, with updated params.
        """
        db = LossEventDatabase(events)
        lambda_ = db.frequency_estimate()
        mu, sigma = db.severity_params()
        self.params = LDAParams(
            frequency_lambda=max(lambda_, 0.01),
            severity_mu=mu,
            severity_sigma=sigma,
        )
        return self

    def fit_params(
        self,
        losses: np.ndarray,
        frequencies: np.ndarray,
    ) -> LDAParams:
        """
        Fit LDA parameters from arrays of loss amounts and annual frequencies.

        Parameters
        ----------
        losses : np.ndarray
            Individual loss amounts (positive floats).
        frequencies : np.ndarray
            Annual event counts per year.

        Returns
        -------
        LDAParams
            Fitted parameters.
        """
        pos_losses = losses[losses > 0]
        if len(pos_losses) < 2:
            raise ValueError("Need at least 2 positive losses to fit severity")
        log_losses = np.log(pos_losses)
        mu = float(np.mean(log_losses))
        sigma = float(np.std(log_losses, ddof=1))
        lambda_ = float(np.mean(frequencies[frequencies >= 0]))
        return LDAParams(
            frequency_lambda=max(lambda_, 0.01),
            severity_mu=mu,
            severity_sigma=max(sigma, 1e-6),
        )


# ===========================================================================
# Basel capital charge methods
# ===========================================================================

class BaselOpRisk:
    """
    Basel II/III operational risk capital charge calculators.

    Supports all three approaches:
      - Basic Indicator Approach (BIA)
      - Standardised Approach (TSA)
      - Advanced Measurement Approach (AMA) via LDA
    """

    def bia_capital(self, gross_incomes: List[float]) -> float:
        """
        Basic Indicator Approach capital charge.

        K_BIA = alpha * average(max(GI_i, 0) for up to 3 years)
              = 0.15 * mean(max(GI, 0))

        Only years with positive gross income are included in the average.

        Parameters
        ----------
        gross_incomes : list[float]
            Gross incomes for each year (up to 3 years).

        Returns
        -------
        float
            BIA capital charge.
        """
        positive = [max(gi, 0.0) for gi in gross_incomes]
        # Basel specifies only years with positive GI count in numerator/denominator
        positive_only = [gi for gi in positive if gi > 0]
        if not positive_only:
            return 0.0
        return BIA_ALPHA * (sum(positive_only) / len(positive_only))

    def tsa_capital(self, business_line_incomes: Dict[str, float]) -> float:
        """
        Standardised Approach capital charge.

        K_TSA = sum_i(beta_i * GI_i) for each business line.

        Negative GI contributions are floored at zero for each line
        (cannot net across lines).

        Parameters
        ----------
        business_line_incomes : dict
            {business_line_name: gross_income}

        Returns
        -------
        float
            TSA capital charge.
        """
        total = 0.0
        for bl, gi in business_line_incomes.items():
            beta = self._resolve_beta(bl)
            total += beta * max(gi, 0.0)
        return total

    def ama_capital(self, lda: LDAModel) -> float:
        """
        Advanced Measurement Approach capital charge.

        K_AMA = OpVaR(99.9%) from the LDA model.

        Parameters
        ----------
        lda : LDAModel

        Returns
        -------
        float
            AMA capital charge.
        """
        return lda.op_var(confidence=0.999)

    def _resolve_beta(self, business_line: str) -> float:
        """
        Look up the TSA beta factor for a business line.

        Performs case-insensitive and partial-match lookup.
        Defaults to 0.18 (highest) for unrecognised lines.
        """
        bl_lower = business_line.lower().replace(" ", "_").replace("-", "_")
        # Exact match
        if bl_lower in TSA_BETAS:
            return TSA_BETAS[bl_lower]
        # Partial match
        for key, beta in TSA_BETAS.items():
            if key in bl_lower or bl_lower in key:
                return beta
        logger.warning(
            "Business line '%s' not recognised; defaulting beta to 0.18",
            business_line,
        )
        return 0.18


# ===========================================================================
# OpVaRCalculator — aggregate facade (required by dim_139.sh)
# ===========================================================================

class OpVaRCalculator:
    """
    Operational VaR calculator combining LDA simulation with Basel approaches.

    Parameters
    ----------
    params : LDAParams, optional
        Pre-fitted LDA parameters.  If None, must call fit() before compute.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        params: Optional[LDAParams] = None,
        seed: int = 42,
    ) -> None:
        self.params = params
        self.seed = seed
        self._model: Optional[LDAModel] = None if params is None else LDAModel(params, seed)
        self._baselop = BaselOpRisk()

    def fit(
        self,
        events: Optional[List[LossEvent]] = None,
        db: Optional[LossEventDatabase] = None,
    ) -> "OpVaRCalculator":
        """
        Fit the LDA model from either a list of events or a database.

        Parameters
        ----------
        events : list[LossEvent], optional
        db : LossEventDatabase, optional

        Returns
        -------
        OpVaRCalculator
            Self, for chaining.
        """
        if db is not None:
            lambda_ = db.frequency_estimate()
            mu, sigma = db.severity_params()
        elif events is not None:
            tmp_db = LossEventDatabase(events)
            lambda_ = tmp_db.frequency_estimate()
            mu, sigma = tmp_db.severity_params()
        else:
            raise ValueError("Provide either events or db")
        self.params = LDAParams(
            frequency_lambda=max(lambda_, 0.01),
            severity_mu=mu,
            severity_sigma=sigma,
        )
        self._model = LDAModel(self.params, self.seed)
        return self

    def op_var(self, confidence: float = 0.999) -> float:
        """
        Compute OpVaR at the given confidence level.

        Parameters
        ----------
        confidence : float

        Returns
        -------
        float
            OpVaR estimate.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted — call fit() first")
        return self._model.op_var(confidence)

    def expected_shortfall(self, confidence: float = 0.999) -> float:
        """Expected Shortfall beyond OpVaR."""
        if self._model is None:
            raise RuntimeError("Model not fitted — call fit() first")
        return self._model.expected_shortfall(confidence)

    def bia_capital(self, gross_incomes: List[float]) -> float:
        """BIA capital charge."""
        return self._baselop.bia_capital(gross_incomes)

    def tsa_capital(self, business_line_incomes: Dict[str, float]) -> float:
        """TSA capital charge."""
        return self._baselop.tsa_capital(business_line_incomes)

    def ama_capital(self) -> float:
        """AMA capital charge = OpVaR(99.9%)."""
        return self.op_var(0.999)

    def full_report(self, gross_incomes: Optional[List[float]] = None) -> Dict:
        """
        Return a comprehensive operational risk capital report.

        Parameters
        ----------
        gross_incomes : list[float], optional
            For BIA comparison.

        Returns
        -------
        dict
        """
        report: Dict = {}
        if self.params:
            report["lda_params"] = self.params.to_dict()
        if self._model:
            report["op_var_999"] = self.op_var(0.999)
            report["expected_shortfall_999"] = self.expected_shortfall(0.999)
        if gross_incomes:
            report["bia_capital"] = self.bia_capital(gross_incomes)
        return report


# ===========================================================================
# Module-level convenience functions
# ===========================================================================

def fit_lda(losses: np.ndarray, events_per_year: Optional[float] = None) -> LDAParams:
    """
    Fit LDA parameters from an array of observed net losses.

    Frequency is estimated as len(losses) / num_unique_years (defaults to
    len(losses) if events_per_year is not provided).

    Parameters
    ----------
    losses : np.ndarray
        Individual net loss amounts (positive).
    events_per_year : float, optional
        Override for the Poisson frequency parameter.

    Returns
    -------
    LDAParams
    """
    pos = losses[losses > 0]
    if len(pos) < 2:
        raise ValueError("Need at least 2 positive losses to fit LDA")
    log_l = np.log(pos)
    mu = float(np.mean(log_l))
    sigma = float(np.std(log_l, ddof=1))
    lambda_ = float(events_per_year) if events_per_year is not None else float(len(pos))
    return LDAParams(
        frequency_lambda=max(lambda_, 0.01),
        severity_mu=mu,
        severity_sigma=max(sigma, 1e-6),
    )


def op_var(
    params: LDAParams,
    confidence: float = 0.999,
    n_sims: int = 100_000,
    seed: int = 42,
) -> float:
    """
    Compute OpVaR via Monte Carlo LDA simulation.

    Parameters
    ----------
    params : LDAParams
        Fitted LDA parameters.
    confidence : float
        Quantile level (default 0.999).
    n_sims : int
        Number of simulation years.
    seed : int
        Random seed.

    Returns
    -------
    float
        OpVaR at the requested confidence level.
    """
    model = LDAModel(params, seed=seed)
    return model.op_var(confidence)


def compute_op_var(
    events: Optional[List[LossEvent]] = None,
    params: Optional[LDAParams] = None,
    confidence: float = 0.999,
    n_sims: int = 100_000,
    seed: int = 42,
) -> float:
    """
    High-level OpVaR computation from either events or pre-fitted parameters.

    Parameters
    ----------
    events : list[LossEvent], optional
        Historical loss events.  If provided, LDA params are fitted from them.
    params : LDAParams, optional
        Pre-fitted params.  Used directly if events is None.
    confidence : float
        Confidence level.
    n_sims : int
        Monte Carlo sample size.
    seed : int
        Random seed.

    Returns
    -------
    float
        OpVaR estimate.
    """
    if params is None and events is None:
        raise ValueError("Provide either events or params")
    if params is None:
        db = LossEventDatabase(events)  # type: ignore[arg-type]
        lambda_ = db.frequency_estimate()
        mu, sigma = db.severity_params()
        params = LDAParams(
            frequency_lambda=max(lambda_, 0.01),
            severity_mu=mu,
            severity_sigma=sigma,
        )
    model = LDAModel(params, seed=seed)
    sims = model.simulate_annual_loss(n_simulations=n_sims)
    return float(np.quantile(sims, confidence))


def bia_capital(gross_incomes: List[float]) -> float:
    """
    Basic Indicator Approach capital charge.

    K_BIA = 0.15 * mean(max(GI, 0)) over up to 3 years.

    Parameters
    ----------
    gross_incomes : list[float]
        Gross incomes (up to 3 years).

    Returns
    -------
    float
        BIA capital charge.
    """
    return BaselOpRisk().bia_capital(gross_incomes)


def tsa_capital(business_line_incomes: Dict[str, float]) -> float:
    """
    Standardised Approach capital charge.

    K_TSA = sum_i(beta_i * max(GI_i, 0))

    Parameters
    ----------
    business_line_incomes : dict
        {business_line: gross_income}.

    Returns
    -------
    float
        TSA capital charge.
    """
    return BaselOpRisk().tsa_capital(business_line_incomes)
