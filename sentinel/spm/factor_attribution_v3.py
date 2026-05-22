"""
sentinel/spm/factor_attribution_v3.py
dim_144: Multi-Factor Return Attribution Engine (score target: 9)

Barra-style multi-factor attribution decomposing portfolio returns into
systematic (factor-driven) and idiosyncratic (stock-specific) components.

Attribution identity:
    R_p = Σ_k (w_p,k * f_k) + Σ_i (w_i * ε_i)
        = factor_return   +   specific_return

Where:
    R_p     = portfolio return
    w_p,k   = portfolio factor exposure to factor k (Σ_i w_i * β_i,k)
    f_k     = factor return for factor k
    w_i     = stock weight
    ε_i     = idiosyncratic (stock-specific) return = r_i - Σ_k (β_i,k * f_k)

Academic references:
    Barra (1975)          "Security Analysis for Portfolio Management"
    Ross (1976)           "The Arbitrage Theory of Capital Asset Pricing"
    Fama & French (1993)  3-factor model (Mkt, SMB, HML)
    Carhart (1997)        Momentum factor
    Frazzini & Pedersen   "Betting Against Beta" — low-vol factor
    Novy-Marx (2013)      "The Other Side of Value" — profitability/quality

Factors supported:
    market   — CAPM beta (Mkt-RF)
    size     — SMB log-market-cap z-score
    value    — HML book-to-market z-score
    momentum — 12-1 month return z-score
    quality  — ROE/profitability z-score
    low_vol  — low volatility z-score
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported factor names (order matters for factor_breakdown dict ordering)
# ---------------------------------------------------------------------------
SUPPORTED_FACTORS: List[str] = [
    "market",
    "size",
    "value",
    "momentum",
    "quality",
    "low_vol",
]

# Map factor name -> FactorExposure attribute name
_FACTOR_ATTR: Dict[str, str] = {
    "market": "market_beta",
    "size": "size",
    "value": "value",
    "momentum": "momentum",
    "quality": "quality",
    "low_vol": "low_vol",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FactorExposure:
    """Factor loadings (betas/z-scores) for a single security.

    Attributes
    ----------
    ticker:       Security identifier.
    market_beta:  CAPM beta — sensitivity to market return.
    size:         Log market-cap z-score (positive = small-cap tilt).
    value:        Book-to-market z-score (positive = value tilt).
    momentum:     12-1 month return z-score (positive = recent winner).
    quality:      ROE/profitability z-score (positive = high quality).
    low_vol:      Idiosyncratic volatility z-score (positive = low vol).
    """
    ticker: str
    market_beta: float = 1.0
    size: float = 0.0
    value: float = 0.0
    momentum: float = 0.0
    quality: float = 0.0
    low_vol: float = 0.0


@dataclass
class SystematicReturn:
    """Factor-driven component of portfolio return.

    Attributes
    ----------
    total_factor_return:    Sum of all factor contributions.
    market_contribution:    Portfolio beta × market factor return.
    size_contribution:      Portfolio size exposure × size factor return.
    value_contribution:     Portfolio value exposure × value factor return.
    momentum_contribution:  Portfolio momentum exposure × momentum factor return.
    quality_contribution:   Portfolio quality exposure × quality factor return.
    low_vol_contribution:   Portfolio low-vol exposure × low-vol factor return.
    factor_breakdown:       Dict mapping factor name → contribution.
    """
    total_factor_return: float
    market_contribution: float
    size_contribution: float
    value_contribution: float
    momentum_contribution: float
    quality_contribution: float
    low_vol_contribution: float
    factor_breakdown: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Auto-populate factor_breakdown if empty
        if not self.factor_breakdown:
            self.factor_breakdown = {
                "market": self.market_contribution,
                "size": self.size_contribution,
                "value": self.value_contribution,
                "momentum": self.momentum_contribution,
                "quality": self.quality_contribution,
                "low_vol": self.low_vol_contribution,
            }


@dataclass
class IdiosyncraticReturn:
    """Stock-specific (residual) component of portfolio return.

    Attributes
    ----------
    total_specific_return:  Weighted sum of idiosyncratic returns.
    stock_contributions:    Dict mapping ticker → weighted specific return.
    top_contributors:       Top 3 positive contributors [(ticker, raw_ε, contribution)].
    bottom_contributors:    Bottom 3 negative contributors.
    """
    total_specific_return: float
    stock_contributions: Dict[str, float] = field(default_factory=dict)
    top_contributors: List[Tuple[str, float, float]] = field(default_factory=list)
    bottom_contributors: List[Tuple[str, float, float]] = field(default_factory=list)


@dataclass
class AttributionResult:
    """Full Barra-style attribution decomposition for a portfolio.

    Attributes
    ----------
    total_portfolio_return:  Σ_i w_i * r_i
    systematic:              Factor-driven return component.
    idiosyncratic:           Stock-specific return component.
    r_squared:               Fraction of portfolio variance explained by factors.
                             Computed as (systematic / total)² bounded to [0,1].
    """
    total_portfolio_return: float
    systematic: SystematicReturn
    idiosyncratic: IdiosyncraticReturn
    r_squared: float

    @property
    def factor_return(self) -> float:
        """Alias for systematic.total_factor_return."""
        return self.systematic.total_factor_return

    @property
    def specific_return(self) -> float:
        """Alias for idiosyncratic.total_specific_return."""
        return self.idiosyncratic.total_specific_return

    def verify_decomposition(self, tol: float = 1e-6) -> bool:
        """Check that factor + specific = total (attribution identity)."""
        residual = abs(
            self.total_portfolio_return
            - self.systematic.total_factor_return
            - self.idiosyncratic.total_specific_return
        )
        return residual < tol


# ---------------------------------------------------------------------------
# Main attribution engine
# ---------------------------------------------------------------------------

class MultiFactorAttribution:
    """Barra-style multi-factor return attribution engine.

    Parameters
    ----------
    factors:  List of factor names to use (default: all 6 SUPPORTED_FACTORS).

    Usage
    -----
    >>> engine = MultiFactorAttribution()
    >>> result = engine.compute(weights, stock_returns, factor_returns, exposures)
    >>> print(result.systematic.market_contribution)
    """

    def __init__(self, factors: Optional[List[str]] = None) -> None:
        if factors is None:
            self.factors = list(SUPPORTED_FACTORS)
        else:
            unsupported = set(factors) - set(SUPPORTED_FACTORS)
            if unsupported:
                logger.warning("Unsupported factors ignored: %s", unsupported)
            self.factors = [f for f in SUPPORTED_FACTORS if f in factors]
            # Preserve any extra custom factors not in SUPPORTED_FACTORS
            for f in factors:
                if f not in SUPPORTED_FACTORS and f not in self.factors:
                    self.factors.append(f)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        weights: Dict[str, float],
        stock_returns: Dict[str, float],
        factor_returns: Dict[str, float],
        exposures: Dict[str, FactorExposure],
    ) -> AttributionResult:
        """Full Barra attribution decomposition.

        Parameters
        ----------
        weights:        {ticker: portfolio weight}.  Weights should sum to ~1.
        stock_returns:  {ticker: total return for the period}.
        factor_returns: {factor_name: factor return for the period}.
        exposures:      {ticker: FactorExposure}.

        Returns
        -------
        AttributionResult with systematic + idiosyncratic breakdown.
        """
        # 1. Portfolio total return = Σ w_i * r_i
        total_return = self._compute_portfolio_return(weights, stock_returns)

        # 2. Systematic (factor) return
        systematic = self.compute_factor_return(weights, exposures, factor_returns)

        # 3. Idiosyncratic (specific) return
        idiosyncratic = self.compute_specific_return(
            weights, stock_returns, systematic, exposures, factor_returns
        )

        # 4. R-squared: fraction explained by factors
        r_squared = self._compute_r_squared(
            total_return, systematic.total_factor_return
        )

        return AttributionResult(
            total_portfolio_return=total_return,
            systematic=systematic,
            idiosyncratic=idiosyncratic,
            r_squared=r_squared,
        )

    def compute_factor_return(
        self,
        weights: Dict[str, float],
        exposures: Dict[str, FactorExposure],
        factor_returns: Dict[str, float],
    ) -> SystematicReturn:
        """Compute the systematic (factor-driven) portfolio return.

        For each factor k:
            portfolio_exposure_k = Σ_i w_i * β_i,k
            factor_contribution_k = portfolio_exposure_k * f_k

        Returns
        -------
        SystematicReturn with per-factor contributions.
        """
        factor_contributions: Dict[str, float] = {}

        for factor in SUPPORTED_FACTORS:
            attr = _FACTOR_ATTR.get(factor, factor)
            # Portfolio exposure = weighted average of stock exposures
            portfolio_exposure = 0.0
            for ticker, w in weights.items():
                exp = exposures.get(ticker)
                if exp is not None:
                    stock_exposure = getattr(exp, attr, 0.0)
                    portfolio_exposure += w * stock_exposure

            f_return = factor_returns.get(factor, 0.0)
            factor_contributions[factor] = portfolio_exposure * f_return

        # Handle any extra custom factors beyond the 6 standard ones
        for factor in self.factors:
            if factor not in SUPPORTED_FACTORS:
                attr = factor
                portfolio_exposure = 0.0
                for ticker, w in weights.items():
                    exp = exposures.get(ticker)
                    if exp is not None:
                        stock_exposure = getattr(exp, attr, 0.0)
                        portfolio_exposure += w * stock_exposure
                f_return = factor_returns.get(factor, 0.0)
                factor_contributions[factor] = portfolio_exposure * f_return

        total_factor_return = sum(factor_contributions.values())

        return SystematicReturn(
            total_factor_return=total_factor_return,
            market_contribution=factor_contributions.get("market", 0.0),
            size_contribution=factor_contributions.get("size", 0.0),
            value_contribution=factor_contributions.get("value", 0.0),
            momentum_contribution=factor_contributions.get("momentum", 0.0),
            quality_contribution=factor_contributions.get("quality", 0.0),
            low_vol_contribution=factor_contributions.get("low_vol", 0.0),
            factor_breakdown=factor_contributions,
        )

    def compute_specific_return(
        self,
        weights: Dict[str, float],
        stock_returns: Dict[str, float],
        systematic: SystematicReturn,
        exposures: Optional[Dict[str, FactorExposure]] = None,
        factor_returns: Optional[Dict[str, float]] = None,
    ) -> IdiosyncraticReturn:
        """Compute the idiosyncratic (stock-specific) portfolio return.

        For each stock i:
            ε_i = r_i - Σ_k (β_i,k * f_k)      # stock-level residual
            specific_contribution_i = w_i * ε_i  # weighted contribution

        total_specific_return = Σ_i specific_contribution_i

        Parameters
        ----------
        weights:        Portfolio weights.
        stock_returns:  Stock-level total returns.
        systematic:     Pre-computed SystematicReturn (for fallback).
        exposures:      Optional stock exposures (needed for stock-level ε).
        factor_returns: Optional factor returns (needed for stock-level ε).

        Returns
        -------
        IdiosyncraticReturn with per-stock specific contributions.
        """
        stock_contributions: Dict[str, float] = {}
        stock_residuals: Dict[str, float] = {}  # ε_i values

        for ticker, w in weights.items():
            r_i = stock_returns.get(ticker, 0.0)

            # Compute stock-level factor return = Σ_k β_i,k * f_k
            stock_factor_return = 0.0
            if exposures is not None and factor_returns is not None:
                exp = exposures.get(ticker)
                if exp is not None:
                    for factor in SUPPORTED_FACTORS:
                        attr = _FACTOR_ATTR.get(factor, factor)
                        beta = getattr(exp, attr, 0.0)
                        f_return = factor_returns.get(factor, 0.0)
                        stock_factor_return += beta * f_return

            epsilon_i = r_i - stock_factor_return
            stock_residuals[ticker] = epsilon_i
            stock_contributions[ticker] = w * epsilon_i

        total_specific = sum(stock_contributions.values())

        # Build top/bottom contributors sorted by weighted contribution
        all_contribs: List[Tuple[str, float, float]] = [
            (ticker, stock_residuals.get(ticker, 0.0), contrib)
            for ticker, contrib in stock_contributions.items()
        ]
        sorted_contribs = sorted(all_contribs, key=lambda x: x[2], reverse=True)

        top_n = 3
        top_contributors = sorted_contribs[:top_n]
        bottom_contributors = sorted_contribs[-top_n:] if len(sorted_contribs) > top_n else []
        bottom_contributors = list(reversed(bottom_contributors))

        return IdiosyncraticReturn(
            total_specific_return=total_specific,
            stock_contributions=stock_contributions,
            top_contributors=top_contributors,
            bottom_contributors=bottom_contributors,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_portfolio_return(
        weights: Dict[str, float],
        stock_returns: Dict[str, float],
    ) -> float:
        """Σ_i w_i * r_i"""
        return sum(
            w * stock_returns.get(ticker, 0.0)
            for ticker, w in weights.items()
        )

    @staticmethod
    def _compute_r_squared(total_return: float, factor_return: float) -> float:
        """R² = (factor_return / total_return)², clipped to [0, 1].

        When total_return ≈ 0 the ratio is undefined; return 1.0 if
        factor_return ≈ 0 as well (perfect explanation of zero), else 0.0.
        """
        if abs(total_return) < 1e-10:
            return 1.0 if abs(factor_return) < 1e-10 else 0.0
        ratio = factor_return / total_return
        return max(0.0, min(1.0, ratio ** 2))


# ---------------------------------------------------------------------------
# Convenience function (module-level API)
# ---------------------------------------------------------------------------

def decompose_returns(
    portfolio_returns: Dict[str, float],
    factor_returns: Dict[str, float],
    factor_exposures: Dict[str, FactorExposure],
    weights: Optional[Dict[str, float]] = None,
) -> AttributionResult:
    """Decompose portfolio returns into systematic and idiosyncratic components.

    Convenience wrapper around MultiFactorAttribution.compute().

    Parameters
    ----------
    portfolio_returns:  {ticker: stock return} or {ticker: portfolio return}.
                        If `weights` is None, treats portfolio_returns as
                        pre-weighted contributions and uses equal weights.
    factor_returns:     {factor_name: factor return for the period}.
    factor_exposures:   {ticker: FactorExposure}.
    weights:            Optional {ticker: portfolio weight}.
                        If None, infers equal weights for all tickers.

    Returns
    -------
    AttributionResult — complete Barra attribution breakdown.

    Example
    -------
    >>> result = decompose_returns(
    ...     portfolio_returns={"AAPL": 0.05, "MSFT": 0.03},
    ...     factor_returns={"market": 0.02, "size": -0.005},
    ...     factor_exposures={"AAPL": FactorExposure("AAPL", market_beta=1.2),
    ...                       "MSFT": FactorExposure("MSFT", market_beta=1.1)},
    ...     weights={"AAPL": 0.6, "MSFT": 0.4},
    ... )
    """
    tickers = list(portfolio_returns.keys())

    if weights is None:
        n = len(tickers)
        weights = {t: 1.0 / n for t in tickers} if n > 0 else {}

    engine = MultiFactorAttribution()
    return engine.compute(
        weights=weights,
        stock_returns=portfolio_returns,
        factor_returns=factor_returns,
        exposures=factor_exposures,
    )
