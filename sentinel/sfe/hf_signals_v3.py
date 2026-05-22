"""
High-frequency microstructure signals for the SFE (Signal Feature Engineering) layer.

Re-exports the full implementation from sentinel.sma.hf_microstructure_v3 and
provides the named classes/functions expected by dim_132 capability test.

dim_132 — HF microstructure signals (roll yield, futures basis, VPIN)

Exports
-------
RollYieldSignal         — signal class wrapping roll-yield computation
FuturesBasisSignal      — signal class wrapping futures basis computation
compute_roll_yield      — convenience function
compute_futures_basis   — convenience function

Full API (re-exported)
----------------------
FuturesMicrostructure, RealizedVolAnalytics, OrderFlowImbalance,
HFSignalGenerator, FuturesBasisMetrics, RealizedVolComponents,
futures_basis, roll_yield, realized_vol_decomposition, vpin, ofi_lambda
"""

from __future__ import annotations

from sentinel.sma.hf_microstructure_v3 import (
    FuturesBasisMetrics,
    FuturesMicrostructure,
    HFSignalGenerator,
    OrderFlowImbalance,
    RealizedVolAnalytics,
    RealizedVolComponents,
    futures_basis,
    ofi_lambda,
    realized_vol_decomposition,
    roll_yield,
    vpin,
)

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Signal wrapper classes (SFE layer)
# ---------------------------------------------------------------------------


class RollYieldSignal:
    """Signal class that computes annualized roll yield between futures contracts.

    Usage
    -----
    sig = RollYieldSignal()
    value = sig.compute(near=4505, far=4515, days_to_roll=30)
    # Positive → backwardation (carry positive)
    # Negative → contango (carry negative)
    """

    def __init__(self) -> None:
        self._fm = FuturesMicrostructure()

    def compute(self, near: float, far: float, days_to_roll: float) -> float:
        """Annualized roll yield.

        Parameters
        ----------
        near         : near-contract price
        far          : far-contract price
        days_to_roll : calendar days until near expiry

        Returns
        -------
        float — annualized roll yield
        """
        return self._fm.roll_yield(near, far, days_to_roll)

    def carry_sharpe(
        self, near: float, far: float, days_to_roll: float, sigma: float
    ) -> float:
        """Roll yield divided by realized vol (carry Sharpe ratio)."""
        return self._fm.carry_signal(near, far, days_to_roll, sigma)

    def __repr__(self) -> str:  # pragma: no cover
        return "RollYieldSignal()"


class FuturesBasisSignal:
    """Signal class that computes futures basis metrics.

    Usage
    -----
    sig = FuturesBasisSignal()
    metrics = sig.compute(spot=4500, futures=4515, T=0.25)
    # metrics.basis, metrics.basis_pct, metrics.implied_carry, metrics.market_regime
    """

    def __init__(self) -> None:
        self._fm = FuturesMicrostructure()

    def compute(
        self, spot: float, futures_price: float, T: float = 0.25
    ) -> FuturesBasisMetrics:
        """Compute full basis metrics.

        Parameters
        ----------
        spot          : spot price
        futures_price : futures price
        T             : time to expiry in years

        Returns
        -------
        FuturesBasisMetrics
        """
        return self._fm.basis(spot, futures_price, T)

    def regime(self, spot: float, futures_price: float) -> str:
        """Return 'contango' or 'backwardation'."""
        return "contango" if futures_price >= spot else "backwardation"

    def __repr__(self) -> str:  # pragma: no cover
        return "FuturesBasisSignal()"


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def compute_roll_yield(near: float, far: float, days_to_roll: float) -> float:
    """Annualized roll yield between near and far futures contracts.

    Positive in backwardation, negative in contango.
    """
    return roll_yield(near, far, days_to_roll)


def compute_futures_basis(
    spot: float, futures_price: float, T: float = 0.25
) -> FuturesBasisMetrics:
    """Compute futures basis metrics.

    Parameters
    ----------
    spot          : spot price
    futures_price : futures price
    T             : time to expiry in years
    """
    return futures_basis(spot, futures_price, T)


__all__ = [
    # SFE signal classes
    "RollYieldSignal",
    "FuturesBasisSignal",
    # Convenience functions
    "compute_roll_yield",
    "compute_futures_basis",
    # Re-exported core types
    "FuturesBasisMetrics",
    "RealizedVolComponents",
    "FuturesMicrostructure",
    "RealizedVolAnalytics",
    "OrderFlowImbalance",
    "HFSignalGenerator",
    # Re-exported convenience functions
    "futures_basis",
    "roll_yield",
    "realized_vol_decomposition",
    "vpin",
    "ofi_lambda",
]
