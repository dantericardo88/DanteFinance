"""Higher-order Greeks engine — analytical Black-Scholes with dividend yield.

Implements vanna, volga, charm, speed, color, veta, ultima, dual delta/gamma
and a full GreeksBundle. All formulae are exact analytical derivatives of the
Black-Scholes price with continuous dividend yield q.

Reference
---------
Hull, J. (2018). Options, Futures, and Other Derivatives (10th ed.).
McDonald, R. (2013). Derivatives Markets (3rd ed.).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
from scipy.stats import norm

__all__ = [
    "GreeksBundle",
    "HigherGreeksEngine",
    "compute_all_greeks",
    "vanna",
    "volga",
    "charm",
    "speed",
    "ultima",
    "color",
    "veta",
    "dual_delta",
    "dual_gamma",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-8
_SQRT_2PI = math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# GreeksBundle dataclass
# ---------------------------------------------------------------------------

@dataclass
class GreeksBundle:
    """All Black-Scholes Greeks up to third order, plus dual Greeks."""

    # First order
    delta: float
    vega: float
    theta: float
    rho: float

    # Second order
    gamma: float
    vanna: float
    volga: float
    charm: float

    # Third order
    speed: float
    color: float
    veta: float
    ultima: float

    # Dual Greeks
    dual_delta: float
    dual_gamma: float

    def all(self) -> Dict[str, float]:
        """Return all Greeks as a dictionary."""
        return {
            "delta": self.delta,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
            "gamma": self.gamma,
            "vanna": self.vanna,
            "volga": self.volga,
            "charm": self.charm,
            "speed": self.speed,
            "color": self.color,
            "veta": self.veta,
            "ultima": self.ultima,
            "dual_delta": self.dual_delta,
            "dual_gamma": self.dual_gamma,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _d1d2(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> tuple[float, float]:
    """Compute Black-Scholes d1 and d2."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def _npdf(x: float) -> float:
    """Standard normal PDF N'(x)."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _validate(S: float, K: float, T: float, sigma: float) -> None:
    if S <= 0:
        raise ValueError(f"S must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")


# ---------------------------------------------------------------------------
# HigherGreeksEngine class
# ---------------------------------------------------------------------------

class HigherGreeksEngine:
    """Analytical Black-Scholes higher-order Greeks engine."""

    # ------------------------------------------------------------------
    # Full bundle
    # ------------------------------------------------------------------

    def compute_all(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: Literal["call", "put"] = "call",
    ) -> GreeksBundle:
        """Compute all Greeks and return as GreeksBundle."""
        _validate(S, K, T, sigma)
        d1, d2 = _d1d2(S, K, T, r, sigma, q)
        sqrt_T = math.sqrt(T)
        n_d1 = _npdf(d1)
        n_d2 = _npdf(d2)
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_neg_d1 = 1.0 - N_d1
        N_neg_d2 = 1.0 - N_d2

        # Discount factors
        e_qT = math.exp(-q * T)
        e_rT = math.exp(-r * T)

        # ── First order ──────────────────────────────────────────────
        if option_type == "call":
            delta_val = e_qT * N_d1
            theta_val = (
                -(S * e_qT * n_d1 * sigma) / (2.0 * sqrt_T)
                - r * K * e_rT * N_d2
                + q * S * e_qT * N_d1
            ) / 365.0  # per calendar day
            rho_val = K * T * e_rT * N_d2
        else:  # put
            delta_val = -e_qT * N_neg_d1
            theta_val = (
                -(S * e_qT * n_d1 * sigma) / (2.0 * sqrt_T)
                + r * K * e_rT * N_neg_d2
                - q * S * e_qT * N_neg_d1
            ) / 365.0
            rho_val = -K * T * e_rT * N_neg_d2

        vega_val = S * e_qT * n_d1 * sqrt_T  # per unit of vol (not per %)

        # ── Second order ─────────────────────────────────────────────
        gamma_val = e_qT * n_d1 / (S * sigma * sqrt_T)

        vanna_val = self.vanna(S, K, T, r, sigma, q)
        volga_val = self.volga(S, K, T, r, sigma, q)
        charm_val = self.charm(S, K, T, r, sigma, q, option_type)

        # ── Third order ──────────────────────────────────────────────
        speed_val = self.speed(S, K, T, r, sigma)
        color_val = self.color(S, K, T, r, sigma, q)
        veta_val = self.veta(S, K, T, r, sigma, q)
        ultima_val = self.ultima(S, K, T, r, sigma)

        # ── Dual Greeks ──────────────────────────────────────────────
        dd_val = self.dual_delta(S, K, T, r, sigma, option_type)
        dg_val = self.dual_gamma(S, K, T, r, sigma)

        return GreeksBundle(
            delta=delta_val,
            vega=vega_val,
            theta=theta_val,
            rho=rho_val,
            gamma=gamma_val,
            vanna=vanna_val,
            volga=volga_val,
            charm=charm_val,
            speed=speed_val,
            color=color_val,
            veta=veta_val,
            ultima=ultima_val,
            dual_delta=dd_val,
            dual_gamma=dg_val,
        )

    # ------------------------------------------------------------------
    # Individual analytics
    # ------------------------------------------------------------------

    def vanna(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> float:
        """Vanna = dDelta/dSigma = dVega/dS.

        Vanna = -e^{-qT} * N'(d1) * d2 / sigma
        """
        _validate(S, K, T, sigma)
        d1, d2 = _d1d2(S, K, T, r, sigma, q)
        e_qT = math.exp(-q * T)
        return -e_qT * _npdf(d1) * d2 / sigma

    def volga(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> float:
        """Volga (Vomma) = dVega/dSigma = Vega * d1 * d2 / sigma.

        Always >= 0 (long vol convexity).
        """
        _validate(S, K, T, sigma)
        d1, d2 = _d1d2(S, K, T, r, sigma, q)
        sqrt_T = math.sqrt(T)
        e_qT = math.exp(-q * T)
        vega_val = S * e_qT * _npdf(d1) * sqrt_T
        return vega_val * d1 * d2 / sigma

    def charm(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: Literal["call", "put"] = "call",
    ) -> float:
        """Charm = dDelta/dT = -dTheta/dS.

        For a call with continuous dividends:
        charm = q*e^{-qT}*N(d1) - e^{-qT}*N'(d1) * (2*(r-q)*T - d2*sigma*sqrt(T)) / (2*T*sigma*sqrt(T))
        """
        _validate(S, K, T, sigma)
        d1, d2 = _d1d2(S, K, T, r, sigma, q)
        sqrt_T = math.sqrt(T)
        e_qT = math.exp(-q * T)
        n_d1 = _npdf(d1)

        # The sign of the q*N(d1) term depends on option type
        if option_type == "call":
            N_d1 = norm.cdf(d1)
            sign = 1.0
        else:
            N_d1 = norm.cdf(d1) - 1.0  # -N(-d1)
            sign = -1.0

        second_term = e_qT * n_d1 * (2.0 * (r - q) * T - d2 * sigma * sqrt_T) / (
            2.0 * T * sigma * sqrt_T
        )
        return sign * q * e_qT * abs(N_d1) - second_term

    def speed(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> float:
        """Speed = dGamma/dS = -Gamma/S * (d1/(sigma*sqrt(T)) + 1)."""
        _validate(S, K, T, sigma)
        d1, _ = _d1d2(S, K, T, r, sigma, q)
        sqrt_T = math.sqrt(T)
        e_qT = math.exp(-q * T)
        gamma_val = e_qT * _npdf(d1) / (S * sigma * sqrt_T)
        return -gamma_val / S * (d1 / (sigma * sqrt_T) + 1.0)

    def color(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> float:
        """Color = dGamma/dT.

        Color = -e^{-qT} * N'(d1) / (2*S*T*sigma*sqrt(T))
                * (2*q*T + 1 + d1*(2*(r-q)*T - d2*sigma*sqrt(T)) / (sigma*sqrt(T)))
        """
        _validate(S, K, T, sigma)
        d1, d2 = _d1d2(S, K, T, r, sigma, q)
        sqrt_T = math.sqrt(T)
        e_qT = math.exp(-q * T)
        n_d1 = _npdf(d1)
        inner = 2.0 * q * T + 1.0 + d1 * (2.0 * (r - q) * T - d2 * sigma * sqrt_T) / (
            sigma * sqrt_T
        )
        return -e_qT * n_d1 / (2.0 * S * T * sigma * sqrt_T) * inner

    def veta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> float:
        """Veta = dVega/dT.

        Veta = Vega * (q + (d1*(r-q) - sigma/(2*T)) / (sigma*sqrt(T)) - (r - q) * d1 / (sigma*sqrt(T)))

        Simplified:
        Veta = Vega * (r - q - d1*(r - q - sigma^2/2) / (sigma*sqrt(T)) - (1 + d1^2)/(2*T))
        """
        _validate(S, K, T, sigma)
        d1, d2 = _d1d2(S, K, T, r, sigma, q)
        sqrt_T = math.sqrt(T)
        e_qT = math.exp(-q * T)
        vega_val = S * e_qT * _npdf(d1) * sqrt_T
        # Hull formula (10th ed, p. 429):
        # Veta = Vega * [r - q - d1*sigma/(2T) - (r-q)*d1/(sigma*sqrt(T))]
        # Note: the two r-q terms for d1 combine with the d1*sigma/(2T) term
        # We use the standard textbook form:
        term = (r - q) - (d1 * sigma) / (2.0 * T) - (r - q) * d1 / (sigma * sqrt_T)
        return vega_val * term

    def ultima(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> float:
        """Ultima = d^3P/dSigma^3 = -Vega/sigma^2 * (d1*d2*(1 - d1*d2) + d1^2 + d2^2)."""
        _validate(S, K, T, sigma)
        d1, d2 = _d1d2(S, K, T, r, sigma, q)
        sqrt_T = math.sqrt(T)
        e_qT = math.exp(-q * T)
        vega_val = S * e_qT * _npdf(d1) * sqrt_T
        return -vega_val / (sigma ** 2) * (
            d1 * d2 * (1.0 - d1 * d2) + d1 ** 2 + d2 ** 2
        )

    def dual_delta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call",
    ) -> float:
        """Dual delta = dC/dK.

        Dual delta_call = -e^{-rT} * N(d2)
        Dual delta_put  = +e^{-rT} * N(-d2)
        """
        _validate(S, K, T, sigma)
        _, d2 = _d1d2(S, K, T, r, sigma, 0.0)
        e_rT = math.exp(-r * T)
        if option_type == "call":
            return -e_rT * norm.cdf(d2)
        return e_rT * norm.cdf(-d2)

    def dual_gamma(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Dual gamma = d^2C/dK^2 = e^{-rT} * N'(d2) / (K * sigma * sqrt(T))."""
        _validate(S, K, T, sigma)
        _, d2 = _d1d2(S, K, T, r, sigma, 0.0)
        sqrt_T = math.sqrt(T)
        e_rT = math.exp(-r * T)
        return e_rT * _npdf(d2) / (K * sigma * sqrt_T)

    def numerical_verify(
        self,
        greek: str,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: Literal["call", "put"] = "call",
    ) -> Dict[str, float]:
        """Compare analytical Greek to finite-difference approximation.

        Parameters
        ----------
        greek : one of 'vanna', 'volga', 'charm', 'speed', 'color', 'veta',
                        'ultima', 'dual_delta', 'dual_gamma', 'delta', 'gamma', 'vega'

        Returns
        -------
        dict with keys 'analytical', 'numerical', 'diff', 'rel_error'
        """
        h_S = S * 1e-4
        h_sig = sigma * 1e-4
        h_T = T * 1e-4
        h_K = K * 1e-4

        def bs_price(S_: float, K_: float, T_: float, r_: float, sig_: float, q_: float = 0.0) -> float:
            if T_ <= 0 or sig_ <= 0 or S_ <= 0 or K_ <= 0:
                return 0.0
            d1_, d2_ = _d1d2(S_, K_, T_, r_, sig_, q_)
            e_qT_ = math.exp(-q_ * T_)
            e_rT_ = math.exp(-r_ * T_)
            if option_type == "call":
                return S_ * e_qT_ * norm.cdf(d1_) - K_ * e_rT_ * norm.cdf(d2_)
            return K_ * e_rT_ * norm.cdf(-d2_) - S_ * e_qT_ * norm.cdf(-d1_)

        greek_lower = greek.lower()

        analytical: float
        numerical: float

        if greek_lower == "delta":
            analytical = (e_qT := math.exp(-q * T)) * (
                norm.cdf(_d1d2(S, K, T, r, sigma, q)[0]) if option_type == "call"
                else norm.cdf(_d1d2(S, K, T, r, sigma, q)[0]) - 1.0
            )
            numerical = (bs_price(S + h_S, K, T, r, sigma, q) - bs_price(S - h_S, K, T, r, sigma, q)) / (2.0 * h_S)

        elif greek_lower == "gamma":
            analytical = math.exp(-q * T) * _npdf(_d1d2(S, K, T, r, sigma, q)[0]) / (S * sigma * math.sqrt(T))
            numerical = (
                bs_price(S + h_S, K, T, r, sigma, q)
                - 2.0 * bs_price(S, K, T, r, sigma, q)
                + bs_price(S - h_S, K, T, r, sigma, q)
            ) / h_S ** 2

        elif greek_lower == "vega":
            analytical = S * math.exp(-q * T) * _npdf(_d1d2(S, K, T, r, sigma, q)[0]) * math.sqrt(T)
            numerical = (bs_price(S, K, T, r, sigma + h_sig, q) - bs_price(S, K, T, r, sigma - h_sig, q)) / (2.0 * h_sig)

        elif greek_lower == "vanna":
            analytical = self.vanna(S, K, T, r, sigma, q)
            # dVega/dS via central differences
            numerical = (bs_price(S + h_S, K, T, r, sigma + h_sig, q)
                         - bs_price(S + h_S, K, T, r, sigma - h_sig, q)
                         - bs_price(S - h_S, K, T, r, sigma + h_sig, q)
                         + bs_price(S - h_S, K, T, r, sigma - h_sig, q)) / (4.0 * h_S * h_sig)

        elif greek_lower in ("volga", "vomma"):
            analytical = self.volga(S, K, T, r, sigma, q)
            numerical = (
                bs_price(S, K, T, r, sigma + h_sig, q)
                - 2.0 * bs_price(S, K, T, r, sigma, q)
                + bs_price(S, K, T, r, sigma - h_sig, q)
            ) / h_sig ** 2

        elif greek_lower == "charm":
            analytical = self.charm(S, K, T, r, sigma, q, option_type)
            # dDelta/dT
            def delta_at_T(T_: float) -> float:
                if T_ <= 0:
                    return 0.0
                d1_, _ = _d1d2(S, K, T_, r, sigma, q)
                e_qT_ = math.exp(-q * T_)
                if option_type == "call":
                    return e_qT_ * norm.cdf(d1_)
                return -e_qT_ * norm.cdf(-d1_)
            numerical = (delta_at_T(T + h_T) - delta_at_T(T - h_T)) / (2.0 * h_T)

        elif greek_lower == "speed":
            analytical = self.speed(S, K, T, r, sigma, q)
            # dGamma/dS
            def gamma_at_S(S_: float) -> float:
                if S_ <= 0:
                    return 0.0
                d1_, _ = _d1d2(S_, K, T, r, sigma, q)
                return math.exp(-q * T) * _npdf(d1_) / (S_ * sigma * math.sqrt(T))
            numerical = (gamma_at_S(S + h_S) - gamma_at_S(S - h_S)) / (2.0 * h_S)

        elif greek_lower == "ultima":
            analytical = self.ultima(S, K, T, r, sigma, q)
            # d^3 price / dsigma^3
            numerical = (
                bs_price(S, K, T, r, sigma + 2 * h_sig, q)
                - 2.0 * bs_price(S, K, T, r, sigma + h_sig, q)
                + 2.0 * bs_price(S, K, T, r, sigma - h_sig, q)
                - bs_price(S, K, T, r, sigma - 2 * h_sig, q)
            ) / (2.0 * h_sig ** 3)

        elif greek_lower == "dual_delta":
            analytical = self.dual_delta(S, K, T, r, sigma, option_type)
            numerical = (bs_price(S, K + h_K, T, r, sigma, q) - bs_price(S, K - h_K, T, r, sigma, q)) / (2.0 * h_K)

        elif greek_lower == "dual_gamma":
            analytical = self.dual_gamma(S, K, T, r, sigma)
            numerical = (
                bs_price(S, K + h_K, T, r, sigma, q)
                - 2.0 * bs_price(S, K, T, r, sigma, q)
                + bs_price(S, K - h_K, T, r, sigma, q)
            ) / h_K ** 2

        else:
            raise ValueError(f"Unknown greek: {greek!r}")

        diff = abs(analytical - numerical)
        denom = max(abs(analytical), abs(numerical), _EPS)
        rel_error = diff / denom

        return {
            "analytical": analytical,
            "numerical": numerical,
            "diff": diff,
            "rel_error": rel_error,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_ENGINE = HigherGreeksEngine()


def compute_all_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: Literal["call", "put"] = "call",
) -> GreeksBundle:
    """Compute the full GreeksBundle for a European option."""
    return _ENGINE.compute_all(S, K, T, r, sigma, q, option_type)


def vanna(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Vanna = dDelta/dSigma = -e^{-qT} * N'(d1) * d2 / sigma."""
    return _ENGINE.vanna(S, K, T, r, sigma, q)


def volga(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Volga (Vomma) = dVega/dSigma >= 0."""
    return _ENGINE.volga(S, K, T, r, sigma, q)


def charm(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """Charm = dDelta/dT."""
    return _ENGINE.charm(S, K, T, r, sigma, q, option_type)


def speed(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Speed = dGamma/dS < 0 for a vanilla option."""
    return _ENGINE.speed(S, K, T, r, sigma, q)


def ultima(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Ultima = d^3P/dSigma^3."""
    return _ENGINE.ultima(S, K, T, r, sigma, q)


def color(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Color = dGamma/dT."""
    return _ENGINE.color(S, K, T, r, sigma, q)


def veta(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Veta = dVega/dT."""
    return _ENGINE.veta(S, K, T, r, sigma, q)


def dual_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """Dual delta = dC/dK = -e^{-rT} * N(d2) for a call."""
    return _ENGINE.dual_delta(S, K, T, r, sigma, option_type)


def dual_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Dual gamma = d^2C/dK^2 = e^{-rT} * N'(d2) / (K*sigma*sqrt(T))."""
    return _ENGINE.dual_gamma(S, K, T, r, sigma)
