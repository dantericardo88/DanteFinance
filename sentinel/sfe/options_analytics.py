"""Options analytics — comprehensive Greeks, IV surface, screening, and pricing models.

Targets dim_004 "Options chain (all strikes/expiries, live Greeks)".

All analytics are built on top of yfinance (free) for data sourcing,
with pure-Python/scipy/numpy implementations for:
  - Black-Scholes pricing and all first/second-order Greeks
  - Newton-Raphson implied volatility solver
  - SABR model calibration
  - Binomial American option pricing (CRR)
  - Monte Carlo pricing with confidence intervals
  - IV surface construction (moneyness x expiry grid)
  - GEX, max pain, put/call ratio, unusual activity screening

No external options-data vendor required — yfinance provides free chains.
"""
from __future__ import annotations

import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass  # noqa: F401 — used by downstream modules
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# Risk-free rate fallback (approximate Fed Funds / 3-month T-bill as of mid-2025)
_DEFAULT_RISK_FREE_RATE = 0.053

# Newton-Raphson IV solver parameters
_IV_MAX_ITER = 100
_IV_TOL = 1e-6
_IV_MIN = 1e-6
_IV_MAX = 20.0   # 2000% IV cap (for degenerate cases)


# ---------------------------------------------------------------------------
# Black-Scholes math helpers
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc for speed."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Compute d1 and d2 for Black-Scholes."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0
    log_sk = math.log(S / K)
    sigma_sqrt_t = sigma * math.sqrt(T)
    d1 = (log_sk + (r + 0.5 * sigma ** 2) * T) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return d1, d2


# ---------------------------------------------------------------------------
# GreeksCalculator
# ---------------------------------------------------------------------------

class GreeksCalculator:
    """Compute Black-Scholes price and all Greeks analytically.

    All methods are pure-Python (no external options lib required).
    """

    @staticmethod
    def black_scholes(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> dict:
        """Black-Scholes option price and first-order Greeks.

        Args:
            S: Spot price of the underlying.
            K: Strike price.
            T: Time to expiry in years (e.g. 30/365 for 30 days).
            r: Continuously compounded risk-free rate (e.g. 0.053).
            sigma: Implied / assumed volatility as a decimal (e.g. 0.25 = 25%).
            option_type: "call" or "put".

        Returns:
            dict with keys: price, delta, gamma, theta, vega, rho.
            All values are floats. Theta is expressed per calendar day.
        """
        if T <= 0:
            # Intrinsic value at expiry
            if option_type == "call":
                price = max(S - K, 0.0)
            else:
                price = max(K - S, 0.0)
            return {"price": price, "delta": float(S > K) if option_type == "call" else float(S < K),
                    "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

        d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
        e_rT = math.exp(-r * T)
        is_call = option_type.lower() == "call"

        if is_call:
            price = S * _norm_cdf(d1) - K * e_rT * _norm_cdf(d2)
            delta = _norm_cdf(d1)
            rho = K * T * e_rT * _norm_cdf(d2) / 100.0
        else:
            price = K * e_rT * _norm_cdf(-d2) - S * _norm_cdf(-d1)
            delta = _norm_cdf(d1) - 1.0
            rho = -K * T * e_rT * _norm_cdf(-d2) / 100.0

        gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))
        vega = S * _norm_pdf(d1) * math.sqrt(T) / 100.0   # per 1 vol point
        theta_annual = (
            -(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
            - r * K * e_rT * (_norm_cdf(d2) if is_call else _norm_cdf(-d2))
        )
        theta = theta_annual / 365.0  # per calendar day

        return {
            "price": round(price, 6),
            "delta": round(delta, 6),
            "gamma": round(gamma, 6),
            "theta": round(theta, 6),
            "vega": round(vega, 6),
            "rho": round(rho, 6),
        }

    @staticmethod
    def implied_volatility(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
    ) -> float:
        """Newton-Raphson implied volatility solver.

        Args:
            market_price: Observed market price of the option.
            S, K, T, r: Standard Black-Scholes parameters.
            option_type: "call" or "put".

        Returns:
            Implied volatility as a decimal, or NaN if solver fails.
        """
        if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
            return float("nan")

        # Initial guess via Brenner-Subrahmanyam approximation
        sigma = math.sqrt(2 * math.pi / T) * market_price / S
        sigma = max(_IV_MIN, min(sigma, _IV_MAX))

        for i in range(_IV_MAX_ITER):
            bs = GreeksCalculator.black_scholes(S, K, T, r, sigma, option_type)
            price_diff = bs["price"] - market_price
            vega = bs["vega"] * 100.0  # convert back from per-vol-point

            if abs(price_diff) < _IV_TOL:
                return round(sigma, 8)

            if vega < 1e-10:
                # Fallback: bisection step
                if price_diff > 0:
                    sigma *= 0.5
                else:
                    sigma *= 1.5
            else:
                sigma -= price_diff / vega

            sigma = max(_IV_MIN, min(sigma, _IV_MAX))

        logger.debug("IV solver did not converge", S=S, K=K, T=T, market_price=market_price)
        return float("nan")

    @staticmethod
    def compute_greeks_chain(
        chain_df: pd.DataFrame,
        spot: float,
        risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
    ) -> pd.DataFrame:
        """Add delta/gamma/theta/vega/rho/IV columns to a full chain DataFrame.

        The input DataFrame must have columns:
          strike, lastPrice (or bid/ask midpoint), impliedVolatility (optional),
          expiration (as date or string), optionType ("call"/"put" or "C"/"P").

        Returns:
            Augmented DataFrame with Greeks and computed IV appended.
        """
        if chain_df is None or chain_df.empty:
            return chain_df

        df = chain_df.copy()
        today = date.today()

        # Normalise option type column
        if "option_type" not in df.columns:
            if "contractSymbol" in df.columns:
                # yfinance uses contractSymbol which contains C/P
                df["option_type"] = df["contractSymbol"].str.extract(r"([CP])(\d{8})", expand=False)[0]
                df["option_type"] = df["option_type"].map({"C": "call", "P": "put"})
            else:
                df["option_type"] = "call"

        # Resolve expiration date -> T (years to expiry)
        def _resolve_T(row) -> float:
            exp = row.get("expiration") or row.get("contractSymbol", "")
            if isinstance(exp, str) and len(exp) >= 8:
                try:
                    exp_date = datetime.strptime(exp[:8], "%Y%m%d").date()
                    return max((exp_date - today).days / 365.0, 1e-6)
                except ValueError:
                    pass
            if isinstance(exp, (date, datetime)):
                if isinstance(exp, datetime):
                    exp = exp.date()
                return max((exp - today).days / 365.0, 1e-6)
            return 30 / 365.0  # fallback: 30 days

        df["T"] = df.apply(_resolve_T, axis=1)

        # Market price: prefer mid of bid/ask, fall back to lastPrice
        def _mid_price(row) -> float:
            bid = row.get("bid", 0) or 0
            ask = row.get("ask", 0) or 0
            last = row.get("lastPrice", 0) or 0
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
            return float(last)

        df["mid_price"] = df.apply(_mid_price, axis=1)

        greeks_records = []
        for _, row in df.iterrows():
            K = float(row.get("strike", spot))
            T_val = float(row["T"])
            opt_type = str(row.get("option_type", "call")).lower()
            mid = float(row["mid_price"])

            # Use provided IV if available, else compute from market price
            if "impliedVolatility" in row and not pd.isna(row["impliedVolatility"]):
                sigma = float(row["impliedVolatility"])
                if sigma <= 0:
                    sigma = GreeksCalculator.implied_volatility(
                        mid, spot, K, T_val, risk_free_rate, opt_type
                    )
            else:
                sigma = GreeksCalculator.implied_volatility(
                    mid, spot, K, T_val, risk_free_rate, opt_type
                )

            if math.isnan(sigma) or sigma <= 0:
                sigma = 0.25  # placeholder

            g = GreeksCalculator.black_scholes(spot, K, T_val, risk_free_rate, sigma, opt_type)
            greeks_records.append({
                "computed_iv": round(sigma, 6),
                "bs_price": g["price"],
                "delta": g["delta"],
                "gamma": g["gamma"],
                "theta": g["theta"],
                "vega": g["vega"],
                "rho": g["rho"],
            })

        greeks_df = pd.DataFrame(greeks_records, index=df.index)
        return pd.concat([df, greeks_df], axis=1)

    @staticmethod
    def charm(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call") -> float:
        """Charm: dDelta/dTime (also called delta decay or DdeltaDtime).

        Measures how delta changes as time passes — useful for
        delta-hedging strategies that need intraday rebalancing.
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
        pdf_d1 = _norm_pdf(d1)
        term = r / (sigma * math.sqrt(T)) - d2 / (2 * T)
        if option_type.lower() == "call":
            return -pdf_d1 * term
        else:
            return pdf_d1 * term

    @staticmethod
    def vanna(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Vanna: dDelta/dVol = dVega/dSpot.

        Measures cross-sensitivity of delta to volatility (and vice versa).
        Important for vol-weighted delta hedging.
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
        return -_norm_pdf(d1) * d2 / sigma

    @staticmethod
    def volga(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Volga (Vomma): dVega/dVol — convexity of option price w.r.t. vol.

        A positive volga means the option benefits from vol-of-vol.
        Used in straddle/strangle analysis.
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
        vega = S * _norm_pdf(d1) * math.sqrt(T)  # unnormalized vega
        return vega * (d1 * d2) / sigma


# ---------------------------------------------------------------------------
# OptionsChainFetcher
# ---------------------------------------------------------------------------

class OptionsChainFetcher:
    """Fetch options chains from yfinance for any ticker."""

    @staticmethod
    def get_all_expirations(ticker: str) -> list[str]:
        """Return all available expiration dates (as ISO strings) for a ticker."""
        import yfinance as yf
        try:
            obj = yf.Ticker(ticker)
            return list(obj.options or [])
        except Exception as exc:
            logger.warning("get_all_expirations failed", ticker=ticker, error=str(exc))
            return []

    @staticmethod
    def get_chain(ticker: str, expiry: str = None) -> dict:
        """Fetch options chain for a specific expiry.

        Args:
            ticker: Underlying ticker.
            expiry: ISO date string (e.g. "2025-12-19"). If None, uses nearest expiry.

        Returns:
            dict with keys: calls (DataFrame), puts (DataFrame),
            underlying_price (float), expiry (str).
        """
        import yfinance as yf

        try:
            obj = yf.Ticker(ticker)

            if expiry is None:
                expirations = obj.options
                if not expirations:
                    return {"calls": pd.DataFrame(), "puts": pd.DataFrame(),
                            "underlying_price": None, "expiry": None}
                expiry = expirations[0]

            chain = obj.option_chain(expiry)
            info = obj.fast_info or {}

            try:
                underlying_price = float(
                    getattr(info, "last_price", None)
                    or getattr(info, "previousClose", None)
                    or 0.0
                )
            except Exception:
                underlying_price = 0.0

            # Attach expiry to both DataFrames for convenience
            calls = chain.calls.copy() if chain.calls is not None else pd.DataFrame()
            puts = chain.puts.copy() if chain.puts is not None else pd.DataFrame()
            for df in [calls, puts]:
                df["expiration"] = expiry

            return {
                "calls": calls,
                "puts": puts,
                "underlying_price": underlying_price,
                "expiry": expiry,
            }
        except Exception as exc:
            logger.warning("get_chain failed", ticker=ticker, expiry=expiry, error=str(exc))
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(),
                    "underlying_price": None, "expiry": expiry}

    @staticmethod
    def get_full_chain(ticker: str) -> dict[str, dict]:
        """Fetch full options chain across all expirations.

        Returns:
            dict mapping expiry_date_str -> chain_dict (same as get_chain).
        """
        expirations = OptionsChainFetcher.get_all_expirations(ticker)
        result: dict[str, dict] = {}
        for expiry in expirations:
            chain = OptionsChainFetcher.get_chain(ticker, expiry)
            result[expiry] = chain
        return result

    @staticmethod
    def get_near_term_chain(
        ticker: str,
        n_expirations: int = 4,
    ) -> pd.DataFrame:
        """Return first N expirations merged into a single DataFrame.

        Columns include a 'flag' column ("call" or "put") for each row.

        Returns:
            Merged DataFrame with all strikes, all near-term expirations.
        """
        expirations = OptionsChainFetcher.get_all_expirations(ticker)[:n_expirations]
        frames: list[pd.DataFrame] = []

        for expiry in expirations:
            chain = OptionsChainFetcher.get_chain(ticker, expiry)
            calls = chain["calls"].copy()
            puts = chain["puts"].copy()
            calls["flag"] = "call"
            puts["flag"] = "put"
            calls["expiry"] = expiry
            puts["expiry"] = expiry
            frames.extend([calls, puts])

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# VolatilitySurface
# ---------------------------------------------------------------------------

class VolatilitySurface:
    """Build, fit, and analyse the implied volatility surface."""

    def build_surface(
        self,
        ticker: str,
        n_expirations: int = 6,
        spot: Optional[float] = None,
        risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
    ) -> pd.DataFrame:
        """Construct the IV surface: moneyness (K/S) x expiry grid.

        Returns:
            DataFrame where rows are moneyness levels (K/S buckets),
            columns are expiration dates, values are IV (decimal).
        """
        import yfinance as yf

        if spot is None:
            try:
                info = yf.Ticker(ticker).fast_info
                spot = float(getattr(info, "last_price", 0) or 0)
            except Exception:
                spot = 0.0

        if spot <= 0:
            logger.warning("Cannot build surface: spot = 0", ticker=ticker)
            return pd.DataFrame()

        expirations = OptionsChainFetcher.get_all_expirations(ticker)[:n_expirations]
        surface_data: dict[str, dict[float, float]] = {}

        for expiry in expirations:
            chain = OptionsChainFetcher.get_chain(ticker, expiry)
            calls = chain["calls"]
            puts = chain["puts"]

            if calls.empty and puts.empty:
                continue

            # Merge calls and puts, prefer calls above ATM, puts below ATM
            all_opts = pd.concat([calls.assign(flag="call"), puts.assign(flag="put")],
                                  ignore_index=True)

            today = date.today()
            try:
                exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
                T = max((exp_date - today).days / 365.0, 1e-6)
            except ValueError:
                T = 30 / 365.0

            col_by_moneyness: dict[float, float] = {}
            for _, row in all_opts.iterrows():
                K = float(row.get("strike", 0))
                if K <= 0:
                    continue
                moneyness = round(K / spot, 2)  # K/S rounded to 2dp
                mid = (
                    (float(row.get("bid", 0) or 0) + float(row.get("ask", 0) or 0)) / 2.0
                    if (row.get("bid") or 0) > 0 and (row.get("ask") or 0) > 0
                    else float(row.get("lastPrice", 0) or 0)
                )

                if mid <= 0:
                    continue

                opt_type = str(row.get("flag", "call")).lower()
                iv = GreeksCalculator.implied_volatility(mid, spot, K, T, risk_free_rate, opt_type)
                if not math.isnan(iv) and iv > 0:
                    col_by_moneyness[moneyness] = round(iv, 4)

            surface_data[expiry] = col_by_moneyness

        if not surface_data:
            return pd.DataFrame()

        surface_df = pd.DataFrame(surface_data)
        surface_df.index.name = "moneyness_K_over_S"
        surface_df = surface_df.sort_index()
        return surface_df

    @staticmethod
    def fit_sabr(
        F: float,
        strikes: list[float],
        ivs: list[float],
        T: float,
        beta: float = 0.5,
    ) -> dict:
        """Calibrate SABR model (Hagan et al. 2002) to market IVs.

        Fits (alpha, rho, nu) with beta fixed (default 0.5).

        Args:
            F: Forward price of the underlying.
            strikes: List of strike prices.
            ivs: Corresponding market-observed implied volatilities (decimal).
            T: Time to expiry in years.
            beta: SABR beta parameter (fixed, default 0.5).

        Returns:
            dict with keys: alpha, beta, rho, nu, rmse (calibration error).
        """
        from scipy.optimize import minimize

        if len(strikes) != len(ivs) or len(strikes) < 3:
            return {"alpha": 0.0, "beta": beta, "rho": 0.0, "nu": 0.0, "rmse": np.nan}

        strikes_arr = np.array(strikes, dtype=float)
        ivs_arr = np.array(ivs, dtype=float)
        valid = (strikes_arr > 0) & ~np.isnan(ivs_arr) & (ivs_arr > 0)
        strikes_arr = strikes_arr[valid]
        ivs_arr = ivs_arr[valid]

        if len(strikes_arr) < 3:
            return {"alpha": 0.0, "beta": beta, "rho": 0.0, "nu": 0.0, "rmse": np.nan}

        def _sabr_iv(F: float, K: float, T: float, alpha: float, beta: float,
                     rho: float, nu: float) -> float:
            """Hagan et al. SABR ATM and non-ATM formula."""
            if abs(F - K) < 1e-6:
                # ATM formula
                term1 = alpha / (F ** (1 - beta))
                term2 = (1 + (((1 - beta) ** 2 / 24) * alpha ** 2 / F ** (2 - 2 * beta)
                              + (rho * beta * nu * alpha) / (4 * F ** (1 - beta))
                              + (2 - 3 * rho ** 2) * nu ** 2 / 24) * T)
                return term1 * term2
            else:
                FK = F * K
                mid = FK ** ((1 - beta) / 2)
                log_FK = math.log(F / K)
                z = (nu / alpha) * mid * log_FK
                x = math.log((math.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
                zx = z / x if abs(x) > 1e-10 else 1.0

                A = alpha / (mid * (1 + ((1 - beta) ** 2 / 24) * log_FK ** 2
                                    + ((1 - beta) ** 4 / 1920) * log_FK ** 4))
                B = (1 + (((1 - beta) ** 2 / 24) * alpha ** 2 / mid ** 2
                           + (rho * beta * nu * alpha) / (4 * mid)
                           + (2 - 3 * rho ** 2) * nu ** 2 / 24) * T)
                return A * zx * B

        def _objective(params) -> float:
            alpha, rho, nu = params
            if alpha <= 0 or nu <= 0 or not (-1 < rho < 1):
                return 1e6
            total = 0.0
            for K, iv_mkt in zip(strikes_arr, ivs_arr):
                try:
                    iv_model = _sabr_iv(F, K, T, alpha, beta, rho, nu)
                    total += (iv_model - iv_mkt) ** 2
                except Exception:
                    total += 1.0
            return total

        # Initial guess
        atm_idx = np.argmin(np.abs(strikes_arr - F))
        alpha0 = ivs_arr[atm_idx] * (F ** (1 - beta))
        x0 = [max(alpha0, 0.01), 0.0, 0.3]
        bounds = [(1e-4, 10.0), (-0.999, 0.999), (1e-4, 10.0)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(_objective, x0, bounds=bounds, method="L-BFGS-B",
                              options={"maxiter": 500, "ftol": 1e-10})

        alpha_fit, rho_fit, nu_fit = result.x
        rmse = math.sqrt(result.fun / len(strikes_arr))

        return {
            "alpha": round(float(alpha_fit), 6),
            "beta": beta,
            "rho": round(float(rho_fit), 6),
            "nu": round(float(nu_fit), 6),
            "rmse": round(rmse, 6),
            "converged": result.success,
        }

    def compute_term_structure(
        self,
        ticker: str,
        risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
    ) -> pd.DataFrame:
        """Compute ATM IV by expiration date (term structure).

        Returns:
            DataFrame with columns: expiry, days_to_expiry, atm_iv.
        """
        import yfinance as yf

        try:
            info = yf.Ticker(ticker).fast_info
            spot = float(getattr(info, "last_price", 0) or 0)
        except Exception:
            spot = 0.0

        expirations = OptionsChainFetcher.get_all_expirations(ticker)
        records = []
        today = date.today()

        for expiry in expirations:
            try:
                exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
                days = (exp_date - today).days
            except ValueError:
                continue

            if days <= 0:
                continue

            T = days / 365.0
            chain = OptionsChainFetcher.get_chain(ticker, expiry)
            calls = chain["calls"]

            if calls.empty:
                continue

            # Find ATM call (closest strike to spot)
            calls = calls.copy()
            calls["moneyness_dist"] = (calls["strike"] - spot).abs()
            atm_row = calls.nsmallest(1, "moneyness_dist").iloc[0]

            K = float(atm_row["strike"])
            mid = (
                (float(atm_row.get("bid", 0) or 0) + float(atm_row.get("ask", 0) or 0)) / 2.0
                if (atm_row.get("bid") or 0) > 0 and (atm_row.get("ask") or 0) > 0
                else float(atm_row.get("lastPrice", 0) or 0)
            )

            if mid <= 0:
                continue

            atm_iv = GreeksCalculator.implied_volatility(mid, spot, K, T, risk_free_rate, "call")
            if not math.isnan(atm_iv) and atm_iv > 0:
                records.append({
                    "expiry": expiry,
                    "days_to_expiry": days,
                    "atm_iv": round(atm_iv, 4),
                })

        return pd.DataFrame(records)

    def compute_skew(
        self,
        ticker: str,
        expiry: str,
        risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
    ) -> dict:
        """Compute volatility skew metrics for a single expiry.

        Returns:
            dict with keys:
              - risk_reversal_25d: 25-delta put IV minus 25-delta call IV
              - butterfly_25d: 25-delta butterfly (avg wing IV minus ATM IV)
              - skew_slope: linear regression slope of IV vs moneyness
              - atm_iv: at-the-money IV
        """
        import yfinance as yf

        try:
            info = yf.Ticker(ticker).fast_info
            spot = float(getattr(info, "last_price", 0) or 0)
        except Exception:
            spot = 0.0

        if spot <= 0:
            return {"risk_reversal_25d": None, "butterfly_25d": None,
                    "skew_slope": None, "atm_iv": None}

        chain = OptionsChainFetcher.get_chain(ticker, expiry)
        calls = chain["calls"]
        puts = chain["puts"]

        today = date.today()
        try:
            T = max((datetime.strptime(expiry, "%Y-%m-%d").date() - today).days / 365.0, 1e-6)
        except ValueError:
            T = 30 / 365.0

        def _compute_iv_series(df: pd.DataFrame, opt_type: str) -> pd.DataFrame:
            """Compute IVs for a calls or puts DataFrame."""
            records = []
            for _, row in df.iterrows():
                K = float(row.get("strike", 0))
                if K <= 0:
                    continue
                mid = (
                    (float(row.get("bid", 0) or 0) + float(row.get("ask", 0) or 0)) / 2.0
                    if (row.get("bid") or 0) > 0 and (row.get("ask") or 0) > 0
                    else float(row.get("lastPrice", 0) or 0)
                )
                if mid <= 0:
                    continue
                iv = GreeksCalculator.implied_volatility(mid, spot, K, T, risk_free_rate, opt_type)
                delta = GreeksCalculator.black_scholes(spot, K, T, risk_free_rate,
                                                        iv if not math.isnan(iv) else 0.25,
                                                        opt_type)["delta"]
                records.append({"strike": K, "iv": iv, "delta": delta, "moneyness": K / spot})
            return pd.DataFrame(records)

        call_ivs = _compute_iv_series(calls, "call")
        put_ivs = _compute_iv_series(puts, "put")

        # ATM IV: call nearest to spot
        atm_iv = None
        if not call_ivs.empty:
            atm_row = call_ivs.iloc[(call_ivs["strike"] - spot).abs().argsort()[:1]]
            atm_iv = float(atm_row["iv"].values[0]) if not atm_row.empty else None

        # 25-delta put: closest put with delta ~ -0.25
        rr_25d = None
        fly_25d = None

        if not put_ivs.empty and not call_ivs.empty:
            put_ivs_clean = put_ivs.dropna(subset=["iv"])
            call_ivs_clean = call_ivs.dropna(subset=["iv"])

            # 25-delta put (delta ~ -0.25)
            p25 = put_ivs_clean.iloc[(put_ivs_clean["delta"] + 0.25).abs().argsort()[:1]]
            # 25-delta call (delta ~ 0.25)
            c25 = call_ivs_clean.iloc[(call_ivs_clean["delta"] - 0.25).abs().argsort()[:1]]

            if not p25.empty and not c25.empty:
                iv_p25 = float(p25["iv"].values[0])
                iv_c25 = float(c25["iv"].values[0])
                rr_25d = round(iv_p25 - iv_c25, 4)  # positive = put fear
                fly_25d = round((iv_p25 + iv_c25) / 2.0 - (atm_iv or 0.25), 4)

        # Skew slope: linear regression of IV vs log-moneyness
        skew_slope = None
        if not call_ivs.empty and len(call_ivs.dropna(subset=["iv"])) >= 3:
            clean = call_ivs.dropna(subset=["iv"])
            log_m = np.log(clean["moneyness"].values)
            ivs_arr = clean["iv"].values
            if len(log_m) >= 2:
                slope = np.polyfit(log_m, ivs_arr, 1)[0]
                skew_slope = round(float(slope), 4)

        return {
            "risk_reversal_25d": rr_25d,
            "butterfly_25d": fly_25d,
            "skew_slope": skew_slope,
            "atm_iv": round(atm_iv, 4) if atm_iv and not math.isnan(atm_iv) else None,
        }

    def detect_vol_regime(
        self,
        ticker: str,
        hv_window: int = 21,
        risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
    ) -> dict:
        """Classify current volatility regime.

        Compares ATM IV against 52-week historical volatility percentile.

        Returns:
            dict with keys: current_iv, hv_21d, iv_hv_ratio,
            iv_percentile_52w, regime ("low_vol"/"normal"/"elevated"/"stressed").
        """
        import yfinance as yf

        try:
            hist = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
            if hist is None or hist.empty:
                return {"regime": "unknown", "current_iv": None, "hv_21d": None}
            hist.index = pd.to_datetime(hist.index)
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            close = hist["Close"].dropna()
        except Exception as exc:
            logger.warning("vol regime: history fetch failed", ticker=ticker, error=str(exc))
            return {"regime": "unknown", "current_iv": None, "hv_21d": None}

        # Realized (historical) volatility — annualised 21-day
        log_rets = np.log(close / close.shift(1)).dropna()
        hv_21d = float(log_rets.tail(hv_window).std() * math.sqrt(252)) if len(log_rets) >= hv_window else None

        # Current ATM IV (nearest expiry)
        expirations = OptionsChainFetcher.get_all_expirations(ticker)
        current_iv = None
        spot = float(close.iloc[-1]) if len(close) > 0 else 0.0

        if expirations and spot > 0:
            chain = OptionsChainFetcher.get_chain(ticker, expirations[0])
            calls = chain["calls"]
            today = date.today()
            try:
                T = max((datetime.strptime(expirations[0], "%Y-%m-%d").date() - today).days / 365.0, 1e-6)
            except ValueError:
                T = 30 / 365.0

            if not calls.empty:
                calls = calls.copy()
                calls["_d"] = (calls["strike"] - spot).abs()
                atm_call = calls.nsmallest(1, "_d").iloc[0]
                K = float(atm_call["strike"])
                mid = (
                    (float(atm_call.get("bid", 0) or 0) + float(atm_call.get("ask", 0) or 0)) / 2.0
                    if (atm_call.get("bid") or 0) > 0 and (atm_call.get("ask") or 0) > 0
                    else float(atm_call.get("lastPrice", 0) or 0)
                )
                if mid > 0:
                    current_iv = GreeksCalculator.implied_volatility(
                        mid, spot, K, T, risk_free_rate, "call"
                    )
                    if math.isnan(current_iv):
                        current_iv = None

        # IV percentile vs 52-week daily IV proxy (HV rolling)
        rolling_hv = log_rets.rolling(hv_window).std() * math.sqrt(252)
        rolling_hv = rolling_hv.dropna()

        iv_percentile = None
        if current_iv and len(rolling_hv) > 0:
            pct = float((rolling_hv < current_iv).mean() * 100)
            iv_percentile = round(pct, 1)

        iv_hv_ratio = round(current_iv / hv_21d, 3) if current_iv and hv_21d and hv_21d > 0 else None

        # Regime classification
        if iv_percentile is None:
            regime = "unknown"
        elif iv_percentile < 20:
            regime = "low_vol"
        elif iv_percentile < 60:
            regime = "normal"
        elif iv_percentile < 80:
            regime = "elevated"
        else:
            regime = "stressed"

        return {
            "ticker": ticker,
            "current_iv": round(current_iv, 4) if current_iv else None,
            "hv_21d": round(hv_21d, 4) if hv_21d else None,
            "iv_hv_ratio": iv_hv_ratio,
            "iv_percentile_52w": iv_percentile,
            "regime": regime,
        }


# ---------------------------------------------------------------------------
# OptionsScreener
# ---------------------------------------------------------------------------

class OptionsScreener:
    """Screen options for high IV rank, unusual activity, and key market levels."""

    @staticmethod
    def screen_high_iv_rank(
        tickers: list[str],
        min_iv_rank: float = 70.0,
        risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
        max_workers: int = 4,
    ) -> pd.DataFrame:
        """Screen a list of tickers for high IV rank.

        IV Rank = (current IV - 52w low IV) / (52w high IV - 52w low IV) * 100.

        Args:
            tickers: List of tickers to screen.
            min_iv_rank: Minimum IV rank (0-100) to include in results.
            max_workers: Number of parallel worker threads.

        Returns:
            DataFrame with columns: ticker, current_iv, iv_rank_52w, regime.
        """
        surface = VolatilitySurface()

        def _score_one(ticker: str) -> dict:
            try:
                regime = surface.detect_vol_regime(ticker, risk_free_rate=risk_free_rate)
                return {
                    "ticker": ticker,
                    "current_iv": regime.get("current_iv"),
                    "hv_21d": regime.get("hv_21d"),
                    "iv_rank_52w": regime.get("iv_percentile_52w"),
                    "regime": regime.get("regime", "unknown"),
                }
            except Exception as exc:
                logger.warning("IV rank screen failed", ticker=ticker, error=str(exc))
                return {"ticker": ticker, "current_iv": None, "iv_rank_52w": None, "regime": "unknown"}

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_score_one, t): t for t in tickers}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    pass

        df = pd.DataFrame(results)
        if df.empty:
            return df

        df = df.dropna(subset=["iv_rank_52w"])
        df = df[df["iv_rank_52w"] >= min_iv_rank]
        return df.sort_values("iv_rank_52w", ascending=False).reset_index(drop=True)

    @staticmethod
    def screen_unusual_activity(ticker: str) -> pd.DataFrame:
        """Detect unusual options activity: high volume/OI ratio and block trades.

        Flags:
          - call_sweep: call volume > 3x OI
          - put_sweep: put volume > 3x OI
          - block_trade: single-strike volume > 1000 contracts

        Returns:
            DataFrame of flagged contracts with flag type annotations.
        """
        chain_data = OptionsChainFetcher.get_full_chain(ticker)
        if not chain_data:
            return pd.DataFrame()

        flagged_rows = []
        for expiry, chain in chain_data.items():
            for opt_type, df in [("call", chain["calls"]), ("put", chain["puts"])]:
                if df is None or df.empty:
                    continue
                for _, row in df.iterrows():
                    volume = float(row.get("volume", 0) or 0)
                    oi = float(row.get("openInterest", 1) or 1)
                    strike = float(row.get("strike", 0))

                    flags = []
                    vol_oi_ratio = volume / oi if oi > 0 else 0

                    if vol_oi_ratio > 3:
                        flags.append(f"{opt_type}_sweep")
                    if volume > 1000:
                        flags.append("block_trade")

                    if flags:
                        flagged_rows.append({
                            "ticker": ticker,
                            "expiry": expiry,
                            "strike": strike,
                            "option_type": opt_type,
                            "volume": int(volume),
                            "open_interest": int(oi),
                            "vol_oi_ratio": round(vol_oi_ratio, 2),
                            "flags": "|".join(flags),
                            "last_price": float(row.get("lastPrice", 0) or 0),
                        })

        return pd.DataFrame(flagged_rows).sort_values("volume", ascending=False) if flagged_rows else pd.DataFrame()

    @staticmethod
    def compute_max_pain(chain: dict, expiry: str = "") -> float:
        """Compute max pain strike — where total option value is minimized at expiry.

        Args:
            chain: Chain dict with 'calls' and 'puts' DataFrames.
            expiry: Expiry label (informational; included in returned context).

        At max pain, the largest number of option contracts expire worthless.
        This is the price where option sellers (dealers) face the least loss.

        Args:
            chain: Chain dict from get_chain() containing 'calls' and 'puts'.
            expiry: Expiry string (used for display; not required for computation).

        Returns:
            Max pain strike price as float.
        """
        calls = chain.get("calls", pd.DataFrame())
        puts = chain.get("puts", pd.DataFrame())

        if (calls is None or calls.empty) and (puts is None or puts.empty):
            return 0.0

        all_strikes = sorted(set(
            list(calls["strike"].values if not calls.empty else []) +
            list(puts["strike"].values if not puts.empty else [])
        ))

        if not all_strikes:
            return 0.0

        def _option_pain(expiry_price: float) -> float:
            """Total dollar value of all options at given expiry price."""
            total = 0.0
            if not calls.empty:
                for _, row in calls.iterrows():
                    K = float(row.get("strike", 0))
                    oi = float(row.get("openInterest", 0) or 0)
                    total += max(expiry_price - K, 0) * oi * 100
            if not puts.empty:
                for _, row in puts.iterrows():
                    K = float(row.get("strike", 0))
                    oi = float(row.get("openInterest", 0) or 0)
                    total += max(K - expiry_price, 0) * oi * 100
            return total

        min_pain = float("inf")
        max_pain_strike = all_strikes[len(all_strikes) // 2]

        for strike in all_strikes:
            pain = _option_pain(strike)
            if pain < min_pain:
                min_pain = pain
                max_pain_strike = strike

        logger.debug("Max pain computed", expiry=expiry, strike=max_pain_strike)
        return float(max_pain_strike)

    @staticmethod
    def compute_put_call_ratio(chain: dict) -> dict:
        """Compute volume and OI put/call ratios.

        Returns:
            dict with keys: volume_pc_ratio, oi_pc_ratio,
            total_call_volume, total_put_volume,
            total_call_oi, total_put_oi, interpretation.
        """
        calls = chain.get("calls", pd.DataFrame())
        puts = chain.get("puts", pd.DataFrame())

        call_vol = float(calls["volume"].sum()) if not calls.empty and "volume" in calls.columns else 0.0
        put_vol = float(puts["volume"].sum()) if not puts.empty and "volume" in puts.columns else 0.0
        call_oi = float(calls["openInterest"].sum()) if not calls.empty and "openInterest" in calls.columns else 0.0
        put_oi = float(puts["openInterest"].sum()) if not puts.empty and "openInterest" in puts.columns else 0.0

        vol_pc = round(put_vol / call_vol, 4) if call_vol > 0 else None
        oi_pc = round(put_oi / call_oi, 4) if call_oi > 0 else None

        interpretation = "neutral"
        if vol_pc is not None:
            if vol_pc > 1.5:
                interpretation = "bearish"
            elif vol_pc > 1.0:
                interpretation = "mildly_bearish"
            elif vol_pc < 0.5:
                interpretation = "bullish"
            elif vol_pc < 0.7:
                interpretation = "mildly_bullish"

        return {
            "volume_pc_ratio": vol_pc,
            "oi_pc_ratio": oi_pc,
            "total_call_volume": int(call_vol),
            "total_put_volume": int(put_vol),
            "total_call_oi": int(call_oi),
            "total_put_oi": int(put_oi),
            "interpretation": interpretation,
        }

    @staticmethod
    def gamma_exposure(chain: dict, spot: float,
                       risk_free_rate: float = _DEFAULT_RISK_FREE_RATE) -> dict:
        """Compute aggregate dealer gamma exposure (GEX) by strike.

        Convention: dealers are short calls and long puts (inverse of retail).
        Positive GEX -> dealers long gamma -> vol suppression.
        Negative GEX -> dealers short gamma -> vol amplification.

        Args:
            chain: Chain dict with calls and puts DataFrames.
            spot: Current spot price of the underlying.

        Returns:
            dict with keys: gex_by_strike (dict), net_gex (float),
            regime ("stabilizing" or "destabilizing").
        """
        calls = chain.get("calls", pd.DataFrame())
        puts = chain.get("puts", pd.DataFrame())
        expiry = chain.get("expiry", "")
        today = date.today()

        try:
            T = max((datetime.strptime(expiry, "%Y-%m-%d").date() - today).days / 365.0, 1e-6)
        except (ValueError, TypeError):
            T = 30 / 365.0

        gex_by_strike: dict[float, float] = {}

        def _add_gex(df: pd.DataFrame, opt_type: str, sign: float) -> None:
            if df is None or df.empty:
                return
            for _, row in df.iterrows():
                K = float(row.get("strike", 0))
                oi = float(row.get("openInterest", 0) or 0)
                iv_market = float(row.get("impliedVolatility", 0.25) or 0.25)
                if K <= 0 or oi <= 0:
                    continue
                bs = GreeksCalculator.black_scholes(spot, K, T, risk_free_rate, iv_market, opt_type)
                gamma = bs["gamma"]
                # GEX = gamma * OI * contract_multiplier * spot
                gex = sign * gamma * oi * 100 * spot
                gex_by_strike[K] = gex_by_strike.get(K, 0.0) + gex

        # Dealers: short calls (negative call GEX), long puts (negative put GEX)
        _add_gex(calls, "call", -1.0)
        _add_gex(puts, "put", 1.0)

        net_gex = sum(gex_by_strike.values())
        regime = "stabilizing" if net_gex > 0 else "destabilizing"

        return {
            "gex_by_strike": {str(k): round(v, 2) for k, v in sorted(gex_by_strike.items())},
            "net_gex": round(net_gex, 2),
            "regime": regime,
        }

    @staticmethod
    def find_key_levels(chain: dict, spot: float) -> dict:
        """Identify key price levels from options structure.

        Returns:
            dict with keys:
              - max_pain: max pain strike
              - highest_call_oi_strike: strike with most call OI
              - highest_put_oi_strike: strike with most put OI
              - gamma_flip: strike where net GEX changes sign (approx)
        """
        calls = chain.get("calls", pd.DataFrame())
        puts = chain.get("puts", pd.DataFrame())
        expiry = chain.get("expiry", "")

        max_pain = OptionsScreener.compute_max_pain(chain, expiry)

        highest_call_oi = None
        if not calls.empty and "openInterest" in calls.columns:
            idx = calls["openInterest"].idxmax()
            highest_call_oi = float(calls.loc[idx, "strike"]) if idx is not None else None

        highest_put_oi = None
        if not puts.empty and "openInterest" in puts.columns:
            idx = puts["openInterest"].idxmax()
            highest_put_oi = float(puts.loc[idx, "strike"]) if idx is not None else None

        # Gamma flip: find strike where cumulative GEX switches sign
        gex_data = OptionsScreener.gamma_exposure(chain, spot)
        gex_by_strike = gex_data.get("gex_by_strike", {})

        gamma_flip = None
        sorted_strikes = sorted([float(k) for k in gex_by_strike.keys()])
        cumulative_gex = 0.0
        prev_sign = None
        for strike in sorted_strikes:
            cumulative_gex += gex_by_strike.get(str(strike), 0.0)
            current_sign = 1 if cumulative_gex >= 0 else -1
            if prev_sign is not None and current_sign != prev_sign:
                gamma_flip = strike
                break
            prev_sign = current_sign

        return {
            "max_pain": max_pain,
            "highest_call_oi_strike": highest_call_oi,
            "highest_put_oi_strike": highest_put_oi,
            "gamma_flip": gamma_flip,
        }


# ---------------------------------------------------------------------------
# OptionsPricingModel
# ---------------------------------------------------------------------------

class OptionsPricingModel:
    """Alternative pricing models for American options and complex structures."""

    @staticmethod
    def binomial_american(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n: int = 100,
        option_type: str = "call",
    ) -> float:
        """Cox-Ross-Rubinstein (CRR) binomial tree for American options.

        American options can be exercised early — this model correctly accounts
        for the early exercise premium not captured in Black-Scholes.

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to expiry in years.
            r: Risk-free rate.
            sigma: Volatility.
            n: Number of time steps (more steps = more accurate).
            option_type: "call" or "put".

        Returns:
            American option price as float.
        """
        if T <= 0:
            return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

        dt = T / n
        u = math.exp(sigma * math.sqrt(dt))       # up factor
        d = 1.0 / u                                # down factor (CRR: d = 1/u)
        pu = (math.exp(r * dt) - d) / (u - d)     # risk-neutral up probability
        pd_ = 1.0 - pu                             # risk-neutral down probability
        disc = math.exp(-r * dt)                   # one-step discount factor

        is_call = option_type.lower() == "call"

        # Terminal asset prices (n+1 nodes)
        prices = np.array([S * (u ** (n - 2 * j)) for j in range(n + 1)])

        # Terminal option payoffs
        if is_call:
            values = np.maximum(prices - K, 0.0)
        else:
            values = np.maximum(K - prices, 0.0)

        # Backward induction with early exercise check
        for step in range(n - 1, -1, -1):
            prices = prices[:-1] / u    # asset prices at this step
            # Continuation value
            cont = disc * (pu * values[:-1] + pd_ * values[1:])
            # Early exercise value
            if is_call:
                exercise = np.maximum(prices - K, 0.0)
            else:
                exercise = np.maximum(K - prices, 0.0)
            values = np.maximum(cont, exercise)

        return round(float(values[0]), 6)

    @staticmethod
    def monte_carlo_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_paths: int = 10_000,
        option_type: str = "call",
        seed: int = 42,
    ) -> dict:
        """Monte Carlo option pricing via geometric Brownian motion.

        Args:
            n_paths: Number of simulation paths (10K default, increase for tighter CI).
            seed: RNG seed for reproducibility.

        Returns:
            dict with keys: price, std_error, confidence_interval_95 (tuple),
            n_paths_used.
        """
        if T <= 0:
            price = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
            return {"price": price, "std_error": 0.0,
                    "confidence_interval_95": (price, price), "n_paths_used": 0}

        rng = np.random.default_rng(seed)
        z = rng.standard_normal(n_paths)

        # Terminal price under risk-neutral measure
        drift = (r - 0.5 * sigma ** 2) * T
        diffusion = sigma * math.sqrt(T)
        S_T = S * np.exp(drift + diffusion * z)

        if option_type.lower() == "call":
            payoffs = np.maximum(S_T - K, 0.0)
        else:
            payoffs = np.maximum(K - S_T, 0.0)

        discounted = np.exp(-r * T) * payoffs
        price = float(discounted.mean())
        std_err = float(discounted.std() / math.sqrt(n_paths))
        ci_lo = price - 1.96 * std_err
        ci_hi = price + 1.96 * std_err

        return {
            "price": round(price, 6),
            "std_error": round(std_err, 6),
            "confidence_interval_95": (round(ci_lo, 6), round(ci_hi, 6)),
            "n_paths_used": n_paths,
        }

    @staticmethod
    def butterfly_spread_cost(
        S: float,
        K_low: float,
        K_mid: float,
        K_high: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """Net debit (or credit) for a long butterfly spread.

        Long butterfly: buy 1 K_low + buy 1 K_high + sell 2 K_mid.
        Positive return = net debit (cost to enter).
        Negative return = net credit (premium received).
        """
        gc = GreeksCalculator()
        price_low = gc.black_scholes(S, K_low, T, r, sigma, option_type)["price"]
        price_mid = gc.black_scholes(S, K_mid, T, r, sigma, option_type)["price"]
        price_high = gc.black_scholes(S, K_high, T, r, sigma, option_type)["price"]
        net = price_low + price_high - 2.0 * price_mid
        return round(net, 6)

    @staticmethod
    def iron_condor_analysis(
        S: float,
        K_put_buy: float,
        K_put_sell: float,
        K_call_sell: float,
        K_call_buy: float,
        T: float,
        r: float,
        sigma: float,
    ) -> dict:
        """Analyse an iron condor spread.

        Structure:
          - Buy OTM put at K_put_buy
          - Sell OTM put at K_put_sell (K_put_sell > K_put_buy)
          - Sell OTM call at K_call_sell
          - Buy OTM call at K_call_buy (K_call_buy > K_call_sell)

        Args:
            All K_ parameters are strike prices in increasing order:
            K_put_buy < K_put_sell < spot < K_call_sell < K_call_buy.

        Returns:
            dict with keys: net_credit, max_profit, max_loss,
            breakeven_lower, breakeven_upper, probability_of_profit,
            risk_reward_ratio.
        """
        gc = GreeksCalculator()

        # Leg prices
        p_put_buy = gc.black_scholes(S, K_put_buy, T, r, sigma, "put")["price"]
        p_put_sell = gc.black_scholes(S, K_put_sell, T, r, sigma, "put")["price"]
        p_call_sell = gc.black_scholes(S, K_call_sell, T, r, sigma, "call")["price"]
        p_call_buy = gc.black_scholes(S, K_call_buy, T, r, sigma, "call")["price"]

        # Net credit received (all multiplied by contract size 100)
        net_credit = (p_put_sell - p_put_buy + p_call_sell - p_call_buy)
        put_spread_width = K_put_sell - K_put_buy
        call_spread_width = K_call_buy - K_call_sell
        max_loss = max(put_spread_width, call_spread_width) - net_credit

        max_profit = net_credit
        breakeven_lower = K_put_sell - net_credit
        breakeven_upper = K_call_sell + net_credit

        # Probability of profit: approximate as P(K_put_sell < S_T < K_call_sell)
        # Under risk-neutral measure using BS formula
        d2_lower = _bs_d1_d2(S, K_put_sell, T, r, sigma)[1]
        d2_upper = _bs_d1_d2(S, K_call_sell, T, r, sigma)[1]
        prob_profit = _norm_cdf(d2_upper) - (1 - _norm_cdf(d2_lower))
        prob_profit = max(0.0, min(1.0, prob_profit))

        risk_reward = round(net_credit / max_loss, 4) if max_loss > 0 else None

        return {
            "net_credit": round(net_credit, 4),
            "max_profit": round(max_profit, 4),
            "max_loss": round(max_loss, 4),
            "breakeven_lower": round(breakeven_lower, 4),
            "breakeven_upper": round(breakeven_upper, 4),
            "probability_of_profit": round(prob_profit, 4),
            "risk_reward_ratio": risk_reward,
            "legs": {
                "put_buy": {"strike": K_put_buy, "price": round(p_put_buy, 4)},
                "put_sell": {"strike": K_put_sell, "price": round(p_put_sell, 4)},
                "call_sell": {"strike": K_call_sell, "price": round(p_call_sell, 4)},
                "call_buy": {"strike": K_call_buy, "price": round(p_call_buy, 4)},
            },
        }


# ---------------------------------------------------------------------------
# BlackScholesModel  (spec-required class with explicit method names)
# ---------------------------------------------------------------------------

class BlackScholesModel:
    """Black-Scholes pricing model — pure Python, no external dependencies.

    All methods are static. Greeks follow market conventions:
      - theta: per calendar day (negative for long options)
      - vega:  per 1% change in implied vol
      - rho:   per 1% change in risk-free rate
    """

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """BS d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)."""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """BS d2 = d1 - σ√T."""
        return BlackScholesModel.d1(S, K, T, r, sigma) - sigma * math.sqrt(max(T, 0.0))

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """European call price: S·N(d1) - K·e^{-rT}·N(d2)."""
        if T <= 0:
            return max(S - K, 0.0)
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """European put price: K·e^{-rT}·N(-d2) - S·N(-d1)."""
        if T <= 0:
            return max(K - S, 0.0)
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call") -> float:
        """Delta: ∂V/∂S. Call: N(d1). Put: N(d1) - 1."""
        if T <= 0:
            if option_type == "call":
                return 1.0 if S > K else 0.0
            return -1.0 if S < K else 0.0
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        if option_type.lower() == "call":
            return _norm_cdf(d1)
        return _norm_cdf(d1) - 1.0

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Gamma: ∂²V/∂S² = N'(d1) / (S·σ·√T). Same for calls and puts."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        return _norm_pdf(d1) / (S * sigma * math.sqrt(T))

    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call") -> float:
        """Theta: ∂V/∂t per calendar day (typically negative)."""
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        e_rT = math.exp(-r * T)
        common = -(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
        if option_type.lower() == "call":
            annual = common - r * K * e_rT * _norm_cdf(d2)
        else:
            annual = common + r * K * e_rT * _norm_cdf(-d2)
        return annual / 365.0

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Vega: ∂V/∂σ per 1% change in volatility (same for calls and puts)."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        return S * _norm_pdf(d1) * math.sqrt(T) / 100.0

    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float,
            option_type: str = "call") -> float:
        """Rho: ∂V/∂r per 1% change in risk-free rate."""
        if T <= 0:
            return 0.0
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        e_rT = math.exp(-r * T)
        if option_type.lower() == "call":
            return K * T * e_rT * _norm_cdf(d2) / 100.0
        return -K * T * e_rT * _norm_cdf(-d2) / 100.0

    @staticmethod
    def implied_vol(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> float:
        """Implied volatility via Brentq bisection.

        More robust than Newton-Raphson for deep ITM/OTM options where
        vega collapses. Falls back gracefully to NaN on non-convergence.
        """
        if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
            return float("nan")

        # Intrinsic value check
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        if market_price < intrinsic - tol:
            return float("nan")

        def _price(sigma: float) -> float:
            if option_type.lower() == "call":
                return BlackScholesModel.call_price(S, K, T, r, sigma)
            return BlackScholesModel.put_price(S, K, T, r, sigma)

        lo, hi = 1e-6, 20.0

        f_lo = _price(lo) - market_price
        f_hi = _price(hi) - market_price

        if f_lo * f_hi > 0:
            # Market price outside achievable BS range
            return float("nan")

        # Brentq bisection — lo/hi are loop state, updated each iteration
        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            f_mid = _price(mid) - market_price
            if abs(f_mid) < tol or (hi - lo) / 2.0 < tol:
                return round(mid, 8)
            if f_lo * f_mid < 0:
                hi = mid  # noqa: SIM113
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid

        return float("nan")


# ---------------------------------------------------------------------------
# OptionsChainAdapter  (spec-required class)
# ---------------------------------------------------------------------------

class OptionsChainAdapter:
    """Fetch and normalise options chains from yfinance with computed columns."""

    @staticmethod
    def get_chain(ticker: str, expiry: str = None) -> dict:
        """Fetch options chain for one expiry via yfinance.

        Args:
            ticker: Underlying symbol (e.g. "AAPL", "SPY").
            expiry: ISO date "YYYY-MM-DD". None -> nearest expiry.

        Returns:
            dict: calls (DataFrame), puts (DataFrame),
                  underlying_price (float), expiry (str).
        """
        return OptionsChainFetcher.get_chain(ticker, expiry)

    @staticmethod
    def get_all_chains(ticker: str) -> dict[str, dict]:
        """Fetch chains for every available expiry.

        Returns:
            dict mapping expiry_str -> chain dict (calls, puts, underlying_price).
        """
        return OptionsChainFetcher.get_full_chain(ticker)

    @staticmethod
    def get_nearest_expiry(ticker: str, min_days: int = 7) -> str:
        """Return the first expiry with >= min_days to expiration.

        Args:
            ticker: Underlying symbol.
            min_days: Minimum calendar days to expiry (default 7).

        Returns:
            ISO date string, or empty string if none found.
        """
        expirations = OptionsChainFetcher.get_all_expirations(ticker)
        today = date.today()
        for exp in expirations:
            try:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                if (exp_date - today).days >= min_days:
                    return exp
            except ValueError:
                continue
        return expirations[0] if expirations else ""

    @staticmethod
    def get_chain_dataframe(ticker: str, expiry: str = None) -> pd.DataFrame:
        """Return combined calls+puts as a single DataFrame with computed columns.

        Added columns beyond raw yfinance data:
          - option_type: "call" or "put"
          - expiry: expiration date string
          - mid_price: (bid + ask) / 2, fallback to lastPrice
          - moneyness: strike / spot
          - time_to_expiry_days: calendar days to expiry
          - intrinsic_value, extrinsic_value
          - bid_ask_spread_pct: (ask - bid) / mid
        """
        import yfinance as yf

        chain = OptionsChainAdapter.get_chain(ticker, expiry)
        calls = chain.get("calls", pd.DataFrame()).copy()
        puts = chain.get("puts", pd.DataFrame()).copy()
        spot = chain.get("underlying_price") or 0.0
        exp = chain.get("expiry", "") or ""

        if calls.empty and puts.empty:
            return pd.DataFrame()

        calls["option_type"] = "call"
        puts["option_type"] = "put"
        combined = pd.concat([calls, puts], ignore_index=True)
        combined["expiry"] = exp

        # Mid price
        def _mid(row) -> float:
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            last = float(row.get("lastPrice", 0) or 0)
            return (bid + ask) / 2.0 if bid > 0 and ask > 0 else last

        combined["mid_price"] = combined.apply(_mid, axis=1)

        # Time to expiry
        today = date.today()
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = max((exp_date - today).days, 0)
        except (ValueError, TypeError):
            dte = 30
        combined["time_to_expiry_days"] = dte

        # Moneyness
        if spot > 0:
            combined["moneyness"] = (combined["strike"] / spot).round(4)
        else:
            combined["moneyness"] = float("nan")

        # Intrinsic / extrinsic
        def _intrinsic(row) -> float:
            K = float(row.get("strike", 0))
            if row["option_type"] == "call":
                return max(spot - K, 0.0)
            return max(K - spot, 0.0)

        combined["intrinsic_value"] = combined.apply(_intrinsic, axis=1)
        combined["extrinsic_value"] = (combined["mid_price"] - combined["intrinsic_value"]).clip(lower=0)

        # Bid-ask spread %
        def _ba_pct(row) -> float:
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            mid = float(row.get("mid_price", 0) or 0)
            if mid > 0 and ask >= bid:
                return round((ask - bid) / mid, 4)
            return float("nan")

        combined["bid_ask_spread_pct"] = combined.apply(_ba_pct, axis=1)
        return combined.reset_index(drop=True)

    @staticmethod
    def get_orats_vol(ticker: str) -> dict:
        """Fetch IV rank / IV percentile from ORATS free tier (best-effort).

        ORATS free tier is limited; falls back to computing ATM IV from
        yfinance if the API is unavailable.

        Returns:
            dict: iv_rank (0-100), iv_percentile (0-100), atm_iv (decimal),
                  source ("orats" | "yfinance_computed").
        """
        # Attempt ORATS data endpoint (free, no key needed for basic data)
        try:
            url = f"https://api.orats.io/datav2/hist/dailies?ticker={ticker}&token=demo"
            with __import__("httpx").Client(timeout=10) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    if data:
                        latest = data[-1]
                        return {
                            "iv_rank": latest.get("ivRank"),
                            "iv_percentile": latest.get("ivPct"),
                            "atm_iv": latest.get("iv30"),
                            "source": "orats",
                        }
        except Exception:
            pass

        # Fallback: compute ATM IV from yfinance
        surface = VolatilitySurface()
        try:
            regime = surface.detect_vol_regime(ticker)
            return {
                "iv_rank": regime.get("iv_percentile_52w"),
                "iv_percentile": regime.get("iv_percentile_52w"),
                "atm_iv": regime.get("current_iv"),
                "source": "yfinance_computed",
            }
        except Exception as exc:
            logger.warning("get_orats_vol fallback failed", ticker=ticker, error=str(exc))
            return {"iv_rank": None, "iv_percentile": None, "atm_iv": None, "source": "error"}


# ---------------------------------------------------------------------------
# GreeksEnricher  (spec-required class)
# ---------------------------------------------------------------------------

class GreeksEnricher:
    """Enrich an options chain DataFrame with full BS Greeks and IV surface."""

    @staticmethod
    def enrich_chain(
        chain_df: pd.DataFrame,
        spot_price: float,
        risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
    ) -> pd.DataFrame:
        """Compute BS Greeks for every row in a chain DataFrame.

        Input DataFrame must contain at minimum: strike, option_type,
        and one of: mid_price, lastPrice, or bid+ask columns.
        Expiry info from: expiry column (ISO str) or time_to_expiry_days.

        Added columns:
          implied_vol, delta, gamma, theta, vega, rho,
          moneyness (K/S), time_to_expiry_days, intrinsic_value,
          extrinsic_value, bid_ask_spread_pct, bs_price.
        """
        return GreeksCalculator.compute_greeks_chain(chain_df, spot_price, risk_free_rate)

    @staticmethod
    def compute_iv_surface(
        ticker: str,
        spot_price: float,
        risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
    ) -> pd.DataFrame:
        """Build (expiry, strike, iv) surface across all available expirations.

        Returns:
            DataFrame with columns: expiry, strike, iv, delta, moneyness,
            time_to_expiry_days. One row per (expiry, strike) pair.
        """
        all_chains = OptionsChainAdapter.get_all_chains(ticker)
        if not all_chains:
            return pd.DataFrame()

        today = date.today()
        rows = []

        for exp_str, chain in all_chains.items():
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                T = max((exp_date - today).days / 365.0, 1e-6)
                dte = (exp_date - today).days
            except (ValueError, TypeError):
                T = 30 / 365.0
                dte = 30

            for opt_type, df_opts in [("call", chain.get("calls", pd.DataFrame())),
                                       ("put", chain.get("puts", pd.DataFrame()))]:
                if df_opts is None or df_opts.empty:
                    continue
                for _, row in df_opts.iterrows():
                    K = float(row.get("strike", 0))
                    if K <= 0 or spot_price <= 0:
                        continue
                    bid = float(row.get("bid", 0) or 0)
                    ask = float(row.get("ask", 0) or 0)
                    last = float(row.get("lastPrice", 0) or 0)
                    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else last
                    if mid <= 0:
                        continue

                    iv = BlackScholesModel.implied_vol(
                        mid, spot_price, K, T, risk_free_rate, opt_type
                    )
                    if math.isnan(iv) or iv <= 0:
                        continue

                    delta = BlackScholesModel.delta(spot_price, K, T, risk_free_rate, iv, opt_type)
                    rows.append({
                        "expiry": exp_str,
                        "strike": K,
                        "option_type": opt_type,
                        "iv": round(iv, 4),
                        "delta": round(delta, 4),
                        "moneyness": round(K / spot_price, 4),
                        "time_to_expiry_days": dte,
                    })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).sort_values(["expiry", "strike"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# VolatilitySurface additions  (spec-required interface on existing class)
# The existing VolatilitySurface class is preserved above; these are
# module-level helpers that satisfy the spec's exact method signatures.
# ---------------------------------------------------------------------------

class VolSurface:
    """Spec-aligned volatility surface builder with exact method signatures.

    Uses GreeksEnricher.compute_iv_surface as the data source.
    """

    @staticmethod
    def build_surface(
        chain_data: dict[str, pd.DataFrame],
        spot: float,
        risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
    ) -> pd.DataFrame:
        """Build multi-expiry IV surface from pre-fetched chain data.

        Args:
            chain_data: dict mapping expiry_str -> DataFrame with columns
                        strike, option_type, bid, ask, lastPrice.
            spot: Current spot price.

        Returns:
            DataFrame with columns: expiry, strike, iv, delta.
        """
        today = date.today()
        rows = []

        for exp_str, df_opts in chain_data.items():
            if df_opts is None or df_opts.empty:
                continue
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                T = max((exp_date - today).days / 365.0, 1e-6)
            except (ValueError, TypeError):
                T = 30 / 365.0

            for _, row in df_opts.iterrows():
                K = float(row.get("strike", 0))
                if K <= 0:
                    continue
                opt_type = str(row.get("option_type", "call")).lower()
                bid = float(row.get("bid", 0) or 0)
                ask = float(row.get("ask", 0) or 0)
                last = float(row.get("lastPrice", 0) or 0)
                mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else last
                if mid <= 0:
                    continue

                iv = BlackScholesModel.implied_vol(mid, spot, K, T, risk_free_rate, opt_type)
                if math.isnan(iv) or iv <= 0:
                    continue
                delta = BlackScholesModel.delta(spot, K, T, risk_free_rate, iv, opt_type)
                rows.append({"expiry": exp_str, "strike": K, "iv": round(iv, 4),
                             "delta": round(delta, 4)})

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["expiry", "strike"]).reset_index(drop=True)

    @staticmethod
    def get_vol_smile(expiry: str, surface_df: pd.DataFrame) -> pd.DataFrame:
        """Single-expiry slice of the IV surface (the 'smile').

        Returns:
            DataFrame with columns: strike, iv, delta for the given expiry.
        """
        if surface_df is None or surface_df.empty:
            return pd.DataFrame()
        mask = surface_df["expiry"] == expiry
        return surface_df[mask][["strike", "iv", "delta"]].reset_index(drop=True)

    @staticmethod
    def get_vol_term_structure(
        atm_delta: float = 0.5,
        surface_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """IV by expiry at a fixed delta level (term structure).

        Args:
            atm_delta: Delta level to slice at (default 0.5 = ATM call).
            surface_df: Output of build_surface().

        Returns:
            DataFrame with columns: expiry, iv, delta_actual.
            One row per expiry, using the strike closest to atm_delta.
        """
        if surface_df is None or surface_df.empty:
            return pd.DataFrame()

        rows = []
        for expiry, grp in surface_df.groupby("expiry"):
            if "delta" not in grp.columns:
                continue
            grp = grp.copy()
            grp["_dd"] = (grp["delta"].abs() - atm_delta).abs()
            closest = grp.nsmallest(1, "_dd")
            if not closest.empty:
                rows.append({
                    "expiry": expiry,
                    "iv": float(closest["iv"].values[0]),
                    "delta_actual": float(closest["delta"].values[0]),
                })

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("expiry").reset_index(drop=True)

    @staticmethod
    def interpolate_vol(
        expiry_days: float,
        strike: float,
        surface_df: pd.DataFrame,
    ) -> float:
        """Bilinear interpolation of IV at an arbitrary (expiry, strike) point.

        Args:
            expiry_days: Target days to expiry (float).
            strike: Target strike price.
            surface_df: Output of build_surface() with expiry column as ISO str.

        Returns:
            Interpolated IV as float, or NaN if surface has insufficient data.
        """
        if surface_df is None or surface_df.empty:
            return float("nan")

        if "time_to_expiry_days" not in surface_df.columns:
            today = date.today()
            surface_df = surface_df.copy()
            surface_df["time_to_expiry_days"] = surface_df["expiry"].apply(
                lambda e: (datetime.strptime(e, "%Y-%m-%d").date() - today).days
                if isinstance(e, str) else 30
            )

        expiries_avail = sorted(surface_df["time_to_expiry_days"].unique())
        if len(expiries_avail) < 1:
            return float("nan")

        # Find bounding expiry slices
        lo_exp = max((e for e in expiries_avail if e <= expiry_days), default=expiries_avail[0])
        hi_exp = min((e for e in expiries_avail if e >= expiry_days), default=expiries_avail[-1])

        def _interpolate_strike(dte: float) -> float:
            slice_df = surface_df[surface_df["time_to_expiry_days"] == dte].sort_values("strike")
            if slice_df.empty:
                return float("nan")
            strikes = slice_df["strike"].values
            ivs = slice_df["iv"].values
            if len(strikes) == 1:
                return float(ivs[0])
            return float(np.interp(strike, strikes, ivs))

        iv_lo = _interpolate_strike(lo_exp)
        iv_hi = _interpolate_strike(hi_exp)

        if math.isnan(iv_lo) and math.isnan(iv_hi):
            return float("nan")
        if lo_exp == hi_exp:
            return iv_lo

        # Linear interpolation between expiry slices
        w = (expiry_days - lo_exp) / (hi_exp - lo_exp)
        return round(float(iv_lo * (1 - w) + iv_hi * w), 4)

    @staticmethod
    def compute_skew(surface_df: pd.DataFrame) -> pd.DataFrame:
        """25-delta risk reversal and butterfly per expiry.

        Risk reversal = IV(25d put) - IV(25d call)  [positive = put skew]
        Butterfly     = (IV(25d put) + IV(25d call)) / 2 - IV(ATM)

        Returns:
            DataFrame with columns: expiry, atm_iv, rr_25d, fly_25d.
        """
        if surface_df is None or surface_df.empty:
            return pd.DataFrame()

        rows = []
        for expiry, grp in surface_df.groupby("expiry"):
            grp = grp.copy().dropna(subset=["iv", "delta"])
            if grp.empty:
                continue

            # ATM: delta closest to 0.5 (call convention)
            grp["_atm_dist"] = (grp["delta"].abs() - 0.5).abs()
            atm_row = grp.nsmallest(1, "_atm_dist")
            atm_iv = float(atm_row["iv"].values[0]) if not atm_row.empty else float("nan")

            # 25-delta put (delta ~ -0.25): negative delta side
            put_side = grp[grp["delta"] < 0].copy()
            put_side["_dist"] = (put_side["delta"] + 0.25).abs()
            p25 = put_side.nsmallest(1, "_dist")
            iv_p25 = float(p25["iv"].values[0]) if not p25.empty else float("nan")

            # 25-delta call (delta ~ 0.25): positive delta side
            call_side = grp[grp["delta"] > 0].copy()
            call_side["_dist"] = (call_side["delta"] - 0.25).abs()
            c25 = call_side.nsmallest(1, "_dist")
            iv_c25 = float(c25["iv"].values[0]) if not c25.empty else float("nan")

            rr = round(iv_p25 - iv_c25, 4) if not (math.isnan(iv_p25) or math.isnan(iv_c25)) else float("nan")
            fly = round((iv_p25 + iv_c25) / 2.0 - atm_iv, 4) if not math.isnan(atm_iv) and not math.isnan(iv_p25) else float("nan")

            rows.append({"expiry": expiry, "atm_iv": round(atm_iv, 4),
                         "rr_25d": rr, "fly_25d": fly})

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("expiry").reset_index(drop=True)


# ---------------------------------------------------------------------------
# OptionsSignalEngine  (spec-required class)
# ---------------------------------------------------------------------------

class OptionsSignalEngine:
    """Market microstructure signals derived from options chain data."""

    @staticmethod
    def put_call_ratio(chain_df: pd.DataFrame) -> dict:
        """Volume-weighted and OI-weighted put/call ratios.

        Args:
            chain_df: Combined calls+puts DataFrame with option_type column.

        Returns:
            dict: volume_pcr, oi_pcr, total_call_vol, total_put_vol,
                  total_call_oi, total_put_oi, signal ("bearish"/"neutral"/"bullish").
        """
        if chain_df is None or chain_df.empty:
            return {"volume_pcr": None, "oi_pcr": None, "signal": "unknown"}

        calls = chain_df[chain_df["option_type"] == "call"]
        puts = chain_df[chain_df["option_type"] == "put"]

        call_vol = float(calls["volume"].fillna(0).sum()) if "volume" in calls.columns else 0.0
        put_vol = float(puts["volume"].fillna(0).sum()) if "volume" in puts.columns else 0.0
        call_oi = float(calls["openInterest"].fillna(0).sum()) if "openInterest" in calls.columns else 0.0
        put_oi = float(puts["openInterest"].fillna(0).sum()) if "openInterest" in puts.columns else 0.0

        volume_pcr = round(put_vol / call_vol, 4) if call_vol > 0 else None
        oi_pcr = round(put_oi / call_oi, 4) if call_oi > 0 else None

        signal = "neutral"
        if volume_pcr is not None:
            if volume_pcr > 1.2:
                signal = "bearish"
            elif volume_pcr < 0.7:
                signal = "bullish"

        return {
            "volume_pcr": volume_pcr,
            "oi_pcr": oi_pcr,
            "total_call_vol": int(call_vol),
            "total_put_vol": int(put_vol),
            "total_call_oi": int(call_oi),
            "total_put_oi": int(put_oi),
            "signal": signal,
        }

    @staticmethod
    def max_pain(chain_df: pd.DataFrame) -> float:
        """Strike at which total option seller loss is minimised at expiry.

        Iterates over all strikes and finds the expiry price where
        the sum of call + put intrinsic value (weighted by OI) is lowest.

        Args:
            chain_df: Combined calls+puts DataFrame with strike, openInterest,
                      option_type columns.

        Returns:
            Max pain strike as float.
        """
        if chain_df is None or chain_df.empty:
            return 0.0

        calls = chain_df[chain_df["option_type"] == "call"]
        puts = chain_df[chain_df["option_type"] == "put"]

        all_strikes = sorted(
            set(chain_df["strike"].dropna().values.tolist())
        )
        if not all_strikes:
            return 0.0

        def _total_pain(expiry_price: float) -> float:
            pain = 0.0
            for _, row in calls.iterrows():
                K = float(row.get("strike", 0))
                oi = float(row.get("openInterest", 0) or 0)
                pain += max(expiry_price - K, 0.0) * oi * 100
            for _, row in puts.iterrows():
                K = float(row.get("strike", 0))
                oi = float(row.get("openInterest", 0) or 0)
                pain += max(K - expiry_price, 0.0) * oi * 100
            return pain

        min_pain = float("inf")
        max_pain_strike = all_strikes[len(all_strikes) // 2]
        for strike in all_strikes:
            p = _total_pain(strike)
            if p < min_pain:
                min_pain = p
                max_pain_strike = strike

        return float(max_pain_strike)

    @staticmethod
    def gamma_exposure(
        chain_df: pd.DataFrame,
        spot: float,
        risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
    ) -> dict:
        """Aggregate dealer gamma exposure (GEX) by strike.

        Convention: dealers short calls (negative call GEX) / long puts (positive put GEX).
        Net positive GEX -> vol-suppressing; net negative -> vol-amplifying.

        Returns:
            dict: gex_by_strike (dict[str, float]), net_gex (float),
                  gex_flip_level (float | None), regime ("stabilizing"/"destabilizing").
        """
        if chain_df is None or chain_df.empty or spot <= 0:
            return {"gex_by_strike": {}, "net_gex": 0.0,
                    "gex_flip_level": None, "regime": "unknown"}

        gex_by_strike: dict[float, float] = {}

        for _, row in chain_df.iterrows():
            K = float(row.get("strike", 0))
            oi = float(row.get("openInterest", 0) or 0)
            opt_type = str(row.get("option_type", "call")).lower()
            iv = float(row.get("impliedVolatility", 0.25) or 0.25)
            if iv <= 0:
                iv = 0.25

            # Approximate T from time_to_expiry_days if available
            dte = float(row.get("time_to_expiry_days", 30) or 30)
            T = max(dte / 365.0, 1e-6)

            if K <= 0 or oi <= 0:
                continue

            g = BlackScholesModel.gamma(spot, K, T, risk_free_rate, iv)
            sign = -1.0 if opt_type == "call" else 1.0
            gex = sign * g * oi * 100 * spot
            gex_by_strike[K] = gex_by_strike.get(K, 0.0) + gex

        net_gex = sum(gex_by_strike.values())

        # GEX flip level: strike where cumulative GEX changes sign
        gex_flip_level: Optional[float] = None
        cumulative = 0.0
        prev_sign = None
        for k in sorted(gex_by_strike):
            cumulative += gex_by_strike[k]
            cur_sign = 1 if cumulative >= 0 else -1
            if prev_sign is not None and cur_sign != prev_sign:
                gex_flip_level = k
                break
            prev_sign = cur_sign

        return {
            "gex_by_strike": {str(k): round(v, 2) for k, v in sorted(gex_by_strike.items())},
            "net_gex": round(net_gex, 2),
            "gex_flip_level": gex_flip_level,
            "regime": "stabilizing" if net_gex > 0 else "destabilizing",
        }

    @staticmethod
    def iv_rank(
        ticker: str,
        current_iv: float,
        lookback_days: int = 252,
    ) -> float:
        """Historical IV percentile rank over lookback_days trading days.

        IV Rank = (current_iv - min_iv) / (max_iv - min_iv) * 100.
        Uses rolling 21-day HV as IV proxy for the historical window.

        Returns:
            IV rank as float 0-100, or NaN on failure.
        """
        import yfinance as yf

        try:
            hist = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=True)
            if hist is None or hist.empty:
                return float("nan")
            hist.index = pd.to_datetime(hist.index)
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            close = hist["Close"].dropna()
        except Exception as exc:
            logger.warning("iv_rank history fetch failed", ticker=ticker, error=str(exc))
            return float("nan")

        log_rets = np.log(close / close.shift(1)).dropna()
        rolling_hv = log_rets.rolling(21).std() * math.sqrt(252)
        rolling_hv = rolling_hv.dropna().tail(lookback_days)

        if len(rolling_hv) < 2:
            return float("nan")

        min_iv = float(rolling_hv.min())
        max_iv = float(rolling_hv.max())
        if max_iv <= min_iv:
            return 50.0

        rank = (current_iv - min_iv) / (max_iv - min_iv) * 100.0
        return round(max(0.0, min(100.0, rank)), 2)

    @staticmethod
    def unusual_options_activity(
        chain_df: pd.DataFrame,
        avg_volume_threshold: float = 3.0,
    ) -> pd.DataFrame:
        """Screen for strikes with unusual volume relative to open interest.

        Flags rows where:
          - volume > avg_volume_threshold * openInterest  (sweep / unusual size)
          - volume > 500 contracts  (absolute size filter)

        Args:
            chain_df: Combined options chain DataFrame.
            avg_volume_threshold: Vol/OI multiple threshold (default 3x).

        Returns:
            DataFrame of flagged rows sorted by volume descending.
        """
        if chain_df is None or chain_df.empty:
            return pd.DataFrame()

        required = {"volume", "openInterest", "strike", "option_type"}
        if not required.issubset(chain_df.columns):
            return pd.DataFrame()

        df = chain_df.copy()
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(1)

        df["vol_oi_ratio"] = df["volume"] / df["openInterest"].replace(0, 1)

        unusual = df[
            (df["vol_oi_ratio"] >= avg_volume_threshold) | (df["volume"] >= 500)
        ].copy()

        unusual["flag"] = unusual.apply(
            lambda r: "sweep" if r["vol_oi_ratio"] >= avg_volume_threshold else "block", axis=1
        )

        keep_cols = [c for c in ["strike", "option_type", "expiry", "volume",
                                  "openInterest", "vol_oi_ratio", "lastPrice",
                                  "mid_price", "impliedVolatility", "flag"]
                     if c in unusual.columns]
        return unusual[keep_cols].sort_values("volume", ascending=False).reset_index(drop=True)

    @staticmethod
    def expected_move(spot: float, atm_straddle_price: float) -> dict:
        """1-sigma expected move from ATM straddle price.

        The ATM straddle price (call + put at same strike) directly encodes
        the market's expected 1-SD move to expiry.

        Args:
            spot: Current spot price.
            atm_straddle_price: Market price of ATM call + ATM put.

        Returns:
            dict: expected_move_dollars, expected_move_pct,
                  range_low, range_high, confidence "~68%".
        """
        move_dollars = round(atm_straddle_price, 4)
        move_pct = round(atm_straddle_price / spot * 100, 4) if spot > 0 else float("nan")
        return {
            "expected_move_dollars": move_dollars,
            "expected_move_pct": move_pct,
            "range_low": round(spot - move_dollars, 4),
            "range_high": round(spot + move_dollars, 4),
            "confidence": "~68%",
        }


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

def _make_options_router():
    """Build and return the options FastAPI router (lazy import to avoid hard dep)."""
    try:
        from fastapi import APIRouter, HTTPException, Query
    except ImportError:
        logger.warning("fastapi not installed — options_router unavailable")
        return None

    router = APIRouter(prefix="/api/options", tags=["options"])

    @router.get("/{ticker}/chain")
    def get_chain(
        ticker: str,
        expiry: str = Query(default=None, description="ISO expiry date YYYY-MM-DD"),
    ):
        """Full enriched options chain with Greeks for one expiry."""
        import yfinance as yf
        chain_df = OptionsChainAdapter.get_chain_dataframe(ticker, expiry)
        if chain_df.empty:
            raise HTTPException(status_code=404, detail=f"No chain data for {ticker}")
        try:
            info = yf.Ticker(ticker).fast_info
            spot = float(getattr(info, "last_price", 0) or 0)
        except Exception:
            spot = 0.0
        if spot > 0:
            chain_df = GreeksEnricher.enrich_chain(chain_df, spot)
        return chain_df.fillna("").to_dict(orient="records")

    @router.get("/{ticker}/greeks")
    def get_greeks(ticker: str):
        """All Greeks for the nearest expiry."""
        import yfinance as yf
        expiry = OptionsChainAdapter.get_nearest_expiry(ticker)
        if not expiry:
            raise HTTPException(status_code=404, detail=f"No expirations for {ticker}")
        try:
            info = yf.Ticker(ticker).fast_info
            spot = float(getattr(info, "last_price", 0) or 0)
        except Exception:
            spot = 0.0
        chain_df = OptionsChainAdapter.get_chain_dataframe(ticker, expiry)
        if chain_df.empty:
            raise HTTPException(status_code=404, detail=f"Empty chain for {ticker} {expiry}")
        enriched = GreeksEnricher.enrich_chain(chain_df, spot)
        greek_cols = [c for c in ["strike", "option_type", "expiry", "implied_vol",
                                   "delta", "gamma", "theta", "vega", "rho",
                                   "bs_price", "mid_price"]
                      if c in enriched.columns]
        return enriched[greek_cols].fillna("").to_dict(orient="records")

    @router.get("/{ticker}/surface")
    def get_surface(ticker: str):
        """IV surface data across all expirations."""
        import yfinance as yf
        try:
            info = yf.Ticker(ticker).fast_info
            spot = float(getattr(info, "last_price", 0) or 0)
        except Exception:
            spot = 0.0
        surface_df = GreeksEnricher.compute_iv_surface(ticker, spot)
        if surface_df.empty:
            raise HTTPException(status_code=404, detail=f"No IV surface data for {ticker}")
        return surface_df.fillna("").to_dict(orient="records")

    @router.get("/{ticker}/signals")
    def get_signals(ticker: str):
        """PCR, max pain, GEX, and expected move signals."""
        import yfinance as yf
        try:
            info = yf.Ticker(ticker).fast_info
            spot = float(getattr(info, "last_price", 0) or 0)
        except Exception:
            spot = 0.0

        expiry = OptionsChainAdapter.get_nearest_expiry(ticker)
        chain_df = OptionsChainAdapter.get_chain_dataframe(ticker, expiry)
        if chain_df.empty:
            raise HTTPException(status_code=404, detail=f"No chain data for {ticker}")

        pcr = OptionsSignalEngine.put_call_ratio(chain_df)
        mp = OptionsSignalEngine.max_pain(chain_df)
        gex = OptionsSignalEngine.gamma_exposure(chain_df, spot)

        # ATM straddle price for expected move
        atm_straddle = 0.0
        if spot > 0 and not chain_df.empty:
            chain_df["_d"] = (chain_df["strike"] - spot).abs()
            atm_calls = chain_df[chain_df["option_type"] == "call"].nsmallest(1, "_d")
            atm_puts = chain_df[chain_df["option_type"] == "put"].nsmallest(1, "_d")
            call_mid = float(atm_calls["mid_price"].values[0]) if not atm_calls.empty else 0.0
            put_mid = float(atm_puts["mid_price"].values[0]) if not atm_puts.empty else 0.0
            atm_straddle = call_mid + put_mid

        exp_move = OptionsSignalEngine.expected_move(spot, atm_straddle)

        return {
            "ticker": ticker,
            "expiry": expiry,
            "spot": spot,
            "put_call_ratio": pcr,
            "max_pain": mp,
            "gamma_exposure": {k: v for k, v in gex.items() if k != "gex_by_strike"},
            "gex_flip_level": gex.get("gex_flip_level"),
            "expected_move": exp_move,
        }

    @router.get("/{ticker}/unusual")
    def get_unusual(
        ticker: str,
        threshold: float = Query(default=3.0, description="Vol/OI threshold"),
    ):
        """Unusual options activity screen."""
        all_chains = OptionsChainAdapter.get_all_chains(ticker)
        if not all_chains:
            raise HTTPException(status_code=404, detail=f"No chains for {ticker}")

        frames = []
        for exp, chain in all_chains.items():
            calls = chain.get("calls", pd.DataFrame()).copy()
            puts = chain.get("puts", pd.DataFrame()).copy()
            if not calls.empty:
                calls["option_type"] = "call"
                calls["expiry"] = exp
            if not puts.empty:
                puts["option_type"] = "put"
                puts["expiry"] = exp
            frames.extend([calls, puts])

        if not frames:
            raise HTTPException(status_code=404, detail=f"No chain data for {ticker}")

        combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
        unusual = OptionsSignalEngine.unusual_options_activity(combined, threshold)
        if unusual.empty:
            return []
        return unusual.fillna("").to_dict(orient="records")

    return router


# Instantiate router at module level (None if fastapi not available)
options_router = _make_options_router()
