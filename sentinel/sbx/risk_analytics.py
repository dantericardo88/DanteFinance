"""
Factor-Based Risk Attribution & Portfolio Risk Analytics — Dimension #076 (target 8+).

Comprehensive portfolio risk module using Fama-French factors, historical VaR,
stress testing, covariance estimation, and risk decomposition.

Public API
----------
FamaFrenchAdapter
    get_ff3_factors(start, end)             -> pd.DataFrame
    get_ff5_factors(start, end)             -> pd.DataFrame
    get_momentum_factor(start)              -> pd.DataFrame
    get_q_factors()                         -> pd.DataFrame

FactorRiskModel
    estimate_factor_loadings(returns, factors, window)  -> dict
    rolling_factor_loadings(returns, factors, window)   -> pd.DataFrame
    factor_attribution(portfolio_returns, factors, ...)  -> dict
    factor_risk_decomposition(betas, factor_cov, ...)   -> dict

PortfolioRiskEngine
    compute_var(returns, confidence, method, window)    -> dict
    compute_portfolio_var(weights, returns_dict, ...)   -> dict
    stress_test(weights, returns_dict)                  -> dict
    covariance_matrix(returns_dict, method)             -> pd.DataFrame
    compute_tracking_error(portfolio_returns, bench)    -> dict
    risk_contribution(weights, cov_matrix)              -> pd.DataFrame
"""
from __future__ import annotations

import io
import os
import time
import warnings
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
import numpy as np
import pandas as pd
from scipy import stats

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
}
_TIMEOUT = 60.0

# Ken French Data Library URLs
_FF3_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
_FF5_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
_MOM_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)

# AQR Q-factor proxy (public dataset)
_AQR_QFACTOR_URL = (
    "https://www.aqr.com/Insights/Datasets/The-q-factor-model-dataset"
)

# Cache directory
_CACHE_DIR = Path(os.environ.get("SENTINEL_HOME", Path.home() / ".sentinel")) / "cache" / "ff_factors"

# Historical stress scenario date ranges (peak-to-trough window)
_STRESS_SCENARIOS: dict[str, tuple[str, str]] = {
    "2008_GFC_Lehman":    ("2008-09-01", "2009-03-09"),
    "2020_COVID_Crash":   ("2020-02-19", "2020-03-23"),
    "2000_DotCom_Bust":   ("2000-03-10", "2002-10-09"),
    "1998_LTCM_Crisis":   ("1998-07-17", "1998-10-08"),
    "2022_Rate_Shock":    ("2022-01-03", "2022-10-12"),
    "2011_EU_Debt":       ("2011-05-02", "2011-10-03"),
}

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _cache_path(name: str) -> Path:
    return _ensure_cache_dir() / f"{name}.parquet"


def _is_cache_fresh(path: Path, max_age_hours: int = 24) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < max_age_hours * 3600


def _download_zip_csv(url: str, timeout: float = _TIMEOUT) -> str:
    """Download a .zip from Ken French data library and return the CSV text."""
    resp = httpx.get(url, headers=_HEADERS, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv"))
        return zf.read(csv_name).decode("utf-8", errors="replace")


def _parse_french_csv(csv_text: str, factor_cols: list[str]) -> pd.DataFrame:
    """
    Parse Ken French CSV format.
    French CSVs have a header block, then a date column (YYYYMMDD) + factor columns,
    then sometimes annual data at the bottom.
    """
    lines = csv_text.splitlines()
    data_lines: list[str] = []
    in_data = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Detect data start: first 8-digit date
        parts = [p.strip() for p in stripped.split(",")]
        if not in_data:
            if parts[0].isdigit() and len(parts[0]) in {6, 8}:
                in_data = True
            else:
                continue
        if in_data:
            # Stop at annual section (6-digit year rows after daily data)
            if parts[0].isdigit() and len(parts[0]) == 4:
                break
            data_lines.append(stripped)

    if not data_lines:
        return pd.DataFrame()

    buf = io.StringIO("\n".join(data_lines))
    try:
        df = pd.read_csv(buf, header=None)
    except Exception:
        return pd.DataFrame()

    # Column 0 is date, then factor columns
    ncols = min(len(factor_cols) + 1, df.shape[1])
    df = df.iloc[:, :ncols].copy()
    df.columns = ["date"] + factor_cols[: ncols - 1]  # type: ignore[assignment]

    # Parse date
    df["date"] = df["date"].astype(str).str.strip()
    date_fmt = "%Y%m%d" if df["date"].str.len().max() == 8 else "%Y%m"
    df["date"] = pd.to_datetime(df["date"], format=date_fmt, errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")

    # Convert to float (Ken French stores as percentages)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    df = df.dropna()
    return df


def _align_returns(
    returns: pd.Series, factors: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:
    """Align returns and factor DataFrame to common dates, drop NaN."""
    combined = pd.concat([returns.rename("__ret__"), factors], axis=1).dropna()
    return combined["__ret__"], combined.drop(columns=["__ret__"])


# ---------------------------------------------------------------------------
# FamaFrenchAdapter
# ---------------------------------------------------------------------------

class FamaFrenchAdapter:
    """
    Download and cache Fama-French factor data from Ken French's data library.

    All factors are returned as decimal daily returns (not percentages).
    """

    def get_ff3_factors(
        self,
        start: str = "1926-07-01",
        end: str = None,
    ) -> pd.DataFrame:
        """
        Fama-French 3-Factor daily data: Mkt-RF, SMB, HML, RF.

        Source: Ken French Data Library (F-F_Research_Data_Factors_daily).
        """
        cache = _cache_path("ff3_daily")
        if _is_cache_fresh(cache):
            df = pd.read_parquet(cache)
        else:
            logger.info("Downloading FF3 daily factors from Ken French library")
            try:
                csv_text = _download_zip_csv(_FF3_DAILY_URL)
                df = _parse_french_csv(csv_text, ["Mkt-RF", "SMB", "HML", "RF"])
                if not df.empty:
                    df.to_parquet(cache)
            except Exception as exc:
                logger.error("FF3 download failed: %s", exc)
                return pd.DataFrame(columns=["Mkt-RF", "SMB", "HML", "RF"])

        df = df.sort_index()
        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end) if end else pd.Timestamp.today()
        return df.loc[start_dt:end_dt]

    def get_ff5_factors(
        self,
        start: str = "1963-07-01",
        end: str = None,
    ) -> pd.DataFrame:
        """
        Fama-French 5-Factor daily data: Mkt-RF, SMB, HML, RMW, CMA, RF.

        Source: Ken French Data Library (F-F_Research_Data_5_Factors_2x3_daily).
        """
        cache = _cache_path("ff5_daily")
        if _is_cache_fresh(cache):
            df = pd.read_parquet(cache)
        else:
            logger.info("Downloading FF5 daily factors from Ken French library")
            try:
                csv_text = _download_zip_csv(_FF5_DAILY_URL)
                df = _parse_french_csv(
                    csv_text, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
                )
                if not df.empty:
                    df.to_parquet(cache)
            except Exception as exc:
                logger.error("FF5 download failed: %s", exc)
                return pd.DataFrame(columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])

        df = df.sort_index()
        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end) if end else pd.Timestamp.today()
        return df.loc[start_dt:end_dt]

    def get_momentum_factor(
        self,
        start: str = "1927-01-01",
    ) -> pd.DataFrame:
        """
        Momentum (Mom) factor daily data.

        Source: Ken French Data Library (F-F_Momentum_Factor_daily).
        """
        cache = _cache_path("mom_daily")
        if _is_cache_fresh(cache):
            df = pd.read_parquet(cache)
        else:
            logger.info("Downloading Momentum factor from Ken French library")
            try:
                csv_text = _download_zip_csv(_MOM_DAILY_URL)
                df = _parse_french_csv(csv_text, ["Mom"])
                if not df.empty:
                    df.to_parquet(cache)
            except Exception as exc:
                logger.error("Momentum download failed: %s", exc)
                return pd.DataFrame(columns=["Mom"])

        df = df.sort_index()
        return df.loc[pd.Timestamp(start):]

    def get_q_factors(self) -> pd.DataFrame:
        """
        Hou-Xue-Zhang q-factors: R_MKT, R_ME, R_IA, R_ROE.

        We proxy using AQR's publicly available factor data. On failure we
        return the FF5 factors as an approximation.
        """
        cache = _cache_path("q_factors")
        if _is_cache_fresh(cache, max_age_hours=168):  # weekly refresh
            return pd.read_parquet(cache)

        # Try AQR's Century of Factor Premia dataset (CSV format)
        AQR_URL = (
            "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
            "Century-of-Factor-Premia-Monthly.xlsx"
        )
        try:
            resp = httpx.get(AQR_URL, headers=_HEADERS, timeout=30, follow_redirects=True)
            if resp.status_code == 200 and len(resp.content) > 1000:
                df = pd.read_excel(io.BytesIO(resp.content), sheet_name=0, header=18)
                df.to_parquet(cache)
                logger.info("AQR q-factor data cached")
                return df
        except Exception as exc:
            logger.warning("AQR q-factor fetch failed (%s); falling back to FF5", exc)

        # Fallback: return FF5 as q-factor proxy
        ff5 = self.get_ff5_factors()
        ff5_renamed = ff5.rename(
            columns={"Mkt-RF": "R_MKT", "SMB": "R_ME", "CMA": "R_IA", "RMW": "R_ROE"}
        )
        return ff5_renamed

    def get_combined_factors(
        self,
        start: str = "1963-07-01",
        end: str = None,
        include_momentum: bool = True,
    ) -> pd.DataFrame:
        """
        Return FF5 + optionally Momentum, aligned on common dates.
        """
        ff5 = self.get_ff5_factors(start=start, end=end)
        if not include_momentum:
            return ff5
        mom = self.get_momentum_factor(start=start)
        if end:
            mom = mom.loc[:pd.Timestamp(end)]
        combined = ff5.join(mom, how="left")
        return combined


# ---------------------------------------------------------------------------
# FactorRiskModel
# ---------------------------------------------------------------------------

class FactorRiskModel:
    """
    OLS-based factor risk model: estimate betas, roll loadings, attribute returns.
    """

    def estimate_factor_loadings(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
        window: int = 252,
    ) -> dict:
        """
        OLS regression: excess_return ~ sum(beta_i * factor_i).

        Returns:
          betas (dict factor->float), alpha_annualized, r_squared,
          t_stats (dict), p_values (dict), residual_vol_annualized
        """
        # Align
        ret, fac = _align_returns(returns, factors)

        # Use at most last `window` observations
        if len(ret) > window:
            ret = ret.iloc[-window:]
            fac = fac.iloc[-window:]

        if len(ret) < 30:
            return {
                "error": "insufficient_data",
                "observations": len(ret),
                "betas": {},
                "alpha_annualized": None,
                "r_squared": None,
                "t_stats": {},
                "p_values": {},
                "residual_vol_annualized": None,
            }

        # Subtract risk-free if available
        rf = fac["RF"].values if "RF" in fac.columns else np.zeros(len(ret))
        y  = ret.values - rf
        factor_cols = [c for c in fac.columns if c != "RF"]
        X  = fac[factor_cols].values
        # Add intercept
        X_int = np.column_stack([np.ones(len(X)), X])

        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X_int, y, rcond=None)
        except np.linalg.LinAlgError as exc:
            return {"error": str(exc)}

        alpha_daily  = coeffs[0]
        betas_arr    = coeffs[1:]
        y_hat        = X_int @ coeffs
        resid        = y - y_hat
        ss_res       = float(np.sum(resid ** 2))
        ss_tot       = float(np.sum((y - y.mean()) ** 2))
        r_sq         = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        n = len(y)
        k = X_int.shape[1]
        dof = max(n - k, 1)
        sigma2 = ss_res / dof

        # Standard errors via (X'X)^-1 * sigma^2
        try:
            XtX_inv = np.linalg.inv(X_int.T @ X_int)
            se_arr  = np.sqrt(np.diag(XtX_inv) * sigma2)
        except np.linalg.LinAlgError:
            se_arr = np.full(k, np.nan)

        all_coeffs = np.concatenate([[alpha_daily], betas_arr])
        t_arr = all_coeffs / np.where(se_arr > 0, se_arr, np.nan)
        p_arr = 2 * (1 - stats.t.cdf(np.abs(t_arr), df=dof))

        betas  = {col: float(b) for col, b in zip(factor_cols, betas_arr)}
        t_stats = {col: float(t) for col, t in zip(factor_cols, t_arr[1:])}
        p_vals  = {col: float(p) for col, p in zip(factor_cols, p_arr[1:])}

        resid_vol_ann = float(np.std(resid) * np.sqrt(TRADING_DAYS_PER_YEAR))
        alpha_ann     = float(alpha_daily * TRADING_DAYS_PER_YEAR)

        return {
            "betas": betas,
            "alpha_annualized": round(alpha_ann, 6),
            "alpha_t_stat": float(t_arr[0]),
            "alpha_p_value": float(p_arr[0]),
            "r_squared": round(r_sq, 4),
            "t_stats": {k: round(v, 4) for k, v in t_stats.items()},
            "p_values": {k: round(v, 6) for k, v in p_vals.items()},
            "residual_vol_annualized": round(resid_vol_ann, 6),
            "observations": n,
        }

    def rolling_factor_loadings(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
        window: int = 60,
    ) -> pd.DataFrame:
        """
        Compute rolling OLS betas. Returns DataFrame with date index and one
        column per factor plus 'alpha'.

        Minimum 30 observations needed per window.
        """
        ret, fac = _align_returns(returns, factors)
        factor_cols = [c for c in fac.columns if c != "RF"]
        rf = fac["RF"].values if "RF" in fac.columns else np.zeros(len(ret))

        y_all = ret.values - rf
        X_all = fac[factor_cols].values
        X_int = np.column_stack([np.ones(len(X_all)), X_all])

        results: list[dict] = []
        dates: list[pd.Timestamp] = []

        for i in range(window, len(y_all) + 1):
            y_w = y_all[i - window: i]
            X_w = X_int[i - window: i]
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
                row = {"alpha": float(coeffs[0] * TRADING_DAYS_PER_YEAR)}
                for j, col in enumerate(factor_cols):
                    row[col] = float(coeffs[j + 1])
            except np.linalg.LinAlgError:
                row = {"alpha": np.nan}
                for col in factor_cols:
                    row[col] = np.nan
            results.append(row)
            dates.append(ret.index[i - 1])

        return pd.DataFrame(results, index=pd.DatetimeIndex(dates))

    def factor_attribution(
        self,
        portfolio_returns: pd.Series,
        factors: pd.DataFrame,
        weights: dict = None,
    ) -> dict:
        """
        Decompose total portfolio return into factor contributions.

        Returns:
          factor_contributions (dict): each factor's total return contribution
          alpha_contribution: residual return not explained by factors
          attribution_df: daily time series of contributions
          total_return: cumulative portfolio return
        """
        ret, fac = _align_returns(portfolio_returns, factors)
        factor_cols = [c for c in fac.columns if c != "RF"]
        rf = fac["RF"].values if "RF" in fac.columns else np.zeros(len(ret))

        # Estimate factor loadings on full sample
        loadings = self.estimate_factor_loadings(ret, fac)
        if "error" in loadings:
            return {"error": loadings["error"]}

        betas = loadings["betas"]
        # Daily factor contributions: beta_i * factor_return_i
        contrib_df = pd.DataFrame(index=fac.index)
        for factor_col in factor_cols:
            beta = betas.get(factor_col, 0.0)
            contrib_df[factor_col] = beta * fac[factor_col].values

        # Alpha contribution: excess_ret - sum(factor_contributions)
        excess_ret = ret.values - rf
        total_factor_contrib = contrib_df.sum(axis=1).values
        contrib_df["alpha"] = excess_ret - total_factor_contrib
        contrib_df["rf"]    = rf
        contrib_df["total"] = ret.values

        # Aggregate contributions
        factor_contributions = {
            col: float(contrib_df[col].sum())
            for col in factor_cols
        }
        alpha_contribution = float(contrib_df["alpha"].sum())
        total_return       = float(ret.sum())

        return {
            "factor_contributions": {
                k: round(v, 6) for k, v in factor_contributions.items()
            },
            "alpha_contribution": round(alpha_contribution, 6),
            "rf_contribution": round(float(contrib_df["rf"].sum()), 6),
            "total_return": round(total_return, 6),
            "betas": {k: round(v, 4) for k, v in betas.items()},
            "r_squared": loadings.get("r_squared"),
            "attribution_df": contrib_df,
        }

    def factor_risk_decomposition(
        self,
        betas: dict,
        factor_covariance: pd.DataFrame,
        idiosyncratic_var: float,
    ) -> dict:
        """
        Decompose total portfolio variance into systematic (factor) and
        idiosyncratic components.

        systematic_var = beta' Σ_factors beta
        total_var      = systematic_var + idiosyncratic_var
        """
        factor_names = list(betas.keys())
        b = np.array([betas[f] for f in factor_names])

        # Filter covariance matrix to available factors
        avail = [f for f in factor_names if f in factor_covariance.index]
        b_avail = np.array([betas[f] for f in avail])
        Sigma   = factor_covariance.loc[avail, avail].values

        try:
            systematic_var = float(b_avail @ Sigma @ b_avail)
        except Exception:
            systematic_var = 0.0

        total_var       = systematic_var + idiosyncratic_var
        systematic_pct  = systematic_var / total_var * 100 if total_var > 0 else 0.0
        idiosyncratic_pct = 100 - systematic_pct

        # Per-factor variance contribution: beta_i * (Sigma @ b)_i
        factor_risk_contribs: dict[str, float] = {}
        if len(avail) > 0:
            sigma_b = Sigma @ b_avail
            for i, factor in enumerate(avail):
                factor_risk_contribs[factor] = float(b_avail[i] * sigma_b[i])

        return {
            "systematic_var": round(systematic_var, 8),
            "idiosyncratic_var": round(idiosyncratic_var, 8),
            "total_var": round(total_var, 8),
            "systematic_pct": round(systematic_pct, 2),
            "idiosyncratic_pct": round(idiosyncratic_pct, 2),
            "factor_risk_contributions": {
                k: round(v, 8) for k, v in factor_risk_contribs.items()
            },
            "systematic_vol_annualized": round(
                float(np.sqrt(max(systematic_var, 0)) * np.sqrt(TRADING_DAYS_PER_YEAR)), 4
            ),
            "total_vol_annualized": round(
                float(np.sqrt(max(total_var, 0)) * np.sqrt(TRADING_DAYS_PER_YEAR)), 4
            ),
        }


# ---------------------------------------------------------------------------
# PortfolioRiskEngine
# ---------------------------------------------------------------------------

class PortfolioRiskEngine:
    """
    VaR, CVaR, stress testing, covariance estimation, and risk decomposition.
    """

    # ------------------------------------------------------------------ #
    # VaR / CVaR                                                           #
    # ------------------------------------------------------------------ #

    def compute_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        method: Literal["historical", "parametric", "cornish_fisher"] = "historical",
        window: int = 252,
    ) -> dict:
        """
        Compute Value at Risk and Expected Shortfall (CVaR).

        Methods:
          historical     — empirical quantile of rolling window
          parametric     — normal distribution (mean, std)
          cornish_fisher — skewness/kurtosis-adjusted Z-score (modified VaR)

        Returns daily VaR and CVaR; annual values (×√252).
        """
        r = returns.dropna()
        if len(r) > window:
            r = r.iloc[-window:]

        if len(r) < 10:
            return {"error": "insufficient_data"}

        alpha = 1 - confidence  # left-tail probability
        mu    = float(r.mean())
        sigma = float(r.std(ddof=1))
        skew  = float(stats.skew(r))
        kurt  = float(stats.kurtosis(r))  # excess kurtosis

        if method == "historical":
            var_daily  = float(-np.percentile(r, alpha * 100))
            cvar_daily = float(-r[r <= -var_daily].mean()) if any(r <= -var_daily) else var_daily

        elif method == "parametric":
            z          = stats.norm.ppf(alpha)
            var_daily  = float(-(mu + z * sigma))
            cvar_daily = float(-(mu - sigma * stats.norm.pdf(z) / alpha))

        elif method == "cornish_fisher":
            z   = stats.norm.ppf(alpha)
            # Cornish-Fisher Z adjustment
            z_cf = (
                z
                + (z ** 2 - 1) * skew / 6
                + (z ** 3 - 3 * z) * kurt / 24
                - (2 * z ** 3 - 5 * z) * skew ** 2 / 36
            )
            var_daily = float(-(mu + z_cf * sigma))
            # CVaR: use numerical integration over empirical distribution below VaR
            threshold = -var_daily
            tail = r[r <= threshold]
            cvar_daily = float(-tail.mean()) if len(tail) > 0 else var_daily

        else:
            raise ValueError(f"Unknown VaR method: {method}")

        return {
            "method": method,
            "confidence": confidence,
            "window_days": len(r),
            "var_daily": round(var_daily, 6),
            "var_annual": round(var_daily * np.sqrt(TRADING_DAYS_PER_YEAR), 6),
            "cvar_daily": round(cvar_daily, 6),
            "cvar_annual": round(cvar_daily * np.sqrt(TRADING_DAYS_PER_YEAR), 6),
            "mean_daily": round(mu, 6),
            "vol_daily": round(sigma, 6),
            "vol_annual": round(sigma * np.sqrt(TRADING_DAYS_PER_YEAR), 4),
            "skewness": round(skew, 4),
            "excess_kurtosis": round(kurt, 4),
        }

    def compute_portfolio_var(
        self,
        weights: dict[str, float],
        returns_dict: dict[str, pd.Series],
        confidence: float = 0.95,
        method: str = "historical",
    ) -> dict:
        """
        Portfolio-level VaR incorporating cross-asset correlations.

        Computes:
          - Individual asset VaRs
          - Portfolio VaR (using historical portfolio returns)
          - Diversification benefit: sum(individual) - portfolio VaR
        """
        tickers = list(weights.keys())

        # Build returns matrix on common dates
        ret_df = pd.DataFrame(
            {t: returns_dict[t] for t in tickers if t in returns_dict}
        ).dropna()
        if ret_df.empty:
            return {"error": "no_aligned_returns"}

        w = np.array([weights.get(t, 0.0) for t in ret_df.columns])
        w = w / w.sum() if w.sum() > 0 else w

        # Portfolio returns
        port_ret = pd.Series(ret_df.values @ w, index=ret_df.index)

        # Portfolio VaR
        port_var = self.compute_var(port_ret, confidence=confidence, method=method)

        # Individual VaRs
        individual_vars: dict[str, dict] = {}
        sum_individual_var = 0.0
        for ticker in ret_df.columns:
            iv = self.compute_var(
                ret_df[ticker], confidence=confidence, method=method
            )
            individual_vars[ticker] = iv
            # Weighted individual VaR
            w_ticker = weights.get(ticker, 0.0)
            sum_individual_var += w_ticker * iv.get("var_daily", 0.0)

        div_benefit = sum_individual_var - port_var.get("var_daily", 0.0)

        return {
            "portfolio_var": port_var,
            "individual_vars": individual_vars,
            "sum_weighted_individual_var_daily": round(sum_individual_var, 6),
            "diversification_benefit_daily": round(div_benefit, 6),
            "diversification_ratio": round(
                div_benefit / sum_individual_var if sum_individual_var > 0 else 0.0, 4
            ),
            "confidence": confidence,
            "method": method,
            "n_assets": len(tickers),
        }

    # ------------------------------------------------------------------ #
    # Stress Testing                                                        #
    # ------------------------------------------------------------------ #

    def stress_test(
        self,
        weights: dict[str, float],
        returns_dict: dict[str, pd.Series],
    ) -> dict:
        """
        Apply 6 historical stress scenarios and compute portfolio P&L.

        Scenarios: 2008 GFC, 2020 COVID, 2000 dot-com, 1998 LTCM,
                   2022 rate shock, 2011 EU debt crisis.
        """
        tickers = [t for t in weights if t in returns_dict]
        ret_df  = pd.DataFrame({t: returns_dict[t] for t in tickers}).dropna()
        w = np.array([weights.get(t, 0.0) for t in ret_df.columns])
        w = w / w.sum() if w.sum() > 0 else w

        port_ret = pd.Series(ret_df.values @ w, index=ret_df.index)
        port_ret.index = pd.to_datetime(port_ret.index)

        scenario_results: dict[str, dict] = {}

        for scenario_name, (start_str, end_str) in _STRESS_SCENARIOS.items():
            start_dt = pd.Timestamp(start_str)
            end_dt   = pd.Timestamp(end_str)
            window   = port_ret.loc[start_dt:end_dt]

            if len(window) < 2:
                scenario_results[scenario_name] = {
                    "status": "no_data_in_window",
                    "start": start_str,
                    "end": end_str,
                    "portfolio_return_pct": None,
                    "max_drawdown_pct": None,
                    "worst_day_pct": None,
                }
                continue

            # Cumulative return
            cum_ret = float((1 + window).prod() - 1)

            # Max drawdown
            cumulative = (1 + window).cumprod()
            rolling_max = cumulative.cummax()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_dd   = float(drawdown.min())

            worst_day = float(window.min())

            # Individual asset performance
            asset_returns: dict[str, float] = {}
            for ticker in ret_df.columns:
                asset_window = ret_df[ticker].loc[start_dt:end_dt]
                if len(asset_window) > 0:
                    asset_returns[ticker] = round(
                        float((1 + asset_window).prod() - 1) * 100, 2
                    )

            scenario_results[scenario_name] = {
                "start": start_str,
                "end": end_str,
                "trading_days": len(window),
                "portfolio_return_pct": round(cum_ret * 100, 2),
                "max_drawdown_pct": round(max_dd * 100, 2),
                "worst_day_pct": round(worst_day * 100, 2),
                "asset_returns": asset_returns,
            }

        # Sort by portfolio loss (worst first)
        sorted_scenarios = dict(
            sorted(
                scenario_results.items(),
                key=lambda x: (
                    x[1].get("portfolio_return_pct") or 0.0
                ),
            )
        )

        return {
            "scenarios": sorted_scenarios,
            "worst_scenario": next(
                (k for k in sorted_scenarios if sorted_scenarios[k].get("portfolio_return_pct") is not None),
                None,
            ),
            "weights": weights,
            "tickers": tickers,
        }

    # ------------------------------------------------------------------ #
    # Covariance Estimation                                                 #
    # ------------------------------------------------------------------ #

    def covariance_matrix(
        self,
        returns_dict: dict[str, pd.Series],
        method: Literal["ledoit_wolf", "sample", "ewma"] = "ledoit_wolf",
    ) -> pd.DataFrame:
        """
        Estimate covariance matrix using one of three methods:

          ledoit_wolf — Oracle Approximating Shrinkage (sklearn)
          sample      — standard sample covariance
          ewma        — Exponentially Weighted Moving Average (RiskMetrics λ=0.94)
        """
        ret_df = pd.DataFrame(returns_dict).dropna()
        if ret_df.empty or ret_df.shape[1] < 2:
            return pd.DataFrame()

        tickers = ret_df.columns.tolist()

        if method == "sample":
            cov = ret_df.cov()
            return cov

        elif method == "ledoit_wolf":
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf()
                lw.fit(ret_df.values)
                cov_arr = lw.covariance_
                return pd.DataFrame(cov_arr, index=tickers, columns=tickers)
            except ImportError:
                logger.warning("sklearn not available; falling back to sample covariance")
                return ret_df.cov()

        elif method == "ewma":
            lam = 0.94  # RiskMetrics daily decay
            n, p = ret_df.shape
            # Compute EWMA covariance
            weights_ew = np.array([(1 - lam) * lam ** i for i in range(n - 1, -1, -1)])
            weights_ew /= weights_ew.sum()
            mu_ew = (ret_df.values.T @ weights_ew)
            demeaned = ret_df.values - mu_ew
            cov_arr = (demeaned * weights_ew[:, None]).T @ demeaned
            return pd.DataFrame(cov_arr, index=tickers, columns=tickers)

        else:
            raise ValueError(f"Unknown covariance method: {method}")

    # ------------------------------------------------------------------ #
    # Tracking Error & Information Ratio                                    #
    # ------------------------------------------------------------------ #

    def compute_tracking_error(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> dict:
        """
        Compute annualized tracking error and information ratio.

        tracking_error = std(active_return) × √252
        information_ratio = active_return_mean / tracking_error × √252
        """
        combined = pd.concat(
            [portfolio_returns.rename("port"), benchmark_returns.rename("bench")],
            axis=1,
        ).dropna()
        if combined.empty:
            return {"error": "no_aligned_returns"}

        active = combined["port"] - combined["bench"]
        te_ann = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        active_return_ann = float(active.mean() * TRADING_DAYS_PER_YEAR)
        ir    = active_return_ann / te_ann if te_ann > 0 else None

        # Beta to benchmark
        cov_matrix = np.cov(combined["port"].values, combined["bench"].values)
        beta_to_bench = (
            float(cov_matrix[0, 1] / cov_matrix[1, 1])
            if cov_matrix[1, 1] > 0 else None
        )

        # Correlation
        corr = float(combined["port"].corr(combined["bench"]))

        return {
            "tracking_error_annualized": round(te_ann, 4),
            "active_return_annualized": round(active_return_ann, 4),
            "information_ratio": round(ir, 4) if ir else None,
            "beta_to_benchmark": round(beta_to_bench, 4) if beta_to_bench else None,
            "correlation_to_benchmark": round(corr, 4),
            "n_observations": len(combined),
        }

    # ------------------------------------------------------------------ #
    # Risk Contribution & ERC                                               #
    # ------------------------------------------------------------------ #

    def risk_contribution(
        self,
        weights: dict[str, float],
        cov_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute per-asset marginal and total risk contributions.

        Marginal Risk Contribution (MRC) = ∂σ_p/∂w_i = (Σw)_i / σ_p
        Total Risk Contribution (TRC)    = w_i × MRC_i
        % Risk Contribution              = TRC_i / σ_p

        Also computes Equal Risk Contribution (ERC) target weights.
        """
        tickers_in_cov = [t for t in weights if t in cov_matrix.index]
        if not tickers_in_cov:
            return pd.DataFrame()

        w_arr = np.array([weights.get(t, 0.0) for t in tickers_in_cov])
        w_arr = w_arr / w_arr.sum() if w_arr.sum() > 0 else w_arr

        Sigma = cov_matrix.loc[tickers_in_cov, tickers_in_cov].values
        port_var = float(w_arr @ Sigma @ w_arr)
        port_vol = float(np.sqrt(max(port_var, 0)))

        Sigma_w  = Sigma @ w_arr
        mrc_arr  = Sigma_w / port_vol if port_vol > 0 else np.zeros(len(w_arr))
        trc_arr  = w_arr * mrc_arr
        pct_arr  = trc_arr / port_vol if port_vol > 0 else np.zeros(len(w_arr))

        # ERC target weights via iterative algorithm
        erc_weights = self._compute_erc_weights(Sigma)

        rows = []
        for i, ticker in enumerate(tickers_in_cov):
            rows.append(
                {
                    "ticker": ticker,
                    "weight": round(float(w_arr[i]), 4),
                    "marginal_risk_contribution": round(float(mrc_arr[i]), 6),
                    "total_risk_contribution": round(float(trc_arr[i]), 6),
                    "pct_of_portfolio_risk": round(float(pct_arr[i]) * 100, 2),
                    "erc_target_weight": round(float(erc_weights[i]), 4),
                }
            )

        df = pd.DataFrame(rows)
        return df

    @staticmethod
    def _compute_erc_weights(
        Sigma: np.ndarray,
        tol: float = 1e-8,
        max_iter: int = 500,
    ) -> np.ndarray:
        """
        Newton-Raphson / cyclical coordinate descent for Equal Risk Contribution.
        Solves: w_i × (Σw)_i = w_j × (Σw)_j for all i, j.
        """
        n = Sigma.shape[0]
        w = np.ones(n) / n  # start at equal weights

        for _ in range(max_iter):
            Sigma_w = Sigma @ w
            port_var = float(w @ Sigma_w)
            port_vol = np.sqrt(max(port_var, 1e-12))
            trc = w * Sigma_w / port_vol
            target_trc = port_vol / n

            # Gradient descent step
            grad = trc - target_trc
            if np.linalg.norm(grad) < tol:
                break

            # Step: cyclical update for each asset
            for i in range(n):
                # Quadratic in w_i: Sigma[i,i]*w_i^2 + sum_{j!=i}Sigma[i,j]*w_j*w_i = target*port_vol
                a = Sigma[i, i]
                b = float(Sigma[i, :] @ w) - Sigma[i, i] * w[i]
                # Solve: a*w_i^2 + b*w_i - target_trc*port_vol = 0 → ignore port_vol update
                # Simple gradient update
                w[i] = max(w[i] - 0.1 * grad[i], 1e-6)

            w = w / w.sum()

        return w

    # ------------------------------------------------------------------ #
    # Portfolio Performance Summary                                         #
    # ------------------------------------------------------------------ #

    def performance_summary(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series = None,
        risk_free_rate: float = 0.05,
    ) -> dict:
        """
        Comprehensive risk-adjusted performance metrics.
        """
        r = returns.dropna()
        if len(r) < 20:
            return {"error": "insufficient_data"}

        mu_ann   = float(r.mean() * TRADING_DAYS_PER_YEAR)
        vol_ann  = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
        excess   = r - rf_daily

        sharpe   = float(excess.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if vol_ann > 0 else None

        # Sortino (downside deviation)
        downside = r[r < rf_daily]
        downside_vol = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(downside) > 1 else vol_ann
        sortino  = (mu_ann - risk_free_rate) / downside_vol if downside_vol > 0 else None

        # Calmar
        cum = (1 + r).cumprod()
        roll_max = cum.cummax()
        dd = (cum - roll_max) / roll_max
        max_dd = float(dd.min())
        calmar = mu_ann / abs(max_dd) if max_dd < 0 else None

        # VaR metrics (all three methods)
        var_hist = self.compute_var(r, method="historical")
        var_param = self.compute_var(r, method="parametric")
        var_cf   = self.compute_var(r, method="cornish_fisher")

        result: dict[str, Any] = {
            "return_annualized": round(mu_ann, 4),
            "volatility_annualized": round(vol_ann, 4),
            "sharpe_ratio": round(sharpe, 4) if sharpe else None,
            "sortino_ratio": round(sortino, 4) if sortino else None,
            "calmar_ratio": round(calmar, 4) if calmar else None,
            "max_drawdown": round(max_dd, 4),
            "skewness": round(float(stats.skew(r)), 4),
            "excess_kurtosis": round(float(stats.kurtosis(r)), 4),
            "var_95_historical": var_hist.get("var_daily"),
            "var_95_parametric": var_param.get("var_daily"),
            "var_95_cornish_fisher": var_cf.get("var_daily"),
            "cvar_95_historical": var_hist.get("cvar_daily"),
            "n_observations": len(r),
        }

        if benchmark_returns is not None:
            te = self.compute_tracking_error(r, benchmark_returns)
            result.update(
                {
                    "tracking_error": te.get("tracking_error_annualized"),
                    "information_ratio": te.get("information_ratio"),
                    "beta_to_benchmark": te.get("beta_to_benchmark"),
                    "correlation_to_benchmark": te.get("correlation_to_benchmark"),
                }
            )

        return result


# ---------------------------------------------------------------------------
# Convenience facade: RiskAnalytics
# ---------------------------------------------------------------------------

class RiskAnalytics:
    """
    Unified interface combining FamaFrench factors, factor risk model,
    and portfolio risk engine.
    """

    def __init__(self):
        self.ff   = FamaFrenchAdapter()
        self.frm  = FactorRiskModel()
        self.pre  = PortfolioRiskEngine()

    def full_factor_report(
        self,
        returns: pd.Series,
        start: str = "2010-01-01",
    ) -> dict:
        """
        Run complete factor risk report for a single return series.

        Downloads FF5 + momentum, estimates loadings, decomposes risk.
        """
        factors = self.ff.get_combined_factors(start=start, include_momentum=True)
        if factors.empty:
            return {"error": "factor_data_unavailable"}

        # Align
        combined = pd.concat([returns.rename("ret"), factors], axis=1).dropna()
        if combined.empty:
            return {"error": "no_aligned_data"}

        ret_aligned = combined["ret"]
        fac_aligned = combined.drop(columns=["ret"])

        # Factor loadings
        loadings = self.frm.estimate_factor_loadings(ret_aligned, fac_aligned)

        # Attribution
        attribution = self.frm.factor_attribution(ret_aligned, fac_aligned)

        # Risk decomposition
        factor_cols = [c for c in fac_aligned.columns if c != "RF"]
        fac_cov = fac_aligned[factor_cols].cov()
        betas   = loadings.get("betas", {})
        idio_var = loadings.get("residual_vol_annualized", 0.0) ** 2 / TRADING_DAYS_PER_YEAR
        risk_decomp = self.frm.factor_risk_decomposition(betas, fac_cov, idio_var)

        # Rolling loadings (last 60 days per window)
        rolling = self.frm.rolling_factor_loadings(ret_aligned, fac_aligned, window=60)

        return {
            "factor_loadings": loadings,
            "factor_attribution": {
                k: v for k, v in attribution.items()
                if k != "attribution_df"
            },
            "risk_decomposition": risk_decomp,
            "rolling_loadings_tail": rolling.tail(20).to_dict(orient="index"),
        }

    def portfolio_risk_report(
        self,
        weights: dict[str, float],
        returns_dict: dict[str, pd.Series],
        benchmark_returns: pd.Series = None,
    ) -> dict:
        """
        Full portfolio risk report: VaR, stress test, covariance, risk contributions.
        """
        cov = self.pre.covariance_matrix(returns_dict, method="ledoit_wolf")
        var_report  = self.pre.compute_portfolio_var(weights, returns_dict)
        stress      = self.pre.stress_test(weights, returns_dict)
        risk_contribs = (
            self.pre.risk_contribution(weights, cov) if not cov.empty else pd.DataFrame()
        )

        # Portfolio performance
        ret_df = pd.DataFrame(returns_dict).dropna()
        w = np.array([weights.get(t, 0.0) for t in ret_df.columns])
        w = w / w.sum() if w.sum() > 0 else w
        port_ret = pd.Series(ret_df.values @ w, index=ret_df.index)

        perf = self.pre.performance_summary(port_ret, benchmark_returns)

        return {
            "performance": perf,
            "var_report": var_report,
            "stress_test": stress,
            "risk_contributions": risk_contribs.to_dict(orient="records") if not risk_contribs.empty else [],
            "covariance_shape": list(cov.shape) if not cov.empty else None,
        }
