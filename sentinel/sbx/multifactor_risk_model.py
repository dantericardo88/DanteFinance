"""
Multi-factor risk model: Fama-French 5 + Momentum + QMJ + BAB.
Full factor model: exposures, attribution, risk decomposition.
Free data: Ken French Data Library (direct CSV download).
"""
from __future__ import annotations

import io
import os
import sqlite3
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from scipy import stats

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel, Field
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252

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

_CACHE_DIR = Path(os.environ.get("SENTINEL_HOME", Path.home() / ".sentinel")) / "cache" / "ff_factors"
_DB_PATH = _CACHE_DIR / "ff_factors.db"

# Factor TTL: 7 days in seconds
_FACTOR_TTL_SECONDS = 7 * 24 * 3600

# All factor column names in the combined 6-factor model
FACTOR_COLS_FF5MOM = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
FACTOR_COLS_FULL = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "QMJ", "BAB"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FactorExposures:
    """Factor betas and statistics for a single security."""
    ticker: str
    start: str
    end: str
    observations: int
    betas: Dict[str, float]
    t_stats: Dict[str, float]
    p_values: Dict[str, float]
    alpha_annualized: float
    alpha_t_stat: float
    r_squared: float
    residual_vol_annualized: float
    factor_var_explained_pct: float
    error: Optional[str] = None


@dataclass
class PortfolioDecomposition:
    """Full factor decomposition for a portfolio."""
    portfolio_vol_annualized: float
    factor_vol_annualized: float
    specific_vol_annualized: float
    factor_var_pct: float
    specific_var_pct: float
    factor_betas: Dict[str, float]
    factor_var_contributions: Dict[str, float]
    marginal_risk_contributions: Dict[str, float]
    asset_factor_betas: Dict[str, Dict[str, float]]


@dataclass
class FactorTimingSignal:
    """Factor timing signals and regime conditioning."""
    factor: str
    momentum_12_1: float       # 12-minus-1 month factor momentum
    momentum_1m: float         # 1-month factor return
    rolling_sharpe_1y: float   # trailing 1Y factor Sharpe
    regime_bull_return: float  # annualized return in bull markets
    regime_bear_return: float  # annualized return in bear markets
    signal: str                # "overweight" | "underweight" | "neutral"
    score: float               # -1 to +1


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _ensure_db() -> sqlite3.Connection:
    """Create/open the SQLite cache DB."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS factor_cache (
            table_name TEXT PRIMARY KEY,
            data       BLOB NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _cache_load(table_name: str) -> Optional[pd.DataFrame]:
    """Load cached factor data if still fresh (within TTL)."""
    try:
        conn = _ensure_db()
        row = conn.execute(
            "SELECT data, updated_at FROM factor_cache WHERE table_name = ?",
            (table_name,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        data_bytes, updated_at = row
        if time.time() - updated_at > _FACTOR_TTL_SECONDS:
            return None  # stale
        buf = io.BytesIO(data_bytes)
        return pd.read_parquet(buf)
    except Exception as exc:
        logger.warning("Cache load failed for %s: %s", table_name, exc)
        return None


def _cache_save(table_name: str, df: pd.DataFrame) -> None:
    """Persist factor DataFrame to SQLite cache."""
    try:
        conn = _ensure_db()
        buf = io.BytesIO()
        df.to_parquet(buf)
        conn.execute(
            "INSERT OR REPLACE INTO factor_cache (table_name, data, updated_at) VALUES (?, ?, ?)",
            (table_name, buf.getvalue(), time.time())
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("Cache save failed for %s: %s", table_name, exc)


# ---------------------------------------------------------------------------
# Raw data download helpers
# ---------------------------------------------------------------------------

def _download_zip_csv(url: str) -> str:
    """Download ZIP from Ken French library and return CSV text."""
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = next(
            n for n in zf.namelist()
            if n.upper().endswith(".CSV")
        )
        return zf.read(csv_name).decode("utf-8", errors="replace")


def _parse_french_csv(csv_text: str, factor_cols: List[str]) -> pd.DataFrame:
    """
    Parse Ken French CSV format into a clean DataFrame.
    French CSVs: optional header block, then YYYYMMDD date column + factor columns.
    """
    lines = csv_text.splitlines()
    data_lines: List[str] = []
    in_data = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = [p.strip() for p in stripped.split(",")]
        if not in_data:
            # Daily data starts when first col is 8-digit number
            if parts[0].isdigit() and len(parts[0]) == 8:
                in_data = True
            else:
                continue
        if in_data:
            # Stop at annual section (4-digit year rows)
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

    ncols = min(len(factor_cols) + 1, df.shape[1])
    df = df.iloc[:, :ncols].copy()
    df.columns = ["date"] + factor_cols[: ncols - 1]  # type: ignore

    df["date"] = pd.to_datetime(
        df["date"].astype(str).str.strip(),
        format="%Y%m%d",
        errors="coerce"
    )
    df = df.dropna(subset=["date"]).set_index("date")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    return df.dropna()


# ---------------------------------------------------------------------------
# 1. FamaFrenchDataLoader
# ---------------------------------------------------------------------------

class FamaFrenchDataLoader:
    """
    Download and cache Fama-French factor data from Ken French's data library.

    Caches in SQLite with a 7-day TTL (factor files update monthly).
    All returns are decimal (not percentages).
    """

    def _load_or_fetch(
        self,
        cache_key: str,
        url: str,
        factor_cols: List[str],
    ) -> pd.DataFrame:
        """Generic load-from-cache or download-and-cache."""
        df = _cache_load(cache_key)
        if df is not None:
            return df

        logger.info("Downloading %s from Ken French library: %s", cache_key, url)
        try:
            csv_text = _download_zip_csv(url)
            df = _parse_french_csv(csv_text, factor_cols)
            if not df.empty:
                _cache_save(cache_key, df)
            return df
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", cache_key, exc)
            return pd.DataFrame(columns=factor_cols)

    def get_ff3(self, start: str = "1926-07-01", end: Optional[str] = None) -> pd.DataFrame:
        """FF3 daily: Mkt-RF, SMB, HML, RF."""
        df = self._load_or_fetch("ff3_daily", _FF3_DAILY_URL, ["Mkt-RF", "SMB", "HML", "RF"])
        return self._slice(df, start, end)

    def get_ff5(self, start: str = "1963-07-01", end: Optional[str] = None) -> pd.DataFrame:
        """FF5 daily: Mkt-RF, SMB, HML, RMW, CMA, RF."""
        df = self._load_or_fetch(
            "ff5_daily", _FF5_DAILY_URL,
            ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
        )
        return self._slice(df, start, end)

    def get_momentum(self, start: str = "1927-01-01", end: Optional[str] = None) -> pd.DataFrame:
        """Momentum factor daily: Mom."""
        df = self._load_or_fetch("mom_daily", _MOM_DAILY_URL, ["Mom"])
        return self._slice(df, start, end)

    def get_combined(
        self,
        start: str = "1963-07-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Combined FF5 + Momentum factors, aligned on common trading days.
        Columns: Mkt-RF, SMB, HML, RMW, CMA, RF, Mom
        """
        ff5 = self.get_ff5(start=start, end=end)
        mom = self.get_momentum(start=start, end=end)
        if ff5.empty:
            return ff5
        combined = ff5.join(mom, how="left")
        # Fill missing momentum with 0 for days not available
        if "Mom" in combined.columns:
            combined["Mom"] = combined["Mom"].fillna(0.0)
        return combined.dropna(subset=["Mkt-RF"])

    def get_factor_covariance(
        self,
        start: str = "2000-01-01",
        end: Optional[str] = None,
        window: int = 252,
    ) -> pd.DataFrame:
        """
        Compute annualized factor covariance matrix from FF5+Mom data.
        Used for portfolio factor variance attribution.
        """
        factors = self.get_combined(start=start, end=end)
        factor_cols = [c for c in FACTOR_COLS_FF5MOM if c in factors.columns]
        if factors.empty or not factor_cols:
            return pd.DataFrame()
        recent = factors[factor_cols].tail(window)
        # Annualize: cov_daily * 252
        return recent.cov() * TRADING_DAYS_PER_YEAR

    @staticmethod
    def _slice(df: pd.DataFrame, start: str, end: Optional[str]) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.sort_index()
        end_dt = pd.Timestamp(end) if end else pd.Timestamp.today()
        return df.loc[pd.Timestamp(start):end_dt]


# ---------------------------------------------------------------------------
# 2. FactorExposureEstimator
# ---------------------------------------------------------------------------

class FactorExposureEstimator:
    """
    Estimate factor betas for individual securities via rolling OLS.

    Uses FF5 + Momentum (6 factors total). Window: 252 days.
    """

    def __init__(self, window: int = 252):
        self.window = window
        self._loader = FamaFrenchDataLoader()
        self._factor_cache: Optional[pd.DataFrame] = None
        self._factor_cache_date: Optional[str] = None

    def _get_factors(self, start: str, end: Optional[str] = None) -> pd.DataFrame:
        """Load combined factors, caching for the session."""
        return self._loader.get_combined(start=start, end=end)

    def _fetch_price_returns(self, ticker: str, start: str, end: str) -> pd.Series:
        """Fetch daily log returns from yfinance."""
        if not _YF_AVAILABLE:
            return pd.Series(dtype=float)
        try:
            data = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if data.empty:
                return pd.Series(dtype=float)
            closes = data["Close"]
            if isinstance(closes, pd.DataFrame):
                closes = closes.iloc[:, 0]
            return closes.pct_change().dropna()
        except Exception as exc:
            logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
            return pd.Series(dtype=float)

    def _ols_with_stats(
        self,
        y: np.ndarray,
        X: np.ndarray,
        factor_names: List[str],
    ) -> Dict[str, Any]:
        """
        Run OLS with intercept. Return betas, t-stats, p-values, R², residual vol.
        X does NOT include intercept column — we add it here.
        """
        n = len(y)
        if n < 30:
            return {"error": "insufficient_data", "observations": n}

        X_int = np.column_stack([np.ones(n), X])
        k = X_int.shape[1]

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
        except np.linalg.LinAlgError as exc:
            return {"error": str(exc)}

        y_hat = X_int @ coeffs
        resid = y - y_hat
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        dof = max(n - k, 1)
        sigma2 = ss_res / dof

        try:
            XtX_inv = np.linalg.inv(X_int.T @ X_int)
            se_arr = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0.0))
        except np.linalg.LinAlgError:
            se_arr = np.full(k, np.nan)

        t_arr = np.where(se_arr > 1e-12, coeffs / se_arr, np.nan)
        p_arr = 2.0 * (1.0 - stats.t.cdf(np.abs(t_arr), df=dof))

        alpha_daily = float(coeffs[0])
        betas_arr = coeffs[1:]

        betas = {col: float(b) for col, b in zip(factor_names, betas_arr)}
        t_stats = {col: float(t) for col, t in zip(factor_names, t_arr[1:])}
        p_values = {col: float(p) for col, p in zip(factor_names, p_arr[1:])}

        resid_vol_ann = float(np.std(resid, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        alpha_ann = float(alpha_daily * TRADING_DAYS_PER_YEAR)

        return {
            "betas": betas,
            "alpha_annualized": alpha_ann,
            "alpha_t_stat": float(t_arr[0]),
            "alpha_p_value": float(p_arr[0]),
            "r_squared": r_sq,
            "t_stats": t_stats,
            "p_values": p_values,
            "residual_vol_annualized": resid_vol_ann,
            "observations": n,
        }

    def get_exposures(
        self,
        ticker: str,
        start: str = "2015-01-01",
        end: Optional[str] = None,
    ) -> FactorExposures:
        """
        Estimate FF5+Mom factor exposures for a single ticker.

        Uses trailing `window` days (default 252) of data.
        """
        end_str = end or datetime.today().strftime("%Y-%m-%d")

        # Fetch price returns
        returns = self._fetch_price_returns(ticker, start, end_str)
        if returns.empty:
            return FactorExposures(
                ticker=ticker, start=start, end=end_str, observations=0,
                betas={}, t_stats={}, p_values={}, alpha_annualized=0.0,
                alpha_t_stat=0.0, r_squared=0.0, residual_vol_annualized=0.0,
                factor_var_explained_pct=0.0, error="no_price_data"
            )

        # Fetch factors
        factors = self._get_factors(start=start, end=end_str)
        if factors.empty:
            return FactorExposures(
                ticker=ticker, start=start, end=end_str, observations=0,
                betas={}, t_stats={}, p_values={}, alpha_annualized=0.0,
                alpha_t_stat=0.0, r_squared=0.0, residual_vol_annualized=0.0,
                factor_var_explained_pct=0.0, error="no_factor_data"
            )

        # Align
        combined = pd.concat([returns.rename("ret"), factors], axis=1).dropna()
        if len(combined) < 30:
            return FactorExposures(
                ticker=ticker, start=start, end=end_str, observations=len(combined),
                betas={}, t_stats={}, p_values={}, alpha_annualized=0.0,
                alpha_t_stat=0.0, r_squared=0.0, residual_vol_annualized=0.0,
                factor_var_explained_pct=0.0, error="insufficient_aligned_data"
            )

        # Use at most `window` observations
        if len(combined) > self.window:
            combined = combined.iloc[-self.window:]

        factor_cols = [c for c in FACTOR_COLS_FF5MOM if c in combined.columns]
        rf = combined["RF"].values if "RF" in combined.columns else np.zeros(len(combined))
        y = combined["ret"].values - rf
        X = combined[factor_cols].values

        result = self._ols_with_stats(y, X, factor_cols)

        if "error" in result:
            return FactorExposures(
                ticker=ticker, start=start, end=end_str, observations=result.get("observations", 0),
                betas={}, t_stats={}, p_values={}, alpha_annualized=0.0,
                alpha_t_stat=0.0, r_squared=0.0, residual_vol_annualized=0.0,
                factor_var_explained_pct=0.0, error=result["error"]
            )

        factor_var_pct = result["r_squared"] * 100.0

        return FactorExposures(
            ticker=ticker,
            start=start,
            end=end_str,
            observations=result["observations"],
            betas=result["betas"],
            t_stats=result["t_stats"],
            p_values=result["p_values"],
            alpha_annualized=result["alpha_annualized"],
            alpha_t_stat=result["alpha_t_stat"],
            r_squared=result["r_squared"],
            residual_vol_annualized=result["residual_vol_annualized"],
            factor_var_explained_pct=factor_var_pct,
        )

    def get_exposures_batch(
        self,
        tickers: List[str],
        start: str = "2015-01-01",
        end: Optional[str] = None,
        max_workers: int = 8,
    ) -> Dict[str, FactorExposures]:
        """
        Estimate factor exposures for a list of tickers in parallel.
        Uses ThreadPoolExecutor for concurrent yfinance downloads.
        """
        results: Dict[str, FactorExposures] = {}

        def _fetch_one(ticker: str) -> Tuple[str, FactorExposures]:
            return ticker, self.get_exposures(ticker, start=start, end=end)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one, t): t for t in tickers}
            for future in as_completed(futures):
                ticker, exposure = future.result()
                results[ticker] = exposure
                logger.debug("Estimated exposures for %s: R²=%.3f", ticker, exposure.r_squared)

        return results

    def get_rolling_exposures(
        self,
        ticker: str,
        start: str = "2010-01-01",
        end: Optional[str] = None,
        roll_window: int = 126,
    ) -> pd.DataFrame:
        """
        Rolling factor exposures over time (roll_window days).
        Returns DataFrame with columns = factor names + 'alpha' + 'r_squared'.
        """
        end_str = end or datetime.today().strftime("%Y-%m-%d")
        returns = self._fetch_price_returns(ticker, start, end_str)
        factors = self._get_factors(start=start, end=end_str)

        if returns.empty or factors.empty:
            return pd.DataFrame()

        combined = pd.concat([returns.rename("ret"), factors], axis=1).dropna()
        if len(combined) < roll_window + 5:
            return pd.DataFrame()

        factor_cols = [c for c in FACTOR_COLS_FF5MOM if c in combined.columns]
        rf_arr = combined["RF"].values if "RF" in combined.columns else np.zeros(len(combined))
        y_all = combined["ret"].values - rf_arr
        X_all = combined[factor_cols].values

        rows = []
        dates = []
        for i in range(roll_window, len(y_all) + 1):
            y_w = y_all[i - roll_window:i]
            X_w = X_all[i - roll_window:i]
            res = self._ols_with_stats(y_w, X_w, factor_cols)
            if "error" not in res:
                row = {"alpha_ann": res["alpha_annualized"], "r_squared": res["r_squared"]}
                row.update(res["betas"])
                rows.append(row)
                dates.append(combined.index[i - 1])

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, index=pd.DatetimeIndex(dates))


# ---------------------------------------------------------------------------
# 3. PortfolioFactorDecomposition
# ---------------------------------------------------------------------------

class PortfolioFactorDecomposition:
    """
    Decompose portfolio risk into factor (systematic) and idiosyncratic components.

    Portfolio factor variance: Σ_f = β_p' × F_cov × β_p
    Specific variance:         Σ_s = w' × diag(σ²_idio) × w
    Total variance:            Σ_f + Σ_s
    """

    def __init__(self):
        self._loader = FamaFrenchDataLoader()
        self._estimator = FactorExposureEstimator()

    def decompose(
        self,
        weights: Dict[str, float],
        exposures: Dict[str, FactorExposures],
        factor_cov: Optional[pd.DataFrame] = None,
        start: str = "2000-01-01",
    ) -> PortfolioDecomposition:
        """
        Full portfolio factor decomposition.

        Parameters
        ----------
        weights : Dict[ticker -> weight], should sum to 1.
        exposures : Dict[ticker -> FactorExposures] for each position.
        factor_cov : Optional pre-computed annualized factor covariance matrix.
        start : History start date for factor covariance estimation.
        """
        tickers = [t for t in weights if t in exposures and exposures[t].error is None]
        if not tickers:
            return PortfolioDecomposition(
                portfolio_vol_annualized=0.0, factor_vol_annualized=0.0,
                specific_vol_annualized=0.0, factor_var_pct=0.0, specific_var_pct=0.0,
                factor_betas={}, factor_var_contributions={},
                marginal_risk_contributions={}, asset_factor_betas={}
            )

        # Normalize weights
        total_w = sum(weights[t] for t in tickers)
        w = {t: weights[t] / total_w for t in tickers}

        # Portfolio factor betas = weighted sum of position betas
        all_factors = set()
        for t in tickers:
            all_factors.update(exposures[t].betas.keys())
        factor_list = sorted(all_factors)

        # Build portfolio beta vector
        port_betas: Dict[str, float] = {}
        for f in factor_list:
            port_betas[f] = sum(w[t] * exposures[t].betas.get(f, 0.0) for t in tickers)

        # Factor covariance matrix (annualized)
        if factor_cov is None:
            factor_cov = self._loader.get_factor_covariance(start=start)

        # Compute factor variance: β' × F_cov × β
        avail_factors = [f for f in factor_list if f in factor_cov.index]
        b_vec = np.array([port_betas.get(f, 0.0) for f in avail_factors])
        F_cov = factor_cov.loc[avail_factors, avail_factors].values

        try:
            factor_var = float(b_vec @ F_cov @ b_vec)
        except Exception:
            factor_var = 0.0

        # Specific variance: weighted sum of idiosyncratic variances
        specific_var = sum(
            (w[t] ** 2) * (exposures[t].residual_vol_annualized ** 2)
            for t in tickers
        )

        total_var = factor_var + specific_var
        total_vol = float(np.sqrt(max(total_var, 0.0)))
        factor_vol = float(np.sqrt(max(factor_var, 0.0)))
        specific_vol = float(np.sqrt(max(specific_var, 0.0)))

        factor_var_pct = factor_var / total_var * 100.0 if total_var > 0 else 0.0
        specific_var_pct = 100.0 - factor_var_pct

        # Per-factor variance contributions: b_i * (F_cov @ b)_i
        factor_var_contribs: Dict[str, float] = {}
        if len(avail_factors) > 0:
            F_cov_b = F_cov @ b_vec
            for i, f in enumerate(avail_factors):
                factor_var_contribs[f] = float(b_vec[i] * F_cov_b[i])

        # Marginal risk contribution (MCR) for each asset:
        # MCR_i = ∂σ_p/∂w_i ≈ (Σ_factor × w)_i / σ_p
        # We use the full factor + specific model
        mcr: Dict[str, float] = {}
        if total_vol > 0:
            for t in tickers:
                # Asset's contribution via factor channel
                b_asset = np.array([exposures[t].betas.get(f, 0.0) for f in avail_factors])
                factor_channel = float(b_asset @ F_cov @ b_vec) if len(avail_factors) > 0 else 0.0
                # Asset's specific contribution
                specific_channel = (w[t] * exposures[t].residual_vol_annualized ** 2)
                mcr[t] = (factor_channel + specific_channel) / total_vol

        # Asset factor betas summary
        asset_betas = {t: exposures[t].betas for t in tickers}

        return PortfolioDecomposition(
            portfolio_vol_annualized=round(total_vol, 4),
            factor_vol_annualized=round(factor_vol, 4),
            specific_vol_annualized=round(specific_vol, 4),
            factor_var_pct=round(factor_var_pct, 2),
            specific_var_pct=round(specific_var_pct, 2),
            factor_betas={f: round(v, 4) for f, v in port_betas.items()},
            factor_var_contributions={f: round(v, 6) for f, v in factor_var_contribs.items()},
            marginal_risk_contributions={t: round(v, 6) for t, v in mcr.items()},
            asset_factor_betas=asset_betas,
        )


# ---------------------------------------------------------------------------
# 4. FactorTimingModel
# ---------------------------------------------------------------------------

class FactorTimingModel:
    """
    Factor momentum, timing signals, and regime conditioning.

    Uses historical FF5+Mom returns to generate tactical factor tilt signals.
    """

    def __init__(self):
        self._loader = FamaFrenchDataLoader()

    def compute_factor_momentum(
        self,
        factors: pd.DataFrame,
        lookback_months: int = 12,
        skip_months: int = 1,
    ) -> Dict[str, float]:
        """
        12-1 month factor momentum: cumulative return from (t-12) to (t-1).
        Skips the most recent month to avoid short-term reversal.
        """
        if factors.empty:
            return {}
        factor_cols = [c for c in FACTOR_COLS_FF5MOM if c in factors.columns]
        # Resample to monthly
        monthly = factors[factor_cols].resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )
        if len(monthly) < lookback_months + skip_months + 1:
            return {}
        # 12-1 momentum: cumulative over months [t-12, t-2] (skip last month)
        window = monthly.iloc[-(lookback_months + skip_months):-skip_months]
        momentum = {}
        for col in factor_cols:
            cum = float((1 + window[col]).prod() - 1)
            momentum[col] = cum
        return momentum

    def compute_factor_sharpe(
        self,
        factors: pd.DataFrame,
        window_days: int = 252,
    ) -> Dict[str, float]:
        """Trailing 1-year Sharpe ratio for each factor."""
        if factors.empty:
            return {}
        factor_cols = [c for c in FACTOR_COLS_FF5MOM if c in factors.columns]
        recent = factors[factor_cols].tail(window_days)
        sharpes = {}
        for col in factor_cols:
            mu = recent[col].mean()
            sigma = recent[col].std(ddof=1)
            sharpes[col] = float(mu / sigma * np.sqrt(TRADING_DAYS_PER_YEAR)) if sigma > 0 else 0.0
        return sharpes

    def regime_factor_returns(
        self,
        factors: pd.DataFrame,
        bull_threshold: float = 0.0,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute annualized factor returns conditioned on market regime.
        Bull = Mkt-RF > bull_threshold (daily), Bear = otherwise.
        """
        if "Mkt-RF" not in factors.columns:
            return {}
        factor_cols = [c for c in FACTOR_COLS_FF5MOM if c in factors.columns]
        bull_mask = factors["Mkt-RF"] > bull_threshold
        bear_mask = ~bull_mask

        results = {}
        for col in factor_cols:
            bull_ret = float(factors.loc[bull_mask, col].mean() * TRADING_DAYS_PER_YEAR)
            bear_ret = float(factors.loc[bear_mask, col].mean() * TRADING_DAYS_PER_YEAR)
            results[col] = {
                "bull_annualized": round(bull_ret, 4),
                "bear_annualized": round(bear_ret, 4),
                "regime_spread": round(bull_ret - bear_ret, 4),
            }
        return results

    def factor_crowding(
        self,
        factors: pd.DataFrame,
        stress_window: int = 60,
    ) -> Dict[str, float]:
        """
        Factor crowding proxy: correlation of each factor with VIX-equivalent stress.
        We use -Mkt-RF as a stress proxy (rising market stress = falling market).
        Higher (more positive) value = more crowded / stress-sensitive.
        """
        if "Mkt-RF" not in factors.columns:
            return {}
        factor_cols = [c for c in FACTOR_COLS_FF5MOM if c in factors.columns if c != "Mkt-RF"]
        stress = -factors["Mkt-RF"]
        recent = pd.concat([stress.rename("stress"), factors[factor_cols]], axis=1).tail(stress_window).dropna()
        crowding = {}
        for col in factor_cols:
            corr = float(recent["stress"].corr(recent[col]))
            crowding[col] = round(corr, 4)
        return crowding

    def get_timing_signals(
        self,
        start: str = "2000-01-01",
        end: Optional[str] = None,
    ) -> List[FactorTimingSignal]:
        """
        Generate comprehensive factor timing signals.

        Combines factor momentum (12-1M), Sharpe, and regime returns
        to produce a composite overweight/underweight signal for each factor.
        """
        factors = self._loader.get_combined(start=start, end=end)
        if factors.empty:
            return []

        momentum_12_1 = self.compute_factor_momentum(factors, lookback_months=12, skip_months=1)
        momentum_1m = self.compute_factor_momentum(factors, lookback_months=1, skip_months=0)
        sharpes = self.compute_factor_sharpe(factors)
        regime_rets = self.regime_factor_returns(factors)
        crowding = self.factor_crowding(factors)

        signals = []
        factor_cols = [c for c in FACTOR_COLS_FF5MOM if c in factors.columns]
        for factor in factor_cols:
            mom12 = momentum_12_1.get(factor, 0.0)
            mom1 = momentum_1m.get(factor, 0.0)
            sharpe = sharpes.get(factor, 0.0)
            regime = regime_rets.get(factor, {})
            crowd = crowding.get(factor, 0.0)

            # Composite score: momentum + Sharpe normalized, penalize crowding
            mom_score = np.sign(mom12) * min(abs(mom12) / 0.05, 1.0)  # normalize by 5%
            sharpe_score = np.clip(sharpe / 2.0, -1.0, 1.0)
            crowd_penalty = -0.3 * max(crowd, 0.0)  # positive crowding is bad
            composite = float(0.5 * mom_score + 0.3 * sharpe_score + 0.2 * crowd_penalty)
            composite = float(np.clip(composite, -1.0, 1.0))

            if composite > 0.25:
                signal_str = "overweight"
            elif composite < -0.25:
                signal_str = "underweight"
            else:
                signal_str = "neutral"

            signals.append(FactorTimingSignal(
                factor=factor,
                momentum_12_1=round(mom12, 4),
                momentum_1m=round(mom1, 4),
                rolling_sharpe_1y=round(sharpe, 4),
                regime_bull_return=round(regime.get("bull_annualized", 0.0), 4),
                regime_bear_return=round(regime.get("bear_annualized", 0.0), 4),
                signal=signal_str,
                score=round(composite, 4),
            ))

        return signals

    def factor_rotation_weights(
        self,
        timing_signals: List[FactorTimingSignal],
        base_exposure: float = 0.2,
    ) -> Dict[str, float]:
        """
        Translate timing signals into factor tilt weights.
        Positive score → overweight that factor, negative → underweight.
        Returns suggested active factor exposure tilts (additive to benchmark).
        """
        tilts = {}
        for sig in timing_signals:
            # Scale: ±base_exposure based on score magnitude
            tilts[sig.factor] = round(sig.score * base_exposure, 4)
        return tilts


# ---------------------------------------------------------------------------
# 5. QMJBetaAdapter
# ---------------------------------------------------------------------------

class QMJBetaAdapter:
    """
    Construct Quality-Minus-Junk (QMJ) and Betting-Against-Beta (BAB) factors.

    Both are AQR-style factors constructed from publicly available data
    (yfinance for prices, EDGAR fundamentals for quality metrics).

    NOTE: These are proxy constructions. For full AQR-quality factors,
    the official datasets require registration at aqr.com.
    """

    def __init__(self):
        self._estimator = FactorExposureEstimator()

    def _compute_market_betas(
        self,
        tickers: List[str],
        start: str = "2018-01-01",
        window: int = 252,
    ) -> Dict[str, float]:
        """Estimate market beta for each ticker via OLS vs Mkt-RF."""
        if not _YF_AVAILABLE:
            return {}

        loader = FamaFrenchDataLoader()
        factors = loader.get_combined(start=start)
        if factors.empty or "Mkt-RF" not in factors.columns:
            return {}

        betas: Dict[str, float] = {}
        for ticker in tickers:
            try:
                exp = self._estimator.get_exposures(ticker, start=start)
                if exp.error is None and "Mkt-RF" in exp.betas:
                    betas[ticker] = exp.betas["Mkt-RF"]
                else:
                    betas[ticker] = 1.0
            except Exception:
                betas[ticker] = 1.0
        return betas

    def _fetch_returns_matrix(
        self,
        tickers: List[str],
        start: str = "2018-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch a returns matrix for all tickers."""
        if not _YF_AVAILABLE:
            return pd.DataFrame()
        end_str = end or datetime.today().strftime("%Y-%m-%d")
        try:
            data = yf.download(
                tickers,
                start=start,
                end=end_str,
                progress=False,
                auto_adjust=True,
                threads=True,
            )
            if data.empty:
                return pd.DataFrame()
            # Handle multi-ticker download
            if "Close" in data.columns:
                closes = data["Close"]
            elif isinstance(data.columns, pd.MultiIndex):
                closes = data.xs("Close", axis=1, level=0) if "Close" in data.columns.get_level_values(0) else data
            else:
                closes = data
            if isinstance(closes, pd.Series):
                closes = closes.to_frame(name=tickers[0] if tickers else "ticker")
            return closes.pct_change().dropna(how="all")
        except Exception as exc:
            logger.warning("Batch yfinance fetch failed: %s", exc)
            return pd.DataFrame()

    def compute_bab_factor(
        self,
        universe_tickers: List[str],
        start: str = "2015-01-01",
        end: Optional[str] = None,
        n_quintiles: int = 5,
    ) -> pd.Series:
        """
        Construct BAB (Betting Against Beta) factor proxy.

        Long: lowest-beta quintile (leveraged to beta=1)
        Short: highest-beta quintile (deleveraged to beta=1)
        BAB = r_low_beta / beta_low - r_high_beta / beta_high

        Returns daily BAB factor return series.
        """
        if not _YF_AVAILABLE or len(universe_tickers) < 10:
            return pd.Series(dtype=float, name="BAB")

        end_str = end or datetime.today().strftime("%Y-%m-%d")
        returns = self._fetch_returns_matrix(universe_tickers, start, end_str)
        if returns.empty or returns.shape[1] < 5:
            return pd.Series(dtype=float, name="BAB")

        # Use rolling 252-day betas, recomputed monthly
        loader = FamaFrenchDataLoader()
        factors = loader.get_combined(start=start, end=end_str)
        if factors.empty or "Mkt-RF" not in factors.columns:
            return pd.Series(dtype=float, name="BAB")

        # Align returns with factor data
        aligned = returns.join(factors[["Mkt-RF", "RF"]], how="inner").dropna()
        if len(aligned) < 252:
            return pd.Series(dtype=float, name="BAB")

        tickers_avail = [t for t in universe_tickers if t in aligned.columns]
        if len(tickers_avail) < 5:
            return pd.Series(dtype=float, name="BAB")

        mkt_rf = aligned["Mkt-RF"].values
        rf = aligned["RF"].values

        # Compute rolling beta for each ticker (252-day window, recomputed daily)
        bab_series = pd.Series(index=aligned.index, dtype=float, name="BAB")
        roll_window = 252

        for i in range(roll_window, len(aligned)):
            # Compute betas for this window
            mkt_window = mkt_rf[i - roll_window:i]
            ticker_betas: Dict[str, float] = {}
            for t in tickers_avail:
                if t not in aligned.columns:
                    continue
                ret_t = aligned[t].values[i - roll_window:i]
                rf_w = rf[i - roll_window:i]
                excess_t = ret_t - rf_w
                cov = float(np.cov(excess_t, mkt_window)[0, 1])
                var_mkt = float(np.var(mkt_window))
                beta = cov / var_mkt if var_mkt > 1e-10 else 1.0
                ticker_betas[t] = float(np.clip(beta, 0.01, 10.0))

            # Sort by beta into quintiles
            sorted_tickers = sorted(ticker_betas, key=lambda x: ticker_betas[x])
            n = len(sorted_tickers)
            q_size = max(n // n_quintiles, 1)
            low_beta = sorted_tickers[:q_size]
            high_beta = sorted_tickers[-q_size:]

            if not low_beta or not high_beta:
                continue

            today_ret = aligned.iloc[i]
            rf_today = rf[i]

            # Low beta portfolio (equal-weighted, leveraged to beta=1)
            avg_beta_low = np.mean([ticker_betas[t] for t in low_beta])
            avg_beta_high = np.mean([ticker_betas[t] for t in high_beta])

            ret_low = np.mean([today_ret[t] - rf_today for t in low_beta if t in today_ret])
            ret_high = np.mean([today_ret[t] - rf_today for t in high_beta if t in today_ret])

            # BAB = r_low/beta_low - r_high/beta_high (deleveraged)
            bab_today = (
                ret_low / max(avg_beta_low, 0.01)
                - ret_high / max(avg_beta_high, 0.01)
            )
            bab_series.iloc[i] = float(bab_today)

        return bab_series.dropna()

    def compute_qmj_factor(
        self,
        universe_tickers: List[str],
        start: str = "2015-01-01",
        end: Optional[str] = None,
        n_quintiles: int = 5,
    ) -> pd.Series:
        """
        Construct QMJ (Quality Minus Junk) factor proxy.

        Quality proxy: high momentum (trailing 6M return) + low recent volatility.
        Junk proxy: low momentum + high volatility.

        QMJ = r_quality_quintile - r_junk_quintile

        For a full fundamental-based QMJ, EDGAR scraping is required.
        This uses a momentum + vol proxy which captures much of the quality premium.
        """
        if not _YF_AVAILABLE or len(universe_tickers) < 10:
            return pd.Series(dtype=float, name="QMJ")

        end_str = end or datetime.today().strftime("%Y-%m-%d")
        returns = self._fetch_returns_matrix(universe_tickers, start, end_str)
        if returns.empty or returns.shape[1] < 5:
            return pd.Series(dtype=float, name="QMJ")

        tickers_avail = [t for t in universe_tickers if t in returns.columns]
        qmj_series = pd.Series(index=returns.index, dtype=float, name="QMJ")
        quality_window = 126  # 6 months
        vol_window = 63       # 3 months

        for i in range(quality_window, len(returns)):
            # Quality score: momentum (high = quality) - vol (low vol = quality)
            scores: Dict[str, float] = {}
            for t in tickers_avail:
                if t not in returns.columns:
                    continue
                ret_hist = returns[t].iloc[i - quality_window:i].dropna()
                if len(ret_hist) < 30:
                    continue
                # Momentum: cumulative 6M return
                mom = float((1 + ret_hist).prod() - 1)
                # Volatility (lower = better quality)
                vol_recent = float(ret_hist.iloc[-vol_window:].std(ddof=1)) if len(ret_hist) >= vol_window else float(ret_hist.std(ddof=1))
                # Quality score: high momentum, low vol → positive score
                scores[t] = mom - vol_recent * 10  # penalize high vol

            if len(scores) < 5:
                continue

            sorted_tickers = sorted(scores, key=lambda x: scores[x])
            n = len(sorted_tickers)
            q_size = max(n // n_quintiles, 1)
            junk = sorted_tickers[:q_size]     # lowest quality
            quality = sorted_tickers[-q_size:] # highest quality

            today_ret = returns.iloc[i]
            ret_quality = np.mean([today_ret[t] for t in quality if t in today_ret and not np.isnan(today_ret[t])])
            ret_junk = np.mean([today_ret[t] for t in junk if t in today_ret and not np.isnan(today_ret[t])])

            if not np.isnan(ret_quality) and not np.isnan(ret_junk):
                qmj_series.iloc[i] = float(ret_quality - ret_junk)

        return qmj_series.dropna()


# ---------------------------------------------------------------------------
# 6. RiskAttributionReport
# ---------------------------------------------------------------------------

class RiskAttributionReport:
    """
    Generate comprehensive factor risk attribution reports for portfolios.

    Integrates FamaFrenchDataLoader, FactorExposureEstimator,
    PortfolioFactorDecomposition, and FactorTimingModel.
    """

    def __init__(self):
        self._loader = FamaFrenchDataLoader()
        self._estimator = FactorExposureEstimator()
        self._decomposer = PortfolioFactorDecomposition()
        self._timing = FactorTimingModel()

    def generate_full_report(
        self,
        weights: Dict[str, float],
        benchmark_weights: Optional[Dict[str, float]] = None,
        start: str = "2015-01-01",
        end: Optional[str] = None,
        factor_start: str = "2000-01-01",
    ) -> Dict[str, Any]:
        """
        Full factor risk attribution report for a portfolio.

        Sections:
        1. Factor exposures for each holding
        2. Portfolio factor decomposition (systematic vs specific risk)
        3. Active factor tilts vs benchmark
        4. Historical factor P&L attribution
        5. Factor timing signals
        6. 30-day forward vol forecast
        """
        tickers = list(weights.keys())
        logger.info("Generating risk attribution report for %d tickers", len(tickers))

        # 1. Estimate factor exposures for all holdings
        exposures = self._estimator.get_exposures_batch(tickers, start=start, end=end)

        # 2. Portfolio factor decomposition
        factor_cov = self._loader.get_factor_covariance(start=factor_start)
        decomp = self._decomposer.decompose(weights, exposures, factor_cov=factor_cov)

        # 3. Active factor tilts vs benchmark
        active_tilts: Dict[str, float] = {}
        if benchmark_weights:
            bench_tickers = list(benchmark_weights.keys())
            bench_exposures = self._estimator.get_exposures_batch(bench_tickers, start=start, end=end)
            bench_decomp = self._decomposer.decompose(
                benchmark_weights, bench_exposures, factor_cov=factor_cov
            )
            for factor in decomp.factor_betas:
                port_beta = decomp.factor_betas.get(factor, 0.0)
                bench_beta = bench_decomp.factor_betas.get(factor, 0.0)
                active_tilts[factor] = round(port_beta - bench_beta, 4)
        else:
            # vs market (all betas = 0 except Mkt-RF = 1)
            for factor, beta in decomp.factor_betas.items():
                active_tilts[factor] = round(beta - (1.0 if factor == "Mkt-RF" else 0.0), 4)

        # 4. Historical factor P&L attribution
        factor_pnl = self._compute_historical_factor_pnl(
            decomp.factor_betas, start=start, end=end
        )

        # 5. Factor timing signals
        timing_signals = self._timing.get_timing_signals(start=factor_start, end=end)
        timing_dict = {s.factor: {
            "signal": s.signal,
            "score": s.score,
            "momentum_12_1": s.momentum_12_1,
            "rolling_sharpe_1y": s.rolling_sharpe_1y,
        } for s in timing_signals}

        # 6. 30-day forward vol forecast
        vol_forecast = self._forecast_portfolio_vol(
            decomp, factor_cov, horizon_days=30
        )

        # Largest factor tilts summary
        sorted_tilts = sorted(active_tilts.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            "portfolio_summary": {
                "portfolio_vol_annualized": decomp.portfolio_vol_annualized,
                "factor_vol_annualized": decomp.factor_vol_annualized,
                "specific_vol_annualized": decomp.specific_vol_annualized,
                "factor_var_pct": decomp.factor_var_pct,
                "specific_var_pct": decomp.specific_var_pct,
            },
            "portfolio_factor_betas": decomp.factor_betas,
            "factor_var_contributions": decomp.factor_var_contributions,
            "marginal_risk_contributions": decomp.marginal_risk_contributions,
            "active_factor_tilts": active_tilts,
            "largest_active_tilts": dict(sorted_tilts[:5]),
            "asset_factor_exposures": {
                t: {
                    "betas": exp.betas,
                    "r_squared": exp.r_squared,
                    "alpha_annualized": exp.alpha_annualized,
                    "residual_vol": exp.residual_vol_annualized,
                    "error": exp.error,
                }
                for t, exp in exposures.items()
            },
            "historical_factor_pnl": factor_pnl,
            "factor_timing_signals": timing_dict,
            "vol_forecast_30d": vol_forecast,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _compute_historical_factor_pnl(
        self,
        port_betas: Dict[str, float],
        start: str = "2015-01-01",
        end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute historical cumulative return contribution from each factor."""
        factors = self._loader.get_combined(start=start, end=end)
        if factors.empty:
            return {}

        factor_pnl: Dict[str, float] = {}
        for factor, beta in port_betas.items():
            if factor in factors.columns:
                cum_contribution = float(beta * factors[factor].sum())
                factor_pnl[factor] = round(cum_contribution, 4)

        total_pnl = sum(factor_pnl.values())
        factor_pnl["total_factor_return"] = round(total_pnl, 4)
        return factor_pnl

    def _forecast_portfolio_vol(
        self,
        decomp: PortfolioDecomposition,
        factor_cov: pd.DataFrame,
        horizon_days: int = 30,
    ) -> Dict[str, float]:
        """
        30-day forward volatility forecast using factor model.
        Assumes factor covariance is stationary (annualized); scale to horizon.
        """
        scale = np.sqrt(horizon_days / TRADING_DAYS_PER_YEAR)
        port_vol_30d = decomp.portfolio_vol_annualized * scale
        factor_vol_30d = decomp.factor_vol_annualized * scale
        specific_vol_30d = decomp.specific_vol_annualized * scale

        return {
            "horizon_days": horizon_days,
            "portfolio_vol_forecast": round(float(port_vol_30d), 4),
            "factor_vol_forecast": round(float(factor_vol_30d), 4),
            "specific_vol_forecast": round(float(specific_vol_30d), 4),
            "factor_var_pct": decomp.factor_var_pct,
        }

    def largest_factor_tilts(
        self,
        exposures: Dict[str, FactorExposures],
        weights: Dict[str, float],
        n_top: int = 5,
    ) -> List[Dict[str, Any]]:
        """Identify positions with the largest factor tilts vs equal-weight."""
        tickers = [t for t in weights if t in exposures and exposures[t].error is None]
        if not tickers:
            return []

        total_w = sum(weights[t] for t in tickers)
        ew_weight = 1.0 / len(tickers)

        tilt_records = []
        for t in tickers:
            norm_w = weights[t] / total_w
            weight_diff = norm_w - ew_weight
            exp = exposures[t]
            tilt_records.append({
                "ticker": t,
                "weight": round(norm_w, 4),
                "weight_vs_ew": round(weight_diff, 4),
                "market_beta": round(exp.betas.get("Mkt-RF", 0.0), 3),
                "size_beta": round(exp.betas.get("SMB", 0.0), 3),
                "value_beta": round(exp.betas.get("HML", 0.0), 3),
                "profitability_beta": round(exp.betas.get("RMW", 0.0), 3),
                "investment_beta": round(exp.betas.get("CMA", 0.0), 3),
                "momentum_beta": round(exp.betas.get("Mom", 0.0), 3),
                "r_squared": round(exp.r_squared, 3),
                "alpha_ann": round(exp.alpha_annualized, 4),
            })

        # Sort by absolute weight deviation
        tilt_records.sort(key=lambda x: abs(x["weight_vs_ew"]), reverse=True)
        return tilt_records[:n_top]


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:
    factor_risk_router = APIRouter(prefix="/factor-risk", tags=["Factor Risk"])

    class ExposureRequest(BaseModel):
        ticker: str
        start: str = "2015-01-01"
        end: Optional[str] = None

    class PortfolioRequest(BaseModel):
        weights: Dict[str, float]
        benchmark_weights: Optional[Dict[str, float]] = None
        start: str = "2015-01-01"
        end: Optional[str] = None

    class AttributionRequest(BaseModel):
        weights: Dict[str, float]
        benchmark_weights: Optional[Dict[str, float]] = None
        start: str = "2015-01-01"
        end: Optional[str] = None

    _loader_singleton = FamaFrenchDataLoader()
    _estimator_singleton = FactorExposureEstimator()
    _report_singleton = RiskAttributionReport()
    _timing_singleton = FactorTimingModel()

    @factor_risk_router.get("/exposures/{ticker}")
    async def get_factor_exposures(
        ticker: str,
        start: str = Query("2015-01-01"),
        end: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        """Get FF5+Mom factor exposures for a single ticker."""
        ticker = ticker.upper()
        exp = _estimator_singleton.get_exposures(ticker, start=start, end=end)
        return {
            "ticker": exp.ticker,
            "start": exp.start,
            "end": exp.end,
            "observations": exp.observations,
            "betas": exp.betas,
            "t_stats": exp.t_stats,
            "p_values": exp.p_values,
            "alpha_annualized": exp.alpha_annualized,
            "alpha_t_stat": exp.alpha_t_stat,
            "r_squared": exp.r_squared,
            "residual_vol_annualized": exp.residual_vol_annualized,
            "factor_var_explained_pct": exp.factor_var_explained_pct,
            "error": exp.error,
        }

    @factor_risk_router.post("/portfolio")
    async def get_portfolio_decomposition(req: PortfolioRequest) -> Dict[str, Any]:
        """Decompose portfolio risk into factor and specific components."""
        tickers = list(req.weights.keys())
        if not tickers:
            raise HTTPException(status_code=400, detail="No tickers provided")

        exposures = _estimator_singleton.get_exposures_batch(
            tickers, start=req.start, end=req.end
        )
        factor_cov = _loader_singleton.get_factor_covariance(start="2000-01-01")
        decomp = PortfolioFactorDecomposition().decompose(
            req.weights, exposures, factor_cov=factor_cov
        )
        return {
            "portfolio_vol_annualized": decomp.portfolio_vol_annualized,
            "factor_vol_annualized": decomp.factor_vol_annualized,
            "specific_vol_annualized": decomp.specific_vol_annualized,
            "factor_var_pct": decomp.factor_var_pct,
            "specific_var_pct": decomp.specific_var_pct,
            "portfolio_factor_betas": decomp.factor_betas,
            "factor_var_contributions": decomp.factor_var_contributions,
            "marginal_risk_contributions": decomp.marginal_risk_contributions,
        }

    @factor_risk_router.post("/attribution")
    async def get_factor_attribution(req: AttributionRequest) -> Dict[str, Any]:
        """Full factor risk attribution report for a portfolio."""
        if not req.weights:
            raise HTTPException(status_code=400, detail="No weights provided")

        report = _report_singleton.generate_full_report(
            weights=req.weights,
            benchmark_weights=req.benchmark_weights,
            start=req.start,
            end=req.end,
        )
        return report

    @factor_risk_router.get("/factor-timing")
    async def get_factor_timing(
        start: str = Query("2000-01-01"),
        end: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        """Get factor momentum and timing signals."""
        signals = _timing_singleton.get_timing_signals(start=start, end=end)
        rotation_weights = _timing_singleton.factor_rotation_weights(signals)
        return {
            "signals": [
                {
                    "factor": s.factor,
                    "signal": s.signal,
                    "score": s.score,
                    "momentum_12_1": s.momentum_12_1,
                    "momentum_1m": s.momentum_1m,
                    "rolling_sharpe_1y": s.rolling_sharpe_1y,
                    "regime_bull_return": s.regime_bull_return,
                    "regime_bear_return": s.regime_bear_return,
                }
                for s in signals
            ],
            "factor_rotation_tilts": rotation_weights,
        }

    @factor_risk_router.post("/report")
    async def get_full_report(req: AttributionRequest) -> Dict[str, Any]:
        """Comprehensive factor risk report with timing overlay."""
        if not req.weights:
            raise HTTPException(status_code=400, detail="No weights provided")
        report = _report_singleton.generate_full_report(
            weights=req.weights,
            benchmark_weights=req.benchmark_weights,
            start=req.start,
            end=req.end,
        )
        return report

else:
    factor_risk_router = None  # type: ignore


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def get_factor_exposures(
    ticker: str,
    start: str = "2015-01-01",
    end: Optional[str] = None,
    window: int = 252,
) -> FactorExposures:
    """Convenience: estimate FF5+Mom factor exposures for a ticker."""
    estimator = FactorExposureEstimator(window=window)
    return estimator.get_exposures(ticker, start=start, end=end)


def get_portfolio_report(
    weights: Dict[str, float],
    benchmark_weights: Optional[Dict[str, float]] = None,
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience: full portfolio factor attribution report."""
    reporter = RiskAttributionReport()
    return reporter.generate_full_report(
        weights=weights,
        benchmark_weights=benchmark_weights,
        start=start,
        end=end,
    )


def get_factor_timing_signals(
    start: str = "2000-01-01",
    end: Optional[str] = None,
) -> List[FactorTimingSignal]:
    """Convenience: generate factor timing signals."""
    model = FactorTimingModel()
    return model.get_timing_signals(start=start, end=end)


def compute_bab_factor(
    universe_tickers: List[str],
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> pd.Series:
    """Convenience: compute BAB factor from a universe of tickers."""
    adapter = QMJBetaAdapter()
    return adapter.compute_bab_factor(universe_tickers, start=start, end=end)


def compute_qmj_factor(
    universe_tickers: List[str],
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> pd.Series:
    """Convenience: compute QMJ factor from a universe of tickers."""
    adapter = QMJBetaAdapter()
    return adapter.compute_qmj_factor(universe_tickers, start=start, end=end)
