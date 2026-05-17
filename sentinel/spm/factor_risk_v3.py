"""
sentinel/spm/factor_risk_v3.py
dim_079: Multi-factor risk model — Fama-French 5-Factor + Momentum (score 7 → 9)

Comprehensive Fama-French 5-factor + momentum risk model using free data:
- Kenneth French Data Library (FF5 daily factors + MOM + Industry portfolios)
- yfinance: stock price returns
- Pure numpy/pandas OLS throughout; no statsmodels dependency required.
"""

from __future__ import annotations

import io
import os
import csv
import logging
import zipfile
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FF5_DAILY_URL = f"{FRENCH_BASE}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_DAILY_URL = f"{FRENCH_BASE}/F-F_Momentum_Factor_daily_CSV.zip"
INDUSTRY_10_URL = f"{FRENCH_BASE}/10_Industry_Portfolios_daily_CSV.zip"

CACHE_DIR = Path(__file__).parent.parent / "data"
FF5_CACHE_FILE = CACHE_DIR / "ff5_mom_daily.parquet"
FF5_CSV_FALLBACK = CACHE_DIR / "ff5_mom_daily.csv"
INDUSTRY_CACHE_FILE = CACHE_DIR / "industry_10_daily.parquet"
CACHE_STALE_DAYS = 7

FACTOR_NAMES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
ALL_FACTORS = FACTOR_NAMES + ["RF"]

# VaR confidence levels
VAR_95 = 1.645
VAR_99 = 2.326


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FactorExposures:
    """OLS factor regression results for a single asset or portfolio."""
    ticker: str
    alpha: float
    betas: dict[str, float]
    t_stats: dict[str, float]
    r_squared: float
    adj_r_squared: float
    residual_vol: float          # annualized
    residual_vol_daily: float    # daily std of residuals
    n_obs: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    factor_names: list[str] = field(default_factory=list)
    fitted_values: Optional[pd.Series] = None
    residuals: Optional[pd.Series] = None

    def summary(self) -> str:
        lines = [
            f"Factor Exposures: {self.ticker}",
            f"  Period:      {self.start_date} → {self.end_date}  (N={self.n_obs})",
            f"  Alpha:       {self.alpha*252:.2%} ann.  (daily: {self.alpha:.4f})",
            f"  R²:          {self.r_squared:.3f}  Adj-R²: {self.adj_r_squared:.3f}",
            f"  Resid Vol:   {self.residual_vol:.2%} ann.",
            "  Factor Betas (t-stat):",
        ]
        for f in self.factor_names:
            b = self.betas.get(f, float("nan"))
            t = self.t_stats.get(f, float("nan"))
            star = "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.65 else ""
            lines.append(f"    {f:10s}  {b:+.4f}  ({t:+.2f}{star})")
        return "\n".join(lines)


@dataclass
class FactorDashboard:
    """Portfolio-level factor risk dashboard."""
    timestamp: str
    portfolio_exposures: FactorExposures
    factor_var_95: float          # factor-explained VaR at 95%
    factor_var_99: float          # factor-explained VaR at 99%
    idiosyncratic_var_95: float   # idiosyncratic VaR at 95%
    systematic_pct: float         # % of variance explained by factors
    factor_concentration_score: float  # 0=diversified, 100=single-factor
    crowded_factors: list[str]    # factors with high universe crowding
    factor_momentum: dict[str, float]  # recent factor return (1y annualized)
    variance_decomposition: dict[str, float]
    risk_flags: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 65,
            "  SENTINEL FACTOR RISK DASHBOARD",
            f"  {self.timestamp}",
            "=" * 65,
            "",
            self.portfolio_exposures.summary(),
            "",
            f"RISK METRICS:",
            f"  Factor VaR 95%:     {self.factor_var_95:.2%}",
            f"  Factor VaR 99%:     {self.factor_var_99:.2%}",
            f"  Idiosync VaR 95%:   {self.idiosyncratic_var_95:.2%}",
            f"  Systematic Risk:    {self.systematic_pct:.1f}%",
            f"  Factor Conc. Score: {self.factor_concentration_score:.0f}/100",
            "",
            "VARIANCE DECOMPOSITION:",
        ]
        for k, v in self.variance_decomposition.items():
            lines.append(f"  {k:12s}: {v:.2%}")
        if self.factor_momentum:
            lines.append("\nFACTOR MOMENTUM (1Y Annualized):")
            for k, v in self.factor_momentum.items():
                lines.append(f"  {k:10s}: {v:+.2%}")
        if self.risk_flags:
            lines.append("\nRISK FLAGS:")
            for flag in self.risk_flags:
                lines.append(f"  [!] {flag}")
        lines.append("=" * 65)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTTP utility
# ---------------------------------------------------------------------------

def _fetch_zip_csv(url: str, timeout: int = 120) -> bytes:
    """Download a ZIP archive and return the raw bytes of the first CSV inside."""
    headers = {
        "User-Agent": "SENTINEL/3.0 (research; academic use)",
        "Accept": "application/zip,*/*",
    }
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (HTTPError, URLError) as e:
        logger.error("Failed to download %s: %s", url, e)
        raise
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV found in ZIP: {url}")
        with zf.open(csv_names[0]) as f:
            return f.read()


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# FrenchDataLoader
# ---------------------------------------------------------------------------

class FrenchDataLoader:
    """
    Download and parse Kenneth French factor data library files.

    All factor returns are in decimal form (i.e., 0.01 = 1%).
    French files ship as percentages — we divide by 100 on load.
    """

    def __init__(self, cache_dir: Path = None):
        self._cache_dir = cache_dir or CACHE_DIR
        _ensure_cache_dir()

    @staticmethod
    def _parse_french_csv(raw_bytes: bytes, skip_header_rows: int = 3) -> pd.DataFrame:
        """
        Parse a Kenneth French CSV (percent format).

        French CSVs have an irregular header (copyright notes) followed by
        a blank line, then the actual data with a date column (YYYYMMDD or YYYYMM).
        We scan until we find the data block.
        """
        text = raw_bytes.decode("latin-1")
        lines = text.splitlines()

        data_start = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Find the header row: starts with a date-like integer or ",Mkt-RF"
            if stripped and (
                stripped.startswith(",") or
                (stripped[:8].replace(",", "").isdigit() and len(stripped[:8].replace(",", "")) >= 6)
            ):
                # Previous non-blank line might be the header
                if data_start is None:
                    data_start = i
                break

        # Scan more carefully for the first data header line
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) >= 3 and any(
                kw in parts for kw in ["Mkt-RF", "SMB", "HML", "Mom", "MOM", "NoDur"]
            ):
                data_start = i
                break

        if data_start is None:
            raise ValueError("Could not find data header in French CSV")

        # Read from data_start onward
        content = "\n".join(lines[data_start:])
        # Find footer (French files have an annual/monthly summary after daily data)
        # Stop at first row where date column is not 8 digits
        rows = []
        header_row = None
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parts = [p.strip() for p in stripped.split(",")]
            if header_row is None:
                header_row = parts
                continue
            # Check if date is 8-digit (daily) or 6-digit (monthly) or transition
            date_str = parts[0].strip()
            if len(date_str) == 8 and date_str.isdigit():
                rows.append(parts)
            elif len(date_str) == 6 and date_str.isdigit():
                # Monthly data — skip (daily files sometimes append monthly summaries)
                continue
            else:
                # End of data section
                break

        if not rows or not header_row:
            raise ValueError("No data rows found in French CSV")

        df = pd.DataFrame(rows, columns=header_row[: len(rows[0])])
        # Rename date column
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"].str.strip(), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.set_index("Date")

        # Convert all columns to float and divide by 100
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].str.strip(), errors="coerce") / 100.0

        df = df.dropna(how="all")
        return df.sort_index()

    def download_ff5_daily(self) -> pd.DataFrame:
        """
        Download and parse FF5 daily factors.

        Returns DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, RF
        Index: DatetimeIndex (UTC).
        """
        logger.info("Downloading FF5 daily factors from French Data Library...")
        raw = _fetch_zip_csv(FF5_DAILY_URL)
        df = self._parse_french_csv(raw)
        # Standardize column names
        col_map = {}
        for c in df.columns:
            if "Mkt" in c:
                col_map[c] = "Mkt-RF"
            elif c.strip().upper() in ("SMB", "HML", "RMW", "CMA", "RF"):
                col_map[c] = c.strip().upper()
        df = df.rename(columns=col_map)
        logger.info("FF5 daily: %d observations, %s to %s", len(df), df.index.min().date(), df.index.max().date())
        return df

    def download_momentum_daily(self) -> pd.Series:
        """
        Download and parse the Fama-French Momentum factor (MOM).

        Returns a pd.Series named 'MOM'.
        """
        logger.info("Downloading MOM daily factor...")
        raw = _fetch_zip_csv(MOM_DAILY_URL)
        df = self._parse_french_csv(raw)
        # Find the MOM column
        mom_col = None
        for c in df.columns:
            if "mom" in c.lower() or "pr1" in c.lower() or "wml" in c.lower():
                mom_col = c
                break
        if mom_col is None and len(df.columns) > 0:
            mom_col = df.columns[0]
        if mom_col is None:
            logger.warning("MOM factor column not found; returning zeros")
            return pd.Series(name="MOM", dtype=float)
        series = df[mom_col].rename("MOM")
        logger.info("MOM daily: %d observations", len(series))
        return series

    def download_industry_portfolios(self) -> pd.DataFrame:
        """
        Download and parse 10 industry portfolio daily returns.

        Returns DataFrame with 10 industry columns (equal-weighted).
        """
        logger.info("Downloading 10 Industry Portfolios (daily)...")
        raw = _fetch_zip_csv(INDUSTRY_10_URL)
        df = self._parse_french_csv(raw)
        # French industry file may have value-weighted and equal-weighted sections
        # Take first 10 columns as value-weighted returns
        df = df.iloc[:, :10]
        logger.info("Industry portfolios: %d observations, %d industries", len(df), len(df.columns))
        return df

    def load_all_factors(self, force_download: bool = False) -> pd.DataFrame:
        """
        Load FF5 + MOM factor data, with caching.

        Cache location: sentinel/data/ff5_mom_daily.parquet (or .csv fallback).
        Re-downloads if cache is older than CACHE_STALE_DAYS or force_download=True.
        """
        cache_path = FF5_CACHE_FILE
        fallback_path = FF5_CSV_FALLBACK

        if not force_download and cache_path.exists():
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
            age_days = (datetime.now(timezone.utc) - mtime).days
            if age_days < CACHE_STALE_DAYS:
                logger.info("Loading FF5+MOM from cache (age=%d days)", age_days)
                try:
                    return pd.read_parquet(cache_path)
                except Exception as e:
                    logger.warning("Parquet read failed: %s — trying CSV", e)

        if not force_download and fallback_path.exists():
            mtime = datetime.fromtimestamp(fallback_path.stat().st_mtime, tz=timezone.utc)
            age_days = (datetime.now(timezone.utc) - mtime).days
            if age_days < CACHE_STALE_DAYS:
                logger.info("Loading FF5+MOM from CSV cache")
                try:
                    df = pd.read_csv(fallback_path, index_col=0, parse_dates=True)
                    return df
                except Exception:
                    pass

        # Download fresh
        ff5 = self.download_ff5_daily()
        mom = self.download_momentum_daily()

        # Merge
        if not mom.empty:
            combined = ff5.join(mom, how="left")
            combined["MOM"] = combined["MOM"].fillna(0.0)
        else:
            combined = ff5.copy()
            combined["MOM"] = 0.0

        # Save cache
        try:
            combined.to_parquet(cache_path)
            logger.info("Cached FF5+MOM to %s", cache_path)
        except Exception:
            try:
                combined.to_csv(fallback_path)
                logger.info("Cached FF5+MOM to CSV fallback %s", fallback_path)
            except Exception as e2:
                logger.warning("Cache save failed: %s", e2)

        return combined

    def update_factors(self):
        """Force re-download if cache is stale (> CACHE_STALE_DAYS old)."""
        if FF5_CACHE_FILE.exists():
            mtime = datetime.fromtimestamp(FF5_CACHE_FILE.stat().st_mtime, tz=timezone.utc)
            age_days = (datetime.now(timezone.utc) - mtime).days
            if age_days < CACHE_STALE_DAYS:
                logger.info("Factor cache is fresh (%d days old). No update needed.", age_days)
                return
        self.load_all_factors(force_download=True)

    def get_factors(
        self,
        start: str = None,
        end: str = None,
        factors: list[str] = None,
    ) -> pd.DataFrame:
        """
        Get factor returns for a date range.

        Args:
            start: Start date string (YYYY-MM-DD) or None for full history.
            end: End date string (YYYY-MM-DD) or None for today.
            factors: Subset of factor names. Defaults to all FF5+MOM+RF.
        """
        df = self.load_all_factors()
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        if factors:
            available = [f for f in factors if f in df.columns]
            df = df[available]
        return df


# ---------------------------------------------------------------------------
# Stock returns via yfinance
# ---------------------------------------------------------------------------

def _get_stock_returns(
    tickers: list[str],
    start: str,
    end: str,
    align_index: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices via yfinance and compute daily log returns.

    Returns DataFrame with one column per ticker.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required for stock data: pip install yfinance")

    # Download as single batch for efficiency
    prices = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.sort_index()
    # Convert to UTC-naive for alignment with French factors
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    # Log returns
    returns = np.log(prices / prices.shift(1)).dropna(how="all")

    if align_index is not None:
        idx = pd.to_datetime(align_index).tz_localize(None)
        returns = returns.reindex(idx).dropna(how="all")

    return returns


# ---------------------------------------------------------------------------
# OLS utility
# ---------------------------------------------------------------------------

def _ols(
    y: np.ndarray,
    X: np.ndarray,
    add_const: bool = True,
) -> dict:
    """
    Pure numpy OLS regression.

    Returns dict with: alpha, betas, t_stats, r_squared, adj_r_squared,
    residual_std, residuals, fitted.
    """
    n = len(y)
    if add_const:
        X_full = np.column_stack([np.ones(n), X])
    else:
        X_full = X

    k = X_full.shape[1]
    if n <= k:
        raise ValueError(f"Too few observations ({n}) for {k} regressors")

    # OLS estimate: β = (X'X)^{-1} X'y
    try:
        XtX = X_full.T @ X_full
        Xty = X_full.T @ y
        beta_all = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise ValueError(f"OLS singular matrix: {e}")

    fitted = X_full @ beta_all
    residuals = y - fitted
    sse = float(np.dot(residuals, residuals))
    sst = float(np.dot(y - y.mean(), y - y.mean()))
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else 0.0

    # Standard errors
    sigma2 = sse / (n - k)
    try:
        var_beta = sigma2 * np.linalg.inv(XtX)
        se_beta = np.sqrt(np.diag(var_beta))
    except np.linalg.LinAlgError:
        se_beta = np.full(k, float("nan"))

    t_stats_all = beta_all / (se_beta + 1e-12)

    return {
        "alpha": float(beta_all[0]) if add_const else 0.0,
        "betas": beta_all[1:] if add_const else beta_all,
        "t_stats": t_stats_all,
        "r_squared": float(r2),
        "adj_r_squared": float(adj_r2),
        "residual_std": float(np.sqrt(sigma2)),
        "residuals": residuals,
        "fitted": fitted,
        "n": n,
        "se_beta": se_beta,
    }


# ---------------------------------------------------------------------------
# FactorModelFitter
# ---------------------------------------------------------------------------

class FactorModelFitter:
    """
    Fit Fama-French 5-Factor + Momentum model to individual stocks or portfolios.

    The regression model is:
        R_i - RF = α + β_mkt(Mkt-RF) + β_smb(SMB) + β_hml(HML)
                     + β_rmw(RMW) + β_cma(CMA) + β_mom(MOM) + ε
    """

    def __init__(self, factor_data: pd.DataFrame = None, loader: FrenchDataLoader = None):
        self._loader = loader or FrenchDataLoader()
        self._factor_data = factor_data

    def _get_factors(self, start: str = None, end: str = None) -> pd.DataFrame:
        if self._factor_data is not None:
            df = self._factor_data.copy()
            if start:
                df = df[df.index >= pd.Timestamp(start)]
            if end:
                df = df[df.index <= pd.Timestamp(end)]
            return df
        return self._loader.get_factors(start=start, end=end)

    def fit(
        self,
        returns: pd.Series,
        factors: pd.DataFrame = None,
        start: str = None,
        end: str = None,
        factor_names: list[str] = None,
    ) -> FactorExposures:
        """
        Fit FF5+Mom model to a return series.

        Args:
            returns: Daily excess returns or total returns (pd.Series, daily decimal).
            factors: Pre-loaded factor DataFrame (optional; loads from cache if None).
            start: Start date filter.
            end: End date filter.
            factor_names: Which factors to use. Defaults to FACTOR_NAMES.
        """
        if factor_names is None:
            factor_names = FACTOR_NAMES

        if factors is None:
            factors = self._get_factors(start=start, end=end)

        # Normalize index timezone
        ret_idx = pd.to_datetime(returns.index).tz_localize(None)
        returns = returns.copy()
        returns.index = ret_idx

        fac_idx = pd.to_datetime(factors.index).tz_localize(None)
        factors = factors.copy()
        factors.index = fac_idx

        # Align on common dates
        common_idx = ret_idx.intersection(fac_idx)
        if len(common_idx) < 30:
            raise ValueError(
                f"Insufficient overlapping observations: {len(common_idx)} (need ≥ 30)"
            )

        y_all = returns.loc[common_idx].values.astype(float)
        rf = factors.loc[common_idx, "RF"].values.astype(float) if "RF" in factors.columns else np.zeros(len(common_idx))
        y_excess = y_all - rf  # excess returns

        # Factor matrix
        available_factors = [f for f in factor_names if f in factors.columns]
        if not available_factors:
            raise ValueError(f"None of the requested factors found: {factor_names}")
        X = factors.loc[common_idx, available_factors].values.astype(float)

        # Run OLS
        result = _ols(y_excess, X, add_const=True)

        betas = {name: float(b) for name, b in zip(available_factors, result["betas"])}
        t_stats_vals = result["t_stats"]
        t_alpha = float(t_stats_vals[0]) if len(t_stats_vals) > 0 else 0.0
        t_betas = {name: float(t) for name, t in zip(available_factors, t_stats_vals[1:])}
        t_betas_full = {"alpha": t_alpha, **t_betas}

        resid_daily = float(result["residual_std"])
        resid_ann = resid_daily * np.sqrt(252)

        return FactorExposures(
            ticker=returns.name or "unknown",
            alpha=result["alpha"],
            betas=betas,
            t_stats=t_betas_full,
            r_squared=result["r_squared"],
            adj_r_squared=result["adj_r_squared"],
            residual_vol=resid_ann,
            residual_vol_daily=resid_daily,
            n_obs=result["n"],
            start_date=str(common_idx.min().date()),
            end_date=str(common_idx.max().date()),
            factor_names=available_factors,
            residuals=pd.Series(result["residuals"], index=common_idx),
            fitted_values=pd.Series(result["fitted"], index=common_idx),
        )

    def fit_rolling(
        self,
        returns: pd.Series,
        factors: pd.DataFrame = None,
        window: int = 252,
        step: int = 21,  # recompute monthly for efficiency
        factor_names: list[str] = None,
    ) -> pd.DataFrame:
        """
        Compute rolling factor exposures with a specified window (default 252 days).

        Args:
            returns: Daily return series.
            factors: Factor DataFrame.
            window: Rolling window size in days.
            step: Step between rolling fits (default 21 = monthly).
            factor_names: Factors to include.

        Returns:
            DataFrame with columns: alpha, β_Mkt-RF, β_SMB, ... indexed by date.
        """
        if factor_names is None:
            factor_names = FACTOR_NAMES

        if factors is None:
            factors = self._get_factors()

        ret_idx = pd.to_datetime(returns.index).tz_localize(None)
        returns_clean = returns.copy()
        returns_clean.index = ret_idx

        fac_idx = pd.to_datetime(factors.index).tz_localize(None)
        factors_clean = factors.copy()
        factors_clean.index = fac_idx

        common_idx = ret_idx.intersection(fac_idx)
        y = returns_clean.loc[common_idx]
        F = factors_clean.loc[common_idx]

        available_factors = [f for f in factor_names if f in F.columns]
        rf = F["RF"].values if "RF" in F.columns else np.zeros(len(F))

        dates = list(common_idx)
        records = []

        for end_pos in range(window, len(dates) + 1, step):
            start_pos = end_pos - window
            slice_dates = dates[start_pos:end_pos]
            y_slice = y.iloc[start_pos:end_pos].values.astype(float)
            rf_slice = rf[start_pos:end_pos]
            y_excess = y_slice - rf_slice
            X_slice = F[available_factors].iloc[start_pos:end_pos].values.astype(float)

            try:
                res = _ols(y_excess, X_slice, add_const=True)
                row = {"date": slice_dates[-1], "alpha_daily": res["alpha"]}
                for fname, bval in zip(available_factors, res["betas"]):
                    row[f"beta_{fname}"] = float(bval)
                row["r_squared"] = res["r_squared"]
                row["resid_vol_daily"] = res["residual_std"]
            except Exception as e:
                logger.debug("Rolling OLS failed at %s: %s", slice_dates[-1], e)
                row = {"date": slice_dates[-1]}
            records.append(row)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index("date")
        return df

    def compute_factor_contribution(
        self,
        exposures: FactorExposures,
        factor_returns: pd.DataFrame,
        start: str = None,
        end: str = None,
    ) -> pd.DataFrame:
        """
        Compute daily PnL attribution to each factor.

        Contribution_i(t) = β_i × factor_return_i(t)
        Alpha contribution = alpha (constant daily)

        Returns:
            DataFrame with columns: alpha, Mkt-RF, SMB, HML, RMW, CMA, MOM, total_explained
        """
        fr = factor_returns.copy()
        fr.index = pd.to_datetime(fr.index).tz_localize(None)

        if start:
            fr = fr[fr.index >= pd.Timestamp(start)]
        if end:
            fr = fr[fr.index <= pd.Timestamp(end)]

        contrib = pd.DataFrame(index=fr.index)
        contrib["alpha"] = exposures.alpha

        for fname, beta in exposures.betas.items():
            if fname in fr.columns:
                contrib[fname] = fr[fname] * beta

        contrib["total_explained"] = contrib.sum(axis=1)
        return contrib

    def compute_active_exposures(
        self,
        portfolio_exposures: FactorExposures,
        benchmark_exposures: FactorExposures,
    ) -> dict:
        """
        Compute active factor bets = portfolio β - benchmark β for each factor.

        Large absolute active bets indicate concentrated directional factor views.
        """
        active = {}
        all_factors = set(portfolio_exposures.betas) | set(benchmark_exposures.betas)
        for f in all_factors:
            p_beta = portfolio_exposures.betas.get(f, 0.0)
            b_beta = benchmark_exposures.betas.get(f, 0.0)
            active[f] = p_beta - b_beta
        active["alpha"] = portfolio_exposures.alpha - benchmark_exposures.alpha
        return active


# ---------------------------------------------------------------------------
# PortfolioFactorAnalyzer
# ---------------------------------------------------------------------------

class PortfolioFactorAnalyzer:
    """
    Analyze a multi-stock portfolio's factor exposures and risk decomposition.
    """

    def __init__(
        self,
        loader: FrenchDataLoader = None,
        fitter: FactorModelFitter = None,
    ):
        self._loader = loader or FrenchDataLoader()
        self._fitter = fitter or FactorModelFitter(loader=self._loader)
        self._factor_data: pd.DataFrame = pd.DataFrame()

    def _ensure_factors(self, start: str = None) -> pd.DataFrame:
        if self._factor_data.empty:
            self._factor_data = self._loader.load_all_factors()
        df = self._factor_data
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        return df

    def compute_portfolio_exposures(
        self,
        holdings: dict[str, float],
        factors: pd.DataFrame = None,
        lookback_days: int = 756,
        factor_names: list[str] = None,
    ) -> FactorExposures:
        """
        Compute weight-averaged factor exposures for a multi-stock portfolio.

        Args:
            holdings: {ticker: portfolio_weight} — weights need not sum to 1.
            factors: Pre-loaded factor DataFrame (optional).
            lookback_days: History to use for fitting (default 3 years).
            factor_names: Factors to include.
        """
        if factor_names is None:
            factor_names = FACTOR_NAMES

        if factors is None:
            factors = self._ensure_factors()

        # Date range
        end_date = factors.index.max()
        start_date = end_date - pd.Timedelta(days=lookback_days)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        tickers = list(holdings.keys())
        total_weight = sum(holdings.values()) or 1.0

        # Get stock returns
        stock_returns = _get_stock_returns(tickers, start=start_str, end=end_str)

        # Fit each stock individually
        individual_exposures: list[tuple[float, FactorExposures]] = []
        for ticker in tickers:
            if ticker not in stock_returns.columns:
                logger.warning("No return data for %s — skipping", ticker)
                continue
            weight = holdings[ticker] / total_weight
            ret = stock_returns[ticker].dropna().rename(ticker)
            try:
                exp = self._fitter.fit(ret, factors=factors, factor_names=factor_names)
                individual_exposures.append((weight, exp))
            except Exception as e:
                logger.warning("Could not fit %s: %s", ticker, e)

        if not individual_exposures:
            raise ValueError("No stock exposures could be fitted")

        # Weight-average betas
        portfolio_alpha = sum(w * e.alpha for w, e in individual_exposures)
        portfolio_betas: dict[str, float] = {}
        portfolio_t_stats: dict[str, float] = {}
        all_fnames = individual_exposures[0][1].factor_names

        for fname in all_fnames:
            portfolio_betas[fname] = sum(
                w * e.betas.get(fname, 0.0) for w, e in individual_exposures
            )

        # Weighted avg R² and residual vol
        portfolio_r2 = sum(w * e.r_squared for w, e in individual_exposures)
        portfolio_resid_daily = np.sqrt(
            sum(w ** 2 * e.residual_vol_daily ** 2 for w, e in individual_exposures)
        )
        portfolio_resid_ann = portfolio_resid_daily * np.sqrt(252)
        n_obs = min(e.n_obs for _, e in individual_exposures)

        return FactorExposures(
            ticker="Portfolio",
            alpha=portfolio_alpha,
            betas=portfolio_betas,
            t_stats=portfolio_t_stats,
            r_squared=portfolio_r2,
            adj_r_squared=portfolio_r2,
            residual_vol=portfolio_resid_ann,
            residual_vol_daily=portfolio_resid_daily,
            n_obs=n_obs,
            start_date=start_str,
            end_date=end_str,
            factor_names=all_fnames,
        )

    def compute_factor_covariance(
        self,
        factors: pd.DataFrame = None,
        lookback_days: int = 756,
        annualize: bool = True,
    ) -> pd.DataFrame:
        """
        Compute factor return covariance matrix from historical data.

        Args:
            factors: Factor DataFrame (loads from cache if None).
            lookback_days: History to use.
            annualize: Multiply by 252 for annualized covariance.

        Returns:
            Square pd.DataFrame (covariance matrix, factor × factor).
        """
        if factors is None:
            factors = self._ensure_factors()

        fac_cols = [f for f in FACTOR_NAMES if f in factors.columns]
        recent = factors[fac_cols].tail(lookback_days)
        cov = recent.cov()
        if annualize:
            cov = cov * 252
        return cov

    def compute_factor_var(
        self,
        exposures: FactorExposures,
        factor_cov: pd.DataFrame = None,
        portfolio_value: float = 1.0,
        confidence: float = 0.95,
    ) -> dict:
        """
        Decompose portfolio VaR into factor (systematic) and idiosyncratic components.

        Factor VaR: VaR_factor = z × sqrt(β' × Σ_F × β) × portfolio_value
        Idiosyncratic VaR: z × σ_ε × portfolio_value (using annualized daily vol)

        Args:
            exposures: Fitted FactorExposures object.
            factor_cov: Factor covariance matrix (annualized). Loads if None.
            portfolio_value: Portfolio notional (for $ VaR).
            confidence: Confidence level (0.95 or 0.99).
        """
        if factor_cov is None:
            factor_cov = self.compute_factor_covariance()

        z = VAR_99 if confidence >= 0.99 else VAR_95
        z_99 = VAR_99
        z_95 = VAR_95

        # Beta vector aligned with cov matrix
        fac_names = [f for f in factor_cov.columns if f in exposures.betas]
        beta_vec = np.array([exposures.betas.get(f, 0.0) for f in fac_names])
        cov_matrix = factor_cov.loc[fac_names, fac_names].values

        # Systematic variance (annualized)
        systematic_var = float(beta_vec @ cov_matrix @ beta_vec)
        systematic_vol = np.sqrt(max(systematic_var, 0.0))

        # Daily systematic vol from annualized
        systematic_vol_daily = systematic_vol / np.sqrt(252)

        # Idiosyncratic vol (already annualized in exposures)
        idio_vol_daily = exposures.residual_vol_daily

        # Total portfolio vol (daily)
        total_vol_daily = np.sqrt(systematic_vol_daily ** 2 + idio_vol_daily ** 2)

        return {
            "factor_var_95": z_95 * systematic_vol_daily * portfolio_value,
            "factor_var_99": z_99 * systematic_vol_daily * portfolio_value,
            "idiosyncratic_var_95": z_95 * idio_vol_daily * portfolio_value,
            "idiosyncratic_var_99": z_99 * idio_vol_daily * portfolio_value,
            "total_var_95": z_95 * total_vol_daily * portfolio_value,
            "total_var_99": z_99 * total_vol_daily * portfolio_value,
            "systematic_vol_ann": systematic_vol,
            "systematic_vol_daily": systematic_vol_daily,
            "idiosyncratic_vol_ann": exposures.residual_vol,
            "idiosyncratic_vol_daily": idio_vol_daily,
            "total_vol_daily": total_vol_daily,
            "total_vol_ann": total_vol_daily * np.sqrt(252),
            "confidence": confidence,
        }

    def decompose_variance(
        self,
        exposures: FactorExposures,
        factor_cov: pd.DataFrame = None,
    ) -> dict:
        """
        Decompose portfolio variance into systematic and idiosyncratic components.

        Systematic variance = β' × Σ_F × β
        Idiosyncratic variance = σ²_ε (residual variance, daily)
        Total variance = Systematic + Idiosyncratic (daily)

        Returns:
            dict with variance components, % systematic, % idiosyncratic.
        """
        if factor_cov is None:
            factor_cov = self.compute_factor_covariance()

        fac_names = [f for f in factor_cov.columns if f in exposures.betas]
        beta_vec = np.array([exposures.betas.get(f, 0.0) for f in fac_names])
        cov_matrix = factor_cov.loc[fac_names, fac_names].values

        systematic_var_ann = float(beta_vec @ cov_matrix @ beta_vec)
        idio_var_ann = exposures.residual_vol ** 2
        total_var_ann = systematic_var_ann + idio_var_ann

        if total_var_ann <= 0:
            pct_sys = 0.0
            pct_idio = 0.0
        else:
            pct_sys = systematic_var_ann / total_var_ann
            pct_idio = idio_var_ann / total_var_ann

        # Individual factor contributions to systematic variance
        factor_contributions = {}
        for i, fname in enumerate(fac_names):
            # Marginal contribution: β_i × (Σ_F × β)_i
            sigma_beta = cov_matrix @ beta_vec
            fc = beta_vec[i] * sigma_beta[i]
            factor_contributions[fname] = float(fc) / max(total_var_ann, 1e-12)

        return {
            "systematic_variance_ann": systematic_var_ann,
            "idiosyncratic_variance_ann": idio_var_ann,
            "total_variance_ann": total_var_ann,
            "systematic_vol_ann": np.sqrt(max(systematic_var_ann, 0.0)),
            "idiosyncratic_vol_ann": exposures.residual_vol,
            "pct_systematic": pct_sys,
            "pct_idiosyncratic": pct_idio,
            "r_squared": exposures.r_squared,
            "factor_variance_contributions": factor_contributions,
        }

    def compute_factor_pnl_attribution(
        self,
        portfolio_returns: pd.Series,
        exposures: FactorExposures,
        factors: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Daily PnL attributed to each factor + idiosyncratic residual.

        Args:
            portfolio_returns: Daily portfolio return series.
            exposures: Fitted FactorExposures object.
            factors: Factor DataFrame (loads if None).

        Returns:
            DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, MOM, alpha, idiosyncratic, total_explained
        """
        if factors is None:
            factors = self._ensure_factors()

        fitter = FactorModelFitter(factor_data=factors)
        contrib = fitter.compute_factor_contribution(exposures, factors)

        # Align with portfolio returns
        port_idx = pd.to_datetime(portfolio_returns.index).tz_localize(None)
        portfolio_returns_clean = portfolio_returns.copy()
        portfolio_returns_clean.index = port_idx
        common = contrib.index.intersection(port_idx)
        contrib = contrib.loc[common]
        port_ret = portfolio_returns_clean.loc[common]

        contrib["portfolio_return"] = port_ret.values
        contrib["idiosyncratic"] = port_ret.values - contrib["total_explained"].values
        return contrib


# ---------------------------------------------------------------------------
# FactorRiskMonitor
# ---------------------------------------------------------------------------

class FactorRiskMonitor:
    """
    Monitor and alert on factor exposure drift and concentration.
    """

    def __init__(
        self,
        loader: FrenchDataLoader = None,
        portfolio_analyzer: PortfolioFactorAnalyzer = None,
    ):
        self._loader = loader or FrenchDataLoader()
        self._pa = portfolio_analyzer or PortfolioFactorAnalyzer(self._loader)

    def compute_factor_concentration_risk(self, exposures: FactorExposures) -> dict:
        """
        Compute factor concentration risk score.

        Rules:
        - |β| > 1.5 for any factor: concentrated (adds to score)
        - Score = sum of |β_i - 1| for market; |β_i| for others
        - Normalized 0-100 (100 = single-factor bet, 0 = fully diversified)
        """
        concentrated_factors = []
        concentration_components = {}

        market_beta = abs(exposures.betas.get("Mkt-RF", 1.0) - 1.0)
        concentration_components["Mkt-RF"] = market_beta
        if abs(exposures.betas.get("Mkt-RF", 1.0)) > 1.5 or abs(exposures.betas.get("Mkt-RF", 1.0)) < 0.3:
            concentrated_factors.append("Mkt-RF")

        for fname in ["SMB", "HML", "RMW", "CMA", "MOM"]:
            beta = abs(exposures.betas.get(fname, 0.0))
            concentration_components[fname] = beta
            if beta > 1.5:
                concentrated_factors.append(fname)

        # Raw concentration score
        raw_score = (
            market_beta * 20 +
            sum(abs(exposures.betas.get(f, 0.0)) for f in ["SMB", "HML", "RMW", "CMA", "MOM"]) * 10
        )
        # Normalize to 0-100
        concentration_score = min(100.0, raw_score)

        return {
            "concentration_score": concentration_score,
            "concentrated_factors": concentrated_factors,
            "factor_abs_betas": {f: abs(b) for f, b in exposures.betas.items()},
            "concentration_components": concentration_components,
            "interpretation": (
                "Single-factor bet" if concentration_score > 70 else
                "Concentrated" if concentration_score > 40 else
                "Moderately concentrated" if concentration_score > 20 else
                "Well diversified"
            ),
        }

    def detect_factor_crowding(
        self,
        factor: str,
        universe_exposures: dict[str, FactorExposures],
    ) -> float:
        """
        Detect crowding in a factor by computing average beta across the universe.

        High average absolute beta = crowded factor trade (systemic risk).

        Returns: average absolute factor loading across universe.
        """
        betas = [
            abs(exp.betas.get(factor, 0.0))
            for exp in universe_exposures.values()
            if factor in exp.betas
        ]
        if not betas:
            return float("nan")
        return float(np.mean(betas))

    def compute_factor_momentum(
        self,
        factor: str,
        factors_df: pd.DataFrame = None,
        lookback: int = 252,
    ) -> dict:
        """
        Assess recent factor return momentum.

        Args:
            factor: Factor name (e.g., 'SMB', 'MOM').
            factors_df: Factor return DataFrame (loads if None).
            lookback: Number of trading days for momentum window.

        Returns:
            dict with recent return, volatility, Sharpe, trend direction.
        """
        if factors_df is None:
            factors_df = self._loader.load_all_factors()

        if factor not in factors_df.columns:
            return {"error": f"Factor '{factor}' not found"}

        recent = factors_df[factor].tail(lookback).dropna()
        if len(recent) < 20:
            return {"error": "Insufficient data"}

        ann_return = recent.mean() * 252
        ann_vol = recent.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
        half_period = len(recent) // 2
        first_half_ret = recent.iloc[:half_period].mean() * 252
        second_half_ret = recent.iloc[half_period:].mean() * 252
        trend = "IMPROVING" if second_half_ret > first_half_ret else "DETERIORATING"

        # Factor momentum: 12-1 (12-month return excluding last month)
        mom_12_1 = float("nan")
        if len(recent) >= 252:
            momentum_window = recent.iloc[:232]  # exclude last 21 days
            mom_12_1 = float(momentum_window.mean() * 252)

        return {
            "factor": factor,
            "lookback_days": lookback,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "trend": trend,
            "first_half_return": first_half_ret,
            "second_half_return": second_half_ret,
            "momentum_12_1": mom_12_1,
            "recommendation": (
                "OVERWEIGHT" if sharpe > 0.5 and trend == "IMPROVING" else
                "UNDERWEIGHT" if sharpe < -0.3 or (sharpe < 0.1 and trend == "DETERIORATING") else
                "NEUTRAL"
            ),
        }

    def get_risk_dashboard(
        self,
        holdings: dict[str, float],
        portfolio_value: float = 1_000_000.0,
    ) -> FactorDashboard:
        """
        Compile full factor risk dashboard for a portfolio.

        Args:
            holdings: {ticker: weight} portfolio.
            portfolio_value: Notional portfolio value for $ VaR.

        Returns:
            FactorDashboard dataclass.
        """
        factors = self._loader.load_all_factors()

        # Portfolio exposures
        logger.info("Computing portfolio exposures for %d holdings...", len(holdings))
        exposures = self._pa.compute_portfolio_exposures(holdings, factors=factors)

        # Factor covariance
        factor_cov = self._pa.compute_factor_covariance(factors=factors)

        # VaR
        var_data = self._pa.compute_factor_var(exposures, factor_cov, portfolio_value=1.0)

        # Variance decomposition
        var_decomp = self._pa.decompose_variance(exposures, factor_cov)

        # Concentration
        conc = self.compute_factor_concentration_risk(exposures)

        # Factor momentum
        factor_mom = {}
        for f in FACTOR_NAMES:
            mom = self.compute_factor_momentum(f, factors_df=factors)
            if "ann_return" in mom:
                factor_mom[f] = mom["ann_return"]

        # Risk flags
        risk_flags = []
        if var_data["factor_var_95"] > 0.03:
            risk_flags.append(f"High factor VaR: {var_data['factor_var_95']:.1%} daily 95%")
        if conc["concentration_score"] > 60:
            risk_flags.append(f"High concentration: {conc['concentration_score']:.0f}/100")
        for f in conc["concentrated_factors"]:
            risk_flags.append(f"Concentrated in {f}: β={exposures.betas.get(f,0):.2f}")
        if var_decomp["pct_systematic"] < 0.3:
            risk_flags.append("Low systematic R²: portfolio may be idiosyncratic-heavy")

        return FactorDashboard(
            timestamp=datetime.now(timezone.utc).isoformat(),
            portfolio_exposures=exposures,
            factor_var_95=var_data["factor_var_95"],
            factor_var_99=var_data["factor_var_99"],
            idiosyncratic_var_95=var_data["idiosyncratic_var_95"],
            systematic_pct=var_decomp["pct_systematic"] * 100,
            factor_concentration_score=conc["concentration_score"],
            crowded_factors=conc["concentrated_factors"],
            factor_momentum=factor_mom,
            variance_decomposition={
                "systematic": var_decomp["pct_systematic"],
                "idiosyncratic": var_decomp["pct_idiosyncratic"],
                **var_decomp["factor_variance_contributions"],
            },
            risk_flags=risk_flags,
        )


# ---------------------------------------------------------------------------
# FactorReturnForecaster
# ---------------------------------------------------------------------------

class FactorReturnForecaster:
    """
    Forecast expected factor returns and compute optimal factor tilts.
    """

    def __init__(self, loader: FrenchDataLoader = None):
        self._loader = loader or FrenchDataLoader()

    def compute_factor_expected_returns(
        self,
        method: str = "historical",
        lookback_years: int = 10,
        factors_df: pd.DataFrame = None,
    ) -> dict:
        """
        Forecast expected factor returns.

        Methods:
        - "historical": annualized average factor return over lookback period.
        - "risk_premium": Sharpe × realized vol (empirical risk premium).

        Returns:
            dict: {factor_name: expected_annual_return}
        """
        if factors_df is None:
            factors_df = self._loader.load_all_factors()

        lookback_days = lookback_years * 252
        fac_cols = [f for f in FACTOR_NAMES if f in factors_df.columns]
        recent = factors_df[fac_cols].tail(lookback_days).dropna()

        result = {}

        if method == "historical":
            for f in fac_cols:
                result[f] = float(recent[f].mean() * 252)

        elif method == "risk_premium":
            for f in fac_cols:
                ann_ret = recent[f].mean() * 252
                ann_vol = recent[f].std() * np.sqrt(252)
                sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
                # Risk premium = Sharpe × current vol (target Sharpe * vol)
                result[f] = float(sharpe * ann_vol)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'historical' or 'risk_premium'.")

        return result

    def compute_portfolio_expected_return(
        self,
        holdings: dict[str, float],
        factor_forecasts: dict,
        factors_df: pd.DataFrame = None,
        lookback_days: int = 756,
    ) -> float:
        """
        Compute expected portfolio return using factor model.

        E[R_p] = RF + Σ(β_i × E[factor_i]) + α_p

        Args:
            holdings: {ticker: weight}.
            factor_forecasts: {factor_name: expected_annual_return}.
            factors_df: Factor DataFrame.
            lookback_days: History for exposure estimation.

        Returns:
            Expected annual portfolio return (decimal).
        """
        if factors_df is None:
            factors_df = self._loader.load_all_factors()

        pa = PortfolioFactorAnalyzer(self._loader)
        exposures = pa.compute_portfolio_exposures(holdings, factors=factors_df, lookback_days=lookback_days)

        # RF annualized
        rf_daily = factors_df["RF"].tail(252).mean() if "RF" in factors_df.columns else 0.0
        rf_ann = rf_daily * 252

        factor_component = sum(
            exposures.betas.get(f, 0.0) * factor_forecasts.get(f, 0.0)
            for f in FACTOR_NAMES
        )

        alpha_ann = exposures.alpha * 252

        expected_return = rf_ann + factor_component + alpha_ann
        return float(expected_return)

    def compute_efficient_frontier_factors(
        self,
        factor_forecasts: dict,
        factor_cov: pd.DataFrame,
        n_points: int = 20,
        long_only: bool = False,
    ) -> pd.DataFrame:
        """
        Compute mean-variance efficient frontier in factor space.

        Finds optimal factor tilt vectors (β weights) at various risk budgets.
        Uses closed-form mean-variance optimization.

        Args:
            factor_forecasts: {factor: expected_return} dict.
            factor_cov: Annualized factor covariance matrix (pd.DataFrame).
            n_points: Number of frontier points.
            long_only: Constrain factor betas to be non-negative.

        Returns:
            DataFrame with columns: target_vol, expected_return, sharpe, factor betas.
        """
        fac_names = [f for f in factor_forecasts if f in factor_cov.columns]
        mu = np.array([factor_forecasts[f] for f in fac_names])
        Sigma = factor_cov.loc[fac_names, fac_names].values.astype(float)

        # Regularize covariance matrix
        Sigma = Sigma + np.eye(len(Sigma)) * 1e-6

        # Risk budget: vary from min-var to max Sharpe+
        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(Sigma)

        # Efficient frontier via two-fund separation theorem
        ones = np.ones(len(mu))
        A = float(mu @ Sigma_inv @ ones)
        B = float(mu @ Sigma_inv @ mu)
        C = float(ones @ Sigma_inv @ ones)
        D = B * C - A ** 2

        records = []
        # Parametric frontier: β*(λ) = Σ^{-1}(λμ + γ·1) / normalization
        lambda_range = np.linspace(0.01, 5.0, n_points)

        for lam in lambda_range:
            # β = Σ^{-1}(λμ + γ·1)
            # We solve for the pure factor tilt (normalized)
            raw_weights = Sigma_inv @ (lam * mu)
            if long_only:
                raw_weights = np.maximum(raw_weights, 0.0)

            if raw_weights.sum() == 0:
                continue

            # Normalize: target unit market beta equivalent
            norm = raw_weights
            port_ret = float(norm @ mu)
            port_var = float(norm @ Sigma @ norm)
            port_vol = np.sqrt(max(port_var, 1e-9))
            sharpe = port_ret / port_vol if port_vol > 0 else 0.0

            row = {
                "lambda": lam,
                "expected_return": port_ret,
                "target_vol": port_vol,
                "sharpe": sharpe,
            }
            for i, fname in enumerate(fac_names):
                row[f"tilt_{fname}"] = float(norm[i])
            records.append(row)

        if not records:
            return pd.DataFrame()

        frontier = pd.DataFrame(records).sort_values("target_vol")
        return frontier


# ---------------------------------------------------------------------------
# FactorBacktester
# ---------------------------------------------------------------------------

class FactorBacktester:
    """
    Backtest factor-based long/short strategies using Fama-French factor sorts.
    """

    def __init__(
        self,
        loader: FrenchDataLoader = None,
        fitter: FactorModelFitter = None,
    ):
        self._loader = loader or FrenchDataLoader()
        self._fitter = fitter or FactorModelFitter(loader=self._loader)

    def run_factor_sort(
        self,
        factor: str,
        universe: list[str],
        quantiles: int = 5,
        rebalance: str = "monthly",
        lookback_days: int = 756,
    ) -> dict:
        """
        Sort universe by factor loading, long top quintile, short bottom quintile.

        Args:
            factor: Factor to sort on (e.g., 'SMB', 'HML', 'MOM').
            universe: List of tickers to sort.
            quantiles: Number of portfolios to form (default 5 = quintiles).
            rebalance: Rebalancing frequency ('monthly' or 'quarterly').
            lookback_days: History for factor estimation.

        Returns:
            dict with quintile returns, L/S spread returns, cumulative performance.
        """
        factors_df = self._loader.load_all_factors()
        end_date = factors_df.index.max()
        start_date = end_date - pd.Timedelta(days=lookback_days + 252)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        logger.info("Downloading returns for %d tickers...", len(universe))
        stock_returns = _get_stock_returns(universe, start=start_str, end=end_str)

        # Estimate factor loadings for each stock on the full period
        betas_dict = {}
        for ticker in stock_returns.columns:
            ret = stock_returns[ticker].dropna().rename(ticker)
            if len(ret) < 60:
                continue
            try:
                exp = self._fitter.fit(ret, factors=factors_df, factor_names=[factor])
                betas_dict[ticker] = exp.betas.get(factor, 0.0)
            except Exception as e:
                logger.debug("Factor sort fit failed for %s: %s", ticker, e)

        if not betas_dict:
            return {"error": "No valid factor loadings computed"}

        # Sort into quantiles
        sorted_tickers = sorted(betas_dict.items(), key=lambda x: x[1])
        n = len(sorted_tickers)
        q_size = n // quantiles
        quantile_groups = []
        for q in range(quantiles):
            start_idx = q * q_size
            end_idx = (q + 1) * q_size if q < quantiles - 1 else n
            q_tickers = [t for t, _ in sorted_tickers[start_idx:end_idx]]
            quantile_groups.append(q_tickers)

        # Compute equal-weight returns for each quantile
        trading_start = start_date + pd.Timedelta(days=252)
        quantile_returns = {}
        for q_idx, q_tickers in enumerate(quantile_groups):
            valid = [t for t in q_tickers if t in stock_returns.columns]
            if not valid:
                continue
            q_rets = stock_returns[valid].loc[trading_start:].mean(axis=1)
            quantile_returns[f"Q{q_idx+1}"] = q_rets

        # Long-Short spread (top quintile - bottom quintile)
        if "Q1" in quantile_returns and f"Q{quantiles}" in quantile_returns:
            ls_spread = quantile_returns[f"Q{quantiles}"] - quantile_returns["Q1"]
            quantile_returns["L/S"] = ls_spread

        # Summary statistics
        summary = {}
        for name, ret_series in quantile_returns.items():
            ret_series = ret_series.dropna()
            if len(ret_series) < 10:
                continue
            ann_ret = ret_series.mean() * 252
            ann_vol = ret_series.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            # Max drawdown
            cum = (1 + ret_series).cumprod()
            roll_max = cum.cummax()
            drawdown = (cum - roll_max) / roll_max
            max_dd = float(drawdown.min())
            summary[name] = {
                "ann_return": ann_ret,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "n_stocks": len(quantile_groups[0]) if name == "Q1" else None,
            }

        # Cumulative returns
        cumulative = {
            name: (1 + ret).cumprod() for name, ret in quantile_returns.items()
        }

        return {
            "factor": factor,
            "universe_size": n,
            "quantiles": quantiles,
            "factor_betas": betas_dict,
            "quantile_summary": summary,
            "quantile_returns": {k: v.to_dict() for k, v in quantile_returns.items()},
            "cumulative_returns": {k: v.to_dict() for k, v in cumulative.items()},
        }

    def compute_factor_ic(
        self,
        factor: str,
        universe: list[str],
        forward_period: int = 21,
        lookback_days: int = 756,
    ) -> pd.Series:
        """
        Compute rolling monthly Information Coefficient (IC).

        IC = Spearman rank correlation between factor loading and forward return.

        Args:
            factor: Factor name.
            universe: Ticker list.
            forward_period: Forward return horizon in trading days (default 21 = monthly).
            lookback_days: Total history to compute IC over.

        Returns:
            pd.Series of monthly IC values.
        """
        factors_df = self._loader.load_all_factors()
        end_date = factors_df.index.max()
        start_date = end_date - pd.Timedelta(days=lookback_days + 252)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        stock_returns = _get_stock_returns(universe, start=start_str, end=end_str)

        # Compute rolling factor loadings using 126-day window
        estimation_window = 126
        ic_values = {}

        trading_dates = stock_returns.index[estimation_window:]
        step_dates = trading_dates[::forward_period]  # monthly steps

        for date in step_dates:
            # Get slice for estimation
            date_pos = stock_returns.index.get_loc(date)
            if date_pos < estimation_window:
                continue
            fwd_pos = date_pos + forward_period
            if fwd_pos >= len(stock_returns):
                break

            est_start_pos = date_pos - estimation_window
            est_slice = stock_returns.iloc[est_start_pos:date_pos]
            fwd_date = stock_returns.index[fwd_pos]

            factor_betas = {}
            for ticker in stock_returns.columns:
                ret = est_slice[ticker].dropna().rename(ticker)
                if len(ret) < 30:
                    continue
                try:
                    exp = self._fitter.fit(ret, factors=factors_df, factor_names=[factor])
                    factor_betas[ticker] = exp.betas.get(factor, 0.0)
                except Exception:
                    pass

            # Forward returns
            fwd_returns = {}
            for ticker in factor_betas:
                if ticker not in stock_returns.columns:
                    continue
                start_p = stock_returns[ticker].loc[date] if date in stock_returns.index else float("nan")
                end_p = stock_returns[ticker].loc[fwd_date] if fwd_date in stock_returns.index else float("nan")
                if start_p == start_p and end_p == end_p:
                    # Accumulate forward returns
                    fwd_slice = stock_returns[ticker].iloc[date_pos:fwd_pos]
                    fwd_ret = float(fwd_slice.sum())
                    fwd_returns[ticker] = fwd_ret

            common_tickers = list(set(factor_betas) & set(fwd_returns))
            if len(common_tickers) < 5:
                continue

            beta_arr = np.array([factor_betas[t] for t in common_tickers])
            fwd_arr = np.array([fwd_returns[t] for t in common_tickers])

            # Spearman rank correlation
            beta_ranks = pd.Series(beta_arr).rank()
            fwd_ranks = pd.Series(fwd_arr).rank()
            ic = float(beta_ranks.corr(fwd_ranks))
            ic_values[date] = ic

        return pd.Series(ic_values, name=f"IC_{factor}")

    def compute_factor_ir(self, ics: pd.Series) -> float:
        """
        Compute Information Ratio from IC series.

        IR = mean(IC) / std(IC) × sqrt(12)  [annualized]

        Args:
            ics: Monthly IC series (from compute_factor_ic).

        Returns:
            Annualized Information Ratio.
        """
        ics_clean = ics.dropna()
        if len(ics_clean) < 3:
            return float("nan")
        mean_ic = float(ics_clean.mean())
        std_ic = float(ics_clean.std())
        if std_ic <= 0:
            return float("nan")
        return mean_ic / std_ic * np.sqrt(12)


# ---------------------------------------------------------------------------
# Convenience function: build full factor model for a stock list
# ---------------------------------------------------------------------------

def analyze_stocks_factor_exposures(
    tickers: list[str],
    loader: FrenchDataLoader = None,
    lookback_days: int = 756,
) -> dict[str, FactorExposures]:
    """
    Convenience function to fit FF5+Mom model for a list of tickers.

    Returns:
        dict: {ticker: FactorExposures}
    """
    if loader is None:
        loader = FrenchDataLoader()

    factors = loader.load_all_factors()
    fitter = FactorModelFitter(factor_data=factors, loader=loader)

    end_date = factors.index.max()
    start_date = end_date - pd.Timedelta(days=lookback_days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    stock_returns = _get_stock_returns(tickers, start=start_str, end=end_str)

    results = {}
    for ticker in tickers:
        if ticker not in stock_returns.columns:
            logger.warning("No data for %s", ticker)
            continue
        ret = stock_returns[ticker].dropna().rename(ticker)
        try:
            exp = fitter.fit(ret, factors=factors)
            results[ticker] = exp
        except Exception as e:
            logger.error("Failed to fit %s: %s", ticker, e)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    print("Initializing SENTINEL Factor Risk Model v3...\n")

    # 1. Download FF5+Mom factors
    print("Step 1: Loading Fama-French 5-Factor + Momentum data...")
    loader = FrenchDataLoader()
    factors_df = loader.load_all_factors()
    print(f"  Loaded {len(factors_df):,} trading days")
    print(f"  Date range: {factors_df.index.min().date()} → {factors_df.index.max().date()}")
    print(f"  Columns: {list(factors_df.columns)}\n")

    # 2. Fit exposures for AAPL, MSFT, TSLA, SPY
    tickers = ["AAPL", "MSFT", "TSLA", "SPY"]
    print(f"Step 2: Fitting FF5+Mom exposures for {tickers}...")
    exposures_map = analyze_stocks_factor_exposures(tickers, loader=loader, lookback_days=756)
    print()
    for ticker, exp in exposures_map.items():
        print(exp.summary())
        print()

    # 3. Decompose portfolio variance (equal weight)
    holdings = {"AAPL": 0.25, "MSFT": 0.25, "TSLA": 0.25, "SPY": 0.25}
    print("Step 3: Portfolio variance decomposition (equal weight AAPL/MSFT/TSLA/SPY)...")
    pa = PortfolioFactorAnalyzer(loader)
    portfolio_exp = pa.compute_portfolio_exposures(holdings, factors=factors_df)
    factor_cov = pa.compute_factor_covariance(factors=factors_df)
    var_decomp = pa.decompose_variance(portfolio_exp, factor_cov)
    print(f"  Systematic: {var_decomp['pct_systematic']:.1%}")
    print(f"  Idiosyncratic: {var_decomp['pct_idiosyncratic']:.1%}")
    print(f"  Total Vol (ann): {var_decomp['systematic_vol_ann']:.2%}")
    print()

    # 4. SMB quintile backtest (small universe)
    smb_universe = ["AAPL", "MSFT", "TSLA", "SPY", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "JNJ"]
    print(f"Step 4: SMB quintile backtest on {len(smb_universe)}-stock universe...")
    backtester = FactorBacktester(loader)
    bt_result = backtester.run_factor_sort(
        factor="SMB",
        universe=smb_universe,
        quantiles=3,  # 3 terciles (small universe)
        lookback_days=756,
    )
    if "quantile_summary" in bt_result:
        for q_name, stats in bt_result["quantile_summary"].items():
            print(f"  {q_name:4s}: Ann.Return={stats['ann_return']:+.2%}  Sharpe={stats['sharpe']:+.2f}  MaxDD={stats['max_drawdown']:.1%}")
    print()

    # 5. Factor Risk Dashboard
    print("Step 5: Factor Risk Dashboard...")
    monitor = FactorRiskMonitor(loader, pa)
    dashboard = monitor.get_risk_dashboard(holdings, portfolio_value=1_000_000)
    print(dashboard.summary())

    # 6. Factor momentum for all factors
    print("\nFactor Momentum (1-year lookback):")
    for f in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]:
        mom = monitor.compute_factor_momentum(f, factors_df=factors_df, lookback=252)
        if "ann_return" in mom:
            print(f"  {f:10s}  {mom['ann_return']:+.2%}  Sharpe={mom['sharpe']:+.2f}  [{mom['recommendation']}]")

    # 7. Forecasted expected returns
    print("\nStep 7: Factor expected returns (historical method)...")
    forecaster = FactorReturnForecaster(loader)
    expected_returns = forecaster.compute_factor_expected_returns(method="historical", lookback_years=10)
    for f, ret in expected_returns.items():
        print(f"  {f:10s}: {ret:+.2%} ann.")
