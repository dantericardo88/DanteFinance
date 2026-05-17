"""
sentinel/spm/correlation_monitor_v3.py
=======================================
Correlation monitoring, regime detection, and pair-trading alerts.
dim_080 — score 6 → 9

Free data only:
  - yfinance for price history
  - FRED CSV for recession indicator (USREC)
  - scipy / statsmodels guarded with try/except

Author: SENTINEL Correlation Monitor
"""

from __future__ import annotations

import io
import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional / guarded imports
# ---------------------------------------------------------------------------
try:
    from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore
    from scipy.spatial.distance import squareform  # type: ignore

    _SCIPY_CLUSTER = True
except ImportError:
    _SCIPY_CLUSTER = False

try:
    from scipy.optimize import minimize  # type: ignore
    from scipy.stats import spearmanr  # type: ignore

    _SCIPY_STATS = True
except ImportError:
    _SCIPY_STATS = False

try:
    from statsmodels.tsa.stattools import adfuller, coint  # type: ignore

    _STATSMODELS = True
except ImportError:
    _STATSMODELS = False

try:
    import yfinance as yf

    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

try:
    import requests

    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

log = logging.getLogger("sentinel.spm.correlation_monitor")

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CorrelationEvent:
    """A detected correlation regime event."""

    date: str = ""
    event_type: str = ""       # HIGH_CORR | LOW_CORR | SPIKE | COLLAPSE
    avg_correlation: float = 0.0
    threshold: float = 0.0
    description: str = ""
    in_recession: bool = False


@dataclass
class CointegrationResult:
    """Result from Engle-Granger cointegration test."""

    ticker1: str = ""
    ticker2: str = ""
    is_cointegrated: bool = False
    p_value: float = 1.0
    adf_stat: float = 0.0
    hedge_ratio: float = 1.0
    half_life_days: float = 0.0
    method: str = "engle-granger"


@dataclass
class PairResult:
    """A cointegrated trading pair with spread statistics."""

    ticker1: str = ""
    ticker2: str = ""
    p_value: float = 1.0
    hedge_ratio: float = 1.0
    spread_mean: float = 0.0
    spread_std: float = 0.0
    half_life_days: float = 0.0
    current_z_score: float = 0.0
    signal: str = "NEUTRAL"


@dataclass
class CorrelationAlert:
    """A fired alert from the monitoring system."""

    alert_type: str = ""
    severity: str = "INFO"       # INFO | WARN | CRITICAL
    message: str = ""
    timestamp: str = ""
    affected_assets: List[str] = field(default_factory=list)
    metric_value: float = 0.0
    threshold: float = 0.0


@dataclass
class CorrelationReport:
    """Full correlation monitoring report."""

    as_of_date: str = ""
    universe: List[str] = field(default_factory=list)
    avg_pairwise_corr: float = 0.0
    corr_percentile: float = 0.0
    regime: str = "UNKNOWN"
    cross_asset_regime: str = "UNKNOWN"
    risk_off_signal: bool = False
    stock_bond_corr: float = 0.0
    diversification_ratio: float = 0.0
    effective_n: float = 0.0
    hhi: float = 0.0
    alerts: List[CorrelationAlert] = field(default_factory=list)
    cointegrated_pairs: List[PairResult] = field(default_factory=list)
    correlation_matrix: Optional[pd.DataFrame] = None
    clusters: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class PortfolioCorrelationSummary:
    """Correlation summary for a specific portfolio."""

    holdings: Dict[str, float] = field(default_factory=dict)
    internal_avg_corr: float = 0.0
    diversification_ratio: float = 0.0
    effective_n: float = 0.0
    hhi: float = 0.0
    regime: str = "UNKNOWN"
    alerts: List[CorrelationAlert] = field(default_factory=list)
    component_correlations: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Price data helper
# ---------------------------------------------------------------------------

_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _fetch_prices(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download adjusted close prices from yfinance, return as DataFrame."""
    if not _YF_AVAILABLE:
        raise ImportError("yfinance required: pip install yfinance")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        col = tickers[0] if len(tickers) == 1 else "Close"
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.ffill().dropna(how="all")
    return prices


def _prices_to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert price DataFrame to log return DataFrame."""
    return np.log(prices / prices.shift(1)).dropna()


def _fetch_usrec() -> pd.Series:
    """Fetch NBER recession indicator from FRED (USREC, monthly)."""
    url = f"{_FRED_BASE}?id=USREC"
    try:
        if _REQUESTS_AVAILABLE:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            txt = resp.text
        else:
            import urllib.request

            with urllib.request.urlopen(url, timeout=15) as r:
                txt = r.read().decode()

        df = pd.read_csv(io.StringIO(txt), parse_dates=["DATE"], index_col="DATE")
        df.columns = ["USREC"]
        return df["USREC"].astype(int)
    except Exception as exc:
        log.warning("Could not fetch USREC from FRED: %s", exc)
        return pd.Series(dtype=int)


# ---------------------------------------------------------------------------
# CorrelationComputer
# ---------------------------------------------------------------------------


class CorrelationComputer:
    """Compute various correlation matrices from returns data."""

    # ------------------------------------------------------------------
    @staticmethod
    def compute_pearson(
        returns: pd.DataFrame, window: Optional[int] = None
    ) -> pd.DataFrame:
        """Full-sample or windowed Pearson correlation matrix."""
        if window:
            return returns.tail(window).corr(method="pearson")
        return returns.corr(method="pearson")

    # ------------------------------------------------------------------
    @staticmethod
    def compute_spearman(
        returns: pd.DataFrame, window: Optional[int] = None
    ) -> pd.DataFrame:
        """Rank-based Spearman correlation matrix."""
        r = returns.tail(window) if window else returns
        if _SCIPY_STATS:
            try:
                n = r.shape[1]
                mat = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            mat[i, j] = 1.0
                        elif i < j:
                            rho, _ = spearmanr(r.iloc[:, i], r.iloc[:, j])
                            mat[i, j] = rho
                            mat[j, i] = rho
                return pd.DataFrame(mat, index=r.columns, columns=r.columns)
            except Exception:
                pass
        # Fallback: rank then pearson
        ranked = r.rank()
        return ranked.corr(method="pearson")

    # ------------------------------------------------------------------
    @staticmethod
    def compute_rolling_correlation(
        r1: pd.Series,
        r2: pd.Series,
        window: int = 60,
    ) -> pd.Series:
        """Rolling Pearson correlation between two return series."""
        combined = pd.concat([r1, r2], axis=1).dropna()
        if combined.shape[1] < 2 or len(combined) < window:
            return pd.Series(dtype=float)
        a = combined.iloc[:, 0]
        b = combined.iloc[:, 1]
        return a.rolling(window).corr(b)

    # ------------------------------------------------------------------
    @staticmethod
    def compute_ewm_correlation(
        returns: pd.DataFrame, span: int = 60
    ) -> pd.DataFrame:
        """
        EWMA correlation matrix.
        Compute via ewm covariance then normalize by ewm variances.
        """
        r = returns.dropna()
        n = r.shape[1]
        # EWMA covariance
        ewm_cov = r.ewm(span=span, adjust=True).cov()
        # Extract last covariance matrix
        last_date = ewm_cov.index.get_level_values(0)[-1]
        cov_mat = ewm_cov.loc[last_date]

        # Convert to correlation
        cols = r.columns
        diag = np.sqrt(np.diag(cov_mat.values))
        diag = np.where(diag < 1e-12, 1.0, diag)
        outer = np.outer(diag, diag)
        corr_mat = cov_mat.values / outer
        np.fill_diagonal(corr_mat, 1.0)
        corr_mat = np.clip(corr_mat, -1.0, 1.0)
        return pd.DataFrame(corr_mat, index=cols, columns=cols)

    # ------------------------------------------------------------------
    def compute_dynamic_conditional_correlation(
        self, returns: pd.DataFrame
    ) -> dict:
        """
        Simplified DCC-GARCH:
          Step 1: GARCH(1,1)-standardise each series (or std-normalise).
          Step 2: EWMA correlation of standardised residuals.

        Returns dict with:
          "corr_matrix": latest correlation DataFrame
          "corr_history": dict of rolling avg pairwise corr by date
        """
        r = returns.dropna()

        # Step 1: Standardise residuals
        std_resids = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
        for col in r.columns:
            series = r[col].values
            # Simple GARCH-like: use rolling std as conditional vol proxy
            roll_std = r[col].ewm(span=60).std().values
            roll_std = np.where(roll_std < 1e-10, 1e-10, roll_std)
            std_resids[col] = series / roll_std

        # Step 2: EWMA correlation of standardised residuals
        ewm_corr = self.compute_ewm_correlation(std_resids, span=60)

        # Build rolling avg pairwise history
        tickers = list(r.columns)
        n = len(tickers)
        history: Dict[str, float] = {}
        for date in r.index[-252:]:
            try:
                date_str = date.strftime("%Y-%m-%d")
                window_end = date
                window_start = r.index[max(0, r.index.get_loc(date) - 60)]
                sub = std_resids.loc[window_start:window_end]
                if len(sub) < 5:
                    continue
                cm = sub.corr()
                n_pairs = n * (n - 1) / 2
                if n_pairs > 0:
                    vals = [cm.iloc[i, j] for i in range(n) for j in range(i + 1, n)]
                    history[date_str] = float(np.mean(vals))
            except Exception:
                continue

        return {
            "corr_matrix": ewm_corr,
            "corr_history": history,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def compute_realized_correlation(
        returns: pd.DataFrame, freq: str = "5min"
    ) -> pd.DataFrame:
        """
        Realized correlation. Falls back to daily if intraday not available.
        Uses daily data resampled to requested freq as proxy.
        """
        if freq in ("5min", "1h", "15min"):
            log.debug("Intraday data not available; using daily as realized correlation proxy")
        # Daily realized: straightforward Pearson
        return returns.corr(method="pearson")


# ---------------------------------------------------------------------------
# CorrelationRegimeDetector
# ---------------------------------------------------------------------------


class CorrelationRegimeDetector:
    """Detect correlation regimes: risk-on vs risk-off."""

    def __init__(self):
        self._usrec: Optional[pd.Series] = None

    def _get_usrec(self) -> pd.Series:
        if self._usrec is None:
            self._usrec = _fetch_usrec()
        return self._usrec

    # ------------------------------------------------------------------
    def compute_average_pairwise_correlation(
        self,
        returns: pd.DataFrame,
        window: int = 60,
    ) -> pd.Series:
        """
        Rolling average pairwise correlation — single number per date
        summarising overall market correlation level.
        """
        tickers = list(returns.columns)
        n = len(tickers)
        if n < 2:
            return pd.Series(dtype=float)

        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        result = {}
        r = returns.dropna()
        dates = r.index[window - 1 :]

        for date in dates:
            loc = r.index.get_loc(date)
            start_loc = max(0, loc - window + 1)
            sub = r.iloc[start_loc : loc + 1]
            cm = sub.corr()
            corrs = [float(cm.iloc[i, j]) for i, j in pairs]
            result[date] = float(np.mean(corrs)) if corrs else 0.0

        return pd.Series(result, name="avg_pairwise_corr")

    # ------------------------------------------------------------------
    def detect_correlation_spike(
        self,
        avg_corr: pd.Series,
        threshold: float = 0.7,
        low_threshold: float = 0.3,
    ) -> List[CorrelationEvent]:
        """
        Detect HIGH_CORR (risk-off) and LOW_CORR (risk-on) regimes.
        Returns list of events where regime starts or intensifies.
        """
        usrec = self._get_usrec()
        events: List[CorrelationEvent] = []
        prev_regime = "NORMAL"

        for date, corr in avg_corr.items():
            # Recession overlay
            in_rec = False
            if usrec is not None and len(usrec) > 0:
                try:
                    month_start = pd.Timestamp(date).to_period("M").to_timestamp()
                    in_rec = bool(usrec.reindex([month_start], method="ffill").iloc[0] == 1)
                except Exception:
                    pass

            if corr > threshold:
                regime = "HIGH_CORR"
            elif corr < low_threshold:
                regime = "LOW_CORR"
            else:
                regime = "NORMAL"

            if regime != prev_regime and regime != "NORMAL":
                event_type = "SPIKE" if regime == "HIGH_CORR" else "COLLAPSE"
                desc = (
                    f"Avg pairwise corr {corr:.3f} crossed "
                    f"{'upper' if regime=='HIGH_CORR' else 'lower'} threshold "
                    f"{'during recession' if in_rec else ''}"
                )
                events.append(
                    CorrelationEvent(
                        date=date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                        event_type=event_type,
                        avg_correlation=corr,
                        threshold=threshold if regime == "HIGH_CORR" else low_threshold,
                        description=desc.strip(),
                        in_recession=in_rec,
                    )
                )
            prev_regime = regime if regime != "NORMAL" else prev_regime

        return events

    # ------------------------------------------------------------------
    def compute_correlation_percentile(
        self,
        current_corr: float,
        avg_corr_history: pd.Series,
        lookback_days: int = 252,
    ) -> float:
        """What percentile is the current correlation level in its history?"""
        hist = avg_corr_history.dropna().tail(lookback_days).values
        if len(hist) == 0:
            return 50.0
        pct = float(np.mean(hist <= current_corr)) * 100.0
        return round(pct, 1)

    # ------------------------------------------------------------------
    @staticmethod
    def detect_correlation_breakdown(
        r1: pd.Series,
        r2: pd.Series,
        short_window: int = 20,
        long_window: int = 90,
        threshold: float = 0.40,
    ) -> bool:
        """
        Detect if a pair that was historically correlated has recently diverged.
        Returns True if breakdown detected (potential arb or regime shift).
        """
        combined = pd.concat([r1, r2], axis=1).dropna()
        if len(combined) < long_window:
            return False
        a, b = combined.iloc[:, 0], combined.iloc[:, 1]
        long_corr = float(a.tail(long_window).corr(b.tail(long_window)))
        short_corr = float(a.tail(short_window).corr(b.tail(short_window)))
        # Breakdown: long-run was correlated, recent is not
        return long_corr > 0.5 and (long_corr - short_corr) > threshold


# ---------------------------------------------------------------------------
# DiversificationAnalyzer
# ---------------------------------------------------------------------------


class DiversificationAnalyzer:
    """Analyse portfolio diversification quality."""

    # ------------------------------------------------------------------
    @staticmethod
    def compute_diversification_ratio(
        returns: pd.DataFrame, weights: np.ndarray
    ) -> float:
        """
        Diversification Ratio = weighted avg individual vol / portfolio vol.
        DR > 1 implies diversification benefit.
        """
        w = np.asarray(weights, float)
        w = w / w.sum()
        r = returns.dropna()

        individual_vols = np.array([float(r.iloc[:, i].std()) for i in range(r.shape[1])])
        weighted_avg_vol = float(w @ individual_vols)

        cov = np.cov(r.values.T)
        port_vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))

        return weighted_avg_vol / port_vol if port_vol > 0 else 1.0

    # ------------------------------------------------------------------
    @staticmethod
    def compute_effective_n(
        correlation_matrix: pd.DataFrame,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Effective number of independent bets.
        Uses eigenvalue decomposition of correlation matrix.
        Method: 1 / sum(lambda_i / sum(lambda)) ^ 2
        """
        corr = correlation_matrix.values.copy()
        n = corr.shape[0]
        # Regularise
        np.fill_diagonal(corr, 1.0)
        try:
            eigenvalues = np.linalg.eigvalsh(corr)
            eigenvalues = np.maximum(eigenvalues, 0)
            total = eigenvalues.sum()
            if total < 1e-12:
                return float(n)
            fracs = eigenvalues / total
            eff_n = 1.0 / float(np.sum(fracs**2))
            return round(eff_n, 2)
        except Exception:
            return float(n)

    # ------------------------------------------------------------------
    @staticmethod
    def compute_hhi(weights: np.ndarray) -> float:
        """Herfindahl-Hirschman Index: sum of squared weights. Range [1/N, 1]."""
        w = np.asarray(weights, float)
        w = w / w.sum()
        return float(np.sum(w**2))

    # ------------------------------------------------------------------
    def find_minimum_correlation_portfolio(
        self, returns: pd.DataFrame
    ) -> np.ndarray:
        """
        Portfolio that maximises the diversification ratio.
        Uses scipy.optimize if available; else simple gradient descent.
        """
        r = returns.dropna()
        n = r.shape[1]
        w0 = np.ones(n) / n

        def neg_dr(w):
            w = np.abs(w)
            w /= w.sum() + 1e-12
            return -self.compute_diversification_ratio(r, w)

        if _SCIPY_STATS:
            try:
                constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
                bounds = [(0.01, 1.0)] * n
                res = minimize(neg_dr, w0, method="SLSQP", bounds=bounds, constraints=constraints)
                if res.success:
                    w_opt = np.abs(res.x)
                    return w_opt / w_opt.sum()
            except Exception as exc:
                log.debug("scipy optimize for MCP failed: %s", exc)

        # Gradient descent fallback
        w = w0.copy()
        lr = 0.01
        for _ in range(500):
            grad = np.zeros(n)
            for i in range(n):
                w_plus = w.copy()
                w_plus[i] += 1e-5
                w_plus /= w_plus.sum()
                w_minus = w.copy()
                w_minus[i] -= 1e-5
                w_minus = np.maximum(w_minus, 1e-8)
                w_minus /= w_minus.sum()
                grad[i] = (neg_dr(w_plus) - neg_dr(w_minus)) / (2e-5)
            w -= lr * grad
            w = np.maximum(w, 0.01)
            w /= w.sum()
        return w

    # ------------------------------------------------------------------
    def get_correlation_cluster(
        self,
        returns: pd.DataFrame,
        n_clusters: int = 5,
    ) -> Dict[str, List[str]]:
        """
        Cluster assets by correlation distance using hierarchical clustering.
        Falls back to simple PCA-based grouping if scipy not available.
        """
        r = returns.dropna()
        tickers = list(r.columns)
        n = len(tickers)
        n_clusters = min(n_clusters, n)

        corr = r.corr().values
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        dist = np.sqrt(np.maximum(2 * (1 - corr), 0.0))

        if _SCIPY_CLUSTER:
            try:
                dist_sq = squareform(dist, checks=False)
                Z = linkage(dist_sq, method="ward")
                labels = fcluster(Z, n_clusters, criterion="maxclust")
                clusters: Dict[str, List[str]] = {}
                for i, t in enumerate(tickers):
                    key = f"cluster_{labels[i]}"
                    clusters.setdefault(key, []).append(t)
                return clusters
            except Exception as exc:
                log.debug("Hierarchical clustering failed: %s", exc)

        # Fallback: PCA-based (SVD)
        try:
            _, _, Vt = np.linalg.svd(corr - np.eye(n))
            pca2 = Vt[:2].T  # first 2 PCs
            # K-means on 2D
            rng = np.random.default_rng(seed=0)
            centroids = pca2[rng.choice(n, n_clusters, replace=False)]
            for _ in range(50):
                dists = np.linalg.norm(pca2[:, None] - centroids[None, :], axis=2)
                labels_km = np.argmin(dists, axis=1)
                new_c = np.array(
                    [
                        pca2[labels_km == k].mean(axis=0)
                        if (labels_km == k).any()
                        else centroids[k]
                        for k in range(n_clusters)
                    ]
                )
                if np.allclose(new_c, centroids, atol=1e-6):
                    break
                centroids = new_c
            clusters = {}
            for i, t in enumerate(tickers):
                key = f"cluster_{labels_km[i]+1}"
                clusters.setdefault(key, []).append(t)
            return clusters
        except Exception:
            return {"cluster_1": tickers}


# ---------------------------------------------------------------------------
# CrossAssetCorrelationTracker
# ---------------------------------------------------------------------------

_CROSS_ASSET_UNIVERSE = {
    "SPY": "US Equities",
    "TLT": "Long Bonds",
    "GLD": "Gold",
    "USO": "Oil",
    "UUP": "USD",
    "^VIX": "Volatility",
    "BTC-USD": "Crypto",
    "EEM": "EM Equities",
}


class CrossAssetCorrelationTracker:
    """
    Monitor correlations and regime across major asset classes.
    Detects risk-off signals from cross-asset price moves.
    """

    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self._returns_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def _fetch_cross_asset_returns(self) -> pd.DataFrame:
        """Fetch log returns for all cross-asset tickers."""
        if self._returns_cache is not None:
            return self._returns_cache
        tickers = list(_CROSS_ASSET_UNIVERSE.keys())
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=self.lookback_days + 30)).strftime("%Y-%m-%d")
        try:
            prices = _fetch_prices(tickers, start, end)
            returns = _prices_to_log_returns(prices)
            self._returns_cache = returns
            return returns
        except Exception as exc:
            log.warning("Cross-asset fetch failed: %s", exc)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    def compute_cross_asset_matrix(
        self, lookback_days: Optional[int] = None
    ) -> pd.DataFrame:
        """Current Pearson correlation matrix across asset classes."""
        r = self._fetch_cross_asset_returns()
        if r.empty:
            return pd.DataFrame()
        lb = lookback_days or self.lookback_days
        r = r.tail(lb)
        return r.corr(method="pearson").round(3)

    # ------------------------------------------------------------------
    def detect_risk_off_signal(self) -> Tuple[bool, float]:
        """
        Risk-off signal: equities down + bonds up + gold up + VIX spike.
        Returns (signal_flag, composite_score 0-1).
        """
        r = self._fetch_cross_asset_returns()
        if r.empty:
            return False, 0.0

        # Recent 5-day returns
        n = 5
        recent = r.tail(n)
        score = 0.0
        checks = 0

        # SPY down
        if "SPY" in recent.columns:
            spy_ret = float(recent["SPY"].sum())
            score += max(-spy_ret * 5, 0.0)  # -1% returns ~0.05 score each
            checks += 1

        # TLT up (flight to safety)
        if "TLT" in recent.columns:
            tlt_ret = float(recent["TLT"].sum())
            score += max(tlt_ret * 5, 0.0)
            checks += 1

        # GLD up
        if "GLD" in recent.columns:
            gld_ret = float(recent["GLD"].sum())
            score += max(gld_ret * 5, 0.0)
            checks += 1

        # VIX up (using VIX price level logic — it's inverted as return)
        if "^VIX" in recent.columns:
            vix_ret = float(recent["^VIX"].sum())
            score += max(vix_ret * 3, 0.0)
            checks += 1

        # Normalise
        composite = min(score / max(checks, 1), 1.0)
        signal = composite > 0.15  # threshold for risk-off call

        return signal, round(composite, 3)

    # ------------------------------------------------------------------
    def get_stock_bond_correlation(self, window: int = 60) -> float:
        """Rolling 60-day stock-bond correlation (SPY vs TLT)."""
        r = self._fetch_cross_asset_returns()
        if r.empty or "SPY" not in r.columns or "TLT" not in r.columns:
            return float("nan")
        sub = r.tail(window)
        return round(float(sub["SPY"].corr(sub["TLT"])), 4)

    # ------------------------------------------------------------------
    def detect_correlation_regime(self) -> str:
        """
        Classify cross-asset correlation regime.

        Returns: "RISK_ON" | "RISK_OFF" | "TRANSITIONAL"
        """
        risk_off, score = self.detect_risk_off_signal()
        sb_corr = self.get_stock_bond_correlation()

        if risk_off and (math.isnan(sb_corr) or sb_corr < 0):
            return "RISK_OFF"
        elif not risk_off and not math.isnan(sb_corr) and sb_corr > 0.2:
            # Positive stock-bond corr = inflationary / unusual
            return "TRANSITIONAL"
        elif not risk_off and score < 0.05:
            return "RISK_ON"
        return "TRANSITIONAL"

    # ------------------------------------------------------------------
    def get_regime_summary(self) -> dict:
        """Full cross-asset regime summary dict."""
        matrix = self.compute_cross_asset_matrix()
        risk_off_flag, risk_off_score = self.detect_risk_off_signal()
        sb_corr = self.get_stock_bond_correlation()
        regime = self.detect_correlation_regime()

        return {
            "regime": regime,
            "risk_off_signal": risk_off_flag,
            "risk_off_score": risk_off_score,
            "stock_bond_correlation_60d": sb_corr,
            "correlation_matrix": matrix.to_dict() if not matrix.empty else {},
        }


# ---------------------------------------------------------------------------
# PairsTradingMonitor
# ---------------------------------------------------------------------------


def _numpy_adf_test(series: np.ndarray, maxlag: int = 5) -> Tuple[float, float]:
    """
    Simplified ADF test via OLS (numpy fallback when statsmodels absent).
    Tests H0: unit root. Returns (adf_stat, p_value_approx).
    p-value approximated from critical value table.
    """
    y = np.asarray(series, float)
    dy = np.diff(y)
    n = len(dy)
    if n < maxlag + 5:
        return 0.0, 1.0

    # Build regression: dy_t = rho * y_{t-1} + sum(beta_i * dy_{t-i}) + e
    k = min(maxlag, n // 4)
    y_lag = y[k : n]  # y_{t-1}
    dy_dep = dy[k:]    # dy_t

    # Build matrix with lagged differences
    X_cols = [y_lag]
    for lag in range(1, k + 1):
        X_cols.append(dy[k - lag : n - lag])
    X = np.column_stack(X_cols)
    ones = np.ones((len(dy_dep), 1))
    X = np.hstack([ones, X])

    try:
        beta, res, rank, sv = np.linalg.lstsq(X, dy_dep, rcond=None)
        fitted = X @ beta
        resid = dy_dep - fitted
        s2 = float(np.sum(resid**2)) / max(len(dy_dep) - X.shape[1], 1)
        # Standard error of rho coefficient
        XtX_inv = np.linalg.pinv(X.T @ X)
        se_rho = math.sqrt(max(s2 * XtX_inv[1, 1], 1e-12))
        adf_stat = (beta[1]) / se_rho
    except Exception:
        return 0.0, 1.0

    # Approximate p-value using asymptotic critical values (no constant case)
    # MacKinnon (1994) approximate critical values: -3.43 (1%), -2.86 (5%), -2.57 (10%)
    if adf_stat < -3.43:
        p_approx = 0.01
    elif adf_stat < -2.86:
        p_approx = 0.05
    elif adf_stat < -2.57:
        p_approx = 0.10
    else:
        p_approx = 0.50
    return adf_stat, p_approx


def _compute_half_life(spread: np.ndarray) -> float:
    """Estimate mean-reversion half-life via AR(1) regression on spread."""
    y = spread[1:]
    x = spread[:-1]
    if len(x) < 5:
        return float("inf")
    try:
        beta = float(np.cov(x, y)[0, 1] / max(np.var(x), 1e-12))
        hl = -math.log(2) / math.log(max(beta, 1e-8))
        return max(hl, 0.5)
    except Exception:
        return float("inf")


class PairsTradingMonitor:
    """
    Test for cointegration in pairs and generate spread signals.
    """

    # ------------------------------------------------------------------
    def test_cointegration(
        self,
        s1: pd.Series,
        s2: pd.Series,
    ) -> CointegrationResult:
        """
        Engle-Granger cointegration test.
        Uses statsmodels if available; falls back to numpy ADF.
        """
        common = s1.index.intersection(s2.index)
        if len(common) < 60:
            return CointegrationResult(
                ticker1=str(s1.name),
                ticker2=str(s2.name),
                is_cointegrated=False,
                p_value=1.0,
            )

        p1 = s1.loc[common].values
        p2 = s2.loc[common].values

        # OLS for hedge ratio
        X = np.column_stack([np.ones(len(p2)), p2])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, p1, rcond=None)
            hedge_ratio = float(beta[1])
        except Exception:
            hedge_ratio = 1.0

        spread = p1 - hedge_ratio * p2

        # ADF on spread
        if _STATSMODELS:
            try:
                coint_t, p_val, _ = coint(p1, p2)
                adf_stat, _ = adfuller(spread, maxlag=5, regression="c")[:2]
                method = "engle-granger-statsmodels"
            except Exception:
                adf_stat, p_val = _numpy_adf_test(spread)
                method = "numpy-adf"
        else:
            adf_stat, p_val = _numpy_adf_test(spread)
            method = "numpy-adf"

        is_coint = p_val < 0.05
        hl = _compute_half_life(spread)

        return CointegrationResult(
            ticker1=str(s1.name),
            ticker2=str(s2.name),
            is_cointegrated=is_coint,
            p_value=round(float(p_val), 4),
            adf_stat=round(adf_stat, 4),
            hedge_ratio=round(hedge_ratio, 4),
            half_life_days=round(hl, 1) if not math.isinf(hl) else 999.0,
            method=method,
        )

    # ------------------------------------------------------------------
    def find_cointegrated_pairs(
        self,
        universe: List[str],
        p_threshold: float = 0.05,
        lookback_days: int = 252,
    ) -> List[PairResult]:
        """
        Test all pairs in universe for cointegration.
        Returns only pairs with p_value < p_threshold.
        """
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")

        try:
            prices = _fetch_prices(universe, start, end)
        except Exception as exc:
            log.warning("Could not fetch prices for pairs: %s", exc)
            return []

        pairs_results: List[PairResult] = []
        tickers = [t for t in universe if t in prices.columns]
        n = len(tickers)

        for i in range(n):
            for j in range(i + 1, n):
                t1, t2 = tickers[i], tickers[j]
                s1 = prices[t1].dropna()
                s2 = prices[t2].dropna()

                coint_res = self.test_cointegration(s1, s2)

                if coint_res.p_value <= p_threshold:
                    spread = self.compute_spread(s1, s2, coint_res.hedge_ratio)
                    z = self._compute_zscore(spread)
                    signal = self.detect_spread_signal(spread)

                    pairs_results.append(
                        PairResult(
                            ticker1=t1,
                            ticker2=t2,
                            p_value=coint_res.p_value,
                            hedge_ratio=coint_res.hedge_ratio,
                            spread_mean=round(float(spread.mean()), 4),
                            spread_std=round(float(spread.std()), 4),
                            half_life_days=coint_res.half_life_days,
                            current_z_score=round(z, 3),
                            signal=signal,
                        )
                    )

        pairs_results.sort(key=lambda x: x.p_value)
        return pairs_results

    # ------------------------------------------------------------------
    @staticmethod
    def compute_spread(
        s1: pd.Series,
        s2: pd.Series,
        hedge_ratio: Optional[float] = None,
    ) -> pd.Series:
        """Compute hedge-ratio adjusted spread: s1 - hr * s2."""
        common = s1.index.intersection(s2.index)
        p1 = s1.loc[common]
        p2 = s2.loc[common]

        if hedge_ratio is None:
            X = np.column_stack([np.ones(len(p2)), p2.values])
            try:
                beta, _, _, _ = np.linalg.lstsq(X, p1.values, rcond=None)
                hedge_ratio = float(beta[1])
            except Exception:
                hedge_ratio = 1.0

        spread = p1 - hedge_ratio * p2
        spread.name = f"spread_{s1.name}_{s2.name}"
        return spread

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_zscore(spread: pd.Series, window: int = 60) -> float:
        """Current z-score of spread vs rolling mean/std."""
        if len(spread) < window:
            window = max(len(spread) // 2, 5)
        if len(spread) < 5:
            return 0.0
        tail = spread.tail(window)
        mu = float(tail.mean())
        sigma = float(tail.std())
        if sigma < 1e-10:
            return 0.0
        return (float(spread.iloc[-1]) - mu) / sigma

    # ------------------------------------------------------------------
    def detect_spread_signal(
        self,
        spread: pd.Series,
        z_threshold: float = 2.0,
        window: int = 60,
    ) -> str:
        """
        Generate trading signal based on spread z-score.
        LONG_SPREAD / SHORT_SPREAD / NEUTRAL
        """
        z = self._compute_zscore(spread, window)
        if z < -z_threshold:
            return "LONG_SPREAD"   # spread below mean → expect mean reversion up
        elif z > z_threshold:
            return "SHORT_SPREAD"  # spread above mean → expect mean reversion down
        return "NEUTRAL"


# ---------------------------------------------------------------------------
# RegimeAlertSystem
# ---------------------------------------------------------------------------


class RegimeAlertSystem:
    """Generate structured alerts for all correlation regime events."""

    def __init__(self):
        self._detector = CorrelationRegimeDetector()
        self._cross = CrossAssetCorrelationTracker()
        self._pairs = PairsTradingMonitor()
        self._garch = None  # lazy init

    # ------------------------------------------------------------------
    def _get_garch(self):
        if self._garch is None:
            # Import here to avoid circular; GARCHModel is in portfolio_risk_v3 but
            # we replicate the core logic inline to keep modules independent
            from sentinel.spm.portfolio_risk_v3 import GARCHModel

            self._garch = GARCHModel()
        return self._garch

    # ------------------------------------------------------------------
    def check_all_alerts(
        self,
        universe: List[str],
        returns: Optional[pd.DataFrame] = None,
    ) -> List[CorrelationAlert]:
        """
        Run all alert checks and return a sorted list of CorrelationAlert objects.
        """
        alerts: List[CorrelationAlert] = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if returns is None:
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
            try:
                prices = _fetch_prices(universe, start, end)
                returns = _prices_to_log_returns(prices)
            except Exception as exc:
                log.warning("Could not fetch returns for alerts: %s", exc)
                return alerts

        # ------------------------------------------------------------------
        # Alert 1: CORRELATION_SPIKE
        # ------------------------------------------------------------------
        try:
            avg_corr_series = self._detector.compute_average_pairwise_correlation(
                returns, window=60
            )
            if not avg_corr_series.empty:
                current_corr = float(avg_corr_series.iloc[-1])
                pct = self._detector.compute_correlation_percentile(
                    current_corr, avg_corr_series
                )
                if current_corr > 0.70:
                    alerts.append(
                        CorrelationAlert(
                            alert_type="CORRELATION_SPIKE",
                            severity="CRITICAL",
                            message=(
                                f"Avg pairwise correlation {current_corr:.3f} > 0.70 "
                                f"({pct:.0f}th percentile). Diversification failing."
                            ),
                            timestamp=now,
                            affected_assets=universe,
                            metric_value=current_corr,
                            threshold=0.70,
                        )
                    )
                elif current_corr > 0.55:
                    alerts.append(
                        CorrelationAlert(
                            alert_type="CORRELATION_SPIKE",
                            severity="WARN",
                            message=(
                                f"Avg pairwise correlation elevated at {current_corr:.3f} "
                                f"({pct:.0f}th percentile)."
                            ),
                            timestamp=now,
                            affected_assets=universe,
                            metric_value=current_corr,
                            threshold=0.55,
                        )
                    )

                # Alert 2: CORRELATION_COLLAPSE
                if current_corr < 0.20:
                    alerts.append(
                        CorrelationAlert(
                            alert_type="CORRELATION_COLLAPSE",
                            severity="WARN",
                            message=(
                                f"Avg pairwise correlation {current_corr:.3f} < 0.20. "
                                "Unusual market calm; stock-picking environment."
                            ),
                            timestamp=now,
                            affected_assets=universe,
                            metric_value=current_corr,
                            threshold=0.20,
                        )
                    )
        except Exception as exc:
            log.debug("Correlation spike check failed: %s", exc)

        # ------------------------------------------------------------------
        # Alert 3: RISK_OFF_SIGNAL
        # ------------------------------------------------------------------
        try:
            risk_off, score = self._cross.detect_risk_off_signal()
            if risk_off:
                alerts.append(
                    CorrelationAlert(
                        alert_type="RISK_OFF_SIGNAL",
                        severity="CRITICAL" if score > 0.4 else "WARN",
                        message=(
                            f"Cross-asset risk-off pattern detected. "
                            f"Score: {score:.3f} (SPY down, TLT up, GLD up, VIX elevated)."
                        ),
                        timestamp=now,
                        affected_assets=list(_CROSS_ASSET_UNIVERSE.keys()),
                        metric_value=score,
                        threshold=0.15,
                    )
                )
        except Exception as exc:
            log.debug("Risk-off check failed: %s", exc)

        # ------------------------------------------------------------------
        # Alert 4: PAIR_BREAKDOWN
        # ------------------------------------------------------------------
        try:
            tickers = [t for t in universe if t in returns.columns]
            n = len(tickers)
            for i in range(min(n, 8)):
                for j in range(i + 1, min(n, 8)):
                    t1, t2 = tickers[i], tickers[j]
                    r1, r2 = returns[t1], returns[t2]
                    if self._detector.detect_correlation_breakdown(r1, r2):
                        alerts.append(
                            CorrelationAlert(
                                alert_type="PAIR_BREAKDOWN",
                                severity="WARN",
                                message=(
                                    f"Correlation breakdown detected for {t1}/{t2}. "
                                    "Long-run correlation diverging from recent. "
                                    "Potential regime shift or arb opportunity."
                                ),
                                timestamp=now,
                                affected_assets=[t1, t2],
                                metric_value=0.0,
                                threshold=0.40,
                            )
                        )
        except Exception as exc:
            log.debug("Pair breakdown check failed: %s", exc)

        # ------------------------------------------------------------------
        # Alert 5: SECTOR_ROTATION
        # ------------------------------------------------------------------
        try:
            n = len([t for t in universe if t in returns.columns])
            if n >= 4:
                avail = [t for t in universe if t in returns.columns]
                mid = n // 2
                intra_grp1 = returns[avail[:mid]].corr().values
                intra_grp2 = returns[avail[mid:]].corr().values
                cross_corr = returns[avail[:mid]].corrwith(returns[avail[mid]]).mean()
                intra_avg = (
                    np.nanmean([intra_grp1[i, j] for i in range(mid) for j in range(i + 1, mid)])
                    + np.nanmean(
                        [
                            intra_grp2[i, j]
                            for i in range(n - mid)
                            for j in range(i + 1, n - mid)
                        ]
                    )
                ) / 2
                if intra_avg > 0.7 and float(cross_corr) < 0.3:
                    alerts.append(
                        CorrelationAlert(
                            alert_type="SECTOR_ROTATION",
                            severity="INFO",
                            message=(
                                f"Intra-group correlation {intra_avg:.2f} rising while "
                                f"cross-group {float(cross_corr):.2f} falling. "
                                "Sector rotation signal."
                            ),
                            timestamp=now,
                            affected_assets=avail,
                            metric_value=intra_avg,
                            threshold=0.70,
                        )
                    )
        except Exception as exc:
            log.debug("Sector rotation check failed: %s", exc)

        # ------------------------------------------------------------------
        # Alert 6: VOLATILITY_CLUSTERING
        # ------------------------------------------------------------------
        try:
            port_r = returns.mean(axis=1).dropna()
            if len(port_r) >= 100:
                try:
                    garch = self._get_garch()
                    garch.fit(port_r.values)
                    forecasts = garch.forecast_variance(h=5)
                    current_vol = math.sqrt(float(forecasts[0])) * math.sqrt(252)
                    hist_vol = float(port_r.std()) * math.sqrt(252)
                    if current_vol > 1.5 * hist_vol:
                        alerts.append(
                            CorrelationAlert(
                                alert_type="VOLATILITY_CLUSTERING",
                                severity="WARN",
                                message=(
                                    f"GARCH vol forecast {current_vol*100:.1f}% vs "
                                    f"historical {hist_vol*100:.1f}%. "
                                    "Volatility clustering / jump risk elevated."
                                ),
                                timestamp=now,
                                affected_assets=universe,
                                metric_value=current_vol,
                                threshold=hist_vol * 1.5,
                            )
                        )
                except Exception:
                    pass
        except Exception as exc:
            log.debug("Volatility clustering check failed: %s", exc)

        # Sort: CRITICAL first, then WARN, then INFO
        severity_rank = {"CRITICAL": 0, "WARN": 1, "INFO": 2}
        alerts.sort(key=lambda a: severity_rank.get(a.severity, 3))
        return alerts

    # ------------------------------------------------------------------
    def get_alert_dashboard(
        self,
        universe: List[str],
        returns: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Return all alerts as a pandas DataFrame."""
        alerts = self.check_all_alerts(universe, returns)
        if not alerts:
            return pd.DataFrame(
                columns=["timestamp", "severity", "alert_type", "message", "metric_value", "threshold"]
            )
        rows = [
            {
                "timestamp": a.timestamp,
                "severity": a.severity,
                "alert_type": a.alert_type,
                "message": a.message,
                "metric_value": round(a.metric_value, 4),
                "threshold": a.threshold,
                "affected_assets": ", ".join(a.affected_assets[:5]),
            }
            for a in alerts
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def generate_regime_report(
        self,
        universe: List[str],
        returns: Optional[pd.DataFrame] = None,
    ) -> str:
        """Generate a human-readable regime report string."""
        if returns is None:
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
            try:
                prices = _fetch_prices(universe, start, end)
                returns = _prices_to_log_returns(prices)
            except Exception as exc:
                return f"Could not fetch data: {exc}"

        detector = CorrelationRegimeDetector()
        avg_corr = detector.compute_average_pairwise_correlation(returns, window=60)
        current_corr = float(avg_corr.iloc[-1]) if not avg_corr.empty else 0.0
        pct = detector.compute_correlation_percentile(current_corr, avg_corr)

        cross = CrossAssetCorrelationTracker()
        regime = cross.detect_correlation_regime()
        risk_off, score = cross.detect_risk_off_signal()
        sb_corr = cross.get_stock_bond_correlation()

        alerts = self.check_all_alerts(universe, returns)
        crit_alerts = [a for a in alerts if a.severity == "CRITICAL"]
        warn_alerts = [a for a in alerts if a.severity == "WARN"]

        lines = [
            "=" * 60,
            f"SENTINEL Correlation Regime Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            f"",
            f"Universe: {', '.join(universe[:10])}{'...' if len(universe)>10 else ''}",
            f"",
            f"--- Current Regime ---",
            f"  Cross-asset regime:       {regime}",
            f"  Risk-off signal:          {'YES' if risk_off else 'NO'} (score: {score:.3f})",
            f"  Stock-bond corr (60d):    {sb_corr:.3f}",
            f"",
            f"--- Correlation Levels ---",
            f"  Avg pairwise corr (60d):  {current_corr:.3f}",
            f"  Corr percentile (1yr):    {pct:.0f}th",
            f"",
            f"--- Alerts ---",
            f"  CRITICAL: {len(crit_alerts)}",
            f"  WARNING:  {len(warn_alerts)}",
        ]
        for a in alerts[:8]:
            lines.append(f"  [{a.severity}] {a.alert_type}: {a.message[:80]}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CorrelationMonitorEngine (orchestrator)
# ---------------------------------------------------------------------------


class CorrelationMonitorEngine:
    """
    Top-level orchestrator for all correlation monitoring operations.
    """

    def __init__(self, lookback_days: int = 365):
        self.lookback_days = lookback_days
        self._computer = CorrelationComputer()
        self._detector = CorrelationRegimeDetector()
        self._diversification = DiversificationAnalyzer()
        self._cross = CrossAssetCorrelationTracker(lookback_days=lookback_days)
        self._pairs = PairsTradingMonitor()
        self._alerts = RegimeAlertSystem()

    # ------------------------------------------------------------------
    def _fetch_returns(self, universe: List[str]) -> pd.DataFrame:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=self.lookback_days + 30)).strftime("%Y-%m-%d")
        prices = _fetch_prices(universe, start, end)
        return _prices_to_log_returns(prices)

    # ------------------------------------------------------------------
    def get_full_report(self, universe: List[str]) -> CorrelationReport:
        """
        Compute the complete correlation monitoring report for a universe.
        """
        log.info("Building full correlation report for %d assets", len(universe))
        returns = self._fetch_returns(universe)

        if returns.empty:
            log.error("No return data; cannot build report")
            return CorrelationReport(universe=universe, as_of_date=datetime.now().strftime("%Y-%m-%d"))

        avail = [t for t in universe if t in returns.columns]
        returns = returns[avail]

        # Weights (equal weight for reporting)
        n = len(avail)
        weights = np.ones(n) / n

        # Correlation metrics
        avg_corr_series = self._detector.compute_average_pairwise_correlation(returns, window=60)
        current_corr = float(avg_corr_series.iloc[-1]) if not avg_corr_series.empty else 0.0
        corr_pct = self._detector.compute_correlation_percentile(current_corr, avg_corr_series)

        # Regime
        cross_regime = self._cross.detect_correlation_regime()
        risk_off, _ = self._cross.detect_risk_off_signal()
        sb_corr = self._cross.get_stock_bond_correlation()

        # Diversification
        dr = self._diversification.compute_diversification_ratio(returns, weights)
        corr_matrix = self._computer.compute_pearson(returns)
        eff_n = self._diversification.compute_effective_n(corr_matrix, weights)
        hhi = self._diversification.compute_hhi(weights)

        # Local regime from avg pairwise corr
        if current_corr > 0.70:
            local_regime = "HIGH_CORR"
        elif current_corr < 0.30:
            local_regime = "LOW_CORR"
        else:
            local_regime = "NORMAL"

        # Clusters
        clusters = self._diversification.get_correlation_cluster(returns, n_clusters=min(5, n // 2 + 1))

        # Alerts
        alerts = self._alerts.check_all_alerts(avail, returns)

        # Cointegrated pairs (test top pairs)
        coint_pairs: List[PairResult] = []
        try:
            coint_pairs = self._pairs.find_cointegrated_pairs(avail[:12], p_threshold=0.10)
        except Exception as exc:
            log.warning("Pairs test failed: %s", exc)

        return CorrelationReport(
            as_of_date=datetime.now().strftime("%Y-%m-%d"),
            universe=avail,
            avg_pairwise_corr=round(current_corr, 4),
            corr_percentile=corr_pct,
            regime=local_regime,
            cross_asset_regime=cross_regime,
            risk_off_signal=risk_off,
            stock_bond_corr=sb_corr,
            diversification_ratio=round(dr, 4),
            effective_n=eff_n,
            hhi=round(hhi, 4),
            alerts=alerts,
            cointegrated_pairs=coint_pairs,
            correlation_matrix=corr_matrix,
            clusters=clusters,
        )

    # ------------------------------------------------------------------
    def monitor_portfolio(
        self, holdings: Dict[str, float]
    ) -> PortfolioCorrelationSummary:
        """
        Compute correlation health metrics for a specific portfolio.
        """
        tickers = list(holdings.keys())
        weights = np.array(list(holdings.values()), dtype=float)
        weights /= weights.sum()

        returns = self._fetch_returns(tickers)
        avail = [t for t in tickers if t in returns.columns]
        avail_w = np.array([holdings[t] for t in avail], dtype=float)
        avail_w /= avail_w.sum()
        returns = returns[avail]

        # Internal correlation metrics
        avg_corr_series = self._detector.compute_average_pairwise_correlation(
            returns, window=60
        )
        current_corr = float(avg_corr_series.iloc[-1]) if not avg_corr_series.empty else 0.0
        dr = self._diversification.compute_diversification_ratio(returns, avail_w)
        corr_matrix = self._computer.compute_pearson(returns)
        eff_n = self._diversification.compute_effective_n(corr_matrix, avail_w)
        hhi = self._diversification.compute_hhi(avail_w)

        if current_corr > 0.70:
            regime = "HIGH_CORR"
        elif current_corr < 0.30:
            regime = "LOW_CORR"
        else:
            regime = "NORMAL"

        alerts = self._alerts.check_all_alerts(avail, returns)

        # Component correlations (each asset vs portfolio)
        port_r = pd.Series(returns.values @ avail_w, index=returns.index)
        comp_corrs = {
            t: round(float(returns[t].corr(port_r)), 4)
            for t in avail
        }
        comp_df = pd.DataFrame(
            [{"ticker": t, "corr_to_portfolio": v} for t, v in comp_corrs.items()]
        ).sort_values("corr_to_portfolio", ascending=False)

        return PortfolioCorrelationSummary(
            holdings=dict(zip(avail, avail_w.tolist())),
            internal_avg_corr=round(current_corr, 4),
            diversification_ratio=round(dr, 4),
            effective_n=eff_n,
            hhi=round(hhi, 4),
            regime=regime,
            alerts=alerts,
            component_correlations=comp_df,
        )

    # ------------------------------------------------------------------
    def run_daily_check(self, universe: List[str]) -> List[CorrelationAlert]:
        """Run all alert checks for daily monitoring workflow."""
        try:
            returns = self._fetch_returns(universe)
            return self._alerts.check_all_alerts(universe, returns)
        except Exception as exc:
            log.error("Daily check failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    def export_correlation_matrix(
        self,
        universe: List[str],
        path: str,
        method: str = "pearson",
    ) -> None:
        """
        Export correlation matrix to CSV.

        Parameters
        ----------
        universe : list of tickers
        path : output file path
        method : "pearson" | "spearman" | "ewm"
        """
        returns = self._fetch_returns(universe)
        avail = [t for t in universe if t in returns.columns]
        returns = returns[avail]

        if method == "spearman":
            cm = self._computer.compute_spearman(returns)
        elif method == "ewm":
            cm = self._computer.compute_ewm_correlation(returns)
        else:
            cm = self._computer.compute_pearson(returns)

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        cm.to_csv(path)
        log.info("Correlation matrix exported to %s", path)
        print(f"Correlation matrix ({method}) exported to: {path}")


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    print("=" * 70)
    print("SENTINEL Correlation Monitor — demo (dim_080)")
    print("=" * 70)

    UNIVERSE = ["SPY", "TLT", "GLD", "USO", "QQQ", "XLF", "XLE", "IWM"]

    engine = CorrelationMonitorEngine(lookback_days=365)

    print("\n[1] Fetching returns...")
    try:
        returns = engine._fetch_returns(UNIVERSE)
        avail = [t for t in UNIVERSE if t in returns.columns]
        print(f"  Available: {avail}")
        print(f"  Date range: {returns.index[0].date()} → {returns.index[-1].date()}")
        print(f"  Observations: {len(returns)}")
    except Exception as exc:
        print(f"  Error: {exc}")
        avail = UNIVERSE[:4]
        returns = pd.DataFrame()

    print("\n[2] Correlation matrix (Pearson, 60-day)...")
    try:
        cc = CorrelationComputer()
        pearson = cc.compute_pearson(returns.tail(60))
        print(pearson.round(3).to_string())
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\n[3] EWMA correlation matrix...")
    try:
        ewm_corr = cc.compute_ewm_correlation(returns, span=60)
        print(ewm_corr.round(3).to_string())
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\n[4] Average pairwise correlation (rolling 60d)...")
    try:
        detector = CorrelationRegimeDetector()
        avg_corr = detector.compute_average_pairwise_correlation(returns, window=60)
        if not avg_corr.empty:
            current = float(avg_corr.iloc[-1])
            pct = detector.compute_correlation_percentile(current, avg_corr)
            print(f"  Current avg pairwise corr: {current:.4f}")
            print(f"  Percentile (1yr):          {pct:.0f}th")
            events = detector.detect_correlation_spike(avg_corr, threshold=0.7)
            print(f"  Detected {len(events)} regime events")
            for e in events[-3:]:
                print(f"    {e.date} {e.event_type}: corr={e.avg_correlation:.3f}")
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\n[5] Cross-asset regime detection...")
    try:
        cross = CrossAssetCorrelationTracker()
        regime = cross.detect_correlation_regime()
        risk_off, score = cross.detect_risk_off_signal()
        sb_corr = cross.get_stock_bond_correlation()
        print(f"  Regime:                 {regime}")
        print(f"  Risk-off signal:        {'YES' if risk_off else 'NO'} (score {score:.3f})")
        print(f"  Stock-bond corr (60d):  {sb_corr:.4f}")
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\n[6] Diversification analysis...")
    try:
        avail_r = returns[[t for t in avail if t in returns.columns]]
        n = avail_r.shape[1]
        w = np.ones(n) / n
        da = DiversificationAnalyzer()
        dr = da.compute_diversification_ratio(avail_r, w)
        cm = cc.compute_pearson(avail_r)
        eff_n = da.compute_effective_n(cm, w)
        hhi = da.compute_hhi(w)
        print(f"  Diversification ratio:    {dr:.4f}")
        print(f"  Effective N:              {eff_n:.2f}")
        print(f"  HHI:                      {hhi:.4f}")
        clusters = da.get_correlation_cluster(avail_r, n_clusters=3)
        print(f"  Clusters:")
        for name, members in clusters.items():
            print(f"    {name}: {members}")
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\n[7] Pairs cointegration test (AAPL vs MSFT)...")
    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        prices_pairs = _fetch_prices(["AAPL", "MSFT"], start, end)
        if "AAPL" in prices_pairs.columns and "MSFT" in prices_pairs.columns:
            pt = PairsTradingMonitor()
            result = pt.test_cointegration(prices_pairs["AAPL"], prices_pairs["MSFT"])
            print(f"  Pair: AAPL / MSFT")
            print(f"  Cointegrated: {result.is_cointegrated}")
            print(f"  p-value:      {result.p_value:.4f}")
            print(f"  ADF stat:     {result.adf_stat:.4f}")
            print(f"  Hedge ratio:  {result.hedge_ratio:.4f}")
            print(f"  Half-life:    {result.half_life_days:.1f} days")
            spread = pt.compute_spread(prices_pairs["AAPL"], prices_pairs["MSFT"], result.hedge_ratio)
            signal = pt.detect_spread_signal(spread)
            print(f"  Spread signal: {signal}")
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\n[8] All alerts...")
    try:
        alert_sys = RegimeAlertSystem()
        alerts = alert_sys.check_all_alerts(avail, returns if not returns.empty else None)
        if alerts:
            for a in alerts:
                print(f"  [{a.severity}] {a.alert_type}: {a.message[:80]}")
        else:
            print("  No alerts fired.")
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\n[9] Alert dashboard (DataFrame)...")
    try:
        alert_sys = RegimeAlertSystem()
        dash = alert_sys.get_alert_dashboard(avail, returns if not returns.empty else None)
        if not dash.empty:
            print(dash[["severity", "alert_type", "metric_value"]].to_string(index=False))
        else:
            print("  No alerts.")
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\n[10] Regime report...")
    try:
        report_str = alert_sys.generate_regime_report(avail, returns if not returns.empty else None)
        print(report_str)
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\nDone.")
