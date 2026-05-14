"""
Real-time correlation monitoring and regime detection for portfolio risk — dim_075.

Implements multi-factor regime classification and Ledoit-Wolf shrinkage
correlation matrices for robust portfolio risk assessment.

Uses:
  yfinance  — price history for all tickers and benchmarks
  scipy     — correlation significance tests
  sklearn   — LedoitWolf shrinkage covariance estimation

Implements:
  CorrelationMonitor.get_correlation_matrix()
  CorrelationMonitor.monitor_portfolio_correlations()
  CorrelationMonitor.detect_regime()
  CorrelationMonitor.get_rolling_correlations()
  CorrelationMonitor.find_low_correlation_assets()
  CorrelationMonitor.stress_test_correlation()
  CorrelationMonitor.classify_regime_from_signals()

Module-level helpers:
  correlation_matrix(), detect_regime(), portfolio_diversification()
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from itertools import combinations
from typing import Optional, Literal

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from sklearn.covariance import LedoitWolf
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard benchmark universe for correlation and regime analysis
BENCHMARK_UNIVERSE: dict[str, str] = {
    "SPY":      "US Large Cap",
    "QQQ":      "US Tech",
    "IWM":      "US Small Cap",
    "TLT":      "Long Treasuries",
    "GLD":      "Gold",
    "DX-Y.NYB": "USD Index",
    "^VIX":     "Volatility",
    "HYG":      "High Yield",
    "EEM":      "Emerging Markets",
    "VNQ":      "Real Estate",
    "LQD":      "Investment Grade Credit",
    "SHY":      "Short Treasuries",
    "USO":      "Oil",
    "XLF":      "Financials",
}

# Default candidate pool for diversification search
DEFAULT_CANDIDATE_POOL: list[str] = [
    "GLD", "TLT", "SHY", "BIL", "PDBC",   # safe havens / commodities
    "EEM", "VEA", "EWJ",                    # international
    "VNQ", "XLRE",                           # real estate
    "IEF", "TIP",                            # bonds
    "USO", "GDX", "SLV",                    # commodities
    "XLU", "XLP", "XLV",                    # defensive sectors
    "BTC-USD", "ETH-USD",                    # crypto (if desired)
]

# Historical stress period date ranges (inclusive)
STRESS_SCENARIOS: dict[str, tuple[str, str]] = {
    "2008_crisis":   ("2008-09-01", "2009-03-31"),
    "covid_crash":   ("2020-02-20", "2020-04-30"),
    "dot_com":       ("2000-03-10", "2002-10-09"),
    "2022_bear":     ("2022-01-01", "2022-12-31"),
    "taper_tantrum": ("2013-05-01", "2013-09-30"),
}

# Regime classification thresholds
_VIX_CRISIS      = 30.0
_VIX_RISK_OFF    = 20.0
_VIX_RISK_ON     = 15.0
_CORR_RISK_OFF   = 0.65   # avg pairwise corr above this = risk_off
_CORR_CRISIS     = 0.80   # avg pairwise corr above this = crisis
_ALERT_THRESHOLD = 0.20   # correlation change magnitude to trigger alert
_DIV_LOSS_CORR   = 0.70   # portfolio avg corr above this = diversification loss alert


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CorrelationMatrix(BaseModel):
    tickers: list[str]
    as_of: date
    window_days: int
    matrix: list[list[float]]          # N x N correlation matrix (row = ticker)
    regime: str                        # "risk_on", "risk_off", "transition", "crisis"
    avg_pairwise_corr: float           # mean of off-diagonal upper triangle
    corr_dispersion: float             # std dev of off-diagonal correlations
    n_observations: int                # rows used to compute matrix


class CorrelationAlert(BaseModel):
    ticker_a: str
    ticker_b: str
    current_corr: float
    prior_corr: float
    change: float
    alert_type: str    # "correlation_spike", "correlation_collapse", "diversification_loss"
    severity: str      # "high", "medium", "low"
    description: str


class PortfolioRegime(BaseModel):
    as_of: date
    regime: Literal["risk_on", "risk_off", "transition", "crisis"]
    regime_confidence: float           # 0-1; fraction of signals in agreement
    # Key signals
    vix_level: Optional[float] = None
    yield_curve_slope: Optional[float] = None   # 10Y-2Y (IEF proxy)
    credit_spread: Optional[float] = None       # HYG vs LQD spread proxy
    equity_momentum: Optional[float] = None     # SPY 50d/200d ratio
    avg_correlation: Optional[float] = None     # average pairwise corr in benchmark universe
    # Signal votes
    signal_votes: dict[str, str] = Field(default_factory=dict)   # signal_name → regime
    # Duration
    regime_start: Optional[date] = None
    days_in_regime: Optional[int] = None


class RollingCorrelationRow(BaseModel):
    as_of: date
    corr_30d: Optional[float]
    corr_63d: Optional[float]
    corr_252d: Optional[float]


class DiversificationCandidate(BaseModel):
    ticker: str
    asset_class: str
    avg_corr_to_portfolio: float
    min_corr_to_portfolio: float   # lowest correlation to any single portfolio ticker
    max_corr_to_portfolio: float


# ---------------------------------------------------------------------------
# CorrelationMonitor
# ---------------------------------------------------------------------------

class CorrelationMonitor:
    """
    Portfolio correlation monitor with Ledoit-Wolf shrinkage and
    multi-factor regime classification.

    All public methods are async and safe to run concurrently.
    Price fetching is delegated to _fetch_returns() which uses
    yfinance batch download in a thread executor.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def get_correlation_matrix(
        self,
        tickers: list[str],
        window_days: int = 63,
        end_date: Optional[date] = None,
    ) -> CorrelationMatrix:
        """
        Compute a Ledoit-Wolf shrinkage correlation matrix for the given tickers.

        Args:
            tickers: List of ticker symbols to correlate.
            window_days: Rolling window (trading days). Default 63 = ~1 quarter.
            end_date: End date for price history. Defaults to today.

        Returns:
            CorrelationMatrix with shrinkage-corrected matrix and regime label.
        """
        end = end_date or date.today()

        # Fetch returns: need window_days + buffer for warmup
        fetch_days = window_days + 30
        returns = await self._fetch_returns(tickers, fetch_days, end_date=end)

        if returns.empty:
            raise ValueError(f"No return data for tickers: {tickers}")

        # Use last window_days rows
        returns_window = returns.iloc[-window_days:] if len(returns) >= window_days else returns
        actual_tickers = list(returns_window.columns)

        corr_matrix, avg_corr, dispersion = self._ledoit_wolf_corr(returns_window)

        # Classify regime from VIX (proxy from avg correlation)
        regime = self._classify_regime_from_corr(avg_corr)

        matrix_list = corr_matrix.tolist()

        logger.info(
            "get_correlation_matrix.complete",
            tickers=actual_tickers,
            window_days=window_days,
            n_obs=len(returns_window),
            avg_corr=round(avg_corr, 4),
            regime=regime,
        )

        return CorrelationMatrix(
            tickers=actual_tickers,
            as_of=end,
            window_days=window_days,
            matrix=matrix_list,
            regime=regime,
            avg_pairwise_corr=round(avg_corr, 4),
            corr_dispersion=round(dispersion, 4),
            n_observations=len(returns_window),
        )

    async def monitor_portfolio_correlations(
        self,
        portfolio_tickers: list[str],
        benchmark_tickers: Optional[list[str]] = None,
    ) -> tuple[CorrelationMatrix, list[CorrelationAlert]]:
        """
        Monitor portfolio correlations and detect significant changes.

        Compares current 63-day correlation matrix against the 63-day matrix
        from 90 trading days ago. Flags:
          - Correlation spikes (increase > 0.20)
          - Correlation collapses (decrease > 0.20)
          - Diversification loss (avg portfolio pair corr > 0.70)

        Args:
            portfolio_tickers: Tickers in the portfolio.
            benchmark_tickers: Optional additional tickers for benchmark context.

        Returns:
            Tuple of (current CorrelationMatrix, list[CorrelationAlert]).
        """
        all_tickers = list(dict.fromkeys(
            portfolio_tickers + (benchmark_tickers or [])
        ))

        today = date.today()
        prior_end = today - timedelta(days=90)

        # Fetch sufficient history for both windows
        returns = await self._fetch_returns(all_tickers, 365, end_date=today)

        if returns.empty:
            raise ValueError("No return data available for portfolio monitoring")

        # Restrict to columns actually available
        available_tickers = list(returns.columns)
        portfolio_cols = [t for t in portfolio_tickers if t in available_tickers]

        if not portfolio_cols:
            raise ValueError(f"None of portfolio_tickers found: {portfolio_tickers}")

        # Current window: last 63 trading days
        current_returns = returns.iloc[-63:] if len(returns) >= 63 else returns
        # Prior window: 63 days ending at prior_end
        prior_mask = returns.index <= pd.Timestamp(prior_end)
        prior_slice_full = returns[prior_mask]
        prior_returns = (
            prior_slice_full.iloc[-63:]
            if len(prior_slice_full) >= 63
            else prior_slice_full
        )

        # Compute current matrix for portfolio tickers
        curr_data = current_returns[portfolio_cols].dropna(how="all")
        curr_corr, curr_avg, curr_disp = self._ledoit_wolf_corr(curr_data)
        regime = self._classify_regime_from_corr(curr_avg)

        current_matrix = CorrelationMatrix(
            tickers=portfolio_cols,
            as_of=today,
            window_days=63,
            matrix=curr_corr.tolist(),
            regime=regime,
            avg_pairwise_corr=round(curr_avg, 4),
            corr_dispersion=round(curr_disp, 4),
            n_observations=len(curr_data),
        )

        alerts: list[CorrelationAlert] = []

        # Prior matrix
        if len(prior_returns) >= 20:
            prior_data = prior_returns[portfolio_cols].dropna(how="all")
            prior_corr, prior_avg, _ = self._ledoit_wolf_corr(prior_data)

            # Detect pair-level changes
            for i, t1 in enumerate(portfolio_cols):
                for j, t2 in enumerate(portfolio_cols):
                    if j <= i:
                        continue
                    current_val = float(curr_corr[i, j])
                    prior_val = float(prior_corr[i, j])
                    change = current_val - prior_val

                    if abs(change) < _ALERT_THRESHOLD:
                        continue

                    if change > 0:
                        alert_type = "correlation_spike"
                        severity = "high" if change > 0.4 else "medium" if change > 0.3 else "low"
                        description = (
                            f"{t1}/{t2} correlation increased by {change:.2f} "
                            f"({prior_val:.2f} → {current_val:.2f}): "
                            "diversification benefit reduced"
                        )
                    else:
                        alert_type = "correlation_collapse"
                        severity = "high" if abs(change) > 0.4 else "medium" if abs(change) > 0.3 else "low"
                        description = (
                            f"{t1}/{t2} correlation decreased by {change:.2f} "
                            f"({prior_val:.2f} → {current_val:.2f}): "
                            "regime or factor break detected"
                        )

                    alerts.append(CorrelationAlert(
                        ticker_a=t1,
                        ticker_b=t2,
                        current_corr=round(current_val, 4),
                        prior_corr=round(prior_val, 4),
                        change=round(change, 4),
                        alert_type=alert_type,
                        severity=severity,
                        description=description,
                    ))

        # Diversification loss alert
        if curr_avg > _DIV_LOSS_CORR:
            alerts.append(CorrelationAlert(
                ticker_a="PORTFOLIO",
                ticker_b="AVG",
                current_corr=round(curr_avg, 4),
                prior_corr=round(curr_avg, 4),
                change=0.0,
                alert_type="diversification_loss",
                severity="high",
                description=(
                    f"Average portfolio correlation {curr_avg:.2f} exceeds {_DIV_LOSS_CORR}: "
                    "portfolio is highly correlated — diversification benefit is minimal"
                ),
            ))

        # Sort: high severity first, then by absolute change
        alerts.sort(
            key=lambda a: (0 if a.severity == "high" else 1 if a.severity == "medium" else 2,
                           -abs(a.change))
        )

        logger.info(
            "monitor_portfolio_correlations.complete",
            portfolio=portfolio_cols,
            n_alerts=len(alerts),
            avg_corr=round(curr_avg, 4),
            regime=regime,
        )

        return current_matrix, alerts

    async def detect_regime(self) -> PortfolioRegime:
        """
        Multi-factor macro regime classifier using publicly available ETF proxies.

        Signals:
          1. VIX level  (^VIX from yfinance)
          2. Yield curve slope: TLT/SHY ratio as 10Y/2Y proxy
          3. Credit spread: HYG vs LQD return spread (HY - IG)
          4. Equity momentum: SPY price relative to its 50d and 200d MA
          5. Average pairwise correlation in benchmark universe

        Voting:
          Each signal casts a vote: "risk_on", "risk_off", "crisis", or "transition".
          Final regime = mode of votes; confidence = vote share.

        Returns:
            PortfolioRegime with regime, confidence, and per-signal breakdown.
        """
        today = date.today()

        # Fetch price data for signal tickers
        signal_tickers = ["^VIX", "SPY", "TLT", "SHY", "HYG", "LQD"]
        benchmark_sample = ["SPY", "QQQ", "TLT", "GLD", "HYG", "EEM", "IWM"]

        try:
            returns = await self._fetch_returns(signal_tickers + benchmark_sample, 252)
        except Exception as exc:
            logger.warning("detect_regime: fetch_returns failed: %s", exc)
            returns = pd.DataFrame()

        signal_votes: dict[str, str] = {}

        # ---- Signal 1: VIX level ----
        vix_level: Optional[float] = None
        if "^VIX" in returns.columns:
            vix_prices = await self._fetch_prices(["^VIX"], 5)
            if not vix_prices.empty and "^VIX" in vix_prices.columns:
                vix_level = float(vix_prices["^VIX"].dropna().iloc[-1])
                signal_votes["vix"] = self._vix_to_regime(vix_level)
        else:
            signal_votes["vix"] = "transition"

        # ---- Signal 2: Yield curve slope (TLT/SHY price ratio as proxy) ----
        yield_curve_slope: Optional[float] = None
        if "TLT" in returns.columns and "SHY" in returns.columns:
            prices = await self._fetch_prices(["TLT", "SHY"], 252)
            if not prices.empty and {"TLT", "SHY"}.issubset(prices.columns):
                tlt = prices["TLT"].dropna()
                shy = prices["SHY"].dropna()
                if len(tlt) > 0 and len(shy) > 0:
                    # Ratio of long vs short: declining ratio = flattening / inversion
                    ratio = float(tlt.iloc[-1]) / float(shy.iloc[-1])
                    ratio_6m_ago = (
                        float(tlt.iloc[-126]) / float(shy.iloc[-126])
                        if len(tlt) >= 126 else ratio
                    )
                    yield_curve_slope = round(ratio - ratio_6m_ago, 4)
                    # Negative slope momentum = yield curve flattening = risk_off signal
                    signal_votes["yield_curve"] = (
                        "risk_off" if yield_curve_slope < -0.05
                        else "risk_on" if yield_curve_slope > 0.05
                        else "transition"
                    )
        if "yield_curve" not in signal_votes:
            signal_votes["yield_curve"] = "transition"

        # ---- Signal 3: Credit spread (HYG return spread vs LQD) ----
        credit_spread: Optional[float] = None
        if "HYG" in returns.columns and "LQD" in returns.columns:
            hyg_ret = returns["HYG"].dropna()
            lqd_ret = returns["LQD"].dropna()
            aligned = pd.concat([hyg_ret, lqd_ret], axis=1).dropna()
            if len(aligned) >= 20:
                # Rolling 20-day cumulative return differential
                hyg_20 = float((1 + aligned.iloc[-20:, 0]).prod() - 1)
                lqd_20 = float((1 + aligned.iloc[-20:, 1]).prod() - 1)
                credit_spread = round(hyg_20 - lqd_20, 4)
                # HYG underperforming LQD = credit risk rising = risk_off
                signal_votes["credit_spread"] = (
                    "risk_off" if credit_spread < -0.02
                    else "crisis" if credit_spread < -0.05
                    else "risk_on" if credit_spread > 0.01
                    else "transition"
                )
        if "credit_spread" not in signal_votes:
            signal_votes["credit_spread"] = "transition"

        # ---- Signal 4: Equity momentum (SPY 50d MA / 200d MA) ----
        equity_momentum: Optional[float] = None
        if "SPY" in returns.columns:
            spy_prices = await self._fetch_prices(["SPY"], 252)
            if not spy_prices.empty and "SPY" in spy_prices.columns:
                spy = spy_prices["SPY"].dropna()
                if len(spy) >= 200:
                    ma50 = float(spy.iloc[-50:].mean())
                    ma200 = float(spy.iloc[-200:].mean())
                    equity_momentum = round(ma50 / ma200 - 1, 4)   # positive = above 200d MA
                    signal_votes["equity_momentum"] = (
                        "risk_on" if equity_momentum > 0.02
                        else "risk_off" if equity_momentum < -0.03
                        else "crisis" if equity_momentum < -0.10
                        else "transition"
                    )
        if "equity_momentum" not in signal_votes:
            signal_votes["equity_momentum"] = "transition"

        # ---- Signal 5: Average pairwise correlation ----
        avg_correlation: Optional[float] = None
        bench_cols = [t for t in benchmark_sample if t in returns.columns]
        if len(bench_cols) >= 3:
            bench_returns = returns[bench_cols].dropna(how="all").iloc[-63:]
            if len(bench_returns) >= 20:
                _, avg_corr, _ = self._ledoit_wolf_corr(bench_returns)
                avg_correlation = round(avg_corr, 4)
                signal_votes["avg_correlation"] = (
                    "crisis" if avg_corr > _CORR_CRISIS
                    else "risk_off" if avg_corr > _CORR_RISK_OFF
                    else "risk_on" if avg_corr < 0.30
                    else "transition"
                )
        if "avg_correlation" not in signal_votes:
            signal_votes["avg_correlation"] = "transition"

        # ---- Aggregate votes ----
        final_regime, confidence = self._aggregate_votes(signal_votes)

        logger.info(
            "detect_regime.complete",
            regime=final_regime,
            confidence=round(confidence, 4),
            vix=vix_level,
            signals=signal_votes,
        )

        return PortfolioRegime(
            as_of=today,
            regime=final_regime,  # type: ignore[arg-type]
            regime_confidence=round(confidence, 4),
            vix_level=vix_level,
            yield_curve_slope=yield_curve_slope,
            credit_spread=credit_spread,
            equity_momentum=equity_momentum,
            avg_correlation=avg_correlation,
            signal_votes=signal_votes,
        )

    async def get_rolling_correlations(
        self,
        ticker1: str,
        ticker2: str,
        lookback_days: int = 365,
    ) -> pd.DataFrame:
        """
        Compute rolling 30d, 63d, and 252d correlations between two tickers.

        Args:
            ticker1: First ticker symbol.
            ticker2: Second ticker symbol.
            lookback_days: Total history to fetch.

        Returns:
            DataFrame with columns: date, corr_30d, corr_63d, corr_252d.
            Index is DatetimeIndex.
        """
        returns = await self._fetch_returns(
            [ticker1, ticker2], lookback_days + 60
        )

        if returns.empty or ticker1 not in returns.columns or ticker2 not in returns.columns:
            logger.warning(
                "get_rolling_correlations: missing data for %s or %s", ticker1, ticker2
            )
            return pd.DataFrame(columns=["corr_30d", "corr_63d", "corr_252d"])

        r1 = returns[ticker1]
        r2 = returns[ticker2]

        corr_30 = r1.rolling(window=30, min_periods=15).corr(r2)
        corr_63 = r1.rolling(window=63, min_periods=30).corr(r2)
        corr_252 = r1.rolling(window=252, min_periods=126).corr(r2)

        result = pd.DataFrame({
            "corr_30d": corr_30,
            "corr_63d": corr_63,
            "corr_252d": corr_252,
        })

        # Trim to requested lookback
        cutoff = datetime.today() - timedelta(days=lookback_days)
        result = result[result.index >= pd.Timestamp(cutoff)].copy()

        logger.info(
            "get_rolling_correlations.complete",
            ticker1=ticker1,
            ticker2=ticker2,
            rows=len(result),
        )

        return result

    async def find_low_correlation_assets(
        self,
        portfolio_tickers: list[str],
        candidate_pool: Optional[list[str]] = None,
        max_avg_corr: float = 0.3,
    ) -> list[DiversificationCandidate]:
        """
        Find assets from the candidate pool that have low average correlation
        to the current portfolio — useful for diversification.

        Args:
            portfolio_tickers: Existing portfolio tickers.
            candidate_pool: Assets to evaluate. Defaults to DEFAULT_CANDIDATE_POOL.
            max_avg_corr: Maximum allowed average correlation to portfolio.

        Returns:
            List of DiversificationCandidate sorted by avg_corr_to_portfolio ascending.
        """
        candidates = [
            t for t in (candidate_pool or DEFAULT_CANDIDATE_POOL)
            if t not in portfolio_tickers
        ]

        all_tickers = list(dict.fromkeys(portfolio_tickers + candidates))
        returns = await self._fetch_returns(all_tickers, 252)

        available = list(returns.columns)
        portfolio_cols = [t for t in portfolio_tickers if t in available]
        candidate_cols = [t for t in candidates if t in available]

        if not portfolio_cols or not candidate_cols:
            return []

        # Compute correlation matrix over full 252-day window
        if len(returns) >= 20:
            clean_returns = returns[portfolio_cols + candidate_cols].dropna(how="all")
            corr_df = clean_returns.corr(method="pearson")
        else:
            return []

        # Asset class lookup from benchmark universe and known categories
        asset_class_map = {**{k: v for k, v in BENCHMARK_UNIVERSE.items()}}
        _extra = {
            "PDBC": "Commodities", "BIL": "Cash", "TIP": "TIPS",
            "IEF": "Int Treasuries", "GDX": "Gold Miners", "SLV": "Silver",
            "XLU": "Utilities", "XLP": "Consumer Staples", "XLV": "Healthcare",
            "VEA": "Developed Markets", "EWJ": "Japan", "XLRE": "Real Estate",
            "BTC-USD": "Crypto", "ETH-USD": "Crypto",
        }
        asset_class_map.update(_extra)

        results: list[DiversificationCandidate] = []

        for candidate in candidate_cols:
            if candidate not in corr_df.index:
                continue

            corrs_to_portfolio: list[float] = []
            for pt in portfolio_cols:
                if pt in corr_df.columns:
                    val = corr_df.loc[candidate, pt]
                    if not np.isnan(val):
                        corrs_to_portfolio.append(float(val))

            if not corrs_to_portfolio:
                continue

            avg_corr = float(np.mean(corrs_to_portfolio))
            if avg_corr > max_avg_corr:
                continue

            results.append(DiversificationCandidate(
                ticker=candidate,
                asset_class=asset_class_map.get(candidate, "Unknown"),
                avg_corr_to_portfolio=round(avg_corr, 4),
                min_corr_to_portfolio=round(float(np.min(corrs_to_portfolio)), 4),
                max_corr_to_portfolio=round(float(np.max(corrs_to_portfolio)), 4),
            ))

        results.sort(key=lambda c: c.avg_corr_to_portfolio)

        logger.info(
            "find_low_correlation_assets.complete",
            portfolio=portfolio_cols,
            n_candidates=len(candidate_cols),
            n_passed=len(results),
            max_avg_corr=max_avg_corr,
        )

        return results

    async def stress_test_correlation(
        self,
        tickers: list[str],
        scenario: str = "2008_crisis",
    ) -> CorrelationMatrix:
        """
        Compute historical correlation matrix during a named stress scenario.

        Available scenarios:
          "2008_crisis"   — Sep 2008–Mar 2009
          "covid_crash"   — Feb–Apr 2020
          "dot_com"       — Mar 2000–Oct 2002
          "2022_bear"     — Full year 2022
          "taper_tantrum" — May–Sep 2013

        Args:
            tickers: Tickers to stress-test.
            scenario: Named stress period from STRESS_SCENARIOS.

        Returns:
            CorrelationMatrix for the stress period.

        Raises:
            ValueError: If scenario is not recognised.
        """
        if scenario not in STRESS_SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Available: {list(STRESS_SCENARIOS.keys())}"
            )

        start_str, end_str = STRESS_SCENARIOS[scenario]
        start_dt = date.fromisoformat(start_str)
        end_dt = date.fromisoformat(end_str)
        window_days = (end_dt - start_dt).days

        returns = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._download_returns_range(tickers, start_str, end_str),
        )

        if returns.empty:
            raise ValueError(
                f"No data for scenario '{scenario}' "
                f"({start_str} → {end_str}). "
                "Some tickers may not have data for this period."
            )

        available_tickers = list(returns.columns)
        corr_matrix, avg_corr, dispersion = self._ledoit_wolf_corr(returns)
        regime = self._classify_regime_from_corr(avg_corr)

        logger.info(
            "stress_test_correlation.complete",
            scenario=scenario,
            tickers=available_tickers,
            n_obs=len(returns),
            avg_corr=round(avg_corr, 4),
        )

        return CorrelationMatrix(
            tickers=available_tickers,
            as_of=end_dt,
            window_days=window_days,
            matrix=corr_matrix.tolist(),
            regime=regime,
            avg_pairwise_corr=round(avg_corr, 4),
            corr_dispersion=round(dispersion, 4),
            n_observations=len(returns),
        )

    def classify_regime_from_signals(
        self,
        vix: float,
        curve_slope: float,
        credit_spread: float,
        momentum: float,
    ) -> tuple[str, float]:
        """
        Stateless rule-based regime classifier from four scalar signals.

        Args:
            vix: Current VIX level.
            curve_slope: TLT/SHY ratio momentum (positive = steepening).
            credit_spread: HYG vs LQD 20-day return differential.
            momentum: SPY 50d/200d MA ratio minus 1 (positive = above 200d MA).

        Returns:
            Tuple of (regime_str, confidence_float).
            Regime is one of: "risk_on", "risk_off", "transition", "crisis".
        """
        signal_votes: dict[str, str] = {
            "vix": self._vix_to_regime(vix),
            "yield_curve": (
                "risk_off" if curve_slope < -0.05
                else "risk_on" if curve_slope > 0.05
                else "transition"
            ),
            "credit_spread": (
                "crisis" if credit_spread < -0.05
                else "risk_off" if credit_spread < -0.02
                else "risk_on" if credit_spread > 0.01
                else "transition"
            ),
            "equity_momentum": (
                "crisis" if momentum < -0.10
                else "risk_off" if momentum < -0.03
                else "risk_on" if momentum > 0.02
                else "transition"
            ),
        }
        return self._aggregate_votes(signal_votes)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    async def _fetch_returns(
        self,
        tickers: list[str],
        days: int,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Download adjusted closes for a list of tickers via yfinance
        (in a thread executor to avoid blocking the event loop) and return
        log returns as a DataFrame.

        Handles MultiIndex columns from multi-ticker yf.download.
        Drops tickers with fewer than 20 valid observations.
        """
        end = end_date or date.today()
        start = end - timedelta(days=days + 30)   # buffer for weekends/holidays

        def _download() -> pd.DataFrame:
            try:
                raw = yf.download(
                    tickers,
                    start=start.strftime("%Y-%m-%d"),
                    end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
            except Exception as exc:
                logger.error("yfinance download failed: %s", exc)
                return pd.DataFrame()

            if raw.empty:
                return pd.DataFrame()

            # Extract "Close" column (auto_adjust renames Adj Close → Close)
            if isinstance(raw.columns, pd.MultiIndex):
                if "Close" in raw.columns.get_level_values(0):
                    closes = raw["Close"]
                else:
                    closes = raw.iloc[:, 0].to_frame()
            else:
                closes = raw[["Close"]] if "Close" in raw.columns else raw

            # Flatten single-ticker MultiIndex
            if isinstance(closes.columns, pd.MultiIndex):
                closes.columns = closes.columns.droplevel(0)

            # Log returns
            log_ret = np.log(closes / closes.shift(1)).dropna(how="all")

            # Drop tickers with fewer than 20 valid observations
            valid_cols = [
                col for col in log_ret.columns
                if log_ret[col].dropna().shape[0] >= 20
            ]
            return log_ret[valid_cols]

        returns = await asyncio.get_event_loop().run_in_executor(None, _download)
        return returns

    async def _fetch_prices(
        self,
        tickers: list[str],
        days: int,
    ) -> pd.DataFrame:
        """Fetch raw adjusted close prices (not returns)."""
        end = date.today()
        start = end - timedelta(days=days + 30)

        def _download() -> pd.DataFrame:
            try:
                raw = yf.download(
                    tickers,
                    start=start.strftime("%Y-%m-%d"),
                    end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
            except Exception as exc:
                logger.error("yfinance price download failed: %s", exc)
                return pd.DataFrame()

            if raw.empty:
                return pd.DataFrame()

            if isinstance(raw.columns, pd.MultiIndex):
                if "Close" in raw.columns.get_level_values(0):
                    closes = raw["Close"]
                else:
                    closes = raw.iloc[:, 0].to_frame()
            else:
                closes = raw[["Close"]] if "Close" in raw.columns else raw

            if isinstance(closes.columns, pd.MultiIndex):
                closes.columns = closes.columns.droplevel(0)

            return closes.dropna(how="all")

        return await asyncio.get_event_loop().run_in_executor(None, _download)

    @staticmethod
    def _download_returns_range(
        tickers: list[str], start_str: str, end_str: str
    ) -> pd.DataFrame:
        """Synchronous price download for a specific date range (used in executor)."""
        try:
            raw = yf.download(
                tickers,
                start=start_str,
                end=end_str,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            logger.error("yfinance range download failed: %s", exc)
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                closes = raw["Close"]
            else:
                closes = raw.iloc[:, 0].to_frame()
        else:
            closes = raw[["Close"]] if "Close" in raw.columns else raw

        if isinstance(closes.columns, pd.MultiIndex):
            closes.columns = closes.columns.droplevel(0)

        log_ret = np.log(closes / closes.shift(1)).dropna(how="all")
        valid_cols = [c for c in log_ret.columns if log_ret[c].dropna().shape[0] >= 10]
        return log_ret[valid_cols]

    @staticmethod
    def _ledoit_wolf_corr(
        returns: pd.DataFrame,
    ) -> tuple[np.ndarray, float, float]:
        """
        Compute Ledoit-Wolf shrinkage correlation matrix.

        1. Drop all-NaN rows; fill remaining NaN with column means.
        2. Fit LedoitWolf on the cleaned return matrix.
        3. Convert shrinkage covariance → correlation matrix.
        4. Compute avg off-diagonal correlation and dispersion.

        Args:
            returns: DataFrame of log returns (rows=dates, cols=tickers).

        Returns:
            Tuple of (corr_matrix: np.ndarray, avg_corr: float, dispersion: float).
        """
        clean = returns.dropna(how="all")
        # Fill remaining NaN with column mean (avoids LedoitWolf failure)
        clean = clean.fillna(clean.mean())

        n_assets = clean.shape[1]
        if n_assets == 0:
            return np.array([[]]), 0.0, 0.0

        if n_assets == 1:
            return np.array([[1.0]]), 1.0, 0.0

        X = clean.values.astype(float)

        try:
            lw = LedoitWolf()
            lw.fit(X)
            cov = lw.covariance_
        except Exception as exc:
            logger.warning("LedoitWolf failed, falling back to sample corr: %s", exc)
            corr_df = clean.corr(method="pearson").fillna(0)
            cov = corr_df.values.astype(float)

        # Convert covariance to correlation
        std = np.sqrt(np.diag(cov))
        std[std == 0] = 1.0   # guard against zero-variance columns
        outer_std = np.outer(std, std)
        corr_matrix = cov / outer_std

        # Clip to [-1, 1] (numerical precision)
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        # Enforce diagonal = 1
        np.fill_diagonal(corr_matrix, 1.0)

        # Off-diagonal values for statistics
        mask = np.ones((n_assets, n_assets), dtype=bool)
        np.fill_diagonal(mask, False)
        upper_tri = corr_matrix[np.triu(mask)]

        avg_corr = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0
        dispersion = float(np.std(upper_tri)) if len(upper_tri) > 0 else 0.0

        return corr_matrix, avg_corr, dispersion

    @staticmethod
    def _classify_regime_from_corr(avg_corr: float) -> str:
        """Classify regime from average pairwise correlation alone."""
        if avg_corr > _CORR_CRISIS:
            return "crisis"
        if avg_corr > _CORR_RISK_OFF:
            return "risk_off"
        if avg_corr < 0.30:
            return "risk_on"
        return "transition"

    @staticmethod
    def _vix_to_regime(vix: float) -> str:
        """Map VIX level to a regime label."""
        if vix >= _VIX_CRISIS:
            return "crisis"
        if vix >= _VIX_RISK_OFF:
            return "risk_off"
        if vix <= _VIX_RISK_ON:
            return "risk_on"
        return "transition"

    @staticmethod
    def _aggregate_votes(
        signal_votes: dict[str, str],
    ) -> tuple[str, float]:
        """
        Aggregate regime signal votes by plurality.

        Crisis votes are escalated: if any vote is "crisis", the overall
        regime can only be "crisis" or "risk_off" (not "risk_on").

        Args:
            signal_votes: Mapping of signal_name → regime_vote.

        Returns:
            Tuple of (winning_regime, confidence) where confidence is the
            fraction of votes for the winning regime.
        """
        from collections import Counter

        if not signal_votes:
            return "transition", 0.5

        counts: Counter = Counter(signal_votes.values())
        total = sum(counts.values())

        # Check for crisis escalation: any crisis vote upgrades risk_on to transition
        has_crisis = counts.get("crisis", 0) > 0
        has_risk_off = counts.get("risk_off", 0) > 0

        # Plurality winner
        winner, winner_count = counts.most_common(1)[0]

        # Escalation override: if crisis signals present but not plurality,
        # and risk_on would win — downgrade to transition
        if winner == "risk_on" and (has_crisis or has_risk_off):
            if counts.get("risk_on", 0) < (counts.get("risk_off", 0) + counts.get("crisis", 0)):
                winner = "transition"
                winner_count = counts.get("transition", 0) + 1

        confidence = winner_count / total if total > 0 else 0.5

        return winner, confidence


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------

async def correlation_matrix(
    tickers: list[str], window: int = 63
) -> CorrelationMatrix:
    """Compute Ledoit-Wolf correlation matrix for a list of tickers."""
    monitor = CorrelationMonitor()
    return await monitor.get_correlation_matrix(tickers, window_days=window)


async def detect_regime() -> PortfolioRegime:
    """Run multi-factor macro regime detection using public ETF proxies."""
    monitor = CorrelationMonitor()
    return await monitor.detect_regime()


async def portfolio_diversification(
    tickers: list[str],
) -> tuple[CorrelationMatrix, list[CorrelationAlert]]:
    """Monitor portfolio correlations and return matrix + alerts."""
    monitor = CorrelationMonitor()
    return await monitor.monitor_portfolio_correlations(tickers)
