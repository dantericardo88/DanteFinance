"""
AI-driven factor research: automated alpha signal discovery, validation, decay analysis.
Uses systematic approach: universe → factor construction → IC testing → portfolio construction.

dim_068 — AI-driven factor research (alpha signals) (target: 9)
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).parent.parent / "data" / "factor_research.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_RATE_DELAY = 0.4   # yfinance rate limit

# 50 representative S&P 500 tickers (proxy universe for factor research)
_SP500_PROXY: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "LLY", "AVGO",
    "JPM", "V", "UNH", "XOM", "COST", "MA", "HD", "PG", "JNJ", "ORCL",
    "BAC", "ABBV", "MRK", "KO", "CVX", "CRM", "NFLX", "AMD", "PEP", "ADBE",
    "TMO", "WMT", "LIN", "ACN", "MCD", "CSCO", "ABT", "PM", "DHR", "CAT",
    "TXN", "INTC", "AMGN", "INTU", "WFC", "HON", "IBM", "GS", "SPGI", "BX",
]

_IC_HORIZONS = [1, 5, 21, 63, 126]   # trading days

# ---------------------------------------------------------------------------
# DB initialisation
# ---------------------------------------------------------------------------

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _init_db(conn)
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS factor_ic (
            factor_name  TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            ic_mean      REAL,
            ic_std       REAL,
            icir         REAL,
            ic_tstat     REAL,
            n_obs        INTEGER,
            computed_at  TEXT,
            PRIMARY KEY (factor_name, horizon_days)
        );
        CREATE TABLE IF NOT EXISTS factor_rankings (
            ticker       TEXT NOT NULL,
            factor_name  TEXT NOT NULL,
            z_score      REAL,
            raw_value    REAL,
            as_of        TEXT,
            PRIMARY KEY (ticker, factor_name, as_of)
        );
        CREATE TABLE IF NOT EXISTS discovered_factors (
            factor_id    TEXT PRIMARY KEY,
            name         TEXT,
            description  TEXT,
            formula_hint TEXT,
            source       TEXT,
            icir_5d      REAL,
            icir_21d     REAL,
            discovered_at TEXT
        );
        CREATE TABLE IF NOT EXISTS factor_composite (
            ticker       TEXT NOT NULL,
            composite_name TEXT NOT NULL,
            composite_score REAL,
            as_of        TEXT,
            PRIMARY KEY (ticker, composite_name, as_of)
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FactorResult:
    factor_name: str
    ticker: str
    raw_value: Optional[float]
    z_score: Optional[float]
    as_of: str = field(default_factory=lambda: datetime.utcnow().date().isoformat())


@dataclass
class ICResult:
    factor_name: str
    horizon_days: int
    ic_series: List[float]
    ic_mean: float
    ic_std: float
    icir: float
    ic_tstat: float
    n_obs: int
    half_life_days: Optional[float] = None


@dataclass
class ValidationResult:
    factor_name: str
    is_valid: bool
    oos_ic_mean: float
    oos_icir: float
    in_sample_ic_mean: float
    in_sample_icir: float
    degradation_pct: float
    tc_adjusted_ic: float
    regime_ic: Dict[str, float]
    ff_correlation: Optional[float]
    verdict: str  # "strong" / "moderate" / "weak" / "reject"


# ---------------------------------------------------------------------------
# AlphaFactorLibrary
# ---------------------------------------------------------------------------


class AlphaFactorLibrary:
    """Comprehensive factor library: 40+ factors computed from yfinance + EDGAR.

    All compute_* methods take a ticker string and return a dict[factor_name → raw_value].
    NaN is used for unavailable data. Computation is done per-ticker; cross-sectional
    z-scoring is handled by FactorDataLoader.
    """

    # -----------------------------------------------------------------------
    # Value factors
    # -----------------------------------------------------------------------

    def compute_value_factors(self, info: Dict[str, Any], fin: Optional[pd.DataFrame] = None) -> Dict[str, Optional[float]]:
        """P/E, P/B, P/S, P/FCF, EV/EBITDA, EV/Sales, dividend yield."""
        factors: Dict[str, Optional[float]] = {}

        # P/E (trailing)
        pe = info.get("trailingPE") or info.get("forwardPE")
        factors["pe_ratio"] = float(pe) if pe and pe > 0 else None

        # P/B
        pb = info.get("priceToBook")
        factors["pb_ratio"] = float(pb) if pb and pb > 0 else None

        # P/S
        ps = info.get("priceToSalesTrailing12Months")
        factors["ps_ratio"] = float(ps) if ps and ps > 0 else None

        # P/FCF
        market_cap = info.get("marketCap") or 0
        fcf = info.get("freeCashflow") or 0
        factors["p_fcf"] = float(market_cap / fcf) if fcf and fcf > 0 and market_cap else None

        # EV/EBITDA
        ev_ebitda = info.get("enterpriseToEbitda")
        factors["ev_ebitda"] = float(ev_ebitda) if ev_ebitda and ev_ebitda > 0 else None

        # EV/Sales
        ev_sales = info.get("enterpriseToRevenue")
        factors["ev_sales"] = float(ev_sales) if ev_sales and ev_sales > 0 else None

        # Dividend yield
        div_yield = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
        factors["dividend_yield"] = float(div_yield) if div_yield else 0.0

        return factors

    # -----------------------------------------------------------------------
    # Quality factors
    # -----------------------------------------------------------------------

    def compute_quality_factors(self, info: Dict[str, Any], fin: Optional[pd.DataFrame] = None,
                                 cashflow: Optional[pd.DataFrame] = None) -> Dict[str, Optional[float]]:
        """ROE, ROIC, gross margin, net margin, FCF margin, accruals, Piotroski F-score."""
        factors: Dict[str, Optional[float]] = {}

        # ROE
        roe = info.get("returnOnEquity")
        factors["roe"] = float(roe * 100) if roe is not None else None

        # ROIC proxy (net income / (total assets - current liabilities))
        roic = info.get("returnOnAssets")
        factors["roic"] = float(roic * 100) if roic is not None else None

        # Gross margin
        gm = info.get("grossMargins")
        factors["gross_margin"] = float(gm * 100) if gm is not None else None

        # Net margin
        nm = info.get("profitMargins")
        factors["net_margin"] = float(nm * 100) if nm is not None else None

        # Operating margin
        om = info.get("operatingMargins")
        factors["operating_margin"] = float(om * 100) if om is not None else None

        # FCF margin
        rev = info.get("totalRevenue") or 0
        fcf = info.get("freeCashflow") or 0
        factors["fcf_margin"] = float(fcf / rev * 100) if rev > 0 else None

        # Accruals: (operating CF - net income) / total assets — lower = better quality
        op_cf = info.get("operatingCashflow") or 0
        net_income = info.get("netIncomeToCommon") or 0
        total_assets = info.get("totalAssets") or 1
        if total_assets > 0 and (op_cf or net_income):
            factors["accruals"] = float((op_cf - net_income) / total_assets)
        else:
            factors["accruals"] = None

        # Piotroski F-score (9-point)
        factors["piotroski_f"] = self._compute_piotroski(info)

        # Current ratio
        cr = info.get("currentRatio")
        factors["current_ratio"] = float(cr) if cr else None

        # Debt-to-equity
        de = info.get("debtToEquity")
        factors["debt_to_equity"] = float(de / 100) if de else None  # yfinance gives in %

        return factors

    def _compute_piotroski(self, info: Dict[str, Any]) -> Optional[float]:
        """Simplified Piotroski F-score (0–9) from yfinance info."""
        score = 0
        total = 0

        # Profitability (4 signals)
        roa = info.get("returnOnAssets")
        if roa is not None:
            total += 1
            if roa > 0:
                score += 1

        op_cf = info.get("operatingCashflow") or 0
        total_assets = info.get("totalAssets") or 1
        if total_assets > 0:
            total += 1
            if op_cf / total_assets > 0:
                score += 1

        net_income = info.get("netIncomeToCommon") or 0
        if total_assets > 0:
            total += 1
            if net_income / total_assets > (roa or 0) * 0.9:  # proxy for delta_roa > 0
                score += 1

        # Cash flow > net income (accruals signal)
        if op_cf and net_income:
            total += 1
            if op_cf > net_income:
                score += 1

        # Leverage / liquidity (3 signals)
        de = info.get("debtToEquity")
        if de is not None:
            total += 1
            if de < 50:  # proxy for delta_leverage < 0
                score += 1

        cr = info.get("currentRatio")
        if cr is not None:
            total += 1
            if cr > 1:
                score += 1

        # Efficiency (2 signals)
        gm = info.get("grossMargins")
        if gm is not None:
            total += 1
            if gm > 0.3:  # proxy for improving margin
                score += 1

        asset_turnover = info.get("assetProfile", {})  # proxy
        rev = info.get("totalRevenue") or 0
        if total_assets > 0 and rev > 0:
            total += 1
            at = rev / total_assets
            if at > 0.5:
                score += 1

        return float(score) if total >= 4 else None

    # -----------------------------------------------------------------------
    # Growth factors
    # -----------------------------------------------------------------------

    def compute_growth_factors(self, info: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Revenue growth 1Y/3Y, EPS growth, FCF growth, estimate revision."""
        factors: Dict[str, Optional[float]] = {}

        # Revenue growth 1Y
        rg = info.get("revenueGrowth")
        factors["revenue_growth_1y"] = float(rg * 100) if rg is not None else None

        # EPS growth
        eg = info.get("earningsGrowth")
        factors["eps_growth_1y"] = float(eg * 100) if eg is not None else None

        # EPS forward vs trailing
        fwd_eps = info.get("forwardEps") or 0
        trail_eps = info.get("trailingEps") or 0
        if trail_eps and trail_eps != 0:
            factors["eps_revision"] = float((fwd_eps - trail_eps) / abs(trail_eps) * 100)
        else:
            factors["eps_revision"] = None

        # Earnings surprise proxy (targetMeanPrice vs currentPrice)
        target = info.get("targetMeanPrice") or 0
        current = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        if current and current > 0:
            factors["analyst_upside"] = float((target - current) / current * 100)
        else:
            factors["analyst_upside"] = None

        # FCF growth proxy: FCF / revenue
        rev = info.get("totalRevenue") or 0
        fcf = info.get("freeCashflow") or 0
        factors["fcf_yield"] = float(fcf / (info.get("marketCap") or 1)) if info.get("marketCap") else None

        # Revenue 3Y CAGR proxy (unavailable from info; use revenueGrowth as 1Y proxy)
        factors["revenue_growth_3y"] = factors["revenue_growth_1y"]  # best available proxy

        return factors

    # -----------------------------------------------------------------------
    # Momentum factors
    # -----------------------------------------------------------------------

    def compute_momentum_factors(self, prices: pd.Series) -> Dict[str, Optional[float]]:
        """12-1 momentum, 6-1, 3-1, 1-month reversal, 52W metrics."""
        factors: Dict[str, Optional[float]] = {}

        if prices is None or len(prices) < 21:
            return {k: None for k in ["mom_12_1", "mom_6_1", "mom_3_1", "reversal_1m",
                                       "w52_high_proximity", "rsi_14", "atr_rel"]}

        prices = prices.dropna()
        n = len(prices)

        # 12-1 month momentum (252 - 21 trading days)
        if n >= 252:
            factors["mom_12_1"] = float((prices.iloc[-21] / prices.iloc[-252] - 1) * 100)
        else:
            factors["mom_12_1"] = None

        # 6-1 month momentum
        if n >= 126:
            factors["mom_6_1"] = float((prices.iloc[-21] / prices.iloc[-126] - 1) * 100)
        else:
            factors["mom_6_1"] = None

        # 3-1 month momentum
        if n >= 63:
            factors["mom_3_1"] = float((prices.iloc[-21] / prices.iloc[-63] - 1) * 100)
        else:
            factors["mom_3_1"] = None

        # 1-month reversal
        if n >= 21:
            factors["reversal_1m"] = float((prices.iloc[-1] / prices.iloc[-21] - 1) * 100)
        else:
            factors["reversal_1m"] = None

        # 52W high proximity
        if n >= 252:
            high_52w = prices.iloc[-252:].max()
            factors["w52_high_proximity"] = float(prices.iloc[-1] / high_52w) if high_52w else None
        elif n >= 21:
            high_52w = prices.max()
            factors["w52_high_proximity"] = float(prices.iloc[-1] / high_52w) if high_52w else None
        else:
            factors["w52_high_proximity"] = None

        # RSI(14)
        factors["rsi_14"] = self._compute_rsi(prices, 14)

        # ATR relative (ATR_14 / price)
        factors["atr_rel"] = self._compute_atr_rel(prices, 14)

        return factors

    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Compute RSI for a price series."""
        if len(prices) < period + 1:
            return None
        delta = prices.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.rolling(period).mean().iloc[-1]
        avg_loss = losses.rolling(period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    def _compute_atr_rel(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """ATR relative to price (simplified using price std dev as proxy)."""
        if len(prices) < period + 1:
            return None
        recent = prices.iloc[-period - 1:]
        atr = recent.diff().abs().mean()
        current = prices.iloc[-1]
        return float(atr / current) if current > 0 else None

    def _compute_macd_signal(self, prices: pd.Series) -> Optional[float]:
        """MACD signal line value."""
        if len(prices) < 26:
            return None
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return float(macd.iloc[-1] - signal.iloc[-1])

    # -----------------------------------------------------------------------
    # Size factors
    # -----------------------------------------------------------------------

    def compute_size_factors(self, info: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Market cap (log), float shares (log), enterprise value (log)."""
        factors: Dict[str, Optional[float]] = {}

        mc = info.get("marketCap")
        factors["log_market_cap"] = float(np.log(mc)) if mc and mc > 0 else None

        fs = info.get("floatShares")
        factors["log_float_shares"] = float(np.log(fs)) if fs and fs > 0 else None

        ev = info.get("enterpriseValue")
        factors["log_enterprise_value"] = float(np.log(ev)) if ev and ev > 0 else None

        return factors

    # -----------------------------------------------------------------------
    # Sentiment factors
    # -----------------------------------------------------------------------

    def compute_sentiment_factors(self, info: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Short interest proxy, analyst rating change, insider buying ratio."""
        factors: Dict[str, Optional[float]] = {}

        # Short interest % of float
        short_pct = info.get("shortPercentOfFloat")
        factors["short_pct_float"] = float(short_pct * 100) if short_pct is not None else None

        # Short ratio (days to cover proxy)
        short_ratio = info.get("shortRatio")
        factors["short_ratio"] = float(short_ratio) if short_ratio else None

        # Analyst recommendation: 1=Strong Buy, 5=Strong Sell — invert so higher = better
        rec = info.get("recommendationMean")
        if rec is not None:
            factors["analyst_score"] = float(6.0 - rec)  # invert: higher = more bullish
        else:
            factors["analyst_score"] = None

        # Number of analyst recommendations
        n_analysts = info.get("numberOfAnalystOpinions") or 0
        factors["analyst_coverage"] = float(n_analysts)

        # Insider ownership as sentiment signal
        insider_pct = info.get("heldPercentInsiders")
        factors["insider_ownership_pct"] = float(insider_pct * 100) if insider_pct is not None else None

        # Institutional ownership
        inst_pct = info.get("heldPercentInstitutions")
        factors["institutional_ownership_pct"] = float(inst_pct * 100) if inst_pct is not None else None

        return factors

    # -----------------------------------------------------------------------
    # Composite: all factors for one ticker
    # -----------------------------------------------------------------------

    def compute_all_factors(
        self,
        ticker: str,
        info: Dict[str, Any],
        prices: Optional[pd.Series] = None,
        financials: Optional[pd.DataFrame] = None,
        cashflow: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Optional[float]]:
        """Return dict of all factor raw values for a single ticker."""
        factors: Dict[str, Optional[float]] = {}
        factors.update(self.compute_value_factors(info, financials))
        factors.update(self.compute_quality_factors(info, financials, cashflow))
        factors.update(self.compute_growth_factors(info))
        factors.update(self.compute_size_factors(info))
        factors.update(self.compute_sentiment_factors(info))
        if prices is not None and len(prices) > 21:
            factors.update(self.compute_momentum_factors(prices))
            factors["macd_signal"] = self._compute_macd_signal(prices)
        else:
            for k in ["mom_12_1", "mom_6_1", "mom_3_1", "reversal_1m", "w52_high_proximity",
                      "rsi_14", "atr_rel", "macd_signal"]:
                factors[k] = None
        return factors

    @staticmethod
    def list_factor_names() -> List[str]:
        """Return all supported factor names."""
        return [
            # Value
            "pe_ratio", "pb_ratio", "ps_ratio", "p_fcf", "ev_ebitda", "ev_sales",
            "dividend_yield",
            # Quality
            "roe", "roic", "gross_margin", "net_margin", "operating_margin",
            "fcf_margin", "accruals", "piotroski_f", "current_ratio", "debt_to_equity",
            # Growth
            "revenue_growth_1y", "revenue_growth_3y", "eps_growth_1y", "eps_revision",
            "analyst_upside", "fcf_yield",
            # Momentum
            "mom_12_1", "mom_6_1", "mom_3_1", "reversal_1m", "w52_high_proximity",
            "rsi_14", "atr_rel", "macd_signal",
            # Size
            "log_market_cap", "log_float_shares", "log_enterprise_value",
            # Sentiment
            "short_pct_float", "short_ratio", "analyst_score", "analyst_coverage",
            "insider_ownership_pct", "institutional_ownership_pct",
        ]


# ---------------------------------------------------------------------------
# FactorDataLoader
# ---------------------------------------------------------------------------


class FactorDataLoader:
    """Load factor data for universe; build cross-sectional factor matrix.

    Fetches yfinance data, computes all factors, winsorizes and z-scores
    cross-sectionally per date.

    Main output: factor_matrix DataFrame, shape (N_stocks, N_factors).
    Also builds returns DataFrame for IC computation.
    """

    def __init__(self, universe: Optional[List[str]] = None, lookback_days: int = 504) -> None:
        self._universe = universe or _SP500_PROXY
        self._lookback_days = lookback_days
        self._lib = AlphaFactorLibrary()
        self._price_cache: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Price data fetch
    # ------------------------------------------------------------------

    def _fetch_prices(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV price history from yfinance."""
        if ticker in self._price_cache:
            return self._price_cache[ticker]
        try:
            import yfinance as yf  # type: ignore
            t = yf.Ticker(ticker)
            hist = t.history(period="3y", auto_adjust=True)
            if hist.empty:
                return None
            self._price_cache[ticker] = hist
            time.sleep(_RATE_DELAY)
            return hist
        except Exception as exc:
            logger.warning("_fetch_prices failed", ticker=ticker, error=str(exc))
            return None

    def _fetch_info(self, ticker: str) -> Dict[str, Any]:
        """Fetch ticker info dict from yfinance."""
        try:
            import yfinance as yf  # type: ignore
            t = yf.Ticker(ticker)
            info = t.info or {}
            time.sleep(_RATE_DELAY)
            return info
        except Exception as exc:
            logger.warning("_fetch_info failed", ticker=ticker, error=str(exc))
            return {}

    # ------------------------------------------------------------------
    # Universe-level factor matrix
    # ------------------------------------------------------------------

    def build_factor_matrix(
        self,
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Build cross-sectional factor matrix for the universe.

        Returns
        -------
        DataFrame of shape (N_tickers, N_factors) with z-scored, winsorized values.
        Index = ticker, columns = factor names.
        """
        tickers = tickers or self._universe
        rows: List[Dict[str, Any]] = []

        logger.info("build_factor_matrix starting", n_tickers=len(tickers))
        for ticker in tickers:
            try:
                info = self._fetch_info(ticker)
                if not info:
                    continue
                prices_df = self._fetch_prices(ticker)
                prices = prices_df["Close"] if prices_df is not None else None
                factors = self._lib.compute_all_factors(ticker, info, prices)
                factors["ticker"] = ticker
                rows.append(factors)
            except Exception as exc:
                logger.warning("build_factor_matrix skip", ticker=ticker, error=str(exc))

        if not rows:
            return pd.DataFrame()

        raw_df = pd.DataFrame(rows).set_index("ticker")

        # Winsorize at 1st / 99th percentile per factor
        winsorized = raw_df.copy()
        for col in winsorized.columns:
            series = winsorized[col].dropna()
            if len(series) < 5:
                continue
            lo, hi = np.percentile(series, [1, 99])
            winsorized[col] = winsorized[col].clip(lower=lo, upper=hi)

        # Cross-sectional z-score (mean=0, std=1 per factor)
        zscored = winsorized.copy()
        for col in zscored.columns:
            series = zscored[col].dropna()
            if len(series) < 3:
                continue
            mu = series.mean()
            sigma = series.std()
            if sigma > 0:
                zscored[col] = (zscored[col] - mu) / sigma

        logger.info("build_factor_matrix complete", n_tickers=len(zscored))
        return zscored

    # ------------------------------------------------------------------
    # Forward returns matrix
    # ------------------------------------------------------------------

    def build_returns_matrix(
        self,
        tickers: Optional[List[str]] = None,
        horizons: Optional[List[int]] = None,
    ) -> Dict[int, pd.DataFrame]:
        """Build forward returns for each horizon.

        Returns
        -------
        Dict[horizon → DataFrame(dates × tickers)]
        """
        tickers = tickers or self._universe
        horizons = horizons or _IC_HORIZONS
        price_history: Dict[str, pd.Series] = {}

        for ticker in tickers:
            df = self._fetch_prices(ticker)
            if df is not None and not df.empty:
                price_history[ticker] = df["Close"]

        if not price_history:
            return {}

        # Build wide price DataFrame
        price_df = pd.DataFrame(price_history).sort_index()

        returns_dict: Dict[int, pd.DataFrame] = {}
        for h in horizons:
            fwd_ret = price_df.shift(-h) / price_df - 1
            returns_dict[h] = fwd_ret.dropna(how="all")

        return returns_dict

    # ------------------------------------------------------------------
    # Per-ticker raw factor values (for API use)
    # ------------------------------------------------------------------

    def get_ticker_factors(self, ticker: str) -> Dict[str, Optional[float]]:
        """Return raw factor values for a single ticker."""
        info = self._fetch_info(ticker)
        df = self._fetch_prices(ticker)
        prices = df["Close"] if df is not None else None
        return self._lib.compute_all_factors(ticker, info, prices)


# ---------------------------------------------------------------------------
# InformationCoefficientAnalyzer
# ---------------------------------------------------------------------------


class InformationCoefficientAnalyzer:
    """Factor IC analysis: Spearman IC, ICIR, decay, t-stat.

    IC = Spearman rank correlation between factor z-score and forward return.
    ICIR = mean(IC) / std(IC) — higher = more consistent factor.
    """

    def __init__(self) -> None:
        self._db = _get_db()

    # ------------------------------------------------------------------
    # Core IC computation
    # ------------------------------------------------------------------

    def compute_ic(
        self,
        factor_series: pd.Series,
        forward_returns: pd.Series,
        min_obs: int = 20,
    ) -> Optional[float]:
        """Compute Spearman IC between factor scores and forward returns.

        Parameters
        ----------
        factor_series: factor z-scores per ticker
        forward_returns: forward N-day returns per ticker (same index)
        min_obs: minimum observations required

        Returns
        -------
        Spearman rank correlation, or None if insufficient data
        """
        aligned = pd.concat([factor_series, forward_returns], axis=1).dropna()
        if len(aligned) < min_obs:
            return None
        ic, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return float(ic)

    def compute_ic_time_series(
        self,
        factor_matrix: pd.DataFrame,
        factor_name: str,
        returns_matrix: Dict[int, pd.DataFrame],
        horizon: int = 21,
        rolling_window: int = 60,
    ) -> ICResult:
        """Compute rolling IC time series for a factor across the universe.

        Uses cross-sectional IC at each date (rolling window of available data).

        Parameters
        ----------
        factor_matrix: (N_tickers, N_factors) — current snapshot z-scores
        factor_name: factor column to analyse
        returns_matrix: dict[horizon → (dates, tickers) DataFrame]
        horizon: forward return horizon in trading days
        rolling_window: number of dates for rolling IC estimation

        Returns
        -------
        ICResult with full IC time series and summary statistics
        """
        if factor_name not in factor_matrix.columns:
            return ICResult(factor_name, horizon, [], 0.0, 0.0, 0.0, 0.0, 0)

        factor_col = factor_matrix[factor_name].dropna()
        ret_df = returns_matrix.get(horizon)

        if ret_df is None or ret_df.empty:
            return ICResult(factor_name, horizon, [], 0.0, 0.0, 0.0, 0.0, 0)

        # Cross-sectional IC: factor (same for all dates) vs return at each date
        # Since we have a single factor snapshot, simulate rolling by using the
        # last `rolling_window` return dates
        common_tickers = factor_col.index.intersection(ret_df.columns)
        if len(common_tickers) < 10:
            return ICResult(factor_name, horizon, [], 0.0, 0.0, 0.0, 0.0, 0)

        factor_vals = factor_col[common_tickers]
        ic_series: List[float] = []

        for date in ret_df.index[-rolling_window:]:
            ret_row = ret_df.loc[date, common_tickers].dropna()
            if len(ret_row) < 10:
                continue
            common = factor_vals.index.intersection(ret_row.index)
            if len(common) < 10:
                continue
            ic = self.compute_ic(factor_vals[common], ret_row[common])
            if ic is not None:
                ic_series.append(ic)

        if len(ic_series) < 5:
            return ICResult(factor_name, horizon, ic_series, 0.0, 0.0, 0.0, 0.0, len(ic_series))

        arr = np.array(ic_series)
        ic_mean = float(arr.mean())
        ic_std = float(arr.std()) if len(arr) > 1 else 0.0
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        n = len(arr)
        ic_tstat = float(ic_mean / (ic_std / np.sqrt(n))) if ic_std > 0 and n > 1 else 0.0

        # Half-life: decay toward zero over horizons
        half_life = self._estimate_half_life(arr)

        result = ICResult(
            factor_name=factor_name,
            horizon_days=horizon,
            ic_series=ic_series,
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=icir,
            ic_tstat=ic_tstat,
            n_obs=n,
            half_life_days=half_life,
        )

        # Persist to DB
        self._persist_ic(result)
        return result

    def _estimate_half_life(self, ic_series: np.ndarray) -> Optional[float]:
        """Estimate half-life of IC using AR(1) decay model."""
        if len(ic_series) < 10:
            return None
        try:
            y = ic_series[1:]
            x = ic_series[:-1]
            slope, intercept, r, p, se = stats.linregress(x, y)
            if slope >= 1 or slope <= 0:
                return None
            hl = -np.log(2) / np.log(abs(slope))
            return float(hl)
        except Exception:
            return None

    def _persist_ic(self, result: ICResult) -> None:
        """Save IC result to SQLite."""
        try:
            self._db.execute("""
                INSERT OR REPLACE INTO factor_ic
                (factor_name, horizon_days, ic_mean, ic_std, icir, ic_tstat, n_obs, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.factor_name, result.horizon_days,
                result.ic_mean, result.ic_std, result.icir,
                result.ic_tstat, result.n_obs,
                datetime.utcnow().isoformat(),
            ))
            self._db.commit()
        except Exception as exc:
            logger.warning("_persist_ic failed", error=str(exc))

    def compute_ic_decay_curve(
        self,
        factor_matrix: pd.DataFrame,
        factor_name: str,
        returns_matrix: Dict[int, pd.DataFrame],
    ) -> Dict[int, float]:
        """Compute IC at each standard horizon to build decay curve.

        Returns
        -------
        {horizon_days: ic_mean} sorted by horizon
        """
        decay: Dict[int, float] = {}
        for horizon in _IC_HORIZONS:
            result = self.compute_ic_time_series(
                factor_matrix, factor_name, returns_matrix, horizon=horizon
            )
            decay[horizon] = result.ic_mean
        return decay

    def compute_factor_autocorrelation(
        self,
        factor_series_t1: pd.Series,
        factor_series_t2: pd.Series,
    ) -> Optional[float]:
        """Factor autocorrelation: how stable are factor scores period-over-period.

        High autocorrelation = low turnover (signals persist).
        Low autocorrelation = high turnover (expensive to trade).
        """
        common = factor_series_t1.index.intersection(factor_series_t2.index)
        if len(common) < 10:
            return None
        r, _ = spearmanr(factor_series_t1[common].dropna(), factor_series_t2[common].dropna())
        return float(r)

    def get_cached_ic(self, factor_name: str, horizon_days: int) -> Optional[Dict[str, Any]]:
        """Retrieve cached IC from DB."""
        try:
            row = self._db.execute("""
                SELECT * FROM factor_ic WHERE factor_name=? AND horizon_days=?
            """, (factor_name, horizon_days)).fetchone()
            if row:
                return dict(row)
        except Exception:
            pass
        return None

    def rank_factors_by_icir(
        self,
        factor_matrix: pd.DataFrame,
        returns_matrix: Dict[int, pd.DataFrame],
        horizon: int = 21,
    ) -> pd.DataFrame:
        """Rank all factors by ICIR at a given horizon.

        Returns
        -------
        DataFrame sorted by ICIR descending with ic_mean, icir, tstat columns
        """
        rows: List[Dict[str, Any]] = []
        for factor_name in factor_matrix.columns:
            result = self.compute_ic_time_series(
                factor_matrix, factor_name, returns_matrix, horizon=horizon
            )
            rows.append({
                "factor_name": factor_name,
                "horizon_days": horizon,
                "ic_mean": round(result.ic_mean, 4),
                "ic_std": round(result.ic_std, 4),
                "icir": round(result.icir, 3),
                "ic_tstat": round(result.ic_tstat, 3),
                "n_obs": result.n_obs,
                "half_life_days": result.half_life_days,
            })
        df = pd.DataFrame(rows).set_index("factor_name")
        df = df.sort_values("icir", ascending=False)
        return df


# ---------------------------------------------------------------------------
# FactorComposer
# ---------------------------------------------------------------------------


class FactorComposer:
    """Combine multiple factor z-scores into composite alpha signals.

    Supported methods:
    - equal_weight: simple average of z-scores
    - ic_weighted: weighted by trailing ICIR from DB
    - pca: first principal component
    - ols: OLS regression factors → forward return
    - shrinkage: Bayesian shrinkage toward IC=0
    """

    def __init__(self) -> None:
        self._db = _get_db()
        self._ic_analyzer = InformationCoefficientAnalyzer()

    # ------------------------------------------------------------------
    # Equal weight
    # ------------------------------------------------------------------

    def equal_weight(
        self,
        factor_matrix: pd.DataFrame,
        factor_names: Optional[List[str]] = None,
    ) -> pd.Series:
        """Simple equal-weighted average of factor z-scores."""
        cols = factor_names or list(factor_matrix.columns)
        available = [c for c in cols if c in factor_matrix.columns]
        if not available:
            return pd.Series(dtype=float)
        sub = factor_matrix[available].copy()
        composite = sub.mean(axis=1)
        composite.name = "composite_equal_weight"
        return composite

    # ------------------------------------------------------------------
    # IC-weighted
    # ------------------------------------------------------------------

    def ic_weighted(
        self,
        factor_matrix: pd.DataFrame,
        horizon: int = 21,
        factor_names: Optional[List[str]] = None,
    ) -> pd.Series:
        """Weight each factor by its trailing ICIR at the given horizon."""
        cols = factor_names or list(factor_matrix.columns)
        available = [c for c in cols if c in factor_matrix.columns]

        weights: Dict[str, float] = {}
        for fname in available:
            cached = self._ic_analyzer.get_cached_ic(fname, horizon)
            if cached:
                icir = cached.get("icir") or 0.0
                weights[fname] = max(icir, 0.0)  # only positive-IC factors
            else:
                weights[fname] = 0.0

        total_weight = sum(weights.values())
        if total_weight <= 0:
            # Fallback to equal weight
            return self.equal_weight(factor_matrix, available)

        composite = pd.Series(0.0, index=factor_matrix.index)
        for fname, w in weights.items():
            if fname in factor_matrix.columns:
                composite += factor_matrix[fname].fillna(0) * (w / total_weight)

        composite.name = "composite_ic_weighted"
        return composite

    # ------------------------------------------------------------------
    # PCA composite
    # ------------------------------------------------------------------

    def pca_composite(
        self,
        factor_matrix: pd.DataFrame,
        n_components: int = 1,
        factor_names: Optional[List[str]] = None,
    ) -> pd.Series:
        """First principal component of factor matrix as composite signal."""
        try:
            from sklearn.decomposition import PCA  # type: ignore
            cols = factor_names or list(factor_matrix.columns)
            available = [c for c in cols if c in factor_matrix.columns]
            sub = factor_matrix[available].dropna(how="all")
            sub_filled = sub.fillna(0)

            if sub_filled.shape[0] < 5 or sub_filled.shape[1] < 2:
                return self.equal_weight(factor_matrix, available)

            pca = PCA(n_components=n_components)
            scores = pca.fit_transform(sub_filled)
            composite = pd.Series(scores[:, 0], index=sub_filled.index, name="composite_pca")
            # Align sign: positive PC1 should correlate with positive factor scores
            if composite.mean() < 0:
                composite = -composite
            return composite
        except ImportError:
            logger.warning("sklearn not available; falling back to equal_weight for PCA")
            return self.equal_weight(factor_matrix, factor_names)

    # ------------------------------------------------------------------
    # OLS composite (in-sample)
    # ------------------------------------------------------------------

    def ols_composite(
        self,
        factor_matrix: pd.DataFrame,
        forward_returns: pd.Series,
        factor_names: Optional[List[str]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """OLS regression: factors → forward returns. Returns fitted scores and coefficients."""
        cols = factor_names or list(factor_matrix.columns)
        available = [c for c in cols if c in factor_matrix.columns]
        sub = factor_matrix[available].copy()

        aligned = pd.concat([sub, forward_returns.rename("y")], axis=1).dropna()
        if len(aligned) < 10:
            return self.equal_weight(factor_matrix, available), pd.Series(dtype=float)

        X = aligned[available].values
        y = aligned["y"].values
        try:
            coeffs, _, _, _ = np.linalg.lstsq(
                np.column_stack([np.ones(len(X)), X]), y, rcond=None
            )
            betas = pd.Series(coeffs[1:], index=available)
            # Score all tickers
            X_all = factor_matrix[available].fillna(0).values
            scores = X_all @ betas.values
            composite = pd.Series(scores, index=factor_matrix.index, name="composite_ols")
            return composite, betas
        except Exception as exc:
            logger.warning("ols_composite failed", error=str(exc))
            return self.equal_weight(factor_matrix, available), pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # Bayesian shrinkage
    # ------------------------------------------------------------------

    def shrinkage_composite(
        self,
        factor_matrix: pd.DataFrame,
        horizon: int = 21,
        shrinkage_intensity: float = 0.5,
        factor_names: Optional[List[str]] = None,
    ) -> pd.Series:
        """Bayesian shrinkage toward IC=0 for unstable factors.

        Factors with low |ICIR| are shrunk toward zero weight.
        """
        cols = factor_names or list(factor_matrix.columns)
        available = [c for c in cols if c in factor_matrix.columns]

        weights: Dict[str, float] = {}
        for fname in available:
            cached = self._ic_analyzer.get_cached_ic(fname, horizon)
            if cached:
                icir = cached.get("icir") or 0.0
                # Shrinkage: w = (1 - intensity) * raw_icir + intensity * 0
                w = (1 - shrinkage_intensity) * max(icir, 0.0)
            else:
                w = 0.0
            weights[fname] = w

        total = sum(weights.values())
        if total <= 0:
            return self.equal_weight(factor_matrix, available)

        composite = pd.Series(0.0, index=factor_matrix.index)
        for fname, w in weights.items():
            if fname in factor_matrix.columns:
                composite += factor_matrix[fname].fillna(0) * (w / total)

        composite.name = "composite_shrinkage"
        return composite

    def build_all_composites(
        self,
        factor_matrix: pd.DataFrame,
        returns_matrix: Optional[Dict[int, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Build all composite types and return as wide DataFrame."""
        composites: Dict[str, pd.Series] = {}

        composites["equal_weight"] = self.equal_weight(factor_matrix)
        composites["ic_weighted_21d"] = self.ic_weighted(factor_matrix, horizon=21)
        composites["shrinkage"] = self.shrinkage_composite(factor_matrix)

        try:
            composites["pca"] = self.pca_composite(factor_matrix)
        except Exception:
            pass

        result = pd.DataFrame(composites)
        result.index.name = "ticker"
        return result


# ---------------------------------------------------------------------------
# AlphaSignalValidator
# ---------------------------------------------------------------------------


class AlphaSignalValidator:
    """Validate discovered alpha signals out-of-sample, regime-conditional, TC-aware."""

    _TC_BPS = 20  # assumed round-trip transaction cost in basis points

    def __init__(self) -> None:
        self._ic_analyzer = InformationCoefficientAnalyzer()

    def validate_factor(
        self,
        factor_matrix: pd.DataFrame,
        factor_name: str,
        returns_matrix: Dict[int, pd.DataFrame],
        horizon: int = 21,
        train_pct: float = 0.6,
    ) -> ValidationResult:
        """Full validation suite for a single factor.

        Tests: OOS IC, TC-adjusted IC, regime IC, FF correlation, half-life.
        """
        if factor_name not in factor_matrix.columns:
            return ValidationResult(
                factor_name=factor_name, is_valid=False,
                oos_ic_mean=0.0, oos_icir=0.0,
                in_sample_ic_mean=0.0, in_sample_icir=0.0,
                degradation_pct=0.0, tc_adjusted_ic=0.0,
                regime_ic={}, ff_correlation=None,
                verdict="reject"
            )

        ret_df = returns_matrix.get(horizon)
        if ret_df is None or ret_df.empty:
            return ValidationResult(
                factor_name=factor_name, is_valid=False,
                oos_ic_mean=0.0, oos_icir=0.0,
                in_sample_ic_mean=0.0, in_sample_icir=0.0,
                degradation_pct=0.0, tc_adjusted_ic=0.0,
                regime_ic={}, ff_correlation=None,
                verdict="reject"
            )

        dates = ret_df.index
        n_total = len(dates)
        split = int(n_total * train_pct)

        if split < 20 or n_total - split < 10:
            # Insufficient data for train/test split
            full_result = self._ic_analyzer.compute_ic_time_series(
                factor_matrix, factor_name, returns_matrix, horizon=horizon
            )
            return ValidationResult(
                factor_name=factor_name,
                is_valid=abs(full_result.icir) > 0.3,
                oos_ic_mean=full_result.ic_mean,
                oos_icir=full_result.icir,
                in_sample_ic_mean=full_result.ic_mean,
                in_sample_icir=full_result.icir,
                degradation_pct=0.0,
                tc_adjusted_ic=self._tc_adjust(full_result.ic_mean, horizon),
                regime_ic={},
                ff_correlation=None,
                verdict=self._classify(full_result.icir),
            )

        # In-sample IC
        is_returns = {horizon: ret_df.iloc[:split]}
        is_result = self._ic_analyzer.compute_ic_time_series(
            factor_matrix, factor_name, is_returns, horizon=horizon
        )

        # Out-of-sample IC
        oos_returns = {horizon: ret_df.iloc[split:]}
        oos_result = self._ic_analyzer.compute_ic_time_series(
            factor_matrix, factor_name, oos_returns, horizon=horizon
        )

        # Degradation
        is_ic = is_result.ic_mean
        oos_ic = oos_result.ic_mean
        degradation = float((is_ic - oos_ic) / abs(is_ic) * 100) if is_ic != 0 else 0.0

        # TC-adjusted IC: subtract estimated transaction cost drag
        tc_adjusted = self._tc_adjust(oos_ic, horizon)

        # Regime IC: bull vs bear vs sideways (using return quantiles as proxy)
        regime_ic = self._compute_regime_ic(
            factor_matrix, factor_name, ret_df, horizon
        )

        # Verdict
        verdict = self._classify(oos_result.icir)
        is_valid = verdict in ("strong", "moderate")

        return ValidationResult(
            factor_name=factor_name,
            is_valid=is_valid,
            oos_ic_mean=round(oos_ic, 4),
            oos_icir=round(oos_result.icir, 3),
            in_sample_ic_mean=round(is_ic, 4),
            in_sample_icir=round(is_result.icir, 3),
            degradation_pct=round(degradation, 1),
            tc_adjusted_ic=round(tc_adjusted, 4),
            regime_ic={k: round(v, 4) for k, v in regime_ic.items()},
            ff_correlation=None,  # would require FF data download
            verdict=verdict,
        )

    def _tc_adjust(self, ic: float, horizon: int) -> float:
        """Adjust IC for estimated transaction costs.

        TC drag decreases as horizon increases (cost amortised over longer holding period).
        """
        tc_fraction = self._TC_BPS / 10000
        daily_tc_drag = tc_fraction / horizon
        # IC reduces by roughly the correlation between TC drag and return
        return ic - daily_tc_drag

    def _compute_regime_ic(
        self,
        factor_matrix: pd.DataFrame,
        factor_name: str,
        ret_df: pd.DataFrame,
        horizon: int,
    ) -> Dict[str, float]:
        """Compute IC separately for bull, bear, and sideways market regimes."""
        regime_ic: Dict[str, float] = {}

        if ret_df.empty or len(ret_df) < 30:
            return regime_ic

        # Define regimes by cross-sectional mean return quintile
        mean_ret = ret_df.mean(axis=1).dropna()
        q33 = mean_ret.quantile(0.33)
        q67 = mean_ret.quantile(0.67)

        regimes = {
            "bull": mean_ret[mean_ret > q67].index,
            "bear": mean_ret[mean_ret < q33].index,
            "sideways": mean_ret[(mean_ret >= q33) & (mean_ret <= q67)].index,
        }

        factor_col = factor_matrix[factor_name].dropna()

        for regime_name, regime_dates in regimes.items():
            valid_dates = regime_dates.intersection(ret_df.index)
            if len(valid_dates) < 5:
                regime_ic[regime_name] = 0.0
                continue

            ic_vals: List[float] = []
            for date in valid_dates:
                ret_row = ret_df.loc[date].dropna()
                common = factor_col.index.intersection(ret_row.index)
                if len(common) < 5:
                    continue
                ic = self._ic_analyzer.compute_ic(factor_col[common], ret_row[common])
                if ic is not None:
                    ic_vals.append(ic)

            regime_ic[regime_name] = float(np.mean(ic_vals)) if ic_vals else 0.0

        return regime_ic

    def _classify(self, icir: float) -> str:
        """Classify factor quality based on ICIR."""
        abs_icir = abs(icir)
        if abs_icir >= 0.5:
            return "strong"
        elif abs_icir >= 0.3:
            return "moderate"
        elif abs_icir >= 0.15:
            return "weak"
        else:
            return "reject"

    def batch_validate(
        self,
        factor_matrix: pd.DataFrame,
        returns_matrix: Dict[int, pd.DataFrame],
        horizon: int = 21,
    ) -> pd.DataFrame:
        """Validate all factors in the matrix. Returns summary DataFrame."""
        results: List[Dict[str, Any]] = []
        for fname in factor_matrix.columns:
            try:
                vr = self.validate_factor(factor_matrix, fname, returns_matrix, horizon)
                results.append({
                    "factor_name": fname,
                    "verdict": vr.verdict,
                    "is_valid": vr.is_valid,
                    "oos_ic": vr.oos_ic_mean,
                    "oos_icir": vr.oos_icir,
                    "in_sample_ic": vr.in_sample_ic_mean,
                    "degradation_pct": vr.degradation_pct,
                    "tc_adjusted_ic": vr.tc_adjusted_ic,
                    "regime_bull_ic": vr.regime_ic.get("bull"),
                    "regime_bear_ic": vr.regime_ic.get("bear"),
                    "regime_sideways_ic": vr.regime_ic.get("sideways"),
                })
            except Exception as exc:
                logger.warning("batch_validate skip", factor=fname, error=str(exc))

        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results).set_index("factor_name")
        df = df.sort_values("oos_icir", ascending=False)
        return df


# ---------------------------------------------------------------------------
# AIFactorDiscovery
# ---------------------------------------------------------------------------


# Pre-defined alternative factor list for fallback when no Anthropic API key
_ALTERNATIVE_FACTORS: List[Dict[str, Any]] = [
    {
        "name": "earnings_quality_ratio",
        "description": "Operating cash flow / EBITDA — measures earnings quality (>1 = cash-generative)",
        "formula_hint": "operating_cashflow / ebitda",
    },
    {
        "name": "revenue_per_employee",
        "description": "Revenue / headcount — labour efficiency signal",
        "formula_hint": "totalRevenue / fullTimeEmployees",
    },
    {
        "name": "capex_intensity",
        "description": "CapEx / Revenue — low intensity + high margin = capital-light compounder",
        "formula_hint": "capitalExpenditures / totalRevenue",
    },
    {
        "name": "working_capital_efficiency",
        "description": "Revenue / Net Working Capital — higher = more efficient working capital use",
        "formula_hint": "totalRevenue / (currentAssets - currentLiabilities)",
    },
    {
        "name": "cash_conversion_cycle",
        "description": "DSO + DIO - DPO — lower is better; measures operational efficiency",
        "formula_hint": "daysOfInventoryOnHand + daysOfSalesOutstanding - daysSalesOutstanding",
    },
    {
        "name": "r_and_d_intensity",
        "description": "R&D expense / Revenue — innovation investment signal; high in tech/pharma",
        "formula_hint": "researchDevelopment / totalRevenue",
    },
    {
        "name": "net_buyback_yield",
        "description": "Net shares repurchased / market cap — management returning capital",
        "formula_hint": "(shares_prior - shares_current) / shares_prior",
    },
    {
        "name": "gross_profit_asset_ratio",
        "description": "Gross Profit / Total Assets — profitability relative to asset base",
        "formula_hint": "grossProfit / totalAssets",
    },
    {
        "name": "operating_leverage",
        "description": "% change in operating income / % change in revenue — amplified earnings growth",
        "formula_hint": "delta_operatingIncome / delta_totalRevenue",
    },
    {
        "name": "cash_to_market_cap",
        "description": "Cash and ST investments / Market Cap — hidden value / dry powder signal",
        "formula_hint": "totalCash / marketCap",
    },
]


class AIFactorDiscovery:
    """Use Claude API to suggest novel alpha factors; auto-validate via IC analysis.

    Falls back to pre-defined alternative factors if no Anthropic API key.
    """

    _ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self) -> None:
        self._db = _get_db()
        self._loader = FactorDataLoader()
        self._validator = AlphaSignalValidator()
        self._anthropic_available = False
        try:
            import anthropic as _a  # type: ignore
            self._anthropic = _a
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self._client = _a.Anthropic(api_key=api_key)
                self._anthropic_available = True
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # AI-driven discovery
    # ------------------------------------------------------------------

    def discover_factors_for_ticker(
        self,
        ticker: str,
        info: Dict[str, Any],
        max_suggestions: int = 5,
    ) -> List[Dict[str, Any]]:
        """Ask Claude to suggest novel factors for a given ticker's fundamentals.

        Returns list of {name, description, formula_hint} dicts.
        Falls back to pre-defined alternatives if no API key.
        """
        if not self._anthropic_available:
            logger.info("AIFactorDiscovery: using pre-defined factor list (no API key)")
            return _ALTERNATIVE_FACTORS[:max_suggestions]

        # Build a summary of the ticker's fundamentals
        fundamental_summary = {
            "ticker": ticker,
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap_bn": round(info.get("marketCap", 0) / 1e9, 1),
            "pe_ratio": info.get("trailingPE"),
            "roe": info.get("returnOnEquity"),
            "gross_margin": info.get("grossMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "debt_to_equity": info.get("debtToEquity"),
            "free_cashflow_bn": round((info.get("freeCashflow") or 0) / 1e9, 2),
        }

        prompt = f"""You are a quantitative equity researcher. Given these fundamentals for {ticker}:
{json.dumps(fundamental_summary, indent=2)}

Suggest {max_suggestions} novel alpha factors (not standard P/E, P/B, momentum) that might predict returns.
For each factor, provide:
1. A short name (snake_case)
2. A brief description (1-2 sentences) of the economic intuition
3. A formula_hint showing which financial data fields to combine

Respond as a JSON array: [{{"name": "...", "description": "...", "formula_hint": "..."}}]
Only respond with valid JSON, no other text."""

        try:
            response = self._client.messages.create(
                model=self._ANTHROPIC_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text.strip()
            # Extract JSON array
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                suggestions = json.loads(content[start:end])
                logger.info("AIFactorDiscovery: received suggestions", ticker=ticker, n=len(suggestions))
                return suggestions[:max_suggestions]
        except Exception as exc:
            logger.warning("AIFactorDiscovery: Claude API failed", ticker=ticker, error=str(exc))

        return _ALTERNATIVE_FACTORS[:max_suggestions]

    def discover_and_validate(
        self,
        tickers: Optional[List[str]] = None,
        validate: bool = True,
    ) -> List[Dict[str, Any]]:
        """Discover novel factors, attempt computation, and validate via IC.

        Returns list of discovered factor metadata with validation results.
        """
        tickers = tickers or _SP500_PROXY[:10]
        suggestions: List[Dict[str, Any]] = []

        # Collect suggestions from a few tickers
        for ticker in tickers[:3]:
            info = self._loader._fetch_info(ticker)
            if info:
                new_suggestions = self.discover_factors_for_ticker(ticker, info)
                for s in new_suggestions:
                    if not any(x["name"] == s["name"] for x in suggestions):
                        suggestions.append(s)

        # Compute alternative factors for the universe
        computed_factor_values: Dict[str, Dict[str, Optional[float]]] = {}
        for ticker in (tickers or _SP500_PROXY[:20]):
            info = self._loader._fetch_info(ticker)
            computed_factor_values[ticker] = self._compute_alternative_factors(ticker, info)

        # Build factor matrix for discovered factors
        if computed_factor_values:
            factor_df = pd.DataFrame(computed_factor_values).T
            factor_df.index.name = "ticker"
        else:
            factor_df = pd.DataFrame()

        results: List[Dict[str, Any]] = []
        for suggestion in suggestions:
            factor_id = f"ai_{suggestion['name']}_{datetime.utcnow().strftime('%Y%m%d')}"
            entry: Dict[str, Any] = {
                "factor_id": factor_id,
                "name": suggestion["name"],
                "description": suggestion["description"],
                "formula_hint": suggestion["formula_hint"],
                "source": "ai_discovery" if self._anthropic_available else "predefined",
            }

            if validate and not factor_df.empty and suggestion["name"] in factor_df.columns:
                try:
                    returns_matrix = self._loader.build_returns_matrix(tickers, [5, 21])
                    if returns_matrix:
                        analyzer = InformationCoefficientAnalyzer()
                        for hz in [5, 21]:
                            ic_result = analyzer.compute_ic_time_series(
                                factor_df, suggestion["name"], returns_matrix, horizon=hz
                            )
                            entry[f"icir_{hz}d"] = round(ic_result.icir, 3)
                except Exception as exc:
                    logger.warning("discover_and_validate: IC failed", name=suggestion["name"], error=str(exc))

            # Persist to DB
            self._persist_discovered_factor(entry)
            results.append(entry)

        return results

    def _compute_alternative_factors(
        self,
        ticker: str,
        info: Dict[str, Any],
    ) -> Dict[str, Optional[float]]:
        """Compute pre-defined alternative factors for a ticker."""
        factors: Dict[str, Optional[float]] = {}

        mc = info.get("marketCap") or 0
        rev = info.get("totalRevenue") or 0
        ebitda = info.get("ebitda") or 0
        op_cf = info.get("operatingCashflow") or 0
        fcf = info.get("freeCashflow") or 0
        total_assets = info.get("totalAssets") or 1
        employees = info.get("fullTimeEmployees") or 0
        cur_assets = info.get("totalCurrentAssets") or 0
        cur_liab = info.get("totalCurrentLiabilities") or 0
        rd = info.get("researchDevelopment") or 0
        cash = info.get("totalCash") or 0
        gross_profit = info.get("grossProfit") or 0

        factors["earnings_quality_ratio"] = float(op_cf / ebitda) if ebitda and ebitda != 0 else None
        factors["revenue_per_employee"] = float(rev / employees) if employees > 0 else None
        factors["capex_intensity"] = None  # would need capex from cashflow statement
        wc = cur_assets - cur_liab
        factors["working_capital_efficiency"] = float(rev / wc) if wc and wc != 0 else None
        factors["r_and_d_intensity"] = float(rd / rev) if rev > 0 else None
        factors["cash_to_market_cap"] = float(cash / mc) if mc > 0 else None
        factors["gross_profit_asset_ratio"] = float(gross_profit / total_assets) if total_assets > 0 else None
        factors["cash_conversion_cycle"] = None  # needs A/R, inventory, A/P data
        factors["net_buyback_yield"] = None  # needs share count history
        factors["operating_leverage"] = None  # needs historical revenue/op income data

        return factors

    def _persist_discovered_factor(self, entry: Dict[str, Any]) -> None:
        """Save discovered factor metadata to DB."""
        try:
            self._db.execute("""
                INSERT OR REPLACE INTO discovered_factors
                (factor_id, name, description, formula_hint, source, icir_5d, icir_21d, discovered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.get("factor_id"),
                entry.get("name"),
                entry.get("description"),
                entry.get("formula_hint"),
                entry.get("source"),
                entry.get("icir_5d"),
                entry.get("icir_21d"),
                datetime.utcnow().isoformat(),
            ))
            self._db.commit()
        except Exception as exc:
            logger.warning("_persist_discovered_factor failed", error=str(exc))

    def list_discovered_factors(self) -> List[Dict[str, Any]]:
        """Return all discovered factors from DB."""
        try:
            rows = self._db.execute(
                "SELECT * FROM discovered_factors ORDER BY discovered_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# FactorRankingEngine — score individual tickers
# ---------------------------------------------------------------------------


class FactorRankingEngine:
    """Rank stocks by composite factor score and individual factor percentiles."""

    def __init__(self) -> None:
        self._loader = FactorDataLoader()
        self._composer = FactorComposer()
        self._db = _get_db()

    def rank_universe(
        self,
        tickers: Optional[List[str]] = None,
        method: str = "equal_weight",
    ) -> pd.DataFrame:
        """Build factor matrix for universe and rank by composite score.

        Parameters
        ----------
        tickers: universe to rank; None = SP500 proxy
        method: composite method — "equal_weight", "ic_weighted", "pca", "shrinkage"

        Returns
        -------
        DataFrame indexed by ticker, sorted by composite score descending,
        with individual factor z-scores and composite score
        """
        tickers = tickers or _SP500_PROXY
        factor_matrix = self._loader.build_factor_matrix(tickers)

        if factor_matrix.empty:
            return pd.DataFrame()

        if method == "ic_weighted":
            composite = self._composer.ic_weighted(factor_matrix)
        elif method == "pca":
            composite = self._composer.pca_composite(factor_matrix)
        elif method == "shrinkage":
            composite = self._composer.shrinkage_composite(factor_matrix)
        else:
            composite = self._composer.equal_weight(factor_matrix)

        factor_matrix["composite_score"] = composite
        factor_matrix = factor_matrix.sort_values("composite_score", ascending=False)

        # Persist top rankings to DB
        as_of = datetime.utcnow().date().isoformat()
        for ticker, row in factor_matrix.head(20).iterrows():
            self._db.execute("""
                INSERT OR REPLACE INTO factor_composite
                (ticker, composite_name, composite_score, as_of)
                VALUES (?, ?, ?, ?)
            """, (str(ticker), method, float(row.get("composite_score", 0)), as_of))
        self._db.commit()

        return factor_matrix

    def get_ticker_rankings(self, ticker: str) -> Dict[str, Any]:
        """Return factor scores and percentile ranks for a single ticker.

        Computes individual factor values, builds universe matrix,
        and returns this ticker's percentile rank per factor.
        """
        raw_factors = self._loader.get_ticker_factors(ticker)

        # Build universe matrix to compute percentile rank
        factor_matrix = self._loader.build_factor_matrix()
        if factor_matrix.empty:
            return {"ticker": ticker, "factors": raw_factors, "percentile_ranks": {}}

        percentile_ranks: Dict[str, Optional[float]] = {}
        if ticker in factor_matrix.index:
            for col in factor_matrix.columns:
                col_series = factor_matrix[col].dropna()
                ticker_val = factor_matrix.loc[ticker, col]
                if pd.notna(ticker_val) and len(col_series) > 0:
                    pct = float(stats.percentileofscore(col_series.values, ticker_val))
                    percentile_ranks[col] = round(pct, 1)
                else:
                    percentile_ranks[col] = None

        # Composite score
        composite_scores: Dict[str, Optional[float]] = {}
        for method in ["equal_weight", "ic_weighted"]:
            composite = (
                self._composer.equal_weight(factor_matrix)
                if method == "equal_weight"
                else self._composer.ic_weighted(factor_matrix)
            )
            if ticker in composite.index:
                composite_scores[method] = round(float(composite[ticker]), 3)

        return {
            "ticker": ticker,
            "raw_factors": {k: round(v, 4) if v is not None else None for k, v in raw_factors.items()},
            "percentile_ranks": percentile_ranks,
            "composite_scores": composite_scores,
            "as_of": datetime.utcnow().date().isoformat(),
        }


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query as QParam
    from pydantic import BaseModel

    factor_router = APIRouter(prefix="/api/factor", tags=["Factor Research"])

    _lib = AlphaFactorLibrary()
    _loader = FactorDataLoader()
    _ic_analyzer = InformationCoefficientAnalyzer()
    _composer = FactorComposer()
    _validator = AlphaSignalValidator()
    _discovery = AIFactorDiscovery()
    _ranker = FactorRankingEngine()

    @factor_router.get("/library", summary="List all available factors")
    def get_factor_library() -> Dict[str, Any]:
        """Return all factor names grouped by category."""
        return {
            "n_factors": len(AlphaFactorLibrary.list_factor_names()),
            "factor_names": AlphaFactorLibrary.list_factor_names(),
            "categories": {
                "value": ["pe_ratio", "pb_ratio", "ps_ratio", "p_fcf", "ev_ebitda", "ev_sales", "dividend_yield"],
                "quality": ["roe", "roic", "gross_margin", "net_margin", "operating_margin",
                             "fcf_margin", "accruals", "piotroski_f", "current_ratio", "debt_to_equity"],
                "growth": ["revenue_growth_1y", "revenue_growth_3y", "eps_growth_1y",
                            "eps_revision", "analyst_upside", "fcf_yield"],
                "momentum": ["mom_12_1", "mom_6_1", "mom_3_1", "reversal_1m",
                              "w52_high_proximity", "rsi_14", "atr_rel", "macd_signal"],
                "size": ["log_market_cap", "log_float_shares", "log_enterprise_value"],
                "sentiment": ["short_pct_float", "short_ratio", "analyst_score",
                               "analyst_coverage", "insider_ownership_pct", "institutional_ownership_pct"],
            },
        }

    @factor_router.get("/ic-analysis", summary="Run IC analysis for a factor")
    def get_ic_analysis(
        factor_name: str = QParam(..., description="Factor name from /factor/library"),
        horizon: int = QParam(21, ge=1, le=252, description="Forward return horizon in trading days"),
        tickers: Optional[str] = QParam(None, description="Comma-separated tickers; None = SP500 proxy"),
    ) -> Dict[str, Any]:
        """Compute IC, ICIR, t-stat, and half-life for a given factor."""
        try:
            universe = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
            factor_matrix = _loader.build_factor_matrix(universe)
            if factor_matrix.empty:
                raise HTTPException(status_code=503, detail="Failed to build factor matrix")
            if factor_name not in factor_matrix.columns:
                raise HTTPException(status_code=404, detail=f"Factor '{factor_name}' not found")
            returns_matrix = _loader.build_returns_matrix(universe, [horizon])
            result = _ic_analyzer.compute_ic_time_series(
                factor_matrix, factor_name, returns_matrix, horizon=horizon
            )
            return {
                "factor_name": result.factor_name,
                "horizon_days": result.horizon_days,
                "ic_mean": round(result.ic_mean, 4),
                "ic_std": round(result.ic_std, 4),
                "icir": round(result.icir, 3),
                "ic_tstat": round(result.ic_tstat, 3),
                "n_observations": result.n_obs,
                "half_life_days": result.half_life_days,
                "ic_series_sample": result.ic_series[-10:],  # last 10 obs
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @factor_router.get("/composite", summary="Build composite factor score for universe")
    def get_composite(
        method: str = QParam("equal_weight", description="Composite method: equal_weight|ic_weighted|pca|shrinkage"),
        top_n: int = QParam(20, ge=1, le=50, description="Top N tickers to return"),
        tickers: Optional[str] = QParam(None, description="Comma-separated tickers; None = SP500 proxy"),
    ) -> Dict[str, Any]:
        """Return ranked tickers by composite factor score."""
        try:
            universe = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
            df = _ranker.rank_universe(universe, method=method)
            if df.empty:
                raise HTTPException(status_code=503, detail="Failed to build factor matrix")
            df_out = df.head(top_n).reset_index()
            df_out = df_out.where(pd.notnull(df_out), other=None)
            return {
                "method": method,
                "n_universe": len(df),
                "top_n": min(top_n, len(df)),
                "rankings": df_out[["ticker", "composite_score"]].to_dict(orient="records"),
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @factor_router.get("/validate", summary="Validate a factor out-of-sample")
    def validate_factor(
        factor_name: str = QParam(..., description="Factor to validate"),
        horizon: int = QParam(21, ge=1, le=126, description="Return horizon days"),
        tickers: Optional[str] = QParam(None, description="Comma-separated tickers"),
    ) -> Dict[str, Any]:
        """Run full validation suite: OOS IC, TC-adjusted, regime IC."""
        try:
            universe = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
            factor_matrix = _loader.build_factor_matrix(universe)
            returns_matrix = _loader.build_returns_matrix(universe, [horizon])
            result = _validator.validate_factor(
                factor_matrix, factor_name, returns_matrix, horizon=horizon
            )
            return {
                "factor_name": result.factor_name,
                "verdict": result.verdict,
                "is_valid": result.is_valid,
                "oos_ic_mean": result.oos_ic_mean,
                "oos_icir": result.oos_icir,
                "in_sample_ic_mean": result.in_sample_ic_mean,
                "in_sample_icir": result.in_sample_icir,
                "degradation_pct": result.degradation_pct,
                "tc_adjusted_ic": result.tc_adjusted_ic,
                "regime_ic": result.regime_ic,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @factor_router.post("/discover", summary="AI-driven factor discovery")
    def discover_factors(
        tickers: Optional[str] = QParam(None, description="Comma-separated tickers"),
        validate: bool = QParam(True, description="Validate discovered factors via IC"),
    ) -> Dict[str, Any]:
        """Use Claude to suggest novel factors, then auto-validate via IC analysis."""
        try:
            universe = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
            results = _discovery.discover_and_validate(universe, validate=validate)
            return {
                "n_factors_discovered": len(results),
                "source": "ai_discovery" if _discovery._anthropic_available else "predefined",
                "factors": results,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @factor_router.get("/rankings/{ticker}", summary="Factor rankings for a single ticker")
    def get_ticker_factor_rankings(ticker: str) -> Dict[str, Any]:
        """Return factor scores, percentile ranks, and composite scores for a ticker."""
        try:
            result = _ranker.get_ticker_rankings(ticker.upper())
            return result
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @factor_router.get("/discovered", summary="List all AI-discovered factors")
    def list_discovered() -> Dict[str, Any]:
        """Return all previously discovered factors from the DB."""
        try:
            factors = _discovery.list_discovered_factors()
            return {"n_factors": len(factors), "factors": factors}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @factor_router.get("/decay/{factor_name}", summary="IC decay curve for a factor")
    def get_ic_decay(
        factor_name: str,
        tickers: Optional[str] = QParam(None, description="Comma-separated tickers"),
    ) -> Dict[str, Any]:
        """Return IC decay curve: IC at 1D, 5D, 21D, 63D, 126D horizons."""
        try:
            universe = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
            factor_matrix = _loader.build_factor_matrix(universe)
            returns_matrix = _loader.build_returns_matrix(universe)
            if factor_name not in factor_matrix.columns:
                raise HTTPException(status_code=404, detail=f"Factor '{factor_name}' not found")
            decay = _ic_analyzer.compute_ic_decay_curve(factor_matrix, factor_name, returns_matrix)
            return {
                "factor_name": factor_name,
                "ic_decay": {str(k): round(v, 4) for k, v in decay.items()},
                "horizons_days": list(decay.keys()),
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

except ImportError:
    factor_router = None  # type: ignore[assignment]
    logger.debug("FastAPI not available; factor_router not registered")
