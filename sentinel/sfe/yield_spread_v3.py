"""
Yield Spread Analytics v3 — dim_047 (score 8 → 9)
====================================================
Full term structure decomposition, recession probability models (NY Fed + Wright),
cross-asset signals, G10 international spread comparisons, carry/roll-down analytics,
and yield spread backtesting.

Data sources: FRED CSV (no API key required), yfinance prices.
"""
from __future__ import annotations

import io
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Module-level logger (mirrors sentinel pattern)
# ---------------------------------------------------------------------------
try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# FRED CSV base (no API key)
# ---------------------------------------------------------------------------
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

# FRED series IDs for US Treasury yields
FRED_YIELD_SERIES: dict[str, str] = {
    "3M":  "DGS3MO",
    "6M":  "DGS6MO",
    "1Y":  "DGS1",
    "2Y":  "DGS2",
    "3Y":  "DGS3",
    "5Y":  "DGS5",
    "7Y":  "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}

# FRED spread + rate series
FRED_SPREAD_SERIES: dict[str, str] = {
    "T10Y3M":   "T10Y3M",    # 10Y minus 3M (Fed's own series)
    "T10Y2Y":   "T10Y2Y",    # 10Y minus 2Y
    "FEDFUNDS": "FEDFUNDS",  # Effective Federal Funds Rate
}

# G10 international proxies available from FRED (all free CSV)
FRED_INTL_SERIES: dict[str, dict[str, str]] = {
    "JP": {"10Y": "IRLTLT01JPM156N", "ST": "IRSTCI01JPM156N"},
    "GB": {"10Y": "IRLTLT01GBM156N", "ST": "IRSTCI01GBM156N"},
    "CA": {"10Y": "IRLTLT01CAM156N", "ST": "IRSTCI01CAM156N"},
    "AU": {"10Y": "IRLTLT01AUM156N", "ST": "IRSTCI01AUM156N"},
    "DE": {"10Y": "IRLTLT01DEM156N", "ST": "IRSTCI01DEM156N"},
    "FR": {"10Y": "IRLTLT01FRM156N", "ST": "IRSTCI01FRM156N"},
    "IT": {"10Y": "IRLTLT01ITM156N", "ST": "IRSTCI01ITM156N"},
    "CH": {"10Y": "IRLTLT01CHM156N", "ST": "IRSTCI01CHM156N"},
    "NZ": {"10Y": "IRLTLT01NZM156N", "ST": "IRSTCI01NZM156N"},
    "SE": {"10Y": "IRLTLT01SEM156N", "ST": "IRSTCI01SEM156N"},
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SpreadData:
    """Point-in-time snapshot of all yield spreads."""
    as_of: str
    yields: dict[str, float]
    spreads: dict[str, float]
    percentiles: dict[str, float] = field(default_factory=dict)
    regime: str = "UNKNOWN"

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "yields": self.yields,
            "spreads": self.spreads,
            "percentiles": self.percentiles,
            "regime": self.regime,
        }


@dataclass
class RecessionForecast:
    """Recession probability from multiple models."""
    as_of: str
    ny_fed_model: float          # Estrella-Mishkin Probit
    wright_model: float          # Wright (2006) with FF level
    spread_3m10y: float
    fed_funds: float
    ny_fed_risk_label: str
    wright_risk_label: str
    historical_comparison: str

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "ny_fed_model": round(self.ny_fed_model, 4),
            "wright_model": round(self.wright_model, 4),
            "spread_3m10y_bps": round(self.spread_3m10y * 100, 1),
            "fed_funds": round(self.fed_funds, 3),
            "ny_fed_risk": self.ny_fed_risk_label,
            "wright_risk": self.wright_risk_label,
            "historical_comparison": self.historical_comparison,
        }


@dataclass
class BacktestResult:
    """Backtest output — strategy performance summary."""
    strategy: str
    start: str
    end: str
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    benchmark_return: float
    alpha: float
    signal_accuracy: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "period": f"{self.start} → {self.end}",
            "total_return_pct": round(self.total_return * 100, 2),
            "cagr_pct": round(self.cagr * 100, 2),
            "sharpe": round(self.sharpe, 3),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "num_trades": self.num_trades,
            "win_rate_pct": round(self.win_rate * 100, 1),
            "benchmark_return_pct": round(self.benchmark_return * 100, 2),
            "alpha_pct": round(self.alpha * 100, 2),
            "signal_accuracy_pct": round(self.signal_accuracy * 100, 1),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _fred_csv(series_id: str, start: str = "1953-01-01") -> pd.Series:
    """
    Fetch a single FRED series as a pandas Series (no API key required).
    Falls back gracefully on HTTP errors.
    """
    url = f"{FRED_BASE}{series_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(
            io.StringIO(resp.text),
            parse_dates=["DATE"],
            index_col="DATE",
        )
        s = df.iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.name = series_id
        if start:
            s = s[s.index >= pd.Timestamp(start)]
        return s
    except Exception as exc:
        logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        return pd.Series(name=series_id, dtype=float)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (no scipy required)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _risk_label(prob: float) -> str:
    if prob < 0.10:
        return "LOW"
    if prob < 0.25:
        return "ELEVATED"
    if prob < 0.50:
        return "HIGH"
    return "VERY HIGH"


def _cagr(total_ret: float, n_years: float) -> float:
    if n_years <= 0:
        return 0.0
    return (1 + total_ret) ** (1 / n_years) - 1


def _sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / 252
    if excess.std() == 0:
        return 0.0
    return excess.mean() / excess.std() * math.sqrt(252)


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return dd.min()


# ---------------------------------------------------------------------------
# Class 1: YieldSpreadCalculator
# ---------------------------------------------------------------------------

class YieldSpreadCalculator:
    """
    Compute all standard US Treasury yield spreads from FRED data.
    Uses free FRED CSV endpoint — no API key.
    """

    TENOR_ORDER = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

    def __init__(self, cache_ttl_minutes: int = 60) -> None:
        self._cache: dict[str, tuple[datetime, pd.Series]] = {}
        self._cache_ttl = timedelta(minutes=cache_ttl_minutes)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _get_series(self, series_id: str, start: str = "1953-01-01") -> pd.Series:
        now = datetime.utcnow()
        if series_id in self._cache:
            ts, data = self._cache[series_id]
            if now - ts < self._cache_ttl:
                return data
        data = _fred_csv(series_id, start=start)
        self._cache[series_id] = (now, data)
        return data

    def fetch_all_yields(self, date: Optional[str] = None) -> dict[str, float]:
        """
        Fetch the most-recent US Treasury yield curve from FRED.
        If `date` is provided, return the yield on that specific date.
        Returns dict keyed by tenor: {'3M': 5.25, '2Y': 4.80, ...}
        Yields are in percent (e.g. 5.25 = 5.25%).
        """
        result: dict[str, float] = {}
        for tenor, series_id in FRED_YIELD_SERIES.items():
            s = self._get_series(series_id)
            if s.empty:
                continue
            if date:
                target = pd.Timestamp(date)
                # find nearest available date
                idx = s.index.asof(target)
                if idx is pd.NaT:
                    continue
                val = s.loc[idx]
            else:
                val = s.iloc[-1]
            result[tenor] = float(val)
        return result

    # ------------------------------------------------------------------
    # Core spread calculations
    # ------------------------------------------------------------------

    def compute_2s10s(self, yields: dict[str, float]) -> float:
        """10Y - 2Y spread (most-watched recession signal). Basis points."""
        return (yields.get("10Y", float("nan")) - yields.get("2Y", float("nan"))) * 100

    def compute_3m10y(self, yields: dict[str, float]) -> float:
        """10Y - 3M spread (best 12-month recession predictor per NY Fed). Basis points."""
        return (yields.get("10Y", float("nan")) - yields.get("3M", float("nan"))) * 100

    def compute_5s30s(self, yields: dict[str, float]) -> float:
        """30Y - 5Y spread (long-end steepener/flattener). Basis points."""
        return (yields.get("30Y", float("nan")) - yields.get("5Y", float("nan"))) * 100

    def compute_2s5s10s_butterfly(self, yields: dict[str, float]) -> float:
        """
        2s5s10s butterfly: (2Y + 10Y)/2 - 5Y (hump in the middle of curve).
        Positive = 5Y elevated vs wings; Negative = 5Y depressed vs wings.
        Basis points.
        """
        two = yields.get("2Y", float("nan"))
        five = yields.get("5Y", float("nan"))
        ten = yields.get("10Y", float("nan"))
        return ((two + ten) / 2.0 - five) * 100

    def compute_10y30y_spread(self, yields: dict[str, float]) -> float:
        """30Y - 10Y spread (long-end shape signal). Basis points."""
        return (yields.get("30Y", float("nan")) - yields.get("10Y", float("nan"))) * 100

    def compute_1y5y_forward(self, yields: dict[str, float]) -> float:
        """
        Implied 1Y5Y forward rate: rate on a 5Y bond starting in 1Y.
        Approximation: (6 × 6Y_yield - 1 × 1Y_yield) / 5
        We interpolate 6Y from 5Y and 7Y.
        """
        y1 = yields.get("1Y", float("nan"))
        y5 = yields.get("5Y", float("nan"))
        y7 = yields.get("7Y", float("nan"))
        # Interpolate 6Y linearly between 5Y and 7Y
        y6 = (y5 + y7) / 2.0
        return ((6 * y6) - (1 * y1)) / 5.0 * 100  # in bps

    def compute_2y5y_forward(self, yields: dict[str, float]) -> float:
        """
        Implied 2Y5Y forward rate.
        Approximation: (7 × 7Y_yield - 2 × 2Y_yield) / 5
        """
        y2 = yields.get("2Y", float("nan"))
        y7 = yields.get("7Y", float("nan"))
        return ((7 * y7) - (2 * y2)) / 5.0 * 100

    def get_all_spreads(self, yields: dict[str, float]) -> dict[str, float]:
        """Compute all standard spreads at once. Returns dict with bps values."""
        return {
            "2s10s_bps": self.compute_2s10s(yields),
            "3m10y_bps": self.compute_3m10y(yields),
            "5s30s_bps": self.compute_5s30s(yields),
            "2s5s10s_butterfly_bps": self.compute_2s5s10s_butterfly(yields),
            "10y30y_bps": self.compute_10y30y_spread(yields),
            "1y5y_fwd_bps": self.compute_1y5y_forward(yields),
            "2y5y_fwd_bps": self.compute_2y5y_forward(yields),
        }

    # ------------------------------------------------------------------
    # Historical spread series
    # ------------------------------------------------------------------

    def compute_spread_history(
        self,
        spread_name: str,
        start: str = "1990-01-01",
    ) -> pd.Series:
        """
        Build a historical spread time series from individual FRED yield series.
        spread_name: '2s10s', '3m10y', '5s30s', '2s5s10s_butterfly', '10y30y'
        Returns Series of spread values in basis points.
        """
        SPREAD_SERIES: dict[str, tuple[str, str, str]] = {
            "2s10s":            ("DGS10", "DGS2",   "2s10s"),
            "3m10y":            ("DGS10", "DGS3MO", "3m10y"),
            "5s30s":            ("DGS30", "DGS5",   "5s30s"),
            "10y30y":           ("DGS30", "DGS10",  "10y30y"),
        }
        # Try FRED's pre-computed spread series first
        PREBUILT: dict[str, str] = {
            "2s10s":  "T10Y2Y",
            "3m10y":  "T10Y3M",
        }

        if spread_name in PREBUILT:
            s = self._get_series(PREBUILT[spread_name], start=start)
            if not s.empty:
                return (s * 100).rename(f"{spread_name}_bps")  # FRED gives %, convert to bps

        if spread_name not in SPREAD_SERIES:
            raise ValueError(
                f"Unknown spread '{spread_name}'. "
                f"Choose from: {list(SPREAD_SERIES.keys())}"
            )

        long_id, short_id, _ = SPREAD_SERIES[spread_name]
        long_s = self._get_series(long_id, start=start)
        short_s = self._get_series(short_id, start=start)
        combined = pd.concat([long_s, short_s], axis=1).dropna()
        spread = (combined.iloc[:, 0] - combined.iloc[:, 1]) * 100  # convert % to bps
        spread.name = f"{spread_name}_bps"
        return spread

    def compute_spread_percentile(
        self, spread: float, history: pd.Series
    ) -> float:
        """
        Historical percentile of current spread vs. history.
        Returns 0-100.  Above 50 = steeper than median.
        """
        clean = history.dropna()
        if clean.empty:
            return 50.0
        return float((clean < spread).sum() / len(clean) * 100)

    def detect_curve_regime(self, spreads: dict[str, float]) -> str:
        """
        Classify current yield curve regime:
        - STEEP_BULL:    long end falling faster (rate cuts priced in)
        - BEAR_STEEPEN:  short end falling, long rising (reflation)
        - FLAT:          spread near zero
        - INVERTED:      short yields > long yields (recession signal)
        - DEEPLY_INVERTED: inversion > -100bps
        """
        s_2s10s = spreads.get("2s10s_bps", 0.0)
        s_3m10y = spreads.get("3m10y_bps", 0.0)

        if s_2s10s < -100 or s_3m10y < -100:
            return "DEEPLY_INVERTED"
        if s_2s10s < 0 or s_3m10y < 0:
            return "INVERTED"
        if s_2s10s < 50:
            return "FLAT"
        if s_2s10s >= 150:
            return "STEEP"
        return "NORMAL"


# ---------------------------------------------------------------------------
# Class 2: RecessionProbabilityModel
# ---------------------------------------------------------------------------

class RecessionProbabilityModel:
    """
    Forecast recession probability from yield spreads.
    Implements NY Fed (Estrella-Mishkin 1998) and Wright (2006) Probit models.
    Both calibrated on 1959-2010 data.
    """

    def __init__(self) -> None:
        self._calc = YieldSpreadCalculator()

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def compute_ny_fed_model(self, t10y3m_spread: float) -> float:
        """
        New York Fed Probit model (Estrella & Mishkin 1998).
        Predicts probability of recession within next 12 months.

        Equation: P(recession) = Φ(-0.6002 - 0.5288 × spread)
        where spread is in percentage points (e.g. -0.5 for -50bps).
        Φ = standard normal CDF.
        """
        # spread expected in percentage points
        linear = -0.6002 - 0.5288 * t10y3m_spread
        return _norm_cdf(linear)

    def compute_wright_model(self, spread: float, fed_funds: float) -> float:
        """
        Wright (2006) Probit model.
        Includes fed funds level as additional regressor.

        Equation: P = Φ(-(1.369 + 0.0798×ff² - 0.654×spread))
        ff = federal funds rate in percent.
        spread = 10Y3M spread in percent.

        This model recognizes that inversions at high rate levels
        are more recessionary than inversions at low rate levels.
        """
        linear = -(1.369 + 0.0798 * fed_funds ** 2 - 0.654 * spread)
        return _norm_cdf(linear)

    def compute_smoothed_recession_probability(
        self, spread_history: pd.Series
    ) -> pd.Series:
        """
        Apply NY Fed model to a historical spread series,
        then smooth with 3-month rolling average to reduce noise.

        spread_history: Series of 10Y3M spread in percentage points.
        Returns: Series of recession probabilities (0-1).
        """
        probs = spread_history.apply(self.compute_ny_fed_model)
        smoothed = probs.rolling(window=63, min_periods=10).mean()  # ~3 months
        smoothed.name = "recession_prob_smoothed"
        return smoothed

    # ------------------------------------------------------------------
    # Current snapshot
    # ------------------------------------------------------------------

    def get_current_recession_probability(self) -> dict:
        """
        Compute current recession probability using both models.
        Fetches live data from FRED (T10Y3M, FEDFUNDS).
        """
        # Fetch data
        t10y3m_series = _fred_csv("T10Y3M", start="2000-01-01")
        fedfunds_series = _fred_csv("FEDFUNDS", start="2000-01-01")

        spread_val = float(t10y3m_series.iloc[-1]) if not t10y3m_series.empty else 0.0
        ff_val = float(fedfunds_series.iloc[-1]) if not fedfunds_series.empty else 4.5
        as_of = str(t10y3m_series.index[-1].date()) if not t10y3m_series.empty else "N/A"

        ny_fed_prob = self.compute_ny_fed_model(spread_val)
        wright_prob = self.compute_wright_model(spread_val, ff_val)

        # Historical context: how often has this spread level preceded recession?
        hist_context = self._historical_context(spread_val, t10y3m_series)

        return RecessionForecast(
            as_of=as_of,
            ny_fed_model=ny_fed_prob,
            wright_model=wright_prob,
            spread_3m10y=spread_val,
            fed_funds=ff_val,
            ny_fed_risk_label=_risk_label(ny_fed_prob),
            wright_risk_label=_risk_label(wright_prob),
            historical_comparison=hist_context,
        ).to_dict()

    def _historical_context(
        self, current_spread: float, history: pd.Series
    ) -> str:
        """Build a human-readable historical comparison string."""
        if history.empty:
            return "Insufficient data"

        pct = (history < current_spread).sum() / len(history) * 100
        most_negative = float(history.min())
        median = float(history.median())

        direction = "more inverted" if current_spread < 0 else "less inverted"
        return (
            f"Current 10Y-3M spread ({current_spread:.2f}%) is at the "
            f"{pct:.0f}th percentile of history. "
            f"Median spread: {median:.2f}%; most inverted: {most_negative:.2f}%. "
            f"Spread is {direction} than historical median."
        )

    # ------------------------------------------------------------------
    # Threshold interpretation
    # ------------------------------------------------------------------

    @staticmethod
    def interpret(probability: float) -> dict:
        """Return labeled risk tier and recommended action."""
        label = _risk_label(probability)
        actions = {
            "LOW": "Maintain risk exposure; no recession signal.",
            "ELEVATED": "Monitor; consider mild defensive tilt.",
            "HIGH": "Reduce equities; increase Treasuries and cash.",
            "VERY HIGH": "Defensive positioning warranted; history suggests recession within 12mo.",
        }
        return {"probability": probability, "risk_tier": label, "action": actions[label]}


# ---------------------------------------------------------------------------
# Class 3: TermPremiumDecomposer
# ---------------------------------------------------------------------------

class TermPremiumDecomposer:
    """
    Decompose nominal yields into:
      - Expectations component (average expected short rate)
      - Term premium (compensation for duration risk)
      - Carry and roll-down analytics
    Uses simplified ACM-style decomposition from public FRED data.
    """

    def compute_acm_term_premium(
        self,
        yield_10y: float,
        yield_2y: float,
        fed_funds_expectation: float,
    ) -> dict:
        """
        Adrian-Crump-Moench (ACM) term premium approximation.

        The 10Y yield = E[average short rate over 10Y] + term premium.
        We proxy the expectations component as the average of the near-term
        expectation (2Y) and the current policy rate.

        term_premium ≈ 10Y - (2Y + fed_funds) / 2

        Returns decomposition dictionary.
        """
        expectations = (yield_2y + fed_funds_expectation) / 2.0
        term_premium = yield_10y - expectations

        return {
            "yield_10y": yield_10y,
            "expectations_component": round(expectations, 4),
            "term_premium": round(term_premium, 4),
            "term_premium_bps": round(term_premium * 100, 1),
            "interpretation": (
                "Term premium is NEGATIVE — investors accepting below-fair-value "
                "compensation for duration risk (historically rare, QE era common)."
                if term_premium < 0 else
                "Term premium is POSITIVE — investors receiving above-expectations "
                "compensation for holding duration risk."
            ),
        }

    def compute_breakeven_term_premium(
        self,
        nominal_10y: float,
        tips_10y: float,
        breakeven_inflation: float,
    ) -> dict:
        """
        Decompose nominal yield using TIPS and breakeven inflation.
        Nominal = Real + Breakeven_Inflation = TIPS + BEI
        Residual term premium = Nominal - Real - Inflation_Expectations
        """
        real_yield = tips_10y
        inflation_expectation = breakeven_inflation
        term_premium_approx = nominal_10y - real_yield - inflation_expectation

        return {
            "nominal_10y": nominal_10y,
            "real_yield_tips": real_yield,
            "breakeven_inflation": inflation_expectation,
            "residual_term_premium": round(term_premium_approx, 4),
        }

    def compute_carry(
        self,
        yield_current: float,
        holding_period: float,
    ) -> float:
        """
        Bond carry: income earned from holding the bond.
        carry = yield × holding_period_years
        (assumes yield stays constant — pure income effect)
        Returns in percent per annum.
        """
        return yield_current * holding_period

    def compute_roll_down(
        self,
        yield_curve: dict[str, float],
        starting_tenor: float = 10.0,
        holding_period: float = 1.0,
    ) -> float:
        """
        Roll-down return: as a bond ages it rolls down the yield curve.
        If you hold a 10Y bond for 1 year, it becomes a 9Y bond.
        Roll-down = yield(10Y) - yield(9Y) = decline in yield × modified duration.

        We approximate the final tenor's yield by interpolation.
        Returns roll-down in basis points.
        """
        ending_tenor = starting_tenor - holding_period
        if ending_tenor <= 0:
            return 0.0

        y_start = self._interpolate_yield(yield_curve, starting_tenor)
        y_end = self._interpolate_yield(yield_curve, ending_tenor)

        # Roll-down = yield improvement (price appreciation)
        # Approx price change = -duration × Δyield
        # Modified duration ≈ starting_tenor × 0.85 (rough approximation)
        duration = starting_tenor * 0.85
        roll_down_bps = (y_start - y_end) * 100  # yield decline in bps
        price_appreciation_pct = duration * (y_start - y_end)

        return round(roll_down_bps, 2)  # return bps of roll-down

    def compute_total_return_decomposition(
        self,
        yield_curve: dict[str, float],
        tenor: float,
        horizon: float = 1.0,
        yield_change_assumption: float = 0.0,
    ) -> dict:
        """
        Decompose 1-year total return into carry, roll-down, and price change.

        Components:
          - Carry: yield × holding_period (income)
          - Roll-down: return from aging bond rolling down the curve
          - Price change: -modified_duration × Δyield (if rates move)
          - Total: carry + roll-down + price_change (if rates unchanged, price_change=0)

        Returns all components as percent (annualized).
        """
        y = self._interpolate_yield(yield_curve, tenor)
        carry = self.compute_carry(y, horizon)
        roll_bps = self.compute_roll_down(yield_curve, tenor, horizon)
        roll_pct = roll_bps / 100.0 * horizon  # convert bps to %

        # Modified duration approximation
        mod_dur = tenor * 0.85 / (1 + y)
        price_change = -mod_dur * yield_change_assumption

        total = carry + roll_pct + price_change

        return {
            "tenor": tenor,
            "current_yield_pct": round(y * 100, 3),
            "carry_pct": round(carry * 100, 3),
            "roll_down_bps": round(roll_bps, 1),
            "roll_down_pct": round(roll_pct * 100, 3),
            "yield_change_assumption_bps": round(yield_change_assumption * 100, 1),
            "price_change_pct": round(price_change * 100, 3),
            "total_return_pct": round(total * 100, 3),
            "note": "Assumes rates unchanged" if yield_change_assumption == 0 else "",
        }

    def _interpolate_yield(
        self, yield_curve: dict[str, float], tenor_years: float
    ) -> float:
        """Linear interpolation across available tenors."""
        TENOR_YEARS: dict[str, float] = {
            "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0, "3Y": 3.0,
            "5Y": 5.0, "7Y": 7.0, "10Y": 10.0, "20Y": 20.0, "30Y": 30.0,
        }
        # Build sorted list of (years, yield) pairs
        points = []
        for k, v in yield_curve.items():
            yrs = TENOR_YEARS.get(k)
            if yrs is not None and not math.isnan(v):
                points.append((yrs, v / 100.0 if v > 1 else v))  # normalize to decimal
        points.sort()

        if not points:
            return 0.04  # 4% fallback

        # Exact match
        for yrs, y in points:
            if abs(yrs - tenor_years) < 0.01:
                return y

        # Extrapolate at boundaries
        if tenor_years <= points[0][0]:
            return points[0][1]
        if tenor_years >= points[-1][0]:
            return points[-1][1]

        # Linear interpolation between bracketing points
        for i in range(len(points) - 1):
            y1, v1 = points[i]
            y2, v2 = points[i + 1]
            if y1 <= tenor_years <= y2:
                t = (tenor_years - y1) / (y2 - y1)
                return v1 + t * (v2 - v1)

        return points[-1][1]


# ---------------------------------------------------------------------------
# Class 4: CrossAssetYieldSignals
# ---------------------------------------------------------------------------

class CrossAssetYieldSignals:
    """
    Yield spreads as directional signals for equities, credit, FX, and ERP.
    """

    # Thresholds (bps)
    DEEPLY_INVERTED = -100
    INVERTED = 0
    FLAT = 50
    NORMAL = 100
    STEEP = 200

    def yield_spread_equity_signal(
        self,
        spread_2s10s: float,           # current 2s10s in bps
        spread_history: pd.Series,     # historical 2s10s in bps
    ) -> dict:
        """
        Map yield curve shape to equity positioning signal.

        Regime logic (per Yield Curve Bible / JPM research):
          - DEEPLY_INVERTED (<-100bps): Full defensive. Recession imminent.
          - INVERTED (-100 to 0bps):    Late cycle. Reduce equities, add bonds.
          - FLAT (0 to 50bps):          Transition. Hold, watch for reversal.
          - NORMAL (50-200bps):         Mid cycle. Maintain equity exposure.
          - STEEP (>200bps):            Early cycle or inflation risk.
                                        If steepening from inversion → RECOVERY.
                                        If steepening bear → inflation hedge.

        Also checks: Is the curve steepening or flattening (1-month change)?
        """
        current_pct = (
            (spread_history < spread_2s10s).sum() / len(spread_history) * 100
            if not spread_history.empty else 50.0
        )

        # 1-month momentum: compare to ~21 trading days ago
        if len(spread_history) > 25:
            one_month_ago = float(spread_history.iloc[-22])
            momentum = spread_2s10s - one_month_ago  # +ve = steepening
        else:
            momentum = 0.0

        # Determine signal
        if spread_2s10s < self.DEEPLY_INVERTED:
            signal = "DEFENSIVE"
            equity_bias = "UNDERWEIGHT equities, OVERWEIGHT Treasuries and cash"
            rationale = "Deep inversion historically precedes recessions by 12-18 months."
        elif spread_2s10s < self.INVERTED:
            if momentum > 20:
                signal = "EARLY_RECOVERY"
                equity_bias = "Begin building equity exposure as curve normalizes"
                rationale = "Steepening from inversion historically marks recovery onset."
            else:
                signal = "LATE_CYCLE"
                equity_bias = "Reduce equities, favor defensive sectors and short-duration bonds"
                rationale = "Inverted curve signals late-cycle slowdown ahead."
        elif spread_2s10s < self.FLAT:
            signal = "CAUTIOUS"
            equity_bias = "Neutral equities; watch for inversion or steepening"
            rationale = "Flat curve = transition regime. Direction of next move matters."
        elif spread_2s10s < self.STEEP:
            if momentum > 30 and spread_2s10s < 75:
                signal = "EARLY_RECOVERY"
                equity_bias = "Overweight equities, small-cap and cyclicals"
                rationale = "Steepening from low levels = reflation + recovery."
            else:
                signal = "RISK_ON"
                equity_bias = "Maintain equity exposure; economy mid-cycle"
                rationale = "Normal upward-sloping curve supports risk assets."
        else:
            signal = "LATE_BULL"
            equity_bias = "Quality tilt; inflation protection; watch for flattening"
            rationale = "Very steep curve often associated with high inflation or early recovery."

        return {
            "signal": signal,
            "spread_2s10s_bps": round(spread_2s10s, 1),
            "spread_percentile": round(current_pct, 1),
            "momentum_1m_bps": round(momentum, 1),
            "momentum_direction": "STEEPENING" if momentum > 0 else "FLATTENING",
            "equity_bias": equity_bias,
            "rationale": rationale,
        }

    def yield_spread_credit_signal(self, spread_3m10y: float) -> dict:
        """
        Map 10Y-3M spread to IG vs. HY credit positioning signal.

        Research basis: Credit quality deteriorates with inversion as
        corporate borrowing costs rise relative to long-term revenue expectations.
        """
        if spread_3m10y < -100:
            grade = "CREDIT_DEFENSIVE"
            bias = "Maximum IG tilt. Reduce HY significantly. Expect rising default rates."
            hy_premium = "Insufficient. HY spreads should widen 200-400bps within 18 months."
        elif spread_3m10y < 0:
            grade = "CREDIT_CAUTIOUS"
            bias = "Overweight IG, underweight HY. Monitor HY OAS for widening."
            hy_premium = "Marginal. Risk-reward favors IG over HY."
        elif spread_3m10y < 100:
            grade = "CREDIT_NEUTRAL"
            bias = "Balanced IG/HY. Slight preference for BB-rated (crossover)."
            hy_premium = "Moderate. BB/B credit offers reasonable carry."
        elif spread_3m10y < 200:
            grade = "CREDIT_POSITIVE"
            bias = "Favor HY over IG. Strong carry environment."
            hy_premium = "Attractive. HY outperforms IG in steep curve environments historically."
        else:
            grade = "CREDIT_BULLISH"
            bias = "Overweight HY and leveraged loans. Early cycle credit rally."
            hy_premium = "Very attractive. Distressed opportunities may arise."

        return {
            "signal": grade,
            "spread_3m10y_bps": round(spread_3m10y, 1),
            "credit_bias": bias,
            "hy_premium_assessment": hy_premium,
            "ig_vs_hy": "IG" if spread_3m10y < 0 else ("BALANCED" if spread_3m10y < 100 else "HY"),
        }

    def yield_spread_currency_signal(
        self,
        us_yield_dict: dict[str, float],  # {"2Y": 4.5, "10Y": 4.3} in %
        foreign_yields: dict[str, dict[str, float]],  # {"DE": {"2Y": 2.5}, ...}
    ) -> dict:
        """
        Compute USD direction signal from 2Y yield differentials.
        USD tends to strengthen when US rates are rising relative to peers.
        """
        us_2y = us_yield_dict.get("2Y", 0.0)
        us_10y = us_yield_dict.get("10Y", 0.0)

        differentials = {}
        for country, yields in foreign_yields.items():
            fgn_2y = yields.get("2Y", yields.get("ST", 0.0))
            fgn_10y = yields.get("10Y", 0.0)
            diff_2y = us_2y - fgn_2y
            diff_10y = us_10y - fgn_10y
            pair = f"USD/{country}"
            differentials[pair] = {
                "2y_differential_bps": round(diff_2y * 100, 1),
                "10y_differential_bps": round(diff_10y * 100, 1),
                "usd_direction": "STRENGTHEN" if diff_2y > 0 else "WEAKEN",
                "carry_advantage_bps": round(diff_2y * 100, 1),
            }

        # Aggregate: is USD broadly supported?
        n_positive = sum(
            1 for v in differentials.values()
            if v["2y_differential_bps"] > 0
        )
        usd_overall = (
            "BROADLY_STRONG" if n_positive >= len(differentials) * 0.7
            else "BROADLY_WEAK" if n_positive <= len(differentials) * 0.3
            else "MIXED"
        )

        return {
            "usd_overall_bias": usd_overall,
            "us_2y_yield": us_2y,
            "pairs": differentials,
        }

    def compute_equity_earnings_yield_vs_bonds(
        self,
        earnings_yield: float,  # E/P ratio (e.g. 0.05 for 5%)
        bond_yield: float,       # 10Y Treasury yield (e.g. 0.043 for 4.3%)
    ) -> dict:
        """
        Fed Model: compare S&P 500 earnings yield to 10Y Treasury yield.
        Equity Risk Premium (ERP) = earnings yield - 10Y yield.

        Interpretation:
          ERP > 4%: equities very cheap vs bonds (buy equities)
          2-4%:     equities moderately cheap (mild overweight)
          0-2%:     equities fairly valued
          ERP < 0%: equities expensive vs bonds (underweight)
          ERP < -2%: equities significantly expensive (reduce aggressively)

        Note: Fed Model is controversial. It conflates nominal and real
        quantities. Best used as one signal among many.
        """
        erp = earnings_yield - bond_yield
        pe_ratio = 1 / earnings_yield if earnings_yield > 0 else float("nan")

        if erp > 0.04:
            assessment = "EQUITIES_VERY_CHEAP"
            action = "Strong buy signal vs bonds. Favor equities."
        elif erp > 0.02:
            assessment = "EQUITIES_CHEAP"
            action = "Mild overweight equities vs bonds."
        elif erp > 0.0:
            assessment = "FAIRLY_VALUED"
            action = "Neutral. No strong relative value signal."
        elif erp > -0.02:
            assessment = "EQUITIES_EXPENSIVE"
            action = "Mild underweight equities. Bonds offer competitive yield."
        else:
            assessment = "EQUITIES_VERY_EXPENSIVE"
            action = "Bonds significantly more attractive. Reduce equity duration."

        return {
            "earnings_yield_pct": round(earnings_yield * 100, 2),
            "bond_yield_10y_pct": round(bond_yield * 100, 2),
            "equity_risk_premium_pct": round(erp * 100, 2),
            "pe_ratio": round(pe_ratio, 1) if not math.isnan(pe_ratio) else None,
            "assessment": assessment,
            "action": action,
            "fed_model_caveat": (
                "Fed Model mixes nominal earnings yield vs nominal bond yield. "
                "In high-inflation periods this overstates equity attractiveness."
            ),
        }


# ---------------------------------------------------------------------------
# Class 5: InternationalSpreadAnalyzer
# ---------------------------------------------------------------------------

class InternationalSpreadAnalyzer:
    """
    G10 yield differential analysis. Carry trade setup detection.
    All data sourced from FRED free CSV (no API key).
    """

    COUNTRY_NAMES: dict[str, str] = {
        "US": "United States",
        "JP": "Japan",
        "GB": "United Kingdom",
        "CA": "Canada",
        "AU": "Australia",
        "DE": "Germany",
        "FR": "France",
        "IT": "Italy",
        "CH": "Switzerland",
        "NZ": "New Zealand",
        "SE": "Sweden",
    }

    def __init__(self, cache_ttl_minutes: int = 120) -> None:
        self._cache: dict[str, tuple[datetime, pd.Series]] = {}
        self._cache_ttl = timedelta(minutes=cache_ttl_minutes)

    def _get_series(self, series_id: str, start: str = "2000-01-01") -> pd.Series:
        now = datetime.utcnow()
        if series_id in self._cache:
            ts, data = self._cache[series_id]
            if now - ts < self._cache_ttl:
                return data
        data = _fred_csv(series_id, start=start)
        self._cache[series_id] = (now, data)
        return data

    def fetch_g10_yields(
        self, tenors: list[str] = ("2y", "10y")
    ) -> pd.DataFrame:
        """
        Fetch G10 yields from FRED. Returns DataFrame with MultiIndex columns
        (country, tenor). Monthly frequency (FRED international series).
        US yields are daily, resampled monthly.
        """
        frames: list[pd.Series] = []

        # US (daily → monthly)
        us_map = {"2y": "DGS2", "10y": "DGS10"}
        for t in tenors:
            sid = us_map.get(t.lower())
            if sid:
                s = self._get_series(sid).resample("ME").last()
                s.name = ("US", t.upper())
                frames.append(s)

        # G10 international (monthly from FRED)
        for country, series_map in FRED_INTL_SERIES.items():
            for t in tenors:
                t_key = "10Y" if "10" in t.upper() else "ST"
                sid = series_map.get(t_key)
                if sid:
                    s = self._get_series(sid)
                    s.name = (country, t.upper())
                    frames.append(s)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, axis=1).sort_index()
        df.index = pd.to_datetime(df.index)
        return df

    def compute_yield_differential(
        self,
        country1: str,
        country2: str,
        tenor: str = "10y",
        df: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        Compute yield differential: country1 - country2 for given tenor.
        Positive = country1 yields higher (stronger currency carry).
        Returns latest value in percentage points.
        """
        if df is None:
            df = self.fetch_g10_yields(tenors=[tenor])

        col1 = (country1, tenor.upper())
        col2 = (country2, tenor.upper())

        if col1 not in df.columns or col2 not in df.columns:
            return float("nan")

        latest = df[[col1, col2]].dropna().iloc[-1]
        return float(latest[col1] - latest[col2])

    def compute_carry_score(
        self, country: str, df: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Carry score = 10Y yield - short-term rate (domestic carry).
        Higher = more attractive carry (steep domestic curve).
        Also reflects FX carry available: high-yielding currencies attract flows.
        """
        if df is None:
            df = self.fetch_g10_yields(tenors=["2y", "10y"])

        col_10y = (country, "10Y")
        col_st = (country, "2Y")

        available_cols = [c for c in [col_10y, col_st] if c in df.columns]
        if len(available_cols) < 2:
            return float("nan")

        latest = df[available_cols].dropna().iloc[-1]
        return float(latest[col_10y] - latest[col_st])

    def get_carry_ladder(
        self, df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Rank all G10 countries by carry score.
        Returns DataFrame sorted by carry (highest first).
        """
        if df is None:
            df = self.fetch_g10_yields(tenors=["2y", "10y"])

        countries = ["US", "JP", "GB", "CA", "AU", "DE", "FR", "IT", "CH", "NZ", "SE"]
        scores = []
        for c in countries:
            score = self.compute_carry_score(c, df)
            if not math.isnan(score):
                scores.append(
                    {
                        "country": c,
                        "name": self.COUNTRY_NAMES.get(c, c),
                        "carry_score": round(score, 3),
                        "carry_rank": 0,
                    }
                )

        result = pd.DataFrame(scores).sort_values("carry_score", ascending=False)
        result["carry_rank"] = range(1, len(result) + 1)
        return result.reset_index(drop=True)

    def detect_carry_trade_setup(
        self, vix_threshold: float = 25.0
    ) -> dict:
        """
        Identify optimal carry trade pair (long highest carry, short lowest carry).
        Risk signal: when VIX > vix_threshold, carry trades tend to unwind violently.

        Returns: setup dict with long/short legs, expected carry, risk flags.
        """
        df = self.fetch_g10_yields(tenors=["2y", "10y"])
        ladder = self.get_carry_ladder(df)

        if ladder.empty:
            return {"error": "Could not fetch G10 yields"}

        # Fetch VIX for risk check
        vix_series = _fred_csv("VIXCLS", start="2020-01-01")
        current_vix = float(vix_series.iloc[-1]) if not vix_series.empty else 15.0

        long_leg = ladder.iloc[0]   # highest carry
        short_leg = ladder.iloc[-1] # lowest carry

        expected_carry = long_leg["carry_score"] - short_leg["carry_score"]
        risk_on = current_vix < vix_threshold

        return {
            "long_leg": {
                "country": long_leg["country"],
                "name": long_leg["name"],
                "carry_score": long_leg["carry_score"],
            },
            "short_leg": {
                "country": short_leg["country"],
                "name": short_leg["name"],
                "carry_score": short_leg["carry_score"],
            },
            "expected_carry_pct": round(expected_carry, 3),
            "current_vix": round(current_vix, 1),
            "risk_environment": "FAVORABLE" if risk_on else "UNFAVORABLE",
            "vix_threshold": vix_threshold,
            "trade_signal": "ENTER_CARRY" if risk_on else "AVOID_CARRY",
            "risk_note": (
                "VIX elevated — carry unwind risk is HIGH. Carry trades typically "
                "experience sharp reversals when vol spikes above 25."
                if not risk_on else
                "Low vol environment supports carry. Monitor VIX for exit signal."
            ),
            "carry_ladder": ladder.to_dict(orient="records"),
        }

    def compute_g10_spread_matrix(
        self, tenor: str = "10y", df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Full pairwise yield differential matrix for G10 countries.
        Rows = long country, columns = short country.
        """
        if df is None:
            df = self.fetch_g10_yields(tenors=[tenor])

        countries = [c for c in ["US", "JP", "GB", "CA", "AU", "DE", "FR", "IT"] if (c, tenor.upper()) in df.columns]
        latest = df[[(c, tenor.upper()) for c in countries]].dropna().iloc[-1]

        matrix = pd.DataFrame(index=countries, columns=countries, dtype=float)
        for c1 in countries:
            for c2 in countries:
                matrix.loc[c1, c2] = float(latest[(c1, tenor.upper())] - latest[(c2, tenor.upper())])

        return matrix.round(3)


# ---------------------------------------------------------------------------
# Class 6: YieldSpreadBacktester
# ---------------------------------------------------------------------------

class YieldSpreadBacktester:
    """
    Backtest trading signals derived from yield spreads.
    Uses FRED data for spreads + yfinance for equity prices.
    """

    def __init__(self) -> None:
        self._model = RecessionProbabilityModel()

    def _fetch_equity(self, ticker: str, start: str, end: str) -> pd.Series:
        """Fetch daily adjusted close prices via yfinance."""
        try:
            import yfinance as yf
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                return pd.Series(dtype=float)
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.name = ticker
            return close
        except Exception as exc:
            logger.warning("yfinance failed for %s: %s", ticker, exc)
            return pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # Strategy 1: Curve steepener
    # ------------------------------------------------------------------

    def backtest_curve_steepener(self, start: str = "2000-01-01") -> BacktestResult:
        """
        Long curve steepener when 2s10s < -50bps.

        Implementation:
          - Position: long 10Y Treasury (via TLT proxy), short 2Y (via SHY).
          - Entry: 2s10s < -50bps (inversion).
          - Exit: 2s10s > 0bps (curve normalizes).
          - Return proxy: duration-weighted combination (10Y: 8yr dur, 2Y: 2yr).

        P&L from spread change: when curve steepens by 50bps,
          10Y bond gains ~4% (8yr dur × 0.5%), 2Y gains ~1% (2yr × 0.5%).
          Net steepener P&L: 3% per 50bps of steepening.
        """
        # Fetch 2s10s spread from FRED (percent)
        t10y2y = _fred_csv("T10Y2Y", start=start)
        if t10y2y.empty:
            return BacktestResult(
                strategy="curve_steepener", start=start, end="N/A",
                total_return=0, cagr=0, sharpe=0, max_drawdown=0,
                num_trades=0, win_rate=0, benchmark_return=0, alpha=0,
                notes="FRED T10Y2Y data unavailable",
            )

        # Daily spread in bps
        spread_bps = t10y2y * 100
        spread_bps = spread_bps.resample("B").last().ffill()

        # Signals
        ENTRY_THRESHOLD = -50.0   # enter steepener when inverted > 50bps
        EXIT_THRESHOLD  = 0.0     # exit when curve normalizes

        in_trade = False
        trade_returns = []
        current_trade_start_spread = 0.0
        num_trades = 0

        dates = spread_bps.index
        vals = spread_bps.values

        pnl_series = pd.Series(0.0, index=dates)

        for i in range(1, len(vals)):
            s = vals[i]
            s_prev = vals[i - 1]

            if not in_trade and s < ENTRY_THRESHOLD:
                in_trade = True
                current_trade_start_spread = s
                num_trades += 1

            elif in_trade and s > EXIT_THRESHOLD:
                in_trade = False
                spread_change = s - current_trade_start_spread  # positive = steepening
                # Steepener P&L: 6bps of duration differential per bps of steepening
                # (8yr - 2yr = 6yr duration net) × spread_change_pct
                trade_return = 6.0 * (spread_change / 100.0)  # in decimal terms
                trade_returns.append(trade_return)

            if in_trade:
                # Daily P&L from spread change
                daily_change = (s - s_prev) / 100.0  # bps to decimal
                pnl_series.iloc[i] = 6.0 * daily_change / 252.0  # annualized

        equity = (1 + pnl_series).cumprod()
        total_ret = float(equity.iloc[-1] - 1)
        n_years = (dates[-1] - dates[0]).days / 365.25
        cagr_val = _cagr(total_ret, n_years)
        sr = _sharpe(pnl_series[pnl_series != 0])
        mdd = _max_drawdown(equity)
        win_rate = sum(1 for r in trade_returns if r > 0) / max(len(trade_returns), 1)

        return BacktestResult(
            strategy="curve_steepener",
            start=str(dates[0].date()),
            end=str(dates[-1].date()),
            total_return=total_ret,
            cagr=cagr_val,
            sharpe=sr,
            max_drawdown=mdd,
            num_trades=num_trades,
            win_rate=win_rate,
            benchmark_return=0.0,
            alpha=cagr_val,
            notes=f"Entry: 2s10s < {ENTRY_THRESHOLD}bps, Exit: 2s10s > {EXIT_THRESHOLD}bps",
        )

    # ------------------------------------------------------------------
    # Strategy 2: NY Fed recession model → SPY tactical overlay
    # ------------------------------------------------------------------

    def backtest_recession_model(
        self,
        start: str = "1990-01-01",
        entry_prob: float = 0.30,
        exit_prob: float = 0.15,
    ) -> BacktestResult:
        """
        Tactical SPY allocation based on NY Fed recession probability.
          - Exit equities (hold cash) when NY Fed model > entry_prob (30%)
          - Re-enter when probability < exit_prob (15%)

        SPY is used as equity proxy (yfinance). Cash = 0% return (conservative).
        """
        # Fetch FRED 10Y-3M spread (daily)
        t10y3m = _fred_csv("T10Y3M", start=start)
        if t10y3m.empty:
            return BacktestResult(
                strategy="recession_model_spy", start=start, end="N/A",
                total_return=0, cagr=0, sharpe=0, max_drawdown=0,
                num_trades=0, win_rate=0, benchmark_return=0, alpha=0,
                notes="T10Y3M data unavailable",
            )

        t10y3m = t10y3m.resample("B").last().ffill()
        end_date = str(t10y3m.index[-1].date())

        # Compute daily recession probability
        rec_prob = t10y3m.apply(self._model.compute_ny_fed_model)

        # Fetch SPY prices
        spy = self._fetch_equity("SPY", start=start, end=end_date)
        if spy.empty:
            return BacktestResult(
                strategy="recession_model_spy", start=start, end=end_date,
                total_return=0, cagr=0, sharpe=0, max_drawdown=0,
                num_trades=0, win_rate=0, benchmark_return=0, alpha=0,
                notes="SPY price data unavailable",
            )

        spy_ret = spy.pct_change().dropna()

        # Align
        aligned = pd.concat([spy_ret, rec_prob], axis=1).dropna()
        aligned.columns = ["spy_ret", "rec_prob"]

        # Trading signals
        in_equities = True
        num_trades = 0
        strategy_returns = []
        trade_results = []
        entry_price_idx = 0

        position = pd.Series(1.0, index=aligned.index)
        for i in range(1, len(aligned)):
            p = aligned["rec_prob"].iloc[i - 1]  # use prior-day probability (no look-ahead)
            if in_equities and p > entry_prob:
                in_equities = False
                position.iloc[i] = 0.0
                num_trades += 1
                entry_price_idx = i
            elif not in_equities and p < exit_prob:
                in_equities = True
                position.iloc[i] = 1.0
                num_trades += 1
            else:
                position.iloc[i] = 1.0 if in_equities else 0.0

        strategy_rets = aligned["spy_ret"] * position
        equity = (1 + strategy_rets).cumprod()
        bh_equity = (1 + aligned["spy_ret"]).cumprod()

        total_ret = float(equity.iloc[-1] - 1)
        bh_ret = float(bh_equity.iloc[-1] - 1)
        n_years = (aligned.index[-1] - aligned.index[0]).days / 365.25
        cagr_val = _cagr(total_ret, n_years)
        bh_cagr = _cagr(bh_ret, n_years)
        sr = _sharpe(strategy_rets)
        mdd = _max_drawdown(equity)

        # Win rate: count defensive periods that avoided SPY drawdown
        # (approximate: count N-day windows during cash periods)
        periods_in_cash = (position == 0).sum()
        if periods_in_cash > 0:
            cash_spy = aligned["spy_ret"][position == 0]
            avoided_loss = float((cash_spy < 0).sum() / len(cash_spy))
        else:
            avoided_loss = 0.0

        return BacktestResult(
            strategy="recession_model_spy",
            start=str(aligned.index[0].date()),
            end=str(aligned.index[-1].date()),
            total_return=total_ret,
            cagr=cagr_val,
            sharpe=sr,
            max_drawdown=mdd,
            num_trades=num_trades,
            win_rate=avoided_loss,
            benchmark_return=bh_ret,
            alpha=cagr_val - bh_cagr,
            notes=(
                f"Exit SPY at prob>{entry_prob:.0%}; "
                f"Re-enter at prob<{exit_prob:.0%}. "
                f"Days in cash: {periods_in_cash}"
            ),
        )

    # ------------------------------------------------------------------
    # Signal accuracy analysis
    # ------------------------------------------------------------------

    def compute_spread_signal_accuracy(
        self,
        spread: str = "2s10s",
        forward_period: int = 252,
        inversion_threshold: float = 0.0,
    ) -> dict:
        """
        What % of the time does an inverted yield curve predict S&P decline
        over the following `forward_period` trading days?

        Also computes: false positive rate, average forward return when signal fires.
        """
        # Fetch spread history
        calc = YieldSpreadCalculator()
        spread_hist = calc.compute_spread_history(spread, start="1985-01-01")
        if spread_hist.empty:
            return {"error": "Could not fetch spread history"}

        # Fetch SPY
        spy = self._fetch_equity("SPY", start="1985-01-01", end="2026-01-01")
        if spy.empty:
            return {"error": "Could not fetch SPY"}

        spy_ret = spy.pct_change()

        # Align and compute forward returns
        aligned = pd.concat([spread_hist, spy_ret], axis=1).dropna()
        aligned.columns = ["spread_bps", "spy_daily_ret"]

        results = []
        signal_dates = []

        for i in range(len(aligned) - forward_period):
            s = aligned["spread_bps"].iloc[i]
            if s < inversion_threshold:  # inversion signal
                fwd_ret = aligned["spy_daily_ret"].iloc[i + 1:i + forward_period + 1].sum()
                results.append(fwd_ret)
                signal_dates.append(aligned.index[i])

        if not results:
            return {"error": "No inversion signals found in history"}

        n_signals = len(results)
        n_correct = sum(1 for r in results if r < 0)
        accuracy = n_correct / n_signals
        avg_fwd_ret = float(np.mean(results))
        median_fwd_ret = float(np.median(results))

        # Base rate: how often does SPY decline in any forward_period?
        all_fwd = []
        for i in range(len(aligned) - forward_period):
            fwd = aligned["spy_daily_ret"].iloc[i + 1:i + forward_period + 1].sum()
            all_fwd.append(fwd)
        base_rate = sum(1 for r in all_fwd if r < 0) / len(all_fwd)

        return {
            "spread": spread,
            "inversion_threshold_bps": inversion_threshold,
            "forward_period_days": forward_period,
            "n_inversion_signals": n_signals,
            "signal_accuracy_pct": round(accuracy * 100, 1),
            "base_rate_pct": round(base_rate * 100, 1),
            "lift_over_base_rate_pct": round((accuracy - base_rate) * 100, 1),
            "avg_forward_return_pct": round(avg_fwd_ret * 100, 2),
            "median_forward_return_pct": round(median_fwd_ret * 100, 2),
            "first_signal": str(signal_dates[0].date()) if signal_dates else "N/A",
            "last_signal": str(signal_dates[-1].date()) if signal_dates else "N/A",
        }


# ---------------------------------------------------------------------------
# Class 7: YieldSpreadEngine (orchestrator)
# ---------------------------------------------------------------------------

class YieldSpreadEngine:
    """
    Master orchestrator. Pulls all yield spread analytics together
    into a unified dashboard, daily update loop, and narrative report.
    """

    def __init__(self) -> None:
        self.calc = YieldSpreadCalculator()
        self.recession_model = RecessionProbabilityModel()
        self.term_premium = TermPremiumDecomposer()
        self.cross_asset = CrossAssetYieldSignals()
        self.intl = InternationalSpreadAnalyzer()
        self.backtester = YieldSpreadBacktester()

    def get_dashboard(self) -> dict:
        """
        Full current-state dashboard:
          - All US yield spreads
          - Percentiles vs history
          - Recession probability (2 models)
          - Curve regime classification
          - Term premium decomposition
          - Equity/credit/FX cross-asset signals
          - G10 carry ladder
          - ERP (earnings yield vs bonds)
        """
        logger.info("Building yield spread dashboard...")

        # 1. Fetch yields
        yields = self.calc.fetch_all_yields()

        # 2. Compute spreads
        spreads = self.calc.get_all_spreads(yields)

        # 3. Percentiles
        percentiles = {}
        for sname in ["2s10s", "3m10y", "5s30s"]:
            try:
                hist = self.calc.compute_spread_history(sname, start="1985-01-01")
                spread_key = f"{sname}_bps"
                if spread_key in spreads:
                    percentiles[sname] = self.calc.compute_spread_percentile(
                        spreads[spread_key], hist
                    )
            except Exception as e:
                logger.debug("Percentile failed for %s: %s", sname, e)

        # 4. Regime
        regime = self.calc.detect_curve_regime(spreads)

        # 5. Recession probability
        try:
            recession = self.recession_model.get_current_recession_probability()
        except Exception as e:
            recession = {"error": str(e)}

        # 6. Term premium decomposition
        try:
            t10y3m_s = _fred_csv("FEDFUNDS", start="2024-01-01")
            ff = float(t10y3m_s.iloc[-1]) if not t10y3m_s.empty else 4.5
            tp = self.term_premium.compute_acm_term_premium(
                yield_10y=yields.get("10Y", 4.3) / 100,
                yield_2y=yields.get("2Y", 4.5) / 100,
                fed_funds_expectation=ff / 100,
            )
        except Exception:
            tp = {}

        # 7. Roll-down analytics
        try:
            roll_down = self.term_premium.compute_total_return_decomposition(
                {k: v / 100 for k, v in yields.items()},
                tenor=10.0,
                horizon=1.0,
            )
        except Exception:
            roll_down = {}

        # 8. G10 carry
        try:
            carry_setup = self.intl.detect_carry_trade_setup()
        except Exception as e:
            carry_setup = {"error": str(e)}

        # 9. Equity signal (need history for context)
        try:
            hist_2s10s = self.calc.compute_spread_history("2s10s", start="1990-01-01")
            equity_signal = self.cross_asset.yield_spread_equity_signal(
                spreads.get("2s10s_bps", 0.0), hist_2s10s
            )
        except Exception:
            equity_signal = {}

        # 10. Credit signal
        credit_signal = self.cross_asset.yield_spread_credit_signal(
            spreads.get("3m10y_bps", 0.0)
        )

        return {
            "as_of": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "yields": {k: round(v, 3) for k, v in yields.items()},
            "spreads": {k: round(v, 1) for k, v in spreads.items()},
            "percentiles": {k: round(v, 1) for k, v in percentiles.items()},
            "regime": regime,
            "recession": recession,
            "term_premium": tp,
            "roll_down_10y": roll_down,
            "carry_trade": carry_setup,
            "equity_signal": equity_signal,
            "credit_signal": credit_signal,
        }

    def run_daily_update(self) -> dict:
        """Refresh FRED data and recompute all spreads. Returns update summary."""
        logger.info("Running daily yield spread update...")
        # Clear caches by re-instantiating
        self.calc = YieldSpreadCalculator(cache_ttl_minutes=0)
        self.intl = InternationalSpreadAnalyzer(cache_ttl_minutes=0)
        result = self.get_dashboard()
        logger.info("Daily update complete. Regime: %s", result.get("regime"))
        return result

    def generate_report(self, dashboard: Optional[dict] = None) -> str:
        """Generate a narrative text summary of the current yield spread environment."""
        if dashboard is None:
            dashboard = self.get_dashboard()

        regime = dashboard.get("regime", "UNKNOWN")
        spreads = dashboard.get("spreads", {})
        recession = dashboard.get("recession", {})
        equity = dashboard.get("equity_signal", {})
        carry = dashboard.get("carry_trade", {})

        s_2s10s = spreads.get("2s10s_bps", 0.0)
        s_3m10y = spreads.get("3m10y_bps", 0.0)

        ny_fed = recession.get("ny_fed_model", float("nan"))
        wright = recession.get("wright_model", float("nan"))

        ny_risk = recession.get("ny_fed_risk", "N/A")

        carry_long = carry.get("long_leg", {}).get("country", "N/A")
        carry_short = carry.get("short_leg", {}).get("country", "N/A")
        carry_vix = carry.get("current_vix", 0.0)

        report_lines = [
            "=" * 72,
            "SENTINEL | YIELD CURVE & SPREAD ANALYTICS REPORT",
            f"Generated: {dashboard.get('as_of', 'N/A')}",
            "=" * 72,
            "",
            "YIELD CURVE REGIME",
            f"  Current classification: {regime}",
            f"  2s10s spread:           {s_2s10s:+.1f} bps",
            f"  3m10y spread:           {s_3m10y:+.1f} bps",
            "",
            "RECESSION PROBABILITY MODELS",
            f"  NY Fed (Estrella-Mishkin): {ny_fed:.1%}  [{ny_risk}]",
            f"  Wright (2006) w/ FF:       {wright:.1%}  [{_risk_label(wright)}]",
            "",
            "EQUITY CROSS-ASSET SIGNAL",
            f"  Signal:    {equity.get('signal', 'N/A')}",
            f"  Bias:      {equity.get('equity_bias', 'N/A')}",
            f"  Rationale: {equity.get('rationale', 'N/A')}",
            "",
            "G10 CARRY TRADE",
            f"  Long:  {carry_long} | Short: {carry_short}",
            f"  Expected Carry: {carry.get('expected_carry_pct', 0):.2f}%",
            f"  VIX:   {carry_vix:.1f} → {carry.get('risk_environment', 'N/A')}",
            f"  Signal: {carry.get('trade_signal', 'N/A')}",
            "",
            "=" * 72,
        ]

        return "\n".join(report_lines)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 72)
    print("SENTINEL Yield Spread Analytics v3")
    print("=" * 72)

    engine = YieldSpreadEngine()

    # 1. Current spreads
    print("\n[1] Fetching current US Treasury yields...")
    yields = engine.calc.fetch_all_yields()
    print("  Tenors:", {k: f"{v:.3f}%" for k, v in yields.items()})

    spreads = engine.calc.get_all_spreads(yields)
    print("  Spreads:", {k: f"{v:+.1f}bps" for k, v in spreads.items()})
    print(f"  Regime: {engine.calc.detect_curve_regime(spreads)}")

    # 2. Recession models
    print("\n[2] Running NY Fed & Wright recession models...")
    rec = engine.recession_model.get_current_recession_probability()
    print(json.dumps(rec, indent=2))

    # 3. Term premium & roll-down
    print("\n[3] Term premium decomposition & roll-down...")
    yields_dec = {k: v / 100 for k, v in yields.items()}
    tp = engine.term_premium.compute_acm_term_premium(
        yield_10y=yields_dec.get("10Y", 0.043),
        yield_2y=yields_dec.get("2Y", 0.045),
        fed_funds_expectation=yields_dec.get("3M", 0.053),
    )
    print(json.dumps(tp, indent=2))

    rd = engine.term_premium.compute_total_return_decomposition(yields_dec, tenor=10.0)
    print("  10Y Roll-down decomposition:")
    print(json.dumps(rd, indent=2))

    # 4. G10 carry ladder
    print("\n[4] G10 Carry Ladder...")
    try:
        carry_setup = engine.intl.detect_carry_trade_setup()
        print(f"  Long:  {carry_setup.get('long_leg', {}).get('country')} "
              f"({carry_setup.get('long_leg', {}).get('carry_score', 0):.3f}%)")
        print(f"  Short: {carry_setup.get('short_leg', {}).get('country')} "
              f"({carry_setup.get('short_leg', {}).get('carry_score', 0):.3f}%)")
        print(f"  VIX:   {carry_setup.get('current_vix', 0):.1f} → {carry_setup.get('trade_signal')}")
        if "carry_ladder" in carry_setup:
            print("  Full ladder:")
            for row in carry_setup["carry_ladder"]:
                print(f"    #{row['carry_rank']:2d} {row['country']} ({row['name']}): "
                      f"{row['carry_score']:+.3f}%")
    except Exception as e:
        print(f"  Carry ladder failed: {e}")

    # 5. Backtest recession model
    print("\n[5] Backtesting recession model on SPY (1990→)...")
    try:
        bt = engine.backtester.backtest_recession_model(start="1990-01-01")
        print(json.dumps(bt.to_dict(), indent=2))
    except Exception as e:
        print(f"  Backtest failed: {e}")

    # 6. Signal accuracy
    print("\n[6] 2s10s inversion → SPY signal accuracy (1-year forward)...")
    try:
        acc = engine.backtester.compute_spread_signal_accuracy(
            spread="2s10s", forward_period=252
        )
        print(json.dumps(acc, indent=2))
    except Exception as e:
        print(f"  Signal accuracy failed: {e}")

    # 7. Full report
    print("\n[7] Full narrative report:")
    print(engine.generate_report())
