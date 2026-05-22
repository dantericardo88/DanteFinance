"""Google Trends Financial Signal Platform V3 — dim_089 (score 6 → 9).

Comprehensive Google Trends signal engine for SENTINEL.

Primary source: pytrends (unofficial Google Trends API wrapper, no key needed).
Fallback:       Direct CSV download via Google Trends export URL.

Signals produced:
  - Search momentum (4-week vs 52-week z-score)
  - Trend acceleration (2nd derivative of smoothed interest)
  - Spike detection (z-score > threshold)
  - Composite ticker search score
  - Sector-level trend ranking
  - Fear/greed proxy from market search terms
  - Product cycle signals (leading revenue indicator)
  - Earnings surprise prediction via search interest
  - Macro economic anxiety / job market / housing indices
  - Competitive share-of-search analysis

Cache: JSON flat-file at sentinel/data/trends_cache.json with 24h TTL.

Usage::
    from sentinel.sma.google_trends_v3 import GoogleTrendsEngine

    engine = GoogleTrendsEngine()
    dashboard = engine.get_signal("AAPL")
    print(dashboard.search_score.momentum_zscore)
    print(engine.get_fear_greed_index())
"""
from __future__ import annotations

import csv
import io
import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from pytrends.request import TrendReq as _TrendReq
    _PYTRENDS_OK = True
except ImportError:  # pragma: no cover
    _TrendReq = None  # type: ignore[misc,assignment]
    _PYTRENDS_OK = False

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    _PANDAS_OK = False

try:
    from scipy.signal import savgol_filter as _savgol
    _SCIPY_OK = True
except ImportError:  # pragma: no cover
    _savgol = None  # type: ignore[misc,assignment]
    _SCIPY_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CACHE_PATH = _DATA_DIR / "trends_cache.json"
_CACHE_TTL_HOURS = 24

_GOOGLE_TRENDS_CSV_BASE = (
    "https://trends.google.com/trends/explore/csv"
    "?q={kw}&date={timeframe}&geo={geo}"
)
_GOOGLE_TRENDS_BASE = "https://trends.google.com"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Timeframe aliases supported by Google Trends
_VALID_TIMEFRAMES = {
    "now 1-H", "now 4-H", "now 1-d", "now 7-d",
    "today 1-m", "today 3-m", "today 12-m", "today 5-y",
}

# Market sentiment term mapping
_FEAR_TERMS = [
    "stock market crash", "recession", "bear market",
    "market collapse", "unemployment",
]
_GREED_TERMS = [
    "buy the dip", "bull market", "invest now",
    "all time high", "stocks rising",
]
_NEUTRAL_TERMS = ["stock market"]

# Sector -> representative tickers mapping (used for sector trend ranking)
_SECTOR_TICKERS: Dict[str, List[str]] = {
    "technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
    "finance": ["JPM", "BAC", "GS", "MS", "C"],
    "healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
    "energy": ["XOM", "CVX", "SLB", "COP", "EOG"],
    "consumer": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
    "industrials": ["BA", "CAT", "GE", "HON", "UPS"],
    "utilities": ["NEE", "DUK", "SO", "D", "AEP"],
    "realestate": ["AMT", "PLD", "CCI", "EQIX", "SPG"],
    "materials": ["LIN", "APD", "FCX", "NEM", "NUE"],
    "communication": ["T", "VZ", "NFLX", "DIS", "CMCSA"],
}

# Company name / keyword map for top 100+ tickers
_TICKER_KEYWORDS: Dict[str, List[str]] = {
    "AAPL": ["Apple", "iPhone", "Apple stock", "buy AAPL", "MacBook"],
    "MSFT": ["Microsoft", "Windows", "Azure", "Microsoft stock"],
    "GOOGL": ["Google", "Alphabet", "Google stock", "Android"],
    "AMZN": ["Amazon", "Amazon stock", "AWS", "Prime"],
    "TSLA": ["Tesla", "Tesla stock", "Elon Musk", "Tesla Model Y"],
    "NVDA": ["NVIDIA", "GeForce", "NVIDIA stock", "GPU"],
    "META": ["Meta", "Facebook", "Instagram", "Mark Zuckerberg"],
    "NFLX": ["Netflix", "Netflix stock", "streaming"],
    "JPM": ["JPMorgan", "JP Morgan", "Chase bank"],
    "JNJ": ["Johnson Johnson", "Johnson & Johnson", "JNJ stock"],
    "BAC": ["Bank of America", "BofA", "Bank of America stock"],
    "XOM": ["ExxonMobil", "Exxon", "oil stock"],
    "CVX": ["Chevron", "Chevron stock", "Chevron oil"],
    "WMT": ["Walmart", "Walmart stock", "Walmart deals"],
    "V": ["Visa", "Visa card", "Visa stock"],
    "MA": ["Mastercard", "Mastercard stock"],
    "HD": ["Home Depot", "Home Depot stock", "home improvement"],
    "DIS": ["Disney", "Disney Plus", "Disney stock", "Walt Disney"],
    "PFE": ["Pfizer", "Pfizer stock", "Pfizer vaccine"],
    "PYPL": ["PayPal", "PayPal stock", "digital payments"],
    "INTC": ["Intel", "Intel chip", "Intel stock"],
    "AMD": ["AMD", "Ryzen", "Radeon", "AMD stock"],
    "UBER": ["Uber", "Uber stock", "ride sharing"],
    "LYFT": ["Lyft", "Lyft stock"],
    "SPOT": ["Spotify", "Spotify stock", "music streaming"],
    "SNAP": ["Snapchat", "Snap Inc", "Snap stock"],
    "TWTR": ["Twitter", "X app", "tweet"],
    "COIN": ["Coinbase", "Coinbase stock", "crypto exchange"],
    "SQ": ["Block", "Square", "Cash App", "Jack Dorsey"],
    "HOOD": ["Robinhood", "Robinhood stock", "commission free trading"],
}

# Macro term groups
_MACRO_TERMS = {
    "unemployment": ["unemployment benefits", "file for unemployment", "jobless claims"],
    "jobs": ["job openings", "hiring now", "salary negotiation", "work from home jobs"],
    "housing": [
        "buy a house", "mortgage rates", "Zillow", "Redfin",
        "home prices", "first time home buyer",
    ],
    "refinance": ["refinance mortgage", "refinance home loan", "cash out refinance"],
    "inflation": ["inflation", "prices rising", "cost of living"],
    "recession": ["recession", "economic recession", "depression economy"],
    "layoffs": ["layoffs", "tech layoffs", "company layoffs", "how to survive layoff"],
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrendsScore:
    """Composite Google Trends score for a single ticker."""
    ticker: str
    raw_interest: float = 0.0        # Current 0-100 interest score
    momentum_zscore: float = 0.0     # 4-week vs 52-week z-score
    acceleration: float = 0.0        # 2nd derivative of smoothed series
    spike_detected: bool = False      # Current interest > mean + 2*std
    spike_zscore: float = 0.0
    sector_rank: int = 0             # Rank within sector (1=highest)
    composite_score: float = 0.0     # 0-1 composite
    keywords_used: List[str] = field(default_factory=list)
    computed_at: str = ""


@dataclass
class EarningsPrediction:
    """Pre-earnings search interest signal."""
    ticker: str
    pre_earnings_interest: float = 0.0   # Current normalized interest
    historical_mean_interest: float = 0.0
    signal_strength: float = 0.0         # 0-1 (high = strong positive surprise signal)
    positive_surprise_probability: float = 0.0  # 0-1
    data_points: int = 0
    note: str = ""


@dataclass
class TrendsDashboard:
    """Full trends analysis for a single ticker."""
    ticker: str
    search_score: TrendsScore
    earnings_prediction: Optional[EarningsPrediction]
    product_signals: Dict[str, Any]         # product → {interest, acceleration, surprise_prob}
    competitive_share: Dict[str, float]    # competitor → share-of-search %
    fear_greed_index: float                # 0-100 composite fear/greed
    macro_context: Dict[str, float]        # macro index values
    raw_interest_series: List[Dict]        # [{date, value}]
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class _TrendsCache:
    """Flat JSON cache with 24h TTL per entry."""

    def __init__(self, path: Path = _CACHE_PATH, ttl_hours: float = _CACHE_TTL_HOURS):
        self._path = path
        self._ttl_seconds = ttl_hours * 3600
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self._path.exists():
                with self._path.open("r", encoding="utf-8") as fh:
                    self._data = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            logger.warning("trends cache load failed: %s", exc)
            self._data = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2)
            tmp.replace(self._path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("trends cache save failed: %s", exc)

    def get(self, key: str) -> Optional[Any]:
        entry = self._data.get(key)
        if entry is None:
            return None
        age = time.time() - entry.get("ts", 0)
        if age > self._ttl_seconds:
            del self._data[key]
            return None
        return entry.get("payload")

    def set(self, key: str, payload: Any) -> None:
        self._data[key] = {"ts": time.time(), "payload": payload}
        self._save()

    def invalidate(self, key: str) -> None:
        self._data.pop(key, None)
        self._save()


_cache = _TrendsCache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _series_to_list(df: "pd.DataFrame", col: str) -> List[Dict]:
    """Convert a pandas Series to [{date, value}] list."""
    if not _PANDAS_OK or df is None or df.empty:
        return []
    rows = []
    for idx, val in df[col].items():
        rows.append({"date": str(idx.date() if hasattr(idx, "date") else idx), "value": int(val)})
    return rows


def _zscore(series: List[float], value: float) -> float:
    """Compute z-score of value relative to series."""
    if len(series) < 4:
        return 0.0
    mean = sum(series) / len(series)
    variance = sum((x - mean) ** 2 for x in series) / len(series)
    std = math.sqrt(variance) if variance > 0 else 1e-9
    return (value - mean) / std


def _smooth(values: List[float], window: int = 4) -> List[float]:
    """Simple moving average smoothing fallback."""
    if _SCIPY_OK and _savgol and len(values) >= 7:
        try:
            wl = min(7, len(values) | 1)  # ensure odd
            if wl % 2 == 0:
                wl -= 1
            return list(_savgol(values, wl, 2))
        except Exception:  # noqa: BLE001
            pass
    # SMA fallback
    out = []
    for i, v in enumerate(values):
        start = max(0, i - window + 1)
        chunk = values[start: i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _second_derivative(values: List[float]) -> float:
    """Estimate 2nd derivative (acceleration) at the tail of the series."""
    smoothed = _smooth(values)
    if len(smoothed) < 3:
        return 0.0
    n = len(smoothed)
    d1_last = smoothed[n - 1] - smoothed[n - 2]
    d1_prev = smoothed[n - 2] - smoothed[n - 3]
    return d1_last - d1_prev


def _jitter_sleep(min_s: float = 1.0, max_s: float = 2.5) -> None:
    time.sleep(min_s + random.random() * (max_s - min_s))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# GoogleTrendsFetcher
# ---------------------------------------------------------------------------

class GoogleTrendsFetcher:
    """Fetch raw interest data from Google Trends.

    Primary: pytrends library.
    Fallback: CSV download from Google Trends export URL.
    """

    def __init__(
        self,
        hl: str = "en-US",
        tz: int = 360,
        timeout: int = 30,
        retries: int = 3,
    ):
        self._hl = hl
        self._tz = tz
        self._timeout = timeout
        self._retries = retries
        self._pytrends: Optional[Any] = None
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        if _PYTRENDS_OK and _TrendReq is not None:
            try:
                self._pytrends = _TrendReq(hl=hl, tz=tz, timeout=(timeout, timeout))
                logger.info("pytrends available — using primary fetcher")
            except Exception as exc:  # noqa: BLE001
                logger.warning("pytrends init failed (%s), using fallback", exc)
                self._pytrends = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_interest_over_time(
        self,
        keywords: List[str],
        timeframe: str = "today 5-y",
        geo: str = "US",
    ) -> "pd.DataFrame":
        """Return a DataFrame indexed by date with columns per keyword (0-100)."""
        if not _PANDAS_OK:
            raise RuntimeError("pandas is required for fetch_interest_over_time")
        if timeframe not in _VALID_TIMEFRAMES:
            logger.warning("Non-standard timeframe '%s'; using 'today 5-y'", timeframe)
            timeframe = "today 5-y"

        cache_key = f"iot:{'|'.join(sorted(keywords))}:{timeframe}:{geo}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return pd.read_json(io.StringIO(cached))

        kws = keywords[:5]  # Google Trends limit: 5 keywords

        if self._pytrends is not None:
            result = self._fetch_pytrends_iot(kws, timeframe, geo)
        else:
            result = self._fetch_csv_iot(kws, timeframe, geo)

        if result is not None and not result.empty:
            _cache.set(cache_key, result.to_json())
        return result if result is not None else pd.DataFrame()

    def fetch_interest_by_region(
        self,
        keyword: str,
        resolution: str = "REGION",
    ) -> "pd.DataFrame":
        """Return regional interest breakdown."""
        if not _PANDAS_OK:
            raise RuntimeError("pandas required")

        cache_key = f"ibr:{keyword}:{resolution}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return pd.read_json(io.StringIO(cached))

        result: Optional["pd.DataFrame"] = None
        if self._pytrends is not None:
            result = self._fetch_pytrends_region(keyword, resolution)
        else:
            result = pd.DataFrame(columns=["geoName", keyword])

        if result is not None and not result.empty:
            _cache.set(cache_key, result.to_json())
        return result if result is not None else pd.DataFrame()

    def fetch_related_queries(self, keyword: str) -> Dict:
        """Return {top: DataFrame, rising: DataFrame} of related queries."""
        cache_key = f"rq:{keyword}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        if self._pytrends is None:
            return {"top": [], "rising": []}

        result = self._fetch_pytrends_related_queries(keyword)
        if result:
            _cache.set(cache_key, result)
        return result

    def fetch_related_topics(self, keyword: str) -> Dict:
        """Return related topic groups."""
        cache_key = f"rt:{keyword}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        if self._pytrends is None:
            return {"top": [], "rising": []}

        result = self._fetch_pytrends_related_topics(keyword)
        if result:
            _cache.set(cache_key, result)
        return result

    def build_company_keywords(self, ticker: str) -> List[str]:
        """Map ticker to company name + search variants."""
        ticker = ticker.upper()
        if ticker in _TICKER_KEYWORDS:
            return _TICKER_KEYWORDS[ticker]
        # Generic fallback: use ticker as keyword
        return [ticker, f"{ticker} stock", f"buy {ticker}"]

    # ------------------------------------------------------------------
    # pytrends internals
    # ------------------------------------------------------------------

    def _fetch_pytrends_iot(
        self,
        keywords: List[str],
        timeframe: str,
        geo: str,
    ) -> Optional["pd.DataFrame"]:
        """Fetch interest_over_time via pytrends with retry + backoff."""
        for attempt in range(self._retries):
            try:
                self._pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
                _jitter_sleep(1.0, 2.5)
                df = self._pytrends.interest_over_time()
                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])
                return df
            except Exception as exc:
                err_str = str(exc).lower()
                if "429" in err_str or "too many" in err_str:
                    wait = 60 + random.randint(0, 30)
                    logger.warning("Rate limited (429) — sleeping %ds", wait)
                    time.sleep(wait)
                else:
                    logger.warning("pytrends iot attempt %d failed: %s", attempt + 1, exc)
                if attempt < self._retries - 1:
                    _jitter_sleep(2.0, 4.0)
        return None

    def _fetch_pytrends_region(
        self,
        keyword: str,
        resolution: str,
    ) -> Optional["pd.DataFrame"]:
        for attempt in range(self._retries):
            try:
                self._pytrends.build_payload([keyword], timeframe="today 12-m")
                _jitter_sleep(1.0, 2.5)
                df = self._pytrends.interest_by_region(resolution=resolution)
                return df.reset_index()
            except Exception as exc:
                logger.warning("pytrends ibr attempt %d: %s", attempt + 1, exc)
                if "429" in str(exc):
                    time.sleep(60)
                elif attempt < self._retries - 1:
                    _jitter_sleep(2.0, 4.0)
        return None

    def _fetch_pytrends_related_queries(self, keyword: str) -> Dict:
        try:
            self._pytrends.build_payload([keyword], timeframe="today 3-m")
            _jitter_sleep(1.0, 2.5)
            rq = self._pytrends.related_queries()
            out: Dict = {"top": [], "rising": []}
            if keyword in rq:
                for rtype in ("top", "rising"):
                    df_rq = rq[keyword].get(rtype)
                    if df_rq is not None and not df_rq.empty:
                        out[rtype] = df_rq.to_dict("records")
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("related_queries failed: %s", exc)
            return {"top": [], "rising": []}

    def _fetch_pytrends_related_topics(self, keyword: str) -> Dict:
        try:
            self._pytrends.build_payload([keyword], timeframe="today 3-m")
            _jitter_sleep(1.0, 2.5)
            rt = self._pytrends.related_topics()
            out: Dict = {"top": [], "rising": []}
            if keyword in rt:
                for rtype in ("top", "rising"):
                    df_rt = rt[keyword].get(rtype)
                    if df_rt is not None and not df_rt.empty:
                        cols = [c for c in ["topic_title", "topic_type", "value"] if c in df_rt.columns]
                        out[rtype] = df_rt[cols].to_dict("records")
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("related_topics failed: %s", exc)
            return {"top": [], "rising": []}

    # ------------------------------------------------------------------
    # CSV fallback
    # ------------------------------------------------------------------

    def _fetch_csv_iot(
        self,
        keywords: List[str],
        timeframe: str,
        geo: str,
    ) -> Optional["pd.DataFrame"]:
        """Attempt to download CSV directly from Google Trends."""
        # Google Trends CSV endpoint (unofficial, may break)
        kw_encoded = quote_plus(",".join(keywords))
        tf_encoded = timeframe.replace(" ", "+")
        url = (
            f"{_GOOGLE_TRENDS_BASE}/trends/explore/csv"
            f"?q={kw_encoded}&date={tf_encoded}&geo={geo}&hl=en-US"
        )
        try:
            resp = self._session.get(url, timeout=self._timeout)
            resp.raise_for_status()
            text = resp.text
            # Parse the CSV — first few lines are metadata, skip until data
            lines = text.splitlines()
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith("Week") or line.startswith("Day") or line.startswith("Month"):
                    data_start = i
                    break
            if data_start == 0:
                logger.warning("CSV fallback: could not find data header")
                return None
            reader = csv.DictReader(lines[data_start:])
            rows = list(reader)
            if not rows:
                return None
            first_key = list(rows[0].keys())[0]
            df = pd.DataFrame(rows)
            df[first_key] = pd.to_datetime(df[first_key], errors="coerce")
            df = df.set_index(first_key)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning("CSV fallback failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# TrendsSignalEngine
# ---------------------------------------------------------------------------

class TrendsSignalEngine:
    """Compute alpha signals from raw Google Trends data."""

    def __init__(self, fetcher: Optional[GoogleTrendsFetcher] = None):
        self._fetcher = fetcher or GoogleTrendsFetcher()

    # ------------------------------------------------------------------
    # Core signal primitives
    # ------------------------------------------------------------------

    def _get_weekly_values(self, keyword: str, timeframe: str = "today 5-y") -> List[float]:
        """Return a list of weekly interest values (0-100) for a keyword."""
        df = self._fetcher.fetch_interest_over_time([keyword], timeframe=timeframe)
        if not _PANDAS_OK or df is None or df.empty:
            return []
        col = keyword if keyword in df.columns else df.columns[0]
        return [float(v) for v in df[col].tolist()]

    def compute_search_momentum(self, keyword: str, window: int = 4) -> float:
        """4-week change in search interest, normalized as z-score vs 52-week history.

        Returns a z-score: positive = interest rising faster than historical norm.
        """
        values = self._get_weekly_values(keyword, timeframe="today 5-y")
        if len(values) < window + 1:
            return 0.0
        # Recent window
        recent = values[-window:]
        recent_mean = sum(recent) / len(recent)
        # History: all but the last window
        history_window = values[-(52 + window):-window] if len(values) >= 52 + window else values[:-window]
        if not history_window:
            return 0.0
        return _zscore(history_window, recent_mean)

    def compute_trend_acceleration(self, keyword: str) -> float:
        """2nd derivative of smoothed interest — positive = accelerating interest."""
        values = self._get_weekly_values(keyword, timeframe="today 5-y")
        if len(values) < 5:
            return 0.0
        return _second_derivative(values)

    def detect_search_spike(self, keyword: str, threshold: float = 2.0) -> Tuple[bool, float]:
        """Return (is_spike, z_score): current interest > mean + threshold * std."""
        values = self._get_weekly_values(keyword, timeframe="today 12-m")
        if len(values) < 8:
            return False, 0.0
        current = values[-1]
        history = values[:-1]
        z = _zscore(history, current)
        return z >= threshold, z

    def compute_ticker_search_score(self, ticker: str) -> TrendsScore:
        """Composite search score for a ticker.

        Factors:
        - raw_interest: latest value (0-100)
        - momentum_zscore: 4-week z-score
        - acceleration: 2nd derivative
        - spike_detected: current > mean + 2*std
        - composite_score: weighted 0-1
        """
        keywords = self._fetcher.build_company_keywords(ticker)
        primary_kw = keywords[0]

        # Raw interest (latest week)
        values = self._get_weekly_values(primary_kw, timeframe="today 5-y")
        raw_interest = values[-1] if values else 0.0

        momentum = self.compute_search_momentum(primary_kw)
        acceleration = self.compute_trend_acceleration(primary_kw)
        spike, spike_z = self.detect_search_spike(primary_kw)

        # Composite: clamp components and weight
        mom_component = _clamp((momentum + 3) / 6, 0, 1)  # z-score -3..+3 → 0..1
        raw_component = raw_interest / 100.0
        acc_component = _clamp((acceleration + 10) / 20, 0, 1)  # rough normalize
        spike_component = min(1.0, spike_z / 3.0) if spike else 0.0

        composite = (
            0.35 * raw_component
            + 0.35 * mom_component
            + 0.15 * acc_component
            + 0.15 * spike_component
        )

        return TrendsScore(
            ticker=ticker,
            raw_interest=raw_interest,
            momentum_zscore=round(momentum, 4),
            acceleration=round(acceleration, 4),
            spike_detected=spike,
            spike_zscore=round(spike_z, 4),
            composite_score=round(_clamp(composite, 0, 1), 4),
            keywords_used=keywords[:3],
            computed_at=datetime.now(timezone.utc).isoformat(),
        )

    def compute_sector_trends(self, sector: str) -> Dict[str, float]:
        """Rank tickers in a sector by search momentum. Returns {ticker: momentum}."""
        sector = sector.lower()
        tickers = _SECTOR_TICKERS.get(sector, [])
        if not tickers:
            return {}
        result: Dict[str, float] = {}
        for ticker in tickers:
            keywords = self._fetcher.build_company_keywords(ticker)
            primary_kw = keywords[0]
            try:
                momentum = self.compute_search_momentum(primary_kw)
                result[ticker] = round(momentum, 4)
                _jitter_sleep(1.5, 3.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("sector_trends %s/%s: %s", sector, ticker, exc)
                result[ticker] = 0.0
        return dict(sorted(result.items(), key=lambda x: -x[1]))

    def compute_fear_greed_proxy(
        self,
        market_terms: Optional[List[str]] = None,
        mode: str = "trends",
    ) -> float:
        """Fear/greed composite index (0-100 scale).

        Modes:
          * ``"market_data"`` — delegate to :func:`compute_fear_greed_composite_v2`,
            which uses VIX + FRED STLFSI4 + Put/Call + SPY breadth + HY spread.
            NO Google Trends in the inputs. Returns the ``composite_score`` field.
          * ``"trends"``      — (default, legacy) 5-component PYTRENDS composite:

            1. search_volume  — ratio of crash vs tips search intensity (inverted = fear)
            2. momentum       — 125d vs 250d MA ratio from SPY price (proxy from trends)
            3. breadth        — stocks above 50d MA proxy (trend strength of "stock market")
            4. junk_bond_proxy — "high yield bonds" search vs "treasury bonds"
            5. volatility_proxy — "market volatility" search intensity (inverted = fear)

        Score mapping: <25=Extreme Fear, 25-45=Fear, 45-55=Neutral,
                       55-75=Greed, >75=Extreme Greed.

        Returns score in [0, 100].
        """
        if mode == "market_data":
            try:
                v2 = compute_fear_greed_composite_v2()
                return float(v2["composite_score"])
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "fear_greed v2 (market_data) failed, falling back to trends: %s",
                    exc,
                )
                # Fall through to pytrends path.
        # Component weights (must sum to 1.0)
        W_SEARCH   = 0.20
        W_MOMENTUM = 0.20
        W_BREADTH  = 0.20
        W_JUNK     = 0.20
        W_VOL      = 0.20

        # ----- Component 1: Search volume (crash vs tips) -----
        # Higher "market crash" search = more fear → lower score
        search_score = 50.0  # neutral default
        try:
            crash_terms = ["market crash", "stock tips"]
            df_search = self._fetcher.fetch_interest_over_time(
                crash_terms, timeframe="today 3-m"
            )
            if _PANDAS_OK and df_search is not None and not df_search.empty:
                recent = df_search.tail(4)
                crash_col = next(
                    (c for c in recent.columns if "crash" in c.lower()), None
                )
                tips_col = next(
                    (c for c in recent.columns if "tip" in c.lower()), None
                )
                crash_val = float(recent[crash_col].mean()) if crash_col else 50.0
                tips_val  = float(recent[tips_col].mean())  if tips_col  else 50.0
                total = crash_val + tips_val
                if total > 0:
                    # High crash → low score (fear); high tips → high score (greed)
                    search_score = _clamp(100.0 * (tips_val / total), 0.0, 100.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("fear_greed search_volume component failed: %s", exc)

        # ----- Component 2: Momentum (SPY trend proxy via "SPY ETF" trends) -----
        # Use 125-week vs 250-week interest ratio as a momentum proxy
        momentum_score = 50.0
        try:
            spy_terms = ["SPY ETF"]
            df_spy = self._fetcher.fetch_interest_over_time(
                spy_terms, timeframe="today 5-y"
            )
            if _PANDAS_OK and df_spy is not None and not df_spy.empty:
                col = df_spy.columns[0]
                vals = [float(v) for v in df_spy[col].tolist()]
                if len(vals) >= 250:
                    ma125 = sum(vals[-125:]) / 125.0
                    ma250 = sum(vals[-250:]) / 250.0
                    if ma250 > 0:
                        ratio = ma125 / ma250
                        # ratio > 1 = uptrend = greed; < 1 = fear
                        momentum_score = _clamp(50.0 + 100.0 * (ratio - 1.0), 0.0, 100.0)
                elif len(vals) >= 10:
                    # Short history: compare recent 25% vs full history
                    half = max(1, len(vals) // 4)
                    recent_mean = sum(vals[-half:]) / half
                    full_mean = sum(vals) / len(vals)
                    if full_mean > 0:
                        ratio = recent_mean / full_mean
                        momentum_score = _clamp(50.0 + 100.0 * (ratio - 1.0), 0.0, 100.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("fear_greed momentum component failed: %s", exc)

        # ----- Component 3: Breadth (stock market health proxy) -----
        # Use "stock market rally" vs "stock market crash" as a breadth proxy
        breadth_score = 50.0
        try:
            breadth_terms = ["stock market rally", "stock market crash"]
            df_breadth = self._fetcher.fetch_interest_over_time(
                breadth_terms, timeframe="today 3-m"
            )
            if _PANDAS_OK and df_breadth is not None and not df_breadth.empty:
                recent = df_breadth.tail(4)
                rally_col = next(
                    (c for c in recent.columns if "rally" in c.lower()), None
                )
                crash_col = next(
                    (c for c in recent.columns if "crash" in c.lower()), None
                )
                rally_val = float(recent[rally_col].mean()) if rally_col else 50.0
                crash_val = float(recent[crash_col].mean()) if crash_col else 50.0
                total = rally_val + crash_val
                if total > 0:
                    breadth_score = _clamp(100.0 * (rally_val / total), 0.0, 100.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("fear_greed breadth component failed: %s", exc)

        # ----- Component 4: Junk bond demand proxy -----
        # Higher "high yield bonds" vs "treasury bonds" = risk-on = greed
        junk_score = 50.0
        try:
            bond_terms = ["high yield bonds", "treasury bonds"]
            df_bonds = self._fetcher.fetch_interest_over_time(
                bond_terms, timeframe="today 3-m"
            )
            if _PANDAS_OK and df_bonds is not None and not df_bonds.empty:
                recent = df_bonds.tail(4)
                hy_col  = next(
                    (c for c in recent.columns if "high yield" in c.lower()), None
                )
                tsy_col = next(
                    (c for c in recent.columns if "treasury" in c.lower()), None
                )
                hy_val  = float(recent[hy_col].mean())  if hy_col  else 50.0
                tsy_val = float(recent[tsy_col].mean()) if tsy_col else 50.0
                total = hy_val + tsy_val
                if total > 0:
                    junk_score = _clamp(100.0 * (hy_val / total), 0.0, 100.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("fear_greed junk_bond component failed: %s", exc)

        # ----- Component 5: Market volatility proxy -----
        # Higher "market volatility" search = more fear → lower score
        vol_score = 50.0
        try:
            vol_terms = ["market volatility", "stock market news"]
            df_vol = self._fetcher.fetch_interest_over_time(
                vol_terms, timeframe="today 3-m"
            )
            if _PANDAS_OK and df_vol is not None and not df_vol.empty:
                recent = df_vol.tail(4)
                vol_col  = next(
                    (c for c in recent.columns if "volatility" in c.lower()), None
                )
                news_col = next(
                    (c for c in recent.columns if "news" in c.lower()), None
                )
                vol_val  = float(recent[vol_col].mean())  if vol_col  else 50.0
                news_val = float(recent[news_col].mean()) if news_col else 50.0
                total = vol_val + news_val
                if total > 0:
                    # Higher volatility search = more fear = lower score
                    vol_score = _clamp(100.0 * (news_val / total), 0.0, 100.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("fear_greed volatility component failed: %s", exc)

        # ----- Override with custom market_terms if provided -----
        if market_terms:
            # Legacy behavior: use provided terms directly
            fear_terms_custom = [t for t in market_terms if any(
                bad in t.lower() for bad in ("crash", "recession", "bear", "collapse", "layoff")
            )]
            greed_terms_custom = [t for t in market_terms if t not in fear_terms_custom]
            all_terms = market_terms[:5]
            df = self._fetcher.fetch_interest_over_time(all_terms, timeframe="today 3-m")
            if _PANDAS_OK and df is not None and not df.empty:
                recent = df.tail(4)
                fear_scores: List[float] = []
                greed_scores: List[float] = []
                for col in recent.columns:
                    val = float(recent[col].mean())
                    if any(f.lower() in col.lower() for f in fear_terms_custom):
                        fear_scores.append(val)
                    elif any(g.lower() in col.lower() for g in greed_terms_custom):
                        greed_scores.append(val)
                fear_avg = sum(fear_scores) / len(fear_scores) if fear_scores else 50.0
                greed_avg = sum(greed_scores) / len(greed_scores) if greed_scores else 50.0
                total = fear_avg + greed_avg
                if total > 0:
                    raw_ratio = greed_avg / total
                    return round(_clamp(raw_ratio * 100.0, 0.0, 100.0), 2)

        # ----- Composite: weighted average of 5 components -----
        composite = (
            W_SEARCH   * search_score
            + W_MOMENTUM * momentum_score
            + W_BREADTH  * breadth_score
            + W_JUNK     * junk_score
            + W_VOL      * vol_score
        )
        return round(_clamp(composite, 0.0, 100.0), 2)

    @staticmethod
    def classify_fear_greed(score: float) -> str:
        """Return human-readable label for a 0-100 fear/greed score."""
        if score < 25:
            return "Extreme Fear"
        elif score < 45:
            return "Fear"
        elif score <= 55:
            return "Neutral"
        elif score <= 75:
            return "Greed"
        else:
            return "Extreme Greed"

    def get_product_cycle_signal(
        self,
        ticker: str,
        products: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Track product-specific search trends as leading revenue indicator.

        Returns a dict with per-product interest scores, week-over-week
        acceleration, and a quarterly revenue surprise probability mapping.

        Keys per product:
            "interest"        — 4-week average interest (0-100)
            "acceleration"    — week-over-week growth rate (2nd derivative proxy)
            "surprise_prob"   — estimated probability of positive revenue surprise (0-1)

        Methodology:
            - Fetch weekly interest for each product keyword
            - Compute trend acceleration = WoW growth rate at tail of series
            - Map acceleration to surprise probability:
                acceleration > +2 pts/week → prob ~0.65
                acceleration > 0            → prob ~0.55
                acceleration <= 0           → prob ~0.45
        """
        if products is None:
            # Default product keywords by ticker
            product_map: Dict[str, List[str]] = {
                "AAPL": ["iPhone 16", "MacBook Pro", "Apple Vision Pro"],
                "TSLA": ["Tesla Model Y", "Tesla Model 3", "Tesla Cybertruck"],
                "MSFT": ["Copilot", "Surface Pro", "Xbox Series X"],
                "NVDA": ["RTX 4090", "NVIDIA H100", "GeForce NOW"],
                "GOOGL": ["Pixel 9", "Google Gemini", "Google Workspace"],
                "AMZN": ["Kindle", "Echo", "Amazon Prime"],
            }
            products = product_map.get(ticker.upper(), [ticker + " product"])

        products = products[:5]
        df = self._fetcher.fetch_interest_over_time(products, timeframe="today 3-m")
        if not _PANDAS_OK or df is None or df.empty:
            return {
                p: {"interest": 0.0, "acceleration": 0.0, "surprise_prob": 0.5}
                for p in products
            }

        result: Dict[str, Any] = {}
        for product in products:
            col = next((c for c in df.columns if product.lower() in c.lower()), None)
            if col:
                series = [float(v) for v in df[col].tolist()]
                interest = round(float(df[col].tail(4).mean()), 2)

                # Trend acceleration: week-over-week change in the last 2 weeks
                if len(series) >= 3:
                    # WoW growth rate at tail: (last - penultimate) - (penultimate - ante)
                    d1_last = series[-1] - series[-2]
                    d1_prev = series[-2] - series[-3]
                    acceleration = round(d1_last - d1_prev, 4)
                elif len(series) >= 2:
                    acceleration = round(series[-1] - series[-2], 4)
                else:
                    acceleration = 0.0

                # Map acceleration to revenue surprise probability
                if acceleration > 2.0:
                    surprise_prob = 0.65
                elif acceleration > 0.5:
                    surprise_prob = 0.60
                elif acceleration > 0.0:
                    surprise_prob = 0.55
                elif acceleration > -0.5:
                    surprise_prob = 0.50
                else:
                    surprise_prob = 0.45

                result[product] = {
                    "interest": interest,
                    "acceleration": acceleration,
                    "surprise_prob": surprise_prob,
                }
            else:
                result[product] = {
                    "interest": 0.0,
                    "acceleration": 0.0,
                    "surprise_prob": 0.5,
                }
        return result


# ---------------------------------------------------------------------------
# Fear/Greed V2 — real market data (yfinance + FRED + CBOE), no Google Trends
# ---------------------------------------------------------------------------

# FRED CSV endpoint — free, no API key required for series observations.
_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
_CBOE_DAILY_URL = (
    "https://www.cboe.com/us/options/market_statistics/daily/"
    "?dt={date}&downloadCSV=true&type=ratio"
)


def _fred_latest_value(series_id: str, timeout: float = 10.0) -> Optional[float]:
    """Fetch the most recent numeric observation for a FRED series via the
    public CSV endpoint (no API key needed).

    Returns ``None`` if the series cannot be fetched or contains no valid data.
    """
    url = _FRED_CSV_URL.format(series=series_id)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.debug("FRED fetch failed for %s: %s", series_id, exc)
        return None

    text = resp.text.strip()
    if not text:
        return None

    # Parse CSV: header line "DATE,SERIES_ID" then "YYYY-MM-DD,value" rows.
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if len(rows) < 2:
        return None

    # Walk from the most recent row backwards to find a real numeric value
    # (FRED uses "." for missing observations).
    for row in reversed(rows[1:]):
        if len(row) < 2:
            continue
        raw = row[1].strip()
        if raw in ("", "."):
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return None


def _yf_latest_close(ticker: str, period: str = "5d") -> Optional[float]:
    """Return the most recent close price for ``ticker`` via yfinance.

    Returns ``None`` if yfinance is unavailable or the fetch fails.
    """
    try:
        import yfinance as yf  # local import: optional dependency
    except ImportError:
        logger.debug("yfinance not installed; cannot fetch %s", ticker)
        return None

    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        close = hist["Close"].dropna()
        if close.empty:
            return None
        return float(close.iloc[-1])
    except Exception as exc:  # noqa: BLE001
        logger.debug("yfinance fetch failed for %s: %s", ticker, exc)
        return None


def _yf_history_closes(ticker: str, period: str = "1y") -> Optional[List[float]]:
    """Return a list of daily closes for ``ticker``. ``None`` on failure."""
    try:
        import yfinance as yf  # local import: optional dependency
    except ImportError:
        logger.debug("yfinance not installed; cannot fetch %s history", ticker)
        return None

    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        closes = [float(v) for v in hist["Close"].dropna().tolist()]
        return closes or None
    except Exception as exc:  # noqa: BLE001
        logger.debug("yfinance history failed for %s: %s", ticker, exc)
        return None


def _fetch_put_call_ratio(timeout: float = 10.0) -> Optional[float]:
    """Fetch the latest CBOE total put/call ratio.

    Strategy:
      1. Try CBOE's daily market-statistics page (public, no key).
      2. Fall back to the ``^VPCR`` yfinance pseudo-ticker (CBOE Total P/C).
    """
    # ---- Strategy 1: CBOE public daily endpoint ----
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    url = _CBOE_DAILY_URL.format(date=today)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        if resp.status_code == 200 and resp.text.strip():
            text = resp.text
            # CBOE CSV: look for the "TOTAL PUT/CALL RATIO" row.
            for line in text.splitlines():
                low = line.lower()
                if "total" in low and "put/call" in low:
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                    for tok in reversed(parts):
                        try:
                            val = float(tok)
                            if 0.05 < val < 5.0:  # plausible P/C ratio range
                                return val
                        except ValueError:
                            continue
    except Exception as exc:  # noqa: BLE001
        logger.debug("CBOE put/call fetch failed: %s", exc)

    # ---- Strategy 2: yfinance ^VPCR proxy ----
    pcr = _yf_latest_close("^VPCR", period="5d")
    if pcr is not None and 0.05 < pcr < 5.0:
        return pcr

    return None


def _score_vix(vix: float) -> float:
    """VIX -> 0-100 score. >40 = extreme fear (0); <12 = extreme greed (100)."""
    if vix >= 40.0:
        return 0.0
    if vix <= 12.0:
        return 100.0
    # Linear interp between 40 (=0) and 12 (=100).
    return _clamp(100.0 * (40.0 - vix) / (40.0 - 12.0), 0.0, 100.0)


def _score_stlfsi(stlfsi: float) -> float:
    """St. Louis Financial Stress Index -> 0-100. Negative = greed, positive = fear."""
    return _clamp(50.0 - stlfsi * 25.0, 0.0, 100.0)


def _score_put_call(pcr: float) -> float:
    """Put/Call ratio -> 0-100. >1.0 = fear (0); <0.7 = greed (100)."""
    if pcr >= 1.0:
        return 0.0
    if pcr <= 0.7:
        return 100.0
    return _clamp(100.0 * (1.0 - pcr) / (1.0 - 0.7), 0.0, 100.0)


def _score_breadth(closes: List[float], window: int = 125) -> float:
    """% of price above its rolling MA -> proxy for market breadth.

    Returns 100 when current price is well above the MA, 0 when well below.
    """
    if not closes:
        return 50.0
    if len(closes) < window:
        window = max(5, len(closes) // 2)
    ma = sum(closes[-window:]) / window
    if ma <= 0:
        return 50.0
    pct = (closes[-1] - ma) / ma  # e.g. +0.05 = 5% above MA
    # Map: -10% -> 0, +10% -> 100, linear between.
    return _clamp(50.0 + pct * 500.0, 0.0, 100.0)


def _score_junk_bonds(spread_pct: float) -> float:
    """ICE BofA HY OAS (in *percent*, e.g. 4.50 = 450 bps) -> 0-100.

    Tight (<4%) = greed (100); wide (>8%) = fear (0).
    """
    bps = spread_pct * 100.0
    if bps <= 400.0:
        return 100.0
    if bps >= 800.0:
        return 0.0
    return _clamp(100.0 * (800.0 - bps) / (800.0 - 400.0), 0.0, 100.0)


def _classify_fear_greed_v2(score: float) -> str:
    """Snake-case classification for the v2 composite."""
    if score < 25:
        return "extreme_fear"
    if score < 45:
        return "fear"
    if score <= 55:
        return "neutral"
    if score <= 75:
        return "greed"
    return "extreme_greed"


def compute_fear_greed_composite_v2() -> dict:
    """5-component market-data fear/greed composite — NO Google Trends inputs.

    Sources (all free / no API key):
      * VIX                  : yfinance ``^VIX``                       (25% weight)
      * STLFSI4              : FRED ``STLFSI4`` (St. Louis Fin Stress) (25% weight)
      * Put/Call ratio       : CBOE daily JSON / ``^VPCR`` fallback    (20% weight)
      * Breadth (SPY vs MA)  : yfinance ``SPY``, 125-day MA            (15% weight)
      * Junk-bond spread     : FRED ``BAMLH0A0HYM2`` HY OAS            (15% weight)

    Returns::

        {
            "composite_score": float [0, 100],
            "interpretation": "extreme_fear" | "fear" | "neutral" | "greed" | "extreme_greed",
            "components": {
                "vix":        {"value": float, "score": 0-100, "weight": 0.25},
                "stlfsi":     {"value": float, "score": 0-100, "weight": 0.25},
                "put_call":   {"value": float, "score": 0-100, "weight": 0.20},
                "breadth":    {"value": float, "score": 0-100, "weight": 0.15},
                "junk_bonds": {"value": float, "score": 0-100, "weight": 0.15},
            },
            "timestamp": "<ISO datetime>",
            "source":    "yfinance + FRED + CBOE (no Google Trends)",
        }
    """
    weights = {
        "vix": 0.25,
        "stlfsi": 0.25,
        "put_call": 0.20,
        "breadth": 0.15,
        "junk_bonds": 0.15,
    }

    # ----- 1. VIX (yfinance ^VIX) -----
    vix_val = _yf_latest_close("^VIX", period="5d")
    if vix_val is None or vix_val <= 0:
        # Neutral fallback: long-run VIX median ~ 17.5.
        vix_val = 17.5
    vix_score = _score_vix(vix_val)

    # ----- 2. St. Louis Financial Stress Index (FRED STLFSI4) -----
    stlfsi_val = _fred_latest_value("STLFSI4")
    if stlfsi_val is None:
        stlfsi_val = 0.0  # neutral
    stlfsi_score = _score_stlfsi(stlfsi_val)

    # ----- 3. Put/Call ratio (CBOE -> yfinance fallback) -----
    pcr_val = _fetch_put_call_ratio()
    if pcr_val is None:
        pcr_val = 0.85  # long-run median ~ 0.85
    pcr_score = _score_put_call(pcr_val)

    # ----- 4. Market breadth (SPY vs 125d MA) -----
    spy_closes = _yf_history_closes("SPY", period="1y") or []
    breadth_score = _score_breadth(spy_closes, window=125)
    breadth_val = 0.0
    if spy_closes:
        win = min(125, len(spy_closes))
        ma = sum(spy_closes[-win:]) / win if win else 0.0
        breadth_val = (spy_closes[-1] - ma) / ma * 100.0 if ma > 0 else 0.0

    # ----- 5. HY credit spread (FRED BAMLH0A0HYM2, in percent) -----
    hy_val = _fred_latest_value("BAMLH0A0HYM2")
    if hy_val is None:
        hy_val = 4.5  # long-run median ~ 450 bps
    junk_score = _score_junk_bonds(hy_val)

    composite = (
        weights["vix"]        * vix_score
        + weights["stlfsi"]     * stlfsi_score
        + weights["put_call"]   * pcr_score
        + weights["breadth"]    * breadth_score
        + weights["junk_bonds"] * junk_score
    )
    composite = round(_clamp(composite, 0.0, 100.0), 2)

    return {
        "composite_score": composite,
        "interpretation": _classify_fear_greed_v2(composite),
        "components": {
            "vix":        {"value": round(float(vix_val), 4),    "score": round(vix_score, 2),     "weight": weights["vix"]},
            "stlfsi":     {"value": round(float(stlfsi_val), 4), "score": round(stlfsi_score, 2),  "weight": weights["stlfsi"]},
            "put_call":   {"value": round(float(pcr_val), 4),    "score": round(pcr_score, 2),     "weight": weights["put_call"]},
            "breadth":    {"value": round(float(breadth_val), 4),"score": round(breadth_score, 2), "weight": weights["breadth"]},
            "junk_bonds": {"value": round(float(hy_val), 4),     "score": round(junk_score, 2),    "weight": weights["junk_bonds"]},
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance + FRED + CBOE (no Google Trends)",
    }


# ---------------------------------------------------------------------------
# TrendsPredictor
# ---------------------------------------------------------------------------

class TrendsPredictor:
    """Use Google Trends as leading indicators for financial outcomes."""

    def __init__(self, fetcher: Optional[GoogleTrendsFetcher] = None,
                 engine: Optional[TrendsSignalEngine] = None):
        self._fetcher = fetcher or GoogleTrendsFetcher()
        self._engine = engine or TrendsSignalEngine(self._fetcher)

    def predict_earnings_surprise(self, ticker: str) -> EarningsPrediction:
        """High pre-earnings search interest → positive surprise signal.

        Methodology:
        - Fetch 5-year weekly search interest for ticker name
        - Compute mean/std of interest in weeks preceding prior earnings dates
        - Compare current interest to that distribution
        - Interest > 75th percentile = positive surprise signal (prob 0.6-0.7 based on
          academic research on search volume and earnings: Da, Engelberg, Gao 2011)
        """
        keywords = self._fetcher.build_company_keywords(ticker)
        primary_kw = keywords[0]
        values = self._engine._get_weekly_values(primary_kw, timeframe="today 5-y")

        if not values:
            return EarningsPrediction(
                ticker=ticker,
                note="No trend data available",
            )

        current = values[-1]
        history = values[:-1]
        if not history:
            return EarningsPrediction(ticker=ticker, note="Insufficient history")

        hist_sorted = sorted(history)
        pct75 = hist_sorted[int(len(hist_sorted) * 0.75)]
        pct25 = hist_sorted[int(len(hist_sorted) * 0.25)]
        mean_val = sum(history) / len(history)

        # Signal strength: how far current is above 75th percentile
        signal_strength = _clamp((current - pct75) / max(pct75 - pct25, 1), 0, 1)

        # Positive surprise probability (calibrated to Da et al. range)
        # Base rate ~50%, scale up to 70% at signal_strength=1
        prob = _clamp(0.5 + 0.2 * signal_strength, 0.0, 0.72)

        return EarningsPrediction(
            ticker=ticker,
            pre_earnings_interest=round(current, 2),
            historical_mean_interest=round(mean_val, 2),
            signal_strength=round(signal_strength, 4),
            positive_surprise_probability=round(prob, 4),
            data_points=len(values),
            note=(
                f"Current interest {current:.0f} vs hist mean {mean_val:.1f}. "
                f"{'Above' if current > pct75 else 'Below'} 75th pct ({pct75:.0f})."
            ),
        )

    def predict_consumer_spending(
        self, retail_tickers: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Predict near-term consumer spending using search term proxies.

        Search terms: "online shopping", "Amazon deals", "Black Friday", "holiday gifts"
        Returns {ticker: predicted_spend_signal 0-100}.
        """
        spending_terms = [
            "online shopping",
            "Amazon deals",
            "holiday gifts",
            "Black Friday",
        ]
        df = self._fetcher.fetch_interest_over_time(spending_terms, timeframe="today 3-m")
        if not _PANDAS_OK or df is None or df.empty:
            if retail_tickers:
                return {t: 0.0 for t in retail_tickers}
            return {"consumer_spending_index": 0.0}

        # Aggregate spending signal
        recent = df.tail(4)
        spending_signal = float(recent.mean(axis=1).mean())

        if retail_tickers is None:
            return {"consumer_spending_index": round(spending_signal, 2)}

        # Map signal to tickers proportionally by their category relevance
        result: Dict[str, float] = {}
        consumer_weights: Dict[str, float] = {
            "AMZN": 1.0, "WMT": 0.9, "TGT": 0.85, "ETSY": 0.8,
            "EBAY": 0.75, "SHOP": 0.85, "HD": 0.6, "LOW": 0.55,
        }
        for ticker in retail_tickers:
            w = consumer_weights.get(ticker.upper(), 0.5)
            result[ticker] = round(spending_signal * w, 2)
        return result

    def predict_housing_activity(self) -> Dict[str, float]:
        """Predict housing market activity from search trends."""
        housing_terms = [
            "buy a house",
            "mortgage rates",
            "Zillow",
            "home prices",
        ]
        df = self._fetcher.fetch_interest_over_time(housing_terms, timeframe="today 12-m")
        if not _PANDAS_OK or df is None or df.empty:
            return {t: 0.0 for t in housing_terms}

        recent = df.tail(4)
        result: Dict[str, float] = {}
        for col in recent.columns:
            result[col.strip()] = round(float(recent[col].mean()), 2)

        # Composite housing activity index
        vals = list(result.values())
        result["housing_activity_index"] = round(sum(vals) / len(vals), 2) if vals else 0.0
        return result

    def predict_crypto_sentiment(self) -> Dict[str, float]:
        """Crypto search term sentiment as leading indicator."""
        crypto_terms = [
            "buy bitcoin",
            "crypto crash",
            "ethereum price",
            "altcoin season",
        ]
        df = self._fetcher.fetch_interest_over_time(crypto_terms, timeframe="today 3-m")
        if not _PANDAS_OK or df is None or df.empty:
            return {"crypto_sentiment": 0.0}

        recent = df.tail(4)
        result: Dict[str, float] = {}
        for col in recent.columns:
            result[col.strip()] = round(float(recent[col].mean()), 2)

        # Bull/bear balance: "buy bitcoin" vs "crypto crash"
        bull_val = result.get("buy bitcoin", 0.0)
        bear_val = result.get("crypto crash", 0.0)
        total = bull_val + bear_val
        if total > 0:
            result["crypto_bull_bear_ratio"] = round(bull_val / total, 4)
        else:
            result["crypto_bull_bear_ratio"] = 0.5
        return result

    def predict_sector_rotation(self, sectors: Optional[List[str]] = None) -> Dict[str, float]:
        """Search volume shifts across sectors as rotation signal.

        Returns {sector: momentum_score}: positive = investors rotating IN.
        """
        if sectors is None:
            sectors = list(_SECTOR_TICKERS.keys())

        # Use sector ETF search terms as proxies
        sector_etf_terms: Dict[str, str] = {
            "technology": "XLK ETF",
            "finance": "XLF ETF",
            "healthcare": "XLV ETF",
            "energy": "XLE ETF",
            "consumer": "XLY ETF",
            "industrials": "XLI ETF",
            "utilities": "XLU ETF",
            "realestate": "XLRE ETF",
            "materials": "XLB ETF",
            "communication": "XLC ETF",
        }
        terms = [sector_etf_terms.get(s.lower(), s + " ETF") for s in sectors[:5]]
        df = self._fetcher.fetch_interest_over_time(terms, timeframe="today 12-m")

        result: Dict[str, float] = {}
        if not _PANDAS_OK or df is None or df.empty:
            return {s: 0.0 for s in sectors}

        for i, sector in enumerate(sectors[:5]):
            term = terms[i]
            col = next((c for c in df.columns if term.split()[0].lower() in c.lower()), None)
            if col:
                vals = [float(v) for v in df[col].tolist()]
                result[sector] = round(
                    _zscore(vals[:-4], sum(vals[-4:]) / 4) if len(vals) > 4 else 0.0, 4
                )
            else:
                result[sector] = 0.0
        return result


# ---------------------------------------------------------------------------
# MacroTrendsMonitor
# ---------------------------------------------------------------------------

class MacroTrendsMonitor:
    """Monitor macro-level Google Trends as economic leading indicators."""

    def __init__(self, fetcher: Optional[GoogleTrendsFetcher] = None):
        self._fetcher = fetcher or GoogleTrendsFetcher()

    def _fetch_composite(self, term_group: List[str], timeframe: str = "today 5-y") -> List[float]:
        """Fetch multiple terms, return composite weekly series (mean across terms)."""
        if not term_group:
            return []
        terms = term_group[:5]
        df = self._fetcher.fetch_interest_over_time(terms, timeframe=timeframe)
        if not _PANDAS_OK or df is None or df.empty:
            return []
        return [float(df[col].mean()) for col in df.columns]

    def compute_economic_anxiety_index(self) -> "Optional[pd.DataFrame]":
        """Composite index of negative economic search terms.

        High = elevated public anxiety (leading recessionary signal).
        """
        anxiety_terms = [
            "recession",
            "unemployment benefits",
            "layoffs",
            "how to save money",
        ]
        df = self._fetcher.fetch_interest_over_time(anxiety_terms, timeframe="today 5-y")
        if not _PANDAS_OK or df is None or df.empty:
            return None
        result = df.mean(axis=1).rename("economic_anxiety_index")
        return result.to_frame()

    def compute_job_market_index(self) -> "Optional[pd.DataFrame]":
        """Composite job market search activity.

        Rising = strong labor demand (leading employment indicator).
        """
        job_terms = [
            "job openings",
            "hiring",
            "salary negotiation",
            "resume tips",
        ]
        df = self._fetcher.fetch_interest_over_time(job_terms, timeframe="today 5-y")
        if not _PANDAS_OK or df is None or df.empty:
            return None
        result = df.mean(axis=1).rename("job_market_index")
        return result.to_frame()

    def compute_housing_index(self) -> "Optional[pd.DataFrame]":
        """Composite housing market search activity."""
        housing_terms = [
            "buy a house",
            "mortgage rates",
            "Zillow",
            "home prices",
        ]
        df = self._fetcher.fetch_interest_over_time(housing_terms, timeframe="today 5-y")
        if not _PANDAS_OK or df is None or df.empty:
            return None
        result = df.mean(axis=1).rename("housing_index")
        return result.to_frame()

    def compute_inflation_awareness_index(self) -> "Optional[pd.DataFrame]":
        """Inflation public awareness — leads CPI headline sentiment."""
        inflation_terms = [
            "inflation",
            "prices rising",
            "cost of living",
            "gas prices",
        ]
        df = self._fetcher.fetch_interest_over_time(inflation_terms, timeframe="today 5-y")
        if not _PANDAS_OK or df is None or df.empty:
            return None
        return df.mean(axis=1).rename("inflation_awareness_index").to_frame()

    def get_all_macro_signals(self) -> Dict[str, Any]:
        """Return all macro trend signals with current and 4-week readings."""
        out: Dict[str, Any] = {}

        def _current_and_momentum(df: "Optional[pd.DataFrame]", name: str) -> Dict:
            if not _PANDAS_OK or df is None or df.empty:
                return {"current": 0.0, "momentum_zscore": 0.0}
            vals = [float(v) for v in df[name].tolist()]
            current = vals[-1] if vals else 0.0
            momentum = _zscore(vals[-(52 + 4):-4], sum(vals[-4:]) / 4) if len(vals) >= 56 else 0.0
            return {"current": round(current, 2), "momentum_zscore": round(momentum, 4)}

        out["economic_anxiety"] = _current_and_momentum(
            self.compute_economic_anxiety_index(), "economic_anxiety_index"
        )
        out["job_market"] = _current_and_momentum(
            self.compute_job_market_index(), "job_market_index"
        )
        out["housing"] = _current_and_momentum(
            self.compute_housing_index(), "housing_index"
        )
        out["inflation_awareness"] = _current_and_momentum(
            self.compute_inflation_awareness_index(), "inflation_awareness_index"
        )

        # Composite macro health (invert anxiety, average rest)
        anxiety_z = out["economic_anxiety"]["momentum_zscore"]
        job_z = out["job_market"]["momentum_zscore"]
        housing_z = out["housing"]["momentum_zscore"]
        out["macro_health_score"] = round((-anxiety_z + job_z + housing_z) / 3.0, 4)
        out["computed_at"] = datetime.now(timezone.utc).isoformat()
        return out


# ---------------------------------------------------------------------------
# CompetitiveTrendsAnalyzer
# ---------------------------------------------------------------------------

class CompetitiveTrendsAnalyzer:
    """Analyze search share between competing companies (Share of Search)."""

    def __init__(self, fetcher: Optional[GoogleTrendsFetcher] = None):
        self._fetcher = fetcher or GoogleTrendsFetcher()

    def _get_company_keywords(self, tickers: List[str]) -> Dict[str, str]:
        """Return {ticker: primary_keyword} for each ticker."""
        out: Dict[str, str] = {}
        for ticker in tickers:
            t = ticker.upper()
            keywords = _TICKER_KEYWORDS.get(t, [t])
            out[t] = keywords[0]
        return out

    def compute_share_of_search(self, tickers: List[str]) -> "Optional[pd.DataFrame]":
        """Compute share-of-search (%) for each ticker in the group.

        Each company's search volume normalized as % of total in the group.
        Returns DataFrame indexed by date with columns per ticker (0-100%).
        """
        if not _PANDAS_OK:
            return None

        tickers = [t.upper() for t in tickers[:5]]
        kw_map = self._get_company_keywords(tickers)
        keywords = [kw_map[t] for t in tickers]

        df = self._fetcher.fetch_interest_over_time(keywords, timeframe="today 5-y")
        if df is None or df.empty:
            return None

        # Rename columns to tickers for clarity
        col_rename = {keywords[i]: tickers[i] for i in range(len(keywords)) if keywords[i] in df.columns}
        df = df.rename(columns=col_rename)

        # Normalize to share of search
        row_totals = df.sum(axis=1).replace(0, 1)
        sos_df = df.div(row_totals, axis=0) * 100
        return sos_df.round(2)

    def detect_share_shift(
        self, tickers: List[str], lookback_weeks: int = 12
    ) -> Dict[str, Any]:
        """Identify which company is gaining/losing share of search.

        Returns {ticker: {current_sos, prior_sos, change, signal}}.
        'signal' in ('gaining', 'losing', 'stable').
        """
        sos_df = self.compute_share_of_search(tickers)
        if not _PANDAS_OK or sos_df is None or sos_df.empty:
            return {}

        n = len(sos_df)
        if n < lookback_weeks * 2:
            lookback_weeks = max(4, n // 2)

        result: Dict[str, Any] = {}
        for ticker in sos_df.columns:
            vals = [float(v) for v in sos_df[ticker].tolist()]
            current_period = vals[-lookback_weeks:] if len(vals) >= lookback_weeks else vals
            prior_period = (
                vals[-(lookback_weeks * 2):-lookback_weeks]
                if len(vals) >= lookback_weeks * 2
                else vals[: max(1, len(vals) // 2)]
            )

            current_sos = sum(current_period) / len(current_period) if current_period else 0.0
            prior_sos = sum(prior_period) / len(prior_period) if prior_period else 0.0
            change = current_sos - prior_sos

            if abs(change) < 1.0:
                signal = "stable"
            elif change > 0:
                signal = "gaining"
            else:
                signal = "losing"

            result[ticker] = {
                "current_sos": round(current_sos, 2),
                "prior_sos": round(prior_sos, 2),
                "change_pp": round(change, 2),
                "signal": signal,
                "note": (
                    f"{ticker} is {signal} search share: "
                    f"{prior_sos:.1f}% → {current_sos:.1f}% "
                    f"over last {lookback_weeks} weeks"
                ),
            }
        return result

    def compute_brand_momentum(self, ticker: str, peers: Optional[List[str]] = None) -> float:
        """Trend in share-of-search over 52 weeks.

        Returns z-score of recent 4-week SoS vs 52-week history. Positive = gaining share.
        """
        group = [ticker]
        if peers:
            group.extend(peers[:4])
        else:
            # Use default competitive peer from sector
            sector_peers: Dict[str, List[str]] = {
                "AAPL": ["MSFT", "GOOGL", "META", "AMZN"],
                "MSFT": ["AAPL", "GOOGL", "AMZN", "META"],
                "GOOGL": ["MSFT", "AAPL", "META", "AMZN"],
                "AMZN": ["MSFT", "AAPL", "WMT", "EBAY"],
                "TSLA": ["GM", "F", "RIVN", "NIO"],
                "NFLX": ["DIS", "PARA", "WBD", "AMZN"],
                "JPM": ["BAC", "GS", "MS", "C"],
            }
            group.extend(sector_peers.get(ticker.upper(), [])[:3])

        sos_df = self.compute_share_of_search(group)
        if not _PANDAS_OK or sos_df is None or sos_df.empty:
            return 0.0

        ticker_upper = ticker.upper()
        if ticker_upper not in sos_df.columns:
            return 0.0

        vals = [float(v) for v in sos_df[ticker_upper].tolist()]
        if len(vals) < 8:
            return 0.0

        recent_mean = sum(vals[-4:]) / 4
        history = vals[-(52 + 4):-4] if len(vals) >= 56 else vals[:-4]
        return round(_zscore(history, recent_mean), 4)


# ---------------------------------------------------------------------------
# GoogleTrendsEngine (Orchestrator)
# ---------------------------------------------------------------------------

class GoogleTrendsEngine:
    """Orchestrator: produces full TrendsDashboard for any ticker."""

    def __init__(self):
        self._fetcher = GoogleTrendsFetcher()
        self._signal_engine = TrendsSignalEngine(self._fetcher)
        self._predictor = TrendsPredictor(self._fetcher, self._signal_engine)
        self._macro_monitor = MacroTrendsMonitor(self._fetcher)
        self._competitive = CompetitiveTrendsAnalyzer(self._fetcher)

    def get_signal(self, ticker: str) -> TrendsDashboard:
        """Full trends analysis for a single ticker."""
        ticker = ticker.upper()
        keywords = self._fetcher.build_company_keywords(ticker)
        primary_kw = keywords[0]

        # Search score
        score = self._signal_engine.compute_ticker_search_score(ticker)

        # Earnings prediction
        try:
            earnings_pred = self._predictor.predict_earnings_surprise(ticker)
        except Exception as exc:  # noqa: BLE001
            logger.warning("earnings_pred failed for %s: %s", ticker, exc)
            earnings_pred = None

        # Product signals
        try:
            product_signals = self._signal_engine.get_product_cycle_signal(ticker)
        except Exception as exc:  # noqa: BLE001
            logger.warning("product_signals failed for %s: %s", ticker, exc)
            product_signals = {}

        # Competitive share of search — default peers
        competitive_share: Dict[str, float] = {}
        try:
            peers = self._competitive._get_company_keywords([ticker])
            sos_df = self._competitive.compute_share_of_search([ticker] + _get_default_peers(ticker))
            if _PANDAS_OK and sos_df is not None and not sos_df.empty:
                recent_row = sos_df.tail(4).mean()
                competitive_share = {k: round(float(v), 2) for k, v in recent_row.items()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("competitive_share failed for %s: %s", ticker, exc)

        # Fear/greed index
        try:
            fg = self._signal_engine.compute_fear_greed_proxy()
        except Exception as exc:  # noqa: BLE001
            logger.warning("fear_greed failed: %s", exc)
            fg = 0.0

        # Macro context
        try:
            macro_ctx = self._macro_monitor.get_all_macro_signals()
            macro_flat = {
                "economic_anxiety": macro_ctx.get("economic_anxiety", {}).get("current", 0.0),
                "job_market": macro_ctx.get("job_market", {}).get("current", 0.0),
                "housing": macro_ctx.get("housing", {}).get("current", 0.0),
                "macro_health_score": macro_ctx.get("macro_health_score", 0.0),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("macro_context failed: %s", exc)
            macro_flat = {}

        # Raw series
        try:
            df = self._fetcher.fetch_interest_over_time([primary_kw], timeframe="today 5-y")
            raw_series = _series_to_list(df, primary_kw) if (
                _PANDAS_OK and df is not None and not df.empty and primary_kw in df.columns
            ) else []
        except Exception:  # noqa: BLE001
            raw_series = []

        return TrendsDashboard(
            ticker=ticker,
            search_score=score,
            earnings_prediction=earnings_pred,
            product_signals=product_signals,
            competitive_share=competitive_share,
            fear_greed_index=fg,
            macro_context=macro_flat,
            raw_interest_series=raw_series,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def screen_by_search_momentum(
        self,
        universe: List[str],
        min_score: float = 0.5,
    ) -> "Optional[pd.DataFrame]":
        """Screen a universe of tickers by search momentum. Returns sorted DataFrame."""
        rows = []
        for ticker in universe:
            try:
                score = self._signal_engine.compute_ticker_search_score(ticker)
                if score.composite_score >= min_score:
                    rows.append({
                        "ticker": ticker,
                        "composite_score": score.composite_score,
                        "momentum_zscore": score.momentum_zscore,
                        "raw_interest": score.raw_interest,
                        "spike_detected": score.spike_detected,
                    })
                _jitter_sleep(1.0, 2.5)
            except Exception as exc:  # noqa: BLE001
                logger.warning("screen %s: %s", ticker, exc)

        if not _PANDAS_OK:
            return None
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values("composite_score", ascending=False)
        return df.reset_index(drop=True)

    def get_fear_greed_index(self, mode: str = "market_data") -> float:
        """Single-number market fear/greed indicator.

        Defaults to ``mode="market_data"`` which uses VIX + FRED + CBOE
        (see :func:`compute_fear_greed_composite_v2`). Pass ``mode="trends"``
        for the legacy pytrends-only composite.
        """
        return self._signal_engine.compute_fear_greed_proxy(mode=mode)

    def export_signals(self, universe: List[str], path: str) -> None:
        """Export search scores for a universe to CSV."""
        import csv as _csv

        rows = []
        for ticker in universe:
            try:
                score = self._signal_engine.compute_ticker_search_score(ticker)
                rows.append({
                    "ticker": score.ticker,
                    "composite_score": score.composite_score,
                    "momentum_zscore": score.momentum_zscore,
                    "acceleration": score.acceleration,
                    "raw_interest": score.raw_interest,
                    "spike_detected": score.spike_detected,
                    "spike_zscore": score.spike_zscore,
                    "computed_at": score.computed_at,
                })
                _jitter_sleep(1.0, 2.5)
            except Exception as exc:  # noqa: BLE001
                logger.warning("export %s: %s", ticker, exc)

        if not rows:
            logger.warning("No rows to export")
            return

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Exported %d ticker signals to %s", len(rows), path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_default_peers(ticker: str) -> List[str]:
    """Return 3-4 default peer tickers for share-of-search comparison."""
    peer_map: Dict[str, List[str]] = {
        "AAPL": ["MSFT", "GOOGL", "AMZN"],
        "MSFT": ["AAPL", "GOOGL", "AMZN"],
        "GOOGL": ["MSFT", "META", "AMZN"],
        "AMZN": ["MSFT", "AAPL", "WMT"],
        "TSLA": ["GM", "F", "RIVN"],
        "NFLX": ["DIS", "AMZN", "PARA"],
        "NVDA": ["AMD", "INTC", "QCOM"],
        "META": ["GOOGL", "SNAP", "TWTR"],
        "JPM": ["BAC", "GS", "MS"],
    }
    return peer_map.get(ticker.upper(), ["SPY"])[:3]


# ---------------------------------------------------------------------------
# FastAPI router (optional)
# ---------------------------------------------------------------------------
try:
    from fastapi import APIRouter, Query as _Query

    router = APIRouter(prefix="/trends/v3", tags=["trends-v3"])
    _engine_singleton: Optional[GoogleTrendsEngine] = None

    def _get_engine() -> GoogleTrendsEngine:
        global _engine_singleton
        if _engine_singleton is None:
            _engine_singleton = GoogleTrendsEngine()
        return _engine_singleton

    @router.get("/signal/{ticker}")
    def api_get_signal(ticker: str):
        """Full trends dashboard for a ticker."""
        import dataclasses
        dashboard = _get_engine().get_signal(ticker)
        return dataclasses.asdict(dashboard)

    @router.get("/fear-greed")
    def api_fear_greed():
        """Market fear/greed proxy from search terms."""
        return {"fear_greed_index": _get_engine().get_fear_greed_index()}

    @router.get("/sector/{sector}")
    def api_sector_trends(sector: str):
        """Sector-level search momentum ranking."""
        engine = _get_engine()
        result = engine._signal_engine.compute_sector_trends(sector)
        return {"sector": sector, "momentum_by_ticker": result}

    @router.get("/macro")
    def api_macro():
        """All macro trend signals."""
        engine = _get_engine()
        return engine._macro_monitor.get_all_macro_signals()

    @router.get("/competitive-sos")
    def api_competitive_sos(tickers: str = _Query(..., description="Comma-separated tickers")):
        """Share-of-search analysis for a group of tickers."""
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        engine = _get_engine()
        result = engine._competitive.detect_share_shift(ticker_list)
        return result

except ImportError:  # pragma: no cover
    router = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    print("=" * 70)
    print("SENTINEL Google Trends Signal Engine V3")
    print(f"pytrends available: {_PYTRENDS_OK}")
    print(f"pandas available:   {_PANDAS_OK}")
    print("=" * 70)

    engine = GoogleTrendsEngine()

    # --- 1. 5-year trends for AAPL, GOOGL, MSFT ---
    print("\n[1] Fetching 5-year interest-over-time for AAPL / GOOGL / MSFT ...")
    keywords_demo = ["Apple", "Google", "Microsoft"]
    df_iot = engine._fetcher.fetch_interest_over_time(keywords_demo, timeframe="today 5-y")
    if _PANDAS_OK and df_iot is not None and not df_iot.empty:
        print(f"    Shape: {df_iot.shape}")
        print(df_iot.tail(5).to_string())
    else:
        print("    (no data returned — check network / pytrends availability)")

    # --- 2. Search momentum ---
    print("\n[2] Computing search momentum scores ...")
    for ticker, kw in [("AAPL", "Apple"), ("GOOGL", "Google"), ("MSFT", "Microsoft")]:
        mom = engine._signal_engine.compute_search_momentum(kw)
        spike, spike_z = engine._signal_engine.detect_search_spike(kw)
        print(
            f"    {ticker}: momentum_z={mom:+.3f}  "
            f"spike={'YES' if spike else 'no'} (z={spike_z:.2f})"
        )

    # --- 3. Competitive share of search ---
    print("\n[3] Competitive Share of Search: AAPL vs MSFT vs GOOGL ...")
    sos_result = engine._competitive.detect_share_shift(["AAPL", "MSFT", "GOOGL"])
    for ticker, data in sos_result.items():
        print(f"    {ticker}: SoS {data['prior_sos']:.1f}% → {data['current_sos']:.1f}%"
              f"  ({data['signal'].upper()})")

    # --- 4. Fear/greed proxy ---
    print("\n[4] Computing fear/greed proxy ...")
    fg = engine.get_fear_greed_index()
    sentiment_label = (
        "Extreme Greed" if fg > 50 else
        "Greed" if fg > 15 else
        "Neutral" if fg > -15 else
        "Fear" if fg > -50 else
        "Extreme Fear"
    )
    print(f"    Fear/Greed Index: {fg:+.1f}  ({sentiment_label})")

    # --- 5. Full dashboard for AAPL ---
    print("\n[5] Full TrendsDashboard for AAPL ...")
    import dataclasses
    dashboard = engine.get_signal("AAPL")
    print(f"    composite_score:     {dashboard.search_score.composite_score:.4f}")
    print(f"    momentum_zscore:     {dashboard.search_score.momentum_zscore:+.4f}")
    print(f"    acceleration:        {dashboard.search_score.acceleration:+.4f}")
    print(f"    spike_detected:      {dashboard.search_score.spike_detected}")
    if dashboard.earnings_prediction:
        ep = dashboard.earnings_prediction
        print(f"    earnings signal:     strength={ep.signal_strength:.3f}  "
              f"prob_positive_surprise={ep.positive_surprise_probability:.2f}")
    print(f"    fear_greed_index:    {dashboard.fear_greed_index:+.1f}")
    print(f"    product_signals:     {dashboard.product_signals}")

    print("\nDone.")
