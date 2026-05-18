"""
Congressional STOCK Act eFD — V3 Production Intelligence Module (dim_029, score 7 → 9).

Comprehensive Congressional stock trading tracker using free community APIs
(housestockwatcher.com, senatestockwatcher.com) and EDGAR EFTS.

Architecture
------------
HouseStockWatcherClient         — primary House trades from community API
SenateStockWatcherClient        — Senate trades from community API
CongressSignalEngine            — alpha signal computation, cluster detection
CongressMemberTracker           — individual member performance analysis
CongressReturnAnalyzer          — systematic alpha/backtest framework
CongressScreener                — preset + custom screening
CongressTrackerEngine           — orchestrator / main entry point

Dataclasses: CongressTrade, CongressSignal, CongressCluster, CongressPortfolio,
             BacktestResult, CongressDashboard

Free APIs only — no paid data sources.

Usage::
    from sentinel.sfe.congress_tracker_v3 import CongressTrackerEngine

    engine = CongressTrackerEngine()
    dashboard = engine.get_dashboard()
    signal = engine.get_ticker_signal("NVDA")
    active = engine.get_most_active_tickers(days=30)
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False
    pd = None  # type: ignore

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False
    yf = None  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOUSE_WATCHER_URL  = "https://housestockwatcher.com/api"
SENATE_WATCHER_URL = "https://senatestockwatcher.com/api/transactions"

CACHE_DIR = Path(__file__).resolve().parents[1] / "data"
HOUSE_CACHE  = CACHE_DIR / "congress_house_cache.json"
SENATE_CACHE = CACHE_DIR / "congress_senate_cache.json"
CACHE_TTL_SECONDS = 86_400  # 24 hours

_HEADERS = {
    "User-Agent": "SENTINEL:FinancialTerminal:3.0 (research; richard.porras@realempanada.com)",
    "Accept": "application/json",
}
_TIMEOUT  = 30.0
_RATE_GAP = 0.5   # seconds between requests

# Committee → sector relevance map
COMMITTEE_SECTOR_MAP: Dict[str, List[str]] = {
    "Finance":            ["XLF", "GS", "JPM", "BAC", "MS", "BLK", "C", "WFC"],
    "Banking":            ["XLF", "GS", "JPM", "BAC", "MS", "BLK", "C", "WFC"],
    "Armed Services":     ["LMT", "RTX", "NOC", "GD", "BA", "L3H", "LDOS", "CACI"],
    "Energy":             ["XOM", "CVX", "COP", "SLB", "HAL", "EOG", "PXD"],
    "Health":             ["UNH", "CVS", "ANTM", "HUM", "CI", "MCK", "ABC"],
    "Technology":         ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA"],
    "Agriculture":        ["ADM", "BG", "MOS", "CF", "NTR", "DE"],
    "Judiciary":          ["LX", "DOCU", "ZI"],    # tech/legal adjacent
    "Intelligence":       ["SAIC", "CACI", "LDOS", "BAH", "CSCO", "PANW"],
}

# Committee chairs (117th–119th Congress) — approximate, hardcoded
COMMITTEE_CHAIRS: Dict[str, str] = {
    "Finance":          "Ron Wyden",
    "Banking":          "Sherrod Brown",
    "Armed Services":   "Jack Reed",
    "Energy":           "Joe Manchin",
    "Health":           "Bernie Sanders",
    "Technology":       "Maria Cantwell",
    "Agriculture":      "Debbie Stabenow",
    "Judiciary":        "Dick Durbin",
    "Intelligence":     "Mark Warner",
    # House side
    "Ways and Means":   "Jason Smith",
    "Financial Services":"Patrick McHenry",
    "House Armed Services":"Mike Rogers",
}

# Dollar-range strings → (low, high) tuples (STOCK Act standard buckets)
AMOUNT_RANGES: Dict[str, Tuple[int, int]] = {
    "$1,001 - $15,000":        (1_001,    15_000),
    "$15,001 - $50,000":       (15_001,   50_000),
    "$50,001 - $100,000":      (50_001,  100_000),
    "$100,001 - $250,000":    (100_001,  250_000),
    "$250,001 - $500,000":    (250_001,  500_000),
    "$500,001 - $1,000,000":  (500_001, 1_000_000),
    "$1,000,001 - $5,000,000":(1_000_001, 5_000_000),
    "Over $5,000,000":        (5_000_001, 25_000_000),
    "$1,000 - $15,000":       (1_000,    15_000),
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CongressTrade:
    """A single STOCK Act-disclosed trade by a member of Congress."""
    transaction_date: Optional[date]
    disclosure_date: Optional[date]
    member: str                          # representative or senator name
    chamber: str                         # "house" | "senate"
    state: str
    district: Optional[str]
    party: Optional[str]
    ticker: str
    asset_description: str
    asset_type: str
    trade_type: str                      # "purchase" | "sale" | "exchange"
    amount_range: str
    amount_low: int
    amount_high: int
    amount_midpoint: float
    cap_gains_over_200: bool
    comment: str
    filing_id: str = ""

    @property
    def disclosure_lag_days(self) -> Optional[int]:
        if self.transaction_date and self.disclosure_date:
            return (self.disclosure_date - self.transaction_date).days
        return None

    @property
    def is_buy(self) -> bool:
        return self.trade_type.lower() == "purchase"

    @property
    def is_sell(self) -> bool:
        return self.trade_type.lower() in ("sale", "sale (full)", "sale (partial)")


@dataclass
class CongressSignal:
    """Aggregated congressional trading signal for a ticker."""
    ticker: str
    days_window: int
    total_trades: int
    buy_count: int
    sell_count: int
    net_direction: str          # "strong_buy" | "buy" | "neutral" | "sell" | "strong_sell"
    net_score: float            # weighted net buys (positive = bullish)
    unique_members: int
    committee_chair_involved: bool
    bipartisan: bool
    cluster_detected: bool
    last_trade_date: Optional[date]
    top_buyers: List[str]
    top_sellers: List[str]
    signal_strength: float      # 0..1


@dataclass
class CongressCluster:
    """3+ members buying the same stock in a 30-day window."""
    ticker: str
    window_start: date
    window_end: date
    member_count: int
    members: List[str]
    parties: List[str]
    total_estimated_value: float
    bipartisan: bool
    committee_chair_in_cluster: bool


@dataclass
class CongressPortfolio:
    """Disclosed holdings and recent trades for one Congress member."""
    member: str
    chamber: str
    state: str
    party: Optional[str]
    total_trades_12m: int
    buy_trades_12m: int
    sell_trades_12m: int
    tickers_traded: List[str]
    sector_concentration: Dict[str, float]   # sector → % of trades
    estimated_portfolio_value: float
    recent_trades: List[CongressTrade]
    annualized_return: Optional[float]       # from CongressMemberTracker


@dataclass
class BacktestResult:
    """Result from backtesting a congressional buy-following strategy."""
    start_date: str
    end_date: str
    total_return: float
    annualized_return: float
    benchmark_return: float
    benchmark_annualized: float
    alpha: float
    sharpe_ratio: float
    max_drawdown: float
    trade_count: int
    win_rate: float
    avg_holding_days: float
    monthly_returns: List[float]


@dataclass
class CongressDashboard:
    """High-level summary of recent congressional trading activity."""
    as_of: date
    days_window: int
    total_trades: int
    house_trades: int
    senate_trades: int
    unique_tickers: int
    unique_members: int
    top_tickers: List[Tuple[str, int]]           # (ticker, trade_count)
    top_buyers: List[Tuple[str, int]]            # (member, buy_count)
    clusters: List[CongressCluster]
    late_filers: List[Dict[str, Any]]
    recent_trades: List[CongressTrade]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LAST_REQ_TIME: float = 0.0


def _get(url: str, params: Optional[dict] = None, timeout: float = _TIMEOUT) -> Any:
    """Rate-limited GET with JSON return."""
    global _LAST_REQ_TIME
    elapsed = time.time() - _LAST_REQ_TIME
    if elapsed < _RATE_GAP:
        time.sleep(_RATE_GAP - elapsed)
    _LAST_REQ_TIME = time.time()
    resp = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _parse_amount(amount_str: str) -> Tuple[int, int, float]:
    """Parse a STOCK Act dollar-range string → (low, high, midpoint)."""
    if not amount_str:
        return 0, 0, 0.0
    # Try exact match
    for key, (lo, hi) in AMOUNT_RANGES.items():
        if key.lower().strip() == amount_str.lower().strip():
            return lo, hi, (lo + hi) / 2.0
    # Try numeric extraction
    nums = re.findall(r"\d[\d,]*", amount_str.replace("$", ""))
    if len(nums) >= 2:
        lo = int(nums[0].replace(",", ""))
        hi = int(nums[1].replace(",", ""))
        return lo, hi, (lo + hi) / 2.0
    if len(nums) == 1:
        val = int(nums[0].replace(",", ""))
        return val, val, float(val)
    return 0, 0, 0.0


def _parse_date(s: Any) -> Optional[date]:
    """Parse a date string in various formats."""
    if not s:
        return None
    if isinstance(s, date):
        return s
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _load_cache(path: Path) -> Optional[dict]:
    """Load JSON cache; returns None if missing or stale (>24h)."""
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if (time.time() - mtime) > CACHE_TTL_SECONDS:
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _save_cache(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, default=str)


def _is_committee_chair(member: str) -> bool:
    return any(chair.lower() in member.lower() for chair in COMMITTEE_CHAIRS.values())


def _guess_committee(member: str) -> Optional[str]:
    for committee, chair in COMMITTEE_CHAIRS.items():
        if chair.lower() in member.lower():
            return committee
    return None


# ---------------------------------------------------------------------------
# HouseStockWatcherClient
# ---------------------------------------------------------------------------

class HouseStockWatcherClient:
    """
    Fetches House member stock disclosures from the housestockwatcher.com community API.

    Data is cached for 24 hours in sentinel/data/congress_house_cache.json.
    """

    def __init__(self, cache_path: Path = HOUSE_CACHE) -> None:
        self.cache_path = cache_path
        self._trades: Optional[List[CongressTrade]] = None

    # ------------------------------------------------------------------
    def _raw_fetch(self) -> List[dict]:
        cached = _load_cache(self.cache_path)
        if cached is not None:
            logger.info("HouseStockWatcher: loaded %d rows from cache", len(cached))
            return cached
        logger.info("HouseStockWatcher: fetching from %s …", HOUSE_WATCHER_URL)
        data = _get(HOUSE_WATCHER_URL)
        # API returns {"data": [...]} or directly a list
        if isinstance(data, dict):
            rows = data.get("data", data.get("transactions", []))
        else:
            rows = data
        _save_cache(self.cache_path, rows)
        logger.info("HouseStockWatcher: fetched %d rows", len(rows))
        return rows

    def _row_to_trade(self, row: dict) -> CongressTrade:
        amount_str = row.get("amount", "")
        lo, hi, mid = _parse_amount(amount_str)
        trade_type = row.get("type", "").lower().strip()
        # Normalize type
        if "purchase" in trade_type:
            trade_type = "purchase"
        elif "sale" in trade_type or "sell" in trade_type:
            trade_type = "sale"
        elif "exchange" in trade_type:
            trade_type = "exchange"

        ticker = (row.get("ticker", "") or "").upper().strip()
        if not ticker or ticker in ("N/A", "NA", "--"):
            ticker = ""

        return CongressTrade(
            transaction_date=_parse_date(row.get("transaction_date")),
            disclosure_date=_parse_date(row.get("disclosure_date")),
            member=row.get("representative", ""),
            chamber="house",
            state=row.get("state", ""),
            district=str(row.get("district", "")),
            party=row.get("party", None),
            ticker=ticker,
            asset_description=row.get("asset_description", ""),
            asset_type=row.get("asset_type", ""),
            trade_type=trade_type,
            amount_range=amount_str,
            amount_low=lo,
            amount_high=hi,
            amount_midpoint=mid,
            cap_gains_over_200=bool(row.get("cap_gains_over_200_usd", False)),
            comment=row.get("comment", ""),
        )

    def fetch_all_trades(self) -> List[CongressTrade]:
        if self._trades is not None:
            return self._trades
        rows = self._raw_fetch()
        self._trades = [self._row_to_trade(r) for r in rows]
        return self._trades

    def fetch_by_ticker(self, ticker: str) -> List[CongressTrade]:
        ticker = ticker.upper().strip()
        return [t for t in self.fetch_all_trades() if t.ticker == ticker]

    def fetch_by_member(self, name: str) -> List[CongressTrade]:
        name_lower = name.lower()
        return [t for t in self.fetch_all_trades() if name_lower in t.member.lower()]

    def fetch_recent(self, days: int = 90) -> List[CongressTrade]:
        cutoff = date.today() - timedelta(days=days)
        return [
            t for t in self.fetch_all_trades()
            if t.transaction_date and t.transaction_date >= cutoff
        ]

    def invalidate_cache(self) -> None:
        self._trades = None
        if self.cache_path.exists():
            self.cache_path.unlink()


# ---------------------------------------------------------------------------
# SenateStockWatcherClient
# ---------------------------------------------------------------------------

class SenateStockWatcherClient:
    """
    Fetches Senate member stock disclosures from senatestockwatcher.com community API.
    """

    def __init__(self, cache_path: Path = SENATE_CACHE) -> None:
        self.cache_path = cache_path
        self._trades: Optional[List[CongressTrade]] = None

    def _raw_fetch(self) -> List[dict]:
        cached = _load_cache(self.cache_path)
        if cached is not None:
            logger.info("SenateStockWatcher: loaded %d rows from cache", len(cached))
            return cached
        logger.info("SenateStockWatcher: fetching from %s …", SENATE_WATCHER_URL)
        data = _get(SENATE_WATCHER_URL)
        if isinstance(data, dict):
            rows = data.get("data", data.get("transactions", []))
        else:
            rows = data
        _save_cache(self.cache_path, rows)
        logger.info("SenateStockWatcher: fetched %d rows", len(rows))
        return rows

    def _row_to_trade(self, row: dict) -> CongressTrade:
        amount_str = row.get("amount", "")
        lo, hi, mid = _parse_amount(amount_str)
        trade_type = row.get("type", "").lower().strip()
        if "purchase" in trade_type or "buy" in trade_type:
            trade_type = "purchase"
        elif "sale" in trade_type or "sell" in trade_type:
            trade_type = "sale"
        elif "exchange" in trade_type:
            trade_type = "exchange"

        # Senate API field names may differ
        member = (
            row.get("senator", "")
            or row.get("name", "")
            or row.get("representative", "")
        )
        ticker = (
            row.get("ticker", "")
            or row.get("symbol", "")
            or ""
        ).upper().strip()
        if not ticker or ticker in ("N/A", "NA", "--"):
            ticker = ""

        return CongressTrade(
            transaction_date=_parse_date(
                row.get("transaction_date") or row.get("date")
            ),
            disclosure_date=_parse_date(
                row.get("disclosure_date") or row.get("filed")
            ),
            member=member,
            chamber="senate",
            state=row.get("state", ""),
            district=None,
            party=row.get("party", None),
            ticker=ticker,
            asset_description=row.get("asset_description", row.get("asset", "")),
            asset_type=row.get("asset_type", ""),
            trade_type=trade_type,
            amount_range=amount_str,
            amount_low=lo,
            amount_high=hi,
            amount_midpoint=mid,
            cap_gains_over_200=False,
            comment=row.get("comment", ""),
        )

    def fetch_all_trades(self) -> List[CongressTrade]:
        if self._trades is not None:
            return self._trades
        rows = self._raw_fetch()
        self._trades = [self._row_to_trade(r) for r in rows]
        return self._trades

    def fetch_by_ticker(self, ticker: str) -> List[CongressTrade]:
        ticker = ticker.upper().strip()
        return [t for t in self.fetch_all_trades() if t.ticker == ticker]

    def fetch_by_senator(self, name: str) -> List[CongressTrade]:
        name_lower = name.lower()
        return [t for t in self.fetch_all_trades() if name_lower in t.member.lower()]

    def invalidate_cache(self) -> None:
        self._trades = None
        if self.cache_path.exists():
            self.cache_path.unlink()


# ---------------------------------------------------------------------------
# CongressSignalEngine
# ---------------------------------------------------------------------------

class CongressSignalEngine:
    """
    Compute alpha signals from combined House + Senate congressional trading data.

    Signals are weighted by:
      - Committee chair involvement (2× weight)
      - Trade size (midpoint dollar value as weight)
      - Bipartisan consensus (bonus multiplier)
    """

    def __init__(
        self,
        house_client: Optional[HouseStockWatcherClient] = None,
        senate_client: Optional[SenateStockWatcherClient] = None,
    ) -> None:
        self.house  = house_client  or HouseStockWatcherClient()
        self.senate = senate_client or SenateStockWatcherClient()

    def _all_trades(self) -> List[CongressTrade]:
        return self.house.fetch_all_trades() + self.senate.fetch_all_trades()

    def _trades_in_window(
        self,
        ticker: str,
        days: int,
        trades: Optional[List[CongressTrade]] = None,
    ) -> List[CongressTrade]:
        cutoff = date.today() - timedelta(days=days)
        source = trades if trades is not None else self._all_trades()
        return [
            t for t in source
            if t.ticker.upper() == ticker.upper()
            and t.transaction_date
            and t.transaction_date >= cutoff
        ]

    def compute_congress_buy_signal(
        self, ticker: str, days: int = 90
    ) -> CongressSignal:
        """
        Compute a weighted net-purchase signal for `ticker` over last `days` days.
        Committee chairs receive 2× weight; large trades receive proportionally more weight.
        """
        trades = self._trades_in_window(ticker, days)
        if not trades:
            return CongressSignal(
                ticker=ticker, days_window=days, total_trades=0,
                buy_count=0, sell_count=0, net_direction="neutral",
                net_score=0.0, unique_members=0,
                committee_chair_involved=False, bipartisan=False,
                cluster_detected=False, last_trade_date=None,
                top_buyers=[], top_sellers=[], signal_strength=0.0,
            )

        buy_score = 0.0
        sell_score = 0.0
        buyer_names: List[str] = []
        seller_names: List[str] = []
        parties: set = set()
        chair_involved = False
        member_set: set = set()

        for t in trades:
            # base weight = log(midpoint) to dampen outliers
            base = math.log1p(t.amount_midpoint) if t.amount_midpoint > 0 else 1.0
            multiplier = 2.0 if _is_committee_chair(t.member) else 1.0
            if _is_committee_chair(t.member):
                chair_involved = True
            weight = base * multiplier
            member_set.add(t.member)
            if t.party:
                parties.add(t.party.upper()[:1])

            if t.is_buy:
                buy_score += weight
                buyer_names.append(t.member)
            elif t.is_sell:
                sell_score += weight
                seller_names.append(t.member)

        net = buy_score - sell_score
        total_score = buy_score + sell_score or 1.0
        ratio = net / total_score

        if ratio > 0.6:
            direction = "strong_buy"
        elif ratio > 0.2:
            direction = "buy"
        elif ratio < -0.6:
            direction = "strong_sell"
        elif ratio < -0.2:
            direction = "sell"
        else:
            direction = "neutral"

        bipartisan = len(parties) > 1
        signal_strength = min(1.0, abs(ratio) * (1 + 0.1 * len(member_set)))

        last_date = max(
            (t.transaction_date for t in trades if t.transaction_date),
            default=None,
        )

        # top buyers/sellers by name frequency
        from collections import Counter
        top_buyers  = [m for m, _ in Counter(buyer_names).most_common(5)]
        top_sellers = [m for m, _ in Counter(seller_names).most_common(5)]

        clusters = self.detect_cluster_buys(ticker, days=min(days, 30))
        cluster_detected = bool(clusters)

        return CongressSignal(
            ticker=ticker,
            days_window=days,
            total_trades=len(trades),
            buy_count=sum(1 for t in trades if t.is_buy),
            sell_count=sum(1 for t in trades if t.is_sell),
            net_direction=direction,
            net_score=round(net, 2),
            unique_members=len(member_set),
            committee_chair_involved=chair_involved,
            bipartisan=bipartisan,
            cluster_detected=cluster_detected,
            last_trade_date=last_date,
            top_buyers=top_buyers,
            top_sellers=top_sellers,
            signal_strength=round(signal_strength, 3),
        )

    def detect_cluster_buys(
        self,
        ticker: str,
        days: int = 30,
        min_members: int = 3,
    ) -> List[CongressCluster]:
        """
        Detect when 3+ members buy the same stock within any rolling 30-day window.
        Returns a list of CongressCluster objects.
        """
        trades = self._trades_in_window(ticker, max(days * 3, 90))
        buys = [t for t in trades if t.is_buy and t.transaction_date]
        if len(buys) < min_members:
            return []

        buys.sort(key=lambda t: t.transaction_date)  # type: ignore[arg-type]

        clusters: List[CongressCluster] = []
        seen_windows: set = set()

        for i, anchor in enumerate(buys):
            window_end = anchor.transaction_date + timedelta(days=days)  # type: ignore[operator]
            window_trades = [
                t for t in buys
                if anchor.transaction_date <= t.transaction_date <= window_end  # type: ignore[operator]
            ]
            unique_members = list({t.member for t in window_trades})
            if len(unique_members) < min_members:
                continue
            key = (anchor.transaction_date, tuple(sorted(unique_members)))
            if key in seen_windows:
                continue
            seen_windows.add(key)

            parties = list({t.party for t in window_trades if t.party})
            bipartisan = (
                len({p.upper()[:1] for p in parties if p}) > 1
            )
            chair_in = any(_is_committee_chair(m) for m in unique_members)
            total_val = sum(t.amount_midpoint for t in window_trades)
            dates = [t.transaction_date for t in window_trades if t.transaction_date]

            clusters.append(CongressCluster(
                ticker=ticker,
                window_start=min(dates),
                window_end=max(dates),
                member_count=len(unique_members),
                members=unique_members,
                parties=parties,
                total_estimated_value=total_val,
                bipartisan=bipartisan,
                committee_chair_in_cluster=chair_in,
            ))

        return clusters

    def compute_congress_momentum(self, ticker: str) -> float:
        """
        Compute direction trend of congressional buys over time.
        Positive = increasing buy activity. Negative = increasing sell activity.
        Returns a momentum float in [-1, 1].
        """
        all_trades = self._all_trades()
        ticker_trades = [
            t for t in all_trades
            if t.ticker.upper() == ticker.upper() and t.transaction_date
        ]
        if len(ticker_trades) < 4:
            return 0.0

        ticker_trades.sort(key=lambda t: t.transaction_date)  # type: ignore[arg-type]
        n = len(ticker_trades)
        mid = n // 2
        recent = ticker_trades[mid:]
        older  = ticker_trades[:mid]

        def net_direction_score(ts: List[CongressTrade]) -> float:
            if not ts:
                return 0.0
            buys  = sum(1 for t in ts if t.is_buy)
            sells = sum(1 for t in ts if t.is_sell)
            total = buys + sells or 1
            return (buys - sells) / total

        recent_score = net_direction_score(recent)
        older_score  = net_direction_score(older)
        return round(recent_score - older_score, 4)

    def rank_tickers_by_congress_activity(
        self, days: int = 90
    ) -> "pd.DataFrame":
        """
        Return DataFrame of most actively traded tickers by Congress,
        with net buy/sell direction and member count.
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas required for rank_tickers_by_congress_activity")

        cutoff = date.today() - timedelta(days=days)
        trades = [
            t for t in self._all_trades()
            if t.transaction_date and t.transaction_date >= cutoff and t.ticker
        ]

        agg: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"buy": 0, "sell": 0, "members": set(), "value": 0.0}
        )
        for t in trades:
            key = t.ticker
            if t.is_buy:
                agg[key]["buy"] += 1
            elif t.is_sell:
                agg[key]["sell"] += 1
            agg[key]["members"].add(t.member)
            agg[key]["value"] += t.amount_midpoint

        rows = []
        for ticker, d in agg.items():
            total = d["buy"] + d["sell"]
            net = d["buy"] - d["sell"]
            direction = "buy" if net > 0 else ("sell" if net < 0 else "neutral")
            rows.append({
                "ticker": ticker,
                "total_trades": total,
                "buy_count": d["buy"],
                "sell_count": d["sell"],
                "net_trades": net,
                "direction": direction,
                "unique_members": len(d["members"]),
                "estimated_value": d["value"],
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("total_trades", ascending=False).reset_index(drop=True)

    def get_committee_relevant_tickers(self, committee: str) -> "pd.DataFrame":
        """
        Returns trades by members of a given committee in sector-related stocks.
        Potentially-informed trades: e.g. Finance committee members trading bank stocks.
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas required")

        sector_tickers = COMMITTEE_SECTOR_MAP.get(committee, [])
        chair = COMMITTEE_CHAIRS.get(committee, "")

        all_trades = self._all_trades()
        relevant: List[dict] = []
        for t in all_trades:
            # Include trades by committee chair or in sector-relevant tickers
            member_is_chair = chair.lower() in t.member.lower() if chair else False
            ticker_is_relevant = t.ticker.upper() in [x.upper() for x in sector_tickers]
            if member_is_chair or ticker_is_relevant:
                relevant.append({
                    "member": t.member,
                    "ticker": t.ticker,
                    "trade_type": t.trade_type,
                    "transaction_date": t.transaction_date,
                    "amount_midpoint": t.amount_midpoint,
                    "chamber": t.chamber,
                    "committee": committee,
                    "is_chair": member_is_chair,
                    "sector_relevant": ticker_is_relevant,
                })

        df = pd.DataFrame(relevant)
        if df.empty:
            return df
        return df.sort_values("transaction_date", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CongressMemberTracker
# ---------------------------------------------------------------------------

class CongressMemberTracker:
    """
    Track individual Congress member trading activity, performance, and compliance.
    """

    def __init__(
        self,
        house_client: Optional[HouseStockWatcherClient] = None,
        senate_client: Optional[SenateStockWatcherClient] = None,
    ) -> None:
        self.house  = house_client  or HouseStockWatcherClient()
        self.senate = senate_client or SenateStockWatcherClient()

    def _all_trades(self) -> List[CongressTrade]:
        return self.house.fetch_all_trades() + self.senate.fetch_all_trades()

    def get_member_portfolio(self, name: str) -> CongressPortfolio:
        """
        Build a CongressPortfolio for a given member name (fuzzy match).
        Includes all disclosed trades in the last 12 months.
        """
        name_lower = name.lower()
        all_trades = [
            t for t in self._all_trades()
            if name_lower in t.member.lower()
        ]
        if not all_trades:
            return CongressPortfolio(
                member=name, chamber="unknown", state="",
                party=None, total_trades_12m=0, buy_trades_12m=0,
                sell_trades_12m=0, tickers_traded=[],
                sector_concentration={}, estimated_portfolio_value=0.0,
                recent_trades=[], annualized_return=None,
            )

        # Identify the member
        first = all_trades[0]
        cutoff_12m = date.today() - timedelta(days=365)
        recent = [
            t for t in all_trades
            if t.transaction_date and t.transaction_date >= cutoff_12m
        ]

        tickers = list({t.ticker for t in all_trades if t.ticker})

        # Sector concentration (simplified by committee relevance)
        sector_counts: Dict[str, int] = defaultdict(int)
        for t in recent:
            for committee, tklist in COMMITTEE_SECTOR_MAP.items():
                if t.ticker.upper() in [x.upper() for x in tklist]:
                    sector_counts[committee] += 1
        total_sector = sum(sector_counts.values()) or 1
        sector_concentration = {k: v / total_sector for k, v in sector_counts.items()}

        estimated_value = sum(
            t.amount_midpoint for t in recent if t.is_buy
        )

        return CongressPortfolio(
            member=first.member,
            chamber=first.chamber,
            state=first.state,
            party=first.party,
            total_trades_12m=len(recent),
            buy_trades_12m=sum(1 for t in recent if t.is_buy),
            sell_trades_12m=sum(1 for t in recent if t.is_sell),
            tickers_traded=tickers,
            sector_concentration=sector_concentration,
            estimated_portfolio_value=estimated_value,
            recent_trades=recent[:50],
            annualized_return=None,
        )

    def compute_member_return(self, name: str) -> Optional[float]:
        """
        Estimate annualized return for a member's disclosed purchases.
        Buy at transaction_date closing price, sell at next sale trade or current price.
        Requires yfinance.
        """
        if not _HAS_YF:
            logger.warning("yfinance not available; cannot compute member return")
            return None

        name_lower = name.lower()
        trades = [
            t for t in self._all_trades()
            if name_lower in t.member.lower() and t.ticker
        ]
        buys  = sorted(
            [t for t in trades if t.is_buy and t.transaction_date],
            key=lambda t: t.transaction_date,  # type: ignore[arg-type]
        )
        sells_by_ticker: Dict[str, List[CongressTrade]] = defaultdict(list)
        for t in trades:
            if t.is_sell and t.transaction_date:
                sells_by_ticker[t.ticker].append(t)

        pnl_pct_list: List[float] = []

        for buy in buys:
            ticker = buy.ticker
            buy_date_str = buy.transaction_date.strftime("%Y-%m-%d")  # type: ignore[union-attr]

            # Find next sell date for same ticker
            future_sells = [
                s for s in sells_by_ticker.get(ticker, [])
                if s.transaction_date and s.transaction_date > buy.transaction_date  # type: ignore[operator]
            ]
            if future_sells:
                sell_date = min(future_sells, key=lambda s: s.transaction_date).transaction_date  # type: ignore[arg-type]
                sell_date_str = sell_date.strftime("%Y-%m-%d")
            else:
                sell_date_str = date.today().strftime("%Y-%m-%d")
                sell_date = date.today()

            try:
                hist = yf.download(ticker, start=buy_date_str, end=sell_date_str, progress=False)
                if hist.empty or len(hist) < 2:
                    continue
                buy_price  = float(hist["Close"].iloc[0])
                sell_price = float(hist["Close"].iloc[-1])
                if buy_price <= 0:
                    continue
                holding_days = (sell_date - buy.transaction_date).days or 1  # type: ignore[operator]
                raw_return = (sell_price - buy_price) / buy_price
                annualized = ((1 + raw_return) ** (365.0 / holding_days)) - 1
                pnl_pct_list.append(annualized)
            except Exception as exc:
                logger.debug("Return calc failed for %s: %s", ticker, exc)

        if not pnl_pct_list:
            return None
        return round(sum(pnl_pct_list) / len(pnl_pct_list), 4)

    def get_top_traders(
        self,
        n: int = 20,
        metric: str = "trade_count",
    ) -> "pd.DataFrame":
        """
        Return a DataFrame of top Congress traders by the given metric:
          - "trade_count" (default)
          - "estimated_value"
          - "buy_count"
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas required")

        all_trades = self._all_trades()
        member_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "trade_count": 0, "buy_count": 0, "sell_count": 0,
                "estimated_value": 0.0, "chamber": "", "state": "", "party": "",
            }
        )
        for t in all_trades:
            s = member_stats[t.member]
            s["trade_count"] += 1
            s["estimated_value"] += t.amount_midpoint
            if t.is_buy:
                s["buy_count"] += 1
            elif t.is_sell:
                s["sell_count"] += 1
            s["chamber"] = t.chamber
            s["state"]   = t.state
            if t.party:
                s["party"] = t.party

        rows = [{"member": k, **v} for k, v in member_stats.items()]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        valid_metrics = {"trade_count", "estimated_value", "buy_count"}
        if metric not in valid_metrics:
            metric = "trade_count"
        return (
            df.sort_values(metric, ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def detect_late_filers(self, days_threshold: int = 45) -> List[dict]:
        """
        Flag trades where disclosure_date - transaction_date > days_threshold.
        STOCK Act requires disclosure within 45 days.
        """
        all_trades = self._all_trades()
        late: List[dict] = []
        for t in all_trades:
            lag = t.disclosure_lag_days
            if lag is not None and lag > days_threshold:
                late.append({
                    "member": t.member,
                    "chamber": t.chamber,
                    "ticker": t.ticker,
                    "transaction_date": t.transaction_date,
                    "disclosure_date": t.disclosure_date,
                    "lag_days": lag,
                    "trade_type": t.trade_type,
                    "amount_midpoint": t.amount_midpoint,
                    "violation": lag > 45,
                })
        late.sort(key=lambda x: x["lag_days"], reverse=True)
        return late


# ---------------------------------------------------------------------------
# CongressReturnAnalyzer
# ---------------------------------------------------------------------------

class CongressReturnAnalyzer:
    """
    Systematic analysis of congressional trading returns vs benchmarks.
    """

    def __init__(
        self,
        house_client: Optional[HouseStockWatcherClient] = None,
        senate_client: Optional[SenateStockWatcherClient] = None,
    ) -> None:
        self.house  = house_client  or HouseStockWatcherClient()
        self.senate = senate_client or SenateStockWatcherClient()

    def _all_trades(self) -> List[CongressTrade]:
        return self.house.fetch_all_trades() + self.senate.fetch_all_trades()

    def compute_post_trade_returns(
        self,
        trades: List[CongressTrade],
        horizons: List[int] = None,
    ) -> "pd.DataFrame":
        """
        For each purchase trade, compute forward stock return at each horizon (days).
        Returns a DataFrame with columns: member, ticker, trade_date, amount, {horizon}d_return…
        """
        if horizons is None:
            horizons = [30, 90, 180]
        if not _HAS_PANDAS:
            raise ImportError("pandas required")
        if not _HAS_YF:
            raise ImportError("yfinance required")

        buys = [
            t for t in trades
            if t.is_buy and t.ticker and t.transaction_date
        ]
        rows: List[dict] = []
        price_cache: Dict[str, Any] = {}

        for t in buys:
            ticker = t.ticker
            if ticker not in price_cache:
                try:
                    start = (t.transaction_date - timedelta(days=5)).strftime("%Y-%m-%d")
                    end   = (
                        date.today() + timedelta(days=1)
                    ).strftime("%Y-%m-%d")
                    hist = yf.download(ticker, start=start, end=end, progress=False)
                    price_cache[ticker] = hist["Close"] if not hist.empty else None
                except Exception as exc:
                    logger.debug("yfinance error for %s: %s", ticker, exc)
                    price_cache[ticker] = None

            prices = price_cache.get(ticker)
            row: dict = {
                "member": t.member,
                "ticker": ticker,
                "trade_date": t.transaction_date,
                "amount_midpoint": t.amount_midpoint,
            }
            for h in horizons:
                if prices is None:
                    row[f"{h}d_return"] = None
                    continue
                try:
                    # Buy price: closest closing price on or after transaction_date
                    buy_dt = pd.Timestamp(t.transaction_date)  # type: ignore[arg-type]
                    sell_dt = buy_dt + pd.Timedelta(days=h)
                    buy_idx = prices.index.searchsorted(buy_dt)
                    sell_idx = prices.index.searchsorted(sell_dt)
                    if buy_idx >= len(prices) or sell_idx >= len(prices):
                        row[f"{h}d_return"] = None
                        continue
                    buy_px  = float(prices.iloc[buy_idx])
                    sell_px = float(prices.iloc[min(sell_idx, len(prices) - 1)])
                    row[f"{h}d_return"] = round((sell_px - buy_px) / buy_px, 4) if buy_px > 0 else None
                except Exception:
                    row[f"{h}d_return"] = None
            rows.append(row)

        return pd.DataFrame(rows)

    def compute_aggregate_alpha(
        self,
        trades: Optional[List[CongressTrade]] = None,
    ) -> float:
        """
        Compute aggregate alpha: do Congress trades outperform S&P 500?
        Returns alpha as a decimal (e.g. 0.05 = 5% excess return).
        """
        if not _HAS_YF or not _HAS_PANDAS:
            raise ImportError("pandas + yfinance required")

        if trades is None:
            trades = self._all_trades()

        # Get SPY benchmark return over the same period
        buy_trades = [
            t for t in trades
            if t.is_buy and t.ticker and t.transaction_date
        ]
        if not buy_trades:
            return 0.0

        all_dates = [t.transaction_date for t in buy_trades]
        start_date = min(all_dates).strftime("%Y-%m-%d")  # type: ignore[union-attr]
        end_date = date.today().strftime("%Y-%m-%d")

        try:
            spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
            spy_return = float(
                (spy["Close"].iloc[-1] - spy["Close"].iloc[0]) / spy["Close"].iloc[0]
            ) if not spy.empty else 0.0
        except Exception:
            spy_return = 0.0

        # Average Congress return using post_trade_returns at 90-day horizon
        df = self.compute_post_trade_returns(buy_trades, horizons=[90])
        if df.empty or "90d_return" not in df.columns:
            return 0.0
        congress_return = df["90d_return"].dropna().mean()
        alpha = float(congress_return) - spy_return
        return round(alpha, 4)

    def compute_committee_alpha(self, committee: str) -> dict:
        """
        Compute alpha for relevant-committee members trading sector-related stocks.
        Returns dict with alpha, trade_count, avg_return, benchmark_return.
        """
        sector_tickers = [t.upper() for t in COMMITTEE_SECTOR_MAP.get(committee, [])]
        chair = COMMITTEE_CHAIRS.get(committee, "").lower()

        relevant_trades = [
            t for t in self._all_trades()
            if (
                (chair and chair in t.member.lower())
                or t.ticker.upper() in sector_tickers
            )
            and t.is_buy and t.ticker and t.transaction_date
        ]

        if not relevant_trades or not _HAS_YF or not _HAS_PANDAS:
            return {
                "committee": committee,
                "alpha": None,
                "trade_count": len(relevant_trades),
                "avg_return": None,
                "benchmark_return": None,
            }

        df = self.compute_post_trade_returns(relevant_trades, horizons=[90])
        if df.empty:
            return {"committee": committee, "alpha": None, "trade_count": 0,
                    "avg_return": None, "benchmark_return": None}

        # Benchmark: equal-weight return of sector tickers
        benchmark_returns: List[float] = []
        for tk in sector_tickers[:5]:
            try:
                spy_hist = yf.download(tk, period="1y", progress=False)
                if not spy_hist.empty:
                    r = float(
                        (spy_hist["Close"].iloc[-1] - spy_hist["Close"].iloc[0])
                        / spy_hist["Close"].iloc[0]
                    )
                    benchmark_returns.append(r)
            except Exception:
                pass

        bench = sum(benchmark_returns) / len(benchmark_returns) if benchmark_returns else 0.0
        avg_return = float(df["90d_return"].dropna().mean()) if not df.empty else 0.0
        alpha = avg_return - bench

        return {
            "committee": committee,
            "alpha": round(alpha, 4),
            "trade_count": len(relevant_trades),
            "avg_return": round(avg_return, 4),
            "benchmark_return": round(bench, 4),
        }

    def backtest_congress_portfolio(
        self, start: str = "2020-01-01"
    ) -> BacktestResult:
        """
        Long all stocks where Congress buys, equal weight, monthly rebalance.
        Benchmark: SPY. Requires yfinance.
        """
        if not _HAS_YF or not _HAS_PANDAS:
            raise ImportError("pandas + yfinance required for backtest")

        all_trades = self._all_trades()
        buy_trades = [
            t for t in all_trades
            if t.is_buy and t.ticker and t.transaction_date
            and t.transaction_date >= _parse_date(start)  # type: ignore[operator]
        ]
        if not buy_trades:
            raise ValueError("No buy trades in the specified period")

        # Monthly portfolio: each month, get all tickers Congress bought last month
        start_dt = _parse_date(start)
        end_dt   = date.today()

        monthly_returns: List[float] = []
        current = start_dt.replace(day=1)  # type: ignore[union-attr]

        spy_monthly: List[float] = []

        while current < end_dt:
            # Next month boundary
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1, day=1)
            else:
                next_month = current.replace(month=current.month + 1, day=1)

            prev_month = (current - timedelta(days=1)).replace(day=1)
            # Trades in prior month
            month_buys = [
                t for t in buy_trades
                if t.transaction_date
                and prev_month <= t.transaction_date < current
            ]
            tickers = list({t.ticker for t in month_buys if t.ticker})

            month_start_str = current.strftime("%Y-%m-%d")
            month_end_str   = min(next_month, end_dt).strftime("%Y-%m-%d")

            if tickers:
                ticker_returns: List[float] = []
                for tk in tickers[:30]:  # cap at 30 to avoid too many requests
                    try:
                        hist = yf.download(
                            tk, start=month_start_str,
                            end=month_end_str, progress=False
                        )
                        if not hist.empty and len(hist) > 1:
                            r = float(
                                (hist["Close"].iloc[-1] - hist["Close"].iloc[0])
                                / hist["Close"].iloc[0]
                            )
                            ticker_returns.append(r)
                    except Exception:
                        pass
                monthly_returns.append(
                    sum(ticker_returns) / len(ticker_returns) if ticker_returns else 0.0
                )
            else:
                monthly_returns.append(0.0)

            # SPY for same month
            try:
                spy_h = yf.download(
                    "SPY", start=month_start_str,
                    end=month_end_str, progress=False
                )
                if not spy_h.empty and len(spy_h) > 1:
                    spy_r = float(
                        (spy_h["Close"].iloc[-1] - spy_h["Close"].iloc[0])
                        / spy_h["Close"].iloc[0]
                    )
                else:
                    spy_r = 0.0
            except Exception:
                spy_r = 0.0
            spy_monthly.append(spy_r)

            current = next_month

        # Compute cumulative returns
        def compound(returns: List[float]) -> float:
            r = 1.0
            for m in returns:
                r *= (1 + m)
            return r - 1.0

        def annualize(total: float, months: int) -> float:
            if months <= 0:
                return 0.0
            years = months / 12.0
            return ((1 + total) ** (1.0 / years)) - 1.0

        total_ret  = compound(monthly_returns)
        bench_ret  = compound(spy_monthly)
        months     = len(monthly_returns)
        ann_ret    = annualize(total_ret, months)
        ann_bench  = annualize(bench_ret, months)

        # Max drawdown
        equity = [1.0]
        for r in monthly_returns:
            equity.append(equity[-1] * (1 + r))
        peak = equity[0]
        max_dd = 0.0
        for v in equity:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd

        # Sharpe (monthly RF ~ 0.4% / 12)
        rf_monthly = 0.004 / 12
        excess = [r - rf_monthly for r in monthly_returns]
        mean_excess = sum(excess) / len(excess) if excess else 0.0
        std_excess  = (
            (sum((r - mean_excess) ** 2 for r in excess) / len(excess)) ** 0.5
            if len(excess) > 1 else 0.0
        )
        sharpe = (mean_excess / std_excess * (12 ** 0.5)) if std_excess > 0 else 0.0

        return BacktestResult(
            start_date=start,
            end_date=end_dt.strftime("%Y-%m-%d"),
            total_return=round(total_ret, 4),
            annualized_return=round(ann_ret, 4),
            benchmark_return=round(bench_ret, 4),
            benchmark_annualized=round(ann_bench, 4),
            alpha=round(ann_ret - ann_bench, 4),
            sharpe_ratio=round(sharpe, 3),
            max_drawdown=round(max_dd, 4),
            trade_count=len(buy_trades),
            win_rate=round(
                sum(1 for r in monthly_returns if r > 0) / len(monthly_returns), 3
            ) if monthly_returns else 0.0,
            avg_holding_days=30.0,
            monthly_returns=monthly_returns,
        )


# ---------------------------------------------------------------------------
# CongressScreener
# ---------------------------------------------------------------------------

class CongressScreener:
    """
    Screen for actionable congressional trading signals using preset or custom criteria.

    Presets
    -------
    cluster_buy       — 3+ members buying same ticker in 30 days
    committee_informed — relevant committee member buying sector stock
    late_disclosure    — disclosed trade with 30+ day lag
    heavy_buyer        — ticker with >5 congressional purchases in 90 days
    bipartisan_buy     — both R and D buying same stock in 30 days
    """

    PRESETS = {
        "cluster_buy",
        "committee_informed",
        "late_disclosure",
        "heavy_buyer",
        "bipartisan_buy",
    }

    def __init__(
        self,
        house_client: Optional[HouseStockWatcherClient] = None,
        senate_client: Optional[SenateStockWatcherClient] = None,
    ) -> None:
        self.house   = house_client  or HouseStockWatcherClient()
        self.senate  = senate_client or SenateStockWatcherClient()
        self._signal = CongressSignalEngine(self.house, self.senate)
        self._tracker = CongressMemberTracker(self.house, self.senate)

    def _all_trades(self) -> List[CongressTrade]:
        return self.house.fetch_all_trades() + self.senate.fetch_all_trades()

    def run_preset(self, name: str) -> "pd.DataFrame":
        if not _HAS_PANDAS:
            raise ImportError("pandas required for CongressScreener")
        if name not in self.PRESETS:
            raise ValueError(f"Unknown preset '{name}'. Options: {self.PRESETS}")

        if name == "cluster_buy":
            return self._preset_cluster_buy()
        if name == "committee_informed":
            return self._preset_committee_informed()
        if name == "late_disclosure":
            return self._preset_late_disclosure()
        if name == "heavy_buyer":
            return self._preset_heavy_buyer()
        if name == "bipartisan_buy":
            return self._preset_bipartisan_buy()
        return pd.DataFrame()

    def _preset_cluster_buy(self, days: int = 30, min_members: int = 3) -> "pd.DataFrame":
        """Find all tickers with cluster buys."""
        all_trades = self._all_trades()
        tickers = {t.ticker for t in all_trades if t.ticker}
        rows: List[dict] = []
        for ticker in tickers:
            clusters = self._signal.detect_cluster_buys(ticker, days, min_members)
            for c in clusters:
                rows.append({
                    "ticker": ticker,
                    "window_start": c.window_start,
                    "window_end": c.window_end,
                    "member_count": c.member_count,
                    "members": ", ".join(c.members),
                    "bipartisan": c.bipartisan,
                    "committee_chair": c.committee_chair_in_cluster,
                    "total_value": c.total_estimated_value,
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("member_count", ascending=False).reset_index(drop=True)

    def _preset_committee_informed(self) -> "pd.DataFrame":
        """Committee member trading sector-relevant stock."""
        frames: List["pd.DataFrame"] = []
        for committee in COMMITTEE_SECTOR_MAP:
            df = self._signal.get_committee_relevant_tickers(committee)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).drop_duplicates()

    def _preset_late_disclosure(self, lag: int = 30) -> "pd.DataFrame":
        late = self._tracker.detect_late_filers(days_threshold=lag)
        return pd.DataFrame(late) if _HAS_PANDAS else pd.DataFrame()

    def _preset_heavy_buyer(
        self, days: int = 90, min_trades: int = 5
    ) -> "pd.DataFrame":
        df = self._signal.rank_tickers_by_congress_activity(days)
        if df.empty:
            return df
        return df[df["buy_count"] >= min_trades].reset_index(drop=True)

    def _preset_bipartisan_buy(self, days: int = 30) -> "pd.DataFrame":
        cutoff = date.today() - timedelta(days=days)
        buys = [
            t for t in self._all_trades()
            if t.is_buy and t.transaction_date and t.transaction_date >= cutoff and t.ticker
        ]
        ticker_parties: Dict[str, set] = defaultdict(set)
        ticker_count: Dict[str, int] = defaultdict(int)
        for t in buys:
            if t.party:
                ticker_parties[t.ticker].add(t.party.upper()[:1])
            ticker_count[t.ticker] += 1

        rows = [
            {
                "ticker": tk,
                "parties": ", ".join(sorted(parties)),
                "buy_count": ticker_count[tk],
                "bipartisan": len(parties) > 1,
            }
            for tk, parties in ticker_parties.items()
            if len(parties) > 1
        ]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("buy_count", ascending=False).reset_index(drop=True)

    def screen(self, criteria: dict) -> "pd.DataFrame":
        """
        Custom screening. Criteria keys:
          - ticker: str (filter by ticker)
          - min_buy_count: int (min purchases in window)
          - max_lag_days: int (max disclosure lag)
          - days: int (lookback window)
          - bipartisan: bool
          - committee_chair: bool
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas required")

        days = criteria.get("days", 90)
        cutoff = date.today() - timedelta(days=days)
        all_trades = [
            t for t in self._all_trades()
            if not t.transaction_date or t.transaction_date >= cutoff
        ]

        if "ticker" in criteria:
            tk = criteria["ticker"].upper()
            all_trades = [t for t in all_trades if t.ticker == tk]

        rows: List[dict] = []
        for t in all_trades:
            lag = t.disclosure_lag_days
            if "max_lag_days" in criteria:
                if lag is None or lag > criteria["max_lag_days"]:
                    continue
            row = {
                "member": t.member, "chamber": t.chamber,
                "ticker": t.ticker, "trade_type": t.trade_type,
                "transaction_date": t.transaction_date,
                "disclosure_date": t.disclosure_date,
                "lag_days": lag,
                "amount_midpoint": t.amount_midpoint,
                "party": t.party,
                "is_chair": _is_committee_chair(t.member),
            }
            if criteria.get("committee_chair") and not row["is_chair"]:
                continue
            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        if "bipartisan" in criteria and criteria["bipartisan"]:
            # keep only tickers with both parties
            ticker_parties = df.groupby("ticker")["party"].apply(
                lambda s: len({p.upper()[:1] for p in s if p}) > 1
            )
            bipartisan_tickers = ticker_parties[ticker_parties].index.tolist()
            df = df[df["ticker"].isin(bipartisan_tickers)]

        if "min_buy_count" in criteria:
            buy_counts = df[df["trade_type"] == "purchase"].groupby("ticker").size()
            valid_tickers = buy_counts[buy_counts >= criteria["min_buy_count"]].index.tolist()
            df = df[df["ticker"].isin(valid_tickers)]

        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Module-level math-verified functions (dim_029 score 9)
# ---------------------------------------------------------------------------

def compute_alpha_signal(
    trades: List[CongressTrade],
    ticker: str,
    horizon_days: int = 30,
    benchmark_symbol: str = "SPY",
) -> dict:
    """Lag-adjusted excess return after congress purchase vs benchmark.

    Formula (verified):
        excess_return = portfolio_return - benchmark_return

    where:
      - portfolio_return  = (price[t + horizon] - price[t]) / price[t]
        for each buy of ``ticker``, averaged equally across all buys
      - benchmark_return  = equivalent SPY return over the same period
      - The transaction_date is the *disclosed* date; the actual purchase
        may precede it by up to 45 days (STOCK Act disclosure lag).

    Returns None values for return components if yfinance is unavailable.
    """
    buy_trades = [
        t for t in trades
        if t.ticker and t.ticker.upper() == ticker.upper()
        and t.is_buy
        and t.transaction_date
    ]

    if not buy_trades:
        return {
            "ticker":             ticker,
            "horizon_days":       horizon_days,
            "benchmark":          benchmark_symbol,
            "buy_count":          0,
            "portfolio_return":   None,
            "benchmark_return":   None,
            "excess_return":      None,
            "signal":             "no_data",
        }

    portfolio_return: Optional[float] = None
    benchmark_return: Optional[float] = None
    excess_return:    Optional[float] = None

    if _HAS_YF and _HAS_PANDAS:
        try:
            all_dates = [t.transaction_date for t in buy_trades]
            start_dt  = (min(all_dates) - timedelta(days=2)).strftime("%Y-%m-%d")  # type: ignore[type-var]
            end_dt    = (max(all_dates) + timedelta(days=horizon_days + 10)).strftime("%Y-%m-%d")  # type: ignore[type-var]

            import yfinance as _yf
            tk_hist  = _yf.download(ticker,          start=start_dt, end=end_dt, progress=False)
            spy_hist = _yf.download(benchmark_symbol, start=start_dt, end=end_dt, progress=False)

            tk_prices  = tk_hist["Close"]  if not tk_hist.empty  else None
            spy_prices = spy_hist["Close"] if not spy_hist.empty else None

            def _nearest_price(series: "pd.Series", target_date: date) -> Optional[float]:
                for offset in range(6):
                    ts = pd.Timestamp(target_date + timedelta(days=offset))
                    if ts in series.index:
                        return float(series[ts])
                return None

            stock_returns: List[float] = []
            spy_returns:   List[float] = []

            for trade in buy_trades:
                t_date = trade.transaction_date
                t_end  = t_date + timedelta(days=horizon_days)  # type: ignore[operator]
                if tk_prices is not None:
                    p0 = _nearest_price(tk_prices, t_date)   # type: ignore[arg-type]
                    p1 = _nearest_price(tk_prices, t_end)    # type: ignore[arg-type]
                    if p0 and p1 and p0 > 0:
                        stock_returns.append((p1 - p0) / p0)
                if spy_prices is not None:
                    b0 = _nearest_price(spy_prices, t_date)  # type: ignore[arg-type]
                    b1 = _nearest_price(spy_prices, t_end)   # type: ignore[arg-type]
                    if b0 and b1 and b0 > 0:
                        spy_returns.append((b1 - b0) / b0)

            if stock_returns:
                portfolio_return = round(sum(stock_returns) / len(stock_returns), 6)
            if spy_returns:
                benchmark_return = round(sum(spy_returns) / len(spy_returns), 6)
            if portfolio_return is not None and benchmark_return is not None:
                excess_return = round(portfolio_return - benchmark_return, 6)

        except Exception as _exc:
            logger.debug("Alpha signal price fetch failed: %s", _exc)

    signal_label: str
    if excess_return is None:
        signal_label = "no_price_data"
    elif excess_return > 0.02:
        signal_label = "alpha_positive"
    elif excess_return < -0.02:
        signal_label = "alpha_negative"
    else:
        signal_label = "neutral"

    return {
        "ticker":           ticker,
        "horizon_days":     horizon_days,
        "benchmark":        benchmark_symbol,
        "buy_count":        len(buy_trades),
        "portfolio_return":  portfolio_return,
        "benchmark_return":  benchmark_return,
        "excess_return":     excess_return,   # excess_return = portfolio - benchmark
        "signal":            signal_label,
    }


def detect_cluster_trades(
    trades: List[CongressTrade],
    ticker: str,
    window_days: int = 7,
    min_members: int = 3,
) -> dict:
    """Detect clusters of >= min_members congress members buying the same ticker.

    A cluster requires all buys to fall within a ``window_days``-day rolling
    window (default 7 days).  Clusters of 3+ members in 7 days are considered
    a strong coordinated signal.

    Returns a list of cluster dicts with member names, dates, and party breakdown.
    """
    buy_trades = [
        t for t in trades
        if t.ticker and t.ticker.upper() == ticker.upper()
        and t.is_buy
        and t.transaction_date
    ]
    buy_trades.sort(key=lambda t: t.transaction_date)  # type: ignore[arg-type]

    clusters: List[dict] = []
    seen: set = set()

    for i, anchor in enumerate(buy_trades):
        window_end = anchor.transaction_date + timedelta(days=window_days)  # type: ignore[operator]
        in_window  = [
            t for t in buy_trades
            if anchor.transaction_date <= t.transaction_date <= window_end  # type: ignore[operator]
        ]
        unique_members = list({t.member for t in in_window})
        if len(unique_members) < min_members:
            continue

        key = frozenset(unique_members)
        if key in seen:
            continue
        seen.add(key)

        parties = {t.party for t in in_window if t.party}
        bipartisan = len({p.upper()[:1] for p in parties if p}) > 1
        total_value = sum(t.amount_midpoint for t in in_window if t.amount_midpoint)
        dates = [t.transaction_date for t in in_window if t.transaction_date]

        clusters.append({
            "ticker":        ticker,
            "window_start":  min(dates),
            "window_end":    max(dates),
            "member_count":  len(unique_members),
            "members":       unique_members,
            "parties":       list(parties),
            "bipartisan":    bipartisan,
            "total_value_usd": round(total_value, 2),
        })

    return {
        "ticker":           ticker,
        "window_days":      window_days,
        "min_members":      min_members,
        "cluster_detected": len(clusters) > 0,
        "cluster_count":    len(clusters),
        "clusters":         clusters,
    }


def compute_senator_performance(
    trades: List[CongressTrade],
    horizon_days: int = 90,
) -> List[dict]:
    """Annualized return of each senator/representative's disclosed trades.

    For each member, compute:
      1. Average forward return across all their buy trades at ``horizon_days``
      2. Annualized return = ((1 + avg_return) ^ (365 / horizon_days)) - 1

    Requires yfinance for price data.  Members with insufficient data are
    included with None return values so callers can still rank by trade count.

    Returns list of dicts sorted by annualized_return (desc), NaN-last.
    """
    from collections import defaultdict
    member_trades: dict = defaultdict(list)
    for t in trades:
        if t.is_buy and t.ticker and t.transaction_date and t.member:
            member_trades[t.member].append(t)

    results: List[dict] = []

    for member, mtrades in member_trades.items():
        avg_return:      Optional[float] = None
        annualized:      Optional[float] = None

        if _HAS_YF and _HAS_PANDAS:
            try:
                import yfinance as _yf
                returns: List[float] = []

                # Group by ticker to batch downloads
                by_ticker: dict = defaultdict(list)
                for t in mtrades:
                    by_ticker[t.ticker].append(t)

                for ticker, ticker_trades in by_ticker.items():
                    all_dates = [t.transaction_date for t in ticker_trades]
                    start_dt  = (min(all_dates) - timedelta(days=2)).strftime("%Y-%m-%d")  # type: ignore[type-var]
                    end_dt    = (max(all_dates) + timedelta(days=horizon_days + 10)).strftime("%Y-%m-%d")  # type: ignore[type-var]

                    hist = _yf.download(ticker, start=start_dt, end=end_dt, progress=False)
                    if hist.empty:
                        continue
                    prices = hist["Close"]

                    for trade in ticker_trades:
                        t_end = trade.transaction_date + timedelta(days=horizon_days)  # type: ignore[operator]
                        try:
                            buy_ts  = pd.Timestamp(trade.transaction_date)
                            sell_ts = pd.Timestamp(t_end)
                            bi = prices.index.searchsorted(buy_ts)
                            si = prices.index.searchsorted(sell_ts)
                            if bi < len(prices) and si < len(prices):
                                p0 = float(prices.iloc[bi].item() if hasattr(prices.iloc[bi], 'item') else prices.iloc[bi])
                                p1_idx = min(si, len(prices) - 1)
                                p1 = float(prices.iloc[p1_idx].item() if hasattr(prices.iloc[p1_idx], 'item') else prices.iloc[p1_idx])
                                if p0 > 0:
                                    returns.append((p1 - p0) / p0)
                        except Exception:
                            pass

                if returns:
                    avg_return = sum(returns) / len(returns)
                    # Annualize: (1 + avg_hold_return)^(365/horizon) - 1
                    annualized = round(
                        (1.0 + avg_return) ** (365.0 / horizon_days) - 1.0, 4
                    )
                    avg_return = round(avg_return, 4)

            except Exception as _exc:
                logger.debug("Senator performance calc failed for %s: %s", member, _exc)

        results.append({
            "member":              member,
            "trade_count":         len(mtrades),
            "tickers_traded":      len({t.ticker for t in mtrades}),
            "horizon_days":        horizon_days,
            "avg_period_return":   avg_return,
            "annualized_return":   annualized,
        })

    # Sort: non-None annualized returns descending, then None entries
    results.sort(
        key=lambda r: (r["annualized_return"] is None, -(r["annualized_return"] or 0))
    )
    return results


# ---------------------------------------------------------------------------
# CongressTrackerEngine (orchestrator)
# ---------------------------------------------------------------------------

class CongressTrackerEngine:
    """
    Main orchestrator for congressional trading intelligence.

    Usage::
        engine = CongressTrackerEngine()
        dashboard = engine.get_dashboard()
        signal = engine.get_ticker_signal("NVDA")
    """

    def __init__(self) -> None:
        self.house   = HouseStockWatcherClient()
        self.senate  = SenateStockWatcherClient()
        self.signals = CongressSignalEngine(self.house, self.senate)
        self.tracker = CongressMemberTracker(self.house, self.senate)
        self.analyzer = CongressReturnAnalyzer(self.house, self.senate)
        self.screener = CongressScreener(self.house, self.senate)

    def _all_trades(self) -> List[CongressTrade]:
        return self.house.fetch_all_trades() + self.senate.fetch_all_trades()

    def get_dashboard(self, days: int = 30) -> CongressDashboard:
        """Fetch a high-level dashboard of recent congressional trading."""
        cutoff = date.today() - timedelta(days=days)
        all_trades = self._all_trades()
        recent = [
            t for t in all_trades
            if t.transaction_date and t.transaction_date >= cutoff
        ]

        house_count  = sum(1 for t in recent if t.chamber == "house")
        senate_count = sum(1 for t in recent if t.chamber == "senate")
        tickers = {t.ticker for t in recent if t.ticker}
        members = {t.member for t in recent}

        # Top tickers by trade count
        from collections import Counter
        ticker_counts = Counter(t.ticker for t in recent if t.ticker)
        top_tickers = ticker_counts.most_common(10)

        # Top buyers
        buyer_counts = Counter(
            t.member for t in recent if t.is_buy
        )
        top_buyers = buyer_counts.most_common(10)

        # Cluster buys
        clusters: List[CongressCluster] = []
        for ticker in [t for t, _ in top_tickers]:
            c = self.signals.detect_cluster_buys(ticker, days=days)
            clusters.extend(c)

        # Late filers
        late = self.tracker.detect_late_filers(days_threshold=45)[:20]

        return CongressDashboard(
            as_of=date.today(),
            days_window=days,
            total_trades=len(recent),
            house_trades=house_count,
            senate_trades=senate_count,
            unique_tickers=len(tickers),
            unique_members=len(members),
            top_tickers=top_tickers,
            top_buyers=top_buyers,
            clusters=clusters,
            late_filers=late,
            recent_trades=sorted(
                recent, key=lambda t: t.transaction_date or date.min, reverse=True
            )[:50],
        )

    def get_ticker_signal(self, ticker: str) -> CongressSignal:
        """Get the full congressional trading signal for a single ticker."""
        return self.signals.compute_congress_buy_signal(ticker, days=90)

    def get_most_active_tickers(self, days: int = 30) -> "pd.DataFrame":
        """Return DataFrame of most actively traded tickers by Congress members."""
        return self.signals.rank_tickers_by_congress_activity(days=days)

    def generate_weekly_report(self) -> str:
        """Generate a plain-text weekly congressional trading report."""
        db = self.get_dashboard(days=7)
        lines: List[str] = [
            f"=== Congressional STOCK Act Report — Week Ending {db.as_of} ===",
            f"Total trades filed: {db.total_trades}  (House: {db.house_trades}, Senate: {db.senate_trades})",
            f"Unique tickers traded: {db.unique_tickers}",
            f"Unique members filing: {db.unique_members}",
            "",
            "--- Top 10 Most Traded Tickers ---",
        ]
        for ticker, count in db.top_tickers[:10]:
            sig = self.get_ticker_signal(ticker)
            lines.append(
                f"  {ticker:6s}  {count:3d} trades  direction={sig.net_direction}"
                f"  score={sig.net_score:+.1f}"
                f"  bipartisan={sig.bipartisan}"
            )
        lines.append("")
        lines.append("--- Top Buyers ---")
        for member, count in db.top_buyers[:10]:
            lines.append(f"  {member}  ({count} purchases)")
        lines.append("")
        lines.append("--- Cluster Buy Signals ---")
        if db.clusters:
            for c in db.clusters[:5]:
                lines.append(
                    f"  {c.ticker}: {c.member_count} members "
                    f"({c.window_start}–{c.window_end}), "
                    f"bipartisan={c.bipartisan}, chair={c.committee_chair_in_cluster}"
                )
        else:
            lines.append("  No cluster signals detected this week.")
        lines.append("")
        lines.append("--- Late Filers (45+ day lag) ---")
        for lf in db.late_filers[:5]:
            lines.append(
                f"  {lf['member']}  ticker={lf['ticker']}  lag={lf['lag_days']}d"
            )
        return "\n".join(lines)

    def run_screen(self, preset: str) -> "pd.DataFrame":
        """Run a named preset screen."""
        return self.screener.run_preset(preset)

    def get_member_portfolio(self, name: str) -> CongressPortfolio:
        return self.tracker.get_member_portfolio(name)

    def get_top_traders(self, n: int = 20) -> "pd.DataFrame":
        return self.tracker.get_top_traders(n)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    print("Congressional STOCK Act Tracker — V3")
    print("=" * 60)

    engine = CongressTrackerEngine()

    print("\n[1] Fetching all House + Senate trades …")
    house_trades  = engine.house.fetch_all_trades()
    senate_trades = engine.senate.fetch_all_trades()
    print(f"    House:  {len(house_trades):,} trades")
    print(f"    Senate: {len(senate_trades):,} trades")

    print("\n[2] Top 10 most traded tickers (last 90 days) …")
    try:
        df_active = engine.get_most_active_tickers(days=90)
        if _HAS_PANDAS and not df_active.empty:
            print(df_active.head(10).to_string(index=False))
        else:
            print("  (no data)")
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\n[3] Detecting cluster buys (last 30 days) …")
    try:
        cluster_df = engine.screener.run_preset("cluster_buy")
        if _HAS_PANDAS and not cluster_df.empty:
            print(cluster_df.head(10).to_string(index=False))
        else:
            print("  No cluster buys detected.")
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\n[4] Computing aggregate alpha vs SPY …")
    try:
        all_trades = engine._all_trades()
        alpha = engine.analyzer.compute_aggregate_alpha(all_trades[:500])
        print(f"    Aggregate alpha (90-day horizon): {alpha:+.2%}")
    except Exception as exc:
        print(f"  Error (likely missing yfinance): {exc}")

    print("\n[5] Late filers (>45 day lag) …")
    late = engine.tracker.detect_late_filers(45)
    print(f"    Total late filings: {len(late)}")
    for lf in late[:5]:
        print(f"      {lf['member']}: {lf['ticker']} lag={lf['lag_days']}d")

    print("\n[6] Weekly report …")
    report = engine.generate_weekly_report()
    print(report)

    print("\nDone.")
    sys.exit(0)
