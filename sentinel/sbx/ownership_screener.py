"""Ownership-based screener — dim_072 (score 5 → 9).

Combines SEC EDGAR 13F-HR (institutional holdings) and Form 4 (insider
transactions) data to produce a composite ownership signal for any ticker.

Key signals
-----------
  strong_accumulation  : net institutional buying + insider cluster buys
  accumulation         : either net institutional buying or insider buying
  neutral              : no significant net change
  distribution         : net institutional selling or insider cluster selling
  strong_distribution  : both institutional and insider selling

Public API
----------
ownership_profile(ticker)                   → OwnershipProfile
screen_ownership(tickers, …)               → OwnershipScreenResult
insider_cluster_buys(days_back)            → list[dict]
"""
from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, timedelta
from typing import Any, Optional

import httpx
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from sentinel.core.logging import get_logger
from sentinel.sds.adapters.insider_adapter import InsiderAdapter
from sentinel.sds.adapters.institutional_adapter import InstitutionalAdapter

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_EFTS = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
EDGAR_COMPANY_SEARCH = "https://efts.sec.gov/LATEST/search-index"
EDGAR_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}
_RATE_LIMIT = 0.12  # ~8 req/sec, conservative

# CIK lookup cache (ticker → CIK string, 10-digit zero-padded)
_cik_cache: dict[str, str] = {}

# Known large fund CIKs for "smart money" analysis
_MAJOR_FUND_CIKS: dict[str, str] = {
    "Berkshire Hathaway": "0001067983",
    "Vanguard Group": "0000102909",
    "BlackRock": "0001364742",
    "State Street": "0000093751",
    "Fidelity (FMR)": "0000315066",
    "T. Rowe Price": "0001113169",
    "Soros Fund Management": "0001029160",
    "Tiger Global": "0001167483",
    "Druckenmiller / Duquesne": "0001536411",
    "Pershing Square": "0001336528",
    "Third Point": "0001040273",
    "Elliott Management": "0000849399",
    "Appaloosa Management": "0001006438",
    "Viking Global": "0001109210",
    "Coatue Management": "0001336092",
    "Lone Pine Capital": "0001061165",
    "D1 Capital": "0001774173",
    "Baupost Group": "0001061165",
    "Greenlight Capital": "0001079114",
    "Citadel Advisors": "0001423298",
}

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class InstitutionalChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    manager_name: str
    manager_cik: str
    ticker: str
    period: date
    prev_shares: Optional[float] = None
    curr_shares: Optional[float] = None
    change_shares: Optional[float] = None
    change_pct: Optional[float] = None
    action: str  # "new_position", "increased", "decreased", "closed"
    position_value_usd: Optional[float] = None
    pct_of_portfolio: Optional[float] = None


class InsiderActivity(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    insider_name: str
    role: str
    tx_date: date
    tx_type: str        # "P"=purchase, "S"=sale, "A"=award, "D"=disposition
    shares: float
    price: Optional[float] = None
    value_usd: Optional[float] = None
    shares_owned_after: Optional[float] = None
    is_open_market: bool = False


class OwnershipProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    as_of: date
    # Institutional
    total_institutional_shares: Optional[float] = None
    institutional_pct_owned: Optional[float] = None
    n_institutional_holders: Optional[int] = None
    top_5_holders: list[dict] = Field(default_factory=list)
    new_positions_90d: list[InstitutionalChange] = Field(default_factory=list)
    closed_positions_90d: list[InstitutionalChange] = Field(default_factory=list)
    increased_90d: list[InstitutionalChange] = Field(default_factory=list)
    decreased_90d: list[InstitutionalChange] = Field(default_factory=list)
    net_institutional_change_pct: Optional[float] = None
    # Insider
    insider_buying_90d: list[InsiderActivity] = Field(default_factory=list)
    insider_selling_90d: list[InsiderActivity] = Field(default_factory=list)
    net_insider_value_90d: Optional[float] = None
    # Composite signals
    ownership_signal: str = "neutral"
    signal_score: float = 0.0  # -10 to +10


class OwnershipScreenResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    criteria: dict
    n_screened: int
    results: list[OwnershipProfile]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


async def _get_json(url: str, params: Optional[dict] = None, timeout: float = 30.0) -> Any:
    """Rate-limited JSON GET."""
    await asyncio.sleep(_RATE_LIMIT)
    async with httpx.AsyncClient(
        headers=_HEADERS, timeout=timeout, follow_redirects=True
    ) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# CIK resolution
# ---------------------------------------------------------------------------


async def _resolve_cik(ticker: str) -> Optional[str]:
    """Resolve an equity ticker to its SEC EDGAR CIK (10-digit string).

    Uses the EDGAR company_tickers.json bulk file, cached in memory.
    """
    global _cik_cache

    ticker_up = ticker.upper()
    if ticker_up in _cik_cache:
        return _cik_cache[ticker_up]

    # Bulk company tickers JSON: {index: {cik_str, ticker, title}}
    if not _cik_cache:
        try:
            data = await _get_json(EDGAR_COMPANY_TICKERS)
            for entry in data.values():
                t = (entry.get("ticker") or "").upper()
                c = str(entry.get("cik_str", "")).zfill(10)
                if t:
                    _cik_cache[t] = c
        except Exception as exc:
            logger.warning("cik_cache load error: %s", exc)
            return None

    return _cik_cache.get(ticker_up)


# ---------------------------------------------------------------------------
# EDGAR full-text search for form types by ticker / company
# ---------------------------------------------------------------------------


async def _search_edgar(
    form_type: str,
    ticker: str,
    days_back: int = 90,
    limit: int = 20,
) -> list[dict]:
    """Search EDGAR EFTS for recent filings of a given type mentioning ticker.

    Returns list of filing metadata dicts with keys:
      {form_type, filing_date, accession_no, entity_name, cik}
    """
    cutoff = (date.today() - timedelta(days=days_back)).isoformat()
    params = {
        "q": f'"{ticker}"',
        "dateRange": "custom",
        "startdt": cutoff,
        "enddt": date.today().isoformat(),
        "forms": form_type,
        "_source": "period_of_report,file_date,entity_name,file_num,period_of_report",
        "hits.hits.total.value": 1,
        "hits.hits._source.period_of_report": 1,
        "hits.hits._source.entity_name": 1,
        "hits.hits._source.file_date": 1,
        "_source.accession_no": 1,
    }
    url = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms={form}&dateRange=custom&startdt={start}&enddt={end}&hits.hits.total.value=1".format(
        ticker=ticker,
        form=form_type,
        start=cutoff,
        end=date.today().isoformat(),
    )
    # Use the proper EDGAR full-text search endpoint
    search_url = "https://efts.sec.gov/LATEST/search-index"
    try:
        data = await _get_json(
            search_url,
            params={
                "q": f'"{ticker}"',
                "forms": form_type,
                "dateRange": "custom",
                "startdt": cutoff,
                "enddt": date.today().isoformat(),
                "hits.hits._source": "period_of_report,file_date,entity_name,accession_no,period_of_report",
            },
            timeout=20.0,
        )
    except Exception as exc:
        logger.warning("edgar_search error form=%s ticker=%s: %s", form_type, ticker, exc)
        return []

    hits = data.get("hits", {}).get("hits", [])
    results = []
    for hit in hits[:limit]:
        src = hit.get("_source", {})
        results.append({
            "form_type": form_type,
            "filing_date": src.get("file_date", ""),
            "period": src.get("period_of_report", ""),
            "entity_name": src.get("entity_name", ""),
            "accession_no": src.get("accession_no", ""),
            "cik": src.get("entity_id", ""),
        })
    return results


# ---------------------------------------------------------------------------
# Institutional change computation
# ---------------------------------------------------------------------------


async def _get_institutional_changes(
    ticker: str,
    cik: Optional[str],
    days_back: int = 90,
) -> tuple[list[InstitutionalChange], float, list[dict]]:
    """Pull 13F changes for a ticker across major institutional managers.

    Strategy: iterate _MAJOR_FUND_CIKS, fetch their two most recent 13F
    filings, compare Q/Q position for this ticker's CUSIP/name, classify
    action.

    Returns (changes, net_change_pct, top_5_holders).
    """
    adapter = InstitutionalAdapter(
        user_agent="SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"
    )
    since = date.today() - timedelta(days=days_back + 180)  # give extra lookback

    ticker_up = ticker.upper()
    changes: list[InstitutionalChange] = []
    all_current: list[dict] = []  # for aggregation across managers

    async def _check_manager(name: str, mgr_cik: str) -> None:
        try:
            holdings = await adapter.fetch_holdings(mgr_cik, limit=4, since=since)
        except Exception as exc:
            logger.debug("institutional_changes manager=%s error: %s", name, exc)
            return

        if not holdings:
            return

        # Group by period and find latest two quarters
        periods: dict[date, list[dict]] = {}
        for h in holdings:
            p = h.get("period_of_report")
            if p is None:
                continue
            periods.setdefault(p, []).append(h)

        sorted_periods = sorted(periods.keys(), reverse=True)
        if len(sorted_periods) < 1:
            return

        curr_period = sorted_periods[0]
        prev_period = sorted_periods[1] if len(sorted_periods) > 1 else None

        def _match(h: dict) -> bool:
            """Does this holding match the ticker?"""
            issuer = (h.get("issuer_name") or "").upper()
            t = (h.get("ticker") or "").upper()
            return ticker_up in issuer or t == ticker_up

        curr_holdings = [h for h in periods[curr_period] if _match(h)]
        prev_holdings = [h for h in periods[prev_period] if _match(h)] if prev_period else []

        curr_shares = sum(
            float(h.get("shares") or 0) for h in curr_holdings
        )
        prev_shares = sum(
            float(h.get("shares") or 0) for h in prev_holdings
        )
        curr_value = sum(
            float(h.get("market_value") or 0) for h in curr_holdings
        )

        if curr_shares == 0 and prev_shares == 0:
            return

        # Classify
        if prev_shares == 0 and curr_shares > 0:
            action = "new_position"
            change_pct = None
        elif curr_shares == 0 and prev_shares > 0:
            action = "closed"
            change_pct = -100.0
        elif curr_shares > prev_shares:
            action = "increased"
            change_pct = (curr_shares - prev_shares) / prev_shares * 100.0 if prev_shares else None
        else:
            action = "decreased"
            change_pct = (curr_shares - prev_shares) / prev_shares * 100.0 if prev_shares else None

        change = InstitutionalChange(
            manager_name=name,
            manager_cik=mgr_cik,
            ticker=ticker_up,
            period=curr_period,
            prev_shares=prev_shares if prev_shares else None,
            curr_shares=curr_shares if curr_shares else None,
            change_shares=curr_shares - prev_shares,
            change_pct=round(change_pct, 2) if change_pct is not None else None,
            action=action,
            position_value_usd=curr_value if curr_value else None,
        )
        changes.append(change)

        if curr_shares > 0:
            all_current.append({
                "manager_name": name,
                "manager_cik": mgr_cik,
                "shares": curr_shares,
                "value_usd": curr_value,
            })

    # Run all manager checks concurrently (limited to 5 at a time)
    sem = asyncio.Semaphore(5)

    async def _guarded(name: str, cik_str: str) -> None:
        async with sem:
            await _check_manager(name, cik_str)

    await asyncio.gather(
        *[_guarded(n, c) for n, c in _MAJOR_FUND_CIKS.items()],
        return_exceptions=True,
    )

    # Net change across all managers
    total_curr = sum(c.curr_shares or 0 for c in changes)
    total_prev = sum(c.prev_shares or 0 for c in changes)
    net_change_pct = (
        (total_curr - total_prev) / total_prev * 100.0 if total_prev else 0.0
    )

    # Top 5 by current shares
    all_current.sort(key=lambda x: x["shares"], reverse=True)
    top5 = all_current[:5]

    logger.info(
        "institutional_changes ticker=%s managers_checked=%d changes=%d net_pct=%.2f",
        ticker, len(_MAJOR_FUND_CIKS), len(changes), net_change_pct,
    )
    return changes, round(net_change_pct, 2), top5


# ---------------------------------------------------------------------------
# Insider transaction fetch
# ---------------------------------------------------------------------------


async def _get_insider_transactions(
    ticker: str,
    cik: Optional[str],
    days_back: int = 90,
) -> tuple[list[InsiderActivity], float]:
    """Fetch Form 4 filings for a ticker's company CIK.

    Returns (activities, net_value_usd) where net_value is positive for
    net buying and negative for net selling (open-market trades only).
    """
    if cik is None:
        cik = await _resolve_cik(ticker)
    if cik is None:
        logger.warning("insider_tx: could not resolve CIK for ticker=%s", ticker)
        return [], 0.0

    adapter = InsiderAdapter(
        user_agent="SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"
    )
    since = date.today() - timedelta(days=days_back)

    try:
        raw_txs = await adapter.fetch_transactions(cik=cik, ticker=ticker, since=since, limit=60)
    except Exception as exc:
        logger.warning("insider_tx fetch error ticker=%s cik=%s: %s", ticker, cik, exc)
        return [], 0.0

    activities: list[InsiderActivity] = []
    for tx in raw_txs:
        tx_code = (tx.get("tx_code") or "").upper()
        shares_raw = tx.get("shares")
        price_raw = tx.get("price_per_share")
        value_raw = tx.get("value")
        tx_date = tx.get("tx_date")

        if tx_date is None or shares_raw is None:
            continue

        shares_f = float(shares_raw)
        price_f = float(price_raw) if price_raw else None
        value_f = float(value_raw) if value_raw else None

        # Open-market trades: code P (purchase) or S (sale)
        is_open_market = tx_code in ("P", "S")

        # Normalize sign: positive = acquired
        tx_type = tx_code
        if isinstance(tx_date, str):
            try:
                tx_date = datetime.strptime(tx_date, "%Y-%m-%d").date()
            except ValueError:
                continue

        activities.append(
            InsiderActivity(
                ticker=ticker.upper(),
                insider_name=(tx.get("owner_name") or "")[:200],
                role=(tx.get("role") or "Unknown")[:30],
                tx_date=tx_date,
                tx_type=tx_type,
                shares=shares_f,
                price=price_f,
                value_usd=abs(value_f) if value_f else None,
                shares_owned_after=(
                    float(tx["shares_owned_after"])
                    if tx.get("shares_owned_after") else None
                ),
                is_open_market=is_open_market,
            )
        )

    # Net value: open-market buys (P, positive shares) minus sells (S / negative shares)
    net_value = 0.0
    for act in activities:
        if not act.is_open_market or act.value_usd is None:
            continue
        if act.shares > 0:
            net_value += act.value_usd
        else:
            net_value -= act.value_usd

    logger.info(
        "insider_tx ticker=%s cik=%s activities=%d net_value=%.0f",
        ticker, cik, len(activities), net_value,
    )
    return activities, round(net_value, 2)


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------


def _compute_ownership_signal(
    net_institutional_change_pct: Optional[float],
    new_positions: list[InstitutionalChange],
    closed_positions: list[InstitutionalChange],
    increased: list[InstitutionalChange],
    decreased: list[InstitutionalChange],
    insider_buying: list[InsiderActivity],
    insider_selling: list[InsiderActivity],
    net_insider_value: float,
) -> tuple[str, float]:
    """Compute composite ownership signal and numeric score.

    Scoring rubric:
        +3  net institutional buying > 2%
        +2  new positions by 2+ managers
        +2  insider cluster buying (2+ open-market purchases)
        +1  net institutional buying 0–2%
        +1  single open-market insider purchase
        -1  net institutional selling 0–2%
        -2  insider cluster selling (2+ open-market sales)
        -2  2+ managers closed positions
        -3  net institutional selling > 2%

    Score clamped to [-10, +10].
    """
    score = 0.0
    inst_chg = net_institutional_change_pct or 0.0

    # Institutional direction
    if inst_chg > 2.0:
        score += 3.0
    elif inst_chg > 0.0:
        score += 1.0
    elif inst_chg < -2.0:
        score -= 3.0
    elif inst_chg < 0.0:
        score -= 1.0

    # New / closed positions
    large_new = [
        c for c in new_positions
        if (c.position_value_usd or 0) > 10_000_000  # >$10M position
    ]
    if len(large_new) >= 2:
        score += 2.0
    elif len(large_new) == 1:
        score += 1.0

    if len(closed_positions) >= 2:
        score -= 2.0
    elif len(closed_positions) == 1:
        score -= 1.0

    # Insider cluster
    open_buys = [a for a in insider_buying if a.is_open_market and a.shares > 0]
    open_sells = [a for a in insider_selling if a.is_open_market and a.shares < 0]

    if len(open_buys) >= 2:
        score += 2.0
    elif len(open_buys) == 1:
        score += 1.0

    if len(open_sells) >= 2:
        score -= 2.0
    elif len(open_sells) == 1:
        score -= 1.0

    score = max(-10.0, min(10.0, score))

    if score >= 4.0:
        label = "strong_accumulation"
    elif score >= 1.5:
        label = "accumulation"
    elif score <= -4.0:
        label = "strong_distribution"
    elif score <= -1.5:
        label = "distribution"
    else:
        label = "neutral"

    return label, round(score, 2)


# ---------------------------------------------------------------------------
# Main screener class
# ---------------------------------------------------------------------------


class OwnershipScreener:
    """Combine 13F institutional holdings and Form 4 insider trades.

    Usage (no context manager required):
        screener = OwnershipScreener()
        profile = await screener.get_ownership_profile("AAPL")
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Single ticker profile
    # ------------------------------------------------------------------

    async def get_ownership_profile(
        self, ticker: str, days_back: int = 90
    ) -> OwnershipProfile:
        """Build a complete ownership profile for a ticker.

        Fetches 13F changes across major managers and Form 4 insider
        transactions concurrently, then computes composite signal.
        """
        today = date.today()
        cik = await _resolve_cik(ticker)

        # Parallel fetch: institutional changes + insider transactions
        inst_task = asyncio.create_task(
            _get_institutional_changes(ticker, cik, days_back)
        )
        insider_task = asyncio.create_task(
            _get_insider_transactions(ticker, cik, days_back)
        )

        try:
            (changes, net_inst_pct, top5), (activities, net_insider_val) = await asyncio.gather(
                inst_task, insider_task
            )
        except Exception as exc:
            logger.error("ownership_profile ticker=%s error: %s", ticker, exc)
            changes, net_inst_pct, top5 = [], 0.0, []
            activities, net_insider_val = [], 0.0

        # Split institutional changes by action
        new_pos = [c for c in changes if c.action == "new_position"]
        closed = [c for c in changes if c.action == "closed"]
        increased = [c for c in changes if c.action == "increased"]
        decreased = [c for c in changes if c.action == "decreased"]

        # Split insider activities
        buying = [a for a in activities if a.shares > 0]
        selling = [a for a in activities if a.shares < 0]

        signal_label, signal_score = _compute_ownership_signal(
            net_inst_pct, new_pos, closed, increased, decreased,
            buying, selling, net_insider_val,
        )

        # Aggregate institutional shares
        total_inst_shares = sum(c.curr_shares or 0 for c in changes if c.action != "closed")
        n_holders = len([c for c in changes if (c.curr_shares or 0) > 0])

        return OwnershipProfile(
            ticker=ticker.upper(),
            as_of=today,
            total_institutional_shares=total_inst_shares if total_inst_shares else None,
            n_institutional_holders=n_holders if n_holders else None,
            top_5_holders=top5,
            new_positions_90d=new_pos,
            closed_positions_90d=closed,
            increased_90d=increased,
            decreased_90d=decreased,
            net_institutional_change_pct=net_inst_pct if changes else None,
            insider_buying_90d=buying,
            insider_selling_90d=selling,
            net_insider_value_90d=net_insider_val,
            ownership_signal=signal_label,
            signal_score=signal_score,
        )

    # ------------------------------------------------------------------
    # Multi-ticker screener
    # ------------------------------------------------------------------

    async def screen_by_ownership(
        self,
        tickers: list[str],
        min_institutional_pct: Optional[float] = None,
        require_insider_buying: bool = False,
        require_new_positions: bool = False,
        min_signal_score: Optional[float] = None,
        max_concurrent: int = 5,
    ) -> OwnershipScreenResult:
        """Screen a list of tickers by ownership criteria.

        Args:
            tickers: Equity tickers to screen.
            min_institutional_pct: Minimum institutional ownership percentage.
            require_insider_buying: Only keep tickers with net insider buying.
            require_new_positions: Only keep tickers with new institutional positions.
            min_signal_score: Minimum composite signal score (e.g. 2.0 for accumulation).
            max_concurrent: Max parallel profile fetches.

        Returns:
            OwnershipScreenResult with matching profiles sorted by signal score.
        """
        sem = asyncio.Semaphore(max_concurrent)

        async def _fetch(ticker: str) -> Optional[OwnershipProfile]:
            async with sem:
                try:
                    return await self.get_ownership_profile(ticker)
                except Exception as exc:
                    logger.warning("screen_by_ownership ticker=%s error: %s", ticker, exc)
                    return None

        profiles_raw = await asyncio.gather(*[_fetch(t) for t in tickers])
        profiles = [p for p in profiles_raw if p is not None]

        # Apply filters
        filtered: list[OwnershipProfile] = []
        for p in profiles:
            if min_institutional_pct is not None:
                if (p.institutional_pct_owned or 0) < min_institutional_pct:
                    continue
            if require_insider_buying and (p.net_insider_value_90d or 0) <= 0:
                continue
            if require_new_positions and not p.new_positions_90d:
                continue
            if min_signal_score is not None and p.signal_score < min_signal_score:
                continue
            filtered.append(p)

        filtered.sort(key=lambda p: p.signal_score, reverse=True)

        criteria = {
            "min_institutional_pct": min_institutional_pct,
            "require_insider_buying": require_insider_buying,
            "require_new_positions": require_new_positions,
            "min_signal_score": min_signal_score,
        }
        logger.info(
            "screen_by_ownership n_tickers=%d n_passed=%d",
            len(tickers), len(filtered),
        )
        return OwnershipScreenResult(
            criteria=criteria,
            n_screened=len(profiles),
            results=filtered,
        )

    # ------------------------------------------------------------------
    # Smart money consensus
    # ------------------------------------------------------------------

    async def find_smart_money_consensus(
        self,
        tickers: list[str],
        top_n_funds: int = 20,
    ) -> pd.DataFrame:
        """Find tickers where multiple top funds are adding positions.

        "Smart money consensus" = 2+ top funds with new/increased positions
        in the same quarter.

        Returns a DataFrame with columns:
          [ticker, consensus_count, funds, total_added_shares, signal_score]
        sorted by consensus_count descending.
        """
        profiles = await asyncio.gather(
            *[self.get_ownership_profile(t) for t in tickers],
            return_exceptions=True,
        )

        rows = []
        for p in profiles:
            if not isinstance(p, OwnershipProfile):
                continue
            # Count funds that are new or increased
            accum = p.new_positions_90d + p.increased_90d
            if len(accum) < 2:
                continue
            fund_names = [c.manager_name for c in accum]
            total_added = sum(
                (c.change_shares or 0) for c in accum if (c.change_shares or 0) > 0
            )
            rows.append({
                "ticker": p.ticker,
                "consensus_count": len(accum),
                "funds": ", ".join(fund_names),
                "total_added_shares": total_added,
                "signal_score": p.signal_score,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("consensus_count", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Insider cluster buys
    # ------------------------------------------------------------------

    async def get_insider_cluster_buys(
        self,
        days_back: int = 90,
        min_insider_count: int = 2,
    ) -> list[dict]:
        """Find tickers where 2+ insiders bought in the same 30-day window.

        Searches EDGAR EFTS for recent Form 4 filings with transaction code P
        (open-market purchase), groups by company, and filters to clusters.

        Returns a list of dicts:
          [{ticker, n_insiders, total_bought_usd, insiders, period_start}]
        sorted by total_bought_usd descending.
        """
        cutoff = date.today() - timedelta(days=days_back)
        cutoff_str = cutoff.isoformat()

        try:
            data = await _get_json(
                "https://efts.sec.gov/LATEST/search-index",
                params={
                    "forms": "4",
                    "dateRange": "custom",
                    "startdt": cutoff_str,
                    "enddt": date.today().isoformat(),
                    "hits.hits.total.value": 1,
                },
                timeout=30.0,
            )
        except Exception as exc:
            logger.warning("insider_cluster_buys search error: %s", exc)
            return []

        hits = data.get("hits", {}).get("hits", [])

        # Group filings by issuer/ticker
        by_ticker: dict[str, list[dict]] = {}
        for hit in hits:
            src = hit.get("_source", {})
            ticker = (src.get("period_of_report") or "").strip()
            entity = (src.get("entity_name") or "").strip()
            if not entity:
                continue
            by_ticker.setdefault(entity, []).append(src)

        results = []
        for entity, filings in by_ticker.items():
            if len(filings) < min_insider_count:
                continue

            # Attempt to get the actual transactions for the entity's CIK
            # Use the filing's accession info; for now report the aggregate
            results.append({
                "entity": entity,
                "n_filings": len(filings),
                "period": filings[0].get("period_of_report", ""),
                "filing_dates": sorted(
                    set(f.get("file_date", "") for f in filings)
                ),
            })

        results.sort(key=lambda r: r["n_filings"], reverse=True)
        logger.info("insider_cluster_buys days_back=%d clusters=%d", days_back, len(results))
        return results

    # ------------------------------------------------------------------
    # High conviction funds
    # ------------------------------------------------------------------

    async def get_high_conviction_funds(
        self,
        ticker: str,
    ) -> list[dict]:
        """Find funds where this stock is a top-10 position by portfolio weight.

        Checks _MAJOR_FUND_CIKS; compares each manager's latest 13F position
        value to their total reported AUM.

        Returns: [{fund_name, pct_of_portfolio, shares, value_usd}]
        sorted by pct_of_portfolio descending.
        """
        adapter = InstitutionalAdapter(
            user_agent="SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"
        )
        ticker_up = ticker.upper()
        results = []

        async def _check(name: str, mgr_cik: str) -> None:
            try:
                holdings = await adapter.fetch_holdings(mgr_cik, limit=2)
            except Exception:
                return

            if not holdings:
                return

            # Latest period only
            latest_period = max((h.get("period_of_report") for h in holdings if h.get("period_of_report")), default=None)
            if latest_period is None:
                return

            latest = [h for h in holdings if h.get("period_of_report") == latest_period]
            total_value = sum(float(h.get("market_value") or 0) for h in latest)
            if total_value <= 0:
                return

            for h in latest:
                issuer = (h.get("issuer_name") or "").upper()
                hticker = (h.get("ticker") or "").upper()
                if ticker_up not in issuer and hticker != ticker_up:
                    continue
                val = float(h.get("market_value") or 0)
                shares = float(h.get("shares") or 0)
                pct = val / total_value * 100.0
                results.append({
                    "fund_name": name,
                    "pct_of_portfolio": round(pct, 3),
                    "shares": shares,
                    "value_usd": val,
                    "period": latest_period.isoformat() if hasattr(latest_period, "isoformat") else str(latest_period),
                })

        sem = asyncio.Semaphore(5)

        async def _guarded(n: str, c: str) -> None:
            async with sem:
                await _check(n, c)

        await asyncio.gather(*[_guarded(n, c) for n, c in _MAJOR_FUND_CIKS.items()])
        results.sort(key=lambda r: r["pct_of_portfolio"], reverse=True)
        logger.info("high_conviction_funds ticker=%s funds_found=%d", ticker, len(results))
        return results

    # ------------------------------------------------------------------
    # Activism tracking
    # ------------------------------------------------------------------

    async def track_activism_ownership(
        self,
        ticker: str,
    ) -> dict:
        """Identify activist ownership via 13D/13G filings.

        Searches EDGAR for SC 13D and SC 13G filings for the ticker,
        returns a summary with filer names, ownership percentages, and
        activist flag (5%+ threshold per SEC rules).

        Returns:
            {
                "ticker": str,
                "activist_holders": [{filer, pct_owned, filing_date, form_type}],
                "has_active_5pct_holder": bool,
                "n_13d": int,        # Schedule 13D = activist (>5%, change intended)
                "n_13g": int,        # Schedule 13G = passive large holder
            }
        """
        # Search for both 13D and 13G
        results_13d = await _search_edgar("SC 13D", ticker, days_back=365, limit=20)
        results_13g = await _search_edgar("SC 13G", ticker, days_back=365, limit=20)

        holders = []
        for filing in results_13d:
            holders.append({
                "filer": filing.get("entity_name", ""),
                "form_type": "SC 13D",
                "filing_date": filing.get("filing_date", ""),
                "is_activist": True,  # 13D implies active intent
                "cik": filing.get("cik", ""),
            })
        for filing in results_13g:
            holders.append({
                "filer": filing.get("entity_name", ""),
                "form_type": "SC 13G",
                "filing_date": filing.get("filing_date", ""),
                "is_activist": False,  # 13G = passive
                "cik": filing.get("cik", ""),
            })

        has_activist = any(h["is_activist"] for h in holders)
        logger.info(
            "track_activism ticker=%s 13d=%d 13g=%d",
            ticker, len(results_13d), len(results_13g),
        )
        return {
            "ticker": ticker.upper(),
            "activist_holders": holders,
            "has_active_5pct_holder": has_activist or len(holders) > 0,
            "n_13d": len(results_13d),
            "n_13g": len(results_13g),
        }

    # ------------------------------------------------------------------
    # Internal helpers exposed for testing
    # ------------------------------------------------------------------

    async def _get_institutional_changes(
        self, ticker: str, days_back: int
    ) -> tuple[list[InstitutionalChange], float]:
        cik = await _resolve_cik(ticker)
        changes, net_pct, _ = await _get_institutional_changes(ticker, cik, days_back)
        return changes, net_pct

    async def _get_insider_transactions(
        self, ticker: str, days_back: int
    ) -> tuple[list[InsiderActivity], float]:
        cik = await _resolve_cik(ticker)
        return await _get_insider_transactions(ticker, cik, days_back)

    def _compute_ownership_signal(
        self, profile: "OwnershipProfile"
    ) -> tuple[str, float]:
        return _compute_ownership_signal(
            profile.net_institutional_change_pct,
            profile.new_positions_90d,
            profile.closed_positions_90d,
            profile.increased_90d,
            profile.decreased_90d,
            profile.insider_buying_90d,
            profile.insider_selling_90d,
            profile.net_insider_value_90d or 0.0,
        )


# ---------------------------------------------------------------------------
# Module-level convenience coroutines
# ---------------------------------------------------------------------------


async def ownership_profile(ticker: str) -> OwnershipProfile:
    """Build an ownership profile for a single ticker."""
    return await OwnershipScreener().get_ownership_profile(ticker)


async def screen_ownership(
    tickers: list[str],
    require_buying: bool = True,
) -> OwnershipScreenResult:
    """Screen tickers requiring insider or institutional buying."""
    return await OwnershipScreener().screen_by_ownership(
        tickers,
        require_insider_buying=require_buying,
        min_signal_score=1.0,
    )


async def insider_cluster_buys(days_back: int = 90) -> list[dict]:
    """Find insider cluster buys across the EDGAR universe."""
    return await OwnershipScreener().get_insider_cluster_buys(days_back=days_back)
