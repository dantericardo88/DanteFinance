"""Continuous futures construction — stitches raw contract chains into a
single price series free of roll gaps.

Three adjustment methods:
  PANAMA   — additive: subtract/add the roll gap at each roll date so the
              oldest bars are shifted. Preserves absolute price differences
              but not ratios. Standard for spread/basis strategies.
  RATIO    — multiplicative: scale all bars before each roll by the ratio
              (next_open / prev_close). Preserves percentage moves.
              Standard for momentum/trend strategies.
  UNADJ    — no price adjustment; pure chain concatenation for volume
              or open-interest analysis.

Usage
-----
    from sentinel.sds.continuous_futures import build_continuous_series, ContractSpec

    spec = ContractSpec(root="ES", exchange="CME", months=[3, 6, 9, 12])
    bars = await build_continuous_series(session, spec, start, end, method="panama")
    # bars: list[OHLCVBar] with ticker="CONT:ES1", figi="CONT:ES"
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.core.logging import get_logger
from sentinel.core.types import OHLCVBar

logger = get_logger(__name__)


# ── Enums & config ─────────────────────────────────────────────────────────────

class AdjustmentMethod(str, Enum):
    PANAMA = "panama"
    RATIO  = "ratio"
    UNADJ  = "unadj"


@dataclass
class ContractSpec:
    """Defines a futures product and its rolling schedule.

    Attributes:
        root:             Bloomberg/CME root symbol, e.g. "ES", "CL", "GC".
        exchange:         Exchange code, e.g. "CME", "NYMEX", "CBOT".
        months:           Contract expiry months as integers (1=Jan … 12=Dec).
        roll_days_before: Number of calendar days before the last trading day
                          to roll to the next contract.  Defaults to 5 (one
                          week), which avoids delivery month liquidity traps.
        calendar:         Optional override of last-trading dates per contract
                          as {YYYYMM: last_trade_date}.  If empty, the module
                          infers LTD as the third Friday of the expiry month
                          (a common but imperfect heuristic — use the override
                          for precision).
    """
    root: str
    exchange: str
    months: list[int]
    roll_days_before: int = 5
    calendar: dict[str, date] = field(default_factory=dict)

    # ── Well-known specs ──────────────────────────────────────────────────────

    @classmethod
    def es(cls) -> "ContractSpec":
        """E-mini S&P 500 (CME, quarterly)."""
        return cls(root="ES", exchange="CME", months=[3, 6, 9, 12])

    @classmethod
    def nq(cls) -> "ContractSpec":
        """E-mini Nasdaq-100 (CME, quarterly)."""
        return cls(root="NQ", exchange="CME", months=[3, 6, 9, 12])

    @classmethod
    def cl(cls) -> "ContractSpec":
        """Crude Oil WTI (NYMEX, monthly)."""
        return cls(root="CL", exchange="NYMEX",
                   months=list(range(1, 13)), roll_days_before=3)

    @classmethod
    def gc(cls) -> "ContractSpec":
        """Gold (COMEX, even months)."""
        return cls(root="GC", exchange="COMEX", months=[2, 4, 6, 8, 10, 12])

    @classmethod
    def zb(cls) -> "ContractSpec":
        """30-Year US Treasury Bond (CBOT, quarterly)."""
        return cls(root="ZB", exchange="CBOT", months=[3, 6, 9, 12])


# ── Contract-month helpers ─────────────────────────────────────────────────────

_MONTH_CODES = {1:"F",2:"G",3:"H",4:"J",5:"K",6:"M",7:"N",8:"Q",9:"U",10:"V",11:"X",12:"Z"}


def _contract_ticker(root: str, year: int, month: int) -> str:
    """Build a standard contract ticker, e.g. 'ESH24'."""
    return f"{root}{_MONTH_CODES[month]}{str(year)[-2:]}"


def _third_friday(year: int, month: int) -> date:
    """Return the third Friday of the given month."""
    d = date(year, month, 1)
    # Advance to the first Friday
    d += timedelta(days=(4 - d.weekday()) % 7)
    return d + timedelta(weeks=2)


def _last_trading_date(spec: ContractSpec, year: int, month: int) -> date:
    key = f"{year}{month:02d}"
    if key in spec.calendar:
        return spec.calendar[key]
    # Heuristic: third Friday of the contract month (CME equity futures)
    return _third_friday(year, month)


def _roll_date(spec: ContractSpec, year: int, month: int) -> date:
    ltd = _last_trading_date(spec, year, month)
    return ltd - timedelta(days=spec.roll_days_before)


def _contract_sequence(spec: ContractSpec, start: date, end: date) -> list[tuple[str, date, date]]:
    """Build [(ticker, roll_from, roll_to), …] covering [start, end].

    Each tuple gives the active contract ticker and the calendar range during
    which it is the front-month contract.
    """
    # Generate (year, month) pairs for contracts that could be active
    contracts: list[tuple[int, int, date]] = []  # (year, month, roll_date)
    for yr in range(start.year - 1, end.year + 2):
        for mo in spec.months:
            rd = _roll_date(spec, yr, mo)
            contracts.append((yr, mo, rd))

    # Sort by roll date (ascending) and filter to window
    contracts.sort(key=lambda x: x[2])

    # Build sequence: each contract is active from the previous roll_date
    # (exclusive) through its own roll_date (inclusive)
    result: list[tuple[str, date, date]] = []
    prev_roll = date(start.year - 1, 1, 1)
    for yr, mo, rd in contracts:
        if rd <= prev_roll:
            continue
        active_from = prev_roll + timedelta(days=1)
        active_to   = rd
        if active_to < start:
            prev_roll = rd
            continue
        if active_from > end:
            break
        ticker = _contract_ticker(spec.root, yr, mo)
        result.append((ticker, max(active_from, start), min(active_to, end)))
        prev_roll = rd

    return result


# ── Database helpers ───────────────────────────────────────────────────────────

async def _fetch_bars(
    session: AsyncSession,
    ticker: str,
    start: date,
    end: date,
) -> list[dict]:
    """Pull OHLCV rows for a single contract ticker from the database."""
    sql = text("""
        SELECT time, open, high, low, close, volume, source
        FROM ohlcv
        WHERE ticker = :ticker
          AND time >= :start
          AND time <= :end
          AND interval = '1d'
        ORDER BY time ASC
    """)
    result = await session.execute(sql, {
        "ticker": ticker,
        "start": datetime.combine(start, datetime.min.time()),
        "end":   datetime.combine(end,   datetime.max.time()),
    })
    rows = result.mappings().fetchall()
    return [dict(r) for r in rows]


# ── Core algorithm ─────────────────────────────────────────────────────────────

def _apply_panama(segments: list[list[dict]]) -> list[dict]:
    """Additive back-adjustment: shift all older bars by each roll gap."""
    if not segments:
        return []
    # Work backwards: last segment is unadjusted; each prior segment is
    # shifted by (next_open - prev_close) at the roll boundary.
    adjusted: list[list[dict]] = []
    cumulative_shift = Decimal("0")

    for i in range(len(segments) - 1, -1, -1):
        seg = segments[i]
        if not seg:
            adjusted.insert(0, [])
            continue

        if i < len(segments) - 1:
            # Compute roll gap: next contract's first open vs this contract's last close
            next_seg = segments[i + 1]
            if next_seg and seg:
                next_open  = Decimal(str(next_seg[0]["open"]))
                prev_close = Decimal(str(seg[-1]["close"]))
                gap = next_open - prev_close
                cumulative_shift += gap

        shifted = []
        for bar in seg:
            shifted.append({**bar,
                "open":  Decimal(str(bar["open"]))  + cumulative_shift,
                "high":  Decimal(str(bar["high"]))  + cumulative_shift,
                "low":   Decimal(str(bar["low"]))   + cumulative_shift,
                "close": Decimal(str(bar["close"])) + cumulative_shift,
            })
        adjusted.insert(0, shifted)

    return [bar for seg in adjusted for bar in seg]


def _apply_ratio(segments: list[list[dict]]) -> list[dict]:
    """Multiplicative back-adjustment: scale older bars by roll ratios."""
    if not segments:
        return []
    adjusted: list[list[dict]] = []
    cumulative_ratio = Decimal("1")

    for i in range(len(segments) - 1, -1, -1):
        seg = segments[i]
        if not seg:
            adjusted.insert(0, [])
            continue

        if i < len(segments) - 1:
            next_seg = segments[i + 1]
            if next_seg and seg:
                next_open  = Decimal(str(next_seg[0]["open"]))
                prev_close = Decimal(str(seg[-1]["close"]))
                if prev_close != 0:
                    ratio = next_open / prev_close
                    cumulative_ratio *= ratio

        scaled = []
        for bar in seg:
            scaled.append({**bar,
                "open":  (Decimal(str(bar["open"]))  * cumulative_ratio).quantize(Decimal("0.01")),
                "high":  (Decimal(str(bar["high"]))  * cumulative_ratio).quantize(Decimal("0.01")),
                "low":   (Decimal(str(bar["low"]))   * cumulative_ratio).quantize(Decimal("0.01")),
                "close": (Decimal(str(bar["close"])) * cumulative_ratio).quantize(Decimal("0.01")),
            })
        adjusted.insert(0, scaled)

    return [bar for seg in adjusted for bar in seg]


def _apply_unadj(segments: list[list[dict]]) -> list[dict]:
    """No adjustment — concatenate segments as-is."""
    return [bar for seg in segments for bar in seg]


# ── Public API ─────────────────────────────────────────────────────────────────

async def build_continuous_series(
    session: AsyncSession,
    spec: ContractSpec,
    start: date,
    end: date,
    method: str = "panama",
) -> list[OHLCVBar]:
    """Build a continuous futures price series from individual contracts.

    Fetches raw contract data from the `ohlcv` table, applies roll logic and
    price adjustment, and returns the stitched series as OHLCVBar objects
    labelled with ticker ``CONT:{root}1`` and figi ``CONT:{root}``.

    Args:
        session:  Async SQLAlchemy session.
        spec:     ContractSpec describing the product.
        start:    First date of the desired series.
        end:      Last date of the desired series.
        method:   "panama" | "ratio" | "unadj".

    Returns:
        List of OHLCVBar sorted by time, adjusted per `method`.
        Returns [] if no data is found for any contract in the range.
    """
    adj = AdjustmentMethod(method)
    seq = _contract_sequence(spec, start, end)
    if not seq:
        logger.warning("No contracts in range", root=spec.root, start=str(start), end=str(end))
        return []

    logger.info("Fetching %d contract segments", len(seq),
                root=spec.root, method=method)

    # Fetch all segments concurrently
    tasks = [_fetch_bars(session, ticker, from_dt, to_dt)
             for ticker, from_dt, to_dt in seq]
    segments = await asyncio.gather(*tasks)

    if adj == AdjustmentMethod.PANAMA:
        raw_bars = _apply_panama(list(segments))
    elif adj == AdjustmentMethod.RATIO:
        raw_bars = _apply_ratio(list(segments))
    else:
        raw_bars = _apply_unadj(list(segments))

    if not raw_bars:
        logger.warning("No bars after adjustment", root=spec.root)
        return []

    cont_ticker = f"CONT:{spec.root}1"
    cont_figi   = f"CONT:{spec.root}"

    result: list[OHLCVBar] = []
    for bar in raw_bars:
        try:
            result.append(OHLCVBar(
                time=bar["time"],
                figi=cont_figi,
                ticker=cont_ticker,
                open=Decimal(str(bar["open"])),
                high=Decimal(str(bar["high"])),
                low=Decimal(str(bar["low"])),
                close=Decimal(str(bar["close"])),
                volume=int(bar.get("volume") or 0),
                source=bar.get("source", "continuous"),
                interval="1d",
            ))
        except Exception as exc:
            logger.warning("Skipping malformed bar", error=str(exc))

    logger.info("Continuous series built", root=spec.root, bars=len(result),
                method=method, start=str(start), end=str(end))
    return result


async def write_continuous_series(
    session: AsyncSession,
    bars: list[OHLCVBar],
) -> int:
    """Persist a continuous series to the ohlcv table.

    Uses INSERT ... ON CONFLICT DO UPDATE so re-running is idempotent.
    Returns the number of rows upserted.
    """
    if not bars:
        return 0

    from sentinel.sds.repository import write_ohlcv_bars
    written = await write_ohlcv_bars(bars, session)
    logger.info("Continuous series persisted", bars=written)
    return written


# ── Convenience: build and persist in one call ──────────────────────────────

async def build_and_persist_continuous(
    session: AsyncSession,
    spec: ContractSpec,
    start: date,
    end: date,
    method: str = "panama",
) -> int:
    """Build continuous series and write to DB. Returns bar count."""
    bars = await build_continuous_series(session, spec, start, end, method)
    return await write_continuous_series(session, bars)
