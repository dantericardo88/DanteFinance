"""Multi-source OHLCV DataCleaner.

Fetches from ALL adapters in parallel, computes a cross-source consensus
(median OHLCV), scores data quality per source, and optionally writes
the consensus + provenance receipt to the database.

Typical usage:
    cleaner = DataCleaner(session)
    report = await cleaner.clean_ticker("AAPL", interval="1d")
    print(report.best_source(), report.consensus_bars)

    reports = await cleaner.clean_universe(["AAPL", "MSFT", "GOOG"])
"""
from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.core.logging import get_logger
from sentinel.core.types import OHLCVBar

logger = get_logger(__name__)

# Grade ordering for best_source() priority lookup
_GRADE_RANK: dict[str, int] = {"A": 4, "B": 3, "C": 2, "F": 1}


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class SourceQualityScore:
    """Per-source quality metrics relative to the consensus bars."""

    source: str
    bars_returned: int
    bars_expected: int
    availability: float        # 0.0–1.0
    mean_abs_deviation: float  # deviation from consensus close, 0.0–1.0
    outlier_bars: int
    grade: str                 # "A" | "B" | "C" | "F"


@dataclass
class CleaningReport:
    """Result of a single-ticker cleaning pass."""

    ticker: str
    interval: str
    start: datetime
    end: datetime
    sources_attempted: list[str]
    sources_with_data: list[str]
    sources_rejected: list[str]   # failed single-source validation
    consensus_bars: int
    bars_outlier_flagged: int
    gaps_detected: int
    quality_scores: list[SourceQualityScore]
    wrote_to_db: bool
    as_of: datetime

    def best_source(self) -> str:
        """Return the name of the highest-grade source (A > B > C > F).

        Ties broken by availability descending, then mean_abs_deviation ascending.
        Returns empty string when no quality scores exist.
        """
        if not self.quality_scores:
            return ""
        return max(
            self.quality_scores,
            key=lambda s: (
                _GRADE_RANK.get(s.grade, 0),
                s.availability,
                -s.mean_abs_deviation,
            ),
        ).source

    def to_dict(self) -> dict:
        """JSON-serializable dict. All datetimes are ISO-8601 strings."""
        return {
            "ticker": self.ticker,
            "interval": self.interval,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "sources_attempted": self.sources_attempted,
            "sources_with_data": self.sources_with_data,
            "sources_rejected": self.sources_rejected,
            "consensus_bars": self.consensus_bars,
            "bars_outlier_flagged": self.bars_outlier_flagged,
            "gaps_detected": self.gaps_detected,
            "quality_scores": [
                {
                    "source": qs.source,
                    "bars_returned": qs.bars_returned,
                    "bars_expected": qs.bars_expected,
                    "availability": qs.availability,
                    "mean_abs_deviation": qs.mean_abs_deviation,
                    "outlier_bars": qs.outlier_bars,
                    "grade": qs.grade,
                }
                for qs in self.quality_scores
            ],
            "wrote_to_db": self.wrote_to_db,
            "as_of": self.as_of.isoformat(),
        }


# ── DataCleaner ───────────────────────────────────────────────────────────────

class DataCleaner:
    """Multi-source OHLCV pipeline: parallel fetch → consensus → quality gate → DB write."""

    # Deviation threshold above which a single source's bar is flagged as an outlier
    OUTLIER_THRESHOLD = 0.02   # 2% from median close

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── public API ────────────────────────────────────────────────────────────

    async def clean_ticker(
        self,
        ticker: str,
        interval: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        write_to_db: bool = True,
    ) -> CleaningReport:
        """Run the full cleaning pipeline for one ticker.

        Steps:
          1. Parallel fetch from ALL adapters.
          2. Per-source single-source validation.
          3. Build source_map of validated bars.
          4. Cross-source consensus (median OHLCV).
          5. Deduplicate and sort consensus bars.
          6. Gap detection.
          7. Write to DB (optional).
          8. Compute SourceQualityScore per source.
          9. Persist quality scores to DB (best-effort).
         10. Return CleaningReport.
        """
        from sentinel.sds import get_all_adapters, OHLCV_CHAIN
        from sentinel.sds.validator import validate_single_source
        from sentinel.sds.normalizer import deduplicate_bars
        from sentinel.sds.gap_detector import detect_gaps
        from sentinel.sds.provenance import create_receipt
        from sentinel.sds import repository

        # Defaults
        end_dt = end or datetime.utcnow()
        start_dt = start or (end_dt - timedelta(days=365))

        all_adapters = get_all_adapters()
        # Restrict to the standard OHLCV chain so we only query market-data adapters
        adapters = {
            name: all_adapters[name]
            for name in OHLCV_CHAIN
            if name in all_adapters
        }

        sources_attempted: list[str] = list(adapters.keys())
        sources_with_data: list[str] = []
        sources_rejected: list[str] = []

        # ── Step 1: Parallel fetch ────────────────────────────────────────────
        async def _fetch_one(name: str, adapter) -> tuple[str, list[OHLCVBar]]:
            try:
                bars = await adapter.fetch_ohlcv(ticker, start_dt, end_dt, interval)
                return name, bars if bars else []
            except Exception as exc:
                logger.warning(
                    "DataCleaner: adapter fetch failed",
                    adapter=name, ticker=ticker, error=str(exc),
                )
                return name, []

        tasks = [_fetch_one(name, adapter) for name, adapter in adapters.items()]
        raw_results: list[tuple[str, list[OHLCVBar]]] = []

        gather_results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in gather_results:
            if isinstance(result, Exception):
                logger.warning(
                    "DataCleaner: gather returned exception", error=str(result)
                )
                continue
            raw_results.append(result)

        # ── Step 2: Per-source validation ─────────────────────────────────────
        source_map: dict[str, list[OHLCVBar]] = {}

        for source_name, raw_bars in raw_results:
            if not raw_bars:
                continue
            validated_bars, _report = validate_single_source(raw_bars, ticker)
            if not validated_bars:
                sources_rejected.append(source_name)
                logger.warning(
                    "DataCleaner: all bars rejected by validator",
                    source=source_name, ticker=ticker,
                )
                continue
            source_map[source_name] = validated_bars
            sources_with_data.append(source_name)

        # ── Step 3–4: Cross-source consensus ──────────────────────────────────
        # Track per-source deviations and outlier counts for quality scoring.
        # Structure: source_name → list[abs_deviation_float]
        source_deviations: dict[str, list[float]] = {s: [] for s in source_map}
        source_outlier_counts: dict[str, int] = {s: 0 for s in source_map}

        consensus_bars: list[OHLCVBar] = []
        bars_outlier_flagged: int = 0

        if source_map:
            # Gather all unique timestamps across every source.
            # For daily bars, normalise to UTC midnight so timestamps align.
            all_timestamps: set[datetime] = set()
            for bars in source_map.values():
                for bar in bars:
                    ts = _normalise_ts(bar.time, interval)
                    all_timestamps.add(ts)

            # Build per-source lookup: normalised_ts → bar
            source_lookups: dict[str, dict[datetime, OHLCVBar]] = {}
            for src_name, bars in source_map.items():
                source_lookups[src_name] = {
                    _normalise_ts(b.time, interval): b for b in bars
                }

            # Derive a reference figi from whichever source has bars first
            reference_figi = ""
            for src_name in OHLCV_CHAIN:
                if src_name in source_map and source_map[src_name]:
                    reference_figi = source_map[src_name][0].figi
                    break

            for ts in sorted(all_timestamps):
                present: list[tuple[str, OHLCVBar]] = [
                    (src, source_lookups[src][ts])
                    for src in source_map
                    if ts in source_lookups[src]
                ]

                if len(present) == 1:
                    # Only one source — use it verbatim; deviation = 0
                    src_name, bar = present[0]
                    source_deviations[src_name].append(0.0)
                    consensus_bars.append(bar)
                    continue

                # 2+ sources: compute median for OHLCV, max volume
                opens  = [float(b.open)   for _, b in present]
                highs  = [float(b.high)   for _, b in present]
                lows   = [float(b.low)    for _, b in present]
                closes = [float(b.close)  for _, b in present]
                vols   = [float(b.volume) for _, b in present]

                median_open  = statistics.median(opens)
                median_high  = statistics.median(highs)
                median_low   = statistics.median(lows)
                median_close = statistics.median(closes)
                max_volume   = max(vols)

                # Track per-source deviations; flag outliers
                any_outlier = False
                for src_name, bar in present:
                    close_f = float(bar.close)
                    if median_close != 0.0:
                        dev = abs(close_f - median_close) / median_close
                    else:
                        dev = 0.0
                    source_deviations[src_name].append(dev)
                    if dev > self.OUTLIER_THRESHOLD:
                        source_outlier_counts[src_name] += 1
                        any_outlier = True

                if any_outlier:
                    bars_outlier_flagged += 1

                contributing = ",".join(src for src, _ in present)
                consensus_source = f"consensus:{contributing}"

                consensus_bar = OHLCVBar(
                    time=ts,
                    figi=reference_figi,
                    ticker=ticker,
                    open=Decimal(str(median_open)),
                    high=Decimal(str(median_high)),
                    low=Decimal(str(median_low)),
                    close=Decimal(str(median_close)),
                    volume=int(max_volume),
                    source=consensus_source,
                )
                consensus_bars.append(consensus_bar)

        # ── Step 5: Dedup and sort ─────────────────────────────────────────────
        consensus_bars.sort(key=lambda b: b.time)
        consensus_bars = deduplicate_bars(consensus_bars)

        # ── Step 6: Gap detection ──────────────────────────────────────────────
        gap_report = detect_gaps(
            consensus_bars,
            ticker,
            start_dt.date(),
            end_dt.date(),
            interval=interval,
        )
        gaps_detected: int = gap_report.gap_count

        # ── Step 7: Write to DB ────────────────────────────────────────────────
        wrote_to_db = False
        if write_to_db and consensus_bars:
            try:
                await repository.write_ohlcv_bars(
                    consensus_bars, ticker, interval, self._session
                )
                receipt = create_receipt(
                    consensus_bars,
                    source="data_cleaner",
                    ticker=ticker,
                    interval=interval,
                    figi=consensus_bars[0].figi if consensus_bars else None,
                )
                await repository.write_provenance_receipt(receipt, self._session)
                wrote_to_db = True
                logger.info(
                    "DataCleaner: consensus written to DB",
                    ticker=ticker, bars=len(consensus_bars), interval=interval,
                )
            except Exception as exc:
                logger.error(
                    "DataCleaner: DB write failed",
                    ticker=ticker, error=str(exc),
                )

        # ── Step 8: Compute SourceQualityScore ────────────────────────────────
        total_consensus = max(len(consensus_bars), 1)
        quality_scores: list[SourceQualityScore] = []

        for src_name in sources_with_data:
            src_bars_count = len(source_map.get(src_name, []))
            availability = src_bars_count / total_consensus
            deviations = source_deviations.get(src_name, [])
            mean_dev = (
                statistics.mean(deviations) if deviations else 0.0
            )
            outlier_ct = source_outlier_counts.get(src_name, 0)

            if availability >= 0.99 and mean_dev <= 0.001:
                grade = "A"
            elif availability >= 0.95 and mean_dev <= 0.005:
                grade = "B"
            elif availability >= 0.90 and mean_dev <= 0.01:
                grade = "C"
            else:
                grade = "F"

            quality_scores.append(SourceQualityScore(
                source=src_name,
                bars_returned=src_bars_count,
                bars_expected=total_consensus,
                availability=availability,
                mean_abs_deviation=mean_dev,
                outlier_bars=outlier_ct,
                grade=grade,
            ))

        # ── Step 9: Persist quality scores (best-effort) ──────────────────────
        if quality_scores:
            try:
                as_of_now = datetime.utcnow()
                score_dicts = [
                    {
                        "ticker": ticker,
                        "interval": interval,
                        "source": qs.source,
                        "as_of": as_of_now,
                        "bars_returned": qs.bars_returned,
                        "bars_expected": qs.bars_expected,
                        "availability": qs.availability,
                        "mean_abs_deviation": qs.mean_abs_deviation,
                        "outlier_bars": qs.outlier_bars,
                        "grade": qs.grade,
                    }
                    for qs in quality_scores
                ]
                await repository.write_quality_scores(score_dicts, self._session)
            except Exception as exc:
                logger.warning(
                    "DataCleaner: quality score write skipped",
                    error=str(exc),
                )

        # ── Step 10: Build and return report ──────────────────────────────────
        return CleaningReport(
            ticker=ticker,
            interval=interval,
            start=start_dt,
            end=end_dt,
            sources_attempted=sources_attempted,
            sources_with_data=sources_with_data,
            sources_rejected=sources_rejected,
            consensus_bars=len(consensus_bars),
            bars_outlier_flagged=bars_outlier_flagged,
            gaps_detected=gaps_detected,
            quality_scores=quality_scores,
            wrote_to_db=wrote_to_db,
            as_of=datetime.utcnow(),
        )

    async def clean_universe(
        self,
        tickers: list[str],
        interval: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        max_concurrent: int = 5,
        write_to_db: bool = True,
    ) -> list[CleaningReport]:
        """Clean all tickers with bounded concurrency.

        Uses an asyncio.Semaphore of ``max_concurrent`` to avoid hammering
        adapters with too many simultaneous requests.

        Returns a list of CleaningReport in the same order as ``tickers``.
        Failed tickers produce a minimal error report rather than propagating
        the exception to the caller.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _clean_one(ticker: str) -> CleaningReport:
            async with semaphore:
                try:
                    return await self.clean_ticker(
                        ticker=ticker,
                        interval=interval,
                        start=start,
                        end=end,
                        write_to_db=write_to_db,
                    )
                except Exception as exc:
                    logger.error(
                        "DataCleaner: clean_universe ticker failed",
                        ticker=ticker, error=str(exc),
                    )
                    now = datetime.utcnow()
                    end_dt = end or now
                    start_dt = start or (end_dt - timedelta(days=365))
                    return CleaningReport(
                        ticker=ticker,
                        interval=interval,
                        start=start_dt,
                        end=end_dt,
                        sources_attempted=[],
                        sources_with_data=[],
                        sources_rejected=[],
                        consensus_bars=0,
                        bars_outlier_flagged=0,
                        gaps_detected=0,
                        quality_scores=[],
                        wrote_to_db=False,
                        as_of=now,
                    )

        tasks = [_clean_one(ticker) for ticker in tickers]
        reports: list[CleaningReport] = list(await asyncio.gather(*tasks))
        return reports


# ── helpers ───────────────────────────────────────────────────────────────────

_DAILY_INTERVALS = {"1d", "1D", "day", "daily", "1wk", "1mo"}


def _normalise_ts(ts: datetime, interval: str) -> datetime:
    """Normalise a timestamp to UTC midnight for daily+ bars.

    This ensures timestamps from different adapters that may disagree on the
    exact time-of-day component (e.g. 00:00 UTC vs 20:00 EST) align correctly
    when building the cross-source consensus map.

    Intraday intervals are returned unchanged.
    """
    if interval in _DAILY_INTERVALS:
        return ts.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    return ts
