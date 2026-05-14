"""Data lake catalog — coverage, quality, and gap analysis.

Answers the question: "What is actually in the database?"
Run at any time to see exactly what's been ingested.

Usage:
    from sentinel.sds.catalog import get_coverage, format_catalog_report
    async with session_factory()() as session:
        summary = await get_coverage(session)
        print(format_catalog_report(summary))

CLI: python scripts/backfill.py --status
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TickerCoverage:
    ticker: str
    figi: str
    interval: str
    first_date: date
    last_date: date
    row_count: int
    sources: list[str] = field(default_factory=list)

    @property
    def years_covered(self) -> float:
        return (self.last_date - self.first_date).days / 365.25


@dataclass
class MacroCoverage:
    series_id: str
    first_date: date
    last_date: date
    row_count: int


@dataclass
class CatalogSummary:
    tickers: list[TickerCoverage]
    macro_series: list[MacroCoverage]
    total_ohlcv_rows: int
    total_macro_rows: int
    total_fact_rows: int
    total_provenance_records: int
    delisted_in_registry: int
    ca_records: int
    insider_count: int
    institutional_count: int
    news_count: int
    cot_count: int
    congress_count: int
    as_of: datetime


async def get_coverage(session: AsyncSession) -> CatalogSummary:
    """Return a full snapshot of what's in the data lake."""

    # OHLCV coverage by ticker + interval
    ohlcv_rows = await session.execute(text("""
        SELECT
            o.figi,
            COALESCE(i.ticker, o.figi) AS ticker,
            o.interval,
            MIN(o.time)::date            AS first_date,
            MAX(o.time)::date            AS last_date,
            COUNT(*)                     AS row_count,
            ARRAY_AGG(DISTINCT o.source) AS sources
        FROM ohlcv o
        LEFT JOIN instruments i ON o.figi = i.figi
        GROUP BY o.figi, i.ticker, o.interval
        ORDER BY ticker, o.interval
    """))
    tickers = [
        TickerCoverage(
            ticker=row["ticker"],
            figi=row["figi"],
            interval=row["interval"],
            first_date=row["first_date"],
            last_date=row["last_date"],
            row_count=row["row_count"],
            sources=list(row["sources"] or []),
        )
        for row in ohlcv_rows.mappings()
    ]

    # Macro series coverage
    macro_rows = await session.execute(text("""
        SELECT
            series_id,
            MIN(time)::date AS first_date,
            MAX(time)::date AS last_date,
            COUNT(*)        AS row_count
        FROM macro_data
        GROUP BY series_id
        ORDER BY series_id
    """))
    macro_series = [
        MacroCoverage(
            series_id=row["series_id"],
            first_date=row["first_date"],
            last_date=row["last_date"],
            row_count=row["row_count"],
        )
        for row in macro_rows.mappings()
    ]

    # Aggregate row counts across all tables
    counts = (await session.execute(text("""
        SELECT
            (SELECT COUNT(*) FROM ohlcv)                    AS ohlcv_rows,
            (SELECT COUNT(*) FROM macro_data)               AS macro_rows,
            (SELECT COUNT(*) FROM financial_facts)          AS fact_rows,
            (SELECT COUNT(*) FROM data_provenance)          AS prov_rows,
            (SELECT COUNT(*) FROM survivorship_registry)    AS surv_rows,
            (SELECT COUNT(*) FROM corporate_actions)        AS ca_rows,
            (SELECT COUNT(*) FROM insider_transactions)     AS insider_rows,
            (SELECT COUNT(*) FROM institutional_holdings)   AS institutional_rows,
            (SELECT COUNT(*) FROM news_articles)            AS news_rows,
            (SELECT COUNT(*) FROM cot_data)                 AS cot_rows,
            (SELECT COUNT(*) FROM congressional_trades)     AS congress_rows
    """))).mappings().fetchone()

    c = dict(counts) if counts else {}
    return CatalogSummary(
        tickers=tickers,
        macro_series=macro_series,
        total_ohlcv_rows=c.get("ohlcv_rows", 0),
        total_macro_rows=c.get("macro_rows", 0),
        total_fact_rows=c.get("fact_rows", 0),
        total_provenance_records=c.get("prov_rows", 0),
        delisted_in_registry=c.get("surv_rows", 0),
        ca_records=c.get("ca_rows", 0),
        insider_count=c.get("insider_rows", 0),
        institutional_count=c.get("institutional_rows", 0),
        news_count=c.get("news_rows", 0),
        cot_count=c.get("cot_rows", 0),
        congress_count=c.get("congress_rows", 0),
        as_of=datetime.utcnow(),
    )


async def get_missing_tickers(
    universe: list[str],
    interval: str,
    session: AsyncSession,
) -> list[str]:
    """Return tickers in universe that have zero OHLCV rows in the lake."""
    result = await session.execute(
        text("""
            SELECT DISTINCT COALESCE(i.ticker, o.figi) AS ticker
            FROM ohlcv o
            LEFT JOIN instruments i ON o.figi = i.figi
            WHERE o.interval = :interval
        """),
        dict(interval=interval),
    )
    have_data = {row[0] for row in result}
    return [t for t in universe if t not in have_data]


async def get_data_quality_stats(session: AsyncSession) -> dict:
    """Provenance-based quality statistics: pass rates, sources, gap counts."""
    result = await session.execute(text("""
        SELECT
            source,
            COUNT(*)                             AS batches,
            SUM(bar_count)                       AS total_bars,
            AVG(bar_count)                       AS avg_bars_per_batch,
            COUNT(*) FILTER (WHERE validated)    AS validated_batches,
            MIN(ingested_at)                     AS first_ingest,
            MAX(ingested_at)                     AS last_ingest
        FROM data_provenance
        GROUP BY source
        ORDER BY total_bars DESC
    """))
    return {
        "by_source": [dict(row) for row in result.mappings()],
        "as_of": datetime.utcnow().isoformat(),
    }


def format_catalog_report(summary: CatalogSummary) -> str:
    """Human-readable report for CLI output."""
    w = 64
    lines = [
        "=" * w,
        " SENTINEL Data Lake Catalog".center(w),
        f" {summary.as_of.strftime('%Y-%m-%d %H:%M UTC')}".center(w),
        "=" * w,
        f"  {'OHLCV bars:':<26} {summary.total_ohlcv_rows:>12,}",
        f"  {'Macro observations:':<26} {summary.total_macro_rows:>12,}",
        f"  {'XBRL financial facts:':<26} {summary.total_fact_rows:>12,}",
        f"  {'Provenance receipts:':<26} {summary.total_provenance_records:>12,}",
        f"  {'Corporate action records:':<26} {summary.ca_records:>12,}",
        f"  {'Delisted in registry:':<26} {summary.delisted_in_registry:>12,}",
        f"  {'Insider transactions:':<26} {summary.insider_count:>12,}",
        f"  {'Institutional holdings:':<26} {summary.institutional_count:>12,}",
        f"  {'News articles:':<26} {summary.news_count:>12,}",
        f"  {'COT records:':<26} {summary.cot_count:>12,}",
        f"  {'Congressional trades:':<26} {summary.congress_count:>12,}",
        "",
        f"  {'Tickers with daily OHLCV:':<26} {len([t for t in summary.tickers if t.interval == '1d']):>12,}",
        f"  {'Macro series:':<26} {len(summary.macro_series):>12,}",
    ]

    daily = sorted(
        [t for t in summary.tickers if t.interval == "1d"],
        key=lambda t: t.row_count,
        reverse=True,
    )
    if daily:
        lines += ["", "  Top tickers (daily bars):", "  " + "-" * 60]
        for t in daily[:15]:
            src = "/".join(t.sources)
            lines.append(
                f"  {t.ticker:<8}  {t.first_date}→{t.last_date}"
                f"  {t.row_count:>6,} bars  {t.years_covered:.1f}y  [{src}]"
            )
        if len(daily) > 15:
            lines.append(f"  ... and {len(daily) - 15} more tickers")

    if summary.macro_series:
        lines += ["", "  Macro series:", "  " + "-" * 60]
        for m in summary.macro_series[:20]:
            lines.append(
                f"  {m.series_id:<14}  {m.first_date}→{m.last_date}"
                f"  {m.row_count:>6,} obs"
            )

    if summary.total_ohlcv_rows == 0:
        lines += [
            "",
            "  ⚠  Data lake is EMPTY. Run: make backfill",
            "     Or: python scripts/backfill.py --mode all --years 10",
        ]

    lines.append("=" * w)
    return "\n".join(lines)
