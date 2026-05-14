"""Master data ingestion orchestrator.

Single entry point for all data ingestion — backfill, scheduler, and API all
call these functions. Every function follows the same pipeline:
    fetch → validate → CA adjust → gap detect → persist → provenance receipt

This is the module that turns raw API responses into durable lake data.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.core.logging import get_logger

logger = get_logger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class IngestResult:
    ticker: str
    interval: str
    status: str   # ok / no_data / silent_throttle / all_rejected / error
    bars: int = 0
    gaps: int = 0
    rejected: int = 0
    source: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass
class MacroIngestResult:
    series_id: str
    status: str   # ok / no_data / error
    points: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass
class EdgarIngestResult:
    cik: str
    ticker: str
    status: str   # ok / no_data / no_facts / error
    facts: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# ── OHLCV ─────────────────────────────────────────────────────────────────────

async def ingest_ticker_ohlcv(
    ticker: str,
    interval: str,
    start: datetime,
    end: datetime,
    session: AsyncSession,
    skip_if_exists: bool = False,
) -> IngestResult:
    """Complete OHLCV pipeline: fetch → validate → CA adjust → gap detect → persist → receipt.

    Args:
        skip_if_exists: If True and provenance already exists for this (ticker, interval),
                        return immediately. Set False for full historical backfills.
    """
    from sentinel.sds.normalizer import fetch_ohlcv_with_fallback
    from sentinel.sds.corporate_actions import fetch_and_apply
    from sentinel.sds.validator import validate_single_source
    from sentinel.sds.gap_detector import detect_gaps
    from sentinel.sds.provenance import create_receipt
    from sentinel.sds import repository

    try:
        # Resume check — skip already-ingested ranges on incremental runs
        if skip_if_exists:
            for src in ("yfinance", "polygon", "alpaca"):
                if await repository.get_latest_provenance_hash(ticker, src, interval, session):
                    return IngestResult(ticker=ticker, interval=interval,
                                        status="skipped_existing")

        # 1. Fetch from adapter chain
        bars = await fetch_ohlcv_with_fallback(ticker, start, end, interval)
        if not bars:
            return IngestResult(ticker=ticker, interval=interval, status="no_data")

        # 2. Single-source validation — hard-reject impossible values
        bars, report = validate_single_source(bars, ticker)
        if not bars:
            return IngestResult(
                ticker=ticker, interval=interval, status="all_rejected",
                rejected=report.rejected,
            )

        # 3. Corporate action backward adjustment (splits; dividends via weekly job)
        bars = await fetch_and_apply(ticker, bars)

        # 4. Gap detection — flag silent throttling
        gap = detect_gaps(bars, ticker, start.date(), end.date(), interval=interval)
        if gap.is_silent_throttle:
            return IngestResult(ticker=ticker, interval=interval, status="silent_throttle")

        # 5. Persist raw bars to TimescaleDB
        n = await repository.write_ohlcv_bars(bars, ticker, interval, session)

        # 6. Provenance receipt — SHA-256 fingerprint, chained
        receipt = create_receipt(
            bars, source=bars[0].source, ticker=ticker, interval=interval,
        )
        await repository.write_provenance_receipt(receipt, session)

        return IngestResult(
            ticker=ticker, interval=interval, status="ok",
            bars=n, gaps=gap.gap_count,
            rejected=report.rejected,
            source=bars[0].source,
        )

    except Exception as exc:
        logger.error("OHLCV ingest error", ticker=ticker, interval=interval, error=str(exc))
        return IngestResult(ticker=ticker, interval=interval, status="error", error=str(exc))


# ── Macro ─────────────────────────────────────────────────────────────────────

async def ingest_macro_series(
    series_id: str,
    session: AsyncSession,
    start: Optional[date] = None,
    vintage_date: Optional[date] = None,
) -> MacroIngestResult:
    """Fetch a FRED series and persist to macro_data (TimescaleDB hypertable)."""
    from sentinel.sds.adapters.fred_adapter import FREDAdapter
    from sentinel.core.config import get_settings
    from sentinel.sds import repository

    try:
        s = get_settings()
        fred = FREDAdapter(api_key=s.fred_api_key)
        points = await fred.fetch_series(series_id, start=start, vintage_date=vintage_date)
        if not points:
            return MacroIngestResult(series_id=series_id, status="no_data")
        n = await repository.write_macro_points(points, session)
        return MacroIngestResult(series_id=series_id, status="ok", points=n)
    except Exception as exc:
        logger.error("Macro ingest error", series_id=series_id, error=str(exc))
        return MacroIngestResult(series_id=series_id, status="error", error=str(exc))


# ── EDGAR Fundamentals ────────────────────────────────────────────────────────

async def ingest_edgar_facts(
    cik: str,
    ticker: str,
    session: AsyncSession,
) -> EdgarIngestResult:
    """Fetch EDGAR XBRL companyfacts and persist to financial_facts table.

    All fact versions are stored (including restatements), keyed by
    (cik, concept, period_end, accession) for point-in-time queries.
    """
    from sentinel.sds.adapters.edgar_adapter import EDGARAdapter
    from sentinel.sfe.xbrl_parser import extract_facts
    from sentinel.core.config import get_settings
    from sentinel.sds import repository

    try:
        s = get_settings()
        edgar = EDGARAdapter(user_agent=s.edgar_user_agent)
        companyfacts = await edgar.fetch_companyfacts(cik)
        if not companyfacts:
            return EdgarIngestResult(cik=cik, ticker=ticker, status="no_data")
        facts = extract_facts(companyfacts, cik)
        if not facts:
            return EdgarIngestResult(cik=cik, ticker=ticker, status="no_facts")
        n = await repository.write_financial_facts(facts, session)
        return EdgarIngestResult(cik=cik, ticker=ticker, status="ok", facts=n)
    except Exception as exc:
        logger.error("EDGAR ingest error", cik=cik, ticker=ticker, error=str(exc))
        return EdgarIngestResult(cik=cik, ticker=ticker, status="error", error=str(exc))


# ── Congressional Trades ──────────────────────────────────────────────────────

@dataclass
class CongressIngestResult:
    status: str   # ok / no_data / error
    house: int = 0
    senate: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


async def ingest_congressional_trades(
    session: AsyncSession,
    since: Optional[date] = None,
) -> CongressIngestResult:
    """Fetch STOCK Act disclosures from both chambers and persist to congressional_trades."""
    from sentinel.sds.adapters.congress_adapter import CongressAdapter
    from sentinel.sds import repository

    try:
        adapter = CongressAdapter()
        house = await adapter.fetch_house_trades(since=since)
        senate = await adapter.fetch_senate_trades(since=since)
        all_trades = house + senate
        if not all_trades:
            return CongressIngestResult(status="no_data")
        await repository.write_congressional_trades(all_trades, session)
        return CongressIngestResult(status="ok", house=len(house), senate=len(senate))
    except Exception as exc:
        logger.error("Congressional trades ingest error", error=str(exc))
        return CongressIngestResult(status="error", error=str(exc))


# ── COT Data ──────────────────────────────────────────────────────────────────

@dataclass
class COTIngestResult:
    status: str   # ok / no_data / error
    markets: int = 0
    records: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


async def ingest_cot_data(
    session: AsyncSession,
    start_year: int = 2015,
) -> COTIngestResult:
    """Load CFTC COT history, compute COT Index for all known markets, persist to cot_data."""
    from sentinel.sma.cot_report import COTClient, MARKET_CODES
    from sentinel.sds import repository

    try:
        client = COTClient()
        await client.load_range(start_year, date.today().year)

        records: list[dict] = []
        for code, market_name in MARKET_CODES.items():
            df = client.compute_cot_index(market_name)
            if df.empty:
                continue
            raw_df = client.get_market_data(market_name)
            date_col = "As of Date in Form YYYY-MM-DD"

            for _, row in df.iterrows():
                report_date = row["date"].date() if hasattr(row["date"], "date") else row["date"]
                raw_row = None
                if not raw_df.empty and date_col in raw_df.columns:
                    mask = raw_df[date_col] == row["date"]
                    if mask.any():
                        raw_row = raw_df[mask].iloc[0]

                def _int(dr, *keys):
                    for k in keys:
                        if dr is not None and k in dr.index:
                            try:
                                return int(float(dr[k]))
                            except (ValueError, TypeError):
                                pass
                    return None

                records.append({
                    "report_date": report_date,
                    "market_name": market_name[:100],
                    "commodity_code": code,
                    "open_interest": _int(raw_row, "Open_Interest_All"),
                    "comm_long": _int(raw_row, "Prod_Merc_Positions_Long_All"),
                    "comm_short": _int(raw_row, "Prod_Merc_Positions_Short_All"),
                    "noncomm_long": _int(raw_row, "M_Money_Positions_Long_All"),
                    "noncomm_short": _int(raw_row, "M_Money_Positions_Short_All"),
                    "nonrept_long": _int(raw_row, "NonRept_Positions_Long_All"),
                    "nonrept_short": _int(raw_row, "NonRept_Positions_Short_All"),
                    "net_speculator": int(row["net_position"]),
                    "cot_index": float(row["cot_index"]) if row.get("cot_index") is not None else None,
                    "signal": row.get("signal"),
                    "source": "cftc",
                })

        if not records:
            return COTIngestResult(status="no_data")

        n = await repository.write_cot_records(records, session)
        return COTIngestResult(status="ok", markets=len(MARKET_CODES), records=n)

    except Exception as exc:
        logger.error("COT ingest error", error=str(exc))
        return COTIngestResult(status="error", error=str(exc))


# ── Insider Transactions (Form 4) ─────────────────────────────────────────────

@dataclass
class InsiderIngestResult:
    cik: str
    ticker: str
    status: str   # ok / no_data / no_filings / error
    transactions: int = 0
    filings: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


async def ingest_insider_transactions(
    cik: str,
    ticker: str,
    session: AsyncSession,
    since: Optional[date] = None,
    figi: str = "",
    limit: int = 40,
) -> InsiderIngestResult:
    """Fetch Form 4 filings from EDGAR and persist to insider_transactions.

    Args:
        since: Only ingest filings on or after this date. None = last 40 filings.
        limit: Max number of Form 4 XMLs to download (per CIK per call).
    """
    from sentinel.sds.adapters.insider_adapter import InsiderAdapter
    from sentinel.core.config import get_settings
    from sentinel.sds import repository

    try:
        s = get_settings()
        adapter = InsiderAdapter(user_agent=s.edgar_user_agent)
        transactions = await adapter.fetch_transactions(
            cik=cik, ticker=ticker, figi=figi, limit=limit, since=since
        )
        if not transactions:
            return InsiderIngestResult(cik=cik, ticker=ticker, status="no_data")

        n = await repository.write_insider_transactions(transactions, session)
        return InsiderIngestResult(cik=cik, ticker=ticker, status="ok", transactions=n)

    except Exception as exc:
        logger.error("Insider transactions ingest error", cik=cik, ticker=ticker, error=str(exc))
        return InsiderIngestResult(cik=cik, ticker=ticker, status="error", error=str(exc))


# ── Institutional Holdings (13F-HR) ──────────────────────────────────────────

@dataclass
class InstitutionalIngestResult:
    manager_cik: str
    status: str   # ok / no_data / no_filings / error
    holdings: int = 0
    filings: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


async def ingest_institutional_holdings(
    manager_cik: str,
    session: AsyncSession,
    since: Optional[date] = None,
    limit: int = 8,
) -> InstitutionalIngestResult:
    """Fetch 13F-HR filings from EDGAR and persist to institutional_holdings.

    Args:
        limit: Max filings to process (default 8 = 2 years of quarterly filings).
    """
    from sentinel.sds.adapters.institutional_adapter import InstitutionalAdapter
    from sentinel.core.config import get_settings
    from sentinel.sds import repository

    try:
        s = get_settings()
        adapter = InstitutionalAdapter(user_agent=s.edgar_user_agent)
        holdings = await adapter.fetch_holdings(manager_cik=manager_cik, limit=limit, since=since)
        if not holdings:
            return InstitutionalIngestResult(manager_cik=manager_cik, status="no_data")

        n = await repository.write_institutional_holdings(holdings, session)
        return InstitutionalIngestResult(
            manager_cik=manager_cik, status="ok",
            holdings=n, filings=limit,
        )

    except Exception as exc:
        logger.error("Institutional holdings ingest error", cik=manager_cik, error=str(exc))
        return InstitutionalIngestResult(manager_cik=manager_cik, status="error", error=str(exc))


# ── News & Sentiment ──────────────────────────────────────────────────────────

@dataclass
class NewsIngestResult:
    ticker: str
    status: str   # ok / no_data / no_key / error
    articles: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


async def ingest_news(
    ticker: str,
    session: AsyncSession,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
) -> NewsIngestResult:
    """Fetch news from Finnhub, score with FinBERT, persist to news_articles.

    Sentiment is scored per headline using the lazy-loaded FinBERT pipeline.
    Articles without a Finnhub API key return no_key status (non-fatal).
    """
    from sentinel.sds.adapters.finnhub_adapter import FinnhubAdapter
    from sentinel.sil.sentiment import score_sentiment_batch
    from sentinel.core.config import get_settings
    from sentinel.sds import repository

    try:
        s = get_settings()
        if not s.finnhub_api_key:
            return NewsIngestResult(ticker=ticker, status="no_key")

        fh = FinnhubAdapter(api_key=s.finnhub_api_key)
        today = date.today()
        from_str = (from_date or (today.replace(day=today.day - min(today.day - 1, 7)))).isoformat()
        to_str = (to_date or today).isoformat()

        raw_articles = await fh.fetch_news(ticker, from_str, to_str)
        if not raw_articles:
            return NewsIngestResult(ticker=ticker, status="no_data")

        headlines = [a.get("headline", "") or "" for a in raw_articles]
        sentiments = await score_sentiment_batch(headlines)

        articles: list[dict] = []
        for raw, sent in zip(raw_articles, sentiments):
            published_ts = raw.get("datetime")
            if published_ts:
                try:
                    from datetime import timezone
                    published_at = datetime.fromtimestamp(int(published_ts), tz=timezone.utc)
                except (ValueError, TypeError, OSError):
                    published_at = None
            else:
                published_at = None

            if published_at is None:
                continue

            articles.append({
                "headline": raw.get("headline", ""),
                "summary": raw.get("summary", ""),
                "source": raw.get("source", "finnhub"),
                "url": raw.get("url"),
                "published_at": published_at,
                "tickers": [ticker.upper()],
                "sentiment_label": sent.label,
                "sentiment_score": float(sent.score),
            })

        if not articles:
            return NewsIngestResult(ticker=ticker, status="no_data")

        n = await repository.write_news_articles(articles, session)
        return NewsIngestResult(ticker=ticker, status="ok", articles=n)

    except Exception as exc:
        logger.error("News ingest error", ticker=ticker, error=str(exc))
        return NewsIngestResult(ticker=ticker, status="error", error=str(exc))


@dataclass
class CryptoIngestResult:
    pair: str
    interval: str
    status: str   # ok / no_data / no_adapter / error
    bars: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# ── Corporate Actions ─────────────────────────────────────────────────────────

async def ingest_corporate_actions(
    ticker: str,
    session: AsyncSession,
) -> int:
    """Fetch splits + dividends (with resolved factors) and persist to corporate_actions."""
    from sentinel.sds.corporate_actions import (
        fetch_splits_yfinance,
        fetch_dividends_yfinance,
        resolve_dividend_factors,
    )
    from sentinel.sds import repository

    loop = asyncio.get_event_loop()
    splits = await loop.run_in_executor(None, lambda: fetch_splits_yfinance(ticker))
    raw_divs = await loop.run_in_executor(None, lambda: fetch_dividends_yfinance(ticker))
    divs = await resolve_dividend_factors(raw_divs, ticker)
    all_actions = splits + divs
    if not all_actions:
        return 0
    return await repository.upsert_corporate_actions(all_actions, session)


async def ingest_crypto_ohlcv(
    pair: str,
    interval: str = "1d",
    start: datetime | None = None,
    end: datetime | None = None,
    exchange: str = "binance",
    session: AsyncSession | None = None,
) -> CryptoIngestResult:
    """Ingest crypto OHLCV via CCXT (Binance/Kraken/etc). Persists to ohlcv table.

    pair: "BTC/USDT", "ETH/USDT", etc.
    exchange: "binance" | "kraken" | "coinbase"
    """
    from sentinel.sds import get_adapter
    from sentinel.sds.validator import validate_single_source
    from sentinel.sds import repository
    from sentinel.sds.provenance import create_receipt

    end = end or datetime.utcnow()
    start = start or (end - timedelta(days=365))

    # Use ccxt adapter
    adapter = get_adapter("ccxt")
    if adapter is None:
        return CryptoIngestResult(pair=pair, interval=interval, status="no_adapter")

    try:
        bars = await adapter.fetch_ohlcv(pair, start, end, interval)
    except Exception as exc:
        return CryptoIngestResult(pair=pair, interval=interval, status="error", error=str(exc))

    if not bars:
        return CryptoIngestResult(pair=pair, interval=interval, status="no_data")

    # Validate
    bars, _report = validate_single_source(bars, pair)
    if not bars:
        return CryptoIngestResult(pair=pair, interval=interval, status="no_data")

    # Assign synthetic FIGI and ticker for crypto
    figi = f"CRYPTO:{pair.replace('/', '_')}"
    for bar in bars:
        bar.figi = figi
        bar.ticker = pair

    # Persist
    _own_session = False
    if session is None:
        from sentinel.sds.db import get_session_factory
        session_factory = get_session_factory()
        _ctx = session_factory()
        session = await _ctx.__aenter__()
        _own_session = True

    try:
        await repository.upsert_instrument(
            figi=figi, ticker=pair, session=session,
            name=pair, asset_class="crypto",
        )
        await repository.write_ohlcv_bars(bars, pair, interval, session)
        receipt = create_receipt(bars, source=bars[0].source, ticker=pair, interval=interval)
        await repository.write_provenance_receipt(receipt, session)
        if _own_session:
            await session.commit()
    finally:
        if _own_session:
            await _ctx.__aexit__(None, None, None)

    logger.info("Crypto OHLCV ingested", pair=pair, interval=interval, bars=len(bars))
    return CryptoIngestResult(pair=pair, interval=interval, status="ok", bars=len(bars))
