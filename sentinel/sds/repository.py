"""SDS Data Repository — all database persistence operations.

Pure functions: every function takes an AsyncSession parameter.
No global state. Fully testable with a test session.

The ohlcv table has a FK to instruments(figi).
Always call upsert_instrument() before write_ohlcv_bars()
(write_ohlcv_bars does this automatically).

Raw prices are stored as-is. adj_factor is NOT a column in ohlcv —
it is computed on-read via get_corporate_actions().
"""
from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.core.types import (
    CorporateAction,
    CorporateActionType,
    DataProvenance,
    FinancialFact,
    MacroDataPoint,
    OHLCVBar,
    SurvivorshipRecord,
)
from sentinel.core.logging import get_logger

logger = get_logger(__name__)


# ── Instruments ───────────────────────────────────────────────────────────────

async def upsert_instrument(
    figi: str,
    ticker: str,
    session: AsyncSession,
    name: str = "",
    exchange: str = "",
    asset_class: str = "equity",
    currency: str = "USD",
) -> None:
    """Ensure an instrument row exists. Inserts minimal record on first seen."""
    await session.execute(
        text("""
            INSERT INTO instruments (figi, ticker, name, exchange, asset_class, currency)
            VALUES (:figi, :ticker, :name, :exchange, :asset_class, :currency)
            ON CONFLICT (figi) DO NOTHING
        """),
        dict(figi=figi, ticker=ticker, name=name,
             exchange=exchange, asset_class=asset_class, currency=currency),
    )


# ── OHLCV ─────────────────────────────────────────────────────────────────────

async def write_ohlcv_bars(
    bars: list[OHLCVBar],
    ticker: str,
    interval: str,
    session: AsyncSession,
) -> int:
    """Upsert OHLCV bars into TimescaleDB. Returns row count attempted.

    Raw close is stored as-is. adj_factor lives in corporate_actions, not here.
    """
    if not bars:
        return 0

    figi = bars[0].figi
    await upsert_instrument(figi, ticker, session)

    inserted = 0
    for bar in bars:
        await session.execute(
            text("""
                INSERT INTO ohlcv
                    (time, figi, open, high, low, close, volume, vwap, source, interval)
                VALUES
                    (:time, :figi, :open, :high, :low, :close, :volume, :vwap, :source, :interval)
                ON CONFLICT (time, figi, interval) DO NOTHING
            """),
            dict(
                time=bar.time,
                figi=figi,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
                vwap=float(bar.vwap) if bar.vwap else None,
                source=bar.source,
                interval=interval,
            ),
        )
        inserted += 1

    await session.commit()
    logger.info("OHLCV written", ticker=ticker, interval=interval, rows=inserted)
    return inserted


async def get_ohlcv_bars(
    figi: str,
    interval: str,
    start: datetime,
    end: datetime,
    session: AsyncSession,
) -> list[dict]:
    """Read raw OHLCV bars from DB by FIGI, ordered ascending."""
    result = await session.execute(
        text("""
            SELECT time, figi, open, high, low, close, volume, vwap, source
            FROM ohlcv
            WHERE figi = :figi AND interval = :interval
              AND time >= :start AND time <= :end
            ORDER BY time ASC
        """),
        dict(figi=figi, interval=interval, start=start, end=end),
    )
    return [dict(row) for row in result.mappings()]


async def get_ohlcv_bars_by_ticker(
    ticker: str,
    interval: str,
    start: datetime,
    end: datetime,
    session: AsyncSession,
) -> list[dict]:
    """Read raw OHLCV bars from DB by ticker symbol, ordered ascending.

    Joins to instruments table so callers don't need to resolve FIGI first.
    """
    result = await session.execute(
        text("""
            SELECT o.time, o.figi, i.ticker, o.open, o.high, o.low,
                   o.close, o.volume, o.vwap, o.source
            FROM ohlcv o
            JOIN instruments i ON o.figi = i.figi
            WHERE i.ticker = :ticker AND o.interval = :interval
              AND o.time >= :start AND o.time <= :end
            ORDER BY o.time ASC
        """),
        dict(ticker=ticker, interval=interval, start=start, end=end),
    )
    return [dict(row) for row in result.mappings()]


# ── Corporate Actions ─────────────────────────────────────────────────────────

async def upsert_corporate_actions(
    actions: list[CorporateAction],
    session: AsyncSession,
) -> int:
    """Insert corporate actions; silently skip duplicates on (figi, action_type, ex_date)."""
    inserted = 0
    for action in actions:
        await session.execute(
            text("""
                INSERT INTO corporate_actions
                    (figi, ticker, action_type, ex_date, ratio_new, ratio_old, factor, source)
                VALUES
                    (:figi, :ticker, :action_type, :ex_date,
                     :ratio_new, :ratio_old, :factor, :source)
                ON CONFLICT (figi, action_type, ex_date) DO NOTHING
            """),
            dict(
                figi=action.figi,
                ticker=action.ticker,
                action_type=action.action_type.value,
                ex_date=action.ex_date,
                ratio_new=float(action.ratio_new),
                ratio_old=float(action.ratio_old),
                factor=float(action.factor),
                source=action.source,
            ),
        )
        inserted += 1

    await session.commit()
    return inserted


async def get_corporate_actions(
    figi: str,
    session: AsyncSession,
) -> list[CorporateAction]:
    """Read all corporate actions for a figi, ordered by ex_date ascending."""
    result = await session.execute(
        text("""
            SELECT figi, ticker, action_type, ex_date,
                   ratio_new, ratio_old, factor, source
            FROM corporate_actions
            WHERE figi = :figi
            ORDER BY ex_date ASC
        """),
        dict(figi=figi),
    )
    actions = []
    for row in result.mappings():
        actions.append(CorporateAction(
            figi=row["figi"],
            ticker=row["ticker"],
            action_type=CorporateActionType(row["action_type"]),
            ex_date=row["ex_date"],
            ratio_new=Decimal(str(row["ratio_new"])),
            ratio_old=Decimal(str(row["ratio_old"])),
            factor=Decimal(str(row["factor"])),
            source=row["source"],
        ))
    return actions


# ── Survivorship Registry ─────────────────────────────────────────────────────

async def upsert_survivorship_records(
    records: list[SurvivorshipRecord],
    session: AsyncSession,
) -> int:
    """Insert survivorship records; silently skip duplicates on cik."""
    inserted = 0
    for r in records:
        await session.execute(
            text("""
                INSERT INTO survivorship_registry
                    (cik, figi, ticker, company_name, delist_date,
                     delist_reason, exchange, notes)
                VALUES
                    (:cik, :figi, :ticker, :company_name, :delist_date,
                     :delist_reason, :exchange, :notes)
                ON CONFLICT (cik) DO NOTHING
            """),
            dict(
                cik=r.cik,
                figi=r.figi,
                ticker=r.ticker,
                company_name=r.company_name,
                delist_date=r.delist_date,
                delist_reason=r.delist_reason.value,
                exchange=r.exchange,
                notes=r.notes,
            ),
        )
        inserted += 1

    await session.commit()
    return inserted


# ── Data Provenance ───────────────────────────────────────────────────────────

async def write_provenance_receipt(
    receipt: DataProvenance,
    session: AsyncSession,
) -> None:
    """Persist a provenance receipt. Silently skips if sha256 already exists."""
    await session.execute(
        text("""
            INSERT INTO data_provenance
                (batch_id, source, ticker, figi, interval,
                 start_time, end_time, bar_count, sha256, prev_hash,
                 validated, validation_delta_pct, ingested_at)
            VALUES
                (:batch_id, :source, :ticker, :figi, :interval,
                 :start_time, :end_time, :bar_count, :sha256, :prev_hash,
                 :validated, :validation_delta_pct, :ingested_at)
            ON CONFLICT (sha256) DO NOTHING
        """),
        dict(
            batch_id=receipt.batch_id,
            source=receipt.source,
            ticker=receipt.ticker,
            figi=receipt.figi,
            interval=receipt.interval,
            start_time=receipt.start_time,
            end_time=receipt.end_time,
            bar_count=receipt.bar_count,
            sha256=receipt.sha256,
            prev_hash=receipt.prev_hash,
            validated=receipt.validated,
            validation_delta_pct=(
                float(receipt.validation_delta_pct)
                if receipt.validation_delta_pct else None
            ),
            ingested_at=receipt.ingested_at,
        ),
    )
    await session.commit()


async def get_latest_provenance_hash(
    ticker: str,
    source: str,
    interval: str,
    session: AsyncSession,
) -> Optional[str]:
    """Return the sha256 of the most recent receipt for (ticker, source, interval)."""
    result = await session.execute(
        text("""
            SELECT sha256 FROM data_provenance
            WHERE ticker = :ticker AND source = :source AND interval = :interval
            ORDER BY ingested_at DESC
            LIMIT 1
        """),
        dict(ticker=ticker, source=source, interval=interval),
    )
    row = result.fetchone()
    return row[0] if row else None


# ── Macro Data ────────────────────────────────────────────────────────────────

async def write_macro_points(
    points: list[MacroDataPoint],
    session: AsyncSession,
) -> int:
    """Upsert macro data points. Updates value on conflict (ALFRED revisions)."""
    if not points:
        return 0

    written = 0
    for pt in points:
        if pt.value is None:
            continue
        await session.execute(
            text("""
                INSERT INTO macro_data (time, series_id, value, vintage)
                VALUES (:time, :series_id, :value, :vintage)
                ON CONFLICT (time, series_id) DO UPDATE
                    SET value   = EXCLUDED.value,
                        vintage = EXCLUDED.vintage
            """),
            dict(
                time=pt.time,
                series_id=pt.series_id,
                value=float(pt.value),
                vintage=pt.vintage,
            ),
        )
        written += 1

    await session.commit()
    logger.info("Macro points written", count=written)
    return written


# ── Financial Facts (EDGAR XBRL) ─────────────────────────────────────────────

async def write_financial_facts(
    facts: list[FinancialFact],
    session: AsyncSession,
) -> int:
    """Upsert EDGAR XBRL facts.

    Requires migration 0001 which adds uq_facts_cik_concept_period_accn.
    Duplicate (cik, concept, period_end, accession) rows are silently skipped,
    so re-ingesting a CIK is always safe and idempotent.

    DB column names differ from the Python type:
        form_type → form
        filed_date → filed
    """
    if not facts:
        return 0

    written = 0
    for fact in facts:
        if fact.value is None:
            continue
        await session.execute(
            text("""
                INSERT INTO financial_facts
                    (cik, figi, concept, label, value, unit,
                     period_start, period_end, form, filed, accession, frame)
                VALUES
                    (:cik, :figi, :concept, :label, :value, :unit,
                     :period_start, :period_end, :form, :filed, :accession, :frame)
                ON CONFLICT ON CONSTRAINT uq_facts_cik_concept_period_accn DO NOTHING
            """),
            dict(
                cik=fact.cik,
                figi=fact.figi or "",
                concept=fact.concept,
                label=fact.label or fact.concept,
                value=float(fact.value),
                unit=fact.unit,
                period_start=fact.period_start,
                period_end=fact.period_end,
                form=fact.form_type,
                filed=fact.filed_date,
                accession=fact.accession,
                frame=fact.frame,
            ),
        )
        written += 1

    await session.commit()
    logger.info("Financial facts written", count=written)
    return written


async def get_financial_facts(
    cik: str,
    session: AsyncSession,
    concept: Optional[str] = None,
    as_of: Optional[datetime] = None,
) -> list[dict]:
    """Read financial facts for a CIK.

    Pass as_of to enforce point-in-time (no look-ahead): only returns facts
    where filed <= as_of, preventing backtest bias from unreleased filings.
    """
    filters = ["cik = :cik"]
    params: dict = {"cik": cik}

    if concept:
        filters.append("concept = :concept")
        params["concept"] = concept

    if as_of:
        filters.append("(filed IS NULL OR filed <= :as_of)")
        params["as_of"] = as_of

    where = " AND ".join(filters)
    result = await session.execute(
        text(f"""
            SELECT cik, figi, concept, label, value, unit,
                   period_start, period_end, form, filed, accession, frame
            FROM financial_facts
            WHERE {where}
            ORDER BY period_end DESC, filed DESC
        """),
        params,
    )
    return [dict(row) for row in result.mappings()]


async def get_financial_facts_by_figi(
    figi: str,
    session: AsyncSession,
    concept: Optional[str] = None,
    as_of: Optional[datetime] = None,
) -> list[dict]:
    """Read financial facts by FIGI (for tickers where CIK is unknown)."""
    filters = ["figi = :figi", "figi != ''"]
    params: dict = {"figi": figi}

    if concept:
        filters.append("concept = :concept")
        params["concept"] = concept

    if as_of:
        filters.append("(filed IS NULL OR filed <= :as_of)")
        params["as_of"] = as_of

    where = " AND ".join(filters)
    result = await session.execute(
        text(f"""
            SELECT cik, figi, concept, label, value, unit,
                   period_start, period_end, form, filed, accession, frame
            FROM financial_facts
            WHERE {where}
            ORDER BY period_end DESC, filed DESC
        """),
        params,
    )
    return [dict(row) for row in result.mappings()]


# ── Congressional Trades ──────────────────────────────────────────────────────

async def write_congressional_trades(
    trades: list[dict],
    session: AsyncSession,
) -> int:
    """Upsert congressional trade disclosures.

    Dedup key: (politician_name, ticker, tx_date, tx_type, amount_low).
    Re-running is safe — duplicates are silently skipped.
    """
    if not trades:
        return 0

    written = 0
    for t in trades:
        if not t.get("tx_date"):
            continue
        await session.execute(
            text("""
                INSERT INTO congressional_trades
                    (politician_name, chamber, party, state,
                     ticker, figi, asset_name,
                     tx_date, filed_date, tx_code, tx_type,
                     amount_low, amount_high, filing_lag_days, late_filing,
                     source, disclosure_url)
                VALUES
                    (:politician_name, :chamber, :party, :state,
                     :ticker, :figi, :asset_name,
                     :tx_date, :filed_date, :tx_code, :tx_type,
                     :amount_low, :amount_high, :filing_lag_days, :late_filing,
                     :source, :disclosure_url)
                ON CONFLICT DO NOTHING
            """),
            dict(
                politician_name=(t.get("politician_name") or "")[:200],
                chamber=(t.get("chamber") or "")[:10],
                party=(t.get("party") or "")[:5],
                state=(t.get("state") or "")[:5],
                ticker=t.get("ticker"),
                figi=t.get("figi"),
                asset_name=(t.get("asset_name") or "")[:500],
                tx_date=t.get("tx_date"),
                filed_date=t.get("filed_date"),
                tx_code=(t.get("tx_code") or "")[:20],
                tx_type=(t.get("tx_type") or "")[:20],
                amount_low=float(t["amount_low"]) if t.get("amount_low") is not None else None,
                amount_high=float(t["amount_high"]) if t.get("amount_high") is not None else None,
                filing_lag_days=t.get("filing_lag_days"),
                late_filing=bool(t.get("late_filing", False)),
                source=(t.get("source") or "")[:30],
                disclosure_url=t.get("disclosure_url"),
            ),
        )
        written += 1

    await session.commit()
    logger.info("Congressional trades written", count=written)
    return written


async def get_congressional_trades(
    session: AsyncSession,
    ticker: Optional[str] = None,
    since: Optional[datetime] = None,
    chamber: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    """Read congressional trades with optional ticker/date/chamber filters."""
    filters = []
    params: dict = {}

    if ticker:
        filters.append("ticker = :ticker")
        params["ticker"] = ticker.upper()
    if since:
        filters.append("tx_date >= :since")
        params["since"] = since
    if chamber:
        filters.append("chamber = :chamber")
        params["chamber"] = chamber.lower()

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    result = await session.execute(
        text(f"""
            SELECT politician_name, chamber, party, state,
                   ticker, asset_name, tx_date, filed_date,
                   tx_code, amount_low, amount_high,
                   filing_lag_days, late_filing, source
            FROM congressional_trades
            {where}
            ORDER BY tx_date DESC
            LIMIT :limit
        """),
        {**params, "limit": limit},
    )
    return [dict(row) for row in result.mappings()]


# ── COT Data ──────────────────────────────────────────────────────────────────

async def write_cot_records(
    records: list[dict],
    session: AsyncSession,
) -> int:
    """Upsert CFTC COT records. Dedup key: (report_date, market_name)."""
    if not records:
        return 0

    written = 0
    for r in records:
        if not r.get("report_date") or not r.get("market_name"):
            continue
        await session.execute(
            text("""
                INSERT INTO cot_data
                    (report_date, market_name, commodity_code,
                     open_interest, comm_long, comm_short,
                     noncomm_long, noncomm_short, nonrept_long, nonrept_short,
                     net_speculator, cot_index, signal, source)
                VALUES
                    (:report_date, :market_name, :commodity_code,
                     :open_interest, :comm_long, :comm_short,
                     :noncomm_long, :noncomm_short, :nonrept_long, :nonrept_short,
                     :net_speculator, :cot_index, :signal, :source)
                ON CONFLICT (report_date, market_name) DO UPDATE
                    SET cot_index = EXCLUDED.cot_index,
                        signal    = EXCLUDED.signal,
                        net_speculator = EXCLUDED.net_speculator
            """),
            dict(
                report_date=r["report_date"],
                market_name=r["market_name"][:100],
                commodity_code=r.get("commodity_code"),
                open_interest=r.get("open_interest"),
                comm_long=r.get("comm_long"),
                comm_short=r.get("comm_short"),
                noncomm_long=r.get("noncomm_long"),
                noncomm_short=r.get("noncomm_short"),
                nonrept_long=r.get("nonrept_long"),
                nonrept_short=r.get("nonrept_short"),
                net_speculator=r.get("net_speculator"),
                cot_index=float(r["cot_index"]) if r.get("cot_index") is not None else None,
                signal=r.get("signal"),
                source=r.get("source", "cftc"),
            ),
        )
        written += 1

    await session.commit()
    logger.info("COT records written", count=written)
    return written


async def get_cot_signals(
    session: AsyncSession,
    market_name: Optional[str] = None,
    since: Optional[datetime] = None,
) -> list[dict]:
    """Read latest COT signals from DB."""
    filters = []
    params: dict = {}

    if market_name:
        filters.append("market_name ILIKE :market_name")
        params["market_name"] = f"%{market_name}%"
    if since:
        filters.append("report_date >= :since")
        params["since"] = since

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    result = await session.execute(
        text(f"""
            SELECT DISTINCT ON (market_name)
                   report_date, market_name, net_speculator,
                   cot_index, signal, open_interest
            FROM cot_data
            {where}
            ORDER BY market_name, report_date DESC
        """),
        params,
    )
    return [dict(row) for row in result.mappings()]


# ── Insider Transactions (Form 4) ─────────────────────────────────────────────

async def write_insider_transactions(
    records: list[dict],
    session: AsyncSession,
) -> int:
    """Insert Form 4 insider transactions. Dedup key: (cik, ticker, tx_date, tx_code, shares).

    Disposals are stored with negative share counts. Re-ingesting is safe.
    """
    if not records:
        return 0

    written = 0
    for r in records:
        if not r.get("tx_date") or r.get("shares") is None:
            continue
        await session.execute(
            text("""
                INSERT INTO insider_transactions
                    (cik, figi, ticker, owner_name, owner_cik, role,
                     security_title, tx_date, tx_code,
                     shares, price_per_share, value,
                     shares_owned_after, is_derivative, exercise_price, expiry_date)
                VALUES
                    (:cik, :figi, :ticker, :owner_name, :owner_cik, :role,
                     :security_title, :tx_date, :tx_code,
                     :shares, :price_per_share, :value,
                     :shares_owned_after, :is_derivative, :exercise_price, :expiry_date)
                ON CONFLICT DO NOTHING
            """),
            dict(
                cik=(r.get("cik") or "")[:10],
                figi=(r.get("figi") or "")[:12],
                ticker=(r.get("ticker") or "")[:20],
                owner_name=(r.get("owner_name") or "")[:200],
                owner_cik=r.get("owner_cik"),
                role=(r.get("role") or "")[:30],
                security_title=(r.get("security_title") or "")[:100],
                tx_date=r["tx_date"],
                tx_code=(r.get("tx_code") or "")[:20],
                shares=float(r["shares"]) if r["shares"] is not None else None,
                price_per_share=float(r["price_per_share"]) if r.get("price_per_share") else None,
                value=float(r["value"]) if r.get("value") else None,
                shares_owned_after=float(r["shares_owned_after"]) if r.get("shares_owned_after") else None,
                is_derivative=bool(r.get("is_derivative", False)),
                exercise_price=float(r["exercise_price"]) if r.get("exercise_price") else None,
                expiry_date=r.get("expiry_date"),
            ),
        )
        written += 1

    await session.commit()
    logger.info("Insider transactions written", count=written)
    return written


async def get_insider_transactions(
    session: AsyncSession,
    ticker: Optional[str] = None,
    cik: Optional[str] = None,
    since: Optional[datetime] = None,
    is_derivative: Optional[bool] = None,
    limit: int = 200,
) -> list[dict]:
    """Read insider transactions with optional filters."""
    filters = []
    params: dict = {}

    if ticker:
        filters.append("ticker = :ticker")
        params["ticker"] = ticker.upper()
    if cik:
        filters.append("cik = :cik")
        params["cik"] = cik.zfill(10)
    if since:
        filters.append("tx_date >= :since")
        params["since"] = since
    if is_derivative is not None:
        filters.append("is_derivative = :is_derivative")
        params["is_derivative"] = is_derivative

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    result = await session.execute(
        text(f"""
            SELECT cik, figi, ticker, owner_name, role,
                   security_title, tx_date, tx_code,
                   shares, price_per_share, value,
                   shares_owned_after, is_derivative,
                   exercise_price, expiry_date
            FROM insider_transactions
            {where}
            ORDER BY tx_date DESC
            LIMIT :limit
        """),
        {**params, "limit": limit},
    )
    return [dict(row) for row in result.mappings()]


# ── Institutional Holdings (13F-HR) ──────────────────────────────────────────

async def write_institutional_holdings(
    records: list[dict],
    session: AsyncSession,
) -> int:
    """Upsert 13F-HR institutional holdings.

    Dedup key: (manager_cik, cusip, period_of_report).
    Re-filing amendments update market_value, shares.
    """
    if not records:
        return 0

    written = 0
    for r in records:
        if not r.get("period_of_report"):
            continue
        await session.execute(
            text("""
                INSERT INTO institutional_holdings
                    (manager_cik, issuer_name, cusip, ticker, figi,
                     period_of_report, filed_date,
                     market_value, shares, share_type,
                     put_call, investment_discretion)
                VALUES
                    (:manager_cik, :issuer_name, :cusip, :ticker, :figi,
                     :period_of_report, :filed_date,
                     :market_value, :shares, :share_type,
                     :put_call, :investment_discretion)
                ON CONFLICT (manager_cik, cusip, period_of_report) DO UPDATE
                    SET market_value        = EXCLUDED.market_value,
                        shares             = EXCLUDED.shares,
                        filed_date         = EXCLUDED.filed_date
            """),
            dict(
                manager_cik=(r.get("manager_cik") or "")[:10],
                issuer_name=(r.get("issuer_name") or "")[:256],
                cusip=(r.get("cusip") or None),
                ticker=r.get("ticker"),
                figi=r.get("figi"),
                period_of_report=r["period_of_report"],
                filed_date=r.get("filed_date"),
                market_value=float(r["market_value"]) if r.get("market_value") else None,
                shares=float(r["shares"]) if r.get("shares") else None,
                share_type=r.get("share_type"),
                put_call=r.get("put_call"),
                investment_discretion=r.get("investment_discretion"),
            ),
        )
        written += 1

    await session.commit()
    logger.info("Institutional holdings written", count=written)
    return written


async def get_institutional_holdings(
    session: AsyncSession,
    ticker: Optional[str] = None,
    cusip: Optional[str] = None,
    manager_cik: Optional[str] = None,
    period: Optional[datetime] = None,
    limit: int = 500,
) -> list[dict]:
    """Read institutional holdings with optional filters."""
    filters = []
    params: dict = {}

    if ticker:
        filters.append("ticker = :ticker")
        params["ticker"] = ticker.upper()
    if cusip:
        filters.append("cusip = :cusip")
        params["cusip"] = cusip
    if manager_cik:
        filters.append("manager_cik = :manager_cik")
        params["manager_cik"] = manager_cik.zfill(10)
    if period:
        filters.append("period_of_report = :period")
        params["period"] = period

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    result = await session.execute(
        text(f"""
            SELECT manager_cik, issuer_name, cusip, ticker, figi,
                   period_of_report, filed_date,
                   market_value, shares, share_type, put_call, investment_discretion
            FROM institutional_holdings
            {where}
            ORDER BY period_of_report DESC, market_value DESC NULLS LAST
            LIMIT :limit
        """),
        {**params, "limit": limit},
    )
    return [dict(row) for row in result.mappings()]


# ── News Articles ─────────────────────────────────────────────────────────────

async def write_news_articles(
    articles: list[dict],
    session: AsyncSession,
) -> int:
    """Insert news articles with sentiment scores. Dedup key: url (unique).

    Articles without a URL use (headline, published_at) as surrogate key.
    Re-ingesting the same URL is silently skipped.
    """
    if not articles:
        return 0

    written = 0
    for a in articles:
        if not a.get("published_at") or not a.get("headline"):
            continue
        # Normalise tickers array to PostgreSQL text[]
        tickers = a.get("tickers") or []
        if isinstance(tickers, str):
            tickers = [tickers]
        tickers_pg = "{" + ",".join(f'"{t}"' for t in tickers if t) + "}"

        await session.execute(
            text("""
                INSERT INTO news_articles
                    (headline, summary, source, url, published_at,
                     tickers, sentiment_label, sentiment_score)
                VALUES
                    (:headline, :summary, :source, :url, :published_at,
                     :tickers::text[], :sentiment_label, :sentiment_score)
                ON CONFLICT DO NOTHING
            """),
            dict(
                headline=a["headline"][:2000],
                summary=(a.get("summary") or "")[:5000],
                source=(a.get("source") or "")[:100],
                url=a.get("url"),
                published_at=a["published_at"],
                tickers=tickers_pg,
                sentiment_label=a.get("sentiment_label"),
                sentiment_score=float(a["sentiment_score"]) if a.get("sentiment_score") is not None else None,
            ),
        )
        written += 1

    await session.commit()
    logger.info("News articles written", count=written)
    return written


async def get_news_articles(
    session: AsyncSession,
    ticker: Optional[str] = None,
    since: Optional[datetime] = None,
    sentiment: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    """Read news articles. ticker matches against the tickers array column."""
    filters = []
    params: dict = {}

    if ticker:
        filters.append(":ticker = ANY(tickers)")
        params["ticker"] = ticker.upper()
    if since:
        filters.append("published_at >= :since")
        params["since"] = since
    if sentiment:
        filters.append("sentiment_label = :sentiment")
        params["sentiment"] = sentiment.lower()

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    result = await session.execute(
        text(f"""
            SELECT headline, summary, source, url, published_at,
                   tickers, sentiment_label, sentiment_score
            FROM news_articles
            {where}
            ORDER BY published_at DESC
            LIMIT :limit
        """),
        {**params, "limit": limit},
    )
    return [dict(row) for row in result.mappings()]


async def get_macro_series(
    series_id: str,
    start: datetime,
    end: datetime,
    session: AsyncSession,
    as_of: Optional[datetime] = None,
) -> list[dict]:
    """Read macro series from DB. Pass as_of for ALFRED point-in-time queries."""
    if as_of:
        result = await session.execute(
            text("""
                SELECT time, series_id, value, vintage FROM macro_data
                WHERE series_id = :series_id
                  AND time >= :start AND time <= :end
                  AND (vintage IS NULL OR vintage <= :as_of)
                ORDER BY time ASC
            """),
            dict(series_id=series_id, start=start, end=end, as_of=as_of),
        )
    else:
        result = await session.execute(
            text("""
                SELECT time, series_id, value, vintage FROM macro_data
                WHERE series_id = :series_id
                  AND time >= :start AND time <= :end
                ORDER BY time ASC
            """),
            dict(series_id=series_id, start=start, end=end),
        )
    return [dict(row) for row in result.mappings()]


# ── Data Quality Scores ───────────────────────────────────────────────────────

async def write_quality_scores(
    scores: list[dict],
    session: AsyncSession,
) -> int:
    """Upsert per-source data quality scores. Returns count written.

    Each score dict must have: ticker, interval, source, as_of (datetime),
    bars_returned, bars_expected, availability, mean_abs_deviation,
    outlier_bars, grade.
    """
    if not scores:
        return 0

    written = 0
    for s in scores:
        await session.execute(
            text("""
                INSERT INTO data_quality_scores
                    (ticker, interval, source, as_of,
                     bars_returned, bars_expected, availability,
                     mean_abs_deviation, outlier_bars, grade)
                VALUES
                    (:ticker, :interval, :source, :as_of,
                     :bars_returned, :bars_expected, :availability,
                     :mean_abs_deviation, :outlier_bars, :grade)
                ON CONFLICT (ticker, interval, source, as_of) DO UPDATE
                    SET bars_returned     = EXCLUDED.bars_returned,
                        bars_expected     = EXCLUDED.bars_expected,
                        availability      = EXCLUDED.availability,
                        mean_abs_deviation = EXCLUDED.mean_abs_deviation,
                        outlier_bars      = EXCLUDED.outlier_bars,
                        grade             = EXCLUDED.grade
            """),
            dict(
                ticker=s["ticker"],
                interval=s["interval"],
                source=s["source"],
                as_of=s["as_of"],
                bars_returned=s["bars_returned"],
                bars_expected=s["bars_expected"],
                availability=float(s["availability"]),
                mean_abs_deviation=float(s["mean_abs_deviation"]),
                outlier_bars=s["outlier_bars"],
                grade=s["grade"],
            ),
        )
        written += 1

    await session.commit()
    logger.info("Quality scores written", count=written)
    return written


async def get_quality_scores(
    ticker: str,
    interval: str,
    session: AsyncSession,
    source: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Read quality scores for ticker/interval, most recent first."""
    params: dict = {"ticker": ticker, "interval": interval, "limit": limit}

    source_filter = ""
    if source:
        source_filter = "AND source = :source"
        params["source"] = source

    result = await session.execute(
        text(f"""
            SELECT ticker, interval, source, as_of,
                   bars_returned, bars_expected, availability,
                   mean_abs_deviation, outlier_bars, grade
            FROM data_quality_scores
            WHERE ticker = :ticker AND interval = :interval
              {source_filter}
            ORDER BY as_of DESC
            LIMIT :limit
        """),
        params,
    )
    return [dict(row) for row in result.mappings()]
