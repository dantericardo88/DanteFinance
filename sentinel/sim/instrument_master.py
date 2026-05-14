"""Instrument Master — unified identifier store backed by SQLAlchemy + OpenFIGI."""
from __future__ import annotations
import asyncio
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, select, Index
from sentinel.core.types import Instrument, AssetClass
from sentinel.core.logging import get_logger
from sentinel.sim.openfigi_client import OpenFIGIClient

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


class InstrumentRow(Base):
    __tablename__ = "instruments"
    figi: Mapped[str] = mapped_column(String(12), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    isin: Mapped[Optional[str]] = mapped_column(String(12), nullable=True, index=True)
    cusip: Mapped[Optional[str]] = mapped_column(String(9), nullable=True, index=True)
    sedol: Mapped[Optional[str]] = mapped_column(String(7), nullable=True)
    name: Mapped[str] = mapped_column(String(256))
    exchange: Mapped[str] = mapped_column(String(20))
    asset_class: Mapped[str] = mapped_column(String(30))
    currency: Mapped[str] = mapped_column(String(3))
    security_type: Mapped[str] = mapped_column(String(50))
    sector: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    industry: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    gics_sector: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    sic_code: Mapped[Optional[str]] = mapped_column(String(6), nullable=True)

    __table_args__ = (
        Index("ix_instruments_ticker_exchange", "ticker", "exchange"),
    )


class InstrumentMaster:
    """Lookup and upsert instruments. Wraps OpenFIGI + DB cache."""

    def __init__(self, db_url: str, openfigi_key: str = "") -> None:
        self._engine = create_async_engine(db_url, pool_size=5, max_overflow=10)
        self._figi_client = OpenFIGIClient(api_key=openfigi_key)
        self._mem_cache: dict[str, Instrument] = {}

    async def init_db(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def get_by_figi(self, figi: str) -> Optional[Instrument]:
        if figi in self._mem_cache:
            return self._mem_cache[figi]
        async with AsyncSession(self._engine) as session:
            row = await session.get(InstrumentRow, figi)
            if row:
                inst = _row_to_instrument(row)
                self._mem_cache[figi] = inst
                return inst
        return None

    async def get_by_ticker(
        self, ticker: str, exchange: str = "US"
    ) -> Optional[Instrument]:
        cache_key = f"{ticker}:{exchange}"
        if cache_key in self._mem_cache:
            return self._mem_cache[cache_key]
        async with AsyncSession(self._engine) as session:
            stmt = select(InstrumentRow).where(
                InstrumentRow.ticker == ticker.upper(),
                InstrumentRow.exchange == exchange,
            ).limit(1)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row:
                inst = _row_to_instrument(row)
                self._mem_cache[cache_key] = inst
                self._mem_cache[row.figi] = inst
                return inst

        # Not in DB — resolve via OpenFIGI
        figi = await self._figi_client.ticker_to_figi(ticker, exch_code=exchange)
        if figi:
            # Fetch full record to get name/security_type
            jobs = [{"idType": "TICKER", "idValue": ticker, "exchCode": exchange}]
            records = await self._figi_client.map_identifiers(jobs)
            if records and records[0]:
                inst = self._figi_client.figi_record_to_instrument(records[0][0], ticker)
                await self.upsert(inst)
                self._mem_cache[cache_key] = inst
                self._mem_cache[figi] = inst
                return inst
        return None

    async def resolve_ticker_to_figi(self, ticker: str, exchange: str = "US") -> Optional[str]:
        inst = await self.get_by_ticker(ticker, exchange)
        return inst.figi if inst else None

    async def upsert(self, instrument: Instrument) -> None:
        async with AsyncSession(self._engine) as session:
            row = await session.get(InstrumentRow, instrument.figi)
            if row is None:
                row = InstrumentRow(figi=instrument.figi)
                session.add(row)
            row.ticker = instrument.ticker
            row.name = instrument.name
            row.exchange = instrument.exchange
            row.asset_class = instrument.asset_class.value
            row.currency = instrument.currency
            row.security_type = instrument.security_type or ""
            row.sector = instrument.sector
            row.industry = instrument.industry
            await session.commit()
        self._mem_cache[instrument.figi] = instrument
        logger.info("Instrument upserted", figi=instrument.figi, ticker=instrument.ticker)

    async def bulk_resolve(
        self, tickers: list[str], exchange: str = "US"
    ) -> dict[str, Optional[str]]:
        """Resolve many tickers to FIGIs using bulk OpenFIGI + DB cache."""
        result: dict[str, Optional[str]] = {}
        uncached = []
        for t in tickers:
            inst = self._mem_cache.get(f"{t}:{exchange}")
            if inst:
                result[t] = inst.figi
            else:
                uncached.append(t)

        if uncached:
            figi_map = await self._figi_client.bulk_resolve_tickers(uncached, exchange)
            result.update({t: f for t, f in figi_map.items()})
        return result

    async def search(self, query: str) -> list[Instrument]:
        records = await self._figi_client.search(query)
        return [self._figi_client.figi_record_to_instrument(r) for r in records]


def _row_to_instrument(row: InstrumentRow) -> Instrument:
    try:
        ac = AssetClass(row.asset_class)
    except ValueError:
        ac = AssetClass.EQUITY
    return Instrument(
        figi=row.figi,
        ticker=row.ticker,
        name=row.name,
        exchange=row.exchange,
        asset_class=ac,
        currency=row.currency,
        security_type=row.security_type,
        sector=row.sector,
        industry=row.industry,
    )
