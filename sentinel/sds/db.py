"""Async SQLAlchemy engine and session management.

Single source of truth for DB connectivity across the entire application.

FastAPI route usage:
    async def route(session: AsyncSession = Depends(get_session)):
        ...

Scheduler / script usage:
    async with get_session_factory()() as session:
        await repository.write_ohlcv_bars(bars, "AAPL", "1d", session)
"""
from __future__ import annotations
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from sentinel.core.config import get_settings
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        s = get_settings()
        _engine = create_async_engine(
            s.database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False,
        )
        logger.info("DB engine initialised", url=s.database_url[:50])
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a transactional async session."""
    async with get_session_factory()() as session:
        yield session


async def dispose_engine() -> None:
    """Call on application shutdown to cleanly close all connections."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        logger.info("DB engine disposed")
