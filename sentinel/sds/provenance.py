"""Data provenance receipt chain.

Every ingestion batch receives a SHA-256 receipt. Receipts are linked via
prev_hash to form an append-only audit chain — any tampering with stored
data is detectable by re-hashing the source.

Usage:
    bars = await fetch_ohlcv_with_fallback("AAPL", start, end)
    receipt = create_receipt(bars, source="yfinance", ticker="AAPL", interval="1d")
    # receipt.sha256 uniquely fingerprints this exact batch
"""
from __future__ import annotations
import hashlib
import json
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional

from sentinel.core.types import DataProvenance, OHLCVBar
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# In-memory chain: (ticker, source, interval) → latest sha256
# Persisted records should populate this on startup for true continuity.
_chain: dict[str, str] = {}


def _bars_to_bytes(bars: list[OHLCVBar]) -> bytes:
    """Deterministic serialization for hashing — canonical field order, sorted by time."""
    records = sorted(
        [
            {
                "time": b.time.isoformat(),
                "open": str(b.open),
                "high": str(b.high),
                "low": str(b.low),
                "close": str(b.close),
                "volume": b.volume,
            }
            for b in bars
        ],
        key=lambda x: x["time"],
    )
    return json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_sha256(bars: list[OHLCVBar]) -> str:
    return hashlib.sha256(_bars_to_bytes(bars)).hexdigest()


def create_receipt(
    bars: list[OHLCVBar],
    source: str,
    ticker: str,
    interval: str,
    figi: Optional[str] = None,
) -> DataProvenance:
    """Create a provenance receipt for a batch of bars.

    Links to the previous receipt for this (ticker, source, interval) tuple
    via prev_hash. The first receipt has prev_hash=None.
    """
    if not bars:
        raise ValueError(f"Cannot create receipt for empty bar list ({ticker})")

    sha = compute_sha256(bars)
    chain_key = f"{ticker}:{source}:{interval}"
    prev = _chain.get(chain_key)

    receipt = DataProvenance(
        batch_id=str(uuid.uuid4()),
        source=source,
        ticker=ticker,
        figi=figi,
        interval=interval,
        start_time=min(b.time for b in bars),
        end_time=max(b.time for b in bars),
        bar_count=len(bars),
        sha256=sha,
        prev_hash=prev,
        validated=False,
        ingested_at=datetime.utcnow(),
    )

    _chain[chain_key] = sha
    logger.info(
        "Provenance receipt created",
        ticker=ticker, source=source, interval=interval,
        bars=len(bars), sha256=sha[:16],
    )
    return receipt


def verify_chain_integrity(receipts: list[DataProvenance]) -> list[str]:
    """Validate that prev_hash links are consistent across a sorted receipt sequence.

    Returns a list of error strings. Empty list = clean chain.
    """
    errors: list[str] = []
    ordered = sorted(receipts, key=lambda r: r.ingested_at)

    prev_sha: Optional[str] = None
    for r in ordered:
        if prev_sha is not None and r.prev_hash != prev_sha:
            errors.append(
                f"Chain break at batch {r.batch_id}: "
                f"prev_hash={r.prev_hash!r} expected={prev_sha!r}"
            )
        prev_sha = r.sha256

    if errors:
        logger.error("Provenance chain integrity failures", count=len(errors))

    return errors


def seed_chain_from_db(records: list[DataProvenance]) -> None:
    """Populate in-memory chain state from persisted provenance records on startup."""
    for r in sorted(records, key=lambda x: x.ingested_at):
        chain_key = f"{r.ticker}:{r.source}:{r.interval}"
        _chain[chain_key] = r.sha256
    logger.info("Provenance chain seeded from DB", records=len(records))
