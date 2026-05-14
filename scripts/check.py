"""SENTINEL system health check — run before backfill to verify everything is wired.

Checks:
  1. PostgreSQL connectivity and all required tables
  2. TimescaleDB hypertable status
  3. Alembic migration state
  4. Adapter chain health (yfinance, FRED, EDGAR)
  5. Data lake population (row counts per table)
  6. Environment variables (warns on missing paid keys, errors on missing required)

Usage:
    python scripts/check.py             # full check
    python scripts/check.py --quick     # skip adapter network calls
    make check
"""
from __future__ import annotations
import argparse
import asyncio
import sys
from datetime import date

from sentinel.core.config import get_settings
from sentinel.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger("check")

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"
INFO = "  [INFO]"

_errors = 0
_warnings = 0


def _ok(msg: str) -> None:
    print(f"{PASS} {msg}")


def _err(msg: str) -> None:
    global _errors
    _errors += 1
    print(f"{FAIL} {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    global _warnings
    _warnings += 1
    print(f"{WARN} {msg}")


def _info(msg: str) -> None:
    print(f"{INFO} {msg}")


# ── 1. Environment variables ──────────────────────────────────────────────────

def check_env() -> None:
    print("\n── Environment Variables ──────────────────────────────────────────")
    s = get_settings()

    required = {
        "database_url": s.database_url,
        "edgar_user_agent": s.edgar_user_agent,
    }
    for key, val in required.items():
        if val and "sentinel@example.com" not in val:
            _ok(f"{key} set")
        elif key == "edgar_user_agent":
            _warn(f"{key} uses placeholder — update to your real email for SEC compliance")
        else:
            _err(f"{key} is not set")

    optional = {
        "fred_api_key": s.fred_api_key,
        "finnhub_api_key": s.finnhub_api_key,
        "polygon_api_key": s.polygon_api_key,
        "alpaca_api_key": s.alpaca_api_key,
        "openfigi_api_key": s.openfigi_api_key,
        "anthropic_api_key": s.anthropic_api_key,
    }
    for key, val in optional.items():
        if val:
            _ok(f"{key} set")
        else:
            _warn(f"{key} not set — some features degraded (free tier still works)")

    if not s.database_url_sync:
        _err("DATABASE_URL_SYNC not set — Alembic migrations will fail")
    else:
        _ok("DATABASE_URL_SYNC set (Alembic ready)")


# ── 2. PostgreSQL connectivity ────────────────────────────────────────────────

REQUIRED_TABLES = [
    "instruments", "ohlcv", "macro_data", "financial_facts",
    "corporate_actions", "survivorship_registry", "data_provenance",
    "congressional_trades", "cot_data",
    "insider_transactions", "institutional_holdings",
    "news_articles", "backtest_results", "strategy_registry",
    "data_quality_scores",
]

HYPERTABLES = ["ohlcv", "macro_data"]


async def check_database() -> None:
    print("\n── PostgreSQL / TimescaleDB ────────────────────────────────────────")
    from sqlalchemy import text
    from sentinel.sds.db import get_session_factory

    try:
        session_factory = get_session_factory()
        async with session_factory() as session:

            # Basic connectivity
            result = await session.execute(text("SELECT version()"))
            version: str = str(result.scalar() or "")
            _ok(f"PostgreSQL connected: {version[:60]}")

            # TimescaleDB extension
            result = await session.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
            )
            row = result.fetchone()
            if row:
                _ok(f"TimescaleDB extension: v{row[0]}")
            else:
                _err("TimescaleDB extension NOT installed — hypertables won't work")

            # Table existence
            for table in REQUIRED_TABLES:
                result = await session.execute(
                    text("""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables
                            WHERE table_name = :tbl AND table_schema = 'public'
                        )
                    """),
                    {"tbl": table},
                )
                exists = result.scalar()
                if exists:
                    _ok(f"Table: {table}")
                else:
                    _err(f"Table MISSING: {table} — run: make db-migrate")

            # Hypertable status
            for ht in HYPERTABLES:
                result = await session.execute(
                    text("""
                        SELECT COUNT(*) FROM timescaledb_information.hypertables
                        WHERE hypertable_name = :ht
                    """),
                    {"ht": ht},
                )
                count = result.scalar()
                if count:
                    _ok(f"Hypertable: {ht}")
                else:
                    _warn(f"Not a hypertable: {ht} — run: make db-init (check 02_hypertables.sql)")

            # Row counts
            print()
            counts = (await session.execute(text("""
                SELECT
                  (SELECT COUNT(*) FROM ohlcv)                   AS ohlcv,
                  (SELECT COUNT(*) FROM macro_data)              AS macro_data,
                  (SELECT COUNT(*) FROM financial_facts)         AS financial_facts,
                  (SELECT COUNT(*) FROM corporate_actions)       AS corporate_actions,
                  (SELECT COUNT(*) FROM survivorship_registry)   AS survivorship_registry,
                  (SELECT COUNT(*) FROM data_provenance)         AS data_provenance,
                  (SELECT COUNT(*) FROM congressional_trades)    AS congressional_trades,
                  (SELECT COUNT(*) FROM cot_data)                AS cot_data,
                  (SELECT COUNT(*) FROM insider_transactions)    AS insider_transactions,
                  (SELECT COUNT(*) FROM institutional_holdings)  AS institutional_holdings,
                  (SELECT COUNT(*) FROM news_articles)           AS news_articles
            """))).mappings().fetchone()

            for table, count in (dict(counts) if counts else {}).items():
                if count == 0:
                    _warn(f"{table}: EMPTY — run: make backfill")
                else:
                    _ok(f"{table}: {count:,} rows")

    except Exception as exc:
        _err(f"Database connection failed: {exc}")
        _info("Is the database running? Try: make docker-up")


# ── 3. Alembic migration state ────────────────────────────────────────────────

async def check_migrations() -> None:
    print("\n── Alembic Migrations ─────────────────────────────────────────────")
    try:
        import subprocess
        result = subprocess.run(
            ["poetry", "run", "alembic", "current"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            current = result.stdout.strip() or "(none)"
            _ok(f"Alembic current: {current}")
            if "0004" not in current:
                if "0002" not in current:
                    _warn("Latest migration 0002 not applied — run: make db-migrate")
                elif "0003" not in current:
                    _warn("Latest migration 0003 not applied — run: make db-migrate")
                else:
                    _warn("Latest migration 0004 not applied — run: make db-migrate")
        else:
            _err(f"Alembic error: {result.stderr[:200]}")
    except Exception as exc:
        _warn(f"Alembic check skipped: {exc}")


# ── 4. Adapter health checks ──────────────────────────────────────────────────

async def check_adapters() -> None:
    print("\n── Data Adapters ──────────────────────────────────────────────────")
    from sentinel.sds import build_default_adapters
    from sentinel.sds.normalizer import run_all_health_checks

    build_default_adapters()
    events = await run_all_health_checks()
    for ev in events:
        sev = ev.severity.value if hasattr(ev.severity, "value") else str(ev.severity)
        if sev in ("ok", "info"):
            _ok(f"{ev.adapter}: {ev.event_type}")
        elif sev == "warning":
            _warn(f"{ev.adapter}: {ev.event_type} — {ev.details}")
        else:
            _err(f"{ev.adapter}: {ev.event_type} — {ev.details}")


# ── 5. Quick import test ───────────────────────────────────────────────────────

def check_imports() -> None:
    print("\n── Critical Imports ───────────────────────────────────────────────")
    modules = [
        ("sentinel.sds.db", "get_session_factory"),
        ("sentinel.sds.repository", "write_ohlcv_bars"),
        ("sentinel.sds.ingest", "ingest_ticker_ohlcv"),
        ("sentinel.sds.catalog", "get_coverage"),
        ("sentinel.sds.scheduler", "start_scheduler"),
        ("sentinel.sds.corporate_actions", "fetch_and_apply"),
        ("sentinel.sds.provenance", "create_receipt"),
        ("sentinel.sds.validator", "validate_single_source"),
        ("sentinel.sds.gap_detector", "detect_gaps"),
        ("sentinel.sds.survivorship", "get_all_delisted"),
        ("sentinel.sds.adapters.congress_adapter", "CongressAdapter"),
        ("sentinel.sds.continuous_futures", "build_continuous_series"),
        ("sentinel.sds.data_cleaner", "DataCleaner"),
        ("sentinel.sma.cot_report", "COTClient"),
        ("sentinel.sfe.xbrl_parser", "extract_facts"),
        ("sentinel.api.main", "app"),
    ]
    for module, attr in modules:
        try:
            mod = __import__(module, fromlist=[attr])
            getattr(mod, attr)
            _ok(f"{module}.{attr}")
        except (ImportError, AttributeError) as exc:
            _err(f"{module}.{attr}: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="SENTINEL system health check")
    parser.add_argument("--quick", action="store_true",
                        help="Skip network calls (adapter health, migration state)")
    args = parser.parse_args()

    print("=" * 64)
    print(" SENTINEL System Health Check".center(64))
    print(f" {date.today().isoformat()}".center(64))
    print("=" * 64)

    check_env()
    check_imports()
    await check_database()

    if not args.quick:
        await check_migrations()
        await check_adapters()

    print("\n" + "=" * 64)
    if _errors == 0 and _warnings == 0:
        print("  ALL CHECKS PASSED — ready to run: make backfill")
    elif _errors == 0:
        print(f"  {_warnings} warning(s) — system functional, some features degraded")
    else:
        print(f"  {_errors} error(s), {_warnings} warning(s) — fix errors before backfill")
    print("=" * 64)

    sys.exit(1 if _errors > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
