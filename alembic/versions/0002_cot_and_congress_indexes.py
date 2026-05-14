"""Add COT data table and congressional trades performance indexes.

Changes:
  1. cot_data — CFTC Commitments of Traders raw positions table
     Enables persisting COT history to DB instead of in-memory only.
     Primary key: (report_date, market_name) — one row per market per week.

  2. Indexes on congressional_trades — ticker + tx_date lookup for terminal CN view

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-07
"""
from __future__ import annotations
from alembic import op

revision: str = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── cot_data table ────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS cot_data (
            id                  BIGSERIAL PRIMARY KEY,
            report_date         DATE NOT NULL,
            market_name         VARCHAR(100) NOT NULL,
            commodity_code      VARCHAR(10),
            open_interest       BIGINT,
            comm_long           BIGINT,    -- commercial longs
            comm_short          BIGINT,    -- commercial shorts
            noncomm_long        BIGINT,    -- non-commercial (speculator) longs
            noncomm_short       BIGINT,    -- non-commercial shorts
            nonrept_long        BIGINT,    -- non-reportable (retail) longs
            nonrept_short       BIGINT,    -- non-reportable shorts
            net_speculator      BIGINT,    -- noncomm_long - noncomm_short
            cot_index           NUMERIC(6,2),   -- 52-week percentile 0-100
            signal              VARCHAR(20),    -- extreme_long / extreme_short / neutral
            source              VARCHAR(30) NOT NULL DEFAULT 'cftc',
            created_at          TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (report_date, market_name)
        );
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_cot_market_date
            ON cot_data (market_name, report_date DESC);
    """)

    # ── congressional_trades indexes (table already in init.sql) ───────────────
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_congress_ticker_date
            ON congressional_trades (ticker, tx_date DESC)
            WHERE ticker IS NOT NULL;
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_congress_politician
            ON congressional_trades (politician_name, tx_date DESC);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_congress_late
            ON congressional_trades (late_filing, filed_date DESC)
            WHERE late_filing = TRUE;
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_congress_late;")
    op.execute("DROP INDEX IF EXISTS ix_congress_politician;")
    op.execute("DROP INDEX IF EXISTS ix_congress_ticker_date;")
    op.execute("DROP INDEX IF EXISTS ix_cot_market_date;")
    op.execute("DROP TABLE IF EXISTS cot_data;")
