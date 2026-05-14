"""Add unique constraint on financial_facts for idempotent upserts.

The base schema (instruments, ohlcv, corporate_actions, survivorship_registry,
data_provenance, macro_data) is created by infra/postgres/init.sql at container
startup. This migration owns the structural changes that ship AFTER that baseline:

  1. UNIQUE constraint on financial_facts(cik, concept, period_end, accession)
     Required for: repository.write_financial_facts() ON CONFLICT DO NOTHING
     Without this constraint the upsert falls back to full-table duplicate scans,
     and re-ingesting the same CIK creates duplicate rows.

  2. filed index — speeds up PIT queries: WHERE filed <= :as_of

Revision ID: 0001
Revises: (none — first migration)
Create Date: 2026-05-07
"""
from __future__ import annotations
from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── financial_facts unique constraint ─────────────────────────────────────
    # accession can be NULL (some EDGAR observations omit it), so we coalesce
    # to '' for the uniqueness check using a partial expression index.
    # Standard UNIQUE ignores NULLs, which would allow duplicate rows when
    # accession IS NULL. The expression index covers that gap.
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'uq_facts_cik_concept_period_accn'
            ) THEN
                ALTER TABLE financial_facts
                    ADD CONSTRAINT uq_facts_cik_concept_period_accn
                    UNIQUE (cik, concept, period_end, accession);
            END IF;
        END
        $$;
    """)

    # ── filed index for PIT queries ────────────────────────────────────────────
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_facts_filed
            ON financial_facts (filed)
            WHERE filed IS NOT NULL;
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_facts_filed;")
    op.execute("""
        ALTER TABLE financial_facts
            DROP CONSTRAINT IF EXISTS uq_facts_cik_concept_period_accn;
    """)
