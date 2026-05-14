"""Create data_quality_scores table.

Revision ID: 0004
Revises: 0003
"""
from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Create table ─────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS data_quality_scores (
            id              BIGSERIAL PRIMARY KEY,
            ticker          VARCHAR(20)       NOT NULL,
            interval        VARCHAR(10)       NOT NULL,
            source          VARCHAR(50)       NOT NULL,
            as_of           TIMESTAMPTZ       NOT NULL DEFAULT NOW(),
            bars_returned   INTEGER           NOT NULL DEFAULT 0,
            bars_expected   INTEGER           NOT NULL DEFAULT 0,
            availability    DOUBLE PRECISION  NOT NULL DEFAULT 0,
            mean_abs_deviation DOUBLE PRECISION NOT NULL DEFAULT 0,
            outlier_bars    INTEGER           NOT NULL DEFAULT 0,
            grade           VARCHAR(2)        NOT NULL DEFAULT 'F'
        );
    """)

    # ── Unique constraint: one score row per (ticker, interval, source, as_of) ─
    # We use DO $$ BEGIN ... END $$ so re-running the migration is safe.
    op.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'uq_dqs_ticker_interval_source_as_of'
            ) THEN
                ALTER TABLE data_quality_scores
                ADD CONSTRAINT uq_dqs_ticker_interval_source_as_of
                UNIQUE (ticker, interval, source, as_of);
            END IF;
        END $$;
    """)

    # ── Indexes ───────────────────────────────────────────────────────────────

    # Primary lookup: all scores for a ticker, most recent first
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_dqs_ticker_as_of
        ON data_quality_scores (ticker, as_of DESC);
    """)

    # Find the latest score for a specific (ticker, interval, source) triple
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_dqs_ticker_interval_source
        ON data_quality_scores (ticker, interval, source, as_of DESC);
    """)

    # Query by grade to surface low-quality sources across the universe
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_dqs_grade
        ON data_quality_scores (grade, as_of DESC);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS data_quality_scores CASCADE;")
