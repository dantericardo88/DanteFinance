"""Add unique constraints and indexes for insider, institutional, and news tables.

Revision ID: 0003
Revises: 0002
"""
from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # insider_transactions: unique on (cik, ticker, tx_date, tx_code, shares)
    # — two transactions on the same day with identical shares/code would be
    #   extremely rare; this prevents double-ingest without losing real data.
    op.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'uq_insider_cik_ticker_date_code_shares'
            ) THEN
                ALTER TABLE insider_transactions
                ADD CONSTRAINT uq_insider_cik_ticker_date_code_shares
                UNIQUE (cik, ticker, tx_date, tx_code, shares);
            END IF;
        END $$;
    """)
    # Index for fast lookups by ticker + date (most common query pattern)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_insider_ticker_date
        ON insider_transactions (ticker, tx_date DESC);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_insider_owner
        ON insider_transactions (owner_name, tx_date DESC);
    """)

    # institutional_holdings: unique on (manager_cik, cusip, period_of_report)
    op.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'uq_holdings_manager_cusip_period'
            ) THEN
                ALTER TABLE institutional_holdings
                ADD CONSTRAINT uq_holdings_manager_cusip_period
                UNIQUE (manager_cik, cusip, period_of_report);
            END IF;
        END $$;
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_holdings_ticker_period
        ON institutional_holdings (ticker, period_of_report DESC)
        WHERE ticker IS NOT NULL;
    """)

    # news_articles: index on url (non-null urls only)
    # NOTE: TimescaleDB hypertable on published_at — unique indexes must include
    # the partitioning column. Using a non-unique index here; deduplication is
    # enforced at the ingestion layer instead.
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_news_url
        ON news_articles (url)
        WHERE url IS NOT NULL;
    """)
    # Partial index for sentiment analysis queries
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_news_sentiment
        ON news_articles (sentiment_label, published_at DESC)
        WHERE sentiment_label IS NOT NULL;
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_news_sentiment")
    op.execute("DROP INDEX IF EXISTS ix_news_url")
    op.execute("DROP INDEX IF EXISTS ix_holdings_ticker_period")
    op.execute("""
        ALTER TABLE institutional_holdings
        DROP CONSTRAINT IF EXISTS uq_holdings_manager_cusip_period
    """)
    op.execute("DROP INDEX IF EXISTS ix_insider_owner")
    op.execute("DROP INDEX IF EXISTS ix_insider_ticker_date")
    op.execute("""
        ALTER TABLE insider_transactions
        DROP CONSTRAINT IF EXISTS uq_insider_cik_ticker_date_code_shares
    """)
