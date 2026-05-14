"""Alembic migration environment.

Reads database_url_sync from Settings so the .env file is the single
source of truth for the connection string. Never hardcode credentials here.
"""
from __future__ import annotations
import sys
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Ensure the project root is on sys.path so 'sentinel' is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentinel.core.config import get_settings

config = context.config

# Override the ini placeholder with the real DSN from Settings
config.set_main_option("sqlalchemy.url", get_settings().database_url_sync)

target_metadata = None  # No ORM models — raw SQL migrations only


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live DB connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live DB connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
