"""Alembic environment — derives the DB URL from SecondBrain's config."""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from secondbrain.config import get_settings


def _db_url() -> str:
    return f"sqlite:///{get_settings().db_path}"


def run_migrations_offline() -> None:
    context.configure(url=_db_url(), literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    settings = get_settings()
    settings.ensure_dirs()
    engine = create_engine(_db_url())
    with engine.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
