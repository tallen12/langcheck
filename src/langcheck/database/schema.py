"""The metadata every table joins, and opening the database."""

from sqlalchemy import Engine, MetaData, create_engine

METADATA = MetaData()
"""Every table is defined on this; a table joins the schema when its module is imported."""


def open_database(url: str) -> Engine:
    """An engine for ``url`` with every imported table created if missing.

    Migrations come with Alembic once the schema first changes; until then creating
    missing tables is enough. Import the table modules (for example
    ``langcheck.auth.sql_store``) before calling this.
    """
    engine = create_engine(url)
    METADATA.create_all(engine)
    return engine
