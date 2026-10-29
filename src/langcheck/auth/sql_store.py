"""``ApiKeyStore`` in the SQL database."""

from collections.abc import Callable
from datetime import datetime

from sqlalchemy import Column, DateTime, Engine, String, Table, delete, insert, select
from sqlalchemy.exc import IntegrityError

from langcheck.auth.models import User, UserExistsError
from langcheck.database.schema import METADATA

API_KEYS = Table(
    "api_keys",
    METADATA,
    Column("user", String(200), primary_key=True),
    Column("token_hash", String(64), unique=True, index=True, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


class SqlApiKeyStore:
    """Stores API key hashes with SQLAlchemy. Calls block; run them off the event loop."""

    def __init__(self, engine: Engine, *, clock: Callable[[], datetime]) -> None:
        self._engine = engine
        self._clock = clock

    def user_for(self, token_hash: str) -> User | None:
        """The user whose key has ``token_hash``, or ``None``."""
        query = select(API_KEYS.c.user).where(API_KEYS.c.token_hash == token_hash)
        with self._engine.connect() as connection:
            name: object = connection.execute(query).scalar_one_or_none()
        return User(name) if isinstance(name, str) else None

    def add(self, name: str, token_hash: str) -> None:
        """Give ``name`` the key with ``token_hash``; ``UserExistsError`` if it has one."""
        row = {"user": name, "token_hash": token_hash, "created_at": self._clock()}
        try:
            with self._engine.begin() as connection:
                connection.execute(insert(API_KEYS).values(row))
        except IntegrityError as error:
            msg = f"user {name!r} already has an API key; revoke it first"
            raise UserExistsError(msg) from error

    def revoke(self, name: str) -> bool:
        """Remove ``name``'s key; ``False`` when there was none."""
        with self._engine.begin() as connection:
            found: object = connection.execute(
                select(API_KEYS.c.user).where(API_KEYS.c.user == name)
            ).scalar_one_or_none()
            connection.execute(delete(API_KEYS).where(API_KEYS.c.user == name))
        return found is not None

    def users(self) -> list[str]:
        """The names of every user with a key, sorted."""
        with self._engine.connect() as connection:
            names: list[object] = list(
                connection.execute(select(API_KEYS.c.user).order_by(API_KEYS.c.user)).scalars()
            )
        return [name for name in names if isinstance(name, str)]
