"""Checkpointer selected by ``CHECKPOINT_BACKEND``.

SQLite (``AsyncSqliteSaver``) is the only backend shipped. It reads the file
path from ``settings.checkpoint_db_path`` (``CHECKPOINT_DB_PATH``, default
``data/graph_checkpoints.sqlite``). ``compile_agent`` already accepts any
``BaseCheckpointSaver``, so switching later does not touch ``main.py`` or the
graph. Register an async opener in ``_OPENERS`` and allow its name in
``settings._CHECKPOINT_BACKENDS``.

Postgres would be that opener around ``AsyncPostgresSaver.from_conn_string``
(package ``langgraph-checkpoint-postgres``), reading a new
``CHECKPOINT_POSTGRES_URI`` setting, then ``await saver.setup()`` before yield.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from settings import settings

Opener = Callable[[], AbstractAsyncContextManager[BaseCheckpointSaver]]


@asynccontextmanager
async def _open_sqlite() -> AsyncIterator[BaseCheckpointSaver]:
    """Open the SQLite file at ``settings.checkpoint_db_path`` and create its tables.

    ``yield`` pauses here so ``from_conn_string`` stays entered while the
    caller (graph execution) uses the saver. Only one saver is produced. A ``return`` would
    leave this function first, exit the ``async with``, and hand back a saver
    whose SQLite connection is already closed. ``@asynccontextmanager`` also
    requires exactly one ``yield``: code before it is setup, the yielded value
    is the saver, and resuming after it closes the file.
    """
    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db_path)) as saver:
        await saver.setup()
        yield saver


# One entry per backend. The key must also be listed in settings._CHECKPOINT_BACKENDS.
_OPENERS: dict[str, Opener] = {
    "sqlite": _open_sqlite,
}


@asynccontextmanager
async def open_checkpointer() -> AsyncIterator[BaseCheckpointSaver]:
    """Open the saver named by ``settings.checkpoint_backend``.

    Re-yields that one saver so the opener stays entered for the caller's
    ``async with``. Returning it would exit the opener and close the connection
    before ``compile_agent`` runs.
    """
    opener = _OPENERS.get(settings.checkpoint_backend)
    if opener is None:
        allowed = ", ".join(sorted(_OPENERS))
        raise ValueError(f"CHECKPOINT_BACKEND must be one of: {allowed}")
    async with opener() as saver:
        yield saver
