"""
Tests that capture pool starvation and nested session issues in PostgresDataStore.

Uses an in-memory SQLite database via aiosqlite — no external Postgres required.

Install deps:
    pip install aiosqlite

Run:
    pytest tests/contract/data_store/test_postgres_pool_starvation.py -v
"""
import asyncio
import os
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from llamphouse.core.data_stores.postgres_store import PostgresDataStore
from llamphouse.core.database.models import Base  # SQLAlchemy declarative base
from llamphouse.core.types.message import CreateMessageRequest
from llamphouse.core.types.thread import CreateThreadRequest
from llamphouse.core.types.run import RunCreateRequest

# ---------------------------------------------------------------------------
# In-memory SQLite engine factory
# ---------------------------------------------------------------------------
# SQLite in-memory databases are per-connection by default.
# StaticPool forces all async sessions to reuse the same connection,
# which is required for the schema to be visible across sessions.
# We also disable pool_size / max_overflow because StaticPool ignores them.

SQLITE_URL = "sqlite+aiosqlite:///:memory:"

# If LLAMPHOUSE_TEST_DB_URL is set, use a real Postgres instance instead.
TEST_DB_URL = os.environ.get("LLAMPHOUSE_TEST_DB_URL")
USE_POSTGRES = TEST_DB_URL is not None

TINY_POOL_SIZE = 2
ZERO_OVERFLOW = 0


def _patch_sqlite_jsonb():
    """
    Teach SQLite's type compiler to render JSONB as TEXT.

    models.py picks JSONType at import time based on DATABASE_URL.
    When the parent conftest already imported llamphouse.core (with JSONB),
    we can't re-run that selection — but we CAN patch the SQLite dialect
    compiler so it knows how to emit JSONB columns.
    """
    from sqlalchemy.dialects.sqlite.base import SQLiteTypeCompiler

    if not hasattr(SQLiteTypeCompiler, "visit_JSONB"):
        SQLiteTypeCompiler.visit_JSONB = SQLiteTypeCompiler.visit_JSON


async def _create_sqlite_store(pool_size: int = TINY_POOL_SIZE) -> PostgresDataStore:
    """
    Build a PostgresDataStore backed by an in-memory SQLite database.

    We bypass the normal __init__ engine creation so we can inject
    StaticPool (required for in-memory SQLite) and create the schema.
    """
    _patch_sqlite_jsonb()

    engine = create_async_engine(
        SQLITE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        # SQLite doesn't support pool_size / max_overflow — omit them
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    store = PostgresDataStore.__new__(PostgresDataStore)
    store._engine = engine
    store._session_factory = session_factory

    # Attach a close helper that drops tables and disposes the engine
    original_close = PostgresDataStore.close

    async def _close(self):
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()

    store.close = lambda: _close(store)

    return store


async def _create_postgres_store(pool_size: int = TINY_POOL_SIZE) -> PostgresDataStore:
    return PostgresDataStore(
        database_url=TEST_DB_URL,
        pool_size=pool_size,
        max_overflow=ZERO_OVERFLOW,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def tiny_store():
    """
    Store with a tiny pool.
    Uses in-memory SQLite unless LLAMPHOUSE_TEST_DB_URL is set.
    """
    if USE_POSTGRES:
        store = await _create_postgres_store(pool_size=TINY_POOL_SIZE)
    else:
        store = await _create_sqlite_store(pool_size=TINY_POOL_SIZE)
    yield store
    await store.close()


@pytest_asyncio.fixture
async def normal_store():
    """
    Store with default pool settings.
    Uses in-memory SQLite unless LLAMPHOUSE_TEST_DB_URL is set.
    """
    if USE_POSTGRES:
        store = await _create_postgres_store(pool_size=5)
    else:
        store = await _create_sqlite_store(pool_size=5)
    yield store
    await store.close()


def _make_thread_request(n_messages: int = 2) -> CreateThreadRequest:
    messages = [
        CreateMessageRequest(role="user", content=f"message {i}")
        for i in range(n_messages)
    ]
    return CreateThreadRequest(messages=messages)


def _make_mock_assistant():
    assistant = MagicMock()
    assistant.id = "asst_test"
    assistant.model = "gpt-4"
    assistant.instructions = "test"
    assistant.tools = []
    assistant.temperature = None
    assistant.top_p = None
    assistant.reasoning_effort = None
    return assistant


# ---------------------------------------------------------------------------
# 1. Nested session deadlock — insert_thread
# ---------------------------------------------------------------------------

class TestNestedSessionDeadlockInsertThread:
    """
    insert_thread() holds session A, then calls self.insert_message() which
    opens session B.  With a tiny pool this causes starvation / deadlock.

    Note: SQLite + StaticPool uses a single shared connection so it will NOT
    deadlock on pool exhaustion — but it WILL deadlock on re-entrant locking.
    Pool exhaustion tests only run meaningfully against Postgres.
    The session-counting tests (TestPostFixRegression) work with both backends.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not USE_POSTGRES, reason="Pool exhaustion only reproducible with Postgres")
    async def test_concurrent_insert_thread_with_messages_does_not_deadlock(self, tiny_store):
        """
        EXPECTED TO FAIL with current code (Postgres only).
        pool_size=2, each insert_thread with 1 message needs 2 connections.
        2 concurrent calls → 4 connections needed, only 2 available → starvation.
        """
        tasks = [
            tiny_store.insert_thread(_make_thread_request(n_messages=1))
            for _ in range(TINY_POOL_SIZE)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        hard_errors = [r for r in results if isinstance(r, Exception)]
        silent_failures = [r for r in results if r is None]
        successes = [r for r in results if r is not None and not isinstance(r, Exception)]

        assert not hard_errors, f"Hard exceptions: {hard_errors}"
        assert not silent_failures, (
            f"Pool starvation: {len(silent_failures)}/{len(results)} calls silently failed (returned None).\n"
            f"Successes: {len(successes)}\n"
            f"Root cause: insert_thread() opens a session then calls insert_message() which opens another.\n"
            f"Fix: pass the existing session into insert_message() instead of acquiring a new one."
        )

    @pytest.mark.asyncio
    async def test_concurrent_insert_thread_no_messages_is_baseline(self, tiny_store):
        """
        Baseline: insert_thread WITHOUT messages should never deadlock.
        Runs on both SQLite and Postgres.
        """
        tasks = [
            tiny_store.insert_thread(CreateThreadRequest(messages=[]))
            for _ in range(TINY_POOL_SIZE * 4)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, f"Baseline failed with exceptions: {errors}"


# ---------------------------------------------------------------------------
# 2. Nested session deadlock — insert_run
# ---------------------------------------------------------------------------

class TestNestedSessionDeadlockInsertRun:

    @pytest.mark.asyncio
    @pytest.mark.skipif(not USE_POSTGRES, reason="Pool exhaustion only reproducible with Postgres")
    async def test_concurrent_insert_run_with_additional_messages_does_not_deadlock(self, tiny_store):
        """EXPECTED TO FAIL with current code (Postgres only)."""
        threads = await asyncio.gather(*[
            tiny_store.insert_thread(CreateThreadRequest(messages=[]))
            for _ in range(TINY_POOL_SIZE)
        ])
        assert all(t is not None for t in threads), "Setup failed: could not create threads"

        tasks = [
            tiny_store.insert_run(
                thread.id,
                RunCreateRequest(
                    assistant_id="asst_test",
                    additional_messages=[
                        CreateMessageRequest(role="user", content=f"msg {i}")
                        for i in range(2)
                    ],
                ),
                _make_mock_assistant(),
            )
            for thread in threads
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        hard_errors = [r for r in results if isinstance(r, Exception)]
        silent_failures = [r for r in results if r is None]

        assert not hard_errors, f"Hard exceptions: {hard_errors}"
        assert not silent_failures, (
            f"Pool starvation in insert_run(): {len(silent_failures)}/{len(results)} calls returned None.\n"
            f"Root cause: insert_run() holds session A and calls insert_message() which needs session B."
        )


# ---------------------------------------------------------------------------
# 3. High-concurrency stress test
# ---------------------------------------------------------------------------

class TestPoolExhaustionUnderConcurrency:

    @pytest.mark.asyncio
    @pytest.mark.skipif(not USE_POSTGRES, reason="Pool exhaustion only reproducible with Postgres")
    async def test_20_concurrent_insert_threads_with_messages(self, normal_store):
        """EXPECTED TO FAIL with current code (Postgres only)."""
        tasks = [
            normal_store.insert_thread(_make_thread_request(n_messages=1))
            for _ in range(20)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        hard_errors = [r for r in results if isinstance(r, Exception)]
        silent_failures = [r for r in results if r is None]
        successes = [r for r in results if r is not None and not isinstance(r, Exception)]

        assert not hard_errors, f"Hard exceptions: {hard_errors}"
        assert not silent_failures, (
            f"{len(silent_failures)}/20 inserts silently returned None (pool starvation swallowed).\n"
            f"Successes: {len(successes)}/20"
        )
        assert len(successes) == 20


# ---------------------------------------------------------------------------
# 4. Error propagation behaviour — documents what is/isn't surfaced to callers
# ---------------------------------------------------------------------------

class TestErrorPropagation:
    """
    Documents the actual error propagation behaviour of PostgresDataStore.

    Structure of every method (e.g. insert_thread):

        async with self._session_factory() as session:   # <-- OUTSIDE try/except
            try:
                ...                                      # <-- errors here ARE caught
            except Exception as e:
                await session.rollback()
                logger.exception("…failed")
                return None

    Consequence:
    - Pool exhaustion (SATimeoutError) raised during session acquisition
      (inside `async with __aenter__`) is OUTSIDE the try block → it
      PROPAGATES to the caller.  Callers must handle it.
    - Errors raised during query execution (inside try) are caught, logged,
      and swallowed → callers receive None with no indication of the cause.
    """

    @pytest.mark.asyncio
    async def test_pool_exhaustion_during_acquisition_propagates(self, tiny_store):
        """
        Pool exhaustion (SATimeoutError) during session __aenter__ propagates.
        The session context manager is OUTSIDE the try/except, so errors there
        are not caught.  Callers WILL see an exception.
        """
        from sqlalchemy.exc import TimeoutError as SATimeoutError

        # Simulate pool exhaustion at session acquisition time
        class _ExhaustedCtx:
            async def __aenter__(self):
                raise SATimeoutError("QueuePool limit of size 2 overflow 0 reached")
            async def __aexit__(self, *args):
                pass

        with patch.object(tiny_store, "_session_factory", return_value=_ExhaustedCtx()):
            with pytest.raises(SATimeoutError):
                await tiny_store.insert_thread(_make_thread_request())

    @pytest.mark.asyncio
    async def test_mid_query_error_is_swallowed_and_returns_none(self, tiny_store):
        """
        Errors raised inside the try block (mid-query) are caught and swallowed.
        The caller receives None with no indication of what failed — this is the
        behaviour that should ideally be improved.
        """
        from sqlalchemy.exc import OperationalError

        # Create a thread first so we have a known-good DB
        thread = await tiny_store.insert_thread(CreateThreadRequest(messages=[]))
        assert thread is not None

        # Patch execute() to raise mid-query
        original_factory = tiny_store._session_factory

        class _BrokenSession:
            async def execute(self, *args, **kwargs):
                raise OperationalError("statement", {}, Exception("disk full"))
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass

        class _BrokenFactory:
            def __call__(self_inner):
                return _BrokenSession()

        tiny_store._session_factory = _BrokenFactory()
        try:
            result = await tiny_store.get_thread_by_id(thread.id)
        finally:
            tiny_store._session_factory = original_factory

        assert result is None, (
            "Expected None: errors inside the try block are swallowed.\n"
            "This is a problematic API contract — callers can't distinguish "
            "'not found' from 'infrastructure failure'."
        )

    @pytest.mark.asyncio
    async def test_mid_query_error_is_logged(self, tiny_store):
        """Errors inside the try block should at least be logged via logger.exception()."""
        from sqlalchemy.exc import OperationalError

        thread = await tiny_store.insert_thread(CreateThreadRequest(messages=[]))
        assert thread is not None

        original_factory = tiny_store._session_factory

        class _BrokenSession:
            async def execute(self, *args, **kwargs):
                raise OperationalError("statement", {}, Exception("disk full"))
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass

        class _BrokenFactory:
            def __call__(self_inner):
                return _BrokenSession()

        # Patch the module-level logger directly to avoid llamphouse's propagate=False
        with patch("llamphouse.core.data_stores.postgres_store.logger") as mock_logger:
            tiny_store._session_factory = _BrokenFactory()
            try:
                await tiny_store.get_thread_by_id(thread.id)
            finally:
                tiny_store._session_factory = original_factory

        mock_logger.exception.assert_called_once()
        log_message = mock_logger.exception.call_args[0][0]
        assert "failed" in log_message.lower(), (
            f"Expected 'failed' in log message, got: {log_message!r}"
        )


# ---------------------------------------------------------------------------
# 5. Post-fix regression tests — run on both backends
# ---------------------------------------------------------------------------

class TestPostFixRegression:
    """
    Define the DESIRED behaviour after mitigation.
    These run against SQLite in CI and Postgres locally/staging.
    """

    @pytest.mark.asyncio
    async def test_insert_thread_with_messages_uses_single_session(self, tiny_store):
        """
        After fix: insert_thread + N messages should acquire exactly ONE session.
        EXPECTED TO FAIL with current code on both backends.
        """
        call_count = 0
        original_factory = tiny_store._session_factory

        class _CountingFactory:
            def __call__(self_inner):
                nonlocal call_count
                call_count += 1
                return original_factory()

        tiny_store._session_factory = _CountingFactory()
        try:
            await tiny_store.insert_thread(_make_thread_request(n_messages=3))
        finally:
            tiny_store._session_factory = original_factory

        assert call_count == 1, (
            f"Expected 1 session acquisition, got {call_count}.\n"
            f"insert_message() is still opening its own session inside insert_thread()."
        )

    @pytest.mark.asyncio
    async def test_insert_run_with_additional_messages_uses_single_session(self, tiny_store):
        """
        After fix: insert_run + additional_messages should acquire exactly ONE session.
        EXPECTED TO FAIL with current code on both backends.
        """
        thread = await tiny_store.insert_thread(CreateThreadRequest(messages=[]))
        assert thread is not None

        call_count = 0
        original_factory = tiny_store._session_factory

        class _CountingFactory:
            def __call__(self_inner):
                nonlocal call_count
                call_count += 1
                return original_factory()

        tiny_store._session_factory = _CountingFactory()
        try:
            await tiny_store.insert_run(
                thread.id,
                RunCreateRequest(
                    assistant_id="asst_test",
                    additional_messages=[
                        CreateMessageRequest(role="user", content=f"msg {i}")
                        for i in range(3)
                    ],
                ),
                _make_mock_assistant(),
            )
        finally:
            tiny_store._session_factory = original_factory

        assert call_count == 1, (
            f"Expected 1 session for insert_run with 3 additional_messages, got {call_count}.\n"
            f"insert_message() is still opening its own session inside insert_run()."
        )

    @pytest.mark.asyncio
    async def test_event_queue_io_happens_outside_session_scope(self, tiny_store):
        """
        After fix: event_queue.add() should be called AFTER the DB session closes.
        EXPECTED TO FAIL with current code on both backends.
        """
        thread = await tiny_store.insert_thread(CreateThreadRequest(messages=[]))
        assert thread is not None

        session_closed_at_event_time = []
        original_factory = tiny_store._session_factory
        session_is_open = False

        class _TrackingCtx:
            async def __aenter__(self_inner):
                nonlocal session_is_open
                self_inner._inner = original_factory()
                session_is_open = True
                return await self_inner._inner.__aenter__()

            async def __aexit__(self_inner, *args):
                nonlocal session_is_open
                result = await self_inner._inner.__aexit__(*args)
                session_is_open = False
                return result

        class _TrackingFactory:
            def __call__(self_inner):
                return _TrackingCtx()

        mock_queue = AsyncMock()

        async def _check_timing(*args, **kwargs):
            session_closed_at_event_time.append(not session_is_open)

        mock_queue.add.side_effect = _check_timing

        tiny_store._session_factory = _TrackingFactory()
        try:
            await tiny_store.insert_message(
                thread.id,
                CreateMessageRequest(role="user", content="test"),
                event_queue=mock_queue,
            )
        finally:
            tiny_store._session_factory = original_factory

        assert mock_queue.add.called, "event_queue.add() was never called"
        assert all(session_closed_at_event_time), (
            "event_queue.add() was called while the DB session was still open.\n"
            "Connection is being held during external I/O — release session before firing events."
        )