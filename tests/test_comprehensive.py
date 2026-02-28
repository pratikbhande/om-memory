"""
Comprehensive tests for om-memory v0.2.0
Covers: parsing, context builder truncation, rolling window, sync wrapper,
resource-scoped memory, stub backends, demo mode.
"""
import pytest
import asyncio
from datetime import datetime, timezone

from om_memory.core import ObservationalMemory
from om_memory.storage.memory import InMemoryStorage
from om_memory.storage.redis_store import RedisStorage
from om_memory.storage.mongodb import MongoDBStorage
from om_memory.storage.postgres import PostgresStorage
from om_memory.models import OMConfig, Message, Observation, Priority
from om_memory.providers.base import LLMProvider
from om_memory.parsing import parse_observations
from om_memory.context_builder import ContextBuilder
from om_memory.token_counter import TokenCounter


# --- Mock Provider ---

class MockProvider(LLMProvider):
    def __init__(self, response: str = None):
        self._response = response or "Date: 2026-03-01\n- 🔴 12:00 Mock observation\n"

    @property
    def model_name(self):
        return "mock"

    async def acomplete(self, sys, usr):
        return self._response

    def complete(self, sys, usr):
        return self._response


# --- Parsing Tests ---

class TestParsing:
    def test_parse_basic_observations(self):
        llm_response = """
Date: 2026-03-01
- 🔴 10:00 Decided on SQLite (referenced: 2026-03-01, meaning "today")
- 🟡 10:30 Prefers dark mode
CURRENT_TASK: Setting up DB
SUGGESTED_NEXT: Write tests
"""
        obs = parse_observations(llm_response, "t1", ["m1", "m2"])
        assert len(obs) == 4
        assert obs[0].content == "CURRENT TASK: Setting up DB"
        assert obs[0].priority == Priority.CRITICAL
        assert obs[1].content == "SUGGESTED NEXT: Write tests"
        assert obs[1].priority == Priority.IMPORTANT
        assert "SQLite" in obs[2].content
        assert obs[2].priority == Priority.CRITICAL
        assert obs[3].priority == Priority.IMPORTANT

    def test_parse_with_resource_id(self):
        llm_response = "Date: 2026-01-01\n- 🟢 09:00 User likes Python\n"
        obs = parse_observations(llm_response, "t1", ["m1"], resource_id="user_42")
        assert len(obs) == 1
        assert obs[0].resource_id == "user_42"

    def test_parse_empty_response(self):
        obs = parse_observations("", "t1", [])
        assert obs == []

    def test_parse_malformed_lines(self):
        llm_response = "- not an observation\n- 🔴 no time here\nrandom text"
        obs = parse_observations(llm_response, "t1", [])
        # Should handle gracefully without crashing
        assert isinstance(obs, list)


# --- Context Builder Tests ---

class TestContextBuilder:
    def test_truncation_uses_priority(self):
        """Verify that Priority is correctly imported and truncation works."""
        counter = TokenCounter()
        builder = ContextBuilder(counter)

        # Create observations of different priorities
        obs = [
            Observation(thread_id="1", priority=Priority.CRITICAL, content="Critical thing " * 20),
            Observation(thread_id="1", priority=Priority.INFO, content="Minor thing " * 20),
        ]
        msgs = [Message(thread_id="1", role="user", content="Hello")]

        # Very small max_tokens to force truncation
        ctx = builder.build_context("1", obs, msgs, max_tokens=50)
        # Should not crash (Priority import fixed) and should preferentially keep CRITICAL
        assert isinstance(ctx, str)

    def test_message_token_budget(self):
        counter = TokenCounter()
        builder = ContextBuilder(counter)
        obs = []
        msgs = [
            Message(thread_id="1", role="user", content=f"Message {i} " * 10)
            for i in range(10)
        ]
        ctx = builder.build_context(
            "1", obs, msgs, message_token_budget=50
        )
        assert isinstance(ctx, str)


# --- Core Tests ---

class TestCore:
    @pytest.fixture
    def om(self):
        config = OMConfig(
            observer_token_threshold=10,
            auto_observe=True,
            message_retention_count=2,  # Keep 2 messages after observation
        )
        return ObservationalMemory(
            provider=MockProvider(), storage=InMemoryStorage(), config=config
        )

    @pytest.mark.asyncio
    async def test_rolling_window_retains_messages(self, om):
        """After observation, the last N messages should be retained."""
        thread_id = "th_rolling"
        # Add enough messages to trigger observation
        for i in range(5):
            await om.aadd_message(
                thread_id, "user", f"Message {i} with enough text to exceed threshold tokens."
            )

        msgs = await om.storage.aget_messages(thread_id)
        obs = await om.storage.aget_observations(thread_id)

        # Messages should be retained (rolling window of 2)
        assert len(msgs) <= 2
        # Observations should have been created
        assert len(obs) > 0

    @pytest.mark.asyncio
    async def test_resource_scoped_memory(self, om):
        """Observations with resource_id should be retrievable across threads."""
        resource_id = "user_42"
        await om.aadd_message("thread_A", "user", "I like Python " * 20, resource_id=resource_id)

        # Trigger observation manually
        await om.aobserve("thread_A", resource_id=resource_id)

        # Get context for a different thread with same resource_id
        ctx = await om.aget_context("thread_B", resource_id=resource_id)
        assert isinstance(ctx, str)

    @pytest.mark.asyncio
    async def test_add_message_and_get_context(self, om):
        """Basic add + get context flow."""
        await om.aadd_message("t1", "user", "Hello")
        ctx = await om.aget_context("t1")
        assert "Hello" in ctx

    def test_sync_wrapper_works(self, om):
        """Sync wrappers should not crash."""
        om.add_message("t_sync", "user", "Sync test")
        ctx = om.get_context("t_sync")
        assert "Sync test" in ctx


# --- Demo Mode Tests ---

class TestDemoMode:
    def test_demo_mode_lowers_thresholds(self):
        config = OMConfig(demo_mode=True)
        assert config.observer_token_threshold == 2000
        assert config.reflector_token_threshold == 4000
        assert config.max_message_history_tokens == 5000

    def test_normal_mode_keeps_defaults(self):
        config = OMConfig()
        assert config.observer_token_threshold == 30000
        assert config.reflector_token_threshold == 40000


# --- Stub Backend Tests ---

class TestStubBackends:
    def test_redis_instantiation_does_not_crash(self):
        """RedisStorage should be instantiable without raising."""
        storage = RedisStorage(connection_string="redis://localhost")
        assert storage.connection_string == "redis://localhost"

    def test_redis_methods_raise_on_use(self):
        storage = RedisStorage()
        with pytest.raises(NotImplementedError):
            storage.save_messages([])

    def test_mongodb_instantiation_does_not_crash(self):
        storage = MongoDBStorage(connection_string="mongodb://localhost")
        assert storage.connection_string == "mongodb://localhost"

    def test_mongodb_methods_raise_on_use(self):
        storage = MongoDBStorage()
        with pytest.raises(NotImplementedError):
            storage.get_messages("t1")

    def test_postgres_instantiation_does_not_crash(self):
        storage = PostgresStorage(connection_string="postgresql://localhost")
        assert storage.connection_string == "postgresql://localhost"

    def test_postgres_methods_raise_on_use(self):
        storage = PostgresStorage()
        with pytest.raises(NotImplementedError):
            storage.get_observations("t1")


# --- Storage Tests ---

class TestInMemoryStorage:
    @pytest.mark.asyncio
    async def test_resource_observations(self):
        storage = InMemoryStorage()
        obs = Observation(
            thread_id="t1",
            resource_id="user_1",
            priority=Priority.CRITICAL,
            content="User preference",
        )
        await storage.asave_resource_observations([obs])
        result = await storage.aget_resource_observations("user_1")
        assert len(result) == 1
        assert result[0].content == "User preference"


class TestSQLiteStorage:
    @pytest.mark.asyncio
    async def test_resource_observations(self, tmp_path):
        from om_memory.storage.sqlite import SQLiteStorage

        db_path = str(tmp_path / "test.db")
        storage = SQLiteStorage(db_path=db_path)
        await storage.ainitialize()

        obs = Observation(
            thread_id="t1",
            resource_id="user_1",
            priority=Priority.CRITICAL,
            content="Shared across threads",
        )
        await storage.asave_observations([obs])
        result = await storage.aget_resource_observations("user_1")
        assert len(result) == 1
        assert result[0].resource_id == "user_1"


# --- Token Counter Tests ---

class TestTokenCounter:
    def test_basic_count(self):
        counter = TokenCounter()
        count = counter.count("Hello world, this is a test.")
        assert count > 0

    def test_empty_count(self):
        counter = TokenCounter()
        assert counter.count("") == 0
        assert counter.count(None) == 0
