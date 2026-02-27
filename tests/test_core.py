import pytest
import asyncio
from om_memory.core import ObservationalMemory
from om_memory.storage.memory import InMemoryStorage
from om_memory.models import OMConfig
from om_memory.providers.base import LLMProvider

class MockProvider(LLMProvider):
    @property
    def model_name(self): return "mock"
    async def acomplete(self, sys, usr):
        return "Date: 2026-03-01\n- 🔴 12:00 Mock observation\n"
    def complete(self, sys, usr):
        return "Date: 2026-03-01\n- 🔴 12:00 Mock observation\n"

@pytest.fixture
def memory_om():
    config = OMConfig(
        observer_token_threshold=10, # Very low to trigger it easily
        auto_observe=True
    )
    provider = MockProvider()
    storage = InMemoryStorage()
    return ObservationalMemory(provider=provider, storage=storage, config=config)

@pytest.mark.asyncio
async def test_om_add_message_triggers_observer(memory_om):
    thread_id = "th_1"
    
    # Needs a bit of text to exceed 10 tokens
    await memory_om.aadd_message(thread_id, "user", "This is a slightly longer message to exceed ten tokens limit.")
    
    # Should automatically observe and compress since we hit the threshold
    # Messages should be deleted, observations should exist
    msgs = await memory_om.storage.aget_messages(thread_id)
    assert len(msgs) == 0  # Block 2 compressed, should be 0 or small
    
    obs = await memory_om.storage.aget_observations(thread_id)
    assert len(obs) > 0 # Should have the mock observation
