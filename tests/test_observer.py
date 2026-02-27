import pytest
from unittest.mock import AsyncMock

from om_memory.observer import Observer
from om_memory.models import Message, OMConfig
from om_memory.providers.base import LLMProvider
from om_memory.token_counter import TokenCounter

class MockProvider(LLMProvider):
    def __init__(self, response: str):
        self.response = response
        
    @property
    def model_name(self): return "mock"
    
    async def acomplete(self, sys, usr):
        return self.response
        
    def complete(self, sys, usr):
        return self.response

@pytest.mark.asyncio
async def test_observer_parsing():
    mock_llm_response = """
Date: 2026-03-01
- 🔴 10:00 Decided on SQLite (referenced: 2026-03-01, meaning "today")
- 🟡 10:30 Prefers dark mode
CURRENT_TASK: Setting up DB
SUGGESTED_NEXT: Write tests
"""
    provider = MockProvider(mock_llm_response)
    config = OMConfig()
    counter = TokenCounter()
    
    observer = Observer(provider, config, counter)
    
    msg = Message(thread_id="1", role="user", content="Let's use sqlite and dark mode.")
    obs = await observer.aobserve("1", [msg])
    
    assert len(obs) == 4 # Task, Next, SQLite, Dark mode
    assert obs[2].priority.value == "🔴"
    assert "SQLite" in obs[2].content
    assert obs[0].content == "CURRENT TASK: Setting up DB"
