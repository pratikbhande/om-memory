import pytest
from datetime import datetime, timezone
from om_memory.context_builder import ContextBuilder
from om_memory.models import Observation, Message, Priority
from om_memory.token_counter import TokenCounter

def test_context_builder_text():
    counter = TokenCounter()
    builder = ContextBuilder(counter)
    
    observations = [
        Observation(thread_id="1", priority=Priority.CRITICAL, content="CURRENT TASK: Testing Context")
    ]
    messages = [
        Message(thread_id="1", role="user", content="Hi")
    ]
    
    ctx = builder.build_context("1", observations, messages, format="text")
    
    assert "[Memory" in ctx
    assert "CURRENT TASK: Testing Context" in ctx
    assert "user: Hi" in ctx

def test_context_builder_dict():
    counter = TokenCounter()
    builder = ContextBuilder(counter)
    
    observations = [
        Observation(thread_id="1", priority=Priority.CRITICAL, content="CURRENT TASK: Testing Context")
    ]
    messages = [
        Message(thread_id="1", role="user", content="Hi")
    ]
    
    ctx = builder.build_context("1", observations, messages, format="dict")
    assert isinstance(ctx, dict)
    assert ctx["current_task"] == "Testing Context"
    assert len(ctx["messages"]) == 1
    assert ctx["messages"][0]["content"] == "Hi"
