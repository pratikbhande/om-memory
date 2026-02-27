import pytest
from om_memory.token_counter import TokenCounter
from om_memory.models import Message, Observation, Priority

def test_token_counter_basic():
    counter = TokenCounter()
    text = "Hello world, this is a test."
    count = counter.count(text)
    assert count > 0

def test_token_counter_messages():
    counter = TokenCounter()
    messages = [
        Message(thread_id="1", role="user", content="Hello"),
        Message(thread_id="1", role="assistant", content="Hi")
    ]
    count = counter.count_messages(messages)
    assert count > 0
    assert messages[0].token_count is not None

def test_token_counter_observations():
    counter = TokenCounter()
    observations = [
        Observation(thread_id="1", priority=Priority.CRITICAL, content="Important thing")
    ]
    count = counter.count_observations(observations)
    assert count > 0
    assert observations[0].token_count is not None
