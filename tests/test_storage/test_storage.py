import pytest
import asyncio
from om_memory.storage.memory import InMemoryStorage
from om_memory.storage.sqlite import SQLiteStorage
from om_memory.models import Message

@pytest.fixture
def memory_storage():
    return InMemoryStorage()

@pytest.mark.asyncio
async def test_memory_storage_save_get(memory_storage):
    msg = Message(thread_id="thread_1", role="user", content="Hello")
    await memory_storage.asave_messages([msg])
    
    msgs = await memory_storage.aget_messages("thread_1")
    assert len(msgs) == 1
    assert msgs[0].content == "Hello"

@pytest.mark.asyncio
async def test_memory_storage_delete(memory_storage):
    msg1 = Message(thread_id="thread_1", role="user", content="Hello 1")
    msg2 = Message(thread_id="thread_1", role="user", content="Hello 2")
    await memory_storage.asave_messages([msg1, msg2])
    
    await memory_storage.adelete_messages([msg1.id])
    
    msgs = await memory_storage.aget_messages("thread_1")
    assert len(msgs) == 1
    assert msgs[0].content == "Hello 2"

@pytest.mark.asyncio
async def test_sqlite_storage(tmp_path):
    db_path = str(tmp_path / "test.db")
    storage = SQLiteStorage(db_path=db_path)
    await storage.ainitialize()
    
    msg = Message(thread_id="thread_sq", role="user", content="Stored in SQL")
    await storage.asave_messages([msg])
    
    msgs = await storage.aget_messages("thread_sq")
    assert len(msgs) == 1
    assert msgs[0].content == "Stored in SQL"
    
    await storage.adelete_messages([msg.id])
    assert len(await storage.aget_messages("thread_sq")) == 0
