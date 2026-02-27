import asyncio
from typing import List
from om_memory.core import ObservationalMemory

try:
    from llama_index.core.memory import BaseMemory
    from llama_index.core.llms import ChatMessage, MessageRole
except ImportError:
    class BaseMemory: ... # fallback
    class ChatMessage: ...

class OMLlamaIndexMemory(BaseMemory):
    """
    Adapter to use ObservationalMemory as LlamaIndex ChatMemoryBuffer alternative.
    """
    def __init__(self, om: ObservationalMemory, thread_id: str):
        self.om = om
        self.thread_id = thread_id

    @classmethod
    def from_defaults(cls, om: ObservationalMemory, thread_id: str, **kwargs: any):
        return cls(om=om, thread_id=thread_id)

    def get(self, initial_token_count: int = 0, **kwargs: any) -> List:
        """Get chat history."""
        context = self.om.get_context(self.thread_id, format="text")
        # Prepend OM context as a system message
        return [ChatMessage(role=MessageRole.SYSTEM, content=context)]

    def get_all(self) -> List:
        return self.get()

    def put(self, message) -> None:
        """Put message into memory."""
        role_map = {
            MessageRole.USER: "user",
            MessageRole.ASSISTANT: "assistant",
            MessageRole.SYSTEM: "system",
            MessageRole.TOOL: "tool"
        }
        mapped_role = role_map.get(message.role, "user")
        self.om.add_message(self.thread_id, mapped_role, message.content)

    def set(self, messages: List) -> None:
        """Set history (replaces current)."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(self.om.aclear_thread(self.thread_id))
        else:
            loop.run_until_complete(self.om.aclear_thread(self.thread_id))
            
        for msg in messages:
            self.put(msg)

    def reset(self) -> None:
        """Reset memory."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(self.om.aclear_thread(self.thread_id))
        else:
            loop.run_until_complete(self.om.aclear_thread(self.thread_id))
