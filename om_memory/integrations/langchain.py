import asyncio
from typing import Any, Dict, List
from om_memory.core import ObservationalMemory

try:
    from langchain_core.memory import BaseMemory
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from pydantic import Field
except ImportError:
    class BaseMemory: ... # fallback
    class Field:
        def __init__(self, *args, **kwargs): pass

class OMLangChainMemory(BaseMemory):
    """
    Adapter to use ObservationalMemory as a LangChain BaseMemory component.
    """
    
    om: ObservationalMemory
    thread_id: str
    memory_key: str = "history"
    return_messages: bool = False
    
    def __init__(self, om: ObservationalMemory, thread_id: str, **kwargs):
        super().__init__(om=om, thread_id=thread_id, **kwargs)
        
    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return memory variables."""
        context = self.om.get_context(self.thread_id, format="text")
        
        if self.return_messages:
            return {self.memory_key: [SystemMessage(content=context)]}
            
        return {self.memory_key: context}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context to memory."""
        # Typically inputs["input"] and outputs["output"]
        user_msg = inputs.get("input") or list(inputs.values())[0]
        ai_msg = outputs.get("output") or list(outputs.values())[0]
        
        self.om.add_message(self.thread_id, "user", str(user_msg))
        self.om.add_message(self.thread_id, "assistant", str(ai_msg))

    def clear(self) -> None:
        """Clear memory."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(self.om.aclear_thread(self.thread_id))
        else:
            loop.run_until_complete(self.om.aclear_thread(self.thread_id))
