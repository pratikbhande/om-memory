from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """
    Abstract interface for LLM providers.
    Used by Observer and Reflector to call the LLM for compression.
    
    Users can implement this to use ANY model.
    """
    
    @abstractmethod
    async def acomplete(self, system_prompt: str, user_prompt: str) -> str:
        """Async completion call."""
        pass
    
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Sync completion call."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""
        pass
