from abc import ABC, abstractmethod
from om_memory.models import Message, Observation, OMStats

class StorageBackend(ABC):
    """
    Abstract interface for storing observations and messages.
    """
    
    # Message operations
    @abstractmethod
    async def asave_messages(self, messages: list[Message]) -> None: ...
    @abstractmethod
    def save_messages(self, messages: list[Message]) -> None: ...
    
    @abstractmethod
    async def aget_messages(self, thread_id: str, limit: int = None) -> list[Message]: ...
    @abstractmethod
    def get_messages(self, thread_id: str, limit: int = None) -> list[Message]: ...
    
    @abstractmethod
    async def adelete_messages(self, message_ids: list[str]) -> None: ...
    @abstractmethod
    def delete_messages(self, message_ids: list[str]) -> None: ...
    
    # Observation operations
    @abstractmethod
    async def asave_observations(self, observations: list[Observation]) -> None: ...
    @abstractmethod
    def save_observations(self, observations: list[Observation]) -> None: ...
    
    @abstractmethod
    async def aget_observations(self, thread_id: str) -> list[Observation]: ...
    @abstractmethod
    def get_observations(self, thread_id: str) -> list[Observation]: ...
    
    @abstractmethod
    async def aupdate_observations(self, observations: list[Observation]) -> None: ...
    @abstractmethod
    def update_observations(self, observations: list[Observation]) -> None: ...
    
    @abstractmethod
    async def adelete_observations(self, observation_ids: list[str]) -> None: ...
    @abstractmethod
    def delete_observations(self, observation_ids: list[str]) -> None: ...
    
    @abstractmethod
    async def areplace_observations(self, thread_id: str, observations: list[Observation]) -> None: ...
    @abstractmethod
    def replace_observations(self, thread_id: str, observations: list[Observation]) -> None: ...
    
    # Resource-scoped observation operations
    async def aget_resource_observations(self, resource_id: str) -> list[Observation]:
        """Get observations shared across all threads for a given resource."""
        return []
    
    def get_resource_observations(self, resource_id: str) -> list[Observation]:
        """Get observations shared across all threads for a given resource (sync)."""
        return []
    
    async def asave_resource_observations(self, observations: list[Observation]) -> None:
        """Save resource-scoped observations. Falls back to regular save if not overridden."""
        await self.asave_observations(observations)
    
    def save_resource_observations(self, observations: list[Observation]) -> None:
        """Save resource-scoped observations (sync). Falls back to regular save if not overridden."""
        self.save_observations(observations)
    
    # Lifecycle
    @abstractmethod
    async def ainitialize(self) -> None: ...
    @abstractmethod
    def initialize(self) -> None: ...
    
    @abstractmethod
    async def aclose(self) -> None: ...
    @abstractmethod
    def close(self) -> None: ...
