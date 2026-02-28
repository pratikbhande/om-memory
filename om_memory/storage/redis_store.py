from om_memory.models import Message, Observation
from om_memory.storage.base import StorageBackend


class RedisStorage(StorageBackend):
    """
    Redis storage backend using redis-py.
    
    Status: Not yet implemented. Instantiation is allowed for configuration
    purposes, but calling any storage method will raise NotImplementedError.
    
    For a working backend, use SQLiteStorage or InMemoryStorage.
    """
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string
        # Note: Does NOT raise on instantiation. Raises on actual usage.

    def _not_implemented(self):
        raise NotImplementedError(
            "RedisStorage is not yet implemented. "
            "Use SQLiteStorage or InMemoryStorage instead."
        )

    async def asave_messages(self, messages: list[Message]) -> None: self._not_implemented()
    def save_messages(self, messages: list[Message]) -> None: self._not_implemented()
    async def aget_messages(self, thread_id: str, limit: int = None) -> list[Message]: self._not_implemented()
    def get_messages(self, thread_id: str, limit: int = None) -> list[Message]: self._not_implemented()
    async def adelete_messages(self, message_ids: list[str]) -> None: self._not_implemented()
    def delete_messages(self, message_ids: list[str]) -> None: self._not_implemented()
    async def asave_observations(self, observations: list[Observation]) -> None: self._not_implemented()
    def save_observations(self, observations: list[Observation]) -> None: self._not_implemented()
    async def aget_observations(self, thread_id: str) -> list[Observation]: self._not_implemented()
    def get_observations(self, thread_id: str) -> list[Observation]: self._not_implemented()
    async def aupdate_observations(self, observations: list[Observation]) -> None: self._not_implemented()
    def update_observations(self, observations: list[Observation]) -> None: self._not_implemented()
    async def adelete_observations(self, observation_ids: list[str]) -> None: self._not_implemented()
    def delete_observations(self, observation_ids: list[str]) -> None: self._not_implemented()
    async def areplace_observations(self, thread_id: str, observations: list[Observation]) -> None: self._not_implemented()
    def replace_observations(self, thread_id: str, observations: list[Observation]) -> None: self._not_implemented()
    async def ainitialize(self) -> None: pass
    def initialize(self) -> None: pass
    async def aclose(self) -> None: pass
    def close(self) -> None: pass
