from typing import Dict, List
from om_memory.models import Message, Observation
from om_memory.storage.base import StorageBackend

class InMemoryStorage(StorageBackend):
    """
    Simple dict-based storage. Good for testing and demos. No persistence.
    """
    def __init__(self):
        self._messages: Dict[str, List[Message]] = {}
        self._observations: Dict[str, List[Observation]] = {}
        self._resource_observations: Dict[str, List[Observation]] = {}
        
    async def asave_messages(self, messages: list[Message]) -> None:
        self.save_messages(messages)
        
    def save_messages(self, messages: list[Message]) -> None:
        for msg in messages:
            if msg.thread_id not in self._messages:
                self._messages[msg.thread_id] = []
            self._messages[msg.thread_id].append(msg)
            
    async def aget_messages(self, thread_id: str, limit: int = None) -> list[Message]:
        return self.get_messages(thread_id, limit)
        
    def get_messages(self, thread_id: str, limit: int = None) -> list[Message]:
        msgs = self._messages.get(thread_id, [])
        msgs = sorted(msgs, key=lambda m: m.timestamp)
        return msgs[-limit:] if limit else msgs
        
    async def adelete_messages(self, message_ids: list[str]) -> None:
        self.delete_messages(message_ids)
        
    def delete_messages(self, message_ids: list[str]) -> None:
        ids = set(message_ids)
        for t_id, msgs in self._messages.items():
            self._messages[t_id] = [m for m in msgs if m.id not in ids]
            
    async def asave_observations(self, observations: list[Observation]) -> None:
        self.save_observations(observations)
        
    def save_observations(self, observations: list[Observation]) -> None:
        for obs in observations:
            if obs.thread_id not in self._observations:
                self._observations[obs.thread_id] = []
            # Upsert: replace existing observation with same ID, or append
            obs_list = self._observations[obs.thread_id]
            replaced = False
            for i, existing in enumerate(obs_list):
                if existing.id == obs.id:
                    obs_list[i] = obs
                    replaced = True
                    break
            if not replaced:
                obs_list.append(obs)
            
    async def aget_observations(self, thread_id: str) -> list[Observation]:
        return self.get_observations(thread_id)
        
    def get_observations(self, thread_id: str) -> list[Observation]:
        return sorted(self._observations.get(thread_id, []), key=lambda o: o.observation_date)
        
    async def aupdate_observations(self, observations: list[Observation]) -> None:
        self.update_observations(observations)
        
    def update_observations(self, observations: list[Observation]) -> None:
        for obs in observations:
            obs_list = self._observations.get(obs.thread_id, [])
            for i, existing in enumerate(obs_list):
                if existing.id == obs.id:
                    obs_list[i] = obs
                    break
                    
    async def adelete_observations(self, observation_ids: list[str]) -> None:
        self.delete_observations(observation_ids)
        
    def delete_observations(self, observation_ids: list[str]) -> None:
        ids = set(observation_ids)
        for t_id, obs_list in self._observations.items():
            self._observations[t_id] = [o for o in obs_list if o.id not in ids]
            
    async def areplace_observations(self, thread_id: str, observations: list[Observation]) -> None:
        self.replace_observations(thread_id, observations)
        
    def replace_observations(self, thread_id: str, observations: list[Observation]) -> None:
        self._observations[thread_id] = observations

    # Resource-scoped operations
    async def aget_resource_observations(self, resource_id: str) -> list[Observation]:
        return self.get_resource_observations(resource_id)
    
    def get_resource_observations(self, resource_id: str) -> list[Observation]:
        return sorted(
            self._resource_observations.get(resource_id, []),
            key=lambda o: o.observation_date,
        )
    
    async def asave_resource_observations(self, observations: list[Observation]) -> None:
        self.save_resource_observations(observations)
    
    def save_resource_observations(self, observations: list[Observation]) -> None:
        for obs in observations:
            rid = obs.resource_id
            if rid:
                if rid not in self._resource_observations:
                    self._resource_observations[rid] = []
                self._resource_observations[rid].append(obs)

    async def ainitialize(self) -> None:
        self.initialize()
        
    def initialize(self) -> None:
        pass
        
    async def aclose(self) -> None:
        self.close()
        
    def close(self) -> None:
        pass
