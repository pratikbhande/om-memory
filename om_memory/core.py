import asyncio
from typing import Callable, Union

from om_memory.models import OMConfig, OMStats, Message, Observation
from om_memory.providers.base import LLMProvider
from om_memory.providers.openai_provider import OpenAIProvider
from om_memory.storage.base import StorageBackend
from om_memory.storage.sqlite import SQLiteStorage
from om_memory.observability.callbacks import CallbackManager, EventType, OMEvent
from om_memory.observability.metrics import MetricsTracker
from om_memory.token_counter import TokenCounter
from om_memory.observer import Observer
from om_memory.reflector import Reflector
from om_memory.context_builder import ContextBuilder
from om_memory.config import default_config


class ObservationalMemory:
    """
    The main entry point for Observational Memory.
    """
    
    def __init__(
        self,
        provider: LLMProvider = None,
        storage: StorageBackend = None,
        config: OMConfig = None,
        api_key: str = None,
        model: str = None,
        database_url: str = None,
    ):
        self.config = config or default_config
        
        # Shortcuts overrides
        if model:
            self.config.observer_model = model
            self.config.reflector_model = model
            
        self.provider = provider
        if not self.provider:
            self.provider = OpenAIProvider(
                model=self.config.observer_model or "gpt-4o-mini",
                api_key=api_key
            )
            
        self.storage = storage
        if not self.storage:
            self.storage = SQLiteStorage(db_path=database_url)
            
        self.token_counter = TokenCounter(model=self.provider.model_name)
        self.callbacks = CallbackManager()
        self.metrics = MetricsTracker(self.config)
        
        self.observer = Observer(self.provider, self.config, self.token_counter)
        self.reflector = Reflector(self.provider, self.config, self.token_counter)
        self.context_builder = ContextBuilder(self.token_counter)
        
        # Register metrics hooks
        self._register_metrics_hooks()
        
        # Thread locks for safety
        self._locks = {}

    def _get_lock(self, thread_id: str) -> asyncio.Lock:
        if thread_id not in self._locks:
            self._locks[thread_id] = asyncio.Lock()
        return self._locks[thread_id]

    def _register_metrics_hooks(self):
        self.callbacks.on(EventType.CONTEXT_BUILT, lambda e: self.metrics.record_context_build(
            e.thread_id, e.data["observation_tokens"], e.data["message_tokens"], e.data["total_tokens"]
        ))
        self.callbacks.on(EventType.OBSERVER_COMPLETED, lambda e: self.metrics.record_observer_run(
            e.thread_id, e.data["input_tokens"], e.data["output_tokens"], e.data["messages_compressed"], e.data["observations_created"]
        ))
        self.callbacks.on(EventType.REFLECTOR_COMPLETED, lambda e: self.metrics.record_reflector_run(
            e.thread_id, e.data["input_tokens"], e.data["output_tokens"], e.data["observations_before"], e.data["observations_after"]
        ))

    async def ainitialize(self) -> None:
        await self.storage.ainitialize()

    def initialize(self) -> None:
        self.storage.initialize()

    async def aclose(self) -> None:
        await self.storage.aclose()
        
    def close(self) -> None:
        self.storage.close()

    async def __aenter__(self):
        await self.ainitialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    # --- PRIMARY API ---
    
    async def aget_context(
        self,
        thread_id: str,
        max_tokens: int = None,
        format: str = "text",
        include_header: bool = True
    ) -> Union[str, dict]:
        # Always init storage if not done explicitly
        await self.storage.ainitialize()
        
        lock = self._get_lock(thread_id)
        async with lock:
            observations = await self.storage.aget_observations(thread_id)
            messages = await self.storage.aget_messages(thread_id)
            
        return self.context_builder.build_context(
            thread_id=thread_id,
            observations=observations,
            messages=messages,
            max_tokens=max_tokens,
            include_header=include_header,
            format=format,
            callbacks=self.callbacks
        )

    def get_context(self, thread_id: str, **kwargs) -> Union[str, dict]:
        # Sync wrapper
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self.aget_context(thread_id, **kwargs))
        except RuntimeError:
            return asyncio.run(self.aget_context(thread_id, **kwargs))

    async def aadd_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        metadata: dict = None
    ) -> None:
        await self.storage.ainitialize()
        
        msg = Message(
            thread_id=thread_id,
            role=role,
            content=content,
            metadata=metadata or {}
        )
        # Pre-count so it's cached
        msg.token_count = self.token_counter.count(f"{msg.role}: {msg.content}")
        
        lock = self._get_lock(thread_id)
        async with lock:
            await self.storage.asave_messages([msg])
            
            # Check thresholds for compression
            all_msgs = await self.storage.aget_messages(thread_id)
            msg_tokens = self.token_counter.count_messages(all_msgs)
            
            if self.config.auto_observe and msg_tokens >= self.config.observer_token_threshold:
                if self.config.blocking_mode:
                    await self._run_observe(thread_id, all_msgs)
                else:
                    asyncio.create_task(self._run_observe(thread_id, all_msgs))

    def add_message(self, thread_id: str, role: str, content: str, **kwargs) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self.aadd_message(thread_id, role, content, **kwargs))
        except RuntimeError:
            asyncio.run(self.aadd_message(thread_id, role, content, **kwargs))

    # --- MANUAL CONTROLS ---

    async def _run_observe(self, thread_id: str, messages: list[Message]) -> list[Observation]:
        observations = await self.storage.aget_observations(thread_id)
        new_obs = await self.observer.aobserve(thread_id, messages, observations, self.callbacks)
        
        if new_obs:
            await self.storage.asave_observations(new_obs)
            await self.storage.adelete_messages([m.id for m in messages])
            
            # Check if reflection is needed
            all_obs = observations + new_obs
            obs_tokens = self.token_counter.count_observations(all_obs)
            
            if self.config.auto_reflect and obs_tokens >= self.config.reflector_token_threshold:
                await self._run_reflect(thread_id, all_obs)
                
        return new_obs

    async def aobserve(self, thread_id: str) -> list[Observation]:
        lock = self._get_lock(thread_id)
        async with lock:
            all_msgs = await self.storage.aget_messages(thread_id)
            if not all_msgs:
                return []
            return await self._run_observe(thread_id, all_msgs)

    async def _run_reflect(self, thread_id: str, observations: list[Observation]) -> list[Observation]:
        new_obs = await self.reflector.areflect(thread_id, observations, self.callbacks)
        if new_obs:
            await self.storage.areplace_observations(thread_id, new_obs)
        return new_obs

    async def areflect(self, thread_id: str) -> list[Observation]:
        lock = self._get_lock(thread_id)
        async with lock:
            observations = await self.storage.aget_observations(thread_id)
            if not observations:
                return []
            return await self._run_reflect(thread_id, observations)

    # --- INTROSPECTION ---
    
    async def aget_observations(self, thread_id: str) -> list[Observation]:
        return await self.storage.aget_observations(thread_id)
        
    async def aget_stats(self, thread_id: str) -> OMStats:
        return self.metrics.get_thread_stats(thread_id)
        
    async def aget_savings_report(self, thread_id: str) -> dict:
        return self.metrics.get_savings_report(thread_id)

    # --- EVENT SYSTEM ---
    
    def on(self, event_type: EventType, callback: Callable) -> None:
        self.callbacks.on(event_type, callback)

    # --- THREAD MANAGEMENT ---

    async def aclear_thread(self, thread_id: str) -> None:
        lock = self._get_lock(thread_id)
        async with lock:
            msgs = await self.storage.aget_messages(thread_id)
            await self.storage.adelete_messages([m.id for m in msgs])
            
            obs = await self.storage.aget_observations(thread_id)
            await self.storage.adelete_observations([o.id for o in obs])
