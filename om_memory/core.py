import asyncio
import logging
import concurrent.futures
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

logger = logging.getLogger("om_memory")


class ObservationalMemory:
    """
    The main entry point for Observational Memory.
    
    Supports:
    - Thread-scoped memory (per conversation)
    - Resource-scoped memory (shared across a user's threads)
    - Rolling window message retention
    - Async buffering for non-blocking mode
    - Demo mode with lower thresholds
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
        self._locks: dict[str, asyncio.Lock] = {}
        
        # Async buffer for non-blocking mode
        self._async_buffer: dict[str, list] = {}
        self._buffer_tasks: dict[str, asyncio.Task] = {}

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
        # Cancel any pending buffer tasks
        for task in self._buffer_tasks.values():
            if not task.done():
                task.cancel()
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
        resource_id: str = None,
        max_tokens: int = None,
        format: str = "text",
        include_header: bool = True
    ) -> Union[str, dict]:
        """
        Get context for a thread, optionally including resource-scoped observations.
        
        Args:
            thread_id: The conversation thread ID.
            resource_id: Optional. If provided, also includes observations shared
                         across all threads for this resource (e.g., a user ID).
            max_tokens: Maximum tokens for the context window.
            format: Output format — "text", "dict", or "json".
            include_header: Whether to include section headers.
        """
        await self.storage.ainitialize()
        
        lock = self._get_lock(thread_id)
        async with lock:
            observations = await self.storage.aget_observations(thread_id)
            
            # Merge resource-scoped observations if available
            if resource_id:
                resource_obs = await self.storage.aget_resource_observations(resource_id)
                # Deduplicate by ID
                obs_ids = {o.id for o in observations}
                for ro in resource_obs:
                    if ro.id not in obs_ids:
                        observations.append(ro)
                observations.sort(key=lambda x: x.observation_date)
            
            messages = await self.storage.aget_messages(thread_id)
            
        return self.context_builder.build_context(
            thread_id=thread_id,
            observations=observations,
            messages=messages,
            max_tokens=max_tokens,
            include_header=include_header,
            format=format,
            callbacks=self.callbacks,
            message_token_budget=self.config.message_token_budget,
            share_token_budget=self.config.share_token_budget,
        )

    def get_context(self, thread_id: str, **kwargs) -> Union[str, dict]:
        """Sync wrapper — safe to call from both sync and async contexts."""
        return self._run_sync(self.aget_context(thread_id, **kwargs))

    async def aadd_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        resource_id: str = None,
        metadata: dict = None
    ) -> None:
        """
        Add a message and trigger observation if threshold is exceeded.
        
        Args:
            thread_id: The conversation thread ID.
            role: Message role ("user", "assistant", "system", "tool").
            content: Message content.
            resource_id: Optional resource ID for cross-thread memory sharing.
            metadata: Optional metadata dict.
        """
        await self.storage.ainitialize()
        
        msg = Message(
            thread_id=thread_id,
            resource_id=resource_id,
            role=role,
            content=content,
            metadata=metadata or {}
        )
        msg.token_count = self.token_counter.count(f"{msg.role}: {msg.content}")
        
        lock = self._get_lock(thread_id)
        async with lock:
            await self.storage.asave_messages([msg])
            
            # Check thresholds for compression
            all_msgs = await self.storage.aget_messages(thread_id)
            msg_tokens = self.token_counter.count_messages(all_msgs)
            
            if self.config.auto_observe and msg_tokens >= self.config.observer_token_threshold:
                if self.config.blocking_mode:
                    await self._run_observe(thread_id, all_msgs, resource_id=resource_id)
                else:
                    await self._buffered_observe(thread_id, all_msgs, resource_id=resource_id)

    def add_message(self, thread_id: str, role: str, content: str, **kwargs) -> None:
        """Sync wrapper — safe to call from both sync and async contexts."""
        self._run_sync(self.aadd_message(thread_id, role, content, **kwargs))

    # --- ASYNC BUFFERING ---

    async def _buffered_observe(self, thread_id: str, messages: list[Message], resource_id: str = None):
        """
        Queue an observation task with proper error handling.
        Unlike bare create_task, this catches and logs failures.
        """
        task = asyncio.create_task(
            self._safe_observe(thread_id, messages, resource_id=resource_id)
        )
        self._buffer_tasks[thread_id] = task

    async def _safe_observe(self, thread_id: str, messages: list[Message], resource_id: str = None):
        """Wrapper that catches and logs observation errors instead of losing them silently."""
        try:
            await self._run_observe(thread_id, messages, resource_id=resource_id)
        except Exception as e:
            logger.error(f"Async observation failed for thread {thread_id}: {e}")
            self.callbacks.emit(OMEvent(
                type=EventType.OBSERVER_ERROR,
                thread_id=thread_id,
                timestamp=__import__('datetime').datetime.now(__import__('datetime').timezone.utc),
                data={"error": str(e), "source": "async_buffer"}
            ))

    # --- INTERNAL OBSERVATION/REFLECTION ---

    async def _run_observe(self, thread_id: str, messages: list[Message], resource_id: str = None) -> list[Observation]:
        observations = await self.storage.aget_observations(thread_id)
        new_obs = await self.observer.aobserve(
            thread_id, messages, observations, self.callbacks, resource_id=resource_id
        )
        
        if new_obs:
            # Set resource_id on observations BEFORE saving (avoids double-insert)
            if resource_id:
                for obs in new_obs:
                    obs.resource_id = resource_id
            
            await self.storage.asave_observations(new_obs)
            
            # ROLLING WINDOW: Keep the last N messages instead of deleting ALL
            retention = self.config.message_retention_count
            if retention > 0 and len(messages) > retention:
                messages_to_delete = messages[:-retention]
                await self.storage.adelete_messages([m.id for m in messages_to_delete])
            elif retention == 0:
                # Delete all (old behavior, configurable)
                await self.storage.adelete_messages([m.id for m in messages])
            # else: retention >= len(messages), keep all
            
            # Check if reflection is needed
            all_obs = observations + new_obs
            obs_tokens = self.token_counter.count_observations(all_obs)
            
            if self.config.auto_reflect and obs_tokens >= self.config.reflector_token_threshold:
                await self._run_reflect(thread_id, all_obs, resource_id=resource_id)
                
        return new_obs

    async def aobserve(self, thread_id: str, resource_id: str = None) -> list[Observation]:
        """Manually trigger observation for a thread."""
        lock = self._get_lock(thread_id)
        async with lock:
            all_msgs = await self.storage.aget_messages(thread_id)
            if not all_msgs:
                return []
            return await self._run_observe(thread_id, all_msgs, resource_id=resource_id)

    async def _run_reflect(self, thread_id: str, observations: list[Observation], resource_id: str = None) -> list[Observation]:
        new_obs = await self.reflector.areflect(
            thread_id, observations, self.callbacks, resource_id=resource_id
        )
        if new_obs:
            await self.storage.areplace_observations(thread_id, new_obs)
        return new_obs

    async def areflect(self, thread_id: str, resource_id: str = None) -> list[Observation]:
        """Manually trigger reflection for a thread."""
        lock = self._get_lock(thread_id)
        async with lock:
            observations = await self.storage.aget_observations(thread_id)
            if not observations:
                return []
            return await self._run_reflect(thread_id, observations, resource_id=resource_id)

    # --- INTROSPECTION ---
    
    async def aget_observations(self, thread_id: str) -> list[Observation]:
        return await self.storage.aget_observations(thread_id)
    
    async def aget_resource_observations(self, resource_id: str) -> list[Observation]:
        """Get observations shared across all threads for a resource."""
        return await self.storage.aget_resource_observations(resource_id)
        
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

    # --- SYNC HELPER ---

    @staticmethod
    def _run_sync(coro):
        """
        Safely run an async coroutine from sync code.
        Works whether or not an event loop is already running.
        """
        try:
            asyncio.get_running_loop()
            # We're inside a running loop — run in a new thread to avoid deadlock
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            # No running loop — safe to use asyncio.run directly
            return asyncio.run(coro)
