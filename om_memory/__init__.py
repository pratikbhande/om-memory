"""
om-memory: Human-like memory for AI agents. 10x cheaper than RAG.

Usage:
    from om_memory import ObservationalMemory
    
    om = ObservationalMemory()
    
    context = om.get_context("thread_1")
    # Use context in your LLM call...
    
    om.add_message("thread_1", "user", "Hello!")
    om.add_message("thread_1", "assistant", "Hi there!")
"""

from om_memory.core import ObservationalMemory
from om_memory.models import (
    OMConfig, 
    OMStats, 
    Message, 
    Observation, 
    Priority,
    ObservationLog,
)
from om_memory.observability.callbacks import EventType, OMEvent, CallbackManager
from om_memory.providers.base import LLMProvider
from om_memory.storage.base import StorageBackend
from om_memory.parsing import parse_observations

# Version
__version__ = "0.2.4"

__all__ = [
    "ObservationalMemory",
    "OMConfig",
    "OMStats", 
    "Message",
    "Observation",
    "Priority",
    "ObservationLog",
    "EventType",
    "OMEvent",
    "CallbackManager",
    "LLMProvider",
    "StorageBackend",
    "parse_observations",
]
