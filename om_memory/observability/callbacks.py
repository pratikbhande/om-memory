from enum import Enum
from typing import Callable, Any, Dict, List
from dataclasses import dataclass
from datetime import datetime

class EventType(Enum):
    # Lifecycle events
    MESSAGE_ADDED = "message_added"
    CONTEXT_BUILT = "context_built"
    
    # Observer events
    OBSERVER_TRIGGERED = "observer_triggered"       # Threshold hit
    OBSERVER_STARTED = "observer_started"           # LLM call starting
    OBSERVER_COMPLETED = "observer_completed"       # Observations created
    OBSERVER_ERROR = "observer_error"
    
    # Reflector events
    REFLECTOR_TRIGGERED = "reflector_triggered"
    REFLECTOR_STARTED = "reflector_started"
    REFLECTOR_COMPLETED = "reflector_completed"
    REFLECTOR_ERROR = "reflector_error"
    
    # Cost events
    TOKENS_USED = "tokens_used"
    CACHE_HIT = "cache_hit"
    COST_CALCULATED = "cost_calculated"
    
    # Storage events
    MESSAGES_COMPRESSED = "messages_compressed"      # Old messages deleted after observation
    OBSERVATIONS_CONSOLIDATED = "observations_consolidated"  # After reflection


@dataclass
class OMEvent:
    type: EventType
    thread_id: str
    timestamp: datetime
    data: Dict[str, Any]


class CallbackManager:
    """
    Manages event callbacks. Users register handlers for specific events.
    """
    
    def __init__(self):
        self._handlers: Dict[EventType, List[Callable[[OMEvent], None]]] = {
            event_type: [] for event_type in EventType
        }
        
    def on(self, event_type: EventType, callback: Callable[[OMEvent], None]) -> None:
        if callback not in self._handlers[event_type]:
            self._handlers[event_type].append(callback)
            
    def remove(self, event_type: EventType, callback: Callable[[OMEvent], None]) -> None:
        if callback in self._handlers[event_type]:
            self._handlers[event_type].remove(callback)
            
    def emit(self, event: OMEvent) -> None:
        for handler in self._handlers[event.type]:
            try:
                handler(event)
            except Exception as e:
                # Log but don't crash the main flow due to callback error
                # In a real system, use standard logging
                print(f"Callback error in {event.type.value}: {str(e)}")
