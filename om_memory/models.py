import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel, Field

def utcnow() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)

class Priority(str, Enum):
    """Observation priority levels — emoji log levels for memory."""
    CRITICAL = "🔴"    # Decisions, deadlines, requirements, key facts
    IMPORTANT = "🟡"   # Preferences, considerations, open questions
    INFO = "🟢"        # Nice-to-know, minor details

class Message(BaseModel):
    """A single conversation message."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str
    role: str  # "user", "assistant", "system", "tool"
    content: str
    # Keep timestamps naive if user provides them, but our default is timezone-aware UTC.
    timestamp: datetime = Field(default_factory=utcnow)
    token_count: Optional[int] = None  # Calculated on creation
    metadata: dict[str, Any] = Field(default_factory=dict)

class Observation(BaseModel):
    """A single compressed observation extracted from conversation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str
    observation_date: datetime = Field(default_factory=utcnow)
    referenced_date: Optional[datetime] = None
    relative_date: Optional[str] = None
    priority: Priority = Priority.INFO
    content: str
    source_message_ids: list[str] = Field(default_factory=list)  # Traceability
    token_count: Optional[int] = None

class ObservationLog(BaseModel):
    """The complete observation log for a thread — Block 1 of context window."""
    thread_id: str
    observations: list[Observation] = Field(default_factory=list)
    total_tokens: int = 0
    last_observation_at: Optional[datetime] = None
    last_reflection_at: Optional[datetime] = None
    reflection_count: int = 0
    
    def to_context_string(self) -> str:
        """Render observations as the text block for the context window."""
        if not self.observations:
            return "No previous memory observed."
            
        # Group observations by date (using YYYY-MM-DD format based on observation_date)
        grouped: dict[str, list[Observation]] = {}
        for obs in self.observations:
            date_str = obs.observation_date.strftime("%Y-%m-%d")
            grouped.setdefault(date_str, []).append(obs)
            
        lines = []
        for date_str in sorted(grouped.keys()):
            lines.append(f"Date: {date_str}")
            # Sort within date chronologically
            sorted_obs = sorted(grouped[date_str], key=lambda x: x.observation_date)
            for obs in sorted_obs:
                time_str = obs.observation_date.strftime("%H:%M")
                
                # Format: 🔴 HH:MM Content (referenced: YYYY-MM-DD, meaning "...")
                ref_part = ""
                if obs.referenced_date or obs.relative_date:
                    ref_inner = []
                    if obs.referenced_date:
                        ref_inner.append(f"referenced: {obs.referenced_date.strftime('%Y-%m-%d')}")
                    if obs.relative_date:
                        ref_inner.append(f"meaning \"{obs.relative_date}\"")
                    ref_part = f" ({', '.join(ref_inner)})"
                    
                lines.append(f"- {obs.priority.value} {time_str} {obs.content}{ref_part}")
        
        return "\n".join(lines)


class OMConfig(BaseModel):
    """Configuration for ObservationalMemory."""
    # Thresholds
    observer_token_threshold: int = 30000    # Trigger Observer when messages exceed this
    reflector_token_threshold: int = 40000   # Trigger Reflector when observations exceed this
    max_message_history_tokens: int = 50000  # Hard cap on raw message history
    
    # Observer/Reflector model
    observer_model: Optional[str] = None     # None = use provider default
    reflector_model: Optional[str] = None
    
    # Behavior
    auto_observe: bool = True       # Auto-trigger Observer when threshold hit
    auto_reflect: bool = True       # Auto-trigger Reflector when threshold hit
    blocking_mode: bool = True      # True = block during compression, False = background async task
    
    # Observation format
    use_emoji_priority: bool = True
    use_three_date_model: bool = True
    
    # Cost tracking
    track_costs: bool = True
    cost_per_1k_input_tokens: float = 0.01   # Default, user should set based on their model
    cost_per_1k_output_tokens: float = 0.03
    cached_token_discount: float = 0.9       # 90% discount for cached tokens


class OMStats(BaseModel):
    """Runtime statistics for a thread."""
    thread_id: str
    total_messages: int = 0
    total_observations: int = 0
    total_reflections: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    estimated_cost_with_om: float = 0.0
    estimated_cost_without_om: float = 0.0   # What it would cost with naive approach
    cost_savings: float = 0.0
    compression_ratio: float = 0.0           # raw_tokens / compressed_tokens
    cache_hit_rate: float = 0.0
    observer_runs: int = 0
    reflector_runs: int = 0
    avg_context_window_tokens: int = 0
