from typing import Union, List, Dict
from om_memory.models import Observation, Message, ObservationLog
from om_memory.token_counter import TokenCounter
from om_memory.observability.callbacks import CallbackManager, OMEvent, EventType
from datetime import datetime, timezone

class ContextBuilder:
    """
    Builds the final context string from observations + messages.
    """
    
    def __init__(self, token_counter: TokenCounter):
        self.token_counter = token_counter

    def build_context(
        self,
        thread_id: str,
        observations: list[Observation],
        messages: list[Message],
        max_tokens: int = None,
        include_header: bool = True,
        format: str = "text",
        callbacks: CallbackManager = None
    ) -> Union[str, Dict]:
        
        obs_log = ObservationLog(
            thread_id=thread_id,
            observations=observations,
            total_tokens=self.token_counter.count_observations(observations)
        )
        
        msg_total_tokens = self.token_counter.count_messages(messages)
        obs_total_tokens = obs_log.total_tokens
        
        # Optionally truncate
        if max_tokens and (obs_total_tokens + msg_total_tokens > max_tokens):
            # Keep as many messages as possible, then keep as many observations as possible
            # But usually we truncate observations (oldest INFO) first, then older messages.
            # For simplicity in this implementation, we will keep messages and truncate oldest non-critical observations.
            sorted_obs = sorted(observations, key=lambda x: (x.priority == Priority.CRITICAL, x.observation_date), reverse=True)
            kept_obs = []
            current_tokens = msg_total_tokens
            for o in sorted_obs:
                t = o.token_count or self.token_counter.count(o.content)
                if current_tokens + t <= max_tokens:
                    kept_obs.append(o)
                    current_tokens += t
            
            # Re-sort chronologically for the context
            observations = sorted(kept_obs, key=lambda x: x.observation_date)
            obs_log.observations = observations
            obs_log.total_tokens = self.token_counter.count_observations(observations)
            
            obs_total_tokens = obs_log.total_tokens

        if format == "dict":
            ctx = self.build_context_dict(obs_log, messages)
        elif format == "json":
            import json
            ctx = json.dumps(self.build_context_dict(obs_log, messages), default=str)
        else:
            # text
            obs_text = obs_log.to_context_string()
            msg_text = "\n".join([f"{m.role}: {m.content}" for m in messages])
            
            blocks = []
            if include_header:
                blocks.append("=== CONVERSATION MEMORY ===")
            blocks.append(obs_text)
            
            if messages:
                if include_header:
                    blocks.append("\n=== RECENT MESSAGES ===")
                blocks.append(msg_text)
                
            ctx = "\n".join(blocks)
            
        if callbacks:
            callbacks.emit(OMEvent(
                type=EventType.CONTEXT_BUILT,
                thread_id=thread_id,
                timestamp=datetime.now(timezone.utc),
                data={
                    "observation_tokens": obs_total_tokens,
                    "message_tokens": msg_total_tokens,
                    "total_tokens": obs_total_tokens + msg_total_tokens
                }
            ))
            
        return ctx
        
    def build_context_dict(
        self,
        obs_log: ObservationLog,
        messages: list[Message],
    ) -> dict:
        
        current_task = ""
        suggested_next = ""
        
        # Extract the special tasks from the observations to place them distinctly in the dict
        # Actually in our parsing we just added them as CRITICAL/IMPORTANT observations with the prefix
        for o in obs_log.observations:
            if o.content.startswith("CURRENT TASK:"):
                current_task = o.content.replace("CURRENT TASK:", "").strip()
            elif o.content.startswith("SUGGESTED NEXT:"):
                suggested_next = o.content.replace("SUGGESTED NEXT:", "").strip()
                
        msg_dicts = [{"role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()} for m in messages]
        
        return {
            "observations_text": obs_log.to_context_string(),
            "messages": msg_dicts,
            "current_task": current_task,
            "suggested_next": suggested_next,
            "stats": {
                "observation_tokens": obs_log.total_tokens,
                "message_tokens": self.token_counter.count_messages(messages),
                "total_tokens": obs_log.total_tokens + self.token_counter.count_messages(messages),
                "cache_eligible_tokens": obs_log.total_tokens  # stable prefix
            }
        }
