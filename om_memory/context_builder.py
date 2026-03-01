from typing import Union, List, Dict
from om_memory.models import Observation, Message, ObservationLog, Priority
from om_memory.token_counter import TokenCounter
from om_memory.observability.callbacks import CallbackManager, OMEvent, EventType
from datetime import datetime, timezone

class ContextBuilder:
    """
    Builds the final context string from observations + messages.
    
    Produces a two-block context:
    - Block 1: Dense observation log (compressed history) — stable, cacheable
    - Block 2: Recent uncompressed messages — small rolling window
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
        callbacks: CallbackManager = None,
        message_token_budget: int = None,
        share_token_budget: bool = False,
    ) -> Union[str, Dict]:
        
        obs_log = ObservationLog(
            thread_id=thread_id,
            observations=observations,
            total_tokens=self.token_counter.count_observations(observations)
        )
        
        msg_total_tokens = self.token_counter.count_messages(messages)
        obs_total_tokens = obs_log.total_tokens
        
        # Apply message_token_budget: trim oldest messages to fit budget
        if message_token_budget and msg_total_tokens > message_token_budget:
            trimmed_messages = []
            budget_remaining = message_token_budget
            # Keep newest messages first
            for m in reversed(messages):
                t = m.token_count or self.token_counter.count(f"{m.role}: {m.content}")
                if budget_remaining >= t:
                    trimmed_messages.insert(0, m)
                    budget_remaining -= t
                else:
                    break
            messages = trimmed_messages
            msg_total_tokens = self.token_counter.count_messages(messages)
        
        # Truncate observations if combined budget exceeded
        if max_tokens and (obs_total_tokens + msg_total_tokens > max_tokens):
            sorted_obs = sorted(
                observations,
                key=lambda x: (x.priority == Priority.CRITICAL, x.observation_date),
                reverse=True,
            )
            kept_obs = []
            current_tokens = msg_total_tokens
            for o in sorted_obs:
                t = o.token_count or self.token_counter.count(o.content)
                if current_tokens + t <= max_tokens:
                    kept_obs.append(o)
                    current_tokens += t
            
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
            # Build compact two-block context
            blocks = []
            
            # Block 1: Observations (only if we have them)
            obs_text = obs_log.to_context_string()
            has_observations = bool(observations)
            
            if has_observations:
                if include_header:
                    blocks.append("[Memory — what I remember from this conversation]")
                blocks.append(obs_text)
            
            # Block 2: Recent messages (only if we have them)
            if messages:
                msg_lines = []
                for m in messages:
                    msg_lines.append(f"{m.role}: {m.content}")
                msg_text = "\n".join(msg_lines)
                
                if include_header:
                    blocks.append("\n[Recent]")
                blocks.append(msg_text)
            elif not has_observations:
                # No observations and no messages — empty context
                blocks.append("No conversation history.")
                
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
                "cache_eligible_tokens": obs_log.total_tokens
            }
        }
