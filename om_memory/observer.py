import re
from datetime import datetime, time, timezone
from typing import Optional

from om_memory.models import Message, Observation, Priority, OMConfig
from om_memory.providers.base import LLMProvider
from om_memory.observability.callbacks import CallbackManager, OMEvent, EventType
from om_memory.token_counter import TokenCounter
from om_memory.prompts.observer_prompt import OBSERVER_SYSTEM_PROMPT


class Observer:
    """
    The Observer agent compresses raw conversation messages into observations.
    """
    
    def __init__(self, provider: LLMProvider, config: OMConfig, token_counter: TokenCounter):
        self.provider = provider
        self.config = config
        self.token_counter = token_counter
        
    async def aobserve(
        self, 
        thread_id: str,
        messages: list[Message],
        existing_observations: list[Observation] = None,
        callbacks: CallbackManager = None
    ) -> list[Observation]:
        
        if callbacks:
            callbacks.emit(OMEvent(type=EventType.OBSERVER_STARTED, thread_id=thread_id, timestamp=datetime.now(timezone.utc), data={}))
            
        existing_observations = existing_observations or []
        
        # Build prompt
        context_str = "No previous context."
        if existing_observations:
            context_str = "\n".join([f"{o.priority.value} {o.observation_date.strftime('%Y-%m-%d %H:%M')} {o.content}" for o in existing_observations])
            
        user_prompt = f"Previous Observations:\n{context_str}\n\nRecent Messages to Compress:\n"
        for msg in messages:
            user_prompt += f"{msg.timestamp.strftime('%Y-%m-%d %H:%M')} {msg.role}: {msg.content}\n"
            
        system_prompt = OBSERVER_SYSTEM_PROMPT
        
        input_tokens = self.token_counter.count(system_prompt + "\n" + user_prompt)

        try:
            llm_response = await self.provider.acomplete(system_prompt, user_prompt)
        except Exception as e:
            if callbacks:
                callbacks.emit(OMEvent(type=EventType.OBSERVER_ERROR, thread_id=thread_id, timestamp=datetime.now(timezone.utc), data={"error": str(e)}))
            # Graceful degradation
            return []
            
        output_tokens = self.token_counter.count(llm_response)
            
        # Parse observations
        new_observations = self._parse_observations(llm_response, thread_id, [m.id for m in messages])
        
        if callbacks:
            callbacks.emit(OMEvent(
                type=EventType.OBSERVER_COMPLETED,
                thread_id=thread_id,
                timestamp=datetime.now(timezone.utc),
                data={
                    "messages_compressed": len(messages),
                    "observations_created": len(new_observations),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
            ))
            
        return new_observations
        
    def _parse_observations(self, llm_response: str, thread_id: str, source_message_ids: list[str]) -> list[Observation]:
        observations = []
        current_date = datetime.now(timezone.utc)
        
        # Extract CURRENT_TASK and SUGGESTED_NEXT
        task_match = re.search(r"CURRENT_TASK:\s*(.*)", llm_response)
        next_match = re.search(r"SUGGESTED_NEXT:\s*(.*)", llm_response)
        
        if task_match:
            observations.append(Observation(
                thread_id=thread_id,
                priority=Priority.CRITICAL,
                content=f"CURRENT TASK: {task_match.group(1).strip()}",
                source_message_ids=source_message_ids
            ))
            
        if next_match:
            observations.append(Observation(
                thread_id=thread_id,
                priority=Priority.IMPORTANT,
                content=f"SUGGESTED NEXT: {next_match.group(1).strip()}",
                source_message_ids=source_message_ids
            ))

        lines = llm_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith("Date:"):
                try:
                    date_str = line.split(":")[1].strip()
                    parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    current_date = datetime.combine(parsed_date, current_date.time(), tzinfo=timezone.utc)
                except Exception:
                    pass # Ignore parse errors, use current utc
                continue
                
            if line.startswith("-") and any(emoji in line for emoji in ["🔴", "🟡", "🟢"]):
                try:
                    # Parse parts
                    priority_val = Priority.INFO
                    for p in Priority:
                        if p.value in line:
                            priority_val = p
                            break
                            
                    # Extract time and content
                    time_match = re.search(r"(\d{2}:\d{2})", line)
                    obs_time = current_date.time()
                    if time_match:
                        obs_time = datetime.strptime(time_match.group(1), "%H:%M").time()
                        
                    content_start = line.find(priority_val.value) + len(priority_val.value)
                    if time_match:
                        content_start = line.find(time_match.group(1)) + len(time_match.group(1))
                        
                    raw_content = line[content_start:].strip()
                    
                    # Extract references
                    ref_date = None
                    rel_date = None
                    ref_match = re.search(r"\(([^)]*referenced[^)]*)\)", raw_content)
                    if ref_match:
                        ref_str = ref_match.group(1)
                        raw_content = raw_content.replace(f"({ref_str})", "").strip()
                        
                        date_match = re.search(r"referenced:\s*(\d{4}-\d{2}-\d{2})", ref_str)
                        if date_match:
                            try:
                                ref_date = datetime.strptime(date_match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
                            except Exception:
                                pass
                                
                        meaning_match = re.search(r"meaning\s*\"([^\"]+)\"", ref_str)
                        if meaning_match:
                            rel_date = meaning_match.group(1)
                            
                    obs_date = datetime.combine(current_date.date(), obs_time, tzinfo=timezone.utc)
                    
                    obs = Observation(
                        thread_id=thread_id,
                        observation_date=obs_date,
                        referenced_date=ref_date,
                        relative_date=rel_date,
                        priority=priority_val,
                        content=raw_content,
                        source_message_ids=source_message_ids
                    )
                    observations.append(obs)
                except Exception as e:
                    # Ignore malformed lines gracefully
                    pass
                    
        return observations
