from datetime import datetime, timezone

from om_memory.models import Message, Observation, OMConfig
from om_memory.providers.base import LLMProvider
from om_memory.observability.callbacks import CallbackManager, OMEvent, EventType
from om_memory.token_counter import TokenCounter
from om_memory.prompts.observer_prompt import OBSERVER_SYSTEM_PROMPT
from om_memory.parsing import parse_observations


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
        callbacks: CallbackManager = None,
        resource_id: str = None,
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
            return []
            
        output_tokens = self.token_counter.count(llm_response)
            
        # Parse observations using shared utility
        new_observations = parse_observations(
            llm_response, thread_id, [m.id for m in messages], resource_id=resource_id
        )
        
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
