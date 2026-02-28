from datetime import datetime, timezone

from om_memory.models import Observation, OMConfig
from om_memory.providers.base import LLMProvider
from om_memory.observability.callbacks import CallbackManager, OMEvent, EventType
from om_memory.token_counter import TokenCounter
from om_memory.prompts.reflector_prompt import REFLECTOR_SYSTEM_PROMPT
from om_memory.parsing import parse_observations


class Reflector:
    """
    The Reflector agent garbage-collects and consolidates old observations.
    """
    
    def __init__(self, provider: LLMProvider, config: OMConfig, token_counter: TokenCounter):
        self.provider = provider
        self.config = config
        self.token_counter = token_counter
        
    async def areflect(
        self,
        thread_id: str,
        observations: list[Observation],
        callbacks: CallbackManager = None,
        resource_id: str = None,
    ) -> list[Observation]:
        
        if callbacks:
            callbacks.emit(OMEvent(type=EventType.REFLECTOR_STARTED, thread_id=thread_id, timestamp=datetime.now(timezone.utc), data={}))
            
        if not observations:
            return []
            
        system_prompt = REFLECTOR_SYSTEM_PROMPT
        
        user_prompt = "Current Observations:\n"
        for o in observations:
            user_prompt += f"{o.priority.value} [{o.observation_date.strftime('%Y-%m-%d %H:%M')}] {o.content}\n"
            
        input_tokens = self.token_counter.count(system_prompt + "\n" + user_prompt)
        
        try:
            llm_response = await self.provider.acomplete(system_prompt, user_prompt)
        except Exception as e:
            if callbacks:
                callbacks.emit(OMEvent(type=EventType.REFLECTOR_ERROR, thread_id=thread_id, timestamp=datetime.now(timezone.utc), data={"error": str(e)}))
            return observations  # Return unchanged on error
            
        output_tokens = self.token_counter.count(llm_response)
        
        all_source_message_ids = []
        for o in observations:
            all_source_message_ids.extend(o.source_message_ids)
        all_source_message_ids = list(set(all_source_message_ids))
        
        # Use shared parsing utility
        new_observations = parse_observations(
            llm_response, thread_id, all_source_message_ids, resource_id=resource_id
        )
        
        if callbacks:
            callbacks.emit(OMEvent(
                type=EventType.REFLECTOR_COMPLETED,
                thread_id=thread_id,
                timestamp=datetime.now(timezone.utc),
                data={
                    "observations_before": len(observations),
                    "observations_after": len(new_observations),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
            ))
            
        return new_observations
