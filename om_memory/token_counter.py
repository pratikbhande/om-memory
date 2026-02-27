from typing import Optional, Callable

from om_memory.models import Message, Observation

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

class TokenCounter:
    """
    Counts tokens for messages and observations.
    
    Uses tiktoken for OpenAI models (fast, accurate).
    Falls back to word-based approximation (1 token ≈ 0.75 words) if tiktoken unavailable.
    Supports custom tokenizers via callback.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", custom_tokenizer: Optional[Callable[[str], int]] = None):
        self.model = model
        self.custom_tokenizer = custom_tokenizer
        
        if HAS_TIKTOKEN and not custom_tokenizer:
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoding = None
            
    def count(self, text: str) -> int:
        if not text:
            return 0
            
        if self.custom_tokenizer:
            return self.custom_tokenizer(text)
            
        if self.encoding:
            return len(self.encoding.encode(text))
            
        # Fallback approximation: 1 token ≈ 0.75 words
        words = len(text.split())
        return int(words / 0.75)
        
    def count_messages(self, messages: list[Message]) -> int:
        total = 0
        for msg in messages:
            if msg.token_count is not None:
                total += msg.token_count
            else:
                formatted = f"{msg.role}: {msg.content}"
                count = self.count(formatted)
                msg.token_count = count
                total += count
        return total
        
    def count_observations(self, observations: list[Observation]) -> int:
        total = 0
        for obs in observations:
            if obs.token_count is not None:
                total += obs.token_count
            else:
                formatted = f"{obs.priority.value} {obs.content}"
                count = self.count(formatted)
                obs.token_count = count
                total += count
        return total
