import os
from om_memory.providers.base import LLMProvider

try:
    from anthropic import AsyncAnthropic, Anthropic
except ImportError:
    pass

class AnthropicProvider(LLMProvider):
    """
    Anthropic provider using the official `anthropic` package.
    """
    
    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: str = None):
        self._model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Pass api_key or set ANTHROPIC_API_KEY.")
            
        self.async_client = AsyncAnthropic(api_key=self.api_key)
        self.sync_client = Anthropic(api_key=self.api_key)
        
    @property
    def model_name(self) -> str:
        return self._model

    async def acomplete(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.async_client.messages.create(
            model=self._model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=2000
        )
        return response.content[0].text

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.sync_client.messages.create(
            model=self._model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=2000
        )
        return response.content[0].text
