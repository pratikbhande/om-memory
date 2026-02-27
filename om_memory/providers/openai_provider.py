import os
from om_memory.providers.base import LLMProvider

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    pass  # We will fail on __init__ if it's missing but actually used


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider using the official `openai` package.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        self._model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Pass api_key or set OPENAI_API_KEY env var.")
            
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.sync_client = OpenAI(api_key=self.api_key)
        
    @property
    def model_name(self) -> str:
        return self._model
        
    def _build_messages(self, system_prompt: str, user_prompt: str) -> list[dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    async def acomplete(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.async_client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(system_prompt, user_prompt)
        )
        return response.choices[0].message.content or ""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.sync_client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(system_prompt, user_prompt)
        )
        return response.choices[0].message.content or ""
