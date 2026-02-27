from om_memory.providers.base import LLMProvider

try:
    from litellm import completion, acompletion
except ImportError:
    pass

class LiteLLMProvider(LLMProvider):
    """
    Provider using `litellm` to support 100+ different LLMs via a single interface.
    """
    
    def __init__(self, model: str):
        # The user must provide a model string compatible with litellm (e.g., 'gemini/gemini-1.5-flash', 'groq/llama3-8b-8192')
        self._model = model
        
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
        response = await acompletion(
            model=self._model,
            messages=self._build_messages(system_prompt, user_prompt)
        )
        return response.choices[0].message.content or ""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = completion(
            model=self._model,
            messages=self._build_messages(system_prompt, user_prompt)
        )
        return response.choices[0].message.content or ""
