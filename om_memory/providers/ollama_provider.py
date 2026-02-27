import httpx
from om_memory.providers.base import LLMProvider

class OllamaProvider(LLMProvider):
    """
    Ollama provider for local models using HTTPX.
    """
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self._model = model
        self.base_url = base_url.rstrip("/")
        
    @property
    def model_name(self) -> str:
        return self._model
        
    def _build_messages(self, system_prompt: str, user_prompt: str) -> list[dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages
        
    def _build_payload(self, system_prompt: str, user_prompt: str) -> dict:
        return {
            "model": self._model,
            "messages": self._build_messages(system_prompt, user_prompt),
            "stream": False
        }

    async def acomplete(self, system_prompt: str, user_prompt: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=self._build_payload(system_prompt, user_prompt),
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/api/chat",
                json=self._build_payload(system_prompt, user_prompt),
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
