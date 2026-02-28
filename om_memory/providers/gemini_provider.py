import os
from om_memory.providers.base import LLMProvider

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class GeminiProvider(LLMProvider):
    """
    Google Gemini provider using the google-generativeai SDK.
    
    Usage:
        provider = GeminiProvider(model="gemini-2.0-flash", api_key="...")
    """
    
    def __init__(self, model: str = "gemini-2.0-flash", api_key: str = None):
        if not HAS_GENAI:
            raise ImportError(
                "google-generativeai is required for GeminiProvider. "
                "Install it with: pip install om-memory[gemini]"
            )
        
        self._model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key is required. Pass api_key or set GOOGLE_API_KEY / GEMINI_API_KEY env var."
            )
        
        genai.configure(api_key=self.api_key)
        self._client = genai.GenerativeModel(self._model)
        
    @property
    def model_name(self) -> str:
        return self._model

    async def acomplete(self, system_prompt: str, user_prompt: str) -> str:
        """Async completion — uses generate_content_async."""
        combined_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        response = await self._client.generate_content_async(combined_prompt)
        return response.text or ""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Sync completion."""
        combined_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        response = self._client.generate_content(combined_prompt)
        return response.text or ""
