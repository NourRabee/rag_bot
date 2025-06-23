import requests

from core.config import settings
import httpx

from services.llm_client import LLMClient


class OllamaClient(LLMClient):
    def __init__(self):
        self.base_url = settings.ollama_base_url

    def fetch_models_name(self):

        response = requests.get(f"{self.base_url}/api/tags")
        data = response.json()

        return [model["name"] for model in data.get("models", [])]

    def get_response(
            self,
            prompt: str,
            uri: str,
            model: str = "llama3.2:1b",
            api_key: str = None,
            stream: bool = True
    ) -> str:

        body = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                 }
            ],
            "stream": stream
        }

        response = requests.post(uri, json=body)
        response.raise_for_status()
        data = response.json()

        return data.get('message', {}).get('content', '')


