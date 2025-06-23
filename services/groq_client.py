import requests

from core.config import settings
from services.llm_client import LLMClient


class GroqClient(LLMClient):
    def __init__(self):
        self.base_url = settings.groq_base_url

    def get_response(
            self,
            prompt: str,
            uri: str,
            model: str,
            api_key: str = settings.groq_api_key,
            stream: bool = False
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

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(uri, json=body, headers=headers)
        response.raise_for_status()
        data = response.json()

        return data['choices'][0]['message']['content']



