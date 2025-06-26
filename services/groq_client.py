import requests

from core.config import settings
from services.llm_client import LLMClient


class GroqClient(LLMClient):
    def __init__(self):
        super().__init__()
        self.base_url = settings.groq_base_url

    def get_response(
            self,
            prompt: str,
            uri: str,
            model: str,
            api_key: str = settings.groq_api_key,
            stream: bool = False
    ) -> str:
        # add_to_conversation(role="user", content=prompt)

        body = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "stream": stream
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(uri, json=body, headers=headers)
        data = response.json()

        answer = data['choices'][0]['message']['content']

        # add_to_conversation(role="assistant", content=answer)

        return answer
