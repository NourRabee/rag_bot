from core.config import settings
import requests


class OllamaClient:
    def __init__(self):
        self.base_url = settings.ollama_base_url

    def get_models_name(self):
        response = requests.get(f"{settings.ollama_base_url}/api/tags")
        data = response.json()

        return [model["name"] for model in data.get("models", [])]
