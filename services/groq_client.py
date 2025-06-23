from core.config import settings
from services.llm_client import LLMClient


class GroqClient(LLMClient):
    def __init__(self):
        self.base_url = settings.groq_base_url

    def get_response(self, prompt, uri, model):
        pass
