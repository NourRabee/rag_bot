import logging

import requests
from langchain_core.messages import HumanMessage

from core.config import settings
from utils.prompts import build_chat_prompt
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from vectorstore.pinecone_vectordb import PineconeService


class LLMClient:
    def __init__(self):
        self.pinecone_db = PineconeService()

    def handle_user_query(self, session_id: str, user_id: str, prompt: str, model: str, provider: str,
                          namespace="general"):

        raw_result = self.pinecone_db.search(prompt, user_id, session_id)
        similar_docs = self.pinecone_db.get_text(raw_result)

        formatted_prompt, parser = build_chat_prompt(similar_docs, prompt)

        llm = self._get_client(
            provider=provider,
            model=model
        )

        message = HumanMessage(content=formatted_prompt)
        response = llm.invoke([message])

        full_text = f"User: {prompt}\nAssistant: {response.content}"

        docs = self.pinecone_db.convert_to_docs(full_text, user_id, session_id)

        self.pinecone_db.upsert(docs, namespace)

        return parser.parse(response.content)

    def _get_client(self, provider: str, model: str):
        if provider == "ollama":
            return ChatOllama(

                model=model,
                base_url=settings.ollama_base_url,
                stream=False
            )
        elif provider == "groq":
            return ChatGroq(
                model=model,
                api_key=settings.groq_api_key
            )
        else:
            return None

    def fetch_ollama_models(self):
        try:
            response = requests.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except requests.RequestException as e:
            print(f"Error fetching models: {e}")
            return []
