from typing import Optional

from utils.embedder import get_embedding
from utils.prompt_builder import build_prompt
from vectorstore.pinecone_vectordb import PineconeService


class LLMClient:
    def __init__(self):
        self.pinecone_db = PineconeService()

    def get_response(self, prompt: str, uri: str, model: str, api_key: Optional[str] = None,
                     stream: Optional[bool] = False) -> str:
        pass

    def handle_user_query(self, session_id: str, user_id: str, prompt: str, uri: str, model: str, stream=False,
                          api_key=None, namespace="general"):
        query_embedding = get_embedding(prompt)

        raw_result = self.pinecone_db.search(query_embedding, session_id, user_id, namespace)
        similar_texts = self.pinecone_db.get_text(raw_result)
        print(similar_texts)

        built_prompt = build_prompt(similar_texts, prompt)

        response = self.get_response(
            prompt=built_prompt,
            uri=uri,
            model=model,
            stream=stream,
            api_key=api_key
        )

        full_text = f"User: {prompt}\nAssistant: {response}"
        full_embedding = get_embedding(full_text)

        self.pinecone_db.upsert(full_embedding, session_id, user_id, full_text, namespace)

        return response
