from typing import Optional

from utils.embedder import get_embedding
from utils.id import generate_vector_id
from utils.prompt_builder import build_prompt
from vectorstore.chroma_store import query_memory, store_memory


class LLMClient:
    def get_response(self, prompt: str, uri: str, model: str, api_key: Optional[str] = None,
                     stream: Optional[bool] = False) -> str:
        pass

    def handle_user_query(self, session_id: str, user_id: str, prompt: str, uri: str, model: str, stream=False,
                          api_key=None):
        query_embedding = get_embedding(prompt)

        results = query_memory(query_embedding, user_id, session_id)
        print(results)
        print("\n\n\n")
        past_docs = results["documents"][0] if results["documents"] else []
        print(past_docs)

        built_prompt = build_prompt(past_docs, prompt)

        print(built_prompt)

        response = self.get_response(
            prompt=built_prompt,
            uri=uri,
            model=model,
            stream=stream,
            api_key=api_key
        )

        full_text = f"User: {prompt}\nAssistant: {response}"
        full_embedding = get_embedding(full_text)

        vector_id = generate_vector_id(session_id)
        store_memory(
            vector_id=vector_id,
            doc=full_text,
            metadata={"user_id": user_id, "session_id": session_id},
            embedding=full_embedding
        )

        return response
