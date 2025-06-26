from pinecone import Pinecone

from core.config import settings
from utils.id import generate_vector_id


class PineconeService:
    def __init__(self):
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.pc.Index(name="chatmemory")

    def upsert(self, embedding, session_id, user_id, text, namespace='general'):
        _id = generate_vector_id(session_id)
        self.index.upsert(
            vectors=[
                {
                    "id": _id,
                    "values": embedding,
                    "metadata": {"session_id": session_id, "user_id": user_id, "text": text}
                }
            ],
            namespace=namespace
        )

    def get_text(self, search_result):
        texts = [match["metadata"]["text"] for match in search_result["matches"]]
        return texts

    def search(self, embedding, session_id, user_id, namespace="general", top_k=3, include_metadata=True,
               include_values=True):
        raw_result = self.index.query(
            namespace=namespace,
            vector=embedding,
            top_k=top_k,
            include_metadata=include_metadata,
            include_values=include_values,
            filter={"$and": [{"user_id": user_id}, {"session_id": session_id}]}

        )
        return raw_result
