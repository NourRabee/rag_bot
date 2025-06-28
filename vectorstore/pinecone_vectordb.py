from typing import List

from pinecone import Pinecone
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from core.config import settings


class PineconeService:
    def __init__(self):
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.pc.Index(name=settings.pinecone_index)

        self.embedding = MistralAIEmbeddings(
            api_key=settings.mistral_api_key,
            model="mistral-embed"
        )

        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embedding,
            text_key="text",
            namespace="general"
        )

    def upsert(self, docs: [Document], namespace: str):
        self.vectorstore.add_documents(docs, namespace=namespace)

    def search(self, prompt, user_id, session_id, top_k=3):
        retriever = self.vectorstore.as_retriever(search_kwargs={
            "k": top_k,
            "filter": {
                "$and": [
                    {"user_id": user_id},
                    {"session_id": session_id}
                ]
            }
        })

        retrieved_docs = retriever.invoke(prompt)
        return retrieved_docs

    def get_text(self, results):
        return [doc.page_content for doc in results]

    def convert_to_docs(self, full_text: str, user_id: str, session_id: str) -> List[Document]:
        return [
            Document(
                page_content=full_text,
                metadata={
                    "user_id": user_id,
                    "session_id": session_id
                }
            )
        ]
