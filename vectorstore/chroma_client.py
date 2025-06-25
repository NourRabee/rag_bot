import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE

from core.config import settings

client = chromadb.PersistentClient(
    path=settings.chroma_db_path,
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE
)

collection = client.get_or_create_collection(
    name=settings.chroma_collection_name,
    metadata={
        "description": "Prompts vector space",
        "created_by": "rag_bot"
    }
)
