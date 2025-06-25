from vectorstore.chroma_client import collection


def query_memory(query_embedding, user_id, session_id, top_k=5):
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"$and": [
            {"user_id": {"$eq": user_id}},
            {"session_id": {"$eq": session_id}}
        ]},
        include=["documents"]
    )


def store_memory(vector_id, doc, metadata, embedding):
    collection.add(
        documents=[doc],
        metadatas=[metadata],
        ids=[vector_id],
        embeddings=[embedding]
    )
