from typing import List, Dict, Any
from app.embeddings.text_embedder import embed_texts
from app.vectorstore.chroma_client import get_collection


def retrieve(query: str, session_id: str, k: int = 6) -> List[Dict[str, Any]]:
    collection = get_collection(session_id)
    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
    )

    documents = results.get("documents")
    metadatas = results.get("metadatas")

    if not documents or not metadatas:
        return []

    documents = documents[0]
    metadatas = metadatas[0]

    retrieved = []
    for doc, meta in zip(documents, metadatas):
        item = ({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", "unknown"),
            "type": meta.get("type", "unknown"),
        })
        retrieved.append(item)

    return retrieved
