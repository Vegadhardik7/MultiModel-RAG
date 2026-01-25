from typing import List, Optional
from chromadb.api.types import Metadata  # ✅ IMPORTANT
from app.embeddings.text_embedder import embed_texts
from app.vectorstore.chroma_client import get_collection


def add_documents(
    texts: List[str],
    metadatas: Optional[List[Metadata]] = None
):
    if not texts:
        return

    # Always provide valid Chroma metadata
    if metadatas is None:
        metadatas = [{"source": "manual"} for _ in texts]

    embeddings = embed_texts(texts)
    collection = get_collection()

    ids = [f"doc_{i}" for i in range(len(texts))]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=metadatas
    )


from typing import List, Dict, Any

def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    collection = get_collection()
    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )

    documents = results.get("documents")
    metadatas = results.get("metadatas")

    if not documents or not metadatas:
        return []

    documents = documents[0]
    metadatas = metadatas[0]

    retrieved = []
    for doc, meta in zip(documents, metadatas):
        retrieved.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", "unknown"),
            "type": meta.get("type", "unknown"),
        })

    return retrieved


