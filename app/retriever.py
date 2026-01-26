from typing import List, Dict, Any
from app.embeddings.text_embedder import embed_texts
from app.vectorstore.chroma_client import get_collection


SUMMARY_KEYWORDS = {
    "summarize",
    "summary",
    "overview",
    "tell me about",
    "what is this pdf",
    "what do you know",
    "describe this document",
}


def _is_summary_query(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in SUMMARY_KEYWORDS)


def retrieve(
    query: str,
    session_id: str,
    k: int = 6
) -> List[Dict[str, Any]]:

    collection = get_collection(session_id)

    # ---------- DOCUMENT-LEVEL RETRIEVAL ----------
    if _is_summary_query(query):
        result = collection.get(
            where={"type": "document_overview"},
            limit=1
        )

        docs = result.get("documents", [])
        metas = result.get("metadatas", [])

        if docs and metas:
            return [{
                "text": docs[0],
                "source": metas[0].get("source"),
                "page": metas[0].get("page"),
                "type": metas[0].get("type"),
            }]

    # ---------- NORMAL SEMANTIC RETRIEVAL ----------
    query_embedding = embed_texts([query])[0]

    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
    )

    docs = (result.get("documents") or [[]])[0] or []
    metas = (result.get("metadatas") or [[]])[0] or []

    retrieved: List[Dict[str, Any]] = []
    for doc, meta in zip(docs, metas):
        retrieved.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", "unknown"),
            "type": meta.get("type", "unknown"),
        })

    return retrieved
