from unstructured.partition.pdf import partition_pdf
from app.embeddings.text_embedder import embed_text
from app.vectorstore.chroma_client import get_collection

def process_pdf(path: str):
    elements = partition_pdf(path)
    texts = [e.text for e in elements if e.text]

    embeddings = embed_text(texts)
    col = get_collection("forgerag")

    for i, text in enumerate(texts):
        col.add(
            ids=[f"{path}_{i}"],
            documents=[text],
            embeddings=[embeddings[i].tolist()]
        )
