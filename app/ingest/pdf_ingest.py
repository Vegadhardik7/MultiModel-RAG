import os
from unstructured.partition.pdf import partition_pdf
from app.retriever import add_documents


def ingest_pdf(pdf_path: str):
    """
    Extracts text from a PDF and stores it in Chroma.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"{pdf_path} not found")

    elements = partition_pdf(pdf_path)

    texts = []
    metadatas = []

    for i, el in enumerate(elements):
        if not hasattr(el, "text"):
            continue
        if not el.text:
            continue

        text = el.text.strip()
        if len(text) < 50:  # skip junk
            continue

        texts.append(text)
        metadatas.append({
            "source": os.path.basename(pdf_path),
            "chunk": i
        })

    if not texts:
        print("No usable text found in PDF.")
        return

    add_documents(texts, metadatas)
    print(f"Ingested {len(texts)} chunks from {pdf_path}")

