# app/ingest/multimodal_pdf_ingest.py

import os
from typing import List, Optional
from chromadb.api.types import Metadata

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    NarrativeText,
    Title,
    Table,
    Image as UnstructuredImage,
)

from app.embeddings.text_embedder import embed_texts
from app.embeddings.clip_helper import describe_image_with_clip
from app.vectorstore.chroma_client import get_collection


def _linearize_table(table: Table) -> str:
    return table.text.strip() if getattr(table, "text", None) else ""


def _describe_image(
    image_element: UnstructuredImage,
    surrounding_text: Optional[str],
) -> Optional[str]:
    parts = []

    if image_element.metadata and image_element.metadata.page_number:
        parts.append(f"Image on page {image_element.metadata.page_number}")

    if image_element.metadata and image_element.metadata.image_path:
        try:
            tag = describe_image_with_clip(image_element.metadata.image_path)
            parts.append(f"Visual content: {tag}")
        except Exception:
            pass

    if surrounding_text:
        parts.append(f"Surrounding context: {surrounding_text}")

    desc = " ".join(parts).strip()
    return desc if len(desc) > 40 else None


# ---------------- MAIN ----------------

def ingest_multimodal_pdf(pdf_path: str, session_id: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    elements = partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,
        extract_images_in_pdf=True,
    )

    texts: List[str] = []
    metadatas: List[Metadata] = []

    prev_text: Optional[str] = None

    for el in elements:

        # -------- TEXT --------
        if isinstance(el, (NarrativeText, Title)) and el.text:
            text = el.text.strip()
            if len(text) < 80:
                continue

            texts.append(text)
            metadatas.append({
                "source": os.path.basename(pdf_path),
                "page": int(el.metadata.page_number or 0),
                "type": "text",
            })
            prev_text = text

        # -------- TABLE --------
        elif isinstance(el, Table):
            table_text = _linearize_table(el)
            if len(table_text) < 80:
                continue

            texts.append(f"Table: {table_text}")
            metadatas.append({
                "source": os.path.basename(pdf_path),
                "page": int(el.metadata.page_number or 0),
                "type": "table",
            })
            prev_text = table_text

        # -------- IMAGE --------
        elif isinstance(el, UnstructuredImage):
            desc = _describe_image(el, prev_text)
            if not desc:
                continue

            texts.append(f"Image context: {desc}")
            metadatas.append({
                "source": os.path.basename(pdf_path),
                "page": int(el.metadata.page_number or 0),
                "type": "image",
            })
            prev_text = None

    if not texts:
        print("⚠️ No usable content extracted")
        return

    assert len(texts) == len(metadatas)

    embeddings = embed_texts(texts)
    collection = get_collection(session_id)

    ids = [f"{os.path.basename(pdf_path)}_{i}" for i in range(len(texts))]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=metadatas,
    )

    print(f"✅ Ingested {len(texts)} chunks for session {session_id}")
