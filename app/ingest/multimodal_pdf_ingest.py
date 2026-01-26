import os
from typing import List
from chromadb.api.types import Metadata
from app.embeddings.clip_helper import describe_image_with_clip

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    NarrativeText,
    Title,
    Table,
    Image as UnstructuredImage,
)

from app.embeddings.text_embedder import embed_texts
from app.vectorstore.chroma_client import get_collection


def _linearize_table(table: Table) -> str:
    if hasattr(table, "text") and table.text:
        return table.text.strip()
    return ""


def _describe_image(
    image_element: UnstructuredImage,
    surrounding_text: str | None
) -> str:
    description_parts: List[str] = []

    # Page info
    if image_element.metadata and image_element.metadata.page_number:
        description_parts.append(
            f"Image on page {image_element.metadata.page_number}"
        )

    # CLIP-based visual semantics
    if image_element.metadata and image_element.metadata.image_path:
        try:
            visual_tag = describe_image_with_clip(
                image_element.metadata.image_path
            )
            description_parts.append(f"Visual content: {visual_tag}")
        except Exception:
            pass

    # Surrounding text (very important signal)
    if surrounding_text:
        description_parts.append(
            f"Surrounding context: {surrounding_text}"
        )

    return " ".join(description_parts).strip()


def ingest_multimodal_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    elements = partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,
        extract_images_in_pdf=True,
    )

    texts: List[str] = []
    metadatas: List[Metadata] = []

    prev_text_buffer = ""

    for el in elements:

        if isinstance(el, (NarrativeText, Title)):
            text = el.text.strip()
            if len(text) < 50:
                continue

            texts.append(text)
            metadatas.append({
                "source": os.path.basename(pdf_path),
                "page": int(el.metadata.page_number or 0),
                "type": "text",
            })

            prev_text_buffer = text

        elif isinstance(el, Table):
            table_text = _linearize_table(el)
            if not table_text or len(table_text) < 50:
                continue

            texts.append(f"Table: {table_text}")
            metadatas.append({
                "source": os.path.basename(pdf_path),
                "page": int(el.metadata.page_number or 0),
                "type": "table",
            })

            prev_text_buffer = table_text

        elif isinstance(el, UnstructuredImage):
            image_description = _describe_image(
                el,
                surrounding_text=prev_text_buffer
            )

            if not image_description:
                continue

            texts.append(f"Image description: {image_description}")
            metadatas.append({
                "source": os.path.basename(pdf_path),
                "page": int(el.metadata.page_number or 0),
                "type": "image",
            })

            prev_text_buffer = ""

    if not texts:
        print("No usable elements extracted.")
        return

    embeddings = embed_texts(texts)

    collection = get_collection()
    ids = [f"{os.path.basename(pdf_path)}_{i}" for i in range(len(texts))]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=metadatas,
    )

    print(f"Ingested {len(texts)} multimodal elements from {pdf_path}")
