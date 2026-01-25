import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Load once (important for performance)
_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def embed_texts(texts: list[str]):
    """
    Convert a list of texts into dense vector embeddings.
    """
    if not texts:
        return []

    embeddings = _model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    return embeddings
