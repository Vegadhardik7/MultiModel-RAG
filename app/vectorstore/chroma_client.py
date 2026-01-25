import os
import chromadb

# Absolute project root
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# ✅ THIS IS THE KEY DIFFERENCE
_client = chromadb.PersistentClient(path=CHROMA_DIR)

def get_collection():
    return _client.get_or_create_collection(
        name="forgerag"
    )
