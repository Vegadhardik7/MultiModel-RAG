import os
import chromadb

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

CHROMA_ROOT = os.path.join(BASE_DIR, "chroma_sessions")
os.makedirs(CHROMA_ROOT, exist_ok=True)

_clients = {}


def init_session_collection(session_id: str):
    if session_id not in _clients:
        _clients[session_id] = chromadb.PersistentClient(
            path=os.path.join(CHROMA_ROOT, session_id)
        )


def get_collection(session_id: str):
    if session_id not in _clients:
        init_session_collection(session_id)

    return _clients[session_id].get_or_create_collection(
        name="documents"
    )
