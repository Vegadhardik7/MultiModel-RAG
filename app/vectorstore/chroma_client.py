# app/vectorstore/chroma_client.py

from pathlib import Path
from typing import Dict, Any
import chromadb


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_ROOT = BASE_DIR / "data" / "chroma_sessions"
CHROMA_ROOT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# CLIENT CACHE
# --------------------------------------------------

_clients: Dict[str, Any] = {}


# --------------------------------------------------
# INTERNAL
# --------------------------------------------------

def _get_client(session_id: str):
    """
    Returns a persistent Chroma client for a session.
    Clients are cached to avoid reopening DB handles.
    """
    if session_id not in _clients:
        session_path = CHROMA_ROOT / session_id
        session_path.mkdir(parents=True, exist_ok=True)

        _clients[session_id] = chromadb.PersistentClient(
            path=str(session_path)
        )

    return _clients[session_id]


# --------------------------------------------------
# PUBLIC API
# --------------------------------------------------

def init_session_collection(session_id: str):
    """
    Create (or load) the session's collection.
    Safe to call multiple times.
    """
    client = _get_client(session_id)
    return client.get_or_create_collection(
        name="documents",
        metadata={"session_id": session_id}
    )


def get_collection(session_id: str):
    """
    Fetch session-specific vector collection.
    """
    client = _get_client(session_id)
    return client.get_or_create_collection(name="documents")


def delete_session_collection(session_id: str):
    """
    Delete vector store for a session.
    Called ONLY when user explicitly deletes a session.
    """
    client = _clients.pop(session_id, None)

    if client:
        try:
            client.delete_collection("documents")
        except Exception:
            pass

    session_path = CHROMA_ROOT / session_id
    if session_path.exists():
        for p in session_path.rglob("*"):
            if p.is_file():
                p.unlink()
        for p in sorted(session_path.glob("**/*"), reverse=True):
            if p.is_dir():
                p.rmdir()
        session_path.rmdir()
