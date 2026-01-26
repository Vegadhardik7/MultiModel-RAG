# app/api.py

import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.ingest.multimodal_pdf_ingest import ingest_multimodal_pdf
from app.rag_pipeline import run_rag, run_rag_stream
from app.vectorstore.chroma_client import init_session_collection
from memory.session_store import SessionStore

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
store = SessionStore()

# --------------------------------------------------
# MODELS
# --------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    question: str


# --------------------------------------------------
# UPLOAD PDF → CREATE **NEW** SESSION (DOES NOT TOUCH OLD ONES)
# --------------------------------------------------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    # 🔒 ALWAYS create a NEW session
    session_id = str(uuid.uuid4())

    pdf_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 🔒 Init isolated vector DB (per session)
    init_session_collection(session_id)

    # 🔒 Create session in SQLite ONCE
    store.create_session(session_id)

    # 🔒 Ingest PDF into this session only
    ingest_multimodal_pdf(
        pdf_path=str(pdf_path),
        session_id=session_id
    )

    return {
        "session_id": session_id,
        "name": Path(file.filename).stem
    }


# --------------------------------------------------
# LIST ALL SESSIONS (FOR SIDEBAR)
# --------------------------------------------------

@app.get("/sessions")
def list_sessions():
    """
    Returns all known sessions (persistent).
    """
    sessions = store.list_sessions()
    return [
        {
            "session_id": sid
        }
        for sid in sessions
    ]


# --------------------------------------------------
# LOAD CHAT HISTORY (READ-ONLY)
# --------------------------------------------------

@app.get("/chat/{session_id}/history")
def get_chat_history(session_id: str):
    history = store.load_history(session_id)
    return history


# --------------------------------------------------
# CHAT (NON-STREAM)
# --------------------------------------------------

@app.post("/chat")
def chat(req: ChatRequest):
    answer = run_rag(
        query=req.question,
        session_id=req.session_id
    )
    return {"answer": answer}


# --------------------------------------------------
# CHAT (STREAM)
# --------------------------------------------------

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):

    def token_stream():
        for token in run_rag_stream(
            query=req.question,
            session_id=req.session_id
        ):
            yield token

    return StreamingResponse(
        token_stream(),
        media_type="text/plain"
    )


# --------------------------------------------------
# DELETE SESSION (EXPLICIT ONLY)
# --------------------------------------------------

@app.delete("/chat/session/{session_id}")
def delete_session(session_id: str):
    deleted = store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    # NOTE:
    # Vector DB cleanup can be added here if desired
    return {"status": "deleted"}


# --------------------------------------------------
# FRONTEND
# --------------------------------------------------

app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend"
)
