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


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
SESSIONS_DIR = BASE_DIR / "data" / "sessions"
FRONTEND_DIR = BASE_DIR / "frontend"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()


# --------------------------------------------------
# MODELS
# --------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    question: str


# --------------------------------------------------
# PDF UPLOAD → NEW SESSION
# --------------------------------------------------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Uploading a PDF automatically:
    - creates a new session
    - creates an isolated vector store
    - clears previous conversation on frontend
    """

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    # ✅ New session
    session_id = str(uuid.uuid4())

    # ✅ Session-specific storage
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Save PDF
    pdf_path = UPLOAD_DIR / file.filename
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ✅ Initialize isolated vector collection
    init_session_collection(session_id)

    # ✅ Ingest PDF into that session’s vector DB
    ingest_multimodal_pdf(str(pdf_path), session_id=session_id)

    return {
        "message": f"{file.filename} ingested successfully",
        "session_id": session_id
    }


# --------------------------------------------------
# CHAT (NON-STREAMING)
# --------------------------------------------------

@app.post("/chat")
def chat(req: ChatRequest):
    answer = run_rag(
        query=req.question,
        session_id=req.session_id
    )
    return {"answer": answer}


# --------------------------------------------------
# CHAT (STREAMING)
# --------------------------------------------------

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):

    def event_generator():
        for token in run_rag_stream(
            query=req.question,
            session_id=req.session_id
        ):
            yield token

    return StreamingResponse(
        event_generator(),
        media_type="text/plain"
    )


# --------------------------------------------------
# FRONTEND
# --------------------------------------------------

app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend"
)
