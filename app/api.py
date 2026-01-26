import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from app.ingest.multimodal_pdf_ingest import ingest_multimodal_pdf
from app.rag_pipeline import run_rag, run_rag_stream
from memory.session_store import SessionStore

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# ---------------- CHAT SESSION STORE ----------------
store = SessionStore()

# ---------------- CHAT SESSION APIS ----------------

@app.post("/chat/new")
def new_chat():
    session_id = str(uuid.uuid4())
    store.create_session(session_id)
    return {"session_id": session_id}


@app.get("/chat/sessions")
def list_chats():
    return store.list_sessions()


# ---------------- CHAT APIS ----------------

class ChatRequest(BaseModel):
    session_id: str
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    answer = run_rag(req.question, session_id=req.session_id)
    return {"answer": answer}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    def event_generator():
        for token in run_rag_stream(req.question, session_id=req.session_id):
            yield token

    return StreamingResponse(
        event_generator(),
        media_type="text/plain"
    )

# ---------------- RETRIEVE IMG ----------------

from fastapi.responses import FileResponse

@app.get("/image")
def serve_image(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


# ---------------- PDF UPLOAD + INGEST ----------------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 🔥 Multimodal ingestion (text + tables + images + CLIP)
    ingest_multimodal_pdf(str(file_path))

    return {"message": f"{file.filename} ingested successfully"}


# ---------------- FRONTEND ----------------

# MUST be mounted LAST
app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend"
)