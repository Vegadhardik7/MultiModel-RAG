import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.ingest.pdf_ingest import ingest_pdf
from app.rag_pipeline import run_rag, run_rag_stream

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# ---------------- FRONTEND ----------------
# MUST be mounted before API routes
app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend"
)

# ---------------- API ----------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid file")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ingest_pdf(str(file_path))
    return {"message": f"{file.filename} ingested successfully"}


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Non-streaming chat (kept for debugging/testing)
    """
    return {"answer": run_rag(req.question)}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint (ChatGPT-style)
    """

    def event_generator():
        for token in run_rag_stream(req.question):
            yield token

    return StreamingResponse(
        event_generator(),
        media_type="text/plain"
    )
