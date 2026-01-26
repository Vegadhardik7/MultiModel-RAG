# app/rag_pipeline.py

from typing import List, Generator
from app.retriever import retrieve
from app.llm.ollama_client import generate, generate_stream
from memory.session_memory import SessionMemory


# -------------------------------------------------
# Prompt Builder
# -------------------------------------------------

def build_prompt(
    context_docs: List[dict],
    query: str,
    history: str | None
) -> str:
    """
    Strongly grounded prompt.
    Assumes the document EXISTS and is authoritative.
    """

    context = "\n\n".join(
        f"[Page {c.get('page', 'N/A')}] {c['text']}"
        for c in context_docs
    )

    return f"""
        You are an expert document analysis assistant.

        You are answering questions about a SINGLE uploaded document.
        This document has already been processed and indexed.

        Rules:
        - Use ONLY the provided context to answer.
        - If information is missing, infer carefully from context.
        - Do NOT say you lack access to the document.
        - Do NOT mention PDFs, uploads, or files unless explicitly asked.
        - Be concise, factual, and confident.

        Conversation history (for continuity only):
        {history or "None"}

        Document context:
        {context}

        User question:
        {query}

        Answer:
        """.strip()


# -------------------------------------------------
# Non-Streaming RAG
# -------------------------------------------------

def run_rag(
    query: str,
    session_id: str,
    k: int = 6
) -> str:
    """
    Deterministic RAG for one session (= one document)
    """

    memory = SessionMemory(session_id)

    # Save user message
    memory.add_user(query)

    # Retrieval MUST be scoped to session
    chunks = retrieve(
        query=query,
        session_id=session_id,
        k=k
    )

    if not chunks:
        answer = "The document does not contain information relevant to this question."
        memory.add_assistant(answer)
        return answer

    prompt = build_prompt(
        context_docs=chunks,
        query=query,
        history=memory.get_context()
    )

    answer = generate(prompt)

    memory.add_assistant(answer)
    return answer


# -------------------------------------------------
# Streaming RAG
# -------------------------------------------------

def run_rag_stream(
    query: str,
    session_id: str,
    k: int = 6
) -> Generator[str, None, None]:
    """
    Streaming RAG.
    Behavior must match run_rag exactly.
    """

    memory = SessionMemory(session_id)

    memory.add_user(query)

    chunks = retrieve(
        query=query,
        session_id=session_id,
        k=k
    )

    if not chunks:
        answer = "The document does not contain information relevant to this question."
        memory.add_assistant(answer)
        yield answer
        return

    prompt = build_prompt(
        context_docs=chunks,
        query=query,
        history=memory.get_context()
    )

    final_answer = ""
    for token in generate_stream(prompt):
        final_answer += token
        yield token

    memory.add_assistant(final_answer)
