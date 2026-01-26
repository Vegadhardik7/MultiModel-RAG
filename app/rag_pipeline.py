from typing import List, Generator
from app.retriever import retrieve
from app.llm.ollama_client import generate, generate_stream
from memory.session_memory import SessionMemory


# -------------------------------------------------
# Prompt
# -------------------------------------------------

def build_prompt(context_docs: List[dict], query: str, history: str | None):
    context = "\n\n".join(c["text"] for c in context_docs)

    return f"""
        You are an expert document understanding assistant.

        Your responsibilities:
        - Summarize documents when asked
        - Explain concepts from the document
        - Answer questions using the provided context
        - Reason even if the context is partial

        If the question is about the document as a whole,
        synthesize an answer instead of refusing.

        Only say "I don't know" if the context is truly unrelated.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """.strip()


# -------------------------------------------------
# Non-streaming RAG
# -------------------------------------------------

def run_rag(query: str, session_id: str, k: int = 6) -> str:
    memory = SessionMemory(session_id)

    memory.add_user(query)
    history = memory.get_context()

    chunks = retrieve(query, session_id=session_id, k=k)
    if not chunks:
        answer = "I don't know."
        memory.add_assistant(answer)
        return answer

    prompt = build_prompt(chunks, query, history)
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

    memory = SessionMemory(session_id)

    memory.add_user(query)
    history = memory.get_context()

    chunks = retrieve(query, session_id=session_id, k=k)
    if not chunks:
        answer = "I don't know."
        memory.add_assistant(answer)
        yield answer
        return

    prompt = build_prompt(chunks, query, history)

    final_answer = ""
    for token in generate_stream(prompt):
        final_answer += token
        yield token

    memory.add_assistant(final_answer)
