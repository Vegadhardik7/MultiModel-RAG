from typing import List, Generator
from app.retriever import retrieve
from app.llm.ollama_client import generate, generate_stream
from memory.session_memory import SessionMemory


# ---------------- PROMPT BUILDING ----------------
def build_prompt(
    context_docs: List[dict],
    query: str,
    history: str | None = None
) -> str:
    context = "\n\n".join(
        c["text"] for c in context_docs
    )

    history_block = f"\nConversation history:\n{history}\n" if history else ""

    prompt = f"""
        You are a precise assistant.
        Use the provided context to answer the question.
        If the answer is not in the context, say "I don't know".

        {history_block}

        Context:
        {context}

        Question:
        {query}

        Answer:
    """
    return prompt.strip()


def build_retrieval_query(query: str, history: str | None) -> str:
    if not history:
        return query
    return f"{history}\nUser question: {query}"


def format_citations(chunks: List[dict]) -> str:
    seen = set()
    citations = []
    idx = 1

    for c in chunks:
        key = (c.get("source"), c.get("page"))
        if key in seen:
            continue
        seen.add(key)

        citations.append(
            f"[{idx}] {c.get('source', 'unknown')} — page {c.get('page', 'unknown')}"
        )
        idx += 1

    return "\n".join(citations)


# ---------------- NON-STREAMING RAG ----------------
def run_rag(query: str, session_id: str, k: int = 5) -> str:
    memory = SessionMemory(session_id)

    memory.add_user(query)
    history = memory.get_context()

    retrieval_query = build_retrieval_query(query, history)
    chunks = retrieve(retrieval_query, k=k)

    if not chunks:
        answer = "I don't know."
        memory.add_assistant(answer)
        return answer

    prompt = build_prompt(chunks, query, history)
    answer = generate(prompt)

    citations = format_citations(chunks)
    final_answer = f"{answer}\n\nSources:\n{citations}"

    memory.add_assistant(final_answer)
    return final_answer


# ---------------- STREAMING RAG ----------------
def run_rag_stream(
    query: str,
    session_id: str,
    k: int = 5
) -> Generator[str, None, None]:

    memory = SessionMemory(session_id)

    memory.add_user(query)
    history = memory.get_context()

    retrieval_query = build_retrieval_query(query, history)
    chunks = retrieve(retrieval_query, k=k)

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
