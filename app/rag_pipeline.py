from typing import List, Generator
from app.retriever import retrieve
from app.llm.ollama_client import generate, generate_stream
from app.memory.chat_memory import ChatMemory

_memory = ChatMemory()


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


def build_retrieval_query(
    query: str,
    history: str | None
) -> str:
    """
    Use conversation history to make retrieval aware
    of previous context for follow-up questions.
    """
    if not history:
        return query

    return f"{history}\nUser question: {query}"

def run_rag(query: str, k: int = 5) -> str:
    _memory.add_user(query)

    history = _memory.get_context()
    retrieval_query = build_retrieval_query(query, history)

    docs = retrieve(retrieval_query, k=k)

    print("\n[DEBUG]")
    print("Retrieval query:")
    print(retrieval_query)
    print("Retrieved docs count:", len(docs))


    if not docs:
        answer = "I don't know."
        _memory.add_assistant(answer)
        return answer

    prompt = build_prompt(docs, query, history)
    answer = generate(prompt)

    _memory.add_assistant(answer)
    return answer



def run_rag_stream(query: str, k: int = 5):
    _memory.add_user(query)

    history = _memory.get_context()
    retrieval_query = build_retrieval_query(query, history)

    docs = retrieve(retrieval_query, k=k)

    if not docs:
        answer = "I don't know."
        _memory.add_assistant(answer)
        yield answer
        return

    prompt = build_prompt(docs, query, history)

    final_answer = ""
    for token in generate_stream(prompt):
        final_answer += token
        yield token

    _memory.add_assistant(final_answer)


# ---------------------------------------------------
# 🔹 MEMORY TEST (LOCAL RUN)
# ---------------------------------------------------
if __name__ == "__main__":
    questions = [
        "What is spinal cord regeneration?",
        "Are there any diagrams related to it?",
        "Explain that simply."
    ]

    for q in questions:
        print("\n==============================")
        print(f"USER: {q}")
        print("ASSISTANT:")

        answer = run_rag(q)
        print(answer)