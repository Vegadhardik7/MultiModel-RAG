import os
import json
import requests
from dotenv import load_dotenv
from typing import Generator

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


def generate(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """
    Non-streaming generation (already working).
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
        "stream": False,
    }

    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"]


def generate_stream(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> Generator[str, None, None]:
    """
    Streaming generation using Ollama.
    Yields tokens as they arrive.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
        "stream": True,
    }

    with requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()

        for line in r.iter_lines():
            if not line:
                continue

            data = json.loads(line.decode("utf-8"))

            # Ollama sends partial tokens in "response"
            if "response" in data:
                yield data["response"]

            # Stop when done
            if data.get("done", False):
                break


if __name__ == "__main__":
    print("Streaming output:\n")

    for token in generate_stream("Explain RAG in one sentence."):
        print(token, end="", flush=True)

    print("\n\n--- done ---")
