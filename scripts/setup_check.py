from dotenv import load_dotenv
import os
import requests

load_dotenv()

print("---- ENV CHECK ----")
print("OLLAMA_BASE_URL:", os.getenv("OLLAMA_BASE_URL"))
print("OLLAMA_MODEL:", os.getenv("OLLAMA_MODEL"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
MODEL = os.getenv("OLLAMA_MODEL")

print("\n---- OLLAMA CHECK ----")
payload = {
    "model": MODEL,
    "prompt": "Say hello in one short sentence.",
    "stream": False
}

try:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=30
    )
    r.raise_for_status()
    print("Ollama response:", r.json()["response"])
    print("\n✅ Ollama is working correctly")
except Exception as e:
    print("\n❌ Ollama check failed")
    print(e)
