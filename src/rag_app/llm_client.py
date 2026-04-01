import os
from dotenv import load_dotenv
import requests
from openai import OpenAI

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

OLLAMA_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def generate(prompt, model=None, provider=None):
    provider = (provider or PROVIDER).lower()
    print(f"Generating with provider '{provider}'...")
    
    if provider == "ollama":
        return _generate_ollama(prompt, model or OLLAMA_MODEL)

    if provider == "openai":
        return _generate_openai(prompt, model or OPENAI_MODEL)

    raise ValueError(f"Unknown LLM provider: {provider}")


def _generate_ollama(prompt, model=None):
    print(f"Generating with Ollama (model={model})...")
    res = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    res.raise_for_status()
    return res.json()["response"]


def _generate_openai(prompt, model):
    if openai_client is None:
        raise ValueError("OPENAI_API_KEY is not set")

    res = openai_client.responses.create(
        model=model,
        input=prompt,
    )
    return res.output_text



def stream_generate(prompt, model="llama3"):
    print(os.getenv("OLLAMA_API_URL"))

    res = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,
        },
        stream=True,
    )
    res.raise_for_status()

    for line in res.iter_lines():
        if not line:
            continue

        chunk = json.loads(line)
        if chunk.get("response"):
            yield chunk["response"]

        if chunk.get("done"):
            break