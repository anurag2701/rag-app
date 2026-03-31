import os
import requests

OLLAMA_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

def generate(prompt, model="llama3"):
    res = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return res.json()["response"]


def chat(messages, model="llama3"):
    res = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False
        }
    )
    return res.json()["message"]["content"]