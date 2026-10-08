import os

import requests


class OllamaLLM:
    server = "local"

    def __init__(self, model="gemma4:e2b"):
        self.model = model
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")

    def generate(self, messages):
        r = requests.post(
            self.url,
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=120,
        )

        r.raise_for_status()
        return r.json()["message"]["content"]
