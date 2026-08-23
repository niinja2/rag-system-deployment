import os
from openai import OpenAI

class OpenRouterLLM:
    server = "cloud"

    def __init__(self, model="mistralai/mistral-small-3.1-24b-instruct", temperature=0.2, max_tokens=500):
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, messages):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""
