from typing import Callable
import ollama


def make_ollama_llm(model_name: str = "mistral") -> Callable[[str], str]:
    def _llm(prompt: str, max_tokens: int = 800) -> str:
        resp = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp["message"]["content"]
    return _llm