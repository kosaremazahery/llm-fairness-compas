import ollama

def llm(prompt: str, model: str = "mistral", max_tokens: int = 500) -> str:
    """
    Simple wrapper around a local Ollama model.
    """
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    # Ollama returns a dict with 'message': {'role': ..., 'content': ...}
    return response["message"]["content"]