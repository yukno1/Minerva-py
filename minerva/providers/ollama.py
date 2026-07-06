from __future__ import annotations

from langchain_ollama import ChatOllama


def create_model(model: str, temperature: float = 0) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url="http://localhost:11434",
        temperature=temperature,
    )
