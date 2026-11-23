"""Minimal client for a locally hosted LLM.

Works with any server that speaks the OpenAI-compatible Chat Completions API:
Ollama (http://localhost:11434/v1), vLLM, llama.cpp server, LM Studio.
Nothing leaves your machine: the base URL points at localhost by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import requests


class LLMError(RuntimeError):
    """Raised when the model server fails or returns something unexpected."""


@dataclass
class Message:
    role: str  # "system", "user" or "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMClient:
    base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    )
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3.2"))
    temperature: float = 0.0  # 0 = most deterministic; good for factual answers
    timeout: float = 120.0

    def chat(self, messages: list[Message]) -> str:
        """Send a conversation to the model and return its reply text."""
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.temperature,
        }
        url = f"{self.base_url.rstrip('/')}/chat/completions"

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach model server at {url}: {exc}") from exc

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape: {data}") from exc

    def ask(self, question: str, system: str | None = None) -> str:
        """Convenience wrapper for a single question."""
        messages = []
        if system:
            messages.append(Message("system", system))
        messages.append(Message("user", question))
        return self.chat(messages)
