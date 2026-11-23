"""Tests run without a real model: we replace the HTTP call with a fake."""

import pytest
import requests

from private_docs_rag.llm.client import LLMClient, LLMError, Message


class FakeResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_chat_returns_reply_and_sends_correct_payload(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse({"choices": [{"message": {"content": "Hello!"}}]})

    monkeypatch.setattr(requests, "post", fake_post)

    client = LLMClient(base_url="http://localhost:11434/v1/", model="test-model")
    reply = client.chat([Message("user", "Hi")])

    assert reply == "Hello!"
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["json"]["model"] == "test-model"
    assert captured["json"]["messages"] == [{"role": "user", "content": "Hi"}]


def test_ask_puts_system_prompt_first(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["messages"] = json["messages"]
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(requests, "post", fake_post)

    LLMClient().ask("Question?", system="Be brief.")

    assert captured["messages"][0] == {"role": "system", "content": "Be brief."}
    assert captured["messages"][1]["role"] == "user"


def test_server_down_raises_clear_error(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(LLMError, match="Could not reach model server"):
        LLMClient().ask("Hi")


def test_bad_response_shape_raises(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse({"error": "oops"})
    )

    with pytest.raises(LLMError, match="Unexpected response shape"):
        LLMClient().ask("Hi")
