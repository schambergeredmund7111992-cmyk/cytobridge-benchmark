from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.llm_client import (
    PROVIDERS,
    chat_completion_with_retry,
    extract_usage_tokens,
    parse_tool_arguments,
)


def test_provider_catalog_includes_openai():
    assert "openai" in PROVIDERS


def test_parse_tool_arguments_accepts_dict_and_json_string():
    assert parse_tool_arguments({"drug_smiles": "CCO"}) == {"drug_smiles": "CCO"}
    assert parse_tool_arguments('{"drug_smiles": "CCO"}') == {"drug_smiles": "CCO"}


def test_extract_usage_tokens_handles_openai_and_input_output_names():
    assert extract_usage_tokens(
        SimpleNamespace(usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7))
    ) == (11, 7)
    assert extract_usage_tokens({"usage": {"input_tokens": 13, "output_tokens": 5}}) == (13, 5)


def test_chat_completion_retries_429_then_returns(monkeypatch):
    sleeps = []
    monkeypatch.setattr("agent.llm_client.time.sleep", lambda seconds: sleeps.append(seconds))

    class RateLimit(Exception):
        status_code = 429

    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RateLimit("rate limited")
            return {"ok": True}

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    assert chat_completion_with_retry(client, max_retries=2, initial_backoff=0.25) == {"ok": True}
    assert sleeps == [0.25]
    assert completions.calls == 2


def test_chat_completion_raises_final_429(monkeypatch):
    monkeypatch.setattr("agent.llm_client.time.sleep", lambda seconds: None)

    class RateLimit(Exception):
        status_code = 429

    class FakeCompletions:
        def create(self, **kwargs):
            raise RateLimit("still rate limited")

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    with pytest.raises(RateLimit):
        chat_completion_with_retry(client, max_retries=1, initial_backoff=0)
