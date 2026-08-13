from __future__ import annotations

import json
from types import SimpleNamespace

from agent.reasoner import CytoReasoner, ReasonerConfig
from agent.llm_client import PROVIDERS


def _message(content: str | None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _tool_call(name: str, args: dict):
    return SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _response(message):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
        choices=[SimpleNamespace(message=message)],
    )


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        return self._responses.pop(0)


class _FakeRegistry:
    def call(self, name: str, **kwargs):
        if name == "query_drugbank":
            return {"name": kwargs.get("drug_name_or_smiles", ""), "targets": ["T"]}
        raise AssertionError(f"unexpected tool call: {name}")


def test_reasoner_rejects_drugbank_only_evidence_without_predict_response():
    reasoner = CytoReasoner.__new__(CytoReasoner)
    reasoner.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=_FakeCompletions(
                [
                    _response(
                        _message(
                            None,
                            [_tool_call("query_drugbank", {"drug_name_or_smiles": "aspirin"})],
                        )
                    ),
                    _response(
                        _message(
                            json.dumps(
                                {
                                    "ranked_candidates": [
                                        {"drug": "aspirin", "rank": 1, "confidence": "high"}
                                    ]
                                }
                            )
                        )
                    ),
                ]
            )
        )
    )
    reasoner.model = "fake-model"
    reasoner.provider_cfg = PROVIDERS["kimi_parasail"]
    reasoner.registry = _FakeRegistry()
    reasoner.cfg = ReasonerConfig(max_tool_iters=2, max_evidence_nudges=0)
    reasoner._token_in = 0
    reasoner._token_out = 0

    out = reasoner.reason("state", ["aspirin"], "cell1")

    assert out["ranking"] is None
    assert "predict_response" in out["error"]
