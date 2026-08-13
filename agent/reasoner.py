"""CytoBridge LLM reasoner: multi-turn tool-calling for drug ranking."""
from __future__ import annotations

import json
from dataclasses import dataclass

from agent.tools import ToolRegistry, TOOL_DEFINITIONS
from agent.llm_client import (
    PROVIDERS,
    parse_tool_arguments,
    extract_usage_tokens,
    chat_completion_with_retry,
)


@dataclass
class ReasonerConfig:
    max_tool_iters: int = 3
    max_evidence_nudges: int = 1
    temperature: float = 0.0


SYSTEM_PROMPT = """You are a pharmacology reasoning assistant. You have access to five tools:
- retrieve_lincs_signatures: find similar drug perturbation signatures for a cell line
- query_drugbank: look up drug targets, mechanism of action, and indications
- get_pathway_info: get genes in an MSigDB Hallmark pathway
- predict_response: predict a drug's perturbation effect on a cell line
- search_pubmed: find literature evidence linking drugs to pathways or phenotypes

For each candidate drug, you MUST:
1. Query DrugBank for MOA/targets
2. Retrieve LINCS signatures for the cell line
3. Call predict_response for the drug in the cell line
4. Search PubMed for evidence linking the drug to relevant pathways

Then rank the candidates by predicted efficacy. Return your answer as JSON:
{"ranked_candidates": [{"drug": "...", "rank": 1, "confidence": "high|medium|low", "rationale": "..."}]}"""


class CytoReasoner:
    """Multi-turn LLM agent for drug ranking with tool-calling."""

    def __init__(
        self,
        registry: ToolRegistry,
        client=None,
        model: str = "gpt-4o",
        provider: str = "openai",
        cfg: ReasonerConfig | None = None,
    ):
        self.registry = registry
        self.client = client
        self.model = model
        self.provider_cfg = PROVIDERS.get(provider, PROVIDERS["openai"])
        self.cfg = cfg or ReasonerConfig()
        self._token_in = 0
        self._token_out = 0

    def reason(
        self,
        state_description: str,
        candidates: list[str],
        cell_line: str,
    ) -> dict:
        """Run multi-turn reasoning to rank drug candidates.

        Returns dict with keys:
            ranking: list[dict] | None  — ranked candidates
            error: str | None           — error description if ranking failed
            tokens_in: int
            tokens_out: int
            tool_calls_made: list[str]
        """
        if self.client is None:
            return {
                "ranking": None,
                "error": "No LLM client configured. Set reasoner.client or provide API key.",
                "tokens_in": 0,
                "tokens_out": 0,
                "tool_calls_made": [],
            }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Cell line: {cell_line}\n"
                f"Disease state: {state_description}\n"
                f"Candidate drugs to rank: {', '.join(candidates)}\n\n"
                f"Please use the tools to gather evidence, then rank these candidates."
            )},
        ]

        tool_defs = [
            {"type": "function", "function": t}
            for t in TOOL_DEFINITIONS
        ]

        evidence_tools_used: set[str] = set()
        nudges_remaining = self.cfg.max_evidence_nudges

        for _ in range(self.cfg.max_tool_iters):
            response = chat_completion_with_retry(
                self.client,
                model=self.model,
                messages=messages,
                tools=tool_defs,
                temperature=self.cfg.temperature,
            )

            tin, tout = extract_usage_tokens(response)
            self._token_in += tin
            self._token_out += tout

            msg = response.choices[0].message

            # If the model returned a text response (no tool calls)
            if msg.content and not msg.tool_calls:
                try:
                    parsed = json.loads(msg.content)
                    if "ranked_candidates" in parsed:
                        # Check evidence requirement
                        if "predict_response" not in evidence_tools_used:
                            if nudges_remaining > 0:
                                nudges_remaining -= 1
                                evidence_tools_used.add("predict_response")
                                messages.append({"role": "user", "content": (
                                    "Before ranking, you MUST call predict_response for each candidate. "
                                    "Please use the predict_response tool first."
                                )})
                                continue
                            return {
                                "ranking": None,
                                "error": (
                                    "predict_response tool not called before ranking. "
                                    "Evidence tools used: " + ", ".join(sorted(evidence_tools_used))
                                ),
                                "tokens_in": self._token_in,
                                "tokens_out": self._token_out,
                                "tool_calls_made": sorted(evidence_tools_used),
                            }
                        return {
                            "ranking": parsed["ranked_candidates"],
                            "error": None,
                            "tokens_in": self._token_in,
                            "tokens_out": self._token_out,
                            "tool_calls_made": sorted(evidence_tools_used),
                        }
                except json.JSONDecodeError:
                    pass

            # Process tool calls
            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ],
                })

                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = parse_tool_arguments(tc.function.arguments)
                    result = self.registry.call(name, **args)
                    evidence_tools_used.add(name)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str),
                    })
            else:
                # No tool calls and no parseable JSON
                messages.append({"role": "user", "content": (
                    "Please use the tools to gather evidence, then output your ranking as JSON."
                )})

        return {
            "ranking": None,
            "error": f"Exceeded max tool iterations ({self.cfg.max_tool_iters}) without ranking.",
            "tokens_in": self._token_in,
            "tokens_out": self._token_out,
            "tool_calls_made": sorted(evidence_tools_used),
        }
