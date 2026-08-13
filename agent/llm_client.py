"""LLM client helpers: provider catalog, retry logic, token accounting."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    base_url: str
    api_key_env: str
    default_model: str


PROVIDERS = {
    "openai": ProviderConfig(
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o",
    ),
    "kimi_parasail": ProviderConfig(
        base_url="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
        default_model="kimi-latest",
    ),
}


def parse_tool_arguments(args: dict | str) -> dict:
    if isinstance(args, dict):
        return args
    return json.loads(args)


def extract_usage_tokens(response) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from an LLM response.
    Handles both prompt_tokens/completion_tokens and input_tokens/output_tokens naming.
    """
    usage = response.usage if hasattr(response, "usage") else response.get("usage", {})
    if hasattr(usage, "prompt_tokens"):
        return (usage.prompt_tokens, usage.completion_tokens)
    if isinstance(usage, dict):
        pin = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
        pout = usage.get("completion_tokens") or usage.get("output_tokens", 0)
        return (pin, pout)
    return (0, 0)


def chat_completion_with_retry(
    client,
    max_retries: int = 2,
    initial_backoff: float = 0.25,
    **kwargs,
):
    """Call client.chat.completions.create with 429 retry logic."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status == 429 and attempt < max_retries:
                time.sleep(initial_backoff * (2 ** attempt))
                last_exc = exc
                continue
            raise
    raise last_exc  # type: ignore[misc]
