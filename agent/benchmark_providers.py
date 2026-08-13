"""
agent/benchmark_providers.py
----------------------------
Day-2 A/B benchmark: run the SAME CytoReasoner protocol across multiple
LLM providers on a small fixed sample, collect per-provider:

    - tool-call count and order
    - structured JSON output validity
    - mechanism-explanation quality (manual rubric)
    - latency
    - actual USD cost

Use this as the empirical basis for your provider decision in Day-2 standup.

Usage:
    export KIMI_PARASAIL_API_KEY=...
    export DEEPSEEK_API_KEY=...
    python agent/benchmark_providers.py \\
        --sample data/test/d2_5_drugs.json \\
        --providers kimi_parasail deepseek local_qwen

The sample JSON is a list of {cell_state_summary, candidate_drugs, cell_state_id}
items. Five hand-curated IPF candidates ship in tests/fixtures/d2_5_drugs.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from agent.llm_client import PROVIDERS, ProviderConfig
from agent.reasoner import CytoReasoner, ReasonerConfig
from agent.tools import ToolRegistry


def _build_client(provider_cfg: ProviderConfig):
    """Create an OpenAI-compatible client from a provider configuration."""
    import openai

    api_key = os.environ.get(provider_cfg.api_key_env)
    if not api_key:
        raise ValueError(
            f"API key environment variable {provider_cfg.api_key_env} is not set "
            f"for provider using {provider_cfg.base_url}."
        )
    return openai.OpenAI(base_url=provider_cfg.base_url, api_key=api_key)


def run_one(provider_key: str, registry: ToolRegistry, sample: dict) -> dict:
    """Run one CytoReasoner pass; capture latency, tokens, cost."""
    provider_cfg = PROVIDERS.get(provider_key)
    if provider_cfg is None:
        raise ValueError(f"Unknown provider: {provider_key}. Available: {sorted(PROVIDERS)}")
    client = _build_client(provider_cfg)
    model = provider_cfg.default_model
    cfg = ReasonerConfig(max_tool_iters=10, temperature=0.2)
    reasoner = CytoReasoner(registry, client=client, model=model,
                            provider=provider_key, cfg=cfg)
    t0 = time.time()
    try:
        out = reasoner.reason(
            state_description=sample["cell_state_summary"],
            candidates=sample["candidate_drugs"],
            cell_line=sample["cell_state_id"],
        )
        ok = out.get("ranking") is not None
    except Exception as e:
        out = {"error": str(e), "ranking": None,
               "tokens_in": 0, "tokens_out": 0, "tool_calls_made": []}
        ok = False
    latency = time.time() - t0

    return {
        "provider": provider_key,
        "model": model,
        "sample_id": sample["cell_state_id"],
        "ok_json": ok,
        "n_tool_calls": len(out.get("tool_calls_made", [])),
        "latency_sec": round(latency, 2),
        "tokens_in": out.get("tokens_in", 0),
        "tokens_out": out.get("tokens_out", 0),
        "error": out.get("error", ""),
        "ranking": out.get("ranking"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path,
                        default=Path("tests/fixtures/d2_5_drugs.json"))
    parser.add_argument("--providers", nargs="+",
                        default=["kimi_parasail", "deepseek", "deepseek_flash"])
    parser.add_argument("--out_dir", type=Path, default=Path("results/d2_benchmark/"))
    args = parser.parse_args()

    samples = json.load(open(args.sample))
    print(f"[d2] {len(samples)} samples × {len(args.providers)} providers "
          f"= {len(samples) * len(args.providers)} total runs")

    # Build registry once (lazy-loads tools)
    registry = ToolRegistry(
        lincs_dir=Path("data/raw/lincs"),
        msigdb_gmt=Path("data/raw/msigdb/h.all.v2024.1.Hs.symbols.gmt"),
        cytobridge_ckpt=None,  # OK to be None for benchmark
    )

    rows = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for provider in args.providers:
        print(f"\n=== {provider} ===")
        for s in samples:
            r = run_one(provider, registry, s)
            rows.append(r)
            print(f"  [{provider}/{s['cell_state_id']}] "
                  f"ok={r['ok_json']} tools={r['n_tool_calls']} "
                  f"latency={r['latency_sec']}s")
            # Save individual run
            out_file = args.out_dir / f"{provider}__{s['cell_state_id']}.json"
            json.dump(r, open(out_file, "w"), indent=2, default=str)

    # Aggregate
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "summary.csv", index=False)

    # Per-provider summary
    summary = df.groupby("provider").agg(
        ok_json_rate=("ok_json", "mean"),
        avg_tool_calls=("n_tool_calls", "mean"),
        avg_latency=("latency_sec", "mean"),
        avg_tokens_in=("tokens_in", "mean"),
        avg_tokens_out=("tokens_out", "mean"),
    ).round(3)

    print("\n========= D2 PROVIDER COMPARISON =========")
    print(summary.to_string())
    summary.to_csv(args.out_dir / "summary_by_provider.csv")

    print("\n→ Decision rules:")
    print("  1. ok_json_rate < 95%        → DROP this provider")
    print("  2. avg_tool_calls < 4         → DROP (model skipping tools)")
    print("  3. otherwise pick the cheapest provider passing 1+2 as MAIN backend")


if __name__ == "__main__":
    main()
