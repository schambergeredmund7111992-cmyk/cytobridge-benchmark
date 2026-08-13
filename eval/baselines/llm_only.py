"""
eval/baselines/llm_only.py
--------------------------
"Pure LLM" baseline: ask an open-source/open-weight LLM provider to rank drugs given only:
    - cell-state summary (top-30 dysregulated genes + cell type)
    - candidate drug list
NO CytoBridge predictions. Tests whether the bridge is even necessary.

Expected: this baseline performs OK on well-known drugs (literature memorized)
but poorly on novel drugs / novel cell types. CytoBridge should win there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agent.llm_client import get_llm_client


SYSTEM = """You are a pharmacologist. Given a disease cell state and candidate drugs,
rank the drugs by likelihood of reverting disease phenotype. Return JSON only:
{"ranked_drugs": [{"drug": "...", "rank": 1, "reasoning": "..."}, ...]}"""


def rank_drugs(cell_summary: str, drugs: list[str], provider: str | None = None):
    client, model, _ = get_llm_client(provider)
    user = f"Cell state:\n{cell_summary}\n\nCandidates: {drugs}"
    resp = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or ""
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_summary_csv", type=Path, required=True,
                        help="csv: cell_state_id, summary, candidate_drugs (json list), gold_drug")
    parser.add_argument("--out", type=Path, default=Path("results/llm_only.csv"))
    parser.add_argument("--provider", default=None, help="defaults to LLM_PROVIDER or kimi_parasail")
    args = parser.parse_args()

    df = pd.read_csv(args.cell_summary_csv)
    rows = []
    for _, r in df.iterrows():
        result = rank_drugs(r["summary"], json.loads(r["candidate_drugs"]), provider=args.provider)
        ranked = result["ranked_drugs"]
        gold = r["gold_drug"]
        rank = next((x["rank"] for x in ranked if x["drug"] == gold), -1)
        rows.append({"cell_state_id": r["cell_state_id"], "gold_drug": gold,
                     "rank_of_gold": rank})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
