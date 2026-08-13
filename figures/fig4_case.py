"""
figures/fig4_case.py
--------------------
Fig 4 — IPF + GBM case studies. Two side-by-side panels showing the
agent-ranked drug list for each disease, colour-coded by FDA approval status.

Inputs
    results/case_studies/ipf.json
    results/case_studies/gbm.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})


# Drugs with FDA approval status — used to colour markers.
FDA_APPROVED = {
    "ipf": {"nintedanib", "pirfenidone"},
    "gbm": {"temozolomide", "bevacizumab", "carmustine", "lomustine"},
}


def _load_case(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = json.load(open(path))
    ranking = raw.get("ranking", raw)
    if isinstance(ranking, dict):
        return ranking.get("ranked_candidates", [])
    if isinstance(ranking, list):
        return ranking
    return []


def _panel_case(ax, case_name: str, candidates: list[dict]) -> None:
    if not candidates:
        ax.text(0.5, 0.5, f"missing\nresults/case_studies/{case_name}.json",
                ha="center", va="center", transform=ax.transAxes,
                color="#999", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(case_name.upper(), fontsize=9)
        return
    rows = []
    for c in candidates[:10]:
        drug = (c.get("drug") or "?").lower()
        rows.append({
            "drug": drug,
            "rank": c.get("rank", len(rows) + 1),
            "confidence": c.get("confidence", "unknown"),
            "fda": drug in FDA_APPROVED.get(case_name, set()),
        })
    df = pd.DataFrame(rows).sort_values("rank")
    palette = {True: "#54A24B", False: "#4C78A8"}
    ax.barh(df["drug"], 11 - df["rank"],
            color=[palette[f] for f in df["fda"]], edgecolor="#222", linewidth=0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Rank score (10 − rank)")
    ax.set_title(f"{case_name.upper()} top-10 retrieved candidates", fontsize=9)
    n_fda = int(df["fda"].sum())
    ax.text(0.99, 0.02, f"{n_fda}/{len(df)} FDA-approved",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color="#54A24B")


def main() -> None:
    out_dir = Path("figures/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    _panel_case(axes[0], "ipf", _load_case(Path("results/case_studies/ipf.json")))
    _panel_case(axes[1], "gbm", _load_case(Path("results/case_studies/gbm.json")))
    fig.suptitle("Fig 4 — Drug-repurposing case studies (IPF, GBM)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_case.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig4_case.png", dpi=200, bbox_inches="tight")
    print(f"saved -> {out_dir}/fig4_case.{{pdf,png}}")


if __name__ == "__main__":
    main()
