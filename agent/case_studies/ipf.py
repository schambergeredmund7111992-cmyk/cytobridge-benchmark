"""
agent/case_studies/ipf.py
-------------------------
IPF (Idiopathic Pulmonary Fibrosis) drug repurposing case study.

Data: GSE135893 (Adams et al. 2020) — IPF patient single-cell RNA-seq, ~300K cells,
      32 IPF samples + 28 controls, multiple lung cell types.

Goal: CytoReasoner should rank
    - nintedanib (FDA approved, Boehringer 2014)
    - pirfenidone (FDA approved, 2014)
    in top-10 candidates from a pool of ~50 drugs (LINCS L1000 screen).
    PLUS propose 1 novel candidate with PubMed-verifiable mechanism.

Run:
    export LLM_PROVIDER=kimi_parasail
    export KIMI_PARASAIL_API_KEY=...
    python agent/case_studies/ipf.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.reasoner import CytoReasoner, ReasonerConfig  # noqa: E402
from agent.tools import ToolRegistry  # noqa: E402


def summarize_cell_state(
    adata,
    cluster: str = "fibroblast",
    cell_type_col: str = "cell_type",
    condition_col: str = "condition",
    disease_label: str = "IPF",
    control_label: str = "control",
    top_k: int = 30,
) -> str:
    """Compute top dysregulated genes (IPF vs control) within a cell type."""
    sub = adata[adata.obs[cell_type_col] == cluster]
    ipf = sub[sub.obs[condition_col] == disease_label].X.mean(axis=0)
    ctl = sub[sub.obs[condition_col] == control_label].X.mean(axis=0)
    if hasattr(ipf, "A1"):
        ipf, ctl = ipf.A1, ctl.A1
    logfc = np.asarray(ipf - ctl).flatten()
    top = np.argsort(-np.abs(logfc))[:top_k]
    genes = sub.var_names[top].tolist()
    direction = ["up" if logfc[i] > 0 else "down" for i in top]
    return (
        f"Cell type: {cluster}; disease label: {disease_label}; control label: {control_label}.\n"
        f"Top {top_k} dysregulated genes (disease vs control):\n"
        + "\n".join(f"  {g} ({d})" for g, d in zip(genes, direction))
    )


def validate_case_inputs(args: argparse.Namespace) -> None:
    missing: list[str] = []
    required_paths = {
        "--adata": args.adata,
        "--msigdb": args.msigdb,
        "--ckpt": args.ckpt,
        "--precomputed_cell_embs": args.precomputed_cell_embs,
        "--cell_state_tokens": args.cell_state_tokens,
    }
    for flag, path in required_paths.items():
        if path is None or not Path(path).exists():
            missing.append(f"{flag}={path}")
    for rel in ("lincs_emb.npy", "lincs_meta.csv"):
        path = args.lincs_dir / rel
        if not path.exists():
            missing.append(f"--lincs_dir missing {rel}: {path}")
    if missing:
        raise FileNotFoundError(
            "Case-study inputs are not ready; refusing to call the LLM with "
            "tools that would only return errors. Missing:\n  - "
            + "\n  - ".join(missing)
            + "\nGenerate these caches first, or pass --skip_preflight only for "
            "deliberate dry-run debugging."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, default=Path("data/raw/ipf/GSE135893_processed.h5ad"))
    parser.add_argument("--lincs_dir", type=Path, default=Path("data/raw/lincs"))
    parser.add_argument("--msigdb", type=Path,
                        default=Path("data/raw/msigdb/h.all.v2024.1.Hs.symbols.gmt"))
    parser.add_argument("--ckpt", type=Path, default=Path("ckpts/v1/last.ckpt"))
    parser.add_argument("--candidates_csv", type=Path,
                        default=Path("data/raw/ipf/candidate_drugs.csv"))
    parser.add_argument("--precomputed_cell_embs", type=Path,
                        default=Path("data/cache/ipf_pooled_cell_embs.npz"))
    parser.add_argument("--cell_state_tokens", type=Path,
                        default=Path("data/cache/ipf_cell_state_tokens.npz"))
    parser.add_argument("--gene_names", type=Path, default=Path("data/cache/sciplex_gene_names.csv"))
    parser.add_argument("--pathway_names", type=Path,
                        default=Path("data/cache/msigdb_hallmark_pathways.csv"))
    parser.add_argument("--molformer_cache_dir", type=Path, default=Path("data/cache/molformer"))
    parser.add_argument("--skip_preflight", action="store_true")
    parser.add_argument("--cluster", default="fibroblast")
    parser.add_argument("--cell_type_col", default="cell_type")
    parser.add_argument("--condition_col", default="condition")
    parser.add_argument("--disease_label", default="IPF")
    parser.add_argument("--control_label", default="control")
    parser.add_argument("--out", type=Path, default=Path("results/case_studies/ipf.json"))
    args = parser.parse_args()

    if not args.skip_preflight:
        validate_case_inputs(args)

    import scanpy as sc

    print("[ipf] loading data ...")
    adata = sc.read_h5ad(args.adata)
    summary = summarize_cell_state(
        adata,
        cluster=args.cluster,
        cell_type_col=args.cell_type_col,
        condition_col=args.condition_col,
        disease_label=args.disease_label,
        control_label=args.control_label,
        top_k=30,
    )
    print(summary)
    if args.candidates_csv.exists():
        candidates = pd.read_csv(args.candidates_csv)["drug_name"].tolist()
    else:
        candidates = ["nintedanib", "pirfenidone", "trametinib", "imatinib", "metformin"]

    registry = ToolRegistry(
        lincs_dir=args.lincs_dir,
        msigdb_gmt=args.msigdb,
        cytobridge_ckpt=args.ckpt,
        precomputed_cell_embs=args.precomputed_cell_embs,
        cell_state_tokens_path=args.cell_state_tokens,
        gene_names_path=args.gene_names,
        pathway_names_path=args.pathway_names,
        molformer_cache_dir=args.molformer_cache_dir,
    )
    reasoner = CytoReasoner(registry, cfg=ReasonerConfig())
    result = reasoner.reason(
        cell_state_summary=summary,
        candidate_drugs=candidates,
        cell_state_id="IPF_fibroblast",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2, default=str)
    print(f"\n[ipf] saved -> {args.out}")
    if result.get("ranking"):
        print("\nTop-10 ranked candidates:")
        for r in result["ranking"].get("ranked_candidates", [])[:10]:
            print(f"  #{r['rank']:2d}  {r['drug']:20s}  conf={r['confidence']}")


if __name__ == "__main__":
    main()
