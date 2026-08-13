#!/usr/bin/env python
"""G3-2: Drug ranking utility for IPF (Idiopathic Pulmonary Fibrosis) case study.

Ranks test+val drugs by predicted IPF disease reversal potential using the
model's pathway attention and predicted gene expression changes.

IPF is driven by TGF-beta-induced fibroblast activation, EMT, and chronic
inflammation. We define an IPF pathway signature and score each drug by its
predicted ability to reverse IPF-associated pathway activation.

Since gene-level overlap between sci-Plex HVGs and MSigDB is only 8.5%, we
operate at the pathway level using the model's 50 Hallmark pathway attentions.

Output:
  - Per-drug IPF reversal scores (ranked)
  - hit@k validation against known pharmacological mechanisms
  - Per-cell-line drug ranking consistency
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from cytobridge.model import CytoBridge, CytoBridgeConfig


# ---------------------------------------------------------------------------
# IPF pathway signature (expert-defined weights on 50 MSigDB Hallmark pathways)
# ---------------------------------------------------------------------------
# Indexed by pathway_names_computed.csv order
IPF_PATHWAY_WEIGHTS = {
    # Core fibrotic drivers
    "TGF-beta Signaling": 1.0,
    "Epithelial Mesenchymal Transition": 0.9,
    # Inflammatory
    "Inflammatory Response": 0.7,
    "TNF-alpha Signaling via NF-kB": 0.7,
    "IL-6/JAK/STAT3 Signaling": 0.6,
    "Interferon Gamma Response": 0.4,
    "Interferon Alpha Response": 0.3,
    "Complement": 0.3,
    "IL-2/STAT5 Signaling": 0.2,
    # Tissue remodeling / fibrosis
    "Hypoxia": 0.6,
    "Apoptosis": 0.5,
    "Coagulation": 0.3,
    "Angiogenesis": 0.3,
    # Proliferation / survival
    "PI3K/AKT/mTOR Signaling": 0.5,
    "mTORC1 Signaling": 0.3,
    "Wnt-beta Catenin Signaling": 0.4,
    "Notch Signaling": 0.3,
    "Hedgehog Signaling": 0.3,
    # Stress / damage
    "Reactive Oxygen Species Pathway": 0.4,
    "Unfolded Protein Response": 0.2,
    "DNA Repair": 0.1,
    "p53 Pathway": 0.1,
    # Metabolic (lower relevance)
    "Glycolysis": 0.2,
    "Oxidative Phosphorylation": 0.1,
    "Fatty Acid Metabolism": 0.1,
    "Cholesterol Homeostasis": 0.1,
    "Xenobiotic Metabolism": 0.1,
    "Bile Acid Metabolism": 0.1,
    "Heme Metabolism": 0.1,
}


# ---------------------------------------------------------------------------
# Known drug mechanism annotations for hit@k validation
# ---------------------------------------------------------------------------
KNOWN_MECHANISMS = {
    # Test drugs
    "AG-490 (Tyrphostin B42)": {
        "mechanism": "JAK2 inhibitor",
        "ipf_relevance": 2,  # 2=strong, 1=plausible, 0=unlikely
        "rationale": "JAK/STAT3 blockade reduces TGFb-induced fibrosis in preclinical IPF models.",
    },
    "Tofacitinib (CP-690550) Citrate": {
        "mechanism": "JAK1/3 inhibitor",
        "ipf_relevance": 2,
        "rationale": "Pan-JAK inhibition attenuates bleomycin-induced pulmonary fibrosis. Phase II trial in IPF.",
    },
    "Celecoxib": {
        "mechanism": "COX-2 inhibitor",
        "ipf_relevance": 1,
        "rationale": "COX-2/PGE2 axis has complex role in IPF; mixed preclinical evidence.",
    },
    "Thalidomide": {
        "mechanism": "Immunomodulatory (anti-TNFa, anti-TGFb)",
        "ipf_relevance": 1,
        "rationale": "Reduces TGFb1 and TNFa. Case reports for IPF-related cough.",
    },
    "Zileuton": {
        "mechanism": "5-lipoxygenase inhibitor",
        "ipf_relevance": 1,
        "rationale": "Leukotrienes promote fibrosis; 5-LO inhibition is antifibrotic in mouse models.",
    },
    "SL-327": {
        "mechanism": "MEK1/2 inhibitor",
        "ipf_relevance": 1,
        "rationale": "ERK/MAPK mediates TGFb-induced EMT. MEK inhibition blocks myofibroblast differentiation.",
    },
    "Fulvestrant": {
        "mechanism": "ER antagonist",
        "ipf_relevance": 0,
        "rationale": "Estrogen signaling not a primary IPF driver. Limited relevance.",
    },
    "Ramelteon": {
        "mechanism": "Melatonin receptor agonist",
        "ipf_relevance": 0,
        "rationale": "No established role in IPF treatment.",
    },
    "SRT3025 HCl": {
        "mechanism": "SIRT1 activator",
        "ipf_relevance": 1,
        "rationale": "SIRT1 attenuates TGFb-induced fibrosis. Emerging antifibrotic target.",
    },
    # Val drugs
    "Vandetanib (ZD6474)": {
        "mechanism": "VEGFR/EGFR inhibitor",
        "ipf_relevance": 2,
        "rationale": "Same TKI class as nintedanib (approved IPF drug targeting VEGFR/PDGFR/FGFR).",
    },
    "Temsirolimus (CCI-779, NSC 683864)": {
        "mechanism": "mTOR inhibitor",
        "ipf_relevance": 1,
        "rationale": "mTOR pathway activated in IPF fibroblasts. Rapamycin reduces fibrosis.",
    },
    "ABT-737": {
        "mechanism": "Bcl-2/Bcl-xL inhibitor",
        "ipf_relevance": 1,
        "rationale": "Myofibroblast apoptosis resistance drives IPF. BH3 mimetics show antifibrotic activity.",
    },
    "Veliparib (ABT-888)": {
        "mechanism": "PARP inhibitor",
        "ipf_relevance": 1,
        "rationale": "PARP1 contributes to TGFb/Smad signaling. PARP inhibition reduces fibrosis.",
    },
    "SRT1720 HCl": {
        "mechanism": "SIRT1 activator",
        "ipf_relevance": 1,
        "rationale": "Same target as SRT3025. SIRT1 agonism attenuates fibrosis.",
    },
    "Sodium Phenylbutyrate": {
        "mechanism": "HDAC inhibitor / chemical chaperone",
        "ipf_relevance": 1,
        "rationale": "HDAC inhibitors reduce myofibroblast activation and collagen production.",
    },
    "Amisulpride": {
        "mechanism": "D2/D3 antagonist",
        "ipf_relevance": 0,
        "rationale": "Dopamine antagonist. No known IPF relevance.",
    },
    "Tie2 kinase inhibitor": {
        "mechanism": "Tie2 (TEK) inhibitor",
        "ipf_relevance": 0,
        "rationale": "Angiopoietin/Tie2 pathway primarily vascular. Limited IPF data.",
    },
}


def load_model(ckpt_path: str, device: str = "cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    hp = ckpt["hyper_parameters"]
    cfg = CytoBridgeConfig(**hp["model_cfg"])
    model = CytoBridge(cfg)
    state = {
        k.replace("model.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def predict_per_cell(model, item: dict, device: str) -> dict:
    """Return dict with delta [G], pathway_attn [K] for one cell."""
    with torch.no_grad():
        out = model(
            cell_tokens=item["cell_tokens"].unsqueeze(0).to(device),
            drug_tokens=item["drug_tokens"].unsqueeze(0).to(device),
            drug_mask=item["drug_mask"].unsqueeze(0).to(device),
            control_counts=item.get("control_counts").unsqueeze(0).to(device),
        )
    result = {}
    delta = out.get("delta")
    if delta is not None:
        result["delta"] = delta.squeeze(0).cpu().numpy()
    else:
        mu = out["mu"].squeeze(0).cpu().numpy()
        ctrl = item["control_counts"].squeeze(0).numpy()
        result["delta"] = np.log1p(np.maximum(mu, 0)) - np.log1p(np.maximum(ctrl, 0))
    pw_attn = out.get("pathway_attn")
    result["pathway_attn"] = pw_attn.squeeze(0).cpu().numpy() if pw_attn is not None else None
    return result


def compute_ipf_reversal(pathway_attn: np.ndarray, pathway_names: list[str],
                         pathway_weights: dict) -> float:
    """Compute IPF reversal score from pathway attention.

    Score = sum_k(w_k * (1/K - attn_k))  — pathways active in IPF should be
    suppressed by an effective drug (attn_k -> 0, score positive).
    A high score means the drug strongly reverses IPF pathway activation.
    """
    K = len(pathway_names)
    score = 0.0
    total_w = 0.0
    for k, pw_name in enumerate(pathway_names):
        w = pathway_weights.get(pw_name, 0.0)
        if w > 0:
            # Reversal: baseline = 1/K (uniform attention), reversal = 1/K - attn_k
            reversal_k = (1.0 / K) - pathway_attn[k]
            score += w * reversal_k
            total_w += w
    if total_w > 0:
        score /= total_w
    return float(score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/v1/epoch19-val_loss325.2686.ckpt")
    ap.add_argument("--splits_dir", default="data/processed/sciplex/splits")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="results/drug_ranking_ipf")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = Path(args.splits_dir)

    # ---- Load pathway names ----
    pathway_names_df = pd.read_csv(splits_dir / "pathway_names_computed.csv")
    pathway_names = pathway_names_df["pathway"].tolist()
    K = len(pathway_names)
    print(f"Pathway count: {K}")

    # ---- Print IPF signature ----
    print("\nIPF Pathway Signature:")
    for pw_name, w in sorted(IPF_PATHWAY_WEIGHTS.items(), key=lambda x: -x[1]):
        if pw_name in pathway_names:
            print(f"  {pw_name:40s} weight={w:.1f}")
    n_ipf_pw = sum(1 for pw in IPF_PATHWAY_WEIGHTS if pw in pathway_names)
    print(f"  Total: {n_ipf_pw} IPF-relevant pathways")

    # ---- Load model ----
    print(f"\nLoading model from {args.ckpt}...")
    model = load_model(args.ckpt, args.device)
    print("Model loaded")

    # ---- Process test + val splits ----
    from cytobridge.data import CytoBridgeDataset

    cell_lines = ["A549", "K562", "MCF7"]
    all_scores = []

    for split_name in ["val", "test"]:
        print(f"\nProcessing {split_name} split...")
        ds = CytoBridgeDataset(
            manifest_path=str(splits_dir / f"sciplex_{split_name}.parquet"),
            cell_emb_path="data/cache/sciplex_scgpt_emb.npy",
            drug_emb_path="data/cache/sciplex_molformer_emb.npz",
            treated_counts_path=str(splits_dir / f"sciplex_{split_name}_treated_counts.npy"),
            pathway_gsea_path=str(splits_dir / f"sciplex_{split_name}_pathway_gsea.npy"),
            control_counts_path=str(splits_dir / f"sciplex_{split_name}_control_counts.npy"),
            n_hard_same_drug=0, n_hard_same_cell=0,
        )

        man = pd.read_parquet(splits_dir / f"sciplex_{split_name}.parquet").reset_index(drop=True)
        for (drug, cl), idx in man.groupby(["drug_id", "cell_line"]).groups.items():
            take = np.asarray(list(idx), dtype=int)

            # Collect predictions
            all_deltas = []
            all_attns = []
            for i in take:
                item = ds._get_one(int(i))
                pred = predict_per_cell(model, item, args.device)
                all_deltas.append(pred["delta"])
                if pred["pathway_attn"] is not None:
                    all_attns.append(pred["pathway_attn"])

            pb_delta = np.mean(all_deltas, axis=0)
            pb_attn = np.mean(all_attns, axis=0) if all_attns else np.ones(K) / K

            # Compute IPF reversal score
            reversal_score = compute_ipf_reversal(pb_attn, pathway_names, IPF_PATHWAY_WEIGHTS)

            # Also compute gene-level reversal for IPF-relevant pathways
            # Mean delta across all genes (negative = antifibrotic direction)
            gene_reversal = -float(np.mean(pb_delta))

            mech = KNOWN_MECHANISMS.get(str(drug), {})
            all_scores.append({
                "drug": str(drug),
                "cell_line": str(cl),
                "split": split_name,
                "ipf_reversal_score": reversal_score,
                "gene_reversal_score": gene_reversal,
                "delta_mean": float(np.mean(pb_delta)),
                "delta_std": float(np.std(pb_delta)),
                "mechanism": mech.get("mechanism", "unknown"),
                "ipf_relevance": mech.get("ipf_relevance", 0),
                "rationale": mech.get("rationale", ""),
            })

    # ---- Aggregate across cell lines ----
    scores_df = pd.DataFrame(all_scores)
    drug_agg = scores_df.groupby("drug").agg(
        ipf_reversal_score=("ipf_reversal_score", "mean"),
        gene_reversal_score=("gene_reversal_score", "mean"),
        delta_mean=("delta_mean", "mean"),
        mechanism=("mechanism", "first"),
        ipf_relevance=("ipf_relevance", "first"),
        rationale=("rationale", "first"),
        n_cell_lines=("cell_line", "count"),
    ).reset_index()

    # Sort by IPF reversal score (higher = more antifibrotic)
    drug_agg = drug_agg.sort_values("ipf_reversal_score", ascending=False).reset_index(drop=True)
    drug_agg["rank"] = range(1, len(drug_agg) + 1)

    # ---- Print ranking table ----
    print("\n" + "=" * 100)
    print("  IPF DRUG REVERSAL RANKING  (higher score = stronger predicted antifibrotic effect)")
    print("=" * 100)
    print(f"{'Rank':<5} {'Drug':<40s} {'Score':>8s} {'Gene Rev':>8s} {'Relev':>6s} {'Mechanism'}")
    print("-" * 100)
    for _, row in drug_agg.iterrows():
        rel_label = {0: "--", 1: "+", 2: "++"}.get(int(row["ipf_relevance"]), "??")
        print(f"{row['rank']:<5} {row['drug']:<40s} {row['ipf_reversal_score']:8.4f} "
              f"{row['gene_reversal_score']:8.4f} {rel_label:>6s} {row['mechanism']}")

    # ---- hit@k validation ----
    print("\n=== HIT@K VALIDATION (known IPF-relevant pharmacology) ===")
    known_positive = drug_agg[drug_agg["ipf_relevance"] >= 1]["drug"].tolist()
    known_strong = drug_agg[drug_agg["ipf_relevance"] == 2]["drug"].tolist()
    print(f"Known positives (relevance >= 1): {len(known_positive)} drugs: {known_positive}")
    print(f"Strong positives (relevance = 2):  {len(known_strong)} drugs: {known_strong}")

    N = len(drug_agg)
    hit_results = {}
    for k in [1, 3, 5, 10]:
        top_k = drug_agg.head(k)["drug"].tolist()
        hit_any = len(set(top_k) & set(known_positive))
        hit_strong = len(set(top_k) & set(known_strong))
        precision_any = hit_any / k
        recall_any = hit_any / len(known_positive) if known_positive else 0
        print(f"  hit@{k}: any={hit_any} (P={precision_any:.2f}, R={recall_any:.2f}), "
              f"strong={hit_strong}")
        hit_results[f"hit@{k}_any"] = hit_any
        hit_results[f"hit@{k}_strong"] = hit_strong
        hit_results[f"precision@{k}"] = precision_any
        hit_results[f"recall@{k}"] = recall_any

    # ---- Per-cell-line consistency ----
    print("\n=== PER-CELL-LINE CONSISTENCY ===")
    for cl in cell_lines:
        cl_scores = scores_df[scores_df["cell_line"] == cl].copy()
        cl_scores = cl_scores.sort_values("ipf_reversal_score", ascending=False)
        cl_top5 = cl_scores.head(5)["drug"].tolist()
        print(f"  {cl}: top-5 = {cl_top5}")

    pivot = scores_df.pivot_table(
        values="ipf_reversal_score", index="drug", columns="cell_line"
    )
    for cl1 in cell_lines:
        for cl2 in cell_lines:
            if cl1 < cl2:
                shared = pivot[[cl1, cl2]].dropna()
                if len(shared) >= 5:
                    sr, sp = spearmanr(shared[cl1], shared[cl2])
                    print(f"  {cl1} vs {cl2}: Spearman r={sr:.4f} (n={len(shared)}, p={sp:.4f})")

    # ---- Top-5 drug details ----
    print("\n=== TOP-5 PREDICTED ANTIFIBROTIC DRUGS ===")
    for _, row in drug_agg.head(5).iterrows():
        drug = row["drug"]
        print(f"\n  Rank {row['rank']}: {drug}")
        print(f"    Mechanism: {row['mechanism']}")
        print(f"    IPF relevance: {row['ipf_relevance']}")
        print(f"    Reversal score: {row['ipf_reversal_score']:.6f}")
        print(f"    Rationale: {row['rationale']}")
        # Per-cell-line breakdown
        drug_cls = scores_df[scores_df["drug"] == drug]
        for _, cl_row in drug_cls.iterrows():
            print(f"    {cl_row['cell_line']:5s}: reversal={cl_row['ipf_reversal_score']:.6f}  "
                  f"delta_mean={cl_row['delta_mean']:.4f}")

    # ---- Save ----
    drug_agg.to_csv(out_dir / "ipf_drug_ranking.csv", index=False)
    scores_df.to_csv(out_dir / "ipf_per_cell_drug_scores.csv", index=False)

    summary = {
        "task": "G3-2 IPF drug ranking case study",
        "disease": "Idiopathic Pulmonary Fibrosis",
        "n_drugs_ranked": N,
        "top_5_drugs": drug_agg.head(5)["drug"].tolist(),
        "top_5_scores": [float(x) for x in drug_agg.head(5)["ipf_reversal_score"]],
        "known_positives": known_positive,
        "hit_results": hit_results,
    }
    with open(out_dir / "ipf_ranking_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}/")
    return summary


if __name__ == "__main__":
    main()
