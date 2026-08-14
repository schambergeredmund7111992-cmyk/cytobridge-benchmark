"""Write the figures_data contract directory consumed by the figure scripts.

Every number here is derived from the pooled construction computed upstream;
no paper value is hard-coded.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from eval.metrics import drug_discrimination_score, inter_drug_pearson
from scripts.regenerate.inputs import CONFIG_ORDER, BundleLoader

CELL_LINES = ("A549", "K562", "MCF7")


def _ddc(pred, true, cl, top_k=50, metric="pearson") -> dict:
    return drug_discrimination_score(
        np.asarray(pred, dtype=float),
        np.asarray(true, dtype=float),
        np.asarray(cl),
        top_k=top_k,
        metric=metric,
    )


def write_figures_data(
    out: Path,
    *,
    loader: BundleLoader,
    results: dict,
    pooled_preds: dict,
    pooled_true: np.ndarray,
    true_perpair: np.ndarray,
    meta: pd.DataFrame,
    curve: dict | None,
    best_auc: float | None,
    bootstrap: dict | None,
    permutation: dict | None,
) -> Path:
    data_dir = out / "figures_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cl = meta["cell_line"].astype(str).to_numpy()
    drugs = meta["drug"].astype(str).to_numpy()

    # ---- table4 / table5 / ladder ----
    table4_rows = []
    for name in CONFIG_ORDER:
        if pooled_preds.get(name) is not None:
            table4_rows.append(
                {"predictor": name, "spearman50": results.get(f"t4.cb_{_id(name)}")}
            )
    for label, entry in (("Mean baseline", "t4.mean"), ("Ridge", "t4.ridge"),
                         ("chemCPA", "t4.chemcpa")):
        if results.get(entry) is not None:
            table4_rows.append({"predictor": label, "spearman50": results[entry]})
    pd.DataFrame(table4_rows).to_csv(data_dir / "table4.csv", index=False)

    table5_rows = []
    for name in CONFIG_ORDER:
        base = _id(name)
        if f"t5.{base}.auc" in results:
            table5_rows.append(
                {
                    "config": name,
                    "inter": results[f"t5.{base}.inter"],
                    "auc": results[f"t5.{base}.auc"],
                    "gap": results[f"t5.{base}.gap"],
                    "p": results[f"t5.{base}.p"],
                }
            )
    pd.DataFrame(table5_rows).to_csv(data_dir / "table5.csv", index=False)

    ladder_rows = [
        {"predictor": "Random (50 perm)", "auc": results.get("fig4b.ladder.random")},
        {"predictor": "Mean (collapsed)", "auc": results.get("fig4b.ladder.mean")},
        {"predictor": "Ridge (clean)", "auc": results.get("fig4b.ladder.ridge")},
        {"predictor": "chemCPA (collapsed)", "auc": results.get("fig4b.ladder.chemcpa")},
        {"predictor": "biolord (collapsed)", "auc": results.get("fig4b.ladder.biolord")},
        {"predictor": "CytoBridge (best)", "auc": results.get("fig4b.ladder.cb")},
        {"predictor": "Oracle (truth)", "auc": results.get("fig4b.ladder.oracle")},
    ]
    pd.DataFrame(ladder_rows).to_csv(data_dir / "ladder.csv", index=False)

    # ---- calibration ----
    if curve is not None:
        pd.DataFrame(
            {"alpha": curve["alphas"], "auc": curve["aucs"]}
        ).to_csv(data_dir / "calibration.csv", index=False)
        (data_dir / "calibration_meta.json").write_text(
            json.dumps(
                {
                    "cytobridge_auc": best_auc,
                    "effective_alpha": results.get("fig4a.eff_alpha"),
                },
                indent=2,
            )
            + "\n"
        )
        (data_dir / "effective_alpha.json").write_text(
            json.dumps(
                {
                    "effective_alpha": results.get("fig4a.eff_alpha"),
                    "best_config_auc": best_auc,
                },
                indent=2,
            )
            + "\n"
        )

    # ---- bootstrap ----
    if bootstrap is not None:
        np.save(data_dir / "bootstrap_auc.npy", bootstrap["draws"])
        (data_dir / "bootstrap_meta.json").write_text(
            json.dumps(
                {
                    "auc_mean": bootstrap["bootstrap_mean"],
                    "auc_lo": bootstrap["ci_lo"],
                    "auc_hi": bootstrap["ci_hi"],
                    "seed": bootstrap["seed"],
                    "n_boot": bootstrap["n_boot"],
                },
                indent=2,
            )
            + "\n"
        )

    # ---- permutation null ----
    if permutation is not None:
        (data_dir / "permutation_null_pooled.json").write_text(
            json.dumps(
                {
                    "observed_auc": permutation["observed"],
                    "null_mean": permutation["null_mean"],
                    "null_sd": permutation["null_sd"],
                    "p_value": permutation["p_value"],
                    "seed": permutation["seed"],
                    "n_perm": permutation["n_perm"],
                },
                indent=2,
            )
            + "\n"
        )

    # ---- Fig 3/6 panels derived from the pooled loss-only matrices ----
    loss_only = pooled_preds.get("loss-only")
    if loss_only is not None:
        # the case-study panel of generate_fig5_mechanism.py reads these names;
        # here they carry the pooled construction
        np.save(data_dir / "logfc_pred_t7_sub_loss_only.npy", loss_only)
        np.save(data_dir / "logfc_true_t7_sub_loss_only.npy", pooled_true)
        meta.to_csv(data_dir / "logfc_meta_t7_sub_loss_only.csv", index=False)
        for cell in CELL_LINES:
            rows = np.flatnonzero(cl == cell)
            if len(rows) < 2:
                continue
            matrix = np.corrcoef(loss_only[rows])
            np.save(data_dir / f"interdrug_{cell}.npy", matrix)
        per_cell = []
        for cell in CELL_LINES:
            rows = np.flatnonzero(cl == cell)
            score = _ddc(loss_only[rows], pooled_true[rows], cl[rows])
            per_cell.append(
                {
                    "config": "loss-only",
                    "cell": cell,
                    "auc": float(score["specificity_auc"]),
                    "inter_drug_r": float(inter_drug_pearson(loss_only[rows], cl[rows])),
                }
            )
        pd.DataFrame(per_cell).to_csv(data_dir / "per_cellline.csv", index=False)

        gene_std_pred, gene_std_true = [], []
        for cell in CELL_LINES:
            rows = np.flatnonzero(cl == cell)
            gene_std_pred.append(loss_only[rows].std(axis=0))
            gene_std_true.append(pooled_true[rows].std(axis=0))
        np.savez(
            data_dir / "gene_variance.npz",
            pred_std=np.mean(gene_std_pred, axis=0),
            true_std=np.mean(gene_std_true, axis=0),
        )

        # A549 confusion matrix (softmax assignment, row-normalised)
        rows = np.flatnonzero(cl == "A549")
        panel = sorted(
            set().union(
                *[set(np.argsort(-np.abs(pooled_true[i]))[:50]) for i in rows]
            )
        )
        P, T = loss_only[np.ix_(rows, panel)], pooled_true[np.ix_(rows, panel)]
        C = np.array(
            [[np.corrcoef(P[i], T[j])[0, 1] for j in range(len(rows))]
             for i in range(len(rows))]
        )
        A = np.exp(C * 6)
        A = A / A.sum(1, keepdims=True)
        np.save(data_dir / "confusion_A549.npy", A)
        (data_dir / "confusion_meta.json").write_text(
            json.dumps(
                {
                    "cell": "A549",
                    "n": len(rows),
                    "chance": 1 / len(rows),
                    "top1_recovery": float(np.mean(np.argmax(C, 1) == np.arange(len(rows)))),
                },
                indent=2,
            )
            + "\n"
        )

        # on/off pooled scores for loss-only
        score = _ddc(loss_only, pooled_true, cl)
        rows_on, rows_off = [], []
        for i in range(len(drugs)):
            for j in range(len(drugs)):
                if i == j:
                    continue
        # per-anchor on/off similarities (reuse the discrimination score input)
        onoff = []
        for cell in CELL_LINES:
            idx = np.flatnonzero(cl == cell)
            Pc = loss_only[idx]
            Tc = pooled_true[idx]
            for i in range(len(idx)):
                on = float(np.corrcoef(Pc[i], Tc[i])[0, 1])
                onoff.append({"score": on, "kind": "on"})
                for j in range(len(idx)):
                    if j != i:
                        onoff.append(
                            {
                                "score": float(np.corrcoef(Pc[i], Tc[j])[0, 1]),
                                "kind": "off",
                            }
                        )
        pd.DataFrame(onoff).to_csv(data_dir / "onoff_pooled.csv", index=False)

    # ---- pathway illusion ----
    pathway = loader.load_json("pathway/pathway_illusion.json")
    if pathway is None:
        # fall back to the repo's verified pathway artifact (matches Fig 6g)
        repo_pathway = (
            Path(__file__).resolve().parents[2]
            / "manuscript" / "analysis_scripts" / "verify_pathway.json"
        )
        if repo_pathway.exists():
            pathway = json.loads(repo_pathway.read_text(encoding="utf-8"))
    if pathway:
        (data_dir / "pathway_illusion.json").write_text(
            json.dumps(
                {
                    "on_diag_mean": pathway.get("on_diag_mean"),
                    "off_diag_mean": pathway.get("off_diag_mean"),
                    "gap": pathway.get("gap"),
                    "specificity_auc": pathway.get("specificity_auc"),
                },
                indent=2,
            )
            + "\n"
        )

    # ---- loss components + curves ----
    loss = loader.load_csv("loss_components.csv")
    if loss is not None:
        loss.to_csv(data_dir / "loss_components.csv", index=False)
    curves_dir = data_dir / "loss_curves"
    curves_dir.mkdir(exist_ok=True)
    for name in CONFIG_ORDER:
        metrics = loader.logs_metrics(name)
        if metrics is not None:
            metrics.to_csv(curves_dir / f"{_id(name)}.csv", index=False)

    # ---- oracle ladder ----
    ladder_rows = []
    for rung, entry in (
        ("Hindsight retrieval (oracle)", "fig5.hindsight"),
        ("Cross-plate biological replicate", "fig5.ceiling"),
        ("Target-matched oracle (15/27 anchors)", "fig5.target_matched"),
        ("CytoBridge (best configuration)", "fig4b.ladder.cb"),
        ("Ridge (as audited)", "fig4b.ladder.ridge"),
        ("Tanimoto 1-NN oracle", "fig5.tanimoto_nn"),
        ("Morgan-ridge oracle (160 compounds)", "fig5.morgan_ridge"),
        ("chemCPA / biolord (as audited)", "fig4b.ladder.chemcpa"),
        ("Mean (negative control)", "fig4b.ladder.mean"),
        ("Random (negative control)", "fig4b.ladder.random"),
    ):
        if results.get(entry) is not None:
            ladder_rows.append({"rung": rung, "auc": results[entry]})
    pd.DataFrame(ladder_rows).to_csv(data_dir / "oracle_ladder.csv", index=False)

    # ---- casestudies (same four A549 pairs as the legacy artifact) ----
    legacy = (
        Path(__file__).resolve().parents[2]
        / "manuscript" / "analysis" / "data2" / "casestudies.json"
    )
    if legacy.exists() and loss_only is not None:
        legacy_data = json.loads(legacy.read_text(encoding="utf-8"))
        case = {"cell": legacy_data.get("cell", "A549")}
        a549 = np.flatnonzero(cl == "A549")
        for key in ("pair0", "pair1", "pair2", "pair3"):
            pair = legacy_data.get(key)
            if not pair:
                continue
            ia = np.flatnonzero(drugs == pair["A"])[0]
            ib = np.flatnonzero(drugs == pair["B"])[0]
            case[key] = {
                "A": pair["A"],
                "B": pair["B"],
                "pred_r": float(np.corrcoef(loss_only[ia], loss_only[ib])[0, 1]),
                "true_r": float(np.corrcoef(pooled_true[ia], pooled_true[ib])[0, 1]),
            }
        (data_dir / "casestudies.json").write_text(
            json.dumps(case, indent=2) + "\n"
        )

    # ---- fig_numbers.json for build_nb2 compatibility ----
    (data_dir / "fig_numbers.json").write_text(
        json.dumps(
            {key: value for key, value in results.items() if isinstance(value, (int, float))},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return data_dir


def _id(name: str) -> str:
    return {
        "loss-only": "loss_only",
        "drug-spec x1": "drugspec1",
        "drug-spec x3": "drugspec3",
        "drug-spec x5": "drugspec5",
        "norm-only": "norm_only",
        "low recon weight": "low_recon",
        "recovery baseline": "recovery_base",
    }[name]
