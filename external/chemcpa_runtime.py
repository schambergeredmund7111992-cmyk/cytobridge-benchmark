"""Protocol-locked runner for official-architecture chemCPA trained from scratch."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from external.benchmark_runtime import (
    paired_control_inputs,
    score_validation_predictions,
    write_external_prediction_npz,
)
from eval.artifacts import sha256_file
from eval.model_selection import epoch_selection_key


OFFICIAL_COMMIT = "43e830eb0958c54e4aa64442c17ec0fed19b3f15"
VALIDATION_EPOCHS = (50, 100, 150, 200, 201)
DEFAULT_HPARAMS = {
    "adversary_depth": 3,
    "adversary_lr": 0.0011926173789223548,
    "adversary_steps": 3,
    "adversary_wd": 0.000009846738873614555,
    "adversary_width": 128,
    "autoencoder_depth": 4,
    "autoencoder_lr": 0.0015751320499779737,
    "autoencoder_wd": 0.0000006251373574521742,
    "autoencoder_width": 256,
    "batch_size": 256,
    "dim": 32,
    "dosers_depth": 3,
    "dosers_lr": 0.0015751320499779737,
    "dosers_wd": 0.0000006251373574521742,
    "dosers_width": 64,
    "dropout": 0.262378,
    "embedding_encoder_depth": 4,
    "embedding_encoder_width": 128,
    "penalty_adversary": 0.4550475813202185,
    "reg_adversary": 9.100951626404369,
    "reg_adversary_cov": 16.165583124257587,
    "reg_multi_task": 0,
    "step_size_lr": 6,
}
TRIAL_KEYS = {
    "autoencoder_lr",
    "autoencoder_wd",
    "adversary_width",
    "adversary_depth",
    "adversary_steps",
    "reg_adversary",
}


def resolve_hparams(trial: dict) -> dict:
    overrides = {key: value for key, value in trial.items() if key != "config_id"}
    unknown = set(overrides) - TRIAL_KEYS
    missing = TRIAL_KEYS - set(overrides)
    if unknown or missing:
        raise ValueError(
            f"chemCPA trial keys differ from protocol: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    resolved = {**DEFAULT_HPARAMS, **overrides}
    if resolved["batch_size"] != 256:
        raise ValueError("chemCPA batch size drifted from the pinned official default.")
    return resolved


def standardize_rdkit_table(
    raw: pd.DataFrame,
    training_smiles: set[str],
    *,
    threshold: float = 0.01,
) -> tuple[pd.DataFrame, dict]:
    if raw.index.astype(str).duplicated().any():
        raise ValueError("RDKit descriptor table contains duplicate SMILES.")
    missing = training_smiles - set(raw.index.astype(str))
    if missing:
        raise ValueError(f"Training SMILES missing from RDKit table: {sorted(missing)[:5]}")
    training = raw.loc[sorted(training_smiles)]
    train_std = training.std(axis=0, ddof=1)
    keep = train_std > threshold
    if not keep.any():
        raise ValueError("No training-variable RDKit descriptors remain.")
    mean = training.loc[:, keep].mean(axis=0)
    std = training.loc[:, keep].std(axis=0, ddof=1)
    normalized = (raw.loc[:, keep] - mean) / std
    if not np.isfinite(normalized.to_numpy(dtype=float)).all():
        raise FloatingPointError("Training-normalized RDKit descriptors are non-finite.")
    manifest = {
        "fit_split": "train",
        "threshold": threshold,
        "raw_features": int(raw.shape[1]),
        "retained_features": int(normalized.shape[1]),
        "training_smiles": len(training_smiles),
        "validation_or_test_response_used": False,
    }
    return normalized.astype(np.float32), manifest


def build_rdkit_embeddings(export_path: Path, output: Path) -> dict:
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite chemCPA embeddings: {output}")
    from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator

    export = ad.read_h5ad(export_path)
    smiles = sorted(export.obs["canonical_smiles"].astype(str).unique())
    generator = MakeGenerator(("RDKit2D",))
    rows = []
    for value in smiles:
        descriptor = generator.process(value)
        if descriptor is None or not bool(descriptor[0]):
            raise ValueError(f"Official RDKit2D generator failed for {value!r}.")
        rows.append(np.asarray(descriptor[1:], dtype=float))
    raw = pd.DataFrame(
        np.stack(rows),
        index=pd.Index(smiles, name="canonical_smiles"),
        columns=[f"latent_{index + 1}" for index in range(len(rows[0]))],
    )
    training_smiles = set(
        export.obs.loc[
            export.obs["benchmark_split"].astype(str).eq("train"),
            "canonical_smiles",
        ].astype(str)
    )
    normalized, manifest = standardize_rdkit_table(raw, training_smiles)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(output)
    payload = {
        "schema_version": 1,
        "export": str(export_path.resolve()),
        "export_sha256": sha256_file(export_path),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        **manifest,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def verify_official_checkout(path: Path) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    )
    if commit != OFFICIAL_COMMIT or status.strip():
        raise ValueError(
            f"chemCPA checkout is not the clean frozen source: commit={commit}, "
            f"dirty={bool(status.strip())}."
        )


def _load_trial(path: Path, config_id: str) -> dict:
    payload = json.loads(path.read_text())
    matches = [row for row in payload["chemcpa"] if row["config_id"] == config_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one chemCPA config {config_id!r}; found {len(matches)}.")
    return matches[0]


def _paired_indices(export: ad.AnnData, benchmark_split: str) -> tuple[np.ndarray, np.ndarray]:
    mask = (
        export.obs["benchmark_split"].astype(str).eq(benchmark_split)
        & export.obs["control"].astype(int).eq(0)
    )
    treated = np.flatnonzero(mask.to_numpy())
    lookup = {str(name): index for index, name in enumerate(export.obs_names.astype(str))}
    control_ids = export.obs.iloc[treated]["input_control_row_id"].astype(str)
    missing = sorted(set(control_ids) - set(lookup))
    if missing:
        raise ValueError(f"chemCPA paired controls are missing: {missing[:5]}")
    controls = np.asarray([lookup[name] for name in control_ids], dtype=int)
    return treated, controls


@torch.no_grad()
def predict_paired(
    lightning_model,
    dataset,
    export: ad.AnnData,
    benchmark_split: str,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    treated, controls = _paired_indices(export, benchmark_split)
    predictions = []
    was_training = lightning_model.training
    lightning_model.eval()
    try:
        for start in range(0, len(treated), batch_size):
            treated_batch = treated[start : start + batch_size]
            control_batch = controls[start : start + batch_size]
            batch = [
                dataset.genes[control_batch],
                dataset.drugs_idx[treated_batch],
                dataset.dosages[treated_batch],
                dataset.degs[treated_batch],
                *[values[treated_batch] for values in dataset.covariates],
            ]
            mean, variance = lightning_model(batch)
            if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
                raise FloatingPointError("chemCPA produced NaN or Inf predictions.")
            predictions.append(mean.detach().cpu().float().numpy())
    finally:
        lightning_model.train(was_training)
    return np.concatenate(predictions, axis=0)


class FiniteStateCallback:
    """Mixin-compatible callback methods loaded without importing Lightning at module import."""

    @staticmethod
    def _check(module, label: str) -> None:
        for name, parameter in module.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(f"chemCPA non-finite {label} parameter: {name}")
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"chemCPA non-finite {label} gradient: {name}")

    def on_after_backward(self, trainer, pl_module) -> None:
        del trainer
        self._check(pl_module, "post-backward")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        del outputs, batch, batch_idx
        for name in (
            "reconstruction_loss",
            "adversary_drugs_loss",
            "adversary_covariates_loss",
        ):
            value = trainer.callback_metrics.get(name)
            if value is not None and not torch.isfinite(torch.as_tensor(value)).all():
                raise FloatingPointError(f"chemCPA non-finite training metric: {name}")
        self._check(pl_module, "post-step")


def _atomic_output_dir(output: Path) -> tuple[Path, Path]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite chemCPA output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    return staging, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chemcpa-source", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--rdkit-embeddings", type=Path, required=True)
    parser.add_argument("--sweeps", type=Path, required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("selection", "final_refit"), required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--treated-counts", type=Path, required=True)
    parser.add_argument("--truth-control-counts", type=Path, required=True)
    parser.add_argument("--gene-panels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "selection" and args.epochs != 201:
        raise ValueError("chemCPA selection must run exactly 201 epochs.")
    if args.mode == "final_refit" and args.epochs not in VALIDATION_EPOCHS:
        raise ValueError("chemCPA final epochs must be a registered validation epoch.")
    verify_official_checkout(args.chemcpa_source)
    if not args.rdkit_embeddings.is_file():
        raise FileNotFoundError(args.rdkit_embeddings)

    import lightning as L
    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.loggers import CSVLogger

    import sys

    sys.path.insert(0, str(args.chemcpa_source))
    from chemCPA.data.data import load_dataset_splits
    from chemCPA.data.perturbation_data_module import PerturbationDataModule
    from chemCPA.lightning_module import ChemCPA

    trial = _load_trial(args.sweeps, args.config_id)
    hparams = resolve_hparams(trial)
    L.seed_everything(args.seed, workers=True)
    datasets, dataset = load_dataset_splits(
        dataset_path=str(args.export),
        perturbation_key="condition",
        dose_key="dose_val",
        covariate_keys="cell_type",
        smiles_key="canonical_smiles",
        degs_key="benchmark_all_genes",
        pert_category="cov_drug_dose_name",
        split_key="split",
        return_dataset=True,
        use_drugs_idx=True,
        verbose=False,
    )
    config = {
        "training": {"run_eval_r2": False},
        "model": {
            "additional_params": {
                "patience": 50,
                "decoder_activation": "ReLU",
                "doser_type": "amortized",
                "seed": 1337,
                "enable_cpa_mode": False,
            },
            "append_ae_layer": False,
            "embedding": {
                "model": "rdkit",
                "datapath": str(args.rdkit_embeddings),
            },
            "hparams": hparams,
            "load_pretrained": False,
        }
    }
    model = ChemCPA(
        config,
        {
            "num_genes": datasets["training"].num_genes,
            "num_drugs": datasets["training"].num_drugs,
            "num_covariates": datasets["training"].num_covariates,
            "use_drugs_idx": dataset.use_drugs_idx,
            "canon_smiles_unique_sorted": dataset.canon_smiles_unique_sorted,
        },
    )
    export = ad.read_h5ad(args.export)
    paired, _ = paired_control_inputs(
        export, "validation" if args.mode == "selection" else "test"
    )
    evaluations: list[dict] = []
    best: dict | None = None
    best_state: dict | None = None

    class ProtocolCallback(FiniteStateCallback, Callback):
        def on_train_epoch_end(self, trainer, pl_module) -> None:
            nonlocal best, best_state
            epoch = int(trainer.current_epoch) + 1
            if args.mode != "selection" or epoch not in VALIDATION_EPOCHS:
                return
            prediction = predict_paired(pl_module, dataset, export, "validation")
            metrics = score_validation_predictions(
                prediction,
                paired,
                args.manifest,
                args.treated_counts,
                args.truth_control_counts,
                args.gene_panels,
            )
            record = {"epoch": epoch, **metrics}
            evaluations.append(record)
            key = epoch_selection_key(
                metrics["conditional_accuracy_drug_macro"],
                metrics["pair_own_spearman_top50_drug_macro"],
                epoch,
            )
            if best is None or key > tuple(best["selection_key"]):
                best = {"epoch": epoch, "metrics": metrics, "selection_key": list(key)}
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in pl_module.state_dict().items()
                }

    staging, destination = _atomic_output_dir(args.out)
    data_module = PerturbationDataModule(
        datasplits=datasets, train_bs=int(hparams["batch_size"])
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        deterministic=True,
        logger=CSVLogger(save_dir=staging, name="lightning"),
        callbacks=[ProtocolCallback()],
        log_every_n_steps=20,
    )
    trainer.fit(model, datamodule=data_module)
    if args.mode == "selection":
        if best is None or best_state is None or len(evaluations) != len(VALIDATION_EPOCHS):
            raise RuntimeError("chemCPA did not complete every registered validation evaluation.")
        model.load_state_dict(best_state, strict=True)
        benchmark_split = "validation"
        selected_epoch = int(best["epoch"])
    else:
        benchmark_split = "test"
        selected_epoch = args.epochs
    prediction = predict_paired(model, dataset, export, benchmark_split)
    prediction_info = write_external_prediction_npz(
        staging / "predictions.npz", prediction, paired, args.manifest
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config_id": args.config_id,
            "hparams": hparams,
            "seed": args.seed,
            "selected_epoch": selected_epoch,
            "initialization": "from_scratch",
        },
        staging / "model.pt",
    )
    result = {
        "schema_version": 1,
        "model": "chemCPA (from scratch)",
        "initialization": "from_scratch",
        "pretrained_weights_used": False,
        "config_id": args.config_id,
        "seed": args.seed,
        "mode": args.mode,
        "selected_epoch": selected_epoch,
        "evaluations": evaluations,
        "best": best,
        "prediction": prediction_info,
        "official_source_commit": OFFICIAL_COMMIT,
        "source_hashes": {
            "export": sha256_file(args.export),
            "rdkit_embeddings": sha256_file(args.rdkit_embeddings),
            "sweeps": sha256_file(args.sweeps),
            "manifest": sha256_file(args.manifest),
            "treated_counts": sha256_file(args.treated_counts),
            "truth_control_counts": sha256_file(args.truth_control_counts),
            "gene_panels": sha256_file(args.gene_panels),
        },
    }
    (staging / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False, sort_keys=True) + "\n"
    )
    os.replace(staging, destination)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
