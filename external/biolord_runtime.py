"""Protocol-locked runner for the pinned official biolord implementation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from external.benchmark_runtime import (
    paired_control_inputs,
    score_validation_predictions,
    write_external_prediction_npz,
)
from eval.artifacts import sha256_file
from eval.model_selection import epoch_selection_key


OFFICIAL_COMMIT = "b7688790e49728d7f8d0906b980a629de484b19b"
REPRO_COMMIT = "16bfefccc0caa013b11d222f9b02aaf535807f85"
VALIDATION_INTERVAL = 20
MAX_EPOCHS = 500
PATIENCE_EVALUATIONS = 20
FIXED_MODULE_PARAMS = {
    "decoder_depth": 4,
    "attribute_nn_width": 2048,
    "attribute_nn_depth": 2,
    "use_batch_norm": False,
    "use_layer_norm": False,
    "attribute_dropout_rate": 0.1,
    "unknown_attribute_noise_param": 20.0,
    "n_latent_attribute_ordered": 256,
    "n_latent_attribute_categorical": 3,
    "gene_likelihood": "normal",
    "reconstruction_penalty": 10000.0,
    "unknown_attribute_penalty": 0.1,
}
FIXED_PLAN_PARAMS = {
    "n_epochs_warmup": 0,
    "latent_lr": 0.0001,
    "latent_wd": 0.0001,
    "decoder_lr": 0.0001,
    "decoder_wd": 0.0001,
    "attribute_nn_lr": 0.01,
    "attribute_nn_wd": 0.00000004,
    "step_size_lr": 45,
    "cosine_scheduler": True,
    "scheduler_final_lr": 0.00001,
}


def resolve_biolord_config(trial: dict, seed: int) -> tuple[dict, dict, int]:
    expected = {"config_id", "decoder_width", "n_latent"}
    if set(trial) != expected:
        raise ValueError(
            f"biolord trial keys differ from protocol: expected={sorted(expected)}, "
            f"observed={sorted(trial)}"
        )
    if int(trial["decoder_width"]) not in (2048, 4096, 8192):
        raise ValueError("biolord decoder width is outside the frozen grid.")
    if int(trial["n_latent"]) not in (128, 256):
        raise ValueError("biolord latent width is outside the frozen grid.")
    module = {
        **FIXED_MODULE_PARAMS,
        "decoder_width": int(trial["decoder_width"]),
        "seed": int(seed),
    }
    return module, dict(FIXED_PLAN_PARAMS), int(trial["n_latent"])


def _verify_checkout(path: Path, expected: str, label: str) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    )
    if commit != expected or status.strip():
        raise ValueError(
            f"{label} checkout is not the clean frozen source: commit={commit}, "
            f"dirty={bool(status.strip())}."
        )


def _load_trial(path: Path, config_id: str) -> dict:
    payload = json.loads(path.read_text())
    matches = [row for row in payload["biolord"] if row["config_id"] == config_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one biolord config {config_id!r}; found {len(matches)}.")
    return matches[0]


def _selection_key(metrics: dict, epoch: int) -> tuple[float, float, int]:
    return epoch_selection_key(
        metrics["conditional_accuracy_drug_macro"],
        metrics["pair_own_spearman_top50_drug_macro"],
        epoch,
    )


def _check_finite(module, label: str) -> None:
    for name, parameter in module.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"biolord non-finite {label} parameter: {name}")
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"biolord non-finite {label} gradient: {name}")


def _check_finite_output(value, label: str) -> None:
    if isinstance(value, dict):
        for name, child in value.items():
            _check_finite_output(child, f"{label}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _check_finite_output(child, f"{label}[{index}]")
    elif isinstance(value, (int, float, np.number, torch.Tensor)):
        if not torch.isfinite(torch.as_tensor(value)).all():
            raise FloatingPointError(f"biolord non-finite value: {label}")


def compose_counterfactual_batch(control: dict, treated: dict) -> dict:
    """Keep exact-control expression/identity and apply treated attributes."""
    if set(control) != set(treated):
        raise ValueError("biolord control and treated registry keys differ.")
    expression_key = "X" if "X" in control else "layers" if "layers" in control else None
    if expression_key is None or "ind_x" not in control:
        raise ValueError("biolord registry lacks expression or sample-index tensors.")
    if control[expression_key].shape[0] != treated[expression_key].shape[0]:
        raise ValueError("biolord control and treated batch sizes differ.")
    return {
        key: control[key] if key in {expression_key, "ind_x"} else treated[key]
        for key in control
    }


@torch.no_grad()
def predict_paired(
    model,
    export: ad.AnnData,
    benchmark_split: str,
    *,
    batch_size: int = 512,
) -> tuple[np.ndarray, ad.AnnData]:
    paired, controls = paired_control_inputs(export, benchmark_split)
    treated_mask = (
        export.obs["benchmark_split"].astype(str).eq(benchmark_split)
        & export.obs["control"].astype(int).eq(0)
    )
    treated = np.flatnonzero(treated_mask.to_numpy())
    if len(treated) != len(controls):
        raise ValueError("biolord paired treated/control index counts differ.")
    predictions = []
    was_training = model.module.training
    model.module.eval()
    try:
        for start in range(0, len(treated), batch_size):
            treated_batch = treated[start : start + batch_size]
            control_batch = controls[start : start + batch_size]
            control_data = model.get_dataset(export, indices=control_batch)
            treated_data = model.get_dataset(export, indices=treated_batch)
            counterfactual = compose_counterfactual_batch(control_data, treated_data)
            _, generative = model.module.forward(counterfactual, compute_loss=False)
            mean = generative["means"]
            variance = generative["variances"]
            if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
                raise FloatingPointError("biolord produced NaN or Inf predictions.")
            predictions.append(mean.detach().cpu().float().numpy())
    finally:
        model.module.train(was_training)
    return np.concatenate(predictions, axis=0), paired


def _atomic_output_dir(output: Path) -> tuple[Path, Path]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite biolord output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    return staging, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biolord-source", type=Path, required=True)
    parser.add_argument("--biolord-repro", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
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

    if args.mode == "selection" and args.epochs != MAX_EPOCHS:
        raise ValueError("biolord selection must use the registered 500-epoch maximum.")
    if args.mode == "final_refit" and (
        args.epochs < VALIDATION_INTERVAL
        or args.epochs > MAX_EPOCHS
        or args.epochs % VALIDATION_INTERVAL
    ):
        raise ValueError("biolord final epochs must be a registered validation epoch.")
    _verify_checkout(args.biolord_source, OFFICIAL_COMMIT, "biolord")
    _verify_checkout(args.biolord_repro, REPRO_COMMIT, "biolord reproducibility")

    import lightning as L
    from lightning.pytorch.callbacks import Callback

    import sys

    sys.path.insert(0, str(args.biolord_source / "src"))
    import biolord

    L.seed_everything(args.seed, workers=True)
    export = ad.read_h5ad(args.export)
    if "rdkit2d_dose" not in export.obsm:
        raise ValueError("biolord export is missing training-normalized rdkit2d_dose.")
    trial = _load_trial(args.sweeps, args.config_id)
    module_params, plan_params, n_latent = resolve_biolord_config(trial, args.seed)
    biolord.Biolord.setup_anndata(
        export,
        ordered_attributes_keys=["rdkit2d_dose"],
        categorical_attributes_keys=["cell_type"],
        retrieval_attribute_key=None,
    )
    model = biolord.Biolord(
        adata=export,
        n_latent=n_latent,
        model_name=f"cytobridge_{args.config_id}_seed{args.seed}",
        module_params=module_params,
        train_classifiers=False,
        split_key="split_ood",
        train_split="train",
        valid_split="test",
        test_split="ood",
    )
    benchmark_split = "validation" if args.mode == "selection" else "test"
    paired, _ = paired_control_inputs(export, benchmark_split)
    evaluations: list[dict] = []
    best: dict | None = None
    best_state: dict | None = None
    bad_evaluations = 0

    class ProtocolCallback(Callback):
        def on_after_backward(self, trainer, pl_module) -> None:
            del trainer
            _check_finite(pl_module, "post-backward")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
            del trainer, batch, batch_idx
            _check_finite_output(outputs, "training_output")
            _check_finite(pl_module, "post-step")

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            nonlocal best, best_state, bad_evaluations
            epoch = int(trainer.current_epoch) + 1
            if args.mode != "selection" or epoch % VALIDATION_INTERVAL:
                return
            values, observed_pairs = predict_paired(
                model, export, "validation", batch_size=512
            )
            if not np.array_equal(observed_pairs.obs_names, paired.obs_names):
                raise ValueError("biolord validation pair order drifted.")
            metrics = score_validation_predictions(
                values,
                paired,
                args.manifest,
                args.treated_counts,
                args.truth_control_counts,
                args.gene_panels,
            )
            record = {"epoch": epoch, **metrics}
            evaluations.append(record)
            key = _selection_key(metrics, epoch)
            if best is None or key > tuple(best["selection_key"]):
                best = {"epoch": epoch, "metrics": metrics, "selection_key": list(key)}
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.module.state_dict().items()
                }
                bad_evaluations = 0
            else:
                bad_evaluations += 1
                if bad_evaluations >= PATIENCE_EVALUATIONS:
                    trainer.should_stop = True

    staging, destination = _atomic_output_dir(args.out)
    model.train(
        max_epochs=args.epochs,
        batch_size=512,
        plan_kwargs=plan_params,
        early_stopping=False,
        accelerator="gpu",
        device=1,
        check_val_every_n_epoch=(
            VALIDATION_INTERVAL if args.mode == "selection" else args.epochs + 1
        ),
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        callbacks=[ProtocolCallback()],
        num_workers=0,
        default_root_dir=str(staging / "lightning"),
    )
    if args.mode == "selection":
        if best is None or best_state is None or not evaluations:
            raise RuntimeError("biolord completed no registered validation evaluation.")
        model.module.load_state_dict(best_state, strict=True)
        selected_epoch = int(best["epoch"])
    else:
        selected_epoch = args.epochs
    values, observed_pairs = predict_paired(
        model, export, benchmark_split, batch_size=512
    )
    if not np.array_equal(observed_pairs.obs_names, paired.obs_names):
        raise ValueError("biolord final pair order drifted.")
    prediction_info = write_external_prediction_npz(
        staging / "predictions.npz", values, paired, args.manifest
    )
    torch.save(
        {
            "state_dict": model.module.state_dict(),
            "config_id": args.config_id,
            "module_params": module_params,
            "plan_params": plan_params,
            "n_latent": n_latent,
            "seed": args.seed,
            "selected_epoch": selected_epoch,
        },
        staging / "model.pt",
    )
    result = {
        "schema_version": 1,
        "model": "biolord",
        "config_id": args.config_id,
        "seed": args.seed,
        "mode": args.mode,
        "selected_epoch": selected_epoch,
        "evaluations": evaluations,
        "best": best,
        "prediction": prediction_info,
        "official_source_commits": {
            "biolord": OFFICIAL_COMMIT,
            "biolord_reproducibility": REPRO_COMMIT,
        },
        "source_hashes": {
            "export": sha256_file(args.export),
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
