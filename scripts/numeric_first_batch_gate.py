"""Pre-execution fp32/bf16 two-step numerical gate for CytoBridge.

This diagnostic makes a precision decision before any screening result exists.  It never
disables gradient auditing and writes a fail-closed JSON record for protocol provenance.
"""

from __future__ import annotations

import argparse
import copy
import json
import traceback
from contextlib import nullcontext
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from cytobridge.data import CytoBridgeDataModule
from cytobridge.gradient_audit import component_weights, measure_component_gradient_norms
from train import LitCytoBridge


def _to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=False)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _forward_loss(lit: LitCytoBridge, batch: dict, *, audit: bool) -> tuple[dict, dict]:
    output = lit.forward(batch)
    positive = lit.forward(batch)
    output["z_pos"] = positive["z"]
    if "hn_cell_emb" in batch and batch["hn_cell_emb"].numel():
        batch_size, n_hard, cell_length, cell_width = batch["hn_cell_emb"].shape
        drug_length = batch["hn_drug_mask"].shape[-1]
        hard_batch = {
            "cell_tokens": batch["hn_cell_emb"].reshape(
                batch_size * n_hard, cell_length, cell_width
            ),
            "drug_tokens": batch["hn_drug_emb"].reshape(
                batch_size * n_hard, drug_length, -1
            ),
            "drug_mask": batch["hn_drug_mask"].reshape(batch_size * n_hard, drug_length),
        }
        hard_output = lit.forward(hard_batch)
        output["z_hard_neg"] = hard_output["z"].reshape(batch_size, n_hard, -1)
    else:
        output["z_hard_neg"] = None
    losses = lit.loss_fn(output, batch, include_raw_components=audit)
    return output, losses


def _tensor_stats(values: dict) -> dict:
    result = {}
    for name, value in values.items():
        if not torch.is_tensor(value):
            continue
        detached = value.detach()
        finite = torch.isfinite(detached)
        result[name] = {
            "dtype": str(detached.dtype),
            "shape": list(detached.shape),
            "finite": bool(finite.all()),
            "nonfinite_elements": int((~finite).sum().cpu()),
            "max_abs": (
                float(detached[finite].abs().max().double().cpu())
                if finite.any()
                else None
            ),
        }
    return result


def _run_precision(
    precision: str,
    lit_template: LitCytoBridge,
    initial_state: dict,
    batch: dict,
    device: torch.device,
) -> dict:
    lit = copy.deepcopy(lit_template).to(device)
    lit.load_state_dict(initial_state, strict=True)
    configured = lit.configure_optimizers()
    optimizer = configured["optimizer"]
    use_bf16 = precision == "bf16-mixed"
    steps = []
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else nullcontext()
        )
        with context:
            outputs, losses = _forward_loss(lit, batch, audit=step == 0)
        raw_components = losses.pop("_raw_components", None)
        audit = None
        if raw_components is not None:
            audit = measure_component_gradient_norms(
                raw_components,
                component_weights(lit.loss_fn.cfg),
                list(lit.model.named_parameters()),
            )
        loss = losses["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{precision} step {step}: non-finite total loss")
        loss.backward()
        gradient_stats = _tensor_stats(
            {
                name: parameter.grad
                for name, parameter in lit.model.named_parameters()
                if parameter.grad is not None
            }
        )
        bad_gradients = [name for name, stats in gradient_stats.items() if not stats["finite"]]
        if bad_gradients:
            raise FloatingPointError(
                f"{precision} step {step}: non-finite gradients in {bad_gradients[:5]}"
            )
        torch.nn.utils.clip_grad_norm_(lit.model.parameters(), max_norm=1.0)
        optimizer.step()
        parameter_stats = _tensor_stats(dict(lit.model.named_parameters()))
        if any(not stats["finite"] for stats in parameter_stats.values()):
            raise FloatingPointError(f"{precision} step {step}: non-finite parameters")
        steps.append(
            {
                "step": step,
                "losses": {
                    name: float(value.detach().double().cpu())
                    for name, value in losses.items()
                    if torch.is_tensor(value)
                },
                "outputs": _tensor_stats(outputs),
                "gradient_audit": audit,
                "gradient_max_abs": max(
                    (stats["max_abs"] or 0.0 for stats in gradient_stats.values()),
                    default=0.0,
                ),
            }
        )
    return {
        "status": "passed",
        "precision": precision,
        "steps": steps,
        "peak_memory_mib": int(torch.cuda.max_memory_allocated(device) / 2**20),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="train/accept_base")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite numerical gate: {args.out}")
    if not torch.cuda.is_available():
        raise RuntimeError("The numerical precision gate requires CUDA.")

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra.compose(config_name=args.config_name)
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    data_cfg.update(
        {
            "num_workers": 0,
            "persistent_workers": False,
            "prefetch_factor": None,
            "multiprocessing_context": None,
        }
    )
    data_module = CytoBridgeDataModule(**data_cfg)
    data_module.setup("fit")
    cpu_batch = next(iter(data_module.train_dataloader()))
    device = torch.device("cuda:0")
    batch = _to_device(cpu_batch, device)
    torch.manual_seed(int(cfg.seed))
    template = LitCytoBridge(
        model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
        loss_cfg=OmegaConf.to_container(cfg.loss, resolve=True),
        opt_cfg=OmegaConf.to_container(cfg.optim, resolve=True),
        metric_cfg={"enabled": False},
        gradient_audit_cfg={"enabled": True},
    )
    initial_state = copy.deepcopy(template.state_dict())
    report = {
        "schema_version": 1,
        "config_name": args.config_name,
        "seed": int(cfg.seed),
        "anchor_indices": cpu_batch["anchor_indices"],
        "hard_negative_indices": cpu_batch["hard_neg_indices"],
        "results": {},
    }
    for precision in ("32-true", "bf16-mixed"):
        try:
            report["results"][precision] = _run_precision(
                precision, template, initial_state, batch, device
            )
        except Exception as exc:  # fail-closed evidence is more useful than a lost traceback
            report["results"][precision] = {
                "status": "failed",
                "precision": precision,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
    fp32_ok = report["results"]["32-true"]["status"] == "passed"
    bf16_ok = report["results"]["bf16-mixed"]["status"] == "passed"
    report["status"] = "passed" if fp32_ok else "failed"
    report["selected_precision"] = "bf16-mixed" if fp32_ok and bf16_ok else "32-true"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "precision": report["selected_precision"]}))
    raise SystemExit(0 if fp32_ok else 1)


if __name__ == "__main__":
    main()
