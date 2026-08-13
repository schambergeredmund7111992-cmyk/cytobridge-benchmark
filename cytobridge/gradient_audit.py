"""Measure weighted per-loss gradient norms without changing accumulated gradients."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

LOSS_WEIGHT_FIELDS = {
    "L_recon": "lam_recon",
    "L_contrast": "lam_contrast",
    "L_pathway": "lam_pathway",
    "L_kl": "lam_kl",
    "L_delta": "lam_delta",
    "L_direction": "lam_direction",
    "L_drugspec": "lam_drugspec",
    "L_logfc": "lam_logfc",
    "L_norm_recon": "lam_norm_recon",
}


def component_weights(loss_config) -> dict[str, float]:
    return {
        component: float(getattr(loss_config, field))
        for component, field in LOSS_WEIGHT_FIELDS.items()
    }


def measure_component_gradient_norms(
    raw_components: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, dict[str, float | int]]:
    """Return L2 norms for active weighted components using ``autograd.grad``."""
    parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    tensors = [parameter for _, parameter in parameters]
    if not tensors:
        raise ValueError("No trainable parameters were provided for gradient auditing.")
    result: dict[str, dict[str, float | int]] = {}
    for component in sorted(raw_components):
        weight = float(weights.get(component, 0.0))
        if weight <= 0.0:
            continue
        raw = raw_components[component]
        if raw.ndim != 0:
            raise ValueError(
                f"{component} must be a scalar loss for gradient auditing."
            )
        connected = 0
        total_elements = 0
        nonfinite_elements = 0
        max_abs_gradient = 0.0
        combined_norm = 0.0
        nonfinite_parameters: list[str] = []
        raw_value = float(raw.detach().double().cpu())
        if not math.isfinite(raw_value):
            raise FloatingPointError(f"Non-finite raw loss for {component}: {raw_value}.")
        if raw.requires_grad:
            gradients = torch.autograd.grad(
                weight * raw,
                tensors,
                retain_graph=True,
                allow_unused=True,
            )
            for (name, _), gradient in zip(parameters, gradients, strict=True):
                if gradient is not None:
                    connected += 1
                    detached = gradient.detach()
                    total_elements += int(detached.numel())
                    finite = torch.isfinite(detached)
                    bad = int((~finite).sum().cpu())
                    nonfinite_elements += bad
                    if bad:
                        nonfinite_parameters.append(name)
                    finite_values = detached[finite]
                    if finite_values.numel():
                        max_abs_gradient = max(
                            max_abs_gradient,
                            float(finite_values.abs().max().double().cpu()),
                        )
                        # Compute each tensor norm in float64 and combine with hypot.
                        # This avoids the false overflow caused by float32 square().sum().
                        tensor_norm = float(torch.linalg.vector_norm(finite_values.double()).cpu())
                        combined_norm = math.hypot(combined_norm, tensor_norm)
        if nonfinite_elements:
            names = ", ".join(nonfinite_parameters[:5])
            raise FloatingPointError(
                f"Non-finite gradient values for {component}: {nonfinite_elements} elements "
                f"across [{names}]."
            )
        if not math.isfinite(combined_norm):
            raise FloatingPointError(f"Non-finite gradient norm for {component}.")
        if connected == 0:
            raise RuntimeError(f"Active loss {component} is disconnected from all parameters.")
        result[component] = {
            "coefficient": weight,
            "raw_loss": raw_value,
            "weighted_gradient_l2": combined_norm,
            "connected_parameter_tensors": connected,
            "gradient_elements": total_elements,
            "nonfinite_gradient_elements": nonfinite_elements,
            "max_abs_gradient": max_abs_gradient,
        }
    return result


def append_gradient_audit(path: Path, record: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
