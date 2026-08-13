"""Fail-closed CUDA, architecture, fp32, and bf16 runtime verification."""

from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch


def _finite_step(device: torch.device, dtype: torch.dtype) -> dict:
    torch.manual_seed(20260710)
    left = torch.randn(128, 256, device=device, dtype=torch.float32, requires_grad=True)
    right = torch.randn(256, 64, device=device, dtype=torch.float32, requires_grad=True)
    if dtype == torch.float32:
        output = left @ right
    else:
        with torch.autocast(device_type="cuda", dtype=dtype):
            output = left @ right
    loss = output.float().square().mean()
    loss.backward()
    tensors = {"output": output, "loss": loss, "left_grad": left.grad, "right_grad": right.grad}
    finite = all(value is not None and bool(torch.isfinite(value).all()) for value in tensors.values())
    return {
        "dtype": str(dtype),
        "finite": finite,
        "loss": float(loss.detach().cpu()),
        "output_dtype": str(output.dtype),
    }


def verify(
    *,
    expected_torch: str,
    required_cuda_prefix: str,
    require_blackwell: bool,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected Python environment.")
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    arch_list = list(torch.cuda.get_arch_list())
    torch_ok = torch.__version__.split("+")[0] == expected_torch
    cuda_ok = str(torch.version.cuda or "").startswith(required_cuda_prefix)
    blackwell_binary = "sm_120" in arch_list or "compute_120" in arch_list
    blackwell_device = capability >= (12, 0)
    fp32 = _finite_step(device, torch.float32)
    bf16 = _finite_step(device, torch.bfloat16)
    checks = {
        "torch_version": torch_ok,
        "cuda_runtime": cuda_ok,
        "fp32_finite": fp32["finite"],
        "bf16_finite": bf16["finite"],
        "blackwell_binary": blackwell_binary,
        "blackwell_device": blackwell_device,
    }
    required = ["torch_version", "cuda_runtime", "fp32_finite", "bf16_finite"]
    if require_blackwell:
        required.extend(("blackwell_binary", "blackwell_device"))
    status = "passed" if all(checks[name] for name in required) else "failed"
    return {
        "schema_version": 1,
        "status": status,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(capability),
        "compiled_arch_list": arch_list,
        "require_blackwell": require_blackwell,
        "checks": checks,
        "fp32": fp32,
        "bf16": bf16,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-torch", default="2.7.1")
    parser.add_argument("--required-cuda-prefix", default="12.8")
    parser.add_argument(
        "--require-blackwell",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require an sm_120-capable binary and a Blackwell device.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite runtime verification: {args.out}")
    report = verify(
        expected_torch=args.expected_torch,
        required_cuda_prefix=args.required_cuda_prefix,
        require_blackwell=args.require_blackwell,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": report["status"], "device": report["device_name"]}))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
