#!/usr/bin/env python3
"""Independent runtime audit for the checkpoint-faithful scGPT wrapper.

This script compares tokenization with scGPT 0.2.4's ``DataCollator`` using the same per-cell
random seeds, checks complete checkpoint coverage, and exercises deterministic embeddings on
synthetic and optional real expression rows. It does not train or score CytoBridge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cytobridge.encoders.scgpt_wrapper import (
    DEFAULT_SEED,
    ScGPTEncoder,
    TokenizedBatch,
    _cell_seed,
    _dense_rows,
    _sha256_file,
    _sha256_strings,
)


def _official_tokenize_one(
    expression: np.ndarray,
    gene_ids: np.ndarray,
    cell_key: str,
    encoder: ScGPTEncoder,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Call upstream DataCollator with isolated seeds, then pad to checkpoint length."""

    from scgpt.data_collator import DataCollator

    nonzero = np.flatnonzero(expression)
    genes = np.insert(gene_ids[nonzero], 0, encoder.cls_id)
    values = np.insert(expression[nonzero], 0, encoder.pad_value)
    example = {
        "id": torch.tensor(0),
        "genes": torch.from_numpy(genes).long(),
        "expressions": torch.from_numpy(values).float(),
    }
    collator = DataCollator(
        do_padding=True,
        pad_token_id=encoder.pad_id,
        pad_value=int(encoder.pad_value),
        do_mlm=False,
        do_binning=True,
        max_length=encoder.max_seq_len,
        sampling=True,
        keep_first_n_tokens=1,
    )
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        np.random.seed(_cell_seed(seed, cell_key, "binning") % (2**32))
        torch.manual_seed(_cell_seed(seed, cell_key, "sampling"))
        official = collator([example])
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)

    official_ids = official["gene"][0].cpu().numpy()
    official_values = official["expr"][0].cpu().numpy()
    padded_ids = np.full(encoder.max_seq_len, encoder.pad_id, dtype=np.int64)
    padded_values = np.full(encoder.max_seq_len, encoder.pad_value, dtype=np.float32)
    padded_ids[: len(official_ids)] = official_ids
    padded_values[: len(official_values)] = official_values
    return padded_ids, padded_values


def _synthetic_inputs(encoder: ScGPTEncoder) -> tuple[np.ndarray, np.ndarray, list[str]]:
    tokens = [
        token
        for token in encoder.vocab.get_itos()
        if token not in {encoder.pad_token, "<cls>", "<eoc>"}
    ][:1400]
    if len(tokens) < 1300:
        raise RuntimeError("checkpoint vocabulary is unexpectedly small")
    gene_ids = np.array([encoder.vocab[token] for token in tokens], dtype=np.int64)
    expression = np.zeros((3, len(tokens)), dtype=np.float32)
    expression[0] = np.arange(1, len(tokens) + 1, dtype=np.float32)
    expression[1] = np.arange(len(tokens), dtype=np.float32) % 17 + 1
    expression[2, :127] = np.linspace(0.1, 12.7, 127, dtype=np.float32)
    return expression, gene_ids, ["synthetic:long_unique", "synthetic:long_ties", "synthetic:short"]


def _embedding_checks(
    encoder: ScGPTEncoder,
    tokenized: TokenizedBatch,
    device: str,
    precision: str,
) -> dict[str, Any]:
    together = encoder.encode_tokenized(tokenized, device=device, precision=precision)
    repeated = encoder.encode_tokenized(tokenized, device=device, precision=precision)
    separate = []
    for index in range(len(tokenized.gene_ids)):
        one = TokenizedBatch(
            gene_ids=tokenized.gene_ids[index : index + 1],
            values=tokenized.values[index : index + 1],
            padding_mask=tokenized.padding_mask[index : index + 1],
            nonzero_counts=(tokenized.nonzero_counts[index],),
            selected_counts=(tokenized.selected_counts[index],),
        )
        separate.append(encoder.encode_tokenized(one, device=device, precision=precision))
    partitioned = np.concatenate(separate, axis=0)

    short_length = tokenized.selected_counts[-1] + 1
    short = TokenizedBatch(
        gene_ids=tokenized.gene_ids[-1:, :short_length],
        values=tokenized.values[-1:, :short_length],
        padding_mask=tokenized.padding_mask[-1:, :short_length],
        nonzero_counts=(tokenized.nonzero_counts[-1],),
        selected_counts=(tokenized.selected_counts[-1],),
    )
    short_embedding = encoder.encode_tokenized(short, device=device, precision=precision)
    repeat_max_abs = float(np.max(np.abs(together - repeated)))
    partition_max_abs = float(np.max(np.abs(together - partitioned)))
    padding_max_abs = float(np.max(np.abs(together[-1:] - short_embedding)))
    norm_error = float(np.max(np.abs(np.linalg.norm(together[:, 0], axis=1) - 1)))
    tolerance = 5e-5 if precision != "fp32" else 1e-5
    checks = {
        "shape": list(together.shape),
        "finite": bool(np.isfinite(together).all()),
        "repeat_max_abs": repeat_max_abs,
        "batch_partition_max_abs": partition_max_abs,
        "padding_max_abs": padding_max_abs,
        "unit_norm_max_abs_error": norm_error,
        "tolerance": tolerance,
    }
    checks["passed"] = bool(
        checks["finite"]
        and repeat_max_abs <= tolerance
        and partition_max_abs <= tolerance
        and padding_max_abs <= tolerance
        and norm_error <= tolerance
    )
    return checks


def _real_minibatch_checks(
    encoder: ScGPTEncoder,
    h5ad_path: Path,
    n_cells: int,
    device: str,
    precision: str,
    seed: int,
) -> dict[str, Any]:
    import scanpy as sc

    adata = sc.read_h5ad(h5ad_path)
    try:
        use_cells = min(n_cells, int(adata.n_obs))
        gene_names = [str(value) for value in adata.var_names.tolist()]
        kept_columns, gene_ids, kept_names = encoder._vocabulary_projection(gene_names)
        expression = _dense_rows(adata.X, 0, use_cells, kept_columns)
        row_ids = [str(value) for value in adata.obs_names[:use_cells].tolist()]
        cell_keys = [f"{index}:{row_ids[index]}" for index in range(use_cells)]
        tokenized = encoder.tokenize(expression, gene_ids, cell_keys, seed=seed)
        embedding_checks = _embedding_checks(encoder, tokenized, device, precision)
        expression_digest = hashlib.sha256(np.ascontiguousarray(expression).tobytes()).hexdigest()
        return {
            "path": str(h5ad_path),
            "sha256": _sha256_file(h5ad_path),
            "source_rows": int(adata.n_obs),
            "source_genes": int(adata.n_vars),
            "tested_rows": use_cells,
            "tested_row_ids_sha256": _sha256_strings(row_ids),
            "vocabulary_overlap_genes": len(kept_names),
            "ordered_overlap_genes_sha256": _sha256_strings(kept_names),
            "expression_slice_sha256": expression_digest,
            "embedding_checks": embedding_checks,
            "passed": embedding_checks["passed"],
        }
    finally:
        adata.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=os.environ.get("SCGPT_CKPT_DIR"), required=False)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--real-h5ad", type=Path, default=None)
    parser.add_argument("--real-cells", type=int, default=4)
    args = parser.parse_args()
    if not args.ckpt:
        parser.error("pass --ckpt or set SCGPT_CKPT_DIR")

    started = time.time()
    encoder = ScGPTEncoder(args.ckpt)
    expression, gene_ids, cell_keys = _synthetic_inputs(encoder)
    tokenized = encoder.tokenize(expression, gene_ids, cell_keys, seed=args.seed)

    oracle_ids = []
    oracle_values = []
    for row, cell_key in zip(expression, cell_keys, strict=True):
        ids, values = _official_tokenize_one(
            row, gene_ids, cell_key, encoder, args.seed
        )
        oracle_ids.append(ids)
        oracle_values.append(values)
    oracle_ids_array = np.stack(oracle_ids)
    oracle_values_array = np.stack(oracle_values)
    token_ids_exact = bool(np.array_equal(tokenized.gene_ids, oracle_ids_array))
    token_values_exact = bool(np.array_equal(tokenized.values, oracle_values_array))
    if not token_ids_exact or not token_values_exact:
        raise RuntimeError("wrapper tokenization differs from the scGPT 0.2.4 DataCollator oracle")

    synthetic_embeddings = _embedding_checks(
        encoder, tokenized, args.device, args.precision
    )
    if not synthetic_embeddings["passed"]:
        raise RuntimeError(f"synthetic embedding checks failed: {synthetic_embeddings}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "scope": "engineering audit only; no model training or benchmark scoring",
        "checkpoint_dir": str(encoder.ckpt_dir),
        "checkpoint_sha256": _sha256_file(encoder.ckpt_dir / "best_model.pt"),
        "vocab_sha256": _sha256_file(encoder.ckpt_dir / "vocab.json"),
        "args_sha256": _sha256_file(encoder.ckpt_dir / "args.json"),
        "wrapper_source_sha256": _sha256_file(
            Path(__file__).parents[1] / "cytobridge" / "encoders" / "scgpt_wrapper.py"
        ),
        "checkpoint_coverage": encoder.coverage_report,
        "official_tokenization_oracle": {
            "implementation": "scgpt.data_collator.DataCollator",
            "scgpt_version": "0.2.4",
            "tested_cells": len(cell_keys),
            "includes_truncation": True,
            "includes_tied_bin_boundaries": True,
            "gene_ids_exact": token_ids_exact,
            "binned_values_exact": token_values_exact,
        },
        "synthetic_embedding_checks": synthetic_embeddings,
        "device": args.device,
        "precision": args.precision,
        "seed": args.seed,
    }
    if args.real_h5ad is not None:
        report["real_minibatch"] = _real_minibatch_checks(
            encoder,
            args.real_h5ad.resolve(),
            args.real_cells,
            args.device,
            args.precision,
            args.seed,
        )
        if not report["real_minibatch"]["passed"]:
            raise RuntimeError("real mini-batch checks failed")
    report["elapsed_seconds"] = round(time.time() - started, 3)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
