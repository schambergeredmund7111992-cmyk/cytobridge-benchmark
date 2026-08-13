"""Checkpoint-faithful, frozen scGPT whole-human cell encoder.

The wrapper follows the inference path used by ``scgpt.tasks.cell_emb``:

* filter genes that are absent from the checkpoint vocabulary;
* retain non-zero genes, prepend ``<cls>``, and bin expression per cell;
* truncate to the checkpoint's ``max_seq_len`` and pad with checkpoint values;
* take the transformer ``<cls>`` state and L2-normalize it.

Unlike the upstream convenience helper, truncation and bin-tie sampling are derived from a
stable per-cell seed. This makes cache rows independent of batch size and reproducible across
restarts. The model is constructed from ``args.json`` and loading fails unless every tensor used
by ``TransformerModel._encode`` is present with the exact expected shape.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

ENCODER_PREFIXES = ("encoder.", "value_encoder.", "transformer_encoder.")
REQUIRED_FILES = ("args.json", "vocab.json", "best_model.pt")
REQUIRED_ARGS = (
    "embsize",
    "nheads",
    "d_hid",
    "nlayers",
    "pad_token",
    "pad_value",
    "input_style",
    "input_emb_style",
    "n_bins",
    "max_seq_len",
    "cell_emb_style",
)
DEFAULT_SEED = 20260710


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a checkpoint cannot faithfully supply the frozen encoder."""


@dataclass(frozen=True)
class TokenizedBatch:
    """Fixed-length scGPT inputs and audit counts for one batch."""

    gene_ids: np.ndarray
    values: np.ndarray
    padding_mask: np.ndarray
    nonzero_counts: tuple[int, ...]
    selected_counts: tuple[int, ...]


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little"))
        digest.update(encoded)
    return digest.hexdigest()


def _update_token_digest(digest: Any, batch: TokenizedBatch) -> None:
    """Update a cache token digest in row order, independent of batch partitioning."""

    for gene_ids, values in zip(batch.gene_ids, batch.values, strict=True):
        digest.update(np.ascontiguousarray(gene_ids).tobytes())
        digest.update(np.ascontiguousarray(values).tobytes())


def _cell_seed(seed: int, cell_key: str, purpose: str) -> int:
    payload = f"{seed}\0{cell_key}\0{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="little") & (
        (1 << 63) - 1
    )


def _digitize_like_scgpt(
    values: np.ndarray,
    bins: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Match scGPT's random tie handling while using an isolated RNG."""

    left_digits = np.digitize(values, bins)
    right_digits = np.digitize(values, bins, right=True)
    random_values = rng.random(len(values))
    return np.ceil(random_values * (right_digits - left_digits) + left_digits).astype(
        np.int64
    )


def _bin_nonzero_values(
    values: np.ndarray,
    n_bins: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Reproduce ``scgpt.preprocess.binning`` for one non-empty cell."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("scGPT binning requires a non-empty one-dimensional array")
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("non-zero scGPT inputs must be finite and strictly positive")
    bins = np.quantile(values, np.linspace(0, 1, n_bins - 1))
    return _digitize_like_scgpt(values, bins, rng)


def tokenize_expression_rows(
    expression: np.ndarray,
    vocab_gene_ids: np.ndarray,
    cell_keys: Sequence[str],
    *,
    cls_id: int,
    pad_id: int,
    pad_value: float,
    n_bins: int,
    max_seq_len: int,
    seed: int = DEFAULT_SEED,
) -> TokenizedBatch:
    """Tokenize dense expression rows with deterministic, per-cell randomness."""

    expression = np.asarray(expression)
    vocab_gene_ids = np.asarray(vocab_gene_ids, dtype=np.int64)
    if expression.ndim != 2:
        raise ValueError(f"expression must be two-dimensional, got {expression.shape}")
    if expression.shape[1] != len(vocab_gene_ids):
        raise ValueError("expression columns and vocab_gene_ids are not aligned")
    if expression.shape[0] != len(cell_keys):
        raise ValueError("expression rows and cell_keys are not aligned")
    if max_seq_len < 2:
        raise ValueError("max_seq_len must leave room for <cls> and at least one gene")
    if not np.isfinite(expression).all() or np.any(expression < 0):
        raise ValueError("scGPT expression input must be finite and non-negative")

    batch_size = expression.shape[0]
    token_ids = np.full((batch_size, max_seq_len), pad_id, dtype=np.int64)
    token_values = np.full((batch_size, max_seq_len), pad_value, dtype=np.float32)
    nonzero_counts: list[int] = []
    selected_counts: list[int] = []

    for row_index, cell_key in enumerate(cell_keys):
        row = expression[row_index]
        nonzero_positions = np.flatnonzero(row)
        if not len(nonzero_positions):
            raise ValueError(f"cell {cell_key!r} has no non-zero genes in the scGPT vocabulary")

        nonzero_values = row[nonzero_positions]
        bin_seed = _cell_seed(seed, str(cell_key), "binning") % (2**32)
        bin_rng = np.random.RandomState(bin_seed)
        binned_values = _bin_nonzero_values(nonzero_values, n_bins, bin_rng)
        selected_positions = np.arange(len(nonzero_positions))
        max_genes = max_seq_len - 1
        if len(selected_positions) > max_genes:
            sample_generator = torch.Generator(device="cpu")
            sample_generator.manual_seed(_cell_seed(seed, str(cell_key), "sampling"))
            selected_positions = torch.randperm(
                len(selected_positions), generator=sample_generator
            ).numpy()[:max_genes]

        selected_genes = nonzero_positions[selected_positions]
        selected_bins = binned_values[selected_positions]
        sequence_length = len(selected_genes) + 1
        token_ids[row_index, 0] = cls_id
        token_ids[row_index, 1:sequence_length] = vocab_gene_ids[selected_genes]
        token_values[row_index, 1:sequence_length] = selected_bins.astype(np.float32)
        nonzero_counts.append(int(len(nonzero_positions)))
        selected_counts.append(int(len(selected_genes)))

    return TokenizedBatch(
        gene_ids=token_ids,
        values=token_values,
        padding_mask=token_ids == pad_id,
        nonzero_counts=tuple(nonzero_counts),
        selected_counts=tuple(selected_counts),
    )


def _load_scgpt_api() -> tuple[type[Any], type[Any]]:
    try:
        from scgpt.model import TransformerModel
        from scgpt.tokenizer import GeneVocab
    except ImportError as exc:
        raise ImportError(
            "scgpt is required for embedding generation. Install the frozen scGPT "
            "environment from env/scgpt_autodl_env.yaml."
        ) from exc
    return TransformerModel, GeneVocab


def _validate_checkpoint_args(args: dict[str, Any]) -> None:
    missing = sorted(set(REQUIRED_ARGS) - set(args))
    if missing:
        raise CheckpointCompatibilityError(f"args.json is missing required keys: {missing}")
    if args["input_style"] != "binned":
        raise CheckpointCompatibilityError(
            f"only checkpoint input_style='binned' is approved, got {args['input_style']!r}"
        )
    if args["cell_emb_style"] != "cls":
        raise CheckpointCompatibilityError(
            f"only checkpoint cell_emb_style='cls' is approved, got {args['cell_emb_style']!r}"
        )


def _normalise_checkpoint_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in payload.items()
    ):
        raise CheckpointCompatibilityError("best_model.pt is not a tensor state dictionary")
    if payload and all(key.startswith("module.") for key in payload):
        payload = {key.removeprefix("module."): value for key, value in payload.items()}
    return {
        key.replace("Wqkv.", "in_proj_"): value
        for key, value in payload.items()
    }


class ScGPTEncoder(nn.Module):
    """Frozen scGPT encoder constructed entirely from checkpoint metadata."""

    def __init__(self, ckpt_dir: str | Path | None = None) -> None:
        super().__init__()
        configured_dir = ckpt_dir or os.environ.get("SCGPT_CKPT_DIR")
        if configured_dir is None:
            raise ValueError("pass ckpt_dir or set SCGPT_CKPT_DIR")
        self.ckpt_dir = Path(configured_dir).expanduser().resolve()
        missing_files = [name for name in REQUIRED_FILES if not (self.ckpt_dir / name).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"checkpoint directory {self.ckpt_dir} is missing {missing_files}"
            )

        self.checkpoint_args = json.loads((self.ckpt_dir / "args.json").read_text())
        _validate_checkpoint_args(self.checkpoint_args)
        TransformerModel, GeneVocab = _load_scgpt_api()
        self.vocab = GeneVocab.from_file(self.ckpt_dir / "vocab.json")

        self.pad_token = str(self.checkpoint_args["pad_token"])
        for token in (self.pad_token, "<cls>", "<eoc>"):
            if token not in self.vocab:
                raise CheckpointCompatibilityError(
                    f"checkpoint vocabulary is missing required special token {token!r}"
                )
        self.pad_id = int(self.vocab[self.pad_token])
        self.cls_id = int(self.vocab["<cls>"])
        self.pad_value = float(self.checkpoint_args["pad_value"])
        self.n_bins = int(self.checkpoint_args["n_bins"])
        self.max_seq_len = int(self.checkpoint_args["max_seq_len"])
        self.emb_dim = int(self.checkpoint_args["embsize"])

        config = self.checkpoint_args
        self.model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=self.emb_dim,
            nhead=int(config["nheads"]),
            d_hid=int(config["d_hid"]),
            nlayers=int(config["nlayers"]),
            nlayers_cls=int(config.get("n_layers_cls", 3)),
            n_cls=int(config.get("n_cls", 1)),
            vocab=self.vocab,
            dropout=float(config.get("dropout", 0.5)),
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            do_mvc=bool(config.get("MVC", False)),
            do_dab=bool(config.get("DAB", False)),
            use_batch_labels=bool(config.get("use_batch_labels", False)),
            num_batch_labels=config.get("num_batch_labels"),
            domain_spec_batchnorm=bool(config.get("domain_spec_batchnorm", False)),
            input_emb_style=str(config["input_emb_style"]),
            n_input_bins=self.n_bins,
            cell_emb_style=str(config["cell_emb_style"]),
            mvc_decoder_style=str(config.get("mvc_decoder_style", "inner product")),
            ecs_threshold=float(config.get("ecs_threshold", 0.3)),
            explicit_zero_prob=bool(config.get("explicit_zero_prob", False)),
            use_fast_transformer=False,
            pre_norm=bool(config.get("pre_norm", False)),
        )
        self.coverage_report = self._load_and_verify_encoder()
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _load_and_verify_encoder(self) -> dict[str, Any]:
        checkpoint_path = self.ckpt_dir / "best_model.pt"
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:  # torch < 2.0 compatibility
            payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_state = _normalise_checkpoint_state(payload)
        model_state = self.model.state_dict()
        required_keys = sorted(
            key for key in model_state if key.startswith(ENCODER_PREFIXES)
        )
        missing_keys = sorted(key for key in required_keys if key not in checkpoint_state)
        shape_mismatches = [
            {
                "key": key,
                "expected": list(model_state[key].shape),
                "observed": list(checkpoint_state[key].shape),
            }
            for key in required_keys
            if key in checkpoint_state and model_state[key].shape != checkpoint_state[key].shape
        ]
        if missing_keys or shape_mismatches:
            raise CheckpointCompatibilityError(
                "checkpoint does not fully cover TransformerModel._encode: "
                f"missing={missing_keys}, shape_mismatches={shape_mismatches}"
            )

        compatible_state = {
            key: value
            for key, value in checkpoint_state.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        load_result = self.model.load_state_dict(compatible_state, strict=False)
        unequal_after_load = sorted(
            key
            for key in required_keys
            if not torch.equal(self.model.state_dict()[key].cpu(), checkpoint_state[key].cpu())
        )
        if unequal_after_load:
            raise CheckpointCompatibilityError(
                f"encoder tensors differ after checkpoint load: {unequal_after_load}"
            )

        ignored_checkpoint_keys = sorted(set(checkpoint_state) - set(compatible_state))
        report = {
            "required_encoder_prefixes": list(ENCODER_PREFIXES),
            "required_encoder_keys": required_keys,
            "required_encoder_key_count": len(required_keys),
            "loaded_encoder_key_count": len(required_keys),
            "missing_encoder_keys": missing_keys,
            "shape_mismatches": shape_mismatches,
            "unequal_after_load": unequal_after_load,
            "loaded_compatible_key_count": len(compatible_state),
            "model_keys_not_loaded": sorted(load_result.missing_keys),
            "ignored_checkpoint_keys": ignored_checkpoint_keys,
            "coverage_fraction": 1.0,
        }
        print(
            "[scgpt] required encoder checkpoint coverage: "
            f"{len(required_keys)}/{len(required_keys)} (100.0%)"
        )
        return report

    def _vocabulary_projection(
        self, gene_names: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        names = [str(name) for name in gene_names]
        if len(names) != len(set(names)):
            raise ValueError("scGPT input gene names must be unique")
        kept_columns = np.array(
            [index for index, gene in enumerate(names) if gene in self.vocab], dtype=np.int64
        )
        if not len(kept_columns):
            raise ValueError("none of the input genes occur in the scGPT vocabulary")
        kept_names = [names[index] for index in kept_columns]
        gene_ids = np.array([self.vocab[gene] for gene in kept_names], dtype=np.int64)
        if np.any(gene_ids == self.pad_id):
            raise CheckpointCompatibilityError("a biological gene resolved to the padding token")
        return kept_columns, gene_ids, kept_names

    def tokenize(
        self,
        expression: np.ndarray,
        gene_ids: np.ndarray,
        cell_keys: Sequence[str],
        *,
        seed: int = DEFAULT_SEED,
    ) -> TokenizedBatch:
        return tokenize_expression_rows(
            expression,
            gene_ids,
            cell_keys,
            cls_id=self.cls_id,
            pad_id=self.pad_id,
            pad_value=self.pad_value,
            n_bins=self.n_bins,
            max_seq_len=self.max_seq_len,
            seed=seed,
        )

    @torch.no_grad()
    def encode_tokenized(
        self,
        batch: TokenizedBatch,
        *,
        device: str | torch.device,
        precision: str = "fp32",
    ) -> np.ndarray:
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"precision must be fp32, fp16, or bf16; got {precision!r}")
        if device.type != "cuda" and precision != "fp32":
            raise ValueError("fp16/bf16 inference is supported only on CUDA")

        self.model.to(device)
        self.model.eval()
        gene_ids = torch.from_numpy(batch.gene_ids).to(device=device, dtype=torch.long)
        values = torch.from_numpy(batch.values).to(device=device, dtype=torch.float32)
        padding_mask = torch.from_numpy(batch.padding_mask).to(device=device, dtype=torch.bool)
        if device.type == "cuda" and precision != "fp32":
            dtype = torch.float16 if precision == "fp16" else torch.bfloat16
            autocast_context = torch.autocast(device_type="cuda", dtype=dtype)
        else:
            autocast_context = nullcontext()
        with autocast_context:
            encoded = self.model._encode(
                gene_ids,
                values,
                padding_mask,
                batch_labels=None,
            )
        cls_embeddings = encoded[:, 0, :].detach().float().cpu().numpy()
        norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
        if not np.isfinite(cls_embeddings).all() or np.any(norms == 0):
            raise RuntimeError("scGPT produced non-finite or zero-norm CLS embeddings")
        return (cls_embeddings / norms).astype(np.float32, copy=False)[:, None, :]

    @torch.no_grad()
    def encode_anndata(
        self,
        adata: Any,
        batch_size: int = 32,
        device: str = "cuda",
        *,
        precision: str = "fp32",
        seed: int = DEFAULT_SEED,
        gene_name_key: str | None = None,
        layer: str | None = None,
    ) -> torch.Tensor:
        """Encode a small in-memory AnnData object to normalized CLS tokens."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        gene_names = (
            [str(value) for value in adata.var[gene_name_key].tolist()]
            if gene_name_key
            else [str(value) for value in adata.var_names.tolist()]
        )
        kept_columns, gene_ids, _ = self._vocabulary_projection(gene_names)
        matrix = adata.layers[layer] if layer else adata.X
        outputs: list[np.ndarray] = []
        for start in range(0, int(adata.n_obs), batch_size):
            stop = min(start + batch_size, int(adata.n_obs))
            rows = _dense_rows(matrix, start, stop, kept_columns)
            cell_keys = [f"{index}:{adata.obs_names[index]}" for index in range(start, stop)]
            tokenized = self.tokenize(rows, gene_ids, cell_keys, seed=seed)
            outputs.append(
                self.encode_tokenized(tokenized, device=device, precision=precision)
            )
        return torch.from_numpy(np.concatenate(outputs, axis=0))

    def forward(
        self,
        gene_ids: torch.Tensor,
        expr_values: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Expose the frozen token encoder for already validated inputs."""

        with torch.no_grad():
            return self.model._encode(
                gene_ids,
                expr_values,
                pad_mask,
                batch_labels=None,
            )


def _dense_rows(
    matrix: Any,
    start: int,
    stop: int,
    kept_columns: np.ndarray,
) -> np.ndarray:
    # Backed CSR/CSC datasets in anndata 0.10 reject mixed ``slice, ndarray``
    # indexing. Materialize only the requested row batch before projecting genes.
    rows = matrix[start:stop]
    if hasattr(rows, "to_memory"):
        rows = rows.to_memory()
    rows = rows[:, kept_columns]
    if hasattr(rows, "toarray"):
        rows = rows.toarray()
    return np.asarray(rows, dtype=np.float32)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def cache_embeddings_to_disk(
    adata_path: str | Path,
    out_path: str | Path,
    ckpt_dir: str | Path | None = None,
    batch_size: int = 32,
    *,
    device: str = "auto",
    precision: str = "fp32",
    seed: int = DEFAULT_SEED,
    gene_name_key: str | None = None,
    layer: str | None = None,
    evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically generate an ``[N, 1, d]`` CLS cache plus provenance JSON."""

    import scanpy as sc

    adata_path = Path(adata_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    evidence_path = (
        Path(evidence_path).expanduser().resolve()
        if evidence_path
        else out_path.with_suffix(out_path.suffix + ".provenance.json")
    )
    if out_path.suffix != ".npy":
        raise ValueError(f"output must use the .npy extension, got {out_path}")
    if out_path.exists() or evidence_path.exists():
        raise FileExistsError(
            f"refusing to overwrite cache or evidence: {out_path}, {evidence_path}"
        )
    if not adata_path.is_file():
        raise FileNotFoundError(adata_path)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    encoder = ScGPTEncoder(ckpt_dir=ckpt_dir)
    resolved_device = _resolve_device(device)
    # Read the sparse AnnData into memory once. anndata 0.10.7 backed sparse slicing is
    # incompatible with scipy 1.15.x on the frozen l20 environment. Embeddings themselves are
    # still streamed to a memmap, so no [N, 1, d] tensor is accumulated in RAM.
    adata = sc.read_h5ad(adata_path)
    try:
        gene_names = (
            [str(value) for value in adata.var[gene_name_key].tolist()]
            if gene_name_key
            else [str(value) for value in adata.var_names.tolist()]
        )
        kept_columns, gene_ids, kept_gene_names = encoder._vocabulary_projection(gene_names)
        matrix = adata.layers[layer] if layer else adata.X
        n_cells = int(adata.n_obs)
        row_ids = [str(value) for value in adata.obs_names.tolist()]
        if not n_cells:
            raise ValueError("input AnnData contains no cells")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.partial.npy")
        if temporary_path.exists():
            raise FileExistsError(f"stale temporary cache exists: {temporary_path}")
        output = np.lib.format.open_memmap(
            temporary_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_cells, 1, encoder.emb_dim),
        )
        token_digest = hashlib.sha256()
        nonzero_counts: list[int] = []
        selected_counts: list[int] = []
        try:
            for start in range(0, n_cells, batch_size):
                stop = min(start + batch_size, n_cells)
                expression = _dense_rows(matrix, start, stop, kept_columns)
                cell_keys = [f"{index}:{row_ids[index]}" for index in range(start, stop)]
                tokenized = encoder.tokenize(expression, gene_ids, cell_keys, seed=seed)
                _update_token_digest(token_digest, tokenized)
                nonzero_counts.extend(tokenized.nonzero_counts)
                selected_counts.extend(tokenized.selected_counts)
                output[start:stop] = encoder.encode_tokenized(
                    tokenized,
                    device=resolved_device,
                    precision=precision,
                )
                print(f"[scgpt] encoded rows {start}:{stop}/{n_cells}")
            output.flush()
            temporary_path.replace(out_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    checkpoint_hashes = {
        name: _sha256_file(encoder.ckpt_dir / name) for name in REQUIRED_FILES
    }
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "encoder": "scGPT whole-human normalized CLS",
        "upstream_semantics": "scgpt.tasks.cell_emb.embed_data/get_batch_cell_embeddings",
        "input_h5ad": str(adata_path),
        "input_h5ad_sha256": _sha256_file(adata_path),
        "expression_source": f"layers[{layer!r}]" if layer else "X",
        "gene_name_source": f"var[{gene_name_key!r}]" if gene_name_key else "var_names",
        "input_gene_count": len(gene_names),
        "vocabulary_gene_count": len(kept_gene_names),
        "ordered_vocabulary_genes_sha256": _sha256_strings(kept_gene_names),
        "row_count": n_cells,
        "ordered_row_ids_sha256": _sha256_strings(row_ids),
        "checkpoint_dir": str(encoder.ckpt_dir),
        "checkpoint_files_sha256": checkpoint_hashes,
        "wrapper_source_sha256": _sha256_file(Path(__file__).resolve()),
        "checkpoint_args": encoder.checkpoint_args,
        "checkpoint_coverage": encoder.coverage_report,
        "tokenization": {
            "input_style": "binned",
            "n_bins": encoder.n_bins,
            "max_seq_len": encoder.max_seq_len,
            "pad_token": encoder.pad_token,
            "pad_id": encoder.pad_id,
            "pad_value": encoder.pad_value,
            "cell_emb_style": "cls",
            "seed": seed,
            "per_cell_seed_key": "sha256(seed, row_index:obs_name, purpose)",
            "token_stream_sha256": token_digest.hexdigest(),
            "nonzero_genes_min": min(nonzero_counts),
            "nonzero_genes_max": max(nonzero_counts),
            "selected_genes_min": min(selected_counts),
            "selected_genes_max": max(selected_counts),
        },
        "runtime": {
            "device": resolved_device,
            "precision": precision,
            "batch_size": batch_size,
            "torch_version": torch.__version__,
            "scgpt_version": importlib.metadata.version("scgpt"),
        },
        "output_cache": str(out_path),
        "output_shape": [n_cells, 1, encoder.emb_dim],
        "output_dtype": "float32",
        "output_sha256": _sha256_file(out_path),
    }
    temporary_evidence = evidence_path.with_name(
        f".{evidence_path.name}.{os.getpid()}.partial"
    )
    temporary_evidence.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    temporary_evidence.replace(evidence_path)
    print(f"[scgpt] cache: {out_path}")
    print(f"[scgpt] provenance: {evidence_path}")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ckpt", default=None, help="Defaults to SCGPT_CKPT_DIR.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gene-name-key", default=None)
    parser.add_argument("--layer", default=None, help="AnnData layer; default uses X.")
    parser.add_argument("--evidence", default=None)
    args = parser.parse_args()
    cache_embeddings_to_disk(
        args.adata,
        args.out,
        ckpt_dir=args.ckpt,
        batch_size=args.batch_size,
        device=args.device,
        precision=args.precision,
        seed=args.seed,
        gene_name_key=args.gene_name_key,
        layer=args.layer,
        evidence_path=args.evidence,
    )


if __name__ == "__main__":
    main()
