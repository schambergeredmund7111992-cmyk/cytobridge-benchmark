"""Versioned, self-validating artifacts for perturbation-model predictions.

An artifact is an immutable directory with exactly three required payloads:

``predictions.npz``
    Numeric ``pred`` and ``true`` matrices plus a one-dimensional ``gene_ids`` array.
``metadata.csv``
    Row-aligned identifiers and split information.
``provenance.json``
    Hashes and execution metadata needed to trace the artifact to its inputs.

The writer stages the complete directory beside its destination, validates the staged
payload, and then publishes it with one atomic rename. Existing artifacts are never
overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0.0"
PREDICTIONS_FILE = "predictions.npz"
METADATA_FILE = "metadata.csv"
PROVENANCE_FILE = "provenance.json"
REQUIRED_METADATA_COLUMNS = (
    "pair_id",
    "drug_id",
    "context_id",
    "split",
    "dataset",
)
REQUIRED_PROVENANCE_FIELDS = (
    "schema_version",
    "dataset",
    "split_name",
    "split_hash",
    "model",
    "seed",
    "config_hash",
    "checkpoint_hash",
    "gene_panel_hash",
    "response_panel_hash",
    "command",
    "git_commit",
    "source_hashes",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class ArtifactValidationError(ValueError):
    """Raised when an artifact violates the versioned contract."""


@dataclass(frozen=True)
class ArtifactBundle:
    """A validated artifact loaded into memory."""

    pred: np.ndarray
    true: np.ndarray
    gene_ids: np.ndarray
    metadata: pd.DataFrame
    provenance: dict[str, Any]


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA256 digest of ``data``."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it fully into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON data using a stable UTF-8, key-sorted serialization."""

    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def sha256_gene_panel(gene_ids: Sequence[str] | np.ndarray) -> str:
    """Hash ordered gene identifiers with unambiguous length-prefix encoding."""

    normalized = _normalize_gene_ids(gene_ids)
    digest = hashlib.sha256(b"cytobridge-gene-panel-v1\0")
    for gene_id in normalized.tolist():
        encoded = gene_id.encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def sha256_files(paths: Mapping[str, str | Path]) -> dict[str, str]:
    """Hash named source files in sorted-name order."""

    hashes: dict[str, str] = {}
    for name in sorted(paths, key=str):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source hash names must be non-empty strings")
        hashes[name] = sha256_file(paths[name])
    return hashes


def _normalize_gene_ids(gene_ids: Sequence[str] | np.ndarray) -> np.ndarray:
    raw = np.asarray(gene_ids)
    if raw.ndim != 1:
        raise ArtifactValidationError(
            f"gene_ids must be one-dimensional; got {raw.shape}"
        )
    values: list[str] = []
    for index, value in enumerate(raw.tolist()):
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ArtifactValidationError(
                    f"gene_ids[{index}] is not valid UTF-8"
                ) from exc
        if not isinstance(value, str):
            raise ArtifactValidationError(f"gene_ids[{index}] must be a string")
        if not value or value != value.strip():
            raise ArtifactValidationError(
                f"gene_ids[{index}] must be non-empty and have no surrounding whitespace"
            )
        values.append(value)
    if len(set(values)) != len(values):
        raise ArtifactValidationError("gene_ids must be unique and order-preserving")
    return np.asarray(values, dtype=str)


def _normalize_matrix(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ArtifactValidationError(
            f"{name} must be two-dimensional; got {array.shape}"
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ArtifactValidationError(f"{name} must have at least one row and one gene")
    if array.dtype.kind not in "iuf":
        raise ArtifactValidationError(
            f"{name} must have a real numeric dtype; got {array.dtype}"
        )
    if not np.isfinite(array).all():
        raise ArtifactValidationError(f"{name} contains NaN or infinite values")
    return array


def _normalize_metadata(metadata: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    if not isinstance(metadata, pd.DataFrame):
        raise ArtifactValidationError("metadata must be a pandas DataFrame")
    if metadata.columns.has_duplicates:
        raise ArtifactValidationError("metadata columns must be unique")
    observed_columns = tuple(metadata.columns)
    if observed_columns != REQUIRED_METADATA_COLUMNS:
        raise ArtifactValidationError(
            "metadata columns must exactly match the contract order; "
            f"expected {list(REQUIRED_METADATA_COLUMNS)}, got {list(observed_columns)}"
        )
    if len(metadata) != n_rows:
        raise ArtifactValidationError(
            f"metadata has {len(metadata)} rows but predictions have {n_rows} rows"
        )

    normalized = metadata.copy()
    for column in REQUIRED_METADATA_COLUMNS:
        if normalized[column].isna().any():
            raise ArtifactValidationError(
                f"metadata column {column!r} contains null values"
            )
        values = normalized[column].astype(str)
        invalid = (
            values.eq("")
            | values.ne(values.str.strip())
            | values.str.contains(r"[\r\n]", regex=True)
        )
        if invalid.any():
            row = int(np.flatnonzero(invalid.to_numpy())[0])
            raise ArtifactValidationError(
                f"metadata column {column!r} has an invalid value at row {row}"
            )
        normalized[column] = values
    if normalized["pair_id"].duplicated().any():
        duplicate = normalized.loc[normalized["pair_id"].duplicated(), "pair_id"].iloc[
            0
        ]
        raise ArtifactValidationError(
            f"metadata pair_id values must be unique; got {duplicate!r}"
        )
    return normalized


def _nonempty_string(
    provenance: Mapping[str, Any], field: str, errors: list[str]
) -> None:
    value = provenance.get(field)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        errors.append(
            f"{field} must be a non-empty string without surrounding whitespace"
        )


def _sha256_field(
    provenance: Mapping[str, Any],
    field: str,
    errors: list[str],
    *,
    nullable: bool = False,
) -> None:
    value = provenance.get(field)
    if nullable and value is None:
        return
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        suffix = " or null" if nullable else ""
        errors.append(f"{field} must be a lowercase SHA256 digest{suffix}")


def _normalize_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ArtifactValidationError("provenance must be a mapping")
    data = dict(provenance)
    missing = [field for field in REQUIRED_PROVENANCE_FIELDS if field not in data]
    extra = sorted(set(data) - set(REQUIRED_PROVENANCE_FIELDS))
    errors: list[str] = []
    if missing:
        errors.append(f"missing required fields: {missing}")
    if extra:
        errors.append(f"unexpected fields: {extra}")
    if missing or extra:
        raise ArtifactValidationError("invalid provenance: " + "; ".join(errors))

    if data["schema_version"] != SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {SCHEMA_VERSION!r}; got {data['schema_version']!r}"
        )
    for field in ("dataset", "split_name", "model", "command"):
        _nonempty_string(data, field, errors)
    seed = data["seed"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        errors.append("seed must be a non-negative integer")
    else:
        data["seed"] = int(seed)

    _sha256_field(data, "split_hash", errors)
    _sha256_field(data, "config_hash", errors)
    _sha256_field(data, "checkpoint_hash", errors, nullable=True)
    _sha256_field(data, "gene_panel_hash", errors)
    _sha256_field(data, "response_panel_hash", errors)

    git_commit = data["git_commit"]
    if git_commit is not None and (
        not isinstance(git_commit, str) or _GIT_COMMIT_RE.fullmatch(git_commit) is None
    ):
        errors.append(
            "git_commit must be a lowercase 40- or 64-character digest or null"
        )

    source_hashes = data["source_hashes"]
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        errors.append("source_hashes must be a non-empty mapping")
    else:
        normalized_sources: dict[str, str] = {}
        for name in sorted(source_hashes, key=str):
            value = source_hashes[name]
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                errors.append("source_hashes keys must be non-empty strings")
                continue
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                errors.append(
                    f"source_hashes[{name!r}] must be a lowercase SHA256 digest"
                )
                continue
            normalized_sources[name] = value
        data["source_hashes"] = normalized_sources

    if errors:
        raise ArtifactValidationError("invalid provenance: " + "; ".join(errors))
    return data


def _validate_components(
    pred: np.ndarray,
    true: np.ndarray,
    gene_ids: Sequence[str] | np.ndarray,
    metadata: pd.DataFrame,
    provenance: Mapping[str, Any],
) -> ArtifactBundle:
    pred_array = _normalize_matrix("pred", pred)
    true_array = _normalize_matrix("true", true)
    if pred_array.shape != true_array.shape:
        raise ArtifactValidationError(
            f"pred and true shapes differ: {pred_array.shape} != {true_array.shape}"
        )
    genes = _normalize_gene_ids(gene_ids)
    if len(genes) != pred_array.shape[1]:
        raise ArtifactValidationError(
            f"gene_ids has {len(genes)} entries but matrices have {pred_array.shape[1]} genes"
        )
    metadata_frame = _normalize_metadata(metadata, pred_array.shape[0])
    provenance_data = _normalize_provenance(provenance)

    datasets = metadata_frame["dataset"].unique().tolist()
    if datasets != [provenance_data["dataset"]]:
        raise ArtifactValidationError(
            "metadata dataset values must all equal provenance.dataset; "
            f"got {datasets!r} versus {provenance_data['dataset']!r}"
        )
    splits = metadata_frame["split"].unique().tolist()
    if splits != [provenance_data["split_name"]]:
        raise ArtifactValidationError(
            "metadata split values must all equal provenance.split_name; "
            f"got {splits!r} versus {provenance_data['split_name']!r}"
        )
    observed_gene_hash = sha256_gene_panel(genes)
    if provenance_data["gene_panel_hash"] != observed_gene_hash:
        raise ArtifactValidationError(
            "provenance.gene_panel_hash does not match the ordered gene_ids array: "
            f"expected {observed_gene_hash}"
        )
    return ArtifactBundle(
        pred=pred_array,
        true=true_array,
        gene_ids=genes,
        metadata=metadata_frame,
        provenance=provenance_data,
    )


def _write_bytes(path: Path, data: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_npz(
    path: Path, pred: np.ndarray, true: np.ndarray, gene_ids: np.ndarray
) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, pred=pred, true=true, gene_ids=gene_ids)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def write_artifact(
    directory: str | Path,
    *,
    pred: np.ndarray,
    true: np.ndarray,
    gene_ids: Sequence[str] | np.ndarray,
    metadata: pd.DataFrame,
    provenance: Mapping[str, Any],
) -> ArtifactBundle:
    """Validate and atomically publish a new immutable artifact directory.

    The destination must not already exist. This avoids ambiguous partial updates and
    makes each artifact content-addressable through the hashes recorded in provenance.
    """

    bundle = _validate_components(pred, true, gene_ids, metadata, provenance)
    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(f"artifact destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-", dir=str(destination.parent)
        )
    )
    try:
        _write_npz(
            stage / PREDICTIONS_FILE,
            bundle.pred,
            bundle.true,
            bundle.gene_ids,
        )
        csv_text = bundle.metadata.to_csv(index=False, lineterminator="\n")
        _write_bytes(stage / METADATA_FILE, csv_text.encode("utf-8"))
        provenance_text = (
            json.dumps(
                bundle.provenance,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        _write_bytes(stage / PROVENANCE_FILE, provenance_text.encode("utf-8"))
        _fsync_directory(stage)

        validated = validate_artifact(stage)
        os.replace(stage, destination)
        _fsync_directory(destination.parent)
        return validated
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def validate_artifact(directory: str | Path) -> ArtifactBundle:
    """Load and strictly validate an artifact directory."""

    root = Path(directory)
    if not root.is_dir():
        raise ArtifactValidationError(f"artifact directory does not exist: {root}")
    required_paths = {
        PREDICTIONS_FILE: root / PREDICTIONS_FILE,
        METADATA_FILE: root / METADATA_FILE,
        PROVENANCE_FILE: root / PROVENANCE_FILE,
    }
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    if missing:
        raise ArtifactValidationError(f"artifact is missing required files: {missing}")
    unexpected = sorted(
        path.name for path in root.iterdir() if path.name not in required_paths
    )
    if unexpected:
        raise ArtifactValidationError(
            f"artifact contains unexpected entries: {unexpected}"
        )

    try:
        with np.load(required_paths[PREDICTIONS_FILE], allow_pickle=False) as archive:
            keys = set(archive.files)
            expected = {"pred", "true", "gene_ids"}
            if keys != expected:
                raise ArtifactValidationError(
                    "predictions.npz must contain exactly pred, true, gene_ids; "
                    f"got {sorted(keys)}"
                )
            pred = np.array(archive["pred"], copy=True)
            true = np.array(archive["true"], copy=True)
            gene_ids = np.array(archive["gene_ids"], copy=True)
    except ArtifactValidationError:
        raise
    except Exception as exc:
        raise ArtifactValidationError(f"could not read predictions.npz: {exc}") from exc

    try:
        metadata = pd.read_csv(
            required_paths[METADATA_FILE],
            dtype=str,
            keep_default_na=False,
            na_filter=False,
        )
    except Exception as exc:
        raise ArtifactValidationError(f"could not read metadata.csv: {exc}") from exc
    try:
        with required_paths[PROVENANCE_FILE].open(encoding="utf-8") as handle:
            provenance = json.load(handle)
    except Exception as exc:
        raise ArtifactValidationError(f"could not read provenance.json: {exc}") from exc
    return _validate_components(pred, true, gene_ids, metadata, provenance)


load_artifact = validate_artifact
