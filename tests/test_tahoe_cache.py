from __future__ import annotations

import pandas as pd
import pytest

from data.build_tahoe_cache import build_cache


def test_build_tahoe_cache_requires_smiles_csv_when_embedding_cache_is_requested(
    tmp_path, monkeypatch
):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    cache_dir = tmp_path / "cache"
    manifest = pd.DataFrame(
        {
            "cell_idx": [0],
            "control_cell_idx": [1],
            "drug_id": ["drugA"],
            "cell_line": ["cellA"],
        }
    )
    (splits_dir / "tahoe_external_1.parquet").write_text("placeholder")
    (splits_dir / "tahoe_external_1.h5ad").write_text("placeholder")
    monkeypatch.setattr(pd, "read_parquet", lambda path: manifest)

    with pytest.raises(FileNotFoundError, match="tahoe_external_1_smiles.csv"):
        build_cache(
            split="external_1",
            splits_dir=splits_dir,
            cache_dir=cache_dir,
            skip_embeddings=False,
        )
