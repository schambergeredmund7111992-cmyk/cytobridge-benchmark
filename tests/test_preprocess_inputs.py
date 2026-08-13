from __future__ import annotations

import pytest

from data.preprocess import load_smiles_map


def test_load_smiles_map_reports_actionable_missing_file(tmp_path):
    missing = tmp_path / "sciplex3_drugs.csv"

    with pytest.raises(FileNotFoundError, match="drug_id,smiles"):
        load_smiles_map(missing)
