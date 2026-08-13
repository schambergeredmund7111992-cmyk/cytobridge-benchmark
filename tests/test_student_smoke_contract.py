from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]


def test_smoke_data_and_model_entrypoints_run(tmp_path):
    create_script = REPO / "scripts" / "create_smoke_data.py"
    model_script = REPO / "scripts" / "run_model_smoke.py"
    assert create_script.exists()
    assert model_script.exists()

    subprocess.run(
        [sys.executable, str(create_script), "--out", str(tmp_path)],
        cwd=REPO,
        check=True,
    )
    expected = [
        "splits/sciplex_train.csv",
        "splits/sciplex_val.csv",
        "splits/sciplex_test.csv",
        "sciplex_scgpt_emb.npy",
        "sciplex_molformer_emb.npz",
        "splits/sciplex_train_treated_counts.npy",
        "splits/sciplex_train_control_counts.npy",
        "splits/sciplex_train_pathway_gsea.npy",
    ]
    for rel in expected:
        assert (tmp_path / rel).exists(), rel

    subprocess.run(
        [sys.executable, str(model_script), "--data-dir", str(tmp_path)],
        cwd=REPO,
        check=True,
    )


def test_default_training_config_does_not_require_missing_pathway_init():
    cfg = yaml.safe_load((REPO / "configs" / "train" / "v1.yaml").read_text())
    assert cfg["model"]["pathway_init_path"] is None


def test_optional_clis_show_help_without_heavy_single_cell_dependencies():
    scripts = [
        REPO / "data" / "build_external_split.py",
        REPO / "agent" / "case_studies" / "ipf.py",
        REPO / "agent" / "case_studies" / "gbm.py",
    ]
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=True,
        )
        assert "usage:" in result.stdout


def test_scgpt_cache_builder_is_valid_shell_and_has_help() -> None:
    script = REPO / "scripts" / "build_sciplex_scgpt_caches.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    result = subprocess.run(
        ["bash", str(script), "--help"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--gpu INDEX --ckpt DIR" in result.stdout
    assert "--scgpt-python PATH" in result.stdout
    assert "/home/zg.peng" not in script.read_text()
