from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd

from eval.baselines.ridge_pseudobulk import (
    build_design_matrix,
    build_parser,
    fit_final_ridge,
    fit_ridge,
    read_split_drugs,
    run_ridge_refit_predict,
    run_ridge_selection,
    smiles_to_morgan,
    validate_split_metadata,
)


def _meta(drugs, smiles, contexts):
    return pd.DataFrame(
        {"drug_id": drugs, "canonical_smiles": smiles, "context_id": contexts}
    )


def test_read_split_drugs_accepts_split_keys_and_smiles_entries():
    splits = {"train": ["CCO"], "test_drugs": [{"drug_id": "drug_c"}]}
    drug_smiles = {"drug_a": "CCO", "drug_b": "CCO", "drug_c": "CCN"}
    valid_drugs = set(drug_smiles)

    train = read_split_drugs(splits, "train", valid_drugs, drug_smiles, "DMSO")
    test = read_split_drugs(splits, "test", valid_drugs, drug_smiles, "DMSO")

    assert train == {"drug_a", "drug_b"}
    assert test == {"drug_c"}
    assert not (train & test)


def test_split_validation_rejects_other_test_drugs_in_fit_pool():
    train = _meta(["a"], ["CCO"], ["A"])
    validation = _meta(["b"], ["CCN"], ["A"])
    test = _meta(["a", "c"], ["CCO", "CCC"], ["B", "A"])

    try:
        validate_split_metadata(train, validation, test)
    except ValueError as exc:
        assert "train/test leakage" in str(exc)
        assert "a" in str(exc)
    else:
        raise AssertionError("test drugs must never enter a fit split")


def test_split_validation_rejects_smiles_alias_leakage():
    train = _meta(["alias_a"], ["CCO"], ["A"])
    validation = _meta(["alias_b"], ["CCO"], ["A"])

    try:
        validate_split_metadata(train, validation)
    except ValueError as exc:
        assert "canonical_smiles_overlap" in str(exc)
    else:
        raise AssertionError("canonical-SMILES aliases must stay in one split")


def test_design_matrix_uses_fingerprint_and_training_context_order():
    metadata = pd.DataFrame({"drug_id": ["d2", "d1"], "context_id": ["B", "A"]})
    fingerprints = {
        "d1": np.array([1.0, 0.0, 1.0]),
        "d2": np.array([0.0, 1.0, 0.0]),
    }

    design = build_design_matrix(metadata, fingerprints, ["A", "B"])

    assert np.array_equal(design[0], [0.0, 1.0, 0.0, 0.0, 1.0])
    assert np.array_equal(design[1], [1.0, 0.0, 1.0, 1.0, 0.0])


def test_fit_apis_have_no_test_target_argument():
    assert all("test" not in name for name in inspect.signature(fit_ridge).parameters)
    assert all(
        "test" not in name for name in inspect.signature(fit_final_ridge).parameters
    )
    assert all(
        "test" not in name for name in inspect.signature(run_ridge_selection).parameters
    )
    assert "test_targets_path" not in inspect.signature(
        run_ridge_refit_predict
    ).parameters


def test_ridge_selection_cli_has_no_test_target_option():
    parser = build_parser()
    selection_parser = next(
        action.choices["select"]
        for action in parser._actions
        if getattr(action, "choices", None) and "select" in action.choices
    )
    option_strings = {
        option
        for action in selection_parser._actions
        for option in action.option_strings
    }
    assert "--test-targets" not in option_strings
    assert "--test-metadata" not in option_strings


def test_final_ridge_refits_only_explicit_train_and_validation_rows():
    train_x = np.array([[0.0], [1.0]])
    train_y = np.array([[0.0], [1.0]])
    validation_x = np.array([[2.0]])
    validation_y = np.array([[2.0]])

    model = fit_final_ridge(train_x, train_y, validation_x, validation_y, alpha=0.1)

    assert model.n_features_in_ == 1
    assert np.isfinite(model.predict([[3.0]])).all()


def test_morgan_defaults_are_radius2_2048_bits():
    signature = inspect.signature(smiles_to_morgan)
    assert signature.parameters["radius"].default == 2
    assert signature.parameters["n_bits"].default == 2048


def test_selection_then_refit_predict_keeps_test_truth_outside_model_api(
    tmp_path, monkeypatch
):
    genes = np.array(["G1", "G2", "G3"])
    contexts = ["A", "B"]
    split_drugs = {
        "train": ["a", "b"],
        "validation": ["c", "d"],
        "test": ["e", "f"],
    }
    paths = {}
    smiles_rows = []
    for split, drugs in split_drugs.items():
        rows = [
            {
                "pair_id": f"{split}-{context}-{drug}",
                "drug_id": drug,
                "context_id": context,
                "canonical_smiles": f"smiles-{drug}",
            }
            for context in contexts
            for drug in drugs
        ]
        metadata = pd.DataFrame(rows)
        metadata_path = tmp_path / f"{split}.csv"
        metadata.to_csv(metadata_path, index=False)
        paths[f"{split}_metadata"] = metadata_path
        if split != "test":
            target = np.asarray(
                [
                    [float(ord(row["drug_id"])), float(index + 1), float(index % 2)]
                    for index, row in enumerate(rows)
                ]
            )
            target_path = tmp_path / f"{split}.npz"
            np.savez_compressed(target_path, true=target, gene_ids=genes)
            paths[f"{split}_targets"] = target_path
        smiles_rows.extend(
            {"drug_id": drug, "canonical_smiles": f"smiles-{drug}"}
            for drug in drugs
        )
    smiles_path = tmp_path / "smiles.csv"
    pd.DataFrame(smiles_rows).to_csv(smiles_path, index=False)
    panels_path = tmp_path / "panels.json"
    panels_path.write_text(json.dumps({"A": [0, 1, 2], "B": [0, 1, 2]}))

    def fake_fingerprints(tables, radius=2, n_bits=2048):
        del radius, n_bits
        drugs = sorted(
            set(pd.concat(tables, ignore_index=True)["drug_id"].astype(str))
        )
        return {
            drug: np.array([float(ord(drug)), float(ord(drug) % 3)])
            for drug in drugs
        }

    monkeypatch.setattr(
        "eval.baselines.ridge_pseudobulk.build_fingerprint_map", fake_fingerprints
    )
    selection_path = tmp_path / "selection.json"
    trials_path = tmp_path / "trials.csv"
    selection = run_ridge_selection(
        train_targets_path=paths["train_targets"],
        train_metadata_path=paths["train_metadata"],
        validation_targets_path=paths["validation_targets"],
        validation_metadata_path=paths["validation_metadata"],
        smiles_csv_path=smiles_path,
        gene_panels_path=panels_path,
        output_selection_path=selection_path,
        output_trials_path=trials_path,
    )
    predictions_path = tmp_path / "predictions.npz"
    result = run_ridge_refit_predict(
        selection_path=selection_path,
        train_targets_path=paths["train_targets"],
        train_metadata_path=paths["train_metadata"],
        validation_targets_path=paths["validation_targets"],
        validation_metadata_path=paths["validation_metadata"],
        test_metadata_path=paths["test_metadata"],
        smiles_csv_path=smiles_path,
        output_predictions_path=predictions_path,
    )

    assert selection["test_artifacts_opened_during_selection"] is False
    assert result["test_targets_opened"] is False
    with np.load(predictions_path, allow_pickle=False) as predictions:
        assert predictions["pred"].shape == (4, 3)
        assert predictions["pair_ids"].tolist() == pd.read_csv(
            paths["test_metadata"]
        )["pair_id"].tolist()
