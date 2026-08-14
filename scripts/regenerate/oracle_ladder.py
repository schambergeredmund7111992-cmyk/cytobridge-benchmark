"""Fig. 5: the oracle ladder.

Every rung is a drug-discrimination AUC measured with the same frozen
off-diagonal control on the nine held-out compounds. Oracles are handed the
measured responses of training compounds only; none sees a held-out response.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.linear_model import Ridge

from eval.metrics import drug_discrimination_score

CELL_INDEX = {"A549": 0, "K562": 1, "MCF7": 2}


def _ddc(pred, true, cl) -> dict:
    return drug_discrimination_score(
        np.asarray(pred, dtype=float),
        np.asarray(true, dtype=float),
        np.asarray(cl),
        top_k=50,
        metric="pearson",
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def _morgan(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def hindsight_retrieval(
    test_true, test_meta, train_responses, train_drugs
) -> tuple[np.ndarray, list[str]]:
    """Per anchor: pick the training compound whose measured response best
    matches the anchor's measured response (in that cell line)."""
    pred = np.zeros_like(test_true, dtype=float)
    chosen: list[str] = []
    for i in range(test_true.shape[0]):
        cell = str(test_meta["cell_line"].iloc[i])
        column = train_responses[:, CELL_INDEX[cell], :]
        sims = [_pearson(test_true[i], column[j]) for j in range(column.shape[0])]
        best = int(np.argmax(sims))
        pred[i] = column[best]
        chosen.append(str(train_drugs["drug_id"].iloc[best]))
    return pred, chosen


def tanimoto_nearest(
    test_true, test_meta, train_responses, test_drugs, train_drugs
) -> np.ndarray:
    """Per anchor: pick the training compound with the most similar Morgan
    fingerprint to the held-out compound (Tanimoto)."""
    train_fps = np.stack(
        [_morgan(smiles) for smiles in train_drugs["canonical_smiles"]]
    )
    smiles_by_drug = dict(
        zip(test_drugs["drug_id"].astype(str), test_drugs["canonical_smiles"].astype(str))
    )
    pred = np.zeros_like(test_true, dtype=float)
    for i in range(test_true.shape[0]):
        test_fp = _morgan(smiles_by_drug[str(test_meta["drug"].iloc[i])])
        sims = 1.0 - np.array(
            [DataStructs.TanimotoSimilarity(_as_bitvect(test_fp), _as_bitvect(fp))
             for fp in train_fps]
        )
        best = int(np.argmin(sims))
        cell = str(test_meta["cell_line"].iloc[i])
        pred[i] = train_responses[best, CELL_INDEX[cell], :]
    return pred


def _as_bitvect(arr: np.ndarray):
    from rdkit.DataStructs import cDataStructs

    bitvect = cDataStructs.ExplicitBitVect(int(len(arr)))
    for position in np.flatnonzero(arr):
        bitvect.SetBit(int(position))
    return bitvect


def morgan_ridge(
    test_true, test_meta, train_responses, test_drugs, train_drugs, alpha: float = 1.0
) -> np.ndarray:
    """Per-gene ridge regression on [Morgan FP | cell-line one-hot] fitted on
    all 160 training compounds (three cell lines each)."""
    train_fps = np.stack(
        [_morgan(smiles) for smiles in train_drugs["canonical_smiles"]]
    )
    # train design: 160 x 3 samples in cell-line order
    train_design = []
    train_targets = []
    for j in range(train_fps.shape[0]):
        for cell_index in range(3):
            onehot = np.zeros(3)
            onehot[cell_index] = 1.0
            train_design.append(np.concatenate([train_fps[j], onehot]))
            train_targets.append(train_responses[j, cell_index, :])
    design = np.asarray(train_design)
    targets = np.asarray(train_targets)  # [480, 3000]

    smiles_by_drug = dict(
        zip(test_drugs["drug_id"].astype(str), test_drugs["canonical_smiles"].astype(str))
    )
    test_design = []
    for i in range(test_true.shape[0]):
        onehot = np.zeros(3)
        onehot[CELL_INDEX[str(test_meta["cell_line"].iloc[i])]] = 1.0
        test_fp = _morgan(smiles_by_drug[str(test_meta["drug"].iloc[i])])
        test_design.append(np.concatenate([test_fp, onehot]))
    test_design = np.asarray(test_design)

    pred = np.zeros_like(test_true, dtype=float)
    for gene in range(targets.shape[1]):
        model = Ridge(alpha=alpha)
        model.fit(design, targets[:, gene])
        pred[:, gene] = model.predict(test_design)
    return pred


def target_matched(
    test_true, test_meta, train_responses, test_drugs, train_drugs
) -> tuple[np.ndarray, np.ndarray]:
    """Mean measured response of the training compounds sharing the held-out
    compound's annotated pharmacological target; NaN rows mark anchors without
    a same-target training compound (scored separately)."""
    pred = np.full_like(test_true, np.nan, dtype=float)
    scored = np.zeros(test_true.shape[0], dtype=bool)
    target_by_drug = dict(
        zip(test_drugs["drug_id"].astype(str), test_drugs["vendor_target"].astype(str))
    )
    for i in range(test_true.shape[0]):
        target = target_by_drug[str(test_meta["drug"].iloc[i])]
        same = train_drugs["vendor_target"].astype(str).eq(target)
        if not same.any():
            continue
        cell = str(test_meta["cell_line"].iloc[i])
        pred[i] = train_responses[same.to_numpy(), CELL_INDEX[cell], :].mean(axis=0)
        scored[i] = True
    return pred, scored


def derangement_null(
    test_true,
    test_meta,
    train_responses,
    train_drugs,
    *,
    n_perm: int = 200,
    seed: int = 7301,
    scorer: Callable | None = None,
) -> dict:
    """The hindsight maximization run against deranged held-out labels."""
    scorer = scorer or hindsight_retrieval
    cl = test_meta["cell_line"].astype(str).to_numpy()
    drugs = test_meta["drug"].astype(str).to_numpy()
    unique = sorted(set(drugs.tolist()), key=repr)
    row_by = {(str(drug), str(cell)): i
              for i, (drug, cell) in enumerate(zip(drugs, cl))}
    rng = np.random.default_rng(seed)
    null_aucs = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        # derangement of the drug labels (reject fixed points)
        while True:
            perm = rng.permutation(len(unique))
            if not np.any(perm == np.arange(len(unique))):
                break
        mapping = {drug: unique[int(p)] for drug, p in zip(unique, perm)}
        deranged_true = np.zeros_like(test_true)
        for i in range(test_true.shape[0]):
            source = row_by[(mapping[str(drugs[i])], str(cl[i]))]
            deranged_true[i] = test_true[int(source)]
        pred, _ = scorer(
            deranged_true, test_meta, train_responses, train_drugs
        )
        null_aucs[k] = float(_ddc(pred, deranged_true, cl)["specificity_auc"])
    observed_pred, _ = scorer(test_true, test_meta, train_responses, train_drugs)
    observed = float(_ddc(observed_pred, test_true, cl)["specificity_auc"])
    p_value = float((int(np.sum(null_aucs >= observed)) + 1) / (n_perm + 1))
    return {
        "observed": observed,
        "null_mean": float(null_aucs.mean()),
        "null": null_aucs,
        "p_value": p_value,
        "seed": seed,
        "n_perm": n_perm,
    }


def build_oracle_ladder(loader, pooled_true, meta) -> dict:
    """Run every oracle rung and return {entry_id: value}."""
    oracle = loader.oracle_inputs()
    out: dict = {}
    if oracle is None:
        return out
    test_drugs = oracle["drugs_172"][
        oracle["drugs_172"]["drug_id"].isin(meta["drug"].astype(str))
    ].reset_index(drop=True)
    if len(test_drugs) != 9:
        return out

    hindsight_pred, _ = hindsight_retrieval(
        pooled_true, meta, oracle["responses"], oracle["training_drugs"]
    )
    out["fig5.hindsight"] = float(
        _ddc(hindsight_pred, pooled_true, meta["cell_line"])["specificity_auc"]
    )
    tanimoto_pred = tanimoto_nearest(
        pooled_true, meta, oracle["responses"], test_drugs, oracle["training_drugs"]
    )
    out["fig5.tanimoto_nn"] = float(
        _ddc(tanimoto_pred, pooled_true, meta["cell_line"])["specificity_auc"]
    )
    ridge_pred = morgan_ridge(
        pooled_true, meta, oracle["responses"], test_drugs, oracle["training_drugs"]
    )
    out["fig5.morgan_ridge"] = float(
        _ddc(ridge_pred, pooled_true, meta["cell_line"])["specificity_auc"]
    )
    target_pred, scored = target_matched(
        pooled_true, meta, oracle["responses"], test_drugs, oracle["training_drugs"]
    )
    if scored.sum() >= 2:
        cl = meta["cell_line"].astype(str).to_numpy()
        out["fig5.target_matched"] = float(
            _ddc(target_pred[scored], pooled_true[scored], cl[scored])["specificity_auc"]
        )
    deranged = derangement_null(pooled_true, meta, oracle["responses"], oracle["training_drugs"])
    out["fig5.derangement_mean"] = deranged["null_mean"]
    out["fig5.hindsight_p"] = deranged["p_value"]
    return out
