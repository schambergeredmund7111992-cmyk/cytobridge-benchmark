"""Resampling and multiplicity utilities for the frozen drug benchmark."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

from eval.metrics import (
    _finite_correlation,
    _resolve_panel,
    conditional_rank_score,
)


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate, percentile interval, and the complete bootstrap distribution."""

    estimate: float
    ci_low: float
    ci_high: float
    draws: np.ndarray

    @property
    def ci(self) -> tuple[float, float]:
        return self.ci_low, self.ci_high


@dataclass(frozen=True)
class PermutationTestResult:
    """Observed conditional rank and its exact or Monte Carlo randomization null."""

    observed: float
    p_value: float
    draws: np.ndarray
    exhaustive: bool
    total_label_permutations: int

    @property
    def null_distribution(self) -> np.ndarray:
        return self.draws

    @property
    def n_permutations(self) -> int:
        return int(self.draws.size)


def _positive_integer(value: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _validate_confidence(confidence: float) -> float:
    confidence = float(confidence)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and strictly between zero and one.")
    return confidence


def _validate_frame(
    data: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    value_col: str,
) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    required = list(dict.fromkeys([*key_columns, value_col]))
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"data is missing required columns: {missing}.")
    if data.empty:
        raise ValueError("data must contain at least one row.")

    frame = data.loc[:, required].copy()
    if frame[list(key_columns)].isna().any().any():
        raise ValueError("identifier columns must not contain missing values.")
    for column in key_columns:
        frame[column] = frame[column].map(str)
    duplicate = frame.duplicated(list(key_columns), keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, list(key_columns)].drop_duplicates().head(5)
        raise ValueError(
            "duplicate benchmark rows are not allowed; repeated keys include "
            f"{examples.to_dict(orient='records')}."
        )
    try:
        values = pd.to_numeric(frame[value_col], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{value_col!r} must be numeric.") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{value_col!r} must contain only finite values.")
    frame[value_col] = values
    return frame


def _assert_balanced(
    frame: pd.DataFrame,
    *,
    cluster_columns: Sequence[str],
    member_col: str,
    label: str,
) -> tuple[str, ...]:
    member_sets = frame.groupby(list(cluster_columns), sort=False)[member_col].agg(
        frozenset
    )
    expected = member_sets.iloc[0]
    if any(members != expected for members in member_sets.iloc[1:]):
        raise ValueError(
            f"unbalanced {label}: every {'/'.join(cluster_columns)} must have the same "
            f"{member_col} set."
        )
    if not expected:
        raise ValueError(f"balanced {label} must contain at least one {member_col}.")
    return tuple(sorted(expected))


def _percentile_result(
    estimate: float,
    draws: np.ndarray,
    confidence: float,
) -> BootstrapResult:
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return BootstrapResult(
        estimate=float(estimate),
        ci_low=float(low),
        ci_high=float(high),
        draws=draws,
    )


def drug_cluster_bootstrap(
    per_pair: pd.DataFrame,
    value_col: str = "conditional_accuracy",
    *,
    drug_col: str = "drug_id",
    context_col: str = "context_id",
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 7301,
) -> BootstrapResult:
    """Bootstrap whole drugs while retaining every context belonging to a sampled drug.

    .. warning::
       This is NOT a valid interval for ``conditional_accuracy``. That endpoint is an
       asymmetric U-statistic: every held-out drug is simultaneously an anchor and a negative
       in every other drug's candidate panel, so the per-drug scores are dependent. Freezing
       them and resampling the numbers estimates only the anchor half of the Hajek projection
       and drops the negative half plus its covariance. In simulation at n = 28 drugs the
       nominal 95% interval covers 70.8% of the time. Use ``drug_jackknife_ci`` for any
       U-statistic endpoint; this function remains valid only for endpoints whose per-drug
       value does not depend on the other drugs (pair-own Spearman, per-pair Pearson, MSE).
    """
    n_boot = _positive_integer(n_boot, "n_boot")
    confidence = _validate_confidence(confidence)
    frame = _validate_frame(
        per_pair,
        key_columns=[drug_col, context_col],
        value_col=value_col,
    )
    _assert_balanced(
        frame,
        cluster_columns=[drug_col],
        member_col=context_col,
        label="drug/context panel",
    )
    per_drug = (
        frame.groupby(drug_col, sort=True)[value_col].mean().to_numpy(dtype=float)
    )
    if per_drug.size < 2:
        raise ValueError("drug-cluster bootstrap requires at least two drugs.")

    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, per_drug.size, size=(n_boot, per_drug.size))
    draws = per_drug[sampled].mean(axis=1)
    return _percentile_result(float(per_drug.mean()), draws, confidence)


@dataclass(frozen=True)
class JackknifeResult:
    """Point estimate, delete-one-drug standard error, and the leave-one-out replicates."""

    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    leave_one_out: np.ndarray
    drugs: tuple[object, ...]

    @property
    def ci(self) -> tuple[float, float]:
        return self.ci_low, self.ci_high


def _drug_macro(
    pred: np.ndarray,
    true: np.ndarray,
    context_ids: np.ndarray,
    drug_ids: np.ndarray,
    gene_panels,
    metric: str,
) -> float:
    result = conditional_rank_score(
        pred, true, context_ids, drug_ids, gene_panels, metric=metric
    )
    return float(result.summary["conditional_accuracy_drug_macro"])


def drug_jackknife_ci(
    pred: np.ndarray,
    true: np.ndarray,
    context_ids: Sequence[object],
    drug_ids: Sequence[object],
    gene_panels,
    *,
    metric: str = "pearson",
    confidence: float = 0.95,
) -> JackknifeResult:
    """Delete-one-drug jackknife interval for the conditional-accuracy U-statistic.

    Deleting a drug removes it from the anchor set AND from every other drug's negative pool,
    so the endpoint is fully recomputed on each subset with the same frozen scoring code. This
    captures both terms of the Hajek projection -- the drug's contribution as an anchor and its
    contribution as a distractor -- which ``drug_cluster_bootstrap`` cannot see.

    Correctness evidence: on a three-drug kernel whose values were computed by hand, the point
    estimate, the leave-one-out replicates and the standard error all reproduce the hand
    values exactly, and the jackknife SE is 2.46x the naive one. In a coverage simulation whose
    generative process couples anchor and distractor roles, nominal 95% intervals cover 93.2%
    (n = 28 drugs) and 94.8% (n = 60) against 70.8% and 75.2% for ``drug_cluster_bootstrap``.
    """
    confidence = _validate_confidence(confidence)
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    contexts = np.asarray(context_ids)
    drugs = np.asarray(drug_ids)
    unique = tuple(sorted(set(drugs.tolist()), key=repr))
    n = len(unique)
    if n < 3:
        raise ValueError("the delete-one-drug jackknife requires at least three held-out drugs.")

    estimate = _drug_macro(pred, true, contexts, drugs, gene_panels, metric)
    loo = np.empty(n, dtype=float)
    for position, drug in enumerate(unique):
        keep = drugs != drug
        loo[position] = _drug_macro(
            pred[keep], true[keep], contexts[keep], drugs[keep], gene_panels, metric
        )

    variance = (n - 1) / n * float(((loo - loo.mean()) ** 2).sum())
    standard_error = float(math.sqrt(max(variance, 0.0)))
    half = float(_scipy_stats.norm.ppf(0.5 * (1.0 + confidence))) * standard_error
    return JackknifeResult(
        estimate=estimate,
        standard_error=standard_error,
        ci_low=estimate - half,
        ci_high=estimate + half,
        leave_one_out=loo,
        drugs=unique,
    )


def paired_hierarchical_bootstrap(
    per_pair: pd.DataFrame,
    model_a: object,
    model_b: object,
    value_col: str = "conditional_accuracy",
    *,
    model_col: str = "model",
    seed_col: str = "seed",
    drug_col: str = "drug_id",
    context_col: str = "context_id",
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 7301,
) -> BootstrapResult:
    """Hierarchically resample seeds then drugs for the paired ``model_a - model_b`` effect.

    Pairing occurs before resampling. Each selected seed/drug cluster therefore carries both
    models and all of its contexts; a missing partner is an error rather than an implicit drop.
    """
    n_boot = _positive_integer(n_boot, "n_boot")
    confidence = _validate_confidence(confidence)
    model_a, model_b = str(model_a), str(model_b)
    if model_a == model_b:
        raise ValueError("model_a and model_b must identify different models.")
    frame = _validate_frame(
        per_pair,
        key_columns=[model_col, seed_col, drug_col, context_col],
        value_col=value_col,
    )
    frame = frame[frame[model_col].isin([model_a, model_b])].copy()
    present = set(frame[model_col])
    if present != {model_a, model_b}:
        raise ValueError(
            f"both requested models must be present; found {sorted(present)}."
        )

    pivot = frame.pivot(
        index=[seed_col, drug_col, context_col],
        columns=model_col,
        values=value_col,
    )
    if pivot[[model_a, model_b]].isna().any().any():
        raise ValueError(
            "unpaired model rows: every seed/drug/context must contain both requested models."
        )
    paired = (pivot[model_a] - pivot[model_b]).rename("difference").reset_index()
    _assert_balanced(
        paired,
        cluster_columns=[seed_col, drug_col],
        member_col=context_col,
        label="seed/drug/context panel",
    )
    drugs = _assert_balanced(
        paired,
        cluster_columns=[seed_col],
        member_col=drug_col,
        label="seed/drug panel",
    )
    seeds = tuple(sorted(paired[seed_col].unique()))
    if len(seeds) < 2 or len(drugs) < 2:
        raise ValueError(
            "hierarchical bootstrap requires at least two seeds and two drugs."
        )

    seed_drug = (
        paired.groupby([seed_col, drug_col], sort=True)["difference"]
        .mean()
        .unstack(drug_col)
        .loc[list(seeds), list(drugs)]
        .to_numpy(dtype=float)
    )
    estimate = float(seed_drug.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for draw_index in range(n_boot):
        sampled_seeds = rng.integers(0, len(seeds), size=len(seeds))
        seed_means = np.empty(len(seeds), dtype=float)
        for output_index, sampled_seed in enumerate(sampled_seeds):
            sampled_drugs = rng.integers(0, len(drugs), size=len(drugs))
            seed_means[output_index] = seed_drug[sampled_seed, sampled_drugs].mean()
        draws[draw_index] = seed_means.mean()
    return _percentile_result(estimate, draws, confidence)


def paired_seed_drug_bootstrap(*args, **kwargs) -> BootstrapResult:
    """Alias with an explicit name for :func:`paired_hierarchical_bootstrap`."""
    return paired_hierarchical_bootstrap(*args, **kwargs)


def seed_drug_hierarchical_bootstrap(
    per_pair: pd.DataFrame,
    value_col: str = "conditional_accuracy",
    *,
    seed_col: str = "seed",
    drug_col: str = "drug_id",
    context_col: str = "context_id",
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 7301,
) -> BootstrapResult:
    """Resample training seeds and whole drugs for one model's drug-macro estimate."""
    n_boot = _positive_integer(n_boot, "n_boot")
    confidence = _validate_confidence(confidence)
    frame = _validate_frame(
        per_pair,
        key_columns=[seed_col, drug_col, context_col],
        value_col=value_col,
    )
    _assert_balanced(
        frame,
        cluster_columns=[seed_col, drug_col],
        member_col=context_col,
        label="seed/drug/context panel",
    )
    drugs = _assert_balanced(
        frame,
        cluster_columns=[seed_col],
        member_col=drug_col,
        label="seed/drug panel",
    )
    seeds = tuple(sorted(frame[seed_col].unique()))
    if len(seeds) < 2 or len(drugs) < 2:
        raise ValueError(
            "hierarchical bootstrap requires at least two seeds and two drugs."
        )
    seed_drug = (
        frame.groupby([seed_col, drug_col], sort=True)[value_col]
        .mean()
        .unstack(drug_col)
        .loc[list(seeds), list(drugs)]
        .to_numpy(dtype=float)
    )
    estimate = float(seed_drug.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for draw_index in range(n_boot):
        sampled_seeds = rng.integers(0, len(seeds), size=len(seeds))
        seed_means = np.empty(len(seeds), dtype=float)
        for output_index, sampled_seed in enumerate(sampled_seeds):
            sampled_drugs = rng.integers(0, len(drugs), size=len(drugs))
            seed_means[output_index] = seed_drug[sampled_seed, sampled_drugs].mean()
        draws[draw_index] = seed_means.mean()
    return _percentile_result(estimate, draws, confidence)


def _conditional_similarity_matrices(
    pred: np.ndarray,
    true: np.ndarray,
    contexts: np.ndarray,
    drugs: np.ndarray,
    drug_order: tuple[str, ...],
    gene_panels: Mapping[str, Sequence[int]] | Sequence[int],
    metric: str,
) -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for context in sorted(set(contexts.tolist())):
        lookup = {
            drug: int(np.flatnonzero((contexts == context) & (drugs == drug))[0])
            for drug in drug_order
        }
        indices = np.asarray([lookup[drug] for drug in drug_order], dtype=int)
        panel = _resolve_panel(gene_panels, context, true.shape[1])
        context_pred = pred[indices][:, panel]
        context_true = true[indices][:, panel]
        similarities = np.empty((len(drug_order), len(drug_order)), dtype=float)
        for pred_index in range(len(drug_order)):
            for truth_index in range(len(drug_order)):
                similarities[pred_index, truth_index] = _finite_correlation(
                    context_pred[pred_index], context_true[truth_index], metric
                )
        matrices.append(similarities)
    return matrices


def _score_label_permutation(
    permutation: Sequence[int],
    similarities: Sequence[np.ndarray],
    tie_atol: float,
) -> float:
    source = np.asarray(permutation, dtype=int)
    n_drugs = source.size
    scores = np.empty((len(similarities), n_drugs), dtype=float)
    for context_index, matrix in enumerate(similarities):
        for drug_index in range(n_drugs):
            row = matrix[source[drug_index]]
            differences = row[drug_index] - np.delete(row, drug_index)
            wins = differences > tie_atol
            ties = np.abs(differences) <= tie_atol
            scores[context_index, drug_index] = np.mean(
                wins.astype(float) + 0.5 * ties.astype(float)
            )
    return float(scores.mean(axis=0).mean())


def conditional_rank_permutation_test(
    pred: np.ndarray,
    true: np.ndarray,
    context_ids: Sequence[object],
    drug_ids: Sequence[object],
    gene_panels: Mapping[str, Sequence[int]] | Sequence[int],
    *,
    metric: str = "pearson",
    tie_atol: float = 1e-12,
    exhaustive_cap: int = 1_000_000,
    n_permutations: int = 10_000,
    seed: int = 7301,
) -> PermutationTestResult:
    """Test conditional rank using one shared drug-label permutation in every context.

    Frozen panels are consumed as provided and never re-derived under the null. All label
    permutations are enumerated when ``n_drugs! <= exhaustive_cap``; otherwise uniformly
    sampled permutations use ``seed`` and a plus-one Monte Carlo p-value.
    """
    n_permutations = _positive_integer(n_permutations, "n_permutations")
    if (
        isinstance(exhaustive_cap, bool)
        or not isinstance(exhaustive_cap, (int, np.integer))
        or exhaustive_cap < 0
    ):
        raise ValueError("exhaustive_cap must be a non-negative integer.")
    if not np.isfinite(tie_atol) or tie_atol < 0:
        raise ValueError("tie_atol must be finite and non-negative.")

    pred_arr = np.asarray(pred, dtype=float)
    true_arr = np.asarray(true, dtype=float)
    raw_contexts = np.asarray(list(context_ids), dtype=object)
    raw_drugs = np.asarray(list(drug_ids), dtype=object)
    if pd.isna(raw_contexts).any() or pd.isna(raw_drugs).any():
        raise ValueError("context_ids and drug_ids must not contain missing values.")
    contexts = np.asarray([str(value) for value in raw_contexts], dtype=object)
    drugs = np.asarray([str(value) for value in raw_drugs], dtype=object)
    observed_result = conditional_rank_score(
        pred_arr,
        true_arr,
        contexts,
        drugs,
        gene_panels,
        metric=metric,
        tie_atol=tie_atol,
    )
    pairs = pd.DataFrame({"context_id": contexts, "drug_id": drugs})
    duplicate = pairs.duplicated(["context_id", "drug_id"], keep=False)
    if duplicate.any():
        raise ValueError(
            "duplicate context/drug rows are not allowed in a permutation panel."
        )
    drug_sets = pairs.groupby("context_id", sort=True)["drug_id"].agg(frozenset)
    expected = drug_sets.iloc[0]
    if any(candidate != expected for candidate in drug_sets.iloc[1:]):
        raise ValueError(
            "unbalanced permutation panel: every context must contain the same drug set."
        )
    drug_order = tuple(sorted(expected))
    if len(drug_order) < 2:
        raise ValueError("conditional-rank permutation requires at least two drugs.")

    similarities = _conditional_similarity_matrices(
        pred_arr,
        true_arr,
        contexts,
        drugs,
        drug_order,
        gene_panels,
        metric,
    )
    observed = float(observed_result.summary["conditional_accuracy_drug_macro"])
    identity_score = _score_label_permutation(
        range(len(drug_order)), similarities, tie_atol
    )
    if not np.isclose(observed, identity_score, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            "permutation scorer does not reproduce the registered endpoint."
        )

    total = math.factorial(len(drug_order))
    exhaustive = total <= int(exhaustive_cap)
    if exhaustive:
        draws = np.fromiter(
            (
                _score_label_permutation(permutation, similarities, tie_atol)
                for permutation in itertools.permutations(range(len(drug_order)))
            ),
            dtype=float,
            count=total,
        )
        p_value = float(np.mean(draws >= observed - 1e-15))
    else:
        rng = np.random.default_rng(seed)
        draws = np.empty(n_permutations, dtype=float)
        for index in range(n_permutations):
            permutation = rng.permutation(len(drug_order))
            draws[index] = _score_label_permutation(permutation, similarities, tie_atol)
        exceedances = int(np.count_nonzero(draws >= observed - 1e-15))
        p_value = float((exceedances + 1) / (n_permutations + 1))
    if not np.isfinite(draws).all():
        raise RuntimeError("permutation null produced non-finite statistics.")
    return PermutationTestResult(
        observed=observed,
        p_value=p_value,
        draws=draws,
        exhaustive=exhaustive,
        total_label_permutations=total,
    )


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted q-values in the input order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence.")
    if not np.isfinite(values).all():
        raise ValueError("p_values must contain only finite values.")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p_values must lie in the closed interval [0, 1].")

    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * values.size / np.arange(1, values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output


def benjamini_hochberg_qvalues(p_values: Sequence[float]) -> np.ndarray:
    """Explicit alias for :func:`benjamini_hochberg`."""
    return benjamini_hochberg(p_values)
