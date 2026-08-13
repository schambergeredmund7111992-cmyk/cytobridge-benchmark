"""Select a configuration using validation metrics only."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY = "conditional_accuracy_drug_macro"
TIE_BREAKER = "pair_own_spearman_top50_drug_macro"


def epoch_selection_key(primary: float, tie_breaker: float, epoch: int) -> tuple[float, float, int]:
    values = np.asarray([primary, tie_breaker], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Epoch-selection metrics must be finite.")
    if int(epoch) != epoch or int(epoch) < 1:
        raise ValueError("Epoch-selection epoch must be a positive integer.")
    return float(primary), float(tie_breaker), -int(epoch)


@dataclass
class EpochSelectionTracker:
    """State machine for the preregistered metric, tie-breaker, earlier-epoch rule."""

    patience: int | None
    best_key: tuple[float, float, int] | None = None
    best_epoch: int | None = None
    best_metrics: dict[str, float] | None = None
    bad_evaluations: int = 0
    evaluations: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.patience is not None and self.patience < 1:
            raise ValueError("Selection patience must be positive or None.")

    def update(self, *, primary: float, tie_breaker: float, epoch: int) -> dict:
        key = epoch_selection_key(primary, tie_breaker, epoch)
        epoch = int(epoch)
        if self.evaluations and epoch <= int(self.evaluations[-1]["epoch"]):
            raise ValueError("Epoch-selection evaluations must be strictly increasing.")
        improved = self.best_key is None or key > self.best_key
        if improved:
            self.best_key = key
            self.best_epoch = epoch
            self.best_metrics = {
                PRIMARY: float(primary),
                TIE_BREAKER: float(tie_breaker),
            }
            self.bad_evaluations = 0
        else:
            self.bad_evaluations += 1
        should_stop = bool(
            self.patience is not None and self.bad_evaluations >= self.patience
        )
        record = {
            "epoch": epoch,
            PRIMARY: float(primary),
            TIE_BREAKER: float(tie_breaker),
            "selection_key": list(key),
            "improved": improved,
            "bad_evaluations": self.bad_evaluations,
            "should_stop": should_stop,
        }
        self.evaluations.append(record)
        return record

    def to_dict(self) -> dict:
        return {
            "patience": self.patience,
            "best_key": list(self.best_key) if self.best_key is not None else None,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "bad_evaluations": self.bad_evaluations,
            "evaluations": self.evaluations,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EpochSelectionTracker":
        tracker = cls(patience=payload["patience"])
        best_key = payload.get("best_key")
        tracker.best_key = tuple(best_key) if best_key is not None else None
        tracker.best_epoch = payload.get("best_epoch")
        tracker.best_metrics = payload.get("best_metrics")
        tracker.bad_evaluations = int(payload.get("bad_evaluations", 0))
        tracker.evaluations = list(payload.get("evaluations", []))
        return tracker


def select_validation_configuration(
    trials: pd.DataFrame,
    *,
    expected_budget: int,
) -> tuple[pd.Series, pd.DataFrame]:
    required = {"config_id", "split", PRIMARY, TIE_BREAKER}
    if missing := required - set(trials.columns):
        raise ValueError(f"Selection table is missing columns: {sorted(missing)}")
    table = trials.copy()
    if len(table) != expected_budget:
        raise ValueError(
            f"Expected exactly {expected_budget} completed trials; observed {len(table)}."
        )
    if table["config_id"].astype(str).duplicated().any():
        raise ValueError("Configuration identifiers must be unique.")
    if set(table["split"].astype(str)) != {"validation"}:
        raise ValueError("Configuration selection may consume validation rows only.")
    for column in (PRIMARY, TIE_BREAKER):
        table[column] = pd.to_numeric(table[column], errors="raise")
        if not np.isfinite(table[column]).all():
            raise ValueError(f"Selection metric {column!r} contains non-finite values.")
    table["config_id"] = table["config_id"].astype(str)
    ranked = table.sort_values(
        [PRIMARY, TIE_BREAKER, "config_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "selection_rank", np.arange(1, len(ranked) + 1))
    return ranked.iloc[0].copy(), ranked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranked-out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.ranked_out.exists():
        raise FileExistsError("Refusing to overwrite frozen model-selection output.")
    selected, ranked = select_validation_configuration(
        pd.read_csv(args.trials), expected_budget=args.budget
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.ranked_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(selected.to_dict(), indent=2, sort_keys=True) + "\n")
    ranked.to_csv(args.ranked_out, index=False)
    print(json.dumps(selected.to_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
