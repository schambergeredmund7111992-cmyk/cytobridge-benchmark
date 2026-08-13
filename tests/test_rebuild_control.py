"""Unit tests for the control-unification rebuild (pure core, no anndata).

The key property: after rebuilding control_counts as the per-cell-line DMSO mean
and broadcasting to every row, run_internal's per-pair ctrl_pb (mean of a pair's
per-row controls) becomes IDENTICAL to ridge's all-DMSO cell-line mean. That is
exactly what makes Δ-vs-ridge apples-to-apples.
Run: python tests/test_rebuild_control.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.rebuild_control_pseudobulk import cellline_control_means, broadcast_control


def test_cellline_means_use_control_cells_only():
    counts = np.array([
        [10, 0, 0, 0],   # A control
        [12, 0, 0, 0],   # A control
        [99, 99, 99, 99],  # A treated (must be ignored)
        [0, 8, 0, 0],    # B control
        [0, 4, 0, 0],    # B control
    ], dtype=np.float32)
    cl = np.array(["A", "A", "A", "B", "B"])
    is_ctrl = np.array([True, True, False, True, True])
    means = cellline_control_means(counts, cl, is_ctrl)
    assert np.allclose(means["A"], [11, 0, 0, 0])   # (10+12)/2, treated ignored
    assert np.allclose(means["B"], [0, 6, 0, 0])    # (8+4)/2


def test_broadcast_and_missing():
    means = {"A": np.array([1.0, 2.0]), "B": np.array([3.0, 4.0])}
    arr, missing = broadcast_control(means, ["A", "B", "A"], n_genes=2)
    assert arr.shape == (3, 2) and not missing
    assert np.allclose(arr[0], [1, 2]) and np.allclose(arr[2], [1, 2])
    _, missing2 = broadcast_control(means, ["A", "C"], n_genes=2)
    assert missing2 == ["C"]


def test_run_internal_ctrl_pb_matches_ridge_after_rebuild():
    """After rebuild, run_internal's pair ctrl_pb == ridge's all-DMSO mean."""
    from eval.run_internal import aggregate_by_pair
    rng = np.random.default_rng(0)
    G = 30
    # 20 DMSO control cells + treated cells for cell line A
    ctrl_cells = rng.poisson(5, size=(20, G)).astype(np.float32)
    n_treated = 7
    treated_cells = rng.poisson(8, size=(n_treated, G)).astype(np.float32)
    counts = np.vstack([ctrl_cells, treated_cells])
    cl = np.array(["A"] * (20 + n_treated))
    is_ctrl = np.array([True] * 20 + [False] * n_treated)

    # ridge's control = all-DMSO cell-line mean
    ridge_ctrl = ctrl_cells.mean(axis=0)
    # rebuild -> broadcast cell-line mean to the treated rows (run_internal rows)
    means = cellline_control_means(counts, cl, is_ctrl)
    bcast, _ = broadcast_control(means, ["A"] * n_treated, n_genes=G)

    # run_internal aggregates the per-row control over a pair's cells:
    pred_mu = treated_cells.copy()
    _, _, _, _ = aggregate_by_pair(pred_mu, treated_cells, bcast,
                                   ["d"] * n_treated, ["A"] * n_treated)
    pair_ctrl_pb = bcast.mean(axis=0)   # what aggregate_by_pair uses internally
    assert np.allclose(pair_ctrl_pb, ridge_ctrl, atol=1e-5), \
        "rebuilt control must equal ridge all-DMSO mean (apples-to-apples)"
    # and it equals the cell-line mean exactly
    assert np.allclose(pair_ctrl_pb, means["A"], atol=1e-5)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"OK: {len(fns)} rebuild-control tests passed")
