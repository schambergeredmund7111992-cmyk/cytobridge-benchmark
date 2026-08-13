"""
agent/case_studies/gbm.py
-------------------------
Glioblastoma (GBM) drug repurposing case study.

Data: Neftel et al. 2019 (Cell) — GBM scRNA-seq, 28 patients, ~24K cells,
      4 cellular states (NPC-like, OPC-like, MES-like, AC-like).

Goal: CytoReasoner should rank
    - temozolomide (FDA, SoC for GBM)
    - bevacizumab (FDA, off-label for GBM)
    in top-10 candidates, plus 1 novel candidate.
"""
from __future__ import annotations

# Same structure as ipf.py — adapt cell_state filter / data path.
# See agent/case_studies/ipf.py for template.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.case_studies.ipf import main as run_case  # noqa: E402

if __name__ == "__main__":
    user_args = sys.argv[1:]
    sys.argv = [sys.argv[0]] + [
        "--adata", "data/raw/gbm/neftel2019_processed.h5ad",
        "--candidates_csv", "data/raw/gbm/candidate_drugs.csv",
        "--precomputed_cell_embs", "data/cache/gbm_pooled_cell_embs.npz",
        "--cell_state_tokens", "data/cache/gbm_cell_state_tokens.npz",
        "--cluster", "MES-like",
        "--disease_label", "GBM",
        "--control_label", "reference",
        "--out", "results/case_studies/gbm.json",
    ] + user_args
    run_case()
