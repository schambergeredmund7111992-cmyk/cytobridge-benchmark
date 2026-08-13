from __future__ import annotations

import json

import numpy as np
import pandas as pd

from agent.tools import ToolRegistry


def test_tool_registry_offline_tools_return_json_serializable(tmp_path):
    lincs = tmp_path / "lincs"
    lincs.mkdir()
    np.save(lincs / "lincs_emb.npy", np.eye(3, dtype="float32"))
    pd.DataFrame({"drug": ["a", "b", "c"], "perturbation": ["p1", "p2", "p3"]}).to_csv(
        lincs / "lincs_meta.csv", index=False
    )
    np.savez(tmp_path / "cells.npz", ids=np.array(["cell1"]), embs=np.array([[1.0, 0.0, 0.0]]))
    (tmp_path / "drugbank_cache.json").write_text(
        json.dumps({"aspirin": {"name": "aspirin", "targets": ["PTGS1"], "moa": "COX inhibitor"}})
    )
    (tmp_path / "hallmark.gmt").write_text("HALLMARK_APOPTOSIS\tna\tBAX\tCASP3\n")

    registry = ToolRegistry(
        lincs_dir=lincs,
        drugbank_cache=tmp_path / "drugbank_cache.json",
        msigdb_gmt=tmp_path / "hallmark.gmt",
        precomputed_cell_embs=tmp_path / "cells.npz",
    )

    lincs_out = registry.retrieve_lincs_signatures("cell1", top_k=2)
    drugbank_out = registry.query_drugbank("aspirin")
    pathway_out = registry.get_pathway_info("HALLMARK_APOPTOSIS")
    pred_out = registry.predict_response("cell1", "CCO")

    json.dumps([lincs_out, drugbank_out, pathway_out, pred_out])
    assert lincs_out[0]["drug"] == "a"
    assert drugbank_out["targets"] == ["PTGS1"]
    assert pathway_out["genes"] == ["BAX", "CASP3"]
    assert "error" in pred_out
    # 5-tool registry — compute_drug_similarity was dropped (overlap with LINCS).
    assert not hasattr(registry, "compute_drug_similarity")
    from agent.tools import TOOL_DEFINITIONS
    assert len(TOOL_DEFINITIONS) == 5
    assert {t["name"] for t in TOOL_DEFINITIONS} == {
        "retrieve_lincs_signatures", "query_drugbank", "get_pathway_info",
        "predict_response", "search_pubmed",
    }
