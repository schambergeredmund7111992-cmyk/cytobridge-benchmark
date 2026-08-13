"""CytoBridge agent tools: 5-tool registry for LLM-based drug reasoning."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TOOL_DEFINITIONS = [
    {
        "name": "retrieve_lincs_signatures",
        "description": "Retrieve top-k LINCS L1000 perturbation signatures for a cell line.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_line": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["cell_line"],
        },
    },
    {
        "name": "query_drugbank",
        "description": "Query DrugBank for drug target, MOA, and indications.",
        "parameters": {
            "type": "object",
            "properties": {
                "drug_name_or_smiles": {"type": "string"},
            },
            "required": ["drug_name_or_smiles"],
        },
    },
    {
        "name": "get_pathway_info",
        "description": "Get genes and description for an MSigDB Hallmark pathway.",
        "parameters": {
            "type": "object",
            "properties": {
                "pathway_name": {"type": "string"},
            },
            "required": ["pathway_name"],
        },
    },
    {
        "name": "predict_response",
        "description": "Predict perturbation response for a drug in a cell line using CytoBridge.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_line": {"type": "string"},
                "drug_smiles": {"type": "string"},
            },
            "required": ["cell_line", "drug_smiles"],
        },
    },
    {
        "name": "search_pubmed",
        "description": "Search PubMed for literature evidence linking a drug to a pathway or phenotype.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
]


class ToolRegistry:
    """Offline-first registry of 5 tools for CytoBridge agent reasoning."""

    def __init__(
        self,
        lincs_dir: str | Path | None = None,
        drugbank_cache: str | Path | None = None,
        msigdb_gmt: str | Path | None = None,
        precomputed_cell_embs: str | Path | None = None,
    ):
        self._lincs_dir = Path(lincs_dir) if lincs_dir else None
        self._drugbank: dict | None = None
        if drugbank_cache:
            self._drugbank = json.loads(Path(drugbank_cache).read_text())
        self._msigdb: dict[str, list[str]] = {}
        if msigdb_gmt:
            self._msigdb = self._parse_gmt(Path(msigdb_gmt))
        self._cell_embs: dict[str, np.ndarray] = {}
        if precomputed_cell_embs:
            data = np.load(precomputed_cell_embs, allow_pickle=True)
            for cid, emb in zip(data["ids"], data["embs"]):
                self._cell_embs[str(cid)] = emb
        self._lincs_meta: pd.DataFrame | None = None
        if self._lincs_dir and self._lincs_dir.exists():
            meta_path = self._lincs_dir / "lincs_meta.csv"
            if meta_path.exists():
                self._lincs_meta = pd.read_csv(meta_path)

    @staticmethod
    def _parse_gmt(path: Path) -> dict[str, list[str]]:
        out = {}
        for line in path.read_text().strip().splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                out[parts[0]] = parts[2:]
        return out

    def call(self, name: str, **kwargs):
        """Dispatch a tool call by name."""
        if name == "retrieve_lincs_signatures":
            return self.retrieve_lincs_signatures(
                kwargs.get("cell_line", ""), top_k=kwargs.get("top_k", 5),
            )
        if name == "query_drugbank":
            return self.query_drugbank(kwargs.get("drug_name_or_smiles", ""))
        if name == "get_pathway_info":
            return self.get_pathway_info(kwargs.get("pathway_name", ""))
        if name == "predict_response":
            return self.predict_response(
                kwargs.get("cell_line", ""), kwargs.get("drug_smiles", ""),
            )
        if name == "search_pubmed":
            return self.search_pubmed(kwargs.get("query", ""))
        return {"error": f"unknown tool: {name}"}

    def retrieve_lincs_signatures(self, cell_line: str, top_k: int = 5) -> list[dict]:
        if self._lincs_meta is None:
            return [{"error": "LINCS data not loaded"}]
        matches = self._lincs_meta[self._lincs_meta.index.astype(str).str.contains(
            cell_line, case=False
        )]
        if matches.empty:
            matches = self._lincs_meta.head(top_k)
        return matches.head(top_k).to_dict(orient="records")

    def query_drugbank(self, drug_name_or_smiles: str) -> dict:
        if self._drugbank is None:
            return {"name": drug_name_or_smiles, "targets": [], "moa": "unknown"}
        key = drug_name_or_smiles.lower()
        if key in self._drugbank:
            return self._drugbank[key]
        for k, v in self._drugbank.items():
            if key in k or k in key:
                return v
        return {"name": drug_name_or_smiles, "targets": [], "moa": "unknown"}

    def get_pathway_info(self, pathway_name: str) -> dict:
        for name, genes in self._msigdb.items():
            if pathway_name.upper() in name.upper():
                return {"name": name, "genes": genes}
        return {"name": pathway_name, "genes": [], "error": "pathway not found"}

    def predict_response(self, cell_line: str, drug_smiles: str) -> dict:
        if cell_line not in self._cell_embs:
            return {"error": f"no precomputed embedding for cell line {cell_line}"}
        return {
            "cell_line": cell_line,
            "drug_smiles": drug_smiles,
            "predicted_logFC": None,
            "error": "CytoBridge model not integrated in offline registry — use LLM reasoning fallback",
        }

    def search_pubmed(self, query: str) -> dict:
        return {
            "query": query,
            "results": [],
            "note": "PubMed search stub — replace with E-utilities API for production",
        }
