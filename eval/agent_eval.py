"""CytoBridge Agent Evaluation: 4 quantitative metrics for LLM drug reasoning.

Metrics:
  1. Tool-call success rate -- fraction of tool invocations returning valid results
  2. Evidence-grounding precision/recall/F1 -- do cited claims match tool outputs?
  3. Rationale-pathway faithfulness -- correlation between LLM rationale and model pathway gate
  4. Repeat consistency -- agreement across N independent runs (same input)

Usage:
  export DEEPSEEK_API_KEY="sk-..."
  cd /path/to/CytoBridge
  python eval/agent_eval.py --cases ipf gbm --n_repeats 3 --out results/agent_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import re
import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.stats import spearmanr

from agent.tools import TOOL_DEFINITIONS


# ============================================================================
# 1. Realistic mock data for IPF / GBM case studies
# ============================================================================

def _build_drugbank() -> dict:
    """Curated DrugBank entries covering IPF and GBM drugs."""
    return {
        "nintedanib": {
            "name": "Nintedanib", "drugbank_id": "DB09079",
            "targets": ["PDGFRA", "PDGFRB", "FGFR1", "FGFR2", "FGFR3", "VEGFR1", "VEGFR2", "VEGFR3"],
            "moa": "Tyrosine kinase inhibitor targeting PDGFR, FGFR, VEGFR -- anti-fibrotic and anti-angiogenic",
            "indication": "Idiopathic pulmonary fibrosis (IPF)",
            "pathways": ["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION", "HALLMARK_TNFA_SIGNALING_VIA_NFKB", "HALLMARK_ANGIOGENESIS"],
        },
        "pirfenidone": {
            "name": "Pirfenidone", "drugbank_id": "DB04951",
            "targets": ["TGFB1", "TNFA"],
            "moa": "Anti-fibrotic; inhibits TGF-beta synthesis and fibroblast proliferation",
            "indication": "Idiopathic pulmonary fibrosis (IPF)",
            "pathways": ["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION", "HALLMARK_TNFA_SIGNALING_VIA_NFKB", "HALLMARK_INFLAMMATORY_RESPONSE"],
        },
        "temozolomide": {
            "name": "Temozolomide", "drugbank_id": "DB00853",
            "targets": ["DNA"],
            "moa": "Alkylating agent; methylates guanine at O6 position causing DNA damage and apoptosis",
            "indication": "Glioblastoma multiforme (GBM)",
            "pathways": ["HALLMARK_DNA_REPAIR", "HALLMARK_P53_PATHWAY", "HALLMARK_APOPTOSIS"],
        },
        "bevacizumab": {
            "name": "Bevacizumab", "drugbank_id": "DB00112",
            "targets": ["VEGFA"],
            "moa": "Anti-VEGF monoclonal antibody; inhibits angiogenesis",
            "indication": "Glioblastoma (recurrent), colorectal cancer, NSCLC",
            "pathways": ["HALLMARK_ANGIOGENESIS", "HALLMARK_HYPOXIA"],
        },
        "dexamethasone": {
            "name": "Dexamethasone", "drugbank_id": "DB01234",
            "targets": ["NR3C1"],
            "moa": "Glucocorticoid receptor agonist; anti-inflammatory and immunosuppressive",
            "indication": "Cerebral edema in GBM, inflammation",
            "pathways": ["HALLMARK_INFLAMMATORY_RESPONSE", "HALLMARK_TNFA_SIGNALING_VIA_NFKB"],
        },
        "losartan": {
            "name": "Losartan", "drugbank_id": "DB00678",
            "targets": ["AGTR1"],
            "moa": "Angiotensin II receptor blocker (ARB); anti-fibrotic via TGF-beta downregulation",
            "indication": "Hypertension; investigated for IPF",
            "pathways": ["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION", "HALLMARK_TNFA_SIGNALING_VIA_NFKB"],
        },
    }


def _build_msigdb() -> dict[str, list[str]]:
    """Curated MSigDB Hallmark pathways relevant to IPF and GBM."""
    return {
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": ["TGFB1", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "CDH1", "CDH2", "VIM", "FN1", "COL1A1", "COL1A2", "ACTA2"],
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB": ["TNF", "NFKB1", "NFKBIA", "RELA", "TNFAIP3", "JUN", "FOS", "IL6", "IL1B", "CCL2", "CXCL8"],
        "HALLMARK_INFLAMMATORY_RESPONSE": ["IL6", "IL1B", "TNF", "CXCL8", "CCL2", "PTGS2", "NFKB1", "RELA", "TLR4", "MYD88"],
        "HALLMARK_ANGIOGENESIS": ["VEGFA", "VEGFR2", "PDGFRB", "FGFR1", "ANGPT1", "ANGPT2", "TEK", "FLT1", "KDR", "NOTCH1"],
        "HALLMARK_APOPTOSIS": ["BCL2", "BAX", "CASP3", "CASP8", "CASP9", "CYCS", "TP53", "BID", "BAK1", "DIABLO"],
        "HALLMARK_DNA_REPAIR": ["BRCA1", "BRCA2", "RAD51", "ATM", "ATR", "TP53", "CHEK1", "CHEK2", "MLH1", "MSH2"],
        "HALLMARK_P53_PATHWAY": ["TP53", "CDKN1A", "MDM2", "BAX", "BBC3", "GADD45A", "RRM2B", "SESN1", "TIGAR"],
        "HALLMARK_HYPOXIA": ["HIF1A", "VEGFA", "LDHA", "HK2", "SLC2A1", "PGK1", "EPO", "BNIP3", "BNIP3L", "PDK1"],
        "HALLMARK_MYC_TARGETS_V1": ["MYC", "NPM1", "NCL", "RPL3", "RPL5", "RPS6", "EIF4E", "DDX21"],
        "HALLMARK_G2M_CHECKPOINT": ["CDK1", "CCNB1", "CCNB2", "CDC20", "BUB1", "PLK1", "AURKA", "AURKB", "TOP2A"],
    }


def _build_cell_embs() -> dict:
    """Mock cell embeddings (128-d) for A549, K562, MCF7, U251."""
    rng = np.random.default_rng(42)
    return {
        "A549": rng.normal(0, 1, 128).astype(np.float32),
        "K562": rng.normal(0.5, 1, 128).astype(np.float32),
        "MCF7": rng.normal(-0.3, 0.9, 128).astype(np.float32),
        "U251": rng.normal(0.2, 0.8, 128).astype(np.float32),
    }


def _build_lincs() -> pd.DataFrame:
    """Mock LINCS L1000 meta table."""
    return pd.DataFrame([
        {"drug": "nintedanib", "perturbation": "TGFB1_down", "cell_line": "A549", "similarity": 0.87},
        {"drug": "pirfenidone", "perturbation": "TGFB1_down", "cell_line": "A549", "similarity": 0.82},
        {"drug": "losartan", "perturbation": "TGFB1_down", "cell_line": "A549", "similarity": 0.65},
        {"drug": "temozolomide", "perturbation": "DNA_damage_up", "cell_line": "U251", "similarity": 0.91},
        {"drug": "bevacizumab", "perturbation": "VEGF_down", "cell_line": "U251", "similarity": 0.78},
        {"drug": "dexamethasone", "perturbation": "NFKB_down", "cell_line": "U251", "similarity": 0.73},
        {"drug": "nintedanib", "perturbation": "TGFB1_down", "cell_line": "U251", "similarity": 0.55},
        {"drug": "pirfenidone", "perturbation": "TGFB1_down", "cell_line": "U251", "similarity": 0.51},
    ])


# ============================================================================
# 2. Evaluation metrics
# ============================================================================

def compute_tool_call_success(tool_results: list[dict]) -> dict:
    """Metric 1: fraction of tool calls returning valid (non-error) results."""
    total = len(tool_results)
    if total == 0:
        return {"tool_call_success_rate": None, "n_calls": 0, "n_success": 0, "by_tool": {}}
    by_tool = {}
    for r in tool_results:
        name = r.get("tool", "unknown")
        by_tool.setdefault(name, {"total": 0, "success": 0})
        by_tool[name]["total"] += 1
        by_tool[name]["success"] += 0 if r.get("error") else 1
    n_ok = sum(v["success"] for v in by_tool.values())
    return {
        "tool_call_success_rate": round(n_ok / total, 4),
        "n_calls": total, "n_success": n_ok,
        "by_tool": {k: round(v["success"] / v["total"], 4) for k, v in by_tool.items()},
    }


def compute_evidence_grounding(
    outputs: list[dict], drugbank: dict,
) -> dict:
    """Metric 2: P/R/F1 -- do the LLM's claims match what the tools returned?

    For each reasoning output, extracts drugs mentioned in the rationale,
    checks if those drugs exist in DrugBank cache, and compares claimed
    MOA/targets with the ground truth.
    """
    tp = fp = fn = 0

    for out in outputs:
        ranking = out.get("ranking") or []
        for entry in ranking:
            drug_name = (entry.get("drug") or "").lower()
            rationale = (entry.get("rationale") or "").lower()

            gt = drugbank.get(drug_name, {})
            gt_targets = set(t.lower() for t in (gt.get("targets") or []))

            for t in gt_targets:
                if t.lower() in rationale:
                    tp += 1
                else:
                    fn += 1

            # Hallucination check: mentions a target NOT in DrugBank
            hallucination_pool = ["egfr", "her2", "braf", "alk", "ros1", "kras", "mtor", "parp"]
            for fake in hallucination_pool:
                if fake in rationale and fake not in gt_targets:
                    fp += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "evidence_precision": round(prec, 4), "evidence_recall": round(rec, 4),
        "evidence_f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn,
    }


def compute_pathway_faithfulness(
    outputs: list[dict],
    pathway_attributions: dict[str, np.ndarray],
) -> dict:
    """Metric 3: correlation between LLM rationale emphasis and model pathway gate.

    For each drug ranked by the LLM, we compare the pathways mentioned in the
    rationale (binary vector over K=10 pathways) with known drug-pathway
    ground truth from DrugBank (as a proxy for CytoBridge pathway gate).
    """
    pathway_list = [
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_ANGIOGENESIS",
        "HALLMARK_APOPTOSIS",
        "HALLMARK_DNA_REPAIR",
        "HALLMARK_P53_PATHWAY",
        "HALLMARK_HYPOXIA",
        "HALLMARK_MYC_TARGETS_V1",
        "HALLMARK_G2M_CHECKPOINT",
    ]

    pairs = []
    for out in outputs:
        ranking = out.get("ranking") or []
        for entry in ranking:
            drug_name = (entry.get("drug") or "").lower()
            rationale = (entry.get("rationale") or "").lower()

            # Binary vector: which pathways does the LLM mention?
            # Match both "epithelial_mesenchymal_transition" and "epithelial mesenchymal transition"
            rationale_flat = rationale.replace(" ", "_")
            llm_vec = np.array(
                [1.0 if pw.lower().replace("hallmark_", "") in rationale_flat else 0.0
                 for pw in pathway_list]
            )

            model_vec = pathway_attributions.get(drug_name)
            if model_vec is None:
                continue

            if llm_vec.sum() > 0 and np.std(model_vec) > 1e-6:
                pairs.append((llm_vec, model_vec))

    if len(pairs) < 2:
        return {"pathway_faithfulness_spearman": None, "n_comparable_pairs": len(pairs),
                "note": "Fewer than 2 comparable drug-pathway pairs"}

    spears = []
    for llm_v, mdl_v in pairs:
        if np.std(llm_v) > 0 and np.std(mdl_v) > 1e-6:
            s = spearmanr(llm_v, mdl_v).statistic
            if not np.isnan(s):
                spears.append(s)

    return {
        "pathway_faithfulness_spearman": round(float(np.mean(spears)), 4) if spears else None,
        "pathway_faithfulness_median": round(float(np.median(spears)), 4) if spears else None,
        "n_comparable_pairs": len(pairs), "n_valid_spearman": len(spears),
    }


def compute_repeat_consistency(all_runs: list) -> dict:
    """Metric 4: agreement across repeated runs (same input).

    all_runs can be either list[dict] (live) or list[list[dict]] (dry-run).
    """
    n_runs = len(all_runs)
    if n_runs < 2:
        return {"repeat_consistency_mean_kendall_tau": None, "note": "Need >= 2 runs"}

    def _extract_ranking(run):
        """Extract drug ranking from a run, which may be a dict or list of dicts."""
        drugs = []
        if isinstance(run, dict):
            items = [run]
        elif isinstance(run, list):
            items = run
        else:
            return drugs
        for item in items:
            if isinstance(item, dict):
                for entry in (item.get("ranking") or []):
                    drugs.append(entry.get("drug", ""))
            elif isinstance(item, str):
                # JSON string from API
                try:
                    parsed = json.loads(item)
                    for entry in (parsed.get("ranking") or []):
                        drugs.append(entry.get("drug", ""))
                except json.JSONDecodeError:
                    pass
        return drugs

    def _pairwise_kendall(a, b):
        common = set(a) & set(b)
        if len(common) < 2:
            return None
        a_idx = {d: i for i, d in enumerate(a) if d in common}
        b_idx = {d: i for i, d in enumerate(b) if d in common}
        concordant = discordant = 0
        drugs = sorted(common)
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                a_order = a_idx[drugs[i]] < a_idx[drugs[j]]
                b_order = b_idx[drugs[i]] < b_idx[drugs[j]]
                if a_order == b_order:
                    concordant += 1
                else:
                    discordant += 1
        total = concordant + discordant
        return (concordant - discordant) / total if total > 0 else 0.0

    ranks = [_extract_ranking(run) for run in all_runs]
    taus = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            tau = _pairwise_kendall(ranks[i], ranks[j])
            if tau is not None:
                taus.append(tau)

    return {
        "repeat_consistency_mean_kendall_tau": round(float(np.mean(taus)), 4) if taus else None,
        "repeat_consistency_std": round(float(np.std(taus)), 4) if taus else None,
        "n_runs": n_runs, "n_pairwise_comparisons": len(taus),
    }


# ============================================================================
# 3. Case study definitions
# ============================================================================

CASES = {
    "ipf": {
        "state": (
            "Idiopathic pulmonary fibrosis (IPF) is a chronic, progressive fibrotic lung disease. "
            "Key pathways: TGF-beta-driven epithelial-mesenchymal transition (EMT), chronic inflammation, "
            "and aberrant wound healing. Current standard of care: nintedanib and pirfenidone."
        ),
        "candidates": ["Nintedanib", "Pirfenidone", "Losartan", "Dexamethasone"],
        "cell_line": "A549",
        "expected_top": ["Nintedanib", "Pirfenidone"],
    },
    "gbm": {
        "state": (
            "Glioblastoma multiforme (GBM) is the most aggressive primary brain tumor. "
            "Key features: angiogenesis (VEGF-driven), DNA repair defects, hypoxia, "
            "and p53 pathway dysregulation. Current standard of care: temozolomide + radiation."
        ),
        "candidates": ["Temozolomide", "Bevacizumab", "Dexamethasone", "Pirfenidone"],
        "cell_line": "U251",
        "expected_top": ["Temozolomide", "Bevacizumab"],
    },
}


# ============================================================================
# 4. Tool simulation
# ============================================================================

def build_realistic_registry() -> dict:
    """Build realistic data buckets for tool simulation."""
    return {
        "drugbank": _build_drugbank(),
        "msigdb": _build_msigdb(),
        "cell_embs": _build_cell_embs(),
    }


def simulate_tool_call(name: str, args: dict, registry: dict) -> dict:
    """Simulate a tool call using registry data."""
    drugbank = registry["drugbank"]
    msigdb = registry["msigdb"]

    if name == "query_drugbank":
        key = args.get("drug_name_or_smiles", "").lower()
        if key in drugbank:
            return {**drugbank[key], "error": None}
        return {"name": key, "targets": [], "moa": "unknown", "error": "Drug not found in cache"}

    if name == "get_pathway_info":
        pw = args.get("pathway_name", "").upper()
        for pname, genes in msigdb.items():
            if pw in pname:
                return {"name": pname, "genes": genes, "n_genes": len(genes), "error": None}
        return {"name": pw, "genes": [], "error": "pathway not found"}

    if name == "retrieve_lincs_signatures":
        cl = args.get("cell_line", "")
        top_k = args.get("top_k", 5)
        lincs = _build_lincs()
        matches = lincs[lincs["cell_line"] == cl].head(top_k)
        if matches.empty:
            return {"signatures": [], "note": f"no LINCS data for {cl}", "error": None}
        return {"signatures": matches.to_dict(orient="records"), "n": len(matches), "error": None}

    if name == "predict_response":
        cl = args.get("cell_line", "")
        return {
            "cell_line": cl, "drug_smiles": args.get("drug_smiles", ""),
            "predicted_logFC": None,
            "error": None,
            "pathway_gate": [round(float(x), 4) for x in
                             np.random.default_rng(42).dirichlet(np.ones(10))],
        }

    if name == "search_pubmed":
        q = args.get("query", "")
        return {"query": q, "results_count": 0, "results": [],
                "note": "PubMed search stub", "error": None}

    return {"error": f"unknown tool: {name}"}


# ============================================================================
# 5. LLM agent run (single case)
# ============================================================================

def run_single_case(
    case_name: str,
    case_def: dict,
    registry: dict,
    client: OpenAI,
    model: str,
) -> dict:
    """Run one case study: call LLM with tool definitions, collect outputs."""
    messages = [
        {"role": "system", "content": (
            "You are a pharmacology reasoning assistant. Use the provided tools to gather "
            "evidence about candidate drugs, then rank them by predicted efficacy. "
            "Return your final answer as JSON: \n"
            '{"ranked_candidates": [{"drug": "...", "rank": 1, "confidence": "high|medium|low", '
            '"rationale": "brief explanation citing specific evidence"}]}'
        )},
        {"role": "user", "content": (
            f"Disease: {case_def['state']}\n\n"
            f"Cell line: {case_def['cell_line']}\n"
            f"Candidate drugs: {', '.join(case_def['candidates'])}\n\n"
            "Please use the tools to gather evidence, then rank these candidates."
        )},
    ]

    tool_defs = [{"type": "function", "function": t} for t in TOOL_DEFINITIONS]
    tool_results: list[dict] = []
    final_ranking = None
    error = None
    tokens_in = 0
    tokens_out = 0

    # Phase 1: tool-calling loop (gather evidence)
    max_tool_iters = 4
    for iteration in range(max_tool_iters):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, tools=tool_defs, temperature=0.0,
            )
        except Exception as exc:
            error = f"API error: {exc}"
            break

        tokens_in += response.usage.prompt_tokens if response.usage else 0
        tokens_out += response.usage.completion_tokens if response.usage else 0

        msg = response.choices[0].message

        if not msg.tool_calls:
            break  # model doesn't want more tools

        tc_list = []
        for tc in msg.tool_calls:
            tc_list.append({
                "id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            })
        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": tc_list,
        })
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                t_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                t_args = {}
            result = simulate_tool_call(name, t_args, registry)
            result["tool"] = name
            tool_results.append(result)
            clean = {k: v for k, v in result.items() if k != "tool"}
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": json.dumps(clean, default=str),
            })

    # Phase 2: ask model to rank WITHOUT tools (force text response)
    messages.append({"role": "user", "content": (
        "You have gathered evidence on all candidates. Now rank them by predicted efficacy. "
        "Output ONLY valid JSON, no other text:\n"
        '{"ranked_candidates": [{"drug": "DrugName", "rank": 1, "confidence": "high|medium|low", '
        '"rationale": "brief explanation citing the evidence you gathered"}]}'
    )})

    try:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0.0,
        )
        tokens_in += response.usage.prompt_tokens if response.usage else 0
        tokens_out += response.usage.completion_tokens if response.usage else 0
        ranking_text = response.choices[0].message.content or ""
        # Parse JSON from response
        try:
            parsed = json.loads(ranking_text)
            if "ranked_candidates" in parsed:
                final_ranking = parsed["ranked_candidates"]
        except json.JSONDecodeError:
            m = re.search(r'\{[^{}]*"ranked_candidates"\s*:\s*\[.*?\][^{}]*\}', ranking_text, re.DOTALL)
            if m:
                try:
                    final_ranking = json.loads(m.group()).get("ranked_candidates")
                except json.JSONDecodeError:
                    error = f"Failed to parse ranking JSON from: {ranking_text[:300]}"
            else:
                error = f"No ranking JSON found in response: {ranking_text[:300]}"
    except Exception as exc:
        error = f"API error in ranking phase: {exc}"

    return {
        "case": case_name, "final_ranking": final_ranking,
        "error": error, "tool_results": tool_results,
        "tokens_in": tokens_in, "tokens_out": tokens_out,
    }


# ============================================================================
# 6. Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["ipf"])
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--out", default="results/agent_eval.json")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # --- dry-run mode ---
    if args.dry_run or not args.api_key:
        print("Running in dry-run mode (no API key or --dry-run flag)")
        results = _dry_run_eval(args)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"Saved to {out_path}")
        return

    # --- live API mode ---
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    registry = build_realistic_registry()

    all_case_runs: dict[str, list] = {}
    for case_name in args.cases:
        if case_name not in CASES:
            print(f"Unknown case: {case_name}. Available: {list(CASES)}")
            continue
        case_def = CASES[case_name]
        runs = []
        for run_idx in range(args.n_repeats):
            print(f"[{case_name}] run {run_idx + 1}/{args.n_repeats} ...")
            out = run_single_case(case_name, case_def, registry, client, args.model)
            runs.append(out)
            if run_idx < args.n_repeats - 1:
                time.sleep(1.0)
        all_case_runs[case_name] = runs

    # --- compute metrics ---
    drugbank = registry["drugbank"]

    pathway_list = [
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB", "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_ANGIOGENESIS", "HALLMARK_APOPTOSIS", "HALLMARK_DNA_REPAIR",
        "HALLMARK_P53_PATHWAY", "HALLMARK_HYPOXIA",
        "HALLMARK_MYC_TARGETS_V1", "HALLMARK_G2M_CHECKPOINT",
    ]
    pathway_attr: dict[str, np.ndarray] = {}
    for drug, info in drugbank.items():
        pw_set = set(info.get("pathways", []))
        pathway_attr[drug] = np.array(
            [1.0 if pw in pw_set else 0.0 for pw in pathway_list], dtype=float,
        )

    metrics: dict[str, dict] = {}
    for case_name in args.cases:
        runs = all_case_runs.get(case_name, [])
        # Normalize: live runs have "final_ranking", metric functions expect "ranking"
        norm_runs = [{**r, "ranking": r.get("final_ranking")} for r in runs]
        all_outputs = [r for r in norm_runs if r["ranking"] is not None]
        all_tool_results = []
        for r in runs:
            all_tool_results.extend(r.get("tool_results", []))

        metrics[case_name] = {
            **compute_tool_call_success(all_tool_results),
            **compute_evidence_grounding(all_outputs, drugbank),
            **compute_pathway_faithfulness(all_outputs, pathway_attr),
            **compute_repeat_consistency(norm_runs),
            "n_runs": len(runs), "n_successful_runs": len(all_outputs),
            "mean_tokens_in": round(float(np.mean([r["tokens_in"] for r in runs])), 1) if runs else 0,
            "mean_tokens_out": round(float(np.mean([r["tokens_out"] for r in runs])), 1) if runs else 0,
        }

    results = {"metrics": metrics, "runs": all_case_runs}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Saved to {out_path}")


def _dry_run_eval(args) -> dict:
    """Plausible mock results when no API key is available."""
    registry = build_realistic_registry()
    drugbank = registry["drugbank"]

    pathway_list = [
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB", "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_ANGIOGENESIS", "HALLMARK_APOPTOSIS", "HALLMARK_DNA_REPAIR",
        "HALLMARK_P53_PATHWAY", "HALLMARK_HYPOXIA",
        "HALLMARK_MYC_TARGETS_V1", "HALLMARK_G2M_CHECKPOINT",
    ]
    pathway_attr = {}
    for drug, info in drugbank.items():
        pw_set = set(info.get("pathways", []))
        pathway_attr[drug] = np.array(
            [1.0 if pw in pw_set else 0.0 for pw in pathway_list], dtype=float,
        )

    metrics = {}
    for case_name in args.cases:
        if case_name not in CASES:
            continue
        case_def = CASES[case_name]

        mock_tool_results = []
        for drug in case_def["candidates"]:
            db_r = simulate_tool_call("query_drugbank",
                                      {"drug_name_or_smiles": drug}, registry)
            db_r["tool"] = "query_drugbank"
            mock_tool_results.append(db_r)
            pr_r = simulate_tool_call("predict_response",
                                      {"cell_line": case_def["cell_line"], "drug_smiles": drug},
                                      registry)
            pr_r["tool"] = "predict_response"
            mock_tool_results.append(pr_r)
            sp_r = simulate_tool_call("search_pubmed",
                                      {"query": f"{drug} {case_def['cell_line']}"}, registry)
            sp_r["tool"] = "search_pubmed"
            mock_tool_results.append(sp_r)

        expected = case_def["expected_top"]
        candidates = case_def["candidates"]
        ranking = []
        for rank, drug in enumerate(expected, 1):
            db = drugbank.get(drug.lower(), {})
            pw_list = db.get("pathways", [])
            pw_mention = ""
            if pw_list:
                pw_short_underscore = [p.replace("HALLMARK_", "").lower() for p in pw_list[:2]]
                pw_mention = f" Key pathways: {', '.join(pw_short_underscore)}. "
            ranking.append({
                "drug": drug, "rank": rank,
                "confidence": "high" if rank == 1 else "medium",
                "rationale": (
                    f"{drug} targets {', '.join(db.get('targets', [])[:3])}. "
                    f"MOA: {db.get('moa', '')[:100]}.{pw_mention}"
                    f"LINCS confirms perturbation similarity in {case_def['cell_line']}. "
                    f"PubMed literature supports efficacy for this indication."
                ),
            })
        remaining = [d for d in candidates if d not in expected]
        for rank, drug in enumerate(remaining, len(expected) + 1):
            db = drugbank.get(drug.lower(), {})
            pw_list = db.get("pathways", [])
            pw_mention = ""
            if pw_list:
                pw_short_uscore2 = [p.replace("HALLMARK_", "").lower() for p in pw_list[:2]]
                pw_mention = f" Related pathways: {', '.join(pw_short_uscore2)}. "
            ranking.append({
                "drug": drug, "rank": rank,
                "confidence": "low",
                "rationale": (
                    f"{drug} targets {', '.join(db.get('targets', [])[:3])}. "
                    f"MOA: {db.get('moa', '')[:80]}.{pw_mention}"
                    f"Limited PubMed evidence in {case_def['cell_line']} for this indication."
                ),
            })

        mock_outputs = [{"ranking": ranking, "case": case_name}]
        mock_runs = [mock_outputs] * args.n_repeats

        metrics[case_name] = {
            **compute_tool_call_success(mock_tool_results),
            **compute_evidence_grounding(mock_outputs, drugbank),
            **compute_pathway_faithfulness(mock_outputs, pathway_attr),
            **compute_repeat_consistency(mock_runs),
            "n_runs": args.n_repeats, "n_successful_runs": args.n_repeats,
            "mean_tokens_in": 0.0, "mean_tokens_out": 0.0,
        }

    return {
        "metrics": metrics,
        "mode": "dry-run (mock LLM responses, no API calls)",
    }


if __name__ == "__main__":
    main()
