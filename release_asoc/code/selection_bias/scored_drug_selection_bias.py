"""
manuscript/analysis/supp_T8/scored_drug_selection_bias.py
=========================================================
T8 (Xichen task book, section T8) — are the 9 scored drugs representative?

WHY: the headline control numbers are computed on 9 "scored" drugs (x3 cell lines = 27
pairs). A reviewer asks whether those 9 are a biased, easy/hard subset of the held-out
drugs. This compares the 9 scored drugs against a background drug set on RDKit physico-
chemical descriptors (MW, logP, TPSA, HBD, HBA, rotatable bonds) with a Mann-Whitney U
test per descriptor; a non-significant difference supports representativeness.

It ALSO reports each scored drug's response magnitude (mean |true logFC| from an E6E7
artifact) as a descriptive check. NOTE: a full response-magnitude COMPARISON to the
background needs per-drug predictions/truth for the background drugs, which were not
produced (only the 9 were scored) -> that comparison is flagged, not fabricated.

REUSE: SMILES from the project's sci-Plex table; descriptors from RDKit. No metric is
reimplemented. REPORTED-NUMBER OWNERSHIP: supervisor-authored. `--selftest` proves the
descriptor + test glue (needs an env with rdkit, e.g. conda `cytobridge`/`drug`).

USAGE
-----
  python scored_drug_selection_bias.py --selftest
  python scored_drug_selection_bias.py \
      --smiles_csv ../../../sciplex3_drugs.clean.csv \
      --artifact t7_sub_loss_only --out_dir .
  # background defaults to ALL drugs in --smiles_csv; pass --background_csv with a
  # drug_id column to restrict it to the exact test-drug set.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_SMILES = _REPO / "sciplex3_drugs.clean.csv"
_DEFAULT_RES = _REPO / "student_progress_E6E7" / "E6E7" / "results"

SCORED_9 = ["AG-490", "Celecoxib", "Fulvestrant", "Ramelteon", "SL-327",
            "SRT3025 HCl", "Thalidomide", "Tofacitinib", "Zileuton"]
DESCRIPTORS = ["MW", "logP", "TPSA", "HBD", "HBA", "RotB"]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def descriptors(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {"MW": Descriptors.MolWt(mol), "logP": Crippen.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol), "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "RotB": Descriptors.NumRotatableBonds(mol)}


def match_drugs(names, smiles_map_norm):
    """Map requested names to SMILES; exact normalised match, then a prefix fallback for
    csv names that append a synonym/salt, e.g. 'AG-490' -> 'AG-490 (Tyrphostin B42)',
    'Tofacitinib' -> 'Tofacitinib (CP-690550) Citrate'.

    Returns (matched, missing, hit_keys). `hit_keys` holds the NORMALISED CSV names that were
    consumed, which is what a caller must exclude by. Keying the exclusion off `matched` (whose
    keys are the REQUESTED names) silently leaves every prefix-matched drug in the background:
    that bug put 'AG-490 (Tyrphostin B42)' and 'Tofacitinib (CP-690550) Citrate' on both sides of
    the Mann-Whitney and is why the shipped artifact reported n_background=176."""
    matched, missing, hit_keys = {}, [], set()
    keys = list(smiles_map_norm.keys())
    for n in names:
        key = _norm(n)
        if key in smiles_map_norm:
            matched[n] = smiles_map_norm[key]
            hit_keys.add(key)
            continue
        cands = [k for k in keys if len(key) >= 4 and k.startswith(key)]
        if cands:
            matched[n] = smiles_map_norm[cands[0]]
            hit_keys.add(cands[0])
            if len(cands) > 1:
                print(f"[warn] '{n}' prefix-matched {len(cands)} csv names; took '{cands[0]}'.")
        else:
            missing.append(n)
    return matched, missing, hit_keys


def descriptor_table(name_to_smiles):
    rows = []
    for n, smi in name_to_smiles.items():
        d = descriptors(smi)
        if d is None:
            continue
        rows.append({"drug": n, **d})
    return pd.DataFrame(rows)


def compare(scored_df, bg_df):
    rows = []
    for desc in DESCRIPTORS:
        s, b = scored_df[desc].values, bg_df[desc].values
        if len(s) < 1 or len(b) < 1:
            continue
        try:
            p = float(stats.mannwhitneyu(s, b, alternative="two-sided").pvalue)
        except ValueError:
            p = float("nan")
        rows.append({"descriptor": desc,
                     "scored_median": round(float(np.median(s)), 3),
                     "background_median": round(float(np.median(b)), 3),
                     "mannwhitney_p": round(p, 4),
                     "n_scored": len(s), "n_background": len(b)})
    return pd.DataFrame(rows)


def response_magnitude(args, scored_names):
    """Mean |true logFC| per scored drug from an E6E7 artifact (descriptive)."""
    res = Path(args.res_dir)
    f = res / f"logfc_true_{args.artifact}.npy"
    if not f.exists():
        print(f"[resp] artifact {f} absent; skipping response-magnitude (descriptive only).")
        return None
    true = np.load(f)
    meta = pd.read_csv(res / f"logfc_meta_{args.artifact}.csv")
    mag = np.abs(true).mean(axis=1)
    meta = meta.assign(resp_mag=mag)
    norm_scored = [_norm(n) for n in scored_names]

    def is_scored(name):
        k = _norm(name)
        return any(k == s or (len(s) >= 4 and k.startswith(s)) for s in norm_scored)

    sub = meta[meta["drug"].map(is_scored)]
    g = sub.groupby("drug")["resp_mag"].mean().round(4)
    return g


def run_real(args):
    smiles = pd.read_csv(args.smiles_csv)
    smap = {_norm(r.drug_id): r.smiles for r in smiles.itertuples()}
    scored_matched, scored_missing, scored_keys = match_drugs(SCORED_9, smap)
    if scored_missing:
        print(f"[warn] {len(scored_missing)} scored drugs unmatched in SMILES table: "
              f"{scored_missing} (fix names; not fabricating).")
    if args.background_csv:
        bg_names = pd.read_csv(args.background_csv)["drug_id"].tolist()
        bg_matched, bg_missing, _ = match_drugs(bg_names, smap)
        if bg_missing:
            print(f"[warn] {len(bg_missing)} background drugs unmatched in SMILES table: "
                  f"{bg_missing} (fix names; not fabricating).")
        print(f"[info] background = {len(bg_matched)} drugs from {args.background_csv}")
    else:
        # background = ALL sci-Plex drugs with SMILES (documented fallback). NOTE this is the
        # full screened library, NOT the profiled library the manuscript refers to; pass
        # --background_csv sciplex_profiled_172.csv to score against the profiled set.
        bg_matched = {r.drug_id: r.smiles for r in smiles.itertuples()}
        print(f"[info] background = all {len(bg_matched)} drugs in {args.smiles_csv} "
              "(pass --background_csv to restrict to the exact test-drug set).")
    # exclude the scored drugs from the background so the two groups are disjoint.
    # Compare against the normalised CSV names actually consumed, not the requested names.
    bg_matched = {n: s for n, s in bg_matched.items() if _norm(n) not in scored_keys}
    leaked = sorted(set(map(_norm, bg_matched)) & scored_keys)
    assert not leaked, f"scored drugs leaked into the background: {leaked}"

    scored_df = descriptor_table(scored_matched)
    bg_df = descriptor_table(bg_matched)
    print(f"[desc] scored n={len(scored_df)}, background n={len(bg_df)}")
    if args.expect_background is not None:
        assert len(bg_df) == args.expect_background, (
            f"background n={len(bg_df)}, expected {args.expect_background}")
    table = compare(scored_df, bg_df)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "selection_bias_descriptors.csv", index=False)
    print("\n=== scored-9 vs background physchem comparison (Mann-Whitney) ===")
    print(table.to_string(index=False))
    n_sig = int((table["mannwhitney_p"] < 0.05).sum())
    print(f"\n[interpret] {n_sig}/{len(table)} descriptors differ at p<0.05; "
          "few/none => the 9 scored drugs are chemically representative.")
    resp = response_magnitude(args, SCORED_9)
    if resp is not None:
        resp.to_csv(out / "scored_response_magnitude.csv")
        print("\nscored-drug response magnitude (mean |true logFC|, descriptive):")
        print(resp.to_string())
        print("[note] background response magnitude NOT available (only the 9 were "
              "scored); full magnitude comparison is a flagged extension, not fabricated.")
    print(f"\n[done] -> {out}/selection_bias_descriptors.csv")


def run_selftest(args):
    try:
        import rdkit  # noqa: F401
    except Exception:
        print("[selftest] rdkit NOT importable in this env -> run in conda `cytobridge`/`drug`. "
              "Skipping descriptor checks (the real run REQUIRES rdkit).")
        raise SystemExit("SELFTEST INCONCLUSIVE: rdkit missing (not a code failure).")
    # a few known drugs with SMILES -> descriptors must compute and the test must return p in [0,1]
    demo = {"aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "thalidomide": "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
            "biotin": "O=C(O)CCCCC1SCC2NC(=O)NC12"}
    df = descriptor_table(demo)
    assert len(df) == len(demo), "rdkit failed to parse a known SMILES"
    g1, g2 = df.iloc[:2], df.iloc[2:]
    tab = compare(g1, g2)
    print("\n=== SELFTEST descriptor table ===")
    print(df.round(2).to_string(index=False))
    print("\n=== SELFTEST Mann-Whitney (2 vs 3 demo split) ===")
    print(tab.to_string(index=False))
    ok = (len(df) == 5 and not tab.empty
          and tab["mannwhitney_p"].between(0, 1).all()
          and df["MW"].gt(0).all())
    print(f"\n[selftest] descriptors computed for all {len(df)} demo drugs; "
          f"p-values in [0,1]: {bool(tab['mannwhitney_p'].between(0,1).all())}")
    if not ok:
        raise SystemExit("SELFTEST FAILED: descriptor/Mann-Whitney glue wrong.")
    print("SELFTEST PASSED: RDKit descriptors + Mann-Whitney comparison verified.")


def main():
    p = argparse.ArgumentParser(description="T8 scored-drug selection-bias analysis")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--smiles_csv", type=Path, default=_DEFAULT_SMILES)
    p.add_argument("--background_csv", type=Path,
                   help="optional CSV with a drug_id column = exact test-drug set")
    p.add_argument("--expect_background", type=int,
                   help="assert the background size after exclusion (e.g. 163 for the "
                        "profiled library minus the nine scored drugs)")
    p.add_argument("--artifact", default="t7_sub_loss_only")
    p.add_argument("--res_dir", type=Path, default=_DEFAULT_RES)
    p.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    if args.selftest:
        run_selftest(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
