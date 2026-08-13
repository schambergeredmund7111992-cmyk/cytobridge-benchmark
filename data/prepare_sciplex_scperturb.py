"""
data/prepare_sciplex_scperturb.py
---------------------------------
Normalize the scPerturb/Zenodo sci-Plex3 h5ad for this repository.

The public scPerturb file is already an AnnData object, but its obs columns
may use names such as `product_name` and `cell_type`, while our preprocessing
pipeline expects:
    obs["drug"]
    obs["cell_line"]
and a companion CSV:
    data/raw/sciplex/sciplex3_drugs.csv  with columns drug_id,smiles

Run after `python data/download.py --target sciplex`:
    python data/prepare_sciplex_scperturb.py --fetch_pubchem
"""
from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


DRUG_COLUMN_CANDIDATES = [
    "drug", "product_name", "perturbation", "condition", "treatment", "compound",
]
CELL_COLUMN_CANDIDATES = [
    "cell_line", "cell_type", "celltype", "cell", "cell_name",
]
SMILES_COLUMN_CANDIDATES = [
    "smiles", "SMILES", "canonical_smiles", "CanonicalSMILES", "rdkit_smiles",
]
CONTROL_VALUES = {
    "DMSO", "dmso", "Vehicle", "vehicle", "control", "Control",
    "vehicle_control", "DMSO control",
}


def choose_column(columns: list[str], explicit: str | None, candidates: list[str], label: str) -> str:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"--{label}_col={explicit!r} not found. Available columns: {columns}")
        return explicit
    for col in candidates:
        if col in columns:
            return col
    raise ValueError(
        f"Could not infer {label} column. Available columns:\n"
        + "\n".join(f"  - {c}" for c in columns)
    )


def fetch_pubchem_smiles(name: str, timeout: int = 8, attempts: int = 2) -> str | None:
    # Use the TXT endpoint: PubChem (2025) renamed the JSON key CanonicalSMILES ->
    # SMILES, so json.get("CanonicalSMILES") now silently returns None. TXT returns
    # the raw SMILES string regardless; RDKit re-canonicalizes downstream.
    quoted = urllib.parse.quote(str(name))
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quoted}/property/CanonicalSMILES/TXT"
    )
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                txt = resp.read().decode("utf-8").strip()
            return txt.splitlines()[0].strip() if txt else None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None          # name genuinely not in PubChem -> fast, no retry
            time.sleep(0.6)          # 503/throttle -> back off and retry once
        except Exception:
            time.sleep(0.6)          # timeout / transient network -> retry once
    return None


def build_smiles_table(
    obs: pd.DataFrame,
    drug_col: str,
    smiles_col: str | None,
    fetch_pubchem: bool,
    control_label: str,
) -> tuple[pd.DataFrame, list[str]]:
    drugs = sorted(d for d in obs[drug_col].dropna().astype(str).unique() if d != control_label)
    smiles_by_drug: dict[str, str] = {}

    if smiles_col is not None:
        sub = obs[[drug_col, smiles_col]].dropna().drop_duplicates()
        for _, row in sub.iterrows():
            drug = str(row[drug_col])
            smi = str(row[smiles_col]).strip()
            if drug != control_label and smi and smi.lower() not in {"nan", "none"}:
                smiles_by_drug.setdefault(drug, smi)

    missing = [d for d in drugs if d not in smiles_by_drug]
    if fetch_pubchem and missing:
        print(f"[prepare] fetching PubChem SMILES for {len(missing)} drugs ...")
        for i, drug in enumerate(missing, start=1):
            smi = fetch_pubchem_smiles(drug)
            if smi:
                smiles_by_drug[drug] = smi
            print(f"  [{i:03d}/{len(missing):03d}] {drug}: {'ok' if smi else 'missing'}", flush=True)
            time.sleep(0.34)          # <=3 req/s: stay under PubChem's rate limit (throttling -> timeouts)

    still_missing = [d for d in drugs if d not in smiles_by_drug]
    rows = [{"drug_id": drug, "smiles": smiles_by_drug[drug]} for drug in drugs if drug in smiles_by_drug]
    return pd.DataFrame(rows), still_missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input scPerturb h5ad. Defaults to data/raw/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad, falling back to sciplex3.h5ad.",
    )
    parser.add_argument("--out_h5ad", type=Path, default=Path("data/raw/sciplex/sciplex3.h5ad"))
    parser.add_argument("--smiles_csv", type=Path, default=Path("data/raw/sciplex/sciplex3_drugs.csv"))
    parser.add_argument("--drug_col", default=None)
    parser.add_argument("--cell_col", default=None)
    parser.add_argument("--smiles_col", default=None)
    parser.add_argument("--control_label", default="DMSO")
    parser.add_argument("--fetch_pubchem", action="store_true")
    args = parser.parse_args()

    import scanpy as sc

    default_input = Path("data/raw/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad")
    fallback_input = Path("data/raw/sciplex/sciplex3.h5ad")
    input_path = args.input or (default_input if default_input.exists() else fallback_input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing input h5ad: {input_path}. Run `python data/download.py --target sciplex` first."
        )
    for output in (args.out_h5ad, args.smiles_csv):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite prepared sci-Plex output: {output}")

    print(f"[prepare] reading {input_path}")
    adata = sc.read_h5ad(input_path)
    columns = list(map(str, adata.obs.columns))
    drug_src = choose_column(columns, args.drug_col, DRUG_COLUMN_CANDIDATES, "drug")
    cell_src = choose_column(columns, args.cell_col, CELL_COLUMN_CANDIDATES, "cell")
    smiles_src = args.smiles_col
    if smiles_src is None:
        smiles_src = next((c for c in SMILES_COLUMN_CANDIDATES if c in columns), None)
    elif smiles_src not in columns:
        raise ValueError(f"--smiles_col={smiles_src!r} not found. Available columns: {columns}")

    print(f"[prepare] drug column: {drug_src}")
    print(f"[prepare] cell column: {cell_src}")
    print(f"[prepare] smiles column: {smiles_src or '(none; will use PubChem if requested)'}")

    adata.obs["drug"] = adata.obs[drug_src].astype(str)
    adata.obs["cell_line"] = adata.obs[cell_src].astype(str)
    adata.obs.loc[adata.obs["drug"].isin(CONTROL_VALUES), "drug"] = args.control_label

    smiles_df, missing = build_smiles_table(
        adata.obs,
        drug_col="drug",
        smiles_col=smiles_src,
        fetch_pubchem=args.fetch_pubchem,
        control_label=args.control_label,
    )
    if missing:
        missing_path = args.smiles_csv.with_name("sciplex3_missing_smiles.txt")
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        missing_path.write_text("\n".join(missing) + "\n")
        raise SystemExit(
            f"[prepare] missing SMILES for {len(missing)} drugs. See {missing_path}. "
            "Fill the missing drug_id,smiles rows in sciplex3_drugs.csv from PubChem/DrugBank "
            "before running preprocess.py."
        )

    args.out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    args.smiles_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_h5ad = args.out_h5ad.with_suffix(".tmp.h5ad")
    tmp_smiles = args.smiles_csv.with_suffix(".tmp.csv")
    for temporary in (tmp_h5ad, tmp_smiles):
        if temporary.exists():
            raise FileExistsError(f"Stale preparation staging file requires review: {temporary}")
    adata.write_h5ad(tmp_h5ad)
    smiles_df.to_csv(tmp_smiles, index=False)
    tmp_h5ad.replace(args.out_h5ad)
    tmp_smiles.replace(args.smiles_csv)
    print(f"[prepare] wrote normalized h5ad -> {args.out_h5ad} ({adata.shape})")
    print(f"[prepare] wrote {len(smiles_df)} SMILES -> {args.smiles_csv}")


if __name__ == "__main__":
    main()
