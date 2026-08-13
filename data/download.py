"""
data/download.py
----------------
Download all datasets to data/raw/.

Run:
    python data/download.py --target sciplex
    python data/prepare_sciplex_scperturb.py --fetch_pubchem
    python data/download.py --target tahoe --slice 50000
    python data/download.py --target replogle
    python data/download.py --target gdsc2
    python data/download.py --target msigdb
    python data/download.py --target lincs
    python data/download.py --target drugbank        # (manual: requires academic license)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path


def download_file(url: str, out_file: Path) -> None:
    """Download `url` to `out_file`.

    Prefer curl because it can resume large interrupted downloads. Fall back to
    Python stdlib so the script still works on machines without curl.
    """
    if shutil.which("curl"):
        try:
            curl_environment = os.environ.copy()
            curl_environment.pop("LD_LIBRARY_PATH", None)
            subprocess.run(
                ["curl", "-L", "-C", "-", "-o", str(out_file), url],
                check=True,
                env=curl_environment,
            )
            return
        except subprocess.CalledProcessError as exc:
            print(
                f"[download] curl failed with exit code {exc.returncode}; trying Python fallback"
            )

    tmp_file = out_file.with_suffix(out_file.suffix + ".part")
    downloaded = 0
    with urllib.request.urlopen(url, timeout=60) as resp, open(
        tmp_file, "wb"
    ) as handle:
        total = int(resp.headers.get("Content-Length") or 0)
        while True:
            chunk = resp.read(1024 * 1024 * 8)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"[download] {downloaded / total:.1%}", end="\r")
            elif downloaded % (1024 * 1024 * 256) < len(chunk):
                print(f"[download] {downloaded / 1024 / 1024:.0f} MB")
    tmp_file.replace(out_file)
    print(f"\n[download] saved {out_file}")


def download_sciplex(out_dir: Path):
    """sci-Plex3 (Srivatsan et al. 2020), prepared h5ad route.

    Do not reconstruct this dataset from GEO raw files for this
    workflow. GEO GSE139944 is the original provenance, but this repository
    expects a single h5ad plus a drug-SMILES table. We therefore download the
    scPerturb/Zenodo h5ad directly, then run
    `python data/prepare_sciplex_scperturb.py --fetch_pubchem`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    url = (
        "https://zenodo.org/records/13350497/files/"
        "SrivatsanTrapnell2020_sciplex3.h5ad?download=1"
    )
    out_file = out_dir / "SrivatsanTrapnell2020_sciplex3.h5ad"
    final_file = out_dir / "sciplex3.h5ad"
    if final_file.exists():
        print(f"[sciplex] found {final_file}; skipping download")
    else:
        print(f"[sciplex] downloading scPerturb h5ad -> {out_file}")
        try:
            download_file(url, out_file)
        except Exception as exc:
            raise SystemExit(
                "[sciplex] download failed. Do not convert GEO raw files for this project.\n"
                f"Direct file URL: {url}\n"
                f"Save it as: {out_file}\n"
                "Then run: python data/prepare_sciplex_scperturb.py --fetch_pubchem"
            ) from exc
        print(f"[sciplex] downloaded {out_file}")
    print("[sciplex] next: python data/prepare_sciplex_scperturb.py --fetch_pubchem")


def download_tahoe(out_dir: Path, slice_size: int = 50000):
    """Download pinned Tahoe metadata only; expression follows bounded selection."""
    if slice_size != 50000:
        print("[tahoe] --slice is ignored by the metadata-first acceptance protocol")
    from data.download_tahoe_metadata import download_tahoe_metadata

    download_tahoe_metadata(out_dir)


def download_replogle(out_dir: Path):
    """Replogle K562 essential genes Perturb-seq (Cell 2022)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    code = """
import pertdata as pt
adata = pt.get_dataset('replogle_k562_essential')
adata.write_h5ad('%s/replogle_k562_essential.h5ad')
""" % str(out_dir)
    subprocess.run(["python", "-c", code], check=True)


def download_gdsc2(out_dir: Path):
    """GDSC2 IC50 data — public CSV from cancerrxgene.org."""
    out_dir.mkdir(parents=True, exist_ok=True)
    url = "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx"
    out_file = out_dir / "GDSC2_fitted_dose_response.xlsx"
    subprocess.run(["curl", "-L", "-o", str(out_file), url], check=True)


def download_msigdb(out_dir: Path):
    """MSigDB Hallmark v2024.1 — GMT format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        "manual step: download h.all.v2024.1.Hs.symbols.gmt from"
        " https://data.broadinstitute.org/gsea-msigdb/msigdb/release/"
        "2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt"
    )
    print("save as", out_dir / "h.all.v2024.1.Hs.symbols.gmt")
    print(
        "expected: 48690 bytes; sha256 "
        "ee2463540042078bfa3f67828e1e223bb354446d9fbb4d22845866835ba5c772"
    )


def download_lincs(out_dir: Path):
    """LINCS L1000 Level 5 (signatures). Requires registration at clue.io."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("manual step: register at https://clue.io and download:")
    print("  - cmap_signature_file (Level 5, ~1.3M signatures, ~10GB)")
    print("  - compoundinfo_beta.txt")
    print("  - cellinfo_beta.txt")
    print(f"save under {out_dir}/lincs/")


def download_drugbank(out_dir: Path):
    """DrugBank — academic license required."""
    print("manual step: academic license at https://go.drugbank.com/releases/latest")
    print(f"save XML under {out_dir}/drugbank/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=[
            "sciplex",
            "tahoe",
            "replogle",
            "gdsc2",
            "msigdb",
            "lincs",
            "drugbank",
            "all",
        ],
        required=True,
    )
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--slice", type=int, default=50000, help="slice size for Tahoe-100M"
    )
    args = parser.parse_args()
    targets = (
        ["sciplex", "tahoe", "replogle", "gdsc2", "msigdb", "lincs", "drugbank"]
        if args.target == "all"
        else [args.target]
    )
    for t in targets:
        print(f"\n=== {t} ===")
        fn = globals()[f"download_{t}"]
        if t == "tahoe":
            fn(args.out / t, slice_size=args.slice)
        else:
            fn(args.out / t)


if __name__ == "__main__":
    main()
