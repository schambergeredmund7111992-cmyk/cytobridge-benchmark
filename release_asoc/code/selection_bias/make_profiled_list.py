"""Emit the profiled sci-Plex compound list that the manuscript's "profiled library" means.

The analysed h5ad carries 173 `perturbation` categories, one of which is `control`, so the
profiled library is 172 compounds. Nine of those are the scored held-out drugs, leaving the
163 that Section 4.1's physicochemical comparison is stated against.

This is deliberately NOT the same set as `sciplex3_drugs.clean.csv` (183 compounds with a
parseable SMILES = the full screened library). Scoring the comparison against that larger set is
what produced the shipped n_background=176.

    python3 make_profiled_list.py          # writes sciplex_profiled_172.csv next to this file
"""
import csv
import re
from pathlib import Path

import h5py

H5 = ("/Users/cgxmac/Desktop/CytoBridge/l20_transfer/data/processed/sciplex/"
      "sciplex_processed.h5ad")
OUT = Path(__file__).resolve().parent / "sciplex_profiled_172.csv"
SCORED_9 = ["AG-490", "Celecoxib", "Fulvestrant", "Ramelteon", "SL-327",
            "SRT3025", "Thalidomide", "Tofacitinib", "Zileuton"]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


with h5py.File(H5, "r") as f:
    cats = [c.decode() if isinstance(c, bytes) else c
            for c in f["obs"]["perturbation"]["categories"][:]]

assert len(cats) == 173, f"expected 173 perturbation categories, got {len(cats)}"
profiled = sorted(c for c in cats if c != "control")
assert len(profiled) == 172, f"expected 172 profiled compounds, got {len(profiled)}"

scored = [c for c in profiled if any(norm(c).startswith(norm(s)) for s in SCORED_9)]
assert len(scored) == 9, f"expected to match 9 scored drugs, matched {len(scored)}: {scored}"
assert len(set(profiled) - set(scored)) == 163

# several sci-Plex names carry a comma inside a synonym, e.g.
# "Baricitinib (INCB028050, LY3009104)", so the field must be quoted.
with OUT.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["drug_id"])
    w.writerows([[c] for c in profiled])
assert len(list(csv.DictReader(OUT.open()))) == 172, "round-trip of the written CSV failed"
print(f"[ok] 173 categories - 1 control = {len(profiled)} profiled compounds")
print(f"[ok] {len(scored)} of them are the scored held-out drugs -> background will be 163")
print(f"[done] -> {OUT}")
