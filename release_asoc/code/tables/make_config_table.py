"""Build the per-configuration hyper-parameter table for the seven audited loss configurations.

Both cold reviews asked for this table: Section 3.4 states that the seven configurations "differ
only in the reconstruction, drug-specificity, and delta weights and in one decoder-normalization
flag", but never prints the values, so a reader cannot tell what "loss-only" or "norm-only" mean.

Source of record is the hparams.yaml Lightning wrote for the run that actually produced each
stored prediction file, NOT the configs/train/*.yaml, which may have been edited afterwards.

    python3 make_config_table.py

Writes data/config_hyperparams.csv and prints the LaTeX tabular body.

KNOWN GAP: t7_sub_drugspec3 has stored predictions under E6E7/results/ but no logs/ directory
and no config file anywhere in the tree, so its weights are NOT recorded. It is emitted with
empty cells rather than with values inferred from the x1/x5 naming convention.
"""
import glob
import os
import re
import csv
from pathlib import Path

E6E7 = "/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7"
OUT = Path(__file__).resolve().parent / "data" / "config_hyperparams.csv"

# paper name -> artifact stem, in the order Table 4 uses
CONFIGS = [("norm-only", "t7_sub_norm_only"), ("recovery baseline", "t6_sub_baseline"),
           ("drug-spec x1", "t7_sub_drugspec1"), ("low recon weight", "t7_sub_lamrecon01"),
           ("loss-only", "t7_sub_loss_only"), ("drug-spec x5", "t7_sub_drugspec5"),
           ("drug-spec x3", "t7_sub_drugspec3")]
VARYING = ["lam_recon", "lam_drugspec", "lam_delta"]
FIXED = {"lam_contrast": "0.5", "lam_pathway": "0.3", "lam_kl": "0.05", "lam_direction": "1.0"}
FLAG = "dec_in_component_norm"


def read_hparams(stem):
    """Return the hparams of the run that produced this stem, or None if not retained."""
    versions = sorted(glob.glob(f"{E6E7}/logs/{stem}/version_*/hparams.yaml"))
    if not versions:
        return None
    parsed = []
    for v in versions:
        txt = open(v).read()
        d = {}
        for k in VARYING + list(FIXED) + [FLAG]:
            m = re.search(rf"^\s+{k}:\s*(\S+)", txt, re.M)
            d[k] = m.group(1) if m else None
        parsed.append((os.path.basename(os.path.dirname(v)), d))
    # several stems have more than one version; they must agree or the table is ambiguous
    first = parsed[0][1]
    for ver, d in parsed[1:]:
        assert d == first, f"{stem}: {ver} disagrees with {parsed[0][0]}; cannot name one config"
    return first


rows, missing = [], []
for name, stem in CONFIGS:
    h = read_hparams(stem)
    if h is None:
        missing.append((name, stem))
        rows.append({"configuration": name, "artifact_stem": stem,
                     **{k: "" for k in VARYING}, FLAG: "", "source": "NOT RETAINED"})
        continue
    for k, want in FIXED.items():
        assert float(h[k]) == float(want), (
            f"{stem}: {k}={h[k]}, but Section 3.4 says it stays fixed at {want}")
    rows.append({"configuration": name, "artifact_stem": stem,
                 **{k: h[k] for k in VARYING}, FLAG: h[FLAG], "source": "logs/*/hparams.yaml"})

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print(f"[ok] fixed weights verified constant across {len(rows)-len(missing)} recorded runs: "
      + ", ".join(f"{k}={v}" for k, v in FIXED.items()))
if missing:
    print(f"[gap] config NOT retained for: {[m[1] for m in missing]} -> emitted as blank cells")
print(f"[done] -> {OUT}\n")

print("% ---- LaTeX tabular body ----")
for r in rows:
    tex = r["configuration"].replace("x1", "$\\times 1$").replace("x3", "$\\times 3$") \
                            .replace("x5", "$\\times 5$")
    if r["source"] == "NOT RETAINED":
        print(f"{tex} & --- & --- & --- & --- \\\\")
        continue
    flag = "yes" if r[FLAG] == "true" else "no"
    print(f"{tex} & {float(r['lam_recon']):g} & {float(r['lam_drugspec']):g} & "
          f"{float(r['lam_delta']):g} & {flag} \\\\")
