"""Fetch sci-Plex drug SMILES from PubChem on a local machine (useful when a
shared server IP has been rate-limited by sustained requests). Rate-limited + 404-fast +
retry. Writes sciplex3_drugs.csv with a header even if 0 rows resolve."""
import csv
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

NAMES_FILE = sys.argv[1] if len(sys.argv) > 1 else "sciplex3_missing_smiles.txt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "sciplex3_drugs.csv"

names = [ln.strip() for ln in open(NAMES_FILE) if ln.strip()]
print(f"fetching SMILES for {len(names)} drug names ...", flush=True)


def fetch(name, timeout=8, attempts=3):
    # PubChem 2025 renamed the JSON key CanonicalSMILES -> SMILES; the TXT endpoint
    # still returns the raw SMILES string regardless. RDKit re-canonicalizes downstream.
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           + urllib.parse.quote(name) + "/property/CanonicalSMILES/TXT")
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                txt = r.read().decode("utf-8").strip()
            return txt.splitlines()[0].strip() if txt else None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.0)
        except Exception:
            time.sleep(1.0)
    return None


rows = []
for i, n in enumerate(names, 1):
    s = fetch(n)
    if s:
        rows.append((n, s))
    print(f"[{i:03d}/{len(names)}] {n}: {'ok' if s else 'MISS'}", flush=True)
    time.sleep(0.3)

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["drug_id", "smiles"])
    for n, s in rows:
        w.writerow([n, s])
print(f"RESOLVED {len(rows)}/{len(names)} -> {OUT}", flush=True)
