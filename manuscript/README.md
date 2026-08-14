# Manuscript figure assets

The paper figures and the graphical abstract are shipped in
[`release_asoc/figures/`](../release_asoc/figures/), and the scripts that
regenerate every reported number (including the figure data) live in
[`release_asoc/code/`](../release_asoc/code/), driven by the single entry
point:

```bash
cd release_asoc
python reproduce.py
```

`reproduce.py` prints a PAPER vs RECOMPUTED verdict for every row of the
manuscript's tables (no training, no GPU, no network).
