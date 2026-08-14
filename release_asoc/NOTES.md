# Execution-environment note

This supplement is the author's release package **as it was run on the
author's macOS machine**. A few analysis scripts under `code/` therefore
contain hard-coded `/Users/cgxmac/...` paths that reflect that execution
environment (e.g. `code/baselines/compute_baseline_control.py`,
`code/consistency_sweep.py`); they are kept byte-identical to the author's
original run and are expected to be re-run in the same environment.

The entry point `reproduce.py` is path-relative and runs on any machine with
numpy / pandas / scipy / pyarrow:

```bash
cd release_asoc
python reproduce.py
```

Software in this package is MIT-licensed; derived data is CC BY 4.0 (see
`LICENSE`).
