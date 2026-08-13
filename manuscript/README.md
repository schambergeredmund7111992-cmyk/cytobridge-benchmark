# Manuscript figure assets

The project paper uses hand-drawn Fig. 1 and Fig. 2. The reproducible code-backed figures are:

- Fig. 3 collapse overview: `generate_fig3_collapse.py`
- Fig. 4 control validation: `generate_fig4_control.py`
- Fig. 5 mechanism and case studies: `generate_fig5_mechanism.py`

Each script reads the small checked-in arrays/tables under `analysis/data*` and writes a PDF under `figs/`.

```bash
python manuscript/generate_fig3_collapse.py
python manuscript/generate_fig4_control.py
python manuscript/generate_fig5_mechanism.py
```
