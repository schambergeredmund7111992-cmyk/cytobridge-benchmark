# T3 / T4 external-dependency feasibility (supervisor recon, 2026-06-22)

Recon for two KBS supplementary experiments. Every claim below was web-verified; URLs cited.
Decision-relevant bottom line up front:

- **T3 (PerturBench / PDS comparison): GO — build a lightweight in-repo skeleton.** The rank
  metric is a published closed form computable on our existing 27×3000 (pred, true) matrix;
  no need to install PerturBench's pipeline. Done: `supp_T3/perturbench_rank.py`.
- **T4 (extra external SOTA): CPA = NO, scGen = NO, Biolord = the only good optional add.**
  The existing audit panel (Ridge + chemCPA + scGPT-zs + CytoBridge, all collapse) already
  satisfies "audit 2–3 SOTA". Any add runs on cityu A100 (supervisor, 铁律 1), so the
  student 5090 / sm_120 / CUDA-12.8 risk is irrelevant to T4.

---

## T3 — PerturBench rank / Perturbation Discrimination Score

**Repo:** [altoslabs/perturbench](https://github.com/altoslabs/perturbench/) (Altos Labs).
**Paper:** PerturBench, [arXiv:2408.10609](https://arxiv.org/abs/2408.10609).

**Exact metric (paper Eq. 2), verbatim:**
> rank_average = (1/p) Σᵢ rank(x̂ᵢ),  rank(x̂ᵢ) = (1/(p−1)) Σ_{j≠i} 𝕀( dist(x̂ⱼ, xᵢ) ≤ dist(x̂ᵢ, xᵢ) )

where p = #perturbations, x̂ᵢ/xᵢ = predicted/observed mean expression of perturbation i, dist
is a generic distance (paper instantiates "Cosine LogFC rank", "RMSE rank"). Range 0 (perfect)
→ 0.5 (random) → 1 (worst).

**Relationship to our control (the novelty hook the supervisor writes formally):** our
off-diagonal `specificity_auc` = mean fraction of OTHER pairs the on-diagonal beats (1 best,
0.5 random); PerturBench `rank` = mean fraction of OTHER predictions closer than the correct one
(0 best, 0.5 random). They are the SAME nearest-neighbour discrimination idea, ≈ `rank ≈ 1 − AUC`
**when computed over the same candidate set**. The DIFFERENTIATION: PerturBench ranks each
prediction against ALL perturbations' truths across the whole matrix (mixes drug- and cell-line
structure), whereas our control conditions WITHIN cell line (same-cell-line off-diagonal),
isolating drug-specificity from shared cell-line structure. That same-cell-line conditioning is
the methodological contribution; T3 shows the numbers side by side to prove it is not a mere
reparameterisation of PerturBench rank.

**Feasibility:** the rank formula is trivial to implement faithfully (~30 lines) and needs only
our (pred, true, cell_line) arrays — NO PerturBench install (their repo pulls a heavy
hydra/lightning stack we don't need just for one metric). Implemented + self-tested in
`supp_T3/perturbench_rank.py` (Oracle → rank 0, random → 0.5, both within- and cross-cell-line).
The "Perturbation Discrimination Score" (PDS) = inverse normalised rank, popularised by Cell-Eval
/ the STATE model ([biorxiv 2025.06.26.661135](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v1.full));
it is the same rank quantity reported as 1 − normalised-rank, so the same script covers it.

**Verdict: GO.** In-repo skeleton, no external dependency, reported-number → supervisor-owned.

---

## T4 — extra external SOTA to audit under the control

Goal: 1–2 MORE published models beyond chemCPA + scGPT-zs (already collapse), to show the
control audits the field. Requirement: must support **unseen-DRUG** holdout (train on a drug set,
predict held-out drugs) — the same protocol as CytoBridge.

### CPA — NO (redundant + degenerate on unseen drugs)
[theislab/cpa](https://github.com/theislab/cpa) ([cpa-tools.readthedocs.io](https://cpa-tools.readthedocs.io/)),
biorxiv 2021. CPA learns a **categorical/one-hot-style perturbation embedding**; an unseen drug
has no learned embedding, so on a truly held-out drug it degenerates exactly like the rolled-back
one-hot Ridge (no molecular structure → no signal). **chemCPA (already done) IS the fix** — it is
CPA + molecular features for unseen compounds. Adding plain CPA would either (a) duplicate chemCPA
or (b) reproduce the one-hot degeneracy we already discredited. No value.

### scGen — NO (wrong task: unseen cell-type, not unseen drug)
[theislab/scgen](https://github.com/theislab/scgen) (Nat Methods 2019, scvi-tools). scGen predicts
a perturbation's effect on a **cell type seen only in control**, via latent vector arithmetic
δ = mean(perturbed) − mean(control) computed from a context where the perturbation **was observed**,
then transferred. For an unseen drug — observed in NO cell line — there is no δ to transfer.
Architecturally inapplicable to the unseen-drug holdout. No-go.

### Biolord — the ONLY good optional add (GO-WITH-FRICTION)
[nitzanlab/biolord](https://github.com/nitzanlab/biolord) (Nat Biotech 2024,
[s41587-023-02079-x](https://www.nature.com/articles/s41587-023-02079-x)). Disentanglement model
that **explicitly predicts unseen-drug response on sci-Plex 3** (benchmarked at the 10 µM strict
setting, the published unseen-drug protocol). This is purpose-built for exactly our holdout, so a
collapse here would be the strongest possible audit point ("even a model designed for unseen-drug
sci-Plex fails the control"). Friction: scvi-tools-style training stack, must be trained (no drop-in
checkpoint for our split), runs on **cityu A100** (supervisor, 铁律 1 reported-number baseline).

### Recommendation
The panel is already sufficient (Ridge / chemCPA / scGPT-zs / CytoBridge all collapse, Oracle
passes). If the supervisor wants ONE more to maximise impact, **Biolord** is the pick — not CPA,
not scGen. This is a supervisor-run cityu task (like chemCPA was), NOT a student skeleton.
**Open decision for the user:** add Biolord (≈ chemCPA-level effort on cityu) or ship the current
4-model panel as-is? Both are defensible; Biolord strengthens the "purpose-built models also
collapse" claim at a real time cost.

---

## Sources
- PerturBench: https://github.com/altoslabs/perturbench/ · https://arxiv.org/abs/2408.10609 · https://arxiv.org/html/2408.10609v4
- PDS / Cell-Eval / STATE: https://www.biorxiv.org/content/10.1101/2025.06.26.661135v1.full
- CPA: https://github.com/theislab/cpa · https://cpa-tools.readthedocs.io/
- scGen: https://github.com/theislab/scgen · https://www.nature.com/articles/s41592-019-0494-8
- Biolord: https://github.com/nitzanlab/biolord · https://www.nature.com/articles/s41587-023-02079-x
- Linear-baseline context (already cited): https://www.nature.com/articles/s41592-025-02772-6
