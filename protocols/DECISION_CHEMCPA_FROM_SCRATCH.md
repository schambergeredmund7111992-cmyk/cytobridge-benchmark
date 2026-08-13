# chemCPA Initialization Decision

Date: 2026-07-11. Status: PI-approved before any revised training result existed.

The frozen selector `4f061dbfc7af05cf84f06a724b0c8563/last.ckpt` is referenced by the
pinned chemCPA and biolord-reproducibility configurations, but it is not present in the authors'
public `chemCPA_models.zip` archive.

Observed archive evidence:

- URL: `https://f003.backblazeb2.com/file/chemCPA-models/chemCPA_models.zip`
- Bytes: `254487527`
- SHA-256: `322606b8a209de44228cf51d9930e4173b21a79dfc780fab3c1a2f1e916bec8f`
- The archive contains published final `.pt` models; its README identifies the pretrained
  extended-gene RDKit result as `d2686f53a55468497195941fac1d7e5e.pt`.

That final model has already been fit on sci-Plex and is therefore inadmissible as initialization
for the frozen held-out-compound benchmark. The approved replacement is the official chemCPA
architecture at the pinned source commit, trained from scratch under the six preregistered
configurations and fixed 201-epoch budget. Results must be labelled `chemCPA (from scratch)`.
