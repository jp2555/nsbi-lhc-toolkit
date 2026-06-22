# Resources

## Primary works compared in this thread

| # | Work | What it is | Trust |
|---|------|-----------|-------|
| R1 | Ghosh, Nachman, Whiteson, *Uncertainty-aware learning for HEP*, PRD 104 056026 (2021), [arXiv:2105.08742](https://arxiv.org/abs/2105.08742) | Parameterized classifier made **aware of nuisance parameters** beats decorrelation/invariance. Conceptual seed for ν-parameterized ratios. | High (PRD) |
| R2 | ATLAS Collaboration, *An implementation of NSBI for parameter estimation in ATLAS*, Rep. Prog. Phys. 88 (2025), [arXiv:2412.01600](https://arxiv.org/abs/2412.01600) | First **full-scale NSBI with many systematics**: per-event learned likelihood ratios, ν as nuisance params, profile-likelihood CIs + diagnostics. Off-shell H→ZZ→4ℓ. | Highest (ATLAS) |
| R3 | Cranmer, Feickert, Heinrich, Held, Sandesara, *HistFactory v2: statistical modeling using Gaussian Process Regression*, SBI Blueprint Workshop (Feb 2026); [pyhf-gpsys](https://github.com/pyhf/pyhf-gpsys) (L. Heinrich) | **GP regression** for the systematic morphing Δ(α): non-parametric, high-dim, drops factorization + analytic ansatz, gives an **interpolation uncertainty**. Binned + unbinned/SBI. | High (IRIS-HEP, pre-pub) |
| R4 | Barrué, Benato, Giordano, Li, Schöfbeck, Schwarz, Shooshtari, Wang, *Unbinned measurements with machine-learned systematic uncertainties*; method in **MLST 6 015007 (2025)**, tested in **PRD 112 052006 (2025)**; tool: GOLLUM | Per-event **ML surrogate for systematics**, parametric ansatz in NPs (arbitrary order, factorizable + non-factorizable), uses kinematics for \|α\|<1, joint LR. **Tested on FAIR Universe**, ~20% tighter 1σ vs binned. | High (MLST+PRD) |
| R5 | Harris et al., *Re-Simulation-based Self-Supervised Learning (RS3L)*, PRD 111 032010 (2025), [arXiv:2403.07066](https://arxiv.org/abs/2403.07066) | Orthogonal philosophy: build a **systematics-robust representation** (re-simulation contrastive SSL) so the discriminant is *insensitive*, rather than modeling ν in the likelihood. | High (PRD) |

## Context / benchmark
- FAIR Universe HiggsML Uncertainty dataset & competition, [arXiv:2410.02867](https://arxiv.org/abs/2410.02867) — the shared benchmark (real TES/JES/soft-MET + bkg-norm nuisances).
- Workshop home: CERN SBI Blueprint Workshop, indico event [1600677](https://indico.cern.ch/event/1600677/).

## Communities (for wisdom)
- **IRIS-HEP / SBI Blueprint** working group & Statistics Forum (the venue where R3 was presented).
- ATLAS/CMS Statistics Committees (internal) — for the profiling/coverage subtleties.
- `scikit-hep`/`pyhf` GitHub discussions — for the HistFactory-v2 / GP implementation.
