# Resources

## Primary works compared in this thread

| # | Work | What it is | Trust |
|---|------|-----------|-------|
| R1 | Ghosh, Nachman, Whiteson, *Uncertainty-aware learning for HEP*, PRD 104 056026 (2021), [arXiv:2105.08742](https://arxiv.org/abs/2105.08742) | Parameterized classifier made **aware of nuisance parameters** beats decorrelation/invariance. Conceptual seed for ν-parameterized ratios. | High (PRD) |
| R2 | ATLAS Collaboration, *An implementation of NSBI for parameter estimation in ATLAS*, Rep. Prog. Phys. 88 (2025), [arXiv:2412.01600](https://arxiv.org/abs/2412.01600) | First **full-scale NSBI with many systematics**: per-event learned likelihood ratios, ν as nuisance params, profile-likelihood CIs + diagnostics. Off-shell H→ZZ→4ℓ. | Highest (ATLAS) |
| R3 | Cranmer, Feickert, Heinrich, Held, Sandesara, *HistFactory v2: statistical modeling using Gaussian Process Regression*, SBI Blueprint Workshop (Feb 2026); [pyhf-gpsys](https://github.com/pyhf/pyhf-gpsys) (L. Heinrich) | **GP regression** for the systematic morphing Δ(α): non-parametric, high-dim, drops factorization + analytic ansatz, gives an **interpolation uncertainty**. Binned + unbinned/SBI. | High (IRIS-HEP, pre-pub) |
| R4 | Barrué, Benato, Giordano, Li, Schöfbeck, Schwarz, Shooshtari, Wang, *Unbinned measurements with machine-learned systematic uncertainties*; method in **MLST 6 015007 (2025)**, tested in **PRD 112 052006 (2025)**; tool: GOLLUM | Per-event **ML surrogate for systematics**, parametric ansatz in NPs (arbitrary order, factorizable + non-factorizable), uses kinematics for \|α\|<1, joint LR. **Tested on FAIR Universe**, ~20% tighter 1σ vs binned. | High (MLST+PRD) |
| R5 | Harris et al., *Re-Simulation-based Self-Supervised Learning (RS3L)*, PRD 111 032010 (2025), [arXiv:2403.07066](https://arxiv.org/abs/2403.07066) | Orthogonal philosophy: build a **systematics-robust representation** (re-simulation contrastive SSL) so the discriminant is *insensitive*, rather than modeling ν in the likelihood. | High (PRD) |

## Data-driven backgrounds & ABCDnn (Lesson 6)

| Source | What it is | Trust |
|--------|-----------|-------|
| Choi, Lim, Oh, *Data-driven Estimation of Background Distribution through Neural Autoregressive Flows* (ABCDnn), [arXiv:2008.03636](https://arxiv.org/abs/2008.03636) | **Primary source for L6.** Normalizing-flow generalization of ABCD → a learned multivariate conditional background density p_bg(x\|control). The object NSBI needs. | High |
| Huang et al., *Neural Autoregressive Flows*, [arXiv:1804.00779](https://arxiv.org/abs/1804.00779) | The NAF architecture ABCDnn is built on (invertible monotonic map). | High |
| CMS, *Search for nonresonant HH→bbττ*, full Run 2, [arXiv:2206.09401](https://arxiv.org/abs/2206.09401) | The bbττ background recipe (QCD ABCD, fakes, DY, tt̄). Pair with the AN for region cuts. | Highest (CMS) |
| CMS, *H→ττ* full Run 2, [arXiv:2204.12957](https://arxiv.org/abs/2204.12957) | Canonical fake-factor (jet→τ_h) treatment + DY handling context. | Highest (CMS) |
| CMS, *τ-embedding technique*, [arXiv:1903.01216](https://arxiv.org/abs/1903.01216) | The data-driven Z→ττ method (data event, simulated τ decay). | High |
| ATLAS, *Universal / flavor-decomposed Fake Factor*, [arXiv:2502.04156](https://arxiv.org/abs/2502.04156) | Relaxes the fake-composition-stability assumption (per-flavor FF). | High |
| *Neural Fake Factor Estimation* (density-ratio FF), [arXiv:2511.06972](https://arxiv.org/abs/2511.06972), JHEP 04(2026)188 | Recasts the fake factor as a per-event neural density ratio — directly NSBI-compatible. | High |
| S. An (CMU), four-top all-hadronic PhD thesis — ExtendedABCD + ABCDnn deployment | The one real ABCDnn-in-an-analysis precedent (BDT+H_T shape, binned fit). | Thesis/internal |
| CMS Open Data Workshop, [ABCD method lesson](https://cms-opendata-workshop.github.io/) | Plain-language ABCD tutorial. | Pedagogical |

## Context / benchmark
- FAIR Universe HiggsML Uncertainty dataset & competition, [arXiv:2410.02867](https://arxiv.org/abs/2410.02867) — the shared benchmark (real TES/JES/soft-MET + bkg-norm nuisances).
- Workshop home: CERN SBI Blueprint Workshop, indico event [1600677](https://indico.cern.ch/event/1600677/).

## Communities (for wisdom)
- **IRIS-HEP / SBI Blueprint** working group & Statistics Forum (the venue where R3 was presented).
- ATLAS/CMS Statistics Committees (internal) — for the profiling/coverage subtleties.
- `scikit-hep`/`pyhf` GitHub discussions — for the HistFactory-v2 / GP implementation.
