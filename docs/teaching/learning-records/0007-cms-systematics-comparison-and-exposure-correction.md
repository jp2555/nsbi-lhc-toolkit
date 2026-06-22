# 0007 — CMS Sec-9 systematics comparison; correcting the "exposure" framing

**Date:** 2026-06-20
**Source read:** CMS HH→bb̄ττ Run-II note AN-18-121 (Amendola et al.), Section 9 (pp. 90–97).
**Lessons updated:** L5 (new §4 CMS-vs-NSBI table + reframed plan); L4 (caveat box on "beats a 1-D fit").

## What CMS actually does (Sec 9)
Binned-template / Combine fit. Normalisation NPs (lumi, e/μ-ID, prefiring, PU, tt̄/QCD/DY norms, bkg σ, VBF
dipole recoil, κ_λ-dependent theory THU_HH = +4/−18%) + shape NPs. Shape mechanism: **shift objects (+MET, +SVfit)
→ re-run the fixed DNN/multiclass → re-histogram → up/down templates.** Observable = **binned DNN score**
(Bayesian-optimised binning); κ_λ via signal reweighting (2D m_HH×cosθ* / NLO).

## The correction (important)
My earlier "NSBI is exposed to more systematics than the histogram" framing assumed a **1-D m_HH histogram**.
CMS's real baseline bins a **multivariate DNN score** and propagates systematics by re-running that DNN on varied
inputs → it already carries the same multi-dimensional systematic info NSBI does. **NSBI is NOT more exposed.**

## The real CMS-vs-NSBI differences (just two)
1. **Counted vs learned:** CMS histograms its re-run DNN (no new estimation, exact up to MC stat); NSBI must *train*
   the per-event systematic weight (~2000 NNs) → the one ML layer that can be wrong even when physics is fine.
2. **What NSBI buys:** unbinning (no info lost to binning the score) + a κ_λ-optimal per-event ratio (CMS's DNN is
   one fixed discriminant). = the off-shell ~2× / FAIR-Universe ~20% gains.
Shared: sources, sizes, object propagation, factorization, profiling, κ_λ theory, QCD/fake modelling.

## Reframed plan
NSBI = "replace CMS's binned-DNN observable with an unbinned, κ_λ-optimal per-event likelihood, keeping the
identical systematic NPs." Gain = information; new cost = learned morphing + coverage validation; systematics budget
unchanged. "Worse than CMS" risk = the learned layer mis-calibrated/under-validated, NOT a heavier systematics burden.

## Learner state
Repeatedly catching my oversimplifications (1-D vs DNN-score; MC-vs-data stats; histogram systematics fairness).
Register is right; the thread has become a genuine method-design dialogue. Teaching series L1–L5 + degeneracy toy
now form a coherent, self-corrected whole.
