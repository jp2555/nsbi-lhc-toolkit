# 0001 — NSBI-systematics landscape established

**Date:** 2026-06-19
**Lesson:** [0001-nsbi-systematics-field-map.html](../lessons/0001-nsbi-systematics-field-map.html)

## What was taught
The unifying frame for how NSBI handles systematics: the unbinned/HistFactory per-event model where
the hard factor is the **systematic morphing Δ_s(x|α)** (intractable, no known form, high-dim α).
Every method is a surrogate for Δ — except RS3L, which instead changes the representation.

Compared five works on two axes (model-in-likelihood vs robust-representation; and *how* Δ(α) is built):
R1 uncertainty-aware (seed), R2 ATLAS NSBI (full-scale, fixed-order classifiers), R3 HistFactory-v2/GP
(non-parametric + interpolation uncertainty), R4 Schöfbeck "refinable" (arbitrary-order ML ansatz,
non-factorizable, tested on FAIR Universe, ~20% tighter), R5 RS3L (robust representation).

## Key insight (non-obvious)
The five papers collapse to **one question** — approximate Δ(x|α) — plus one orthogonal idea (RS3L).
This reframes the user's own work: R4 is the closest match (same FAIR Universe benchmark); R3's
interpolation-uncertainty is the honest defense against morphing bias in a low-stat fit; RS3L is a
*complement* (robust input), not a substitute for the likelihood treatment.

## Zone of proximal development / next lessons (candidates)
- **L2:** Deep-dive on R4's factorization R(x|θ)·S(x|α)·g(x) and how to reproduce it on her FAIR Universe testbed (ties to the `parameter_fitting.py` / GOLLUM-style fit).
- **L2-alt:** The profile likelihood & coverage mechanics — how a learned Δ enters t_μ and where bias creeps in (calibration of fractions; the 0.3–0.4 bias cautionary tale).
- **L3:** GP morphing hands-on (pyhf-gpsys) — kernel/prior as inductive bias; propagating the interpolation covariance.
- Later: connect RS3L's pivot/info trade-off to her robustness diagnostic (shape-shift vs μ-CI).

## Open question to revisit
Does she want the next lesson to be **methods-deep (R4 reproduction)** or **stats-deep (profiling/coverage of learned ratios)**? Ask before building L2.
