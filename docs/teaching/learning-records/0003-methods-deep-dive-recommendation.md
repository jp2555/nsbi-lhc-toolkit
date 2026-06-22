# 0003 — Methods deep-dive + κ_λ recommendation

**Date:** 2026-06-19
**Lesson:** [0003-methods-expanded-and-recommendation.html](../lessons/0003-methods-expanded-and-recommendation.html)

## What was taught
Each of the 5 works expanded (idea / mechanism / strength / limit), the "family tree" relating them,
a reasoned comparison (expressiveness vs honesty trade-off), and a **layered recommendation** for
HH→bbττ / κ_λ.

## Key framings introduced
- **Family tree:** R1 (principle) → R2 (ATLAS, fixed-order factorized baseline) → {R3 GP, R4 refinable}
  as two upgrades of the morphing; R5 (RS3L) is off-tree (changes inputs, composes with R2–R4).
- **The core trade-off inside "model-it":** R4 = expressiveness; R3 = honesty (uncertainty on the
  morphing). Combinable.
- **κ_λ lives in R(x|θ)**; systematics in S(x|α); fractions in g(x) — modelled separately. This makes
  R4's R·S·g factorization the natural home for her existing κ_λ morphing.

## Recommendation (layered)
1. Foundation = ATLAS machinery (R2): fit/CI/diagnostics/training-stat uncertainty.
2. Core = R4 refinable surrogate for the systematic morphing (proven on her FAIR Universe benchmark;
   handles JES/τES + cross-terms; ~20% CI gain). Watch fraction calibration (0.3–0.4 bias if dropped).
3. Honesty = R3 GP interpolation-uncertainty for dominant, sparsely-simulated knobs (she's stat-starved).
4. Complement = R5 RS3L robust inputs for shower/hadronization, later — connects to her FM/Sophon thread.

## Next-lesson fork (asked in lesson)
(a) hands-on R4 R·S·g factorization on her FAIR Universe pipeline, (b) deeper GP morphing, or
(c) a diagram of the family-tree / trade-off. Await her pick. Register now well-calibrated (Lesson 2 level).
