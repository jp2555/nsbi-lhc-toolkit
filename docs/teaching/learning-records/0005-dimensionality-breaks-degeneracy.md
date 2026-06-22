# 0005 — Refinement: more features break the κ_λ–systematic degeneracy

**Date:** 2026-06-19
**Revises:** [0004-kappa-lambda-capstone.md](0004-kappa-lambda-capstone.md) (and the Lesson-4 diagram)

## Trigger
User pushed back: "we tested 10 event-level features with NSBI — can we not consider all of them?"
Correctly spotted that the Lesson-4 m_HH degeneracy picture is a **1-D projection artifact**.

## The corrected understanding
- κ_λ and JES are degenerate only in a *single projection*. In the **full feature space** they leave
  **different fingerprints** (κ_λ = coherent interference deformation; JES = detector rescaling that moves
  m_bb + b-jet pT together). Using all features lets the per-event likelihood disentangle them.
- "Use all features" ≠ decorrelation; it's the **aware/model-it route at full power** (opposite of throwing
  info away). It both maximizes κ_λ sensitivity AND breaks the degeneracy, and lets the data constrain
  nuisances **in situ**.
- Empirical proof on her own benchmark: R4 FAIR Universe unbinned fit tightened JES ±0.095 → ±0.045.
- **Cost / the catch:** every added feature means modeling how each systematic deforms the *joint* density →
  more features raise the stakes on the morphing (R4 expressive, R3 honest), not remove it. Diminishing κ_λ
  returns vs growing morphing burden = the sweet-spot trade for "how many features / constituents."

## Why it matters
This is the load-bearing reason multivariate NSBI beats a 1-D m_HH fit for κ_λ — *systematic
disentanglement*, not just statistical efficiency. Lesson 4 updated with a "dimensionality breaks the
degeneracy" box so the diagram isn't read as a literal dead-end.

## Learner state
Sharp — is now reasoning about the method from first principles and catching oversimplifications. Ready for
hands-on (R·S·g factorization on FAIR Universe, or a toy quantifying the κ_λ–JES degeneracy and how it
shrinks as features are added).
