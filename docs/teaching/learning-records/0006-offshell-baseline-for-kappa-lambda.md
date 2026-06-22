# 0006 — Off-shell NSBI recipe as the κ_λ baseline (+ what's harder)

**Date:** 2026-06-20
**Lesson:** [0005-offshell-nsbi-recipe-for-kappa-lambda.html](../lessons/0005-offshell-nsbi-recipe-for-kappa-lambda.html)
**Source read:** ATLAS off-shell NSBI note HIGG-2023-01 (Sandesara et al. = arXiv:2412.01600), Section 5 (pp. 79–90).

## The recipe (how systematics is done in the first NSBI measurement)
Factorization to avoid a 100-NP network:
1. Reference hypothesis P_ref; NPs in rate + unbinned density; Gaussian constraints (5.1.1).
2. Per-event density ratio = α-independent piece × systematic weight; weight factorizes over NPs and is built by
   per-event polynomial-exponential interpolation between ±1σ anchors (5.1.2–5.1.3). They explicitly reject
   feeding α to the NN (Cranmer-style) as infeasible for O(100) NPs.
3. Each ±1σ per-event weight is itself a density ratio → a separate cross-entropy classifier; ~O(2000) NNs total
   (2 × NP × process) (5.1.4).
4. MC-stat uncertainty = NN-ensemble spread, propagated as one coherent NP α_stat (5.2).
5. JAX + iminuit profile fit; exact Hessian; interference-aware (implicit-function-theorem) impact, not finite
   difference (5.3).
This IS the "R2 = factorized, fixed-order, 2-classifiers-per-NP" approach from Lesson 3.

## Baseline for di-Higgs? Yes — closest precedent
Same problem class ("coupling morphs an interference shape"). Map POI coefficients
{μ−√μ, √μ, 1−√μ} → κ_λ morphing basis {κ_λ², κ_λ, 1} (gg→HH = κ_λ·triangle + box). Reuse reference trick,
factorized weights, MC-stat NP, JAX fit, and especially the interference-aware impact.

## Why di-Higgs is harder (the delta)
(i) κ_λ–systematic shape degeneracy is acute (JES/τES deform m_HH; 4ℓ mostly dodges this — leptons clean);
(ii) far more/larger experimental systematics (b-jet, τ, MET, fakes); (iii) data-driven fakes have no natural
per-event density for SBI; (iv) ~10³× lower signal stats → morphing under-determined; (v) κ_λ-dependent theory
uncertainties entangle systematic with POI. → motivates upgrading the morphing (R4 expressive + R3 honest) on top
of the R2 skeleton.

## Next-lesson candidate (offered)
A concrete "port plan": write the κ_λ mixture model + factorized morphing + NP list against her FAIR Universe /
Delphes pipeline, flag which morphing to upgrade first. Would be the bridge from teaching → implementation.
