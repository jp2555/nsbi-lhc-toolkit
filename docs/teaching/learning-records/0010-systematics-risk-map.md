# 0010 — The systematics risk map (open vs numerical) + the C4 deep-dive

**Date:** 2026-06-23
**Lesson:** [0008-systematics-risk-map-open-vs-numerical.html](../lessons/0008-systematics-risk-map-open-vs-numerical.html)
**Source:** the §5 "Literature maturity & risk" subsection of the κ_λ challenges note (grounded by a 2024–26
literature sweep + maturity-skeptic workflow; Shyamsundar critique verified via WebFetch). See
[[reference_nsbi_systematics_risk]].

## The framing the lesson teaches
Two kinds of risk in a to-do list: **open-conceptual** (no method in the literature → research risk) vs
**numerical-quantification** (method exists → engineering risk). The triage verdict for the 8 systematics
challenges: **none is cleanly numerical** (bbττ exceeds where any per-event method has been demonstrated), but
**only C4 is fully open**; the rest are **hybrid** (method exists in an easier setting → each pilot is a
*method-selection gate*, not just a number).

## The fact that reframes it
Largest published learned-NP-morphing demos: ~3 calibration NPs on simulated data (GOLLUM) + ~50 NPs on clean 4ℓ
(ATLAS). Nothing exercises ~40 *shape-degenerate* NPs deforming a *reconstructed* m_HH across 3 channels with
data-driven backgrounds.

## C1 vs C4 (the conceptual crux, captured with two diagrams)
- C1 (degeneracy) = a FIT problem: weights stay POI-independent; κ_λ and JES just push m_HH the same way →
  off-diagonal Fisher → wider interval. Fixed by dimensionality. *Sensitivity* question.
- C4 (POI-coupled theory) = a MODEL problem: the theory weight's shape changes with κ_λ → must be w_θ(x;κ_λ,α_θ),
  which breaks the factorized ansatz. *Modelability* question; features don't help. Distinct from C5 (NP×NP, which
  GOLLUM solves on a toy).
- Why off-shell escapes: THU sits on the signal whose triangle/box/interference carry *different* bands that κ_λ
  re-mixes (fractional band diverges at the κ_λ≈2.45 σ-minimum); off-shell's leading theory NP is on the
  μ-independent background, σ(μ) monotonic.
- Way out (→ "open" not "impossible"): per-component POI-independent theory weights blended by the κ_λ coefficients;
  missing the component-differential bands + an NSBI closure.

## The two coverage threats
C6 (Shyamsundar 2505.19156 — published claim the ATLAS double-bootstrap MC-stat estimator is *incorrect*) and C2
(NP-scale may force a conditional surrogate / GP morphing over ~thousands of classifiers). Both attack a
*tighter-but-must-cover* interval — settle before quoting a κ_λ number.

## Lesson-design note
Lesson 8 reuses the two chat-built SVGs (C1-vs-C4; σ(κ_λ) schematic) folded inline — the "build a widget live, then
fold it into the durable lesson" pattern again. The teaching curriculum now spans Lessons 1–8 (systematics field-map
→ jargon → methods → systematics-for-κ_λ → off-shell recipe → DD backgrounds → DD mechanics+bridge → risk map).

## Next-lesson / next-action candidates (offered, not yet chosen)
The two gates as concrete pilots: (a) the C4 POI×NP theory-morphing toy; (b) the C6 MC-stat estimator comparison
(ATLAS vs Shyamsundar). Also still open: a skills/build-it lesson (neural fake-factor; calibrate-a-ratio + C2ST).
