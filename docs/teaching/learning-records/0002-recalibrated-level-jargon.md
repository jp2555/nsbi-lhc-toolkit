# 0002 — Recalibrated level; added jargon decoder

**Date:** 2026-06-19
**Lesson:** [0002-decoding-the-jargon.html](../lessons/0002-decoding-the-jargon.html)

## What happened
User reported they couldn't follow a large part of Lesson 1 — too much statistics/HistFactory
jargon introduced at once (nuisance parameter, morphing, profile likelihood, the decomposition
equation, factorizable, etc.). My ZPD estimate was wrong: strong on ML/physics, but NOT assuming
fluency in profile-likelihood/HistFactory *statistics* vocabulary.

## Correction
Built Lesson 2 as a plain-language decoder: one running example (τ energy scale shifting H→ττ
mass spectra → measuring μ), each term introduced with intuition first, symbols minimized, and the
Lesson-1 equation decoded symbol-by-symbol in English at the end. Updated NOTES.md with the revised
register.

## Key teaching insight
For this learner, the load-bearing prerequisites are the *statistics* terms, not the ML ones:
likelihood/likelihood-ratio, nuisance parameter (knob α) + constraint, anchor points, morphing
Δ(x|α), profiling + t_μ/CI. Once those land, the method comparison (Lesson 1) is readable.

## Next
Wait for feedback on which terms still feel fuzzy (asked explicitly in Lesson 2). Offer a *visual*
(before/after τ-energy-scale distribution) if intuition needs anchoring. Hold the Lesson-2 fork
(reproduce-R4 / profiling-coverage / GP) until the vocabulary is solid.
