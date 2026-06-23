# 0009 — Embedding, and the classifier-to-reference bridge (DD backgrounds → NSBI ratio)

**Date:** 2026-06-22
**Lesson:** [0007-dd-background-mechanics-and-the-nsbi-bridge.html](../lessons/0007-dd-background-mechanics-and-the-nsbi-bridge.html)
**Reference updated:** [data-driven-backgrounds-cheatsheet.html](../reference/data-driven-backgrounds-cheatsheet.html) (+bridge, +embedding)
**Grounding:** Q&A walk-throughs (ABCD, fake-factor) + verified τ-embedding via arXiv:1903.01216 (single Explore
agent) + CARL arXiv:1506.02169 + neural fake-factor arXiv:2511.06972.

## What this lesson consolidates
Lesson 7 is the mechanics deep-dive companion to Lesson 6. It captures, durably, the three diagrams built live in
chat (ABCD regions, fake-factor SR/AR/DR, the classifier bridge) plus a new embedding section.

## The spine (the durable takeaway)
Three data-driven methods, three output shapes, **none an evaluable p(x)**:
- ABCD (QCD) → yield + histogram; assumption = the two region variables factorize for the background.
- fake-factor (jet→τ_h) → per-event weights on a real loose-ID sample; assumption = fake composition stable DR→SR.
- embedding (Z→ττ) → a sample of hybrid events; data for jets/PU/UE, simulation only for the τ decay.

## Embedding — the new piece (+ the bbττ caveat that matters)
Real Z→μμ event → remove muons → insert simulated τ with the muons' kinematics → decay → merge.
**Key bbττ caveat (verified):** the b-jets come from the original μμ event, so embedding does NOT model genuine
Z→ττ+heavy-flavor (Z+bb) — CMS uses MC + data corrections for Z+HF instead. Also semi-data-driven: μ/τ-ID + trigger
SFs + QED radiative corrections still applied. Output is a sample, not a density.

## The bridge (the keystone)
Density-ratio-by-classification (CARL): train a classifier (component sample vs reference); calibrated optimal score
s(x) → r(x) = s(x)/(1−s(x)). You never write a density — you only need *samples*, which all three methods provide
(ABCDnn→flow samples; fake-factor→weighted sample; embedding→sample). Guardrails: calibration (isotonic),
weighted/sign-aware BCE, one common P_ref, C2ST closure + support checks. This homogenizes every component into the
same currency (a ratio to P_ref), so DD backgrounds become "just more components."

## The honest framing (repeat it)
The bridge is the *easy middle*. The risk is upstream (a faithful **high-dimensional** DD sample — the ABCDnn-in-10-D
problem) and downstream (per-event systematic morphing of r(x)). The prelim result ([[project_bbtautau_nsbi_prelim_result]])
already runs the bridge for MC components (calibrated ratio classifiers, isotonic, label-flip negatives, single
distant κ_λ=5 reference) — extending it to DD backgrounds is "same trick, harder sample."

## Learner state
Mission now fully covers backgrounds as well as systematics. Lessons 6–7 + the cheat-sheet form a self-contained
"data-driven backgrounds for NSBI" unit. The teaching uses inline SVG diagrams now (built first as chat widgets,
then folded into the lesson for durability) — a pattern worth repeating. Remaining offered threads: a build-it lesson
on the neural fake-factor; a hands-on "calibrate a classifier ratio + C2ST closure" exercise (skills, not just knowledge).
