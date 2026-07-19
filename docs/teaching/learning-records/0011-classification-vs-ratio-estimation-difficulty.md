# 0011 — Learner catch: classification difficulty ≠ ratio-estimation difficulty

**Date:** 2026-07-13
**Trigger:** Reviewing the Lesson-1 Q&A note's "why factor into two hops" reason 1, the learner challenged:
*"for training a classifier, the more different the two distributions are, the easier the training — isn't the
note's easy/hard swapped? Low-stats + similar Family-2 ratios are much harder to learn?"*
**Doc fixed:** `notes/2026-07-11-lesson1-qna-rates-profiling-ratios.tex` §1.2 reason 1 (rewritten; pointer to
this record added inline).

## The catch was half right — which exposed a real imprecision

The note used one word ("easy/hard") for two different axes that run in **opposite directions**:

- **Classification difficulty** (the learner's intuition, correct): separation is fuel. Different = easy
  (high AUC, strong gradients); nearly identical = hard (AUC ≈ 0.5, weak gradients).
- **Ratio-estimation difficulty** (the task the fit actually cares about): the fit consumes the *value* of
  r = s/(1−s), not the ranking. Difficulty is set by the **dynamic range / boundedness of the target** and
  **support overlap** — and it inverts:
  - **Family-1** (component vs reference — very different): target spans orders of magnitude, unbounded where
    the reference support dies; sigmoid saturates exactly where the value matters; calibration errors multiply
    the whole density. High AUC ≠ good ratio. This is the *dangerous* regime → gets the ample nominal MC,
    trained once. (Cf. the prelim's delicate reference choice: pooled ref rejected for zero learning signal —
    accuracy 0.50; too-distant refs diverge. Family-1 lives in a sweet spot Family-2 never needs.)
  - **Family-2** (varied vs nominal, same process): target = bounded perturbative correction around w ≡ 1.
    Failure is *graceful* (under-trained → w = 1: a few-% absolute error that understates a systematic, never
    corrupts the density). Required accuracy is relative-to-the-deviation. Feasible-by-construction from small
    samples.

## The valid half of the objection = the C6 burden, named precisely

"Similar + low stats is hard" is true — but the hardness is **variance, not feasibility**: on the small ±1σ
samples the training *noise* on the few-% deviation rivals the deviation itself, and nets can fit the sample's
own fluctuations. That is a quantifiable statistical problem on a bounded quantity (ensembles, C2ST
signal-existence checks, the wifi band) — exactly C6 — not an intractable learning problem. The one-hop
alternative (learn p_s(x|α)/P_ref directly per variation sample) would suffer *both* hardnesses at once with
the catastrophic failure mode; the two-hop factorization exists precisely to route the dangerous learning to
the ample sample and leave the starved samples only benign-but-noisy corrections.

## Corrected formulation (now in the note)

**Family-1: hard in the dangerous sense (unbounded target, tails/calibration) → needs big statistics.
Family-2: easy in the feasibility sense (bounded, perturbative, graceful) but hard in the variance sense
(noise ~ signal on small samples — the C6 burden, insured by wifi).**

## Learner state

The learner is now reliably catching compressed/ambiguous claims in the teaching material (cf. records 0002,
0005, 0007) — register is right, but one-word difficulty labels ("easy", "hard", "exposed") should always be
qualified with *in what sense*. Apply this rule to future lessons.
