# 0008 — Data-driven backgrounds in bbττ and the ABCDnn bridge to NSBI

**Date:** 2026-06-22
**Lesson:** [0006-data-driven-backgrounds-and-abcdnn.html](../lessons/0006-data-driven-backgrounds-and-abcdnn.html)
**Reference made:** [data-driven-backgrounds-cheatsheet.html](../reference/data-driven-backgrounds-cheatsheet.html)
**Grounding:** 4-reader research workflow over public sources (arXiv:2008.03636, 2206.09401, 2204.12957,
1903.01216, 2502.04156, 2511.06972, 1804.00779) + local challenges note §4 + S. An four-top thesis.

## The organising idea
"Data-driven" is a slider, not a flag: per background ask **which of {shape, normalization} do I trust MC for?**
- QCD multijet → ABCD (data shape+norm); jet→τ_h fakes → fake-factor (data, per-event weight);
  DY Z→ττ → τ-embedding (data except the simulated τ decay); tt̄ → MC shape + data normalization;
  single-H/diboson → pure MC.

## The two workhorses (one assumption each)
- **ABCD:** N_D = N_B·N_C/N_A. Assumes the two region variables *factorize* for the background. Output = yield +
  coarse binned shape.
- **Fake-factor:** FF = N_tight/(N_loose − N_genuine) in a DR, applied as a per-event weight to the AR. Assumes the
  fake *composition* is stable DR→SR. Output = reweighted sample.

## The ladder
ABCD (assumes factorization → yield) → ExtendedABCD (measures the breaking via extra sidebands → better yield;
four-top closure ~18–19%→~1–7%, illustrative) → **ABCDnn** (NAF learns p_bg(x|control) → full conditional density).

## The crux (the reason this lesson exists)
NSBI needs a per-event density/ratio for **every** component. ABCD gives a histogram, fake-factor gives weights,
embedding gives a sample — **none is an evaluable p(x)**. This is the structural gap the ATLAS off-shell measurement
never faced (its background was one MC density). **ABCDnn bridges it** (it *is* a learned density → classifier vs
reference → r(x)); **neural fake-factor** (2511.06972) is the analogue for fakes (density-ratio FF(x)).

## The honest gap + the pragmatic compromise (both flagged as such in the lesson)
- Gap: ABCDnn-to-date targets a *low-dim* object (BDT+H_T for a *binned* fit); NSBI wants the full 10+-dim per-event
  density → high-dim flow + more data + punishing closure + per-event systematic morphing. No published bbττ NSBI
  does this yet.
- Compromise: ABCDnn/neural-FF model only the data-driven *shape of the 1-D NSBI discriminant*; bin QCD/fakes as an
  auxiliary term; per-event morph only the MC components. **Design proposal, not published.**

## Hedges baked into the lesson (do not let these harden into "facts")
Exact ABCD region cuts, the DY embedding-vs-18-SF choice, tt̄ normalization specifics, and any κ_λ limit number are
analysis-note-specific — the lesson explicitly says "verify, don't quote from memory." ExtendedABCD closure numbers
are illustrative of the four-top analysis only.

## Links to the measurement
This completes the background half of the [κ_λ challenges note](../notes/2026-06-20-klambda-nsbi-vs-offshell-challenges.tex)
§4. It is the teaching backing for studies #1 (ABCDnn flow), #2 (ExtendedABCD), #4 (binned-vs-unbinned cost),
#12 (in-situ CRs). The prelim result handled fakes via per-event mis-ID *weights* on MC — i.e. it already gave fakes
a per-event density by a different route than ABCDnn (see [[project_bbtautau_nsbi_prelim_result]]).

## Learner state / next-lesson candidates
Mission has widened from systematics → systematics **+ backgrounds** (both the "new bbττ difficulty"). Natural next
lessons: (a) "Converting a flow/estimate into a per-event ratio" (the classifier-to-reference trick, hands-on);
(b) "τ-embedding, properly" (how the data-driven DY actually works); (c) a build-it lesson on the neural fake-factor.
