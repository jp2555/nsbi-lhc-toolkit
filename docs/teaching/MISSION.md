# Mission

## Who & why
Jingjing Pan — HEP researcher constraining the Higgs trilinear self-coupling **κ_λ** via
non-resonant **HH→bbττ** (resolved regime) with **neural simulation-based inference (NSBI)**.
The measurement is **statistics- and systematics-limited**, and she is exploring whether a
**foundation-model + fine-tuning** paradigm (and/or systematics-robust representations like
RS3L) can help.

## What she wants to learn (this thread)
A clear, well-grounded **map of how systematic uncertainties are handled inside NSBI** — the
methods, how they differ, and their trade-offs — so she can:
1. **choose/design the systematics treatment** for her κ_λ NSBI fit;
2. **situate ML-for-robustness** (RS3L-style representation) relative to *modeling* systematics
   in the likelihood;
3. read the current literature critically (ATLAS NSBI, HistFactory-v2/GP, the "refinable"
   unbinned-ML-systematics work, uncertainty-aware learning).

## Success criteria
- Can state the **single core problem** every NSBI-systematics method is solving
  (the ν-dependence of the per-event likelihood ratio is intractable → needs a surrogate).
- Can place each of the 5 works on a **2-axis map** (model-in-likelihood vs robust-representation;
  and *how* the ν-interpolation is built).
- Can argue which approach(es) fit her FAIR Universe testbed and her κ_λ analysis.

## Constraints / context
- FAIR Universe H→ττ is her current **systematics methodology testbed** (real TES/JES + backgrounds).
  Note: the "refinable" unbinned-ML-systematics work (Schöfbeck et al.) uses *the same benchmark*.
- Related project memory: NSBI-for-κ_λ, foundation-model studies, the systematics-robustness scaffold.
