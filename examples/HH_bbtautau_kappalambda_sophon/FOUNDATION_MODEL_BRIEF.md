# Foundation models for κ_λ NSBI in HH→bbττ — delegation brief

**Purpose.** Self-contained handoff for a separate session that will pursue (or rule out) the
*foundation-model + fine-tuning* direction for the κ_λ measurement. Everything needed to start cold is here.
**Date:** 2026-06-23. **Repo/branch:** `nsbi-lhc-toolkit` @ `claude/omnilearn-nsbi`. **Code home:**
`examples/HH_bbtautau_kappalambda_sophon/`.

> Sourcing note: the SOTA landscape below comes from a 2024–2026 literature sweep (4 readers + synthesis).
> Treat arXiv ids and especially *public-weight availability* as **verify-before-relying** — flagged inline.

---

## 1. The question, in one paragraph

The κ_λ NSBI downstream is **per-event likelihood-ratio estimation** `r(x;κ_λ)=p(x|κ_λ)/p_ref(x)` over
**event-level** objects (b-jets + hadronic-τ + MET + SVfit `m_ττ`). The κ_λ signal lives in the **m_HH lineshape and
the triangle–box interference** — i.e. correlations *across* objects, with a non-monotone dependence on κ_λ (the
degeneracy). The question: does pretraining a backbone and fine-tuning beat training the ratio estimator from
scratch?

## 2. The anchor result (why this is being re-examined, not assumed)

Fine-tuning **Sophon-AK4** (jet foundation model, JetClass-II 188-class flavour/substructure pretext, ParT backbone)
did **NOT** help κ_λ=0-vs-5 separation: **scratch ≥ fine-tune > frozen**. A cheap high-level **kinematic-ceiling
BDT/MLP** matched or beat the constituent nets. Diagnosis (see §5): the pretraining **horizon** (single jet) and
**objective** (flavour classification) are *both* wrong for an event-level continuous-ratio problem.

Reproduce/extend with the existing scripts:
- `scripts/run_kappa_diagnostic.py` — the κ_λ=0-vs-5 frozen/finetune/scratch diagnostic (the experiment that produced the null).
- `scripts/run_kinematic_baseline.py` — the high-level-feature BDT/MLP "kinematic ceiling" (the bar to beat).
- `scripts/eval_to_ratios.py` — convert a trained classifier into the per-event ratio `r(x)` (the NSBI downstream).
- `scripts/run_systematics_robustness.py` — the systematics axis (relevant to RS3L-style SSL, §4).
- `scripts/delphes_to_clouds.py` — object-cloud builder (input format for any backbone).

## 3. SOTA foundation-model landscape (assessed for κ_λ)

| Model | Family | Pretext / inputs | Weights | κ_λ relevance | One-liner |
|---|---|---|---|---|---|
| **EveNet** (arXiv:2601.17126) | **event-level** (SSL + physics-supervised) | masked-particle + MET/neutrino reconstruction + multi-task; **inputs = full object cloud incl. MET** | yes* (heads/scope unconfirmed) | **HIGH** | the only released event-level multi-object+MET FM — prime candidate to test |
| **Bumblebee** (arXiv:2412.07867) | event-level (BERT masked-particle) | mask & predict particles; high-level objects + MET; permutation-invariant | unclear | MEDIUM | architecturally apt; coupling-shape untested; no confirmed weights |
| **PECM** (arXiv:2412.10665) | event-level (supervised) | 12-process event classification (120M events) | no | MED-LOW | between-process ≠ within-process coupling-shape; partial mismatch |
| **OmniLearn / OmniLearned** (2404.16091 / **2510.24066**) | jet multi-task+generative | class + next-token + contrastive on jets; >1B jets; advertised LR mode | yes | LOW | billion-jet but jet-horizon; LR mode unproven on parameter tasks |
| **Sophon-AK4** (arXiv:2405.12972) | jet-discriminative | 188-class JetClass-II flavour; ParT | yes (HF `jet-universe/sophon`) | LOW (falsified in-project) | the anchor null |
| **ParT** (arXiv:2202.03772) | jet-discriminative | 10-class JetClass tagging | yes | LOW | same mismatch as Sophon + constituent cost |
| **OmniJet-α** (2403.05618; cont. 2512.04149) | jet generative | autoregressive on VQ-VAE-tokenized jet constituents | yes (cont. unclear) | LOW | tokenization discards continuous kinematics; jet-only |
| **HEP-JEPA / JetParticle-JEPA** (2502.03933 / 2606.14813) | SSL (JEPA, jet) | predict latent of masked jet constituents | yes / unclear | LOW | few-shot tagging gains; **event-level JEPA does not yet exist** |
| **RS3L** (arXiv:2403.07066) | SSL-contrastive (jet, physics-aug) | invariance to re-simulated shower/hadronization/detector | yes | **MEDIUM (systematics only)** | shower/sim robustness, not separation |
| **RINO** (arXiv:2509.07486) | SSL (self-distillation, jet) | RG/scale-invariant embeddings; can pretrain on real data | unclear | LOW-MED | sim↔real domain-shift; coupling-insensitive by construction |
| **MPM** (arXiv:2401.13537) | SSL (masked-token, set) | masked-particle identity over a set; **set protocol allows event-level** | yes | MEDIUM (conditional) | event-capable in principle; pretext not shape-aware |
| **L-GATr** (**2405.14806** + 2411.00446) | **equivariant arch (NOT an FM)** | none — Lorentz-equivariant; amplitude regression, tagging, generation | yes (`heidelberg-hepml/lorentz-gatr`) | **HIGH as scratch baseline** | amplitude regression ≈ closest analog to interference-aware shape learning |
| **PELICAN** (arXiv:2211.00454) | equivariant arch (not FM) | none — perm-equivariant + Lorentz; 4-momentum regression | yes | MED-HIGH (scratch) | low-complexity ⇒ NSBI-cost-friendly |
| **LorentzNet** (arXiv:2201.08187) | equivariant arch (not FM) | none — Lorentz-equivariant message passing | yes | MEDIUM (scratch) | sample-efficient; costlier than FC |
| **Kinematic-ceiling MLP/BDT** (incumbent) | task-specific, no pretrain | LR-trick ratio on high-level event features | n/a | **HIGH — the bar to beat** | cheap, already near-optimal; matched constituent nets in-project |

\* EveNet weight/checkpoint scope is **LOW-confidence** — verify first (open question #1).

## 4. Three buckets

**Likely helps (must be measured, not assumed):**
- **Event-level pretraining on the same multi-object final state** (b-jets+τ_h+MET). Only **EveNet** is released in-class; Bumblebee/PECM are conceptually in-class. Alignment is on *input structure & event correlations*, not coupling — "plausibly helps."
- **Equivariant architectures from scratch** (L-GATr, PELICAN, LorentzNet) — raise the kinematic ceiling via a correct 4-vector inductive bias; L-GATr's amplitude-regression track record is the closest analog to interference-sensitive learning. (Baselines, *not* the FM paradigm.)
- **Physics-augmented / domain-invariant SSL on the SYSTEMATICS axis only** (RS3L, RINO, MPM) — shower/hadronization & sim↔real robustness for the κ_λ *uncertainty budget*, not for kl0-vs-kl5 power. MEDIUM, scope-limited.
- **Data-efficiency in stat-limited regions** (rare interference-dominated phase space) — every event-level FM reports low-stat gains; a narrower, testable claim than global separation.

**Likely won't help (and why):**
- **Any single-jet-horizon FM for the core separation** (Sophon, ParT, OmniJet-α, OmniLearn/ed, HEP-JEPA, JetCLR): **pretext-task mismatch** — jet flavour/substructure vs event-level m_HH/interference. The in-project Sophon null should replicate across this whole family.
- **Scale as a fix** — OmniLearned's billion-jet scale + advertised LR mode still yields only single-jet results. Don't credit "tops a flavour benchmark" or "has an LR mode."
- **Tokenized/generative single-jet objectives** for fine shape inference — VQ-VAE tokenization discards the continuous kinematics m_HH needs.
- **Frozen backbones** of any kind — frozen was empirically the *worst* setting.
- **Constituent/transformer FMs on COST grounds** — ~10–100× FC-net cost × (systematics × processes × ensemble); even a marginal gain can be net-negative (see §8).

**Untested but worth trying (ranked):**
1. **EveNet fine-tune vs scratch on the real κ_λ ratio task** — the single most informative experiment (§7).
2. **Coupling-aware pretext** — pretrain/continue-pretrain with parameterized-classifier or reweighting targets across a κ_λ grid (teach the backbone interference/m_HH during pretraining). Highest upside, highest effort, untested in literature.
3. **Event-level JEPA** — multi-object context → predict m_HH/latent event kinematics. Does not exist; speculative prototype.
4. **L-GATr/PELICAN from scratch as the new kinematic ceiling** — if an equivariant net beats the MLP/BDT, FMs may be unnecessary.
5. **RS3L/RINO/MPM on the event object set, evaluated only on systematics robustness.**
6. **In-domain event-level SSL** (Bumblebee-style masked-particle on *this project's own* unlabeled events) then fine-tune — isolates "event-level" from "pretrained-elsewhere."

## 5. Core reasoning (the diagnosis the whole brief turns on)

Two **independent** mismatches explain the Sophon null and predict what can escape it:

1. **Horizon / pretext mismatch.** κ_λ is event-level (m_HH lineshape, triangle–box interference, jet→Higgs pairing, MET-constrained phase space). Jet-level FMs are trained to be sensitive to *within-jet* features that are nearly irrelevant here, and blind to the *cross-object* kinematics that carry the coupling. Fine-tuning must then learn the right physics while fighting a wrongly-tilted initialization → scratch ≥ fine-tune, frozen worst.
2. **Downstream-objective mismatch.** The task is not classification between fixed classes but estimation of a **continuous, interference-driven (non-monotone)** per-event ratio. Pretexts built on discrete identity/flavour (or even process labels, PECM) encode category separability, not a smooth coupling response.

**Reconciliation:** the failure was *not* "pretraining is useless" — it was that the pretraining **horizon (jet)** and **objective (classification)** were both wrong. Falsifiable prediction: the only FMs with a chance fix at least the **horizon** (EveNet/Bumblebee/PECM) and ideally the **objective** (coupling/shape-aware pretext, which essentially no released model has). This also explains why the cheap kinematic ceiling competes (right horizon, right objective, nothing to un-learn), and why equivariant archs help by a *different* mechanism (correct inductive bias, orthogonal to transfer).

## 6. Recommendations

1. **Keep the kinematic-ceiling MLP/BDT as the fixed decision bar.** No FM adoption unless it beats this on **both** separation (kl0-vs-5 *and* the degeneracy-breaking points) **and** value-per-compute.
2. **Test exactly one FM next — EveNet** (the cleanest "fix the horizon" test).
3. **Run the same three-way contrast** that produced the original finding — frozen vs fine-tune vs scratch — for commensurability.
4. **Add an equivariant-from-scratch track** (L-GATr or PELICAN) in parallel; it may raise the ceiling with no FM at all.
5. **Separate separation from systematics** — evaluate RS3L/RINO/MPM only on shower/sim↔real robustness.
6. **If event-level transfer also nulls, do not iterate on more released FMs** — pivot to a coupling-aware pretext (the only direction addressing both mismatches).
7. **Fix citations before any writeup:** L-GATr = arXiv:2405.14806 (+2411.00446); OmniLearned = Oct 2025 (arXiv:2510.24066).
8. **Upstream dependency:** settle the NSBI MC-stat/estimator critique (Shyamsundar arXiv:2505.19156) before quoting any FM-vs-scratch κ_λ *significance* — see the challenges note §5 and `reference_nsbi_systematics_risk`.

## 7. The first experiment (do this first)

**EveNet three-way head-to-head on the real κ_λ ratio task.** Object cloud = b-jets + τ_h + MET + SVfit `m_ττ`.
Train the *same* NSBI ratio head `r(x;κ_λ)` in three configs, everything else fixed (data, head, ensemble, budget):
(a) **frozen** EveNet + head, (b) **fine-tuned** EveNet + head, (c) **from-scratch** identical-capacity net.
- **Primary metric:** kl0-vs-5 separation (AUC / cross-entropy of the learned ratio) **and** a degeneracy-sensitive metric near the κ_λ≈2.45 σ-minimum.
- **Secondary:** wall-clock + params (value-per-compute).
- **Decision rule (mirrors the Sophon test):** if fine-tune does **not** exceed scratch (expected: scratch ≥ finetune > frozen) → current FMs, even event-level, don't help → pivot to coupling-aware pretext + equivariant-from-scratch. If fine-tune > scratch beyond the compute penalty → EveNet becomes the backbone to develop.
- **Prerequisite:** verify EveNet ships usable weights/heads and that its inputs map onto the bbττ object set (τ_h + SVfit + MET). If not → fall back to in-domain Bumblebee-style SSL (bucket item 6).

## 8. Compute caveats (a first-class selection criterion)

- NSBI needs **many** nets: ≈ (systematic variations) × (processes) × (ensemble members) × (κ_λ grid). Per-net cost is multiplied by all of these.
- Constituent/transformer FMs are **~10–100× FC-net cost**; in-project they gave at best parity with the cheap ceiling. The compute case for any heavy backbone must clear that bar.
- **Frozen** is the cheap FM mode but was the *worst* for separation; the useful mode (**fine-tune**) re-incurs near-full per-net training cost — eroding the "pretrain once, reuse cheaply" economics *for NSBI specifically*.
- EveNet/Bumblebee/OmniLearned are large; fine-tuning one **per (systematic × process × ensemble) cell** may be infeasible on the Perlmutter allocation. Budget per-cell cost *before* the experiment; consider adapters/LoRA-style partial fine-tuning (and whether that's enough to learn event-level kinematics the backbone never saw).
- A conditional surrogate (one net conditioned on all NPs) is the cost lever on the NSBI side (see challenges note §5) — it interacts with the FM choice.

## 9. Open questions for the delegated session

1. Does **EveNet** ship usable public weights/heads, and do its inputs map onto the bbττ object set? (LOW-confidence — verify first.)
2. Is the Sophon null caused by **horizon**, **objective**, or both? The EveNet test (event-level but not coupling-aware) **disentangles** them: EveNet null ⇒ objective; EveNet win ⇒ horizon.
3. Can any FM beat the cheap ceiling on **value-per-compute** once net multiplicity is counted — even if it wins on raw AUC? (The real adoption criterion.)
4. Does a **coupling-aware pretext** transfer better than flavour pretext? (Highest-upside untested idea.)
5. Do SSL robustness gains (RS3L/RINO/MPM) translate into a measurable reduction of the κ_λ **systematic** uncertainty?
6. Is the upstream NSBI MC-stat concern (arXiv:2505.19156) resolved for this analysis? If not, FM-vs-scratch *significance* comparisons are confounded.
7. Would in-domain event-level SSL beat an out-of-domain FM (isolating "event-level" from "pretrained-elsewhere")?

## 10. Pointers

- **Challenges/risk context:** `docs/teaching/notes/2026-06-20-klambda-nsbi-vs-offshell-challenges.{tex,pdf}` (§2 physics/POI; §5 systematics maturity, width, compute).
- **Prelim physics result:** Ghosh–Klute–Pan note (`NSBI-pheno/dihiggs_bbtautau`) — full-sim, no systematics in fit, 2.75σ→7.1σ.
- **Memories:** `project_sophon_hh_nsbi`, `project_sophon_kl_diagnostic_findings`, `reference_nsbi_systematics_risk`, `project_bbtautau_nsbi_prelim_result`.
- **Scripts to reuse:** see §2.
