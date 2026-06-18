# Foundation-model pretraining (Sophon) for NSBI density-ratio estimation — design

- **Date:** 2026-06-18
- **Branch:** `claude/omnilearn-nsbi` (fresh clone of `iris-hep/nsbi-lhc-toolkit`)
- **Status:** Design — awaiting user review before writing the implementation plan
- **Author:** drafted with Claude Code via the brainstorming workflow

---

## 1. Motivation

The H1 fully-differential ("all-particle") measurement
([H1prelim-25-031](https://www-h1.desy.de/h1/www/publications/htmlsplit/H1prelim-25-031.long.html))
showed that a pretrained foundation model (OmniLearn, integrated into OmniFold;
[H1Unfold](https://github.com/ViniciusMikuni/H1Unfold)) improves the training of
the precision classifier used for density estimation in many dimensions: **better
and faster convergence, and a large reduction in the per-task statistics needed**,
because fine-tuning a pretrained backbone for a specific process is far more
data-efficient than training from scratch.

This project brings that idea into `nsbi-lhc-toolkit` for **parameter estimation
via Neural Simulation-Based Inference (NSBI)**. The toolkit estimates per-event
density ratios `p_A(x)/p_B(x)` with neural classifiers; those ratios feed an
unbinned likelihood that is profiled to measure parameters of interest. The
hypothesis to demonstrate: **a pretrained jet-physics foundation model improves
the density-ratio classifiers** — measured first by training diagnostics, then by
a tighter physics constraint.

### Ultimate physics goal

Constrain the Higgs trilinear self-coupling **κ_λ** via **non-resonant
HH→bbττ**. The κ_λ sensitivity is largest near the di-Higgs production threshold,
**m_HH ∈ [250, 400] GeV**, i.e. the **resolved** regime: two separate small-radius
(AK4) b-jets from H→bb, plus the ττ system (τ_had τ_had / τ_had τ_lep / τ_lep
τ_lep), plus MET. The non-resonant HH signal is rate-starved, which is exactly the
low-statistics regime where foundation-model fine-tuning is expected to help most.

### Foundation model

[**sophon-ak4**](https://huggingface.co/jet-universe/sophon-ak4) — the Sophon
variant for AK4 small-radius jets (resolved). It is a Particle-Transformer (ParT)
model (6 particle-attention + 2 class-attention blocks, 64-d embedding, 8 heads,
~5.5×10⁵ params), pretrained on a JetClass-II-style Delphes dataset to a 23-class
flavor labelling (b/c/s/d/u/g/e/μ/τ_had + two-prong combinations), MIT-licensed,
checkpoint on Hugging Face. Its 64-d class-token output is a per-jet embedding that
we use as a per-AK4-jet encoder. (Lineage:
[Sophon, arXiv 2405.12972](https://arxiv.org/pdf/2405.12972);
[OmniLearned, arXiv 2510.24066](https://arxiv.org/html/2510.24066);
di-Higgs precedent [HH→4b jet-free, arXiv 2508.15048](https://arxiv.org/pdf/2508.15048).)

---

## 2. Goals and non-goals

### Goals
- **G1 (Phase 1, core):** Diagnostics showing sophon-ak4 pretraining improves the
  **event-level** HH-signal-vs-background density-ratio classifier — faster/better
  convergence and, especially, data-efficiency in the low-statistics signal regime.
- **G2 (Phase 2, eventual money figure):** A **κ_λ profile-likelihood** from the
  toolkit's NSBI fit, comparing pretrained vs from-scratch classifiers, to show the
  foundation model tightens the κ_λ constraint.
- **G3:** Land the new capability as **composable additions** that follow the
  toolkit's documented extension pattern (custom `pl.LightningModule` + reuse of
  `save_model`/`predict_with_onnx`/calibration/diagnostics), without breaking the
  existing tabular density-ratio path or the JAX fit layer.

### Non-goals (this iteration)
- Pretraining a foundation model from scratch on JetClass-II — we **fine-tune the
  released sophon-ak4 checkpoint**.
- A boosted/resonant X→HH analysis (sophon, large-R) — out of scope; we target
  resolved non-resonant HH.
- Reusing the FAIR Universe H→ττ tabular example — dropped from the critical path
  (wrong granularity: it has no constituents). It may remain as an independent
  object-level baseline only if cheap; not required.
- Full systematic-uncertainty model tuning for the κ_λ fit beyond a representative
  template.

---

## 3. Current toolkit architecture (what we build on)

Verified by reading the repo at HEAD of `main` (`fc09848`). The stack is
**PyTorch Lightning (training) → ONNX (interchange) → JAX (inference/fit)**.

- `src/nsbi_common_utils/lightning_tools/density_ratio_model.py` —
  `DensityRatioLightning(pl.LightningModule)`: `self.mlp` (Sequential) + `self.out`
  (Linear), flat 2-D input `x`, weighted BCE (or `binary_cross_entropy_with_logits`
  when `use_log_loss`), NAdam + StepLR. Signature
  `(n_hidden, n_neurons, input_dim, learning_rate, use_log_loss, activation,
  callback_factor, callback_patience)`.
- `lightning_tools/multiclass_model.py` — `MultiClassLightning`, same MLP shape,
  softmax head, weighted cross-entropy (used by preselection).
- `lightning_tools/datasets.py` — `WeightedTensorDataset(x, y, w)` → `(features,
  label, weight)`.
- `lightning_tools/callbacks.py` — `LossHistory`, `PrintEpochMetrics`.
- `training/neural_ratio_estimation.py` — `density_ratio_trainer` (`__init__(...)`,
  `.train(...)`): stratified train/holdout split, sklearn `ColumnTransformer`
  scaling, builds `DensityRatioLightning` at one instantiation site, trains via
  `pl.Trainer` (EarlyStopping, LearningRateMonitor, ModelCheckpoint), optional
  isotonic/histogram calibration, **ONNX export**, ensembles via `ensemble_index`.
  Diagnostics: `make_overfit_plots`, `make_calib_plots`, `make_reweighted_plots`
  (closure), `test_normalization`.
- `training/utils.py` — `save_model(lightning_model, input_sample, path, scaler,
  scaler_path, softmax_output)` (`torch.onnx.export`, opset 17, dynamic batch,
  `input_names=['features']`, `output_names=['output']`),
  `load_trained_model`, `predict_with_onnx`, `predict_with_model`,
  `convert_torch_to_onnx`, `convert_logLR_to_score`, `convert_score_to_ratio`.
  The default (non-softmax) export traces `model.forward` directly; only the
  `softmax_output=True` wrapper touches `.mlp`/`.out`.
- `models/sbi_parametric_model.py` — JAX, HistFactory-style (normfactors +
  NormPlusShape interpolation). `model(param_vec)` → NLL, `model_grad`. Consumes
  per-event ratio arrays loaded from `.npy` paths in the workspace.
- `inference.py` — `inference(model_nll, initial_values, list_parameters,
  num_unconstrained_params, model_grad)`: iminuit MIGRAD + profile scans.
- `workspace_builder.py`, `configuration.py` (+ `schemas/config.json`),
  `datasets.py` (uproot/coffea/awkward ROOT loading).
- Example layout: `examples/FAIR_universe_Higgs_tautau/` with `N_*.ipynb`,
  mirror `scripts/*.py`, `htcondor/*` DAGs, `config.pipeline.yaml`,
  `config_fit_nsbi.yml`. Docs in `docs/basics/*.rst`;
  `docs/basics/density_ratio_training.rst` explicitly documents the
  custom-`LightningModule` extension pattern we follow.
- Env (`pixi.toml`): torch ≥2.10, lightning ≥2.6, jax ≥0.5, onnx/onnxruntime,
  scikit-learn, uproot/coffea/awkward, iminuit, cabinetry. `image.def` = Apptainer
  recipe for HPC. Python ≥3.12.

**Key compatibility fact (Phase 1 only).** The JAX fit (`sbi_parametric_model`) is
model-agnostic: it `np.load`s pre-computed per-event **ratio** arrays (one per
sample, nominal + per-systematic up/dn) plus a per-event **Asimov weight** array,
all on one fixed shared event ordering. The ONNX model never touches the fit. A
separate **evaluation step** (`data_nn_eval.py`) runs ONNX inference → score →
`convert_score_to_ratio` (r = s/(1−s)) → ensemble aggregation → saves
`ratio_<process>.npy` on the Asimov event set. So for the **binary / signal-strength
(μ-style)** case, adopting a constituent model means re-implementing only the *eval
step* (constituent ONNX inference, same Asimov ordering as the weights); the fit is
untouched. **This does NOT hold for κ_λ:** κ_λ enters quadratically and reshapes the
signal, while the current unbinned term is linear in normfactors, so
`sbi_parametric_model` must be extended (§8). κ_λ is the one place the fit layer
changes.

---

## 4. Locked decisions (from brainstorming)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Self-pretrain → fine-tune** paradigm (here: use the released sophon-ak4 checkpoint, fine-tune per task) | Maps onto NSBI's one-backbone/many-ratios structure and the low-stat systematic/signal variations. |
| D2 | **Evidence = diagnostics primary + fit as money figure** | Diagnostics isolate the training-quality claim; the κ_λ fit is the eventual physics deliverable. |
| D3 | **Pretraining objective = supervised (sophon-ak4's 23-class flavor task)**, reused as a frozen/fine-tuned encoder | Released, abundant, directly relevant (b/τ flavor). No from-scratch pretraining. |
| D4 | **Target = κ_λ, non-resonant HH→bbττ, resolved (m_HH∈[250,400])** | The physics goal; determines AK4 (sophon-ak4) over boosted Sophon. |
| D5 | **Inputs = constituent particle-clouds**; user can produce constituents | Required for ParT/Sophon; user confirmed feasible. |
| D6 | **Architecture = hierarchical** (sophon-ak4 per-AK4-jet encoder → event-level set-transformer) | Resolved κ_λ NSBI is event-level; sophon-ak4 is per-jet. |
| D7 | **Integration = new composable modules**, existing tabular path + JAX fit untouched | Follows the documented extension pattern; low blast radius. |

---

## 5. Architecture

```
 per AK4 jet:  constituents[n_part, F_part] ──► sophon-ak4 (ParT)        ──► 64-d jet embedding
                                                (frozen OR fine-tuned)

 event:        { jet embeddings }  +  τ_had tokens  +  lepton tokens  +  MET token
                          │
                          ▼
               event set-transformer  ──►  pooled event embedding  ──►  density-ratio head
                          │
                          ▼
               ONNX export ─► eval step (ONNX score → r=s/(1-s) → ensemble agg) ─► per-event ratio .npy ─► JAX fit
               (Phase 1: fit unchanged. Phase 2 κ_λ: morphing-basis ratios + sbi_parametric_model extended.)
```

- **Per-jet (Level 1):** each selected AK4 jet's constituents → sophon-ak4 →
  64-d embedding. This is where pretraining transfer lives (b-tag/flavor priors).
  Supported modes: **frozen** (precompute/cache embeddings; fastest), or
  **fine-tuned** end-to-end.
- **Event (Level 2):** a set-transformer over typed object tokens — the AK4-jet
  embeddings plus τ_had/lepton/MET tokens carrying object four-vectors. Because the
  event transformer sees four-vectors, it can form **m_HH** and bb/ττ correlations,
  which carry the κ_λ information near threshold.
- **Head:** binary sigmoid (Phase 1; same weighted-BCE/`use_log_loss` contract as
  `DensityRatioLightning`); κ_λ-parameterized in Phase 2 (§8).
- **Export & fit:** ONNX export with a constituent-shaped input; per-event score →
  `.npy`; JAX fit unchanged in Phase 1.

### Data representation and ONNX contract
- Per event we materialize: a padded jet-constituent tensor `[n_jets_max,
  n_part_max, F_part]` with a particle mask; an object-token tensor `[n_obj_max,
  F_obj]` (τ_had, leptons, MET) with an object mask; per-jet kinematics for the jet
  tokens. `F_part` matches sophon-ak4's expected per-particle features
  (`px,py,pz,energy, Δη,Δφ, d0/dz(+err), charge, isElectron/isMuon/isPhoton/
  isChargedHadron/isNeutralHadron`).
- **MET and isolated leptons** are not native to sophon-ak4; they enter as
  **event-level object tokens with a type flag**, never as jet constituents.
- ONNX input is the (fixed-max, padded) constituent + object + mask tensors; the
  exported model returns a per-event score. A new constituent-aware
  inference/export path is required (the existing `predict_with_onnx` is tabular
  2-D). The downstream `.npy` **ratio** format is unchanged — but the **eval step**
  (the constituent analog of `data_nn_eval.py`) must produce ratios on the **same
  ordered Asimov event set as the weights array**, then score→ratio→aggregate, so
  the fit's `np.load` contract is met exactly.

### Input data: Delphes production (confirmed)

Samples are produced with the CMS HH→bbττ Delphes card
(`delphes_card_CMS_hhbbtt_v0.tcl`, Delphes 3.5.1pre09): **AK4 (R=0.4)** jets
(`JetPTMin=15` GeV for headroom under a 20 GeV analysis cut), FatJet (R=0.8) kept
as a boosted hook, `JetFlavorAssociation` filling `Jet.Flavor` (b-tag truth/WPs),
gen taus retained, MET from EFlow. The card is written "following the JetClass-II
dataset configurations" — i.e. **deliberately aligned with sophon-ak4's training
domain**, which de-risks the input-preprocessing match (R3).

- **Output = complete Delphes tree** (`Particle`, `Track`, `Tower`,
  `EFlowTrack`, `EFlowPhoton`, `EFlowNeutralHadron`, `GenJet`, `GenMissingET`,
  `Jet`, `FatJet`, `Electron`, `Photon`, `Muon`, `MissingET`, `ScalarHT`). The
  EFlow objects are the particle-flow constituents from which we build per-jet
  clouds; format is adjustable downstream.
- **New data-layer component — a Delphes→cloud converter** (example script,
  promotable to the library): cluster/associate EFlow constituents to each AK4 jet
  and emit sophon-ak4-schema per-particle features — `px,py,pz,energy`, `Δη,Δφ`
  vs jet axis, `d0/dz(+err)` (from `EFlowTrack`), `charge` and
  `isElectron/isMuon/isPhoton/isChargedHadron/isNeutralHadron` (from track PID +
  EFlow type) — plus event-object tokens from `Electron/Muon/MissingET` and the
  τ_had-candidate jets. Built with the toolkit's existing uproot/awkward stack.
- **full vs lite samples:** the card offers a `full` TreeWriter (with
  constituents; signal + shape-critical backgrounds tt→2ℓ2ν, DY, single-H) and a
  `lite` TreeWriter (no constituents; fake-source bulk — QCD HT slices, W+jets).
  The hierarchical constituent model applies to **full** classes; **lite/fake**
  classes must enter the likelihood via jet-level features or data-driven fake
  weights, not the constituent encoder. This split must be explicit in the data
  contract and the workspace.

---

## 6. New / changed components (isolation-first)

Each unit lists *what it does / how it's used / what it depends on*.

### 6.1 `lightning_tools/particle_cloud_dataset.py` (new)
- **What:** ragged→padded dataset yielding `(jet_constituents, part_mask,
  object_tokens, obj_mask, y, w)`; the constituent analog of
  `WeightedTensorDataset`.
- **How:** constructed from awkward arrays / DataFrames produced by the data layer;
  consumed by a `DataLoader` in the trainer.
- **Depends on:** torch, awkward/numpy. No toolkit-internal coupling.

### 6.2 `lightning_tools/sophon_ak4_encoder.py` (new)
- **What:** `SophonAK4Encoder(nn.Module)` — loads the sophon-ak4 ParT checkpoint
  from Hugging Face, exposes a 64-d per-jet embedding; `freeze` flag; optional
  projection. Provides the random-init ParT-of-same-shape ("from scratch") for the
  control.
- **How:** instantiated by the event model; called per jet (batched over jets).
- **Depends on:** ParT model definition (via `weaver-core`/`particle_transformer`
  or vendored), `huggingface_hub` for the checkpoint.

### 6.3 `lightning_tools/event_transformer.py` (new)
- **What:** `EventSetTransformer(nn.Module)` — masked multi-head self-attention over
  {jet embeddings, τ/lepton/MET tokens} → pooled event embedding; a binary
  density-ratio head.
- **Depends on:** torch only.

### 6.4 `lightning_tools/hh_density_ratio_model.py` (new)
- **What:** `HHDensityRatioLightning(pl.LightningModule)` composing 6.2 + 6.3;
  mirrors `DensityRatioLightning`'s training/validation/optimizer contract
  (weighted BCE, `use_log_loss`, NAdam) so the trainer and ONNX/calibration
  utilities apply. Exposes a `_get_model_for_export()` hook for constituent ONNX.
- **Depends on:** 6.2, 6.3, pytorch_lightning.

### 6.5 `training/particle_ratio_estimation.py` (new)
- **What:** `particle_density_ratio_trainer` paralleling `density_ratio_trainer`:
  split, scaling (per-feature, applied to constituent/object features),
  `pl.Trainer` loop, ensembles, calibration, the four diagnostics, constituent ONNX
  export. Reuses `lightning_tools/callbacks.py` and `training/utils.py` helpers.
- **Depends on:** 6.1, 6.4, training/utils.

### 6.6 `training/utils.py` (extend, backward-compatible)
- **What:** add `save_model_constituents(...)` and
  `predict_with_onnx_constituents(...)` for padded-cloud + mask I/O. Existing
  tabular functions untouched.

### 6.7 `models/sbi_parametric_model.py` (extend — Phase 2)
- **What:** add a **κ_λ-parameterized signal**: signal rate and per-event ratio are
  quadratic in κ_λ via a morphing basis (§8). Backgrounds remain
  κ_λ-independent. Existing μ + NormPlusShape behavior preserved.

### 6.8 Example: `examples/HH_bbtautau_kappalambda_sophon/` (new)
- `config_train.yml`, `config_fit_nsbi.yml` — data paths, constituent/object
  schema, model + training hyperparameters, sophon-ak4 checkpoint id, κ_λ points
  ({0,1,5}(+{-1,3})), full/lite class map.
- `scripts/delphes_to_clouds.py` — Delphes ROOT → per-jet constituent clouds
  (sophon-ak4 schema) + event-object tokens; the first data-layer step.
- `scripts/eval_to_ratios.py` — constituent analog of `data_nn_eval.py`: build the
  ordered Asimov set, run constituent ONNX inference → score → ratio → ensemble
  aggregate → save `ratio_<process>.npy` + Asimov `weights.npy` (the contract the
  JAX fit loads).
- `scripts/` + `N_*.ipynb` mirroring the FAIR example's style; `compare.py` (the
  three-control harness: pretrained-frozen / pretrained-finetuned / from-scratch /
  high-level MLP), `smoke_test.py`, `README.md`.

### 6.9 `docs/basics/sophon_hh_density_ratio.rst` (new)
- Documents the hierarchical model and how to extend it, in the same style as
  `density_ratio_training.rst`.

### 6.10 `pixi.toml`, `image.def` (extend)
- Add `weaver-core`/ParT and `huggingface_hub`. Rebuild the Apptainer image.

---

## 7. Phase 1 — Diagnostics (core deliverable)

**Task:** binary **HH-signal vs background** event-level density ratio at fixed κ_λ
(SM, κ_λ=1), using the hierarchical model.

**Three controls** (run by `compare.py`):
1. **sophon-ak4 pretrained** — frozen *and* fine-tuned variants.
2. **ParT-from-scratch** — identical architecture and inputs, random init →
   isolates the *pretraining* benefit.
3. **MLP on high-level objects** — the "no foundation model" production-style
   baseline (b-jet/τ/MET kinematics + derived masses incl. m_HH).

**Diagnostics / plots:**
- Loss-vs-epoch convergence (speed + floor) across controls.
- Reweighting-closure (`make_reweighted_plots`) and normalization
  (`test_normalization`).
- Calibration curves (`make_calib_plots`).
- **Low-statistics ablation:** fine-tune the signal ratio at {100, 25, 10, 5}% of
  the HH signal sample; plot a fixed metric (val loss and/or closure χ²) vs
  fraction, pretrained vs scratch. This is the headline figure.

**Success criteria (Phase 1):** pretrained (esp. fine-tuned) reaches a given
validation loss / closure quality in fewer epochs and/or with fewer signal events
than from-scratch, and matches or beats the high-level-object MLP baseline.

---

## 8. Phase 2 — κ_λ NSBI fit (eventual money figure)

**Physics:** non-resonant HH is quadratic in κ_λ at the amplitude level
(box + κ_λ·triangle), so the differential rate is
`dσ/dx (κ_λ) = κ_λ²·a(x) + κ_λ·b(x) + c(x)` (SM = κ_λ=1).

**Approach (recommended): morphing basis.** Train density-ratio networks for the
κ_λ basis and combine them quadratically inside the JAX model; per-component ratios
remain `.npy` files, matching the toolkit's pattern. (Alternative: a κ_λ-conditioned
network `r(x, κ_λ)` — more ML-elegant but harder to wire into the JAX fit; deferred.)

**Production plan (confirmed):** HH samples at **κ_λ ∈ {0, 1, 5}** as the baseline
basis, optionally extended with **{−1, 3}**, **≥500k events per value**. Three
points exactly determine the quadratic (a, b, c); the extra points over-constrain
it, improving the morphing fit and giving a closure cross-check. All κ_λ signal
samples are produced with the `full` TreeWriter (constituents available).

**Work required:**
- Extend `sbi_parametric_model` for κ_λ-quadratic signal morphing (rate + shape),
  with κ_λ as a parameter of interest, backgrounds κ_λ-independent. **This is the
  one place the fit layer itself changes** — Phase 1 leaves it untouched; the
  current unbinned term is linear in normfactors, so the κ_λ²·a+κ_λ·b+c combination
  of the basis ratios must be added here.
- The eval step (`eval_to_ratios.py`) must produce the **per-basis / per-component
  ratio arrays** on the shared Asimov ordering for the morphing.
- Build the κ_λ workspace via `workspace_builder`/config; profile κ_λ with
  `inference` (MIGRAD + scan).

**Success criteria (Phase 2):** a κ_λ profile-likelihood whose interval is tighter
(or converges with less training data) when using sophon-ak4-pretrained ratios vs
from-scratch.

---

## 9. Dependencies, environment, scope of "scaffold"

- New deps: `weaver-core` (or vendored ParT model code) and `huggingface_hub`.
  Update `pixi.toml` (linux-64 CUDA + osx-arm64 CPU) and `image.def`.
- Modules are **runnable**, not stubs; notebooks default to small/subsampled runs
  with a `FULL_RUN` flag.
- `smoke_test.py`: builds a tiny synthetic constituent+object batch, runs a 2-epoch
  fine-tune of the hierarchical model, asserts shapes and an ONNX round-trip.

---

## 10. Risks and open questions

- **κ_λ samples (R1, RESOLVED):** κ_λ ∈ {0,1,5} baseline (+ optional −1,3), ≥500k
  each, `full` TreeWriter. Sufficient for the quadratic morphing basis. Phase 1 is
  unblocked (fixed κ_λ).
- **Sample format & location (R2, RESOLVED):** complete Delphes ROOT output;
  downstream format adjustable. A Delphes→cloud converter (uproot/awkward) is the
  first data-layer step. Location (Perlmutter?) still to confirm for the data layer
  paths.
- **sophon-ak4 input preprocessing (R3, de-risked):** the Delphes card follows the
  JetClass-II configurations, so the domain matches sophon-ak4's training. Still
  must reproduce the checkpoint's exact per-particle feature definitions and
  normalization; validate against the sophon repo's preprocessing.
- **ONNX export of ParT with masks (R4):** ParT/Sophon support ONNX, but the
  masked, padded constituent input + the hierarchical wrapper need a verified
  export + onnxruntime path. Fallback: native-torch inference to produce `.npy`.
- **weaver-core dependency (R5):** decide depend-vs-vendor for the ParT definition
  and checkpoint loading; confirm it coexists with the toolkit's pinned torch.
- **Frozen vs fine-tuned default (R6, RESOLVED):** run **both, frozen-first** —
  frozen is fastest and strongest for the low-stat ablation; fine-tuned is the
  fuller story.
- **Mixed constituent availability (R9, new):** `lite`/fake-source classes (QCD,
  W+jets) have no constituents, so they cannot pass through the sophon-ak4 encoder.
  The likelihood must handle them via jet-level features or data-driven fake
  weights. Decide their treatment when templating the workspace.
- **MET/lepton tokenization (R7):** exact feature set and type-embedding scheme for
  non-jet tokens.
- **Confounder control (R8):** PET/sophon uses richer inputs than the high-level
  MLP baseline; the *pretrained-vs-scratch* control (same arch+inputs) is the clean
  isolation of the pretraining benefit and is the primary claim.

---

## 11. File layout (summary)

```
src/nsbi_common_utils/lightning_tools/
    particle_cloud_dataset.py        # ragged->padded constituent+object dataset
    sophon_ak4_encoder.py            # sophon-ak4 ParT per-jet encoder (HF checkpoint)
    event_transformer.py             # event-level set-transformer + ratio head
    hh_density_ratio_model.py        # HHDensityRatioLightning (composes the above)
src/nsbi_common_utils/training/
    particle_ratio_estimation.py     # particle_density_ratio_trainer
    utils.py                         # + constituent ONNX export/inference (compat)
src/nsbi_common_utils/models/
    sbi_parametric_model.py          # + kappa_lambda morphing (Phase 2)
examples/HH_bbtautau_kappalambda_sophon/
    config_train.yml, config_fit_nsbi.yml
    scripts/{delphes_to_clouds.py, eval_to_ratios.py}, scripts/, *.ipynb, compare.py, smoke_test.py, README.md
docs/basics/sophon_hh_density_ratio.rst
pixi.toml, image.def                 # + weaver-core, huggingface_hub
```

---

## 12. Out of scope / future

- From-scratch JetClass-II pretraining; boosted X→HH; κ_λ-conditioned single network
  (vs morphing basis); full systematics campaign; HTCondor DAGs for the new example
  (can mirror the FAIR `htcondor/` layer later).
