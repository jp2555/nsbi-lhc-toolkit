# HANDOFF — EveNet data-efficiency money plots for κ_λ (HH→bbττ)

**For:** a fresh session continuing on **Perlmutter**. **Repo/branch:** `jp2555/nsbi-lhc-toolkit @ fm`,
code in `examples/HH_bbtautau_kappalambda_sophon/`. **Date:** 2026-06-26.

## Mission (one paragraph)
Produce **money plots** showing a pretrained event-level foundation model (**EveNet**, arXiv:2601.17126)
improves training in the **low-statistics** regime — especially for **systematic variations** — for the
κ_λ measurement in HH→bbττ. Success metric = **data efficiency** (AUC / closure vs training-set fraction),
NOT raw separation. Keep the kinematic-ceiling MLP/BDT as the bar. **Do NOT quote σ_κλ significance**
(that needs the deferred MC-stat/wifi work — separate project). Full rationale + contract:
[`EVENET_INTEGRATION_PLAN.md`](EVENET_INTEGRATION_PLAN.md).

## State: what's done (all on branch `fm`, latest commit `833e2d1`)
Built locally on a bare Mac (no torch/data) and **verified at the numpy/yaml layer only**:

| File | Role | Verified locally |
|---|---|---|
| `scripts/delphes_to_evenet_npz.py` | Delphes ROOT → EveNet NPZ (`--smoke` mode) | ✅ schema/mask/non-neg |
| `scripts/inject_systematic.py` | parametric `jes`/`jes_mhh` systematic | ✅ scaling/shape |
| `configs/event_info_klambda.yaml` | binary `[ref, kl_hyp]` event_info | template (load-test pending) |
| `configs/workflow_klambda.yaml` | sweep control file (`<PLACEHOLDER>` paths) | template |
| `configs/train_klambda.yaml` | global train config | template |
| `scripts/make_klambda_configs.py` | emits 3×5×5=75 configs + `train-evenet.sh` | ✅ wiring/counts |
| `scripts/plot_data_efficiency.py` | AUC-vs-fraction money plot | ✅ `weighted_auc` |

**Nothing torch/EveNet-touching has run yet** — that's this session's job. Reference EveNet source is
cloned on the Mac at `~/Desktop/evenet-src/{Core,EveNet-Full,EveNet-Lite,Exotic-Higgs-Study}`; on
Perlmutter, clone EveNet-Full + Exotic-Higgs-Study fresh (the harness mirrors Exotic-Higgs-Study's).

## Decision log (settled — do not re-open)
- **Input dataset = Delphes bbττ** (matches EveNet's Delphes pretraining domain; pheno-paper aligned). Not full-sim.
- **Ratio head = binary classifier** `CLASSLABEL [ref, kl_hyp]` → `softmax[...,1]` → `eval_to_ratios.py` LR trick. (Regression head rejected as less battle-tested.)
- **τ_h → jet, photons → generic object**: EveNet has *no* tau/photon type (only `btag/isLepton/charge`). Stock 7+10 schema only (adding features breaks pretrained-weight reuse). Accepts τ-ID / SVfit-m_ττ loss — valid for the data-efficiency comparison.
- **3-way contrast via the options file**: finetune=`options_pretrain.yaml`(freeze none), frozen=`options_frozen.yaml`(freeze full), scratch=`options.yaml`(pretrain null).
- **wifi/MC-stat uncertainty (2506.00113) = DEFERRED** to a separate project (was explored: frozen-FM-as-wifi-basis; real but out of scope now).
- **bbγγ = cheap later increment** (only the adapter's object-filling changes); deferred (cleaner m_HH but smaller FM headroom + no infra yet).

## Next actions on Perlmutter (detail in EVENET_INTEGRATION_PLAN.md §5–§7)
1. `shifterimg -v pull docker:avencast1994/evenet:1.5`; `hf download Avencast/EveNet --local-dir $STORE/pretrain-weights`.
2. **Create `options_frozen.yaml`**: `cp` EveNet's `options_pretrain.yaml`, set `Training.GlobalEmbedding.freeze.type: full`.
3. Fill every `<PLACEHOLDER>` in `configs/workflow_klambda.yaml` + `train_klambda.yaml` ($STORE, account, wandb entity, network/resonance/predict paths).
4. **Load-test before launching 75 jobs** (inside the image): `python -c "from evenet.control.global_config import global_config as g; g.load_yaml('config_farm/<one>.yaml'); print('ok')"`. **Most likely snag = train/event_info schema mismatch vs the real loader** — fix the templates against the actual error.
5. Run: `delphes_to_evenet_npz.py` (real ROOT, one NPZ/sample, ref=0/hyp=1) → `preprocess.py` → `make_klambda_configs.py` → `bash config_farm/train-evenet.sh` → predict → `plot_data_efficiency.py`.
6. **Confirm the prediction key is `classification/klambda`** (the plot reads it; one-line fix if EveNet names it differently).
7. **Money Plot 2 (systematics)**: same pipeline with `inject_systematic.py` producing the varied sample (class 1) vs nominal (class 0) — no harness change.

## Constraints & gotchas
- All real runs are Perlmutter (containerized, not pixi). The Mac venv only validated numpy/yaml logic.
- `seed` is set at `options.Training.seed` in generated configs — **verify EveNet actually reads that key**, else wire to the correct one.
- Delphes branch assumptions in the adapter (`Jet.BTag` etc.) follow `delphes_to_sophon_clouds.py`; verify against the real files with `uproot.open(f)["Delphes"].keys()`.
- Keep commits brief (6–10 words), **no Claude attribution** (user's CLAUDE.md). Auto-push to `origin/fm`.

## Pointers
- Plan/contract: `EVENET_INTEGRATION_PLAN.md` (NPZ schema, object mapping, run env, money plots).
- Landscape/why-EveNet: `FOUNDATION_MODEL_BRIEF.md`.
- Prelim physics: Ghosh–Klute–Pan note (NSBI-pheno/dihiggs_bbtautau) — full-sim 2.75σ→7.1σ, no systematics in fit.
- Memory slug: `fm-systematics-wifi-direction` (has STATUS + REMAINING).

## Suggested skills
- `superpowers:verification-before-completion` — run/inspect before claiming any EveNet step works (evidence before assertions).
- `superpowers:systematic-debugging` — for the first config-load / preprocess / train failure (expected at step 4).
- `remember` — save state at the end of the Perlmutter session.
