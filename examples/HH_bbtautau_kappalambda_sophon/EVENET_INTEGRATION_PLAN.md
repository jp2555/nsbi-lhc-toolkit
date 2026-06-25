# EveNet integration for κ_λ low-stat money plots — concrete plan

**Goal.** Money plots showing a pretrained event-level FM (EveNet) improves training in the
low-statistics regime — especially for **systematic variations** — for HH→bbττ κ_λ. Metrics stay at
the ML/closure level (AUC/SIC, closure-vs-N); **no σ_κλ significance** (that needs the deferred
MC-stat/wifi work). Decision bar = the kinematic-ceiling MLP/BDT (`run_kinematic_baseline.py`).

Reverse-engineered from the EveNet codebase (cloned to `~/Desktop/evenet-src/{Core,EveNet-Full,EveNet-Lite,Exotic-Higgs-Study}`,
upstream `github.com/EveNet-HEP`, paper arXiv:2601.17126). `Exotic-Higgs-Study` is the worked
downstream template we mirror.

---

## 1. Run environment (Perlmutter, containerized — NOT the toolkit pixi env)

- **Container:** `shifterimg -v pull docker:avencast1994/evenet:1.5`
- **Code:** `EveNet-Full` (training pipeline, Ray+Lightning+wandb) + `Core` (network). Python 3.12.
- **Pretrained weights:** `hf download Avencast/EveNet --local-dir pretrain-weights`
  → `checkpoints.20M.a4.last.ckpt` (nominal, 20M params) · `SSL.20M.last.ckpt` (SSL variant).
- This Mac is bare (no torch/numpy) → **all runs are Perlmutter**; locally we can only author code/configs.

## 2. Input contract (what the adapter must emit)

Per-file **NPZ** (then `preprocess.py` → parquet). Required keys (`EveNet-Full/preprocessing/helper.py`):

| key | shape / dtype | meaning |
|---|---|---|
| `x` | (N, N_obj_max, 7) float | Source objects: `[energy, pt, eta, phi, btag, isLepton, charge]` |
| `conditions` | (N, 10) float | Globals: `[met, met_phi, nLepton, nbJet, nJet, HT, HT_lep, M_all, M_leps, M_bjets]` |
| `num_sequential_vectors` | (N,) int | # real objects per event (padding/mask) |
| `event_weight` | (N,) float | per-event weight |
| `classification` | (N,) int | class id — binary: `0`=reference, `1`=κ_λ-hyp |

Optional: `x_invisible` (neutrinos), `regression-data`, `segmentation-*`, `assignments-mask`, `process_names`.

Convert: `python preprocessing/preprocess.py --files *.npz --split_ratio 0.8,0.1,0.1 --store_dir <out> --config <global.yaml>`

## 3. bbττ → EveNet object mapping (stock 7+10 schema — preserves pretrained weights)

- **b-jets** → Source object, `btag=1, isLepton=0, charge=0`
- **τ_h** → Source object as a jet, `btag≈0, isLepton=0` — **loses explicit τ-ID** (no tau type exists)
- **e/μ** (incl. from τ_lep) → Source object, `isLepton=1, charge=±1`
- **MET** → `conditions[met, met_phi]`
- **globals** computed from objects: `nLepton, nbJet, nJet, HT, HT_lep, M_all, M_leps, M_bjets`
- **CAVEAT:** no slot for SVfit `m_ττ` or τ-ID. Adding columns breaks pretrained-weight reuse →
  **first pass uses the stock schema only**. Hurts absolute m_HH power, NOT the validity of the
  frozen/finetune/scratch *data-efficiency* comparison. Adding τ-features (partial weight init) is a later extension.

## 4. event_info_klambda.yaml (adapt `Exotic-Higgs-Study/configs/event_info_30.yaml`)

- `INPUTS.SEQUENTIAL.Source` + `GLOBAL.Conditions`: **identical to pretrain** (so the backbone weights bind).
- `CLASSIFICATIONS.EVENT: [klambda]`; `CLASSLABEL.EVENT.klambda: [ref, kl_hyp]` (binary).
- `EVENT`/`PERMUTATIONS`: optional HH→`{h_bb:[b1,b2], h_tt:[t1,t2]}` for assignment — **OFF for first pass**.
- Ratio: classifier score `softmax(classification/klambda)[...,1]` → `r(x)` via existing `eval_to_ratios.py` (LR trick).

## 5. Money Plot 1 — κ_λ separation data-efficiency

- Sweep `dataset_size_choice = [0.01, 0.03, 0.1, 0.3, 1.0]` × config × ≥5 seeds.
  - **finetune:** `Training.pretrain_model_load_path = checkpoints.20M.a4.last.ckpt`, `GlobalEmbedding.freeze.type = none`
  - **frozen:** same checkpoint, `GlobalEmbedding.freeze.type = full` (train head only)
  - **scratch:** `pretrain_model_load_path = null`
- Generate configs with a `Make_script.py` adapted from Exotic-Higgs (`Dataset.dataset_limit = size`); run `scripts/train.py <cfg> --ray_dir <tmp>`.
- y = AUC/SIC of `classification/klambda` vs N_train; overlay **kinematic-ceiling** curve. Money = FM left-shift + equivalent-data multiplier.

## 6. Money Plot 2 — systematic-variation data-efficiency (parametric, the headline)

- `inject_systematic.py`: take nominal bbττ NPZ, apply a **known** smooth distortion (JES-like jet-energy scale `f`,
  or an m_HH-dependent reweight), emit a "varied" NPZ + truth weight.
- Binary classification = **nominal(0) vs varied(1)**. Same `dataset_size` × {finetune, frozen, scratch} sweep over N_var.
- y = AUC/closure of nominal-vs-varied vs N_var. **Money = FM recovers the systematic shape at small N_var where scratch fails.**
- Note: leans on EveNet's demonstrated *data-efficiency*, NOT its robustness/insensitivity result (opposite framing).

## 7. Files to create (in `examples/HH_bbtautau_kappalambda_sophon/`)

1. `scripts/<input>_to_evenet_npz.py` — input events → §2 NPZ schema. **[input dataset TBD — see §8]**
2. `configs/event_info_klambda.yaml` — §4
3. `configs/workflow_evenet.yaml` + `Make_script.py` adaptation — §5
4. `scripts/inject_systematic.py` — §6 (pure numpy, unit-testable)
5. `scripts/plot_data_efficiency.py` — §5/§6 curves + ceiling overlay

(1,4 = numpy/uproot; 2,3 = config; all execute on Perlmutter via the shifter image.)

## 8. Open decision (blocks the adapter's input side)

**Which dataset feeds EveNet?**
- (a) **Delphes signal** (`/pscratch/.../dihiggs/raw`, the sophon/Delphes path) — matches EveNet's Delphes pretraining domain; pheno-paper aligned.
- (b) **CMS full-sim bbττ** (`NSBI-pheno/dihiggs_bbtautau`, the prelim-result path) — matches the published 2.75σ→7.1σ result.

This sets the adapter's input parsing. Mapping (§3) and everything downstream are identical either way.

## 9. What EveNet already demonstrates (motivation; arXiv:2601.17126)

Data-efficiency: Exotic-Higgs at **5% data beats both baselines at full data**; pairing at 1.5% = 48% vs 24% scratch.
Low-mass with ~1–2k signal events: EveNet holds, scratch fails. Fine-tune > scratch on all 4 tasks.
**Not** an NSBI/likelihood-ratio method itself — we add the ratio head. Our novelty = data-efficiency for NSBI systematic morphing.
