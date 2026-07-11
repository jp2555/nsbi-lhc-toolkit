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

**RESOLVED 2026-07-10: (b) CMS samples first** — `scripts/nanoaod_to_evenet_npz.py` feeds the
sweep from the CMS 2024 NanoAOD bbττ samples on Perlmutter; the Delphes path (a) stays available
via `delphes_to_evenet_npz.py` (same NPZ contract, so a Delphes cross-check is a re-run of §10
step 2). Original options for the record:
- (a) **Delphes signal** (`/pscratch/.../dihiggs/raw`, the sophon/Delphes path) — matches EveNet's Delphes pretraining domain; pheno-paper aligned.
- (b) **CMS full-sim bbττ** (`NSBI-pheno/dihiggs_bbtautau`, the prelim-result path) — matches the published 2.75σ→7.1σ result.

## 9. What EveNet already demonstrates (motivation; arXiv:2601.17126)

Data-efficiency: Exotic-Higgs at **5% data beats both baselines at full data**; pairing at 1.5% = 48% vs 24% scratch.
Low-mass with ~1–2k signal events: EveNet holds, scratch fails. Fine-tune > scratch on all 4 tasks.
**Not** an NSBI/likelihood-ratio method itself — we add the ratio head. Our novelty = data-efficiency for NSBI systematic morphing.

## 10. Option A runbook — CMS bbττ samples on Perlmutter (gates G0 / G0.5 / G1)

Gates and kill criteria: EVENET_FEASIBILITY_NOTE.pdf §8. Everything below runs on Perlmutter;
steps 2–3 are plain numpy/uproot (login node fine), 4–6 use the shifter image.

**Driver script:** `./run_option_a.sh {smoke|setup|ceiling|check <f.root>|convert|preprocess|`
`configs|train|train-local|predict|eval}` stages all of the below (env: `NTUPLES`, `ACCOUNT`;
optional `TAU_ENCODING=corner` for the G1 A/B, `TASK=syst` for Money Plot 2, `KL_HYP=0|5`).

**Input resolution (2026-07-10):** the sweep input is the **CROWN analysis ntuples** — the
same post-selection mt/et files `NSBI-pheno/dihiggs_bbtautau/convert_powheg_to_sbi.py`
reads (they carry the b-pair + ττ-leg four-vectors; `scripts/crown_to_evenet_npz.py`
builds the 4-token cloud + MET globals from them, ttbar shares the branch contract).
The flat 12-feature files on pscratch (`dihiggs_powheg_data.root`) feed the `ceiling`
stage only. If the CROWN ntuples live at KIT (`/work/jpan/bbtautau_2024`), run `convert`
there and copy the **NPZs** to Perlmutter — they are far smaller than the ntuples.
The manual commands are kept for reference (NanoAOD path via `INPUT_FORMAT=nanoaod`):

```bash
STORE=<your store>; NANO=<CMS NanoAOD base>          # kl0/ kl1/ kl5/ sample dirs

# 0. one-time: EveNet source + pretrained checkpoints
git clone https://github.com/UW-EPE-ML/EveNet_Public
hf download Avencast/EveNet --local-dir $STORE/pretrain-weights   # checkpoints.20M.a4.last.ckpt

# 1. one-time: verify branch names + era WP against a real file
python3 -c "import uproot; e=uproot.open('$NANO/kl1/<file>.root')['Events']; \
  print(e.keys(filter_name='Tau_id*')); print(e.keys(filter_name='Jet_btag*'))"
#   -> adjust --tau-id-branch / --btag-branch if needed; look up --btag-wp in the BTV tables
#      (adapter falls back UParT->DeepFlavB and PuppiMET->MET with a warning, never silently)

# 2. NanoAOD -> NPZ (G0 pairing: class 0 = SM kl=1 reference, class 1 = kl=0 or kl=5)
for kl in 0 1 5; do
  python3 scripts/nanoaod_to_evenet_npz.py --input "$NANO/kl$kl/*.root" \
    --class-id $([ $kl -eq 1 ] && echo 0 || echo 1) --btag-wp <WP> \
    --output $STORE/npz/kl$kl.npz
done
#   G1 corner A/B: rerun with --tau-encoding corner into $STORE/npz_corner/
#   (identical kinematics+conditions by construction; only tau token flags differ)

# 3. Money Plot 2 input: parametric systematic on the nominal
python3 scripts/inject_systematic.py --input $STORE/npz/kl1.npz \
  --output $STORE/npz/kl1_jesmhh.npz --kind jes_mhh --alpha 0.05

# 4. EveNet preprocess (shared 0.8/0.1/0.1 split -> evenet-train / evenet-test)
shifter --image=avencast1994/evenet:1.5 python3 EveNet_Public/preprocessing/preprocess.py \
  --files $STORE/npz/kl1.npz $STORE/npz/kl0.npz --split_ratio 0.8,0.1,0.1 \
  --store_dir $STORE/evenet-train --config <global_klambda.yaml>

# 5. sweep: 75 configs (3 modes x 5 fractions x 5 seeds) + predictions
python3 scripts/make_klambda_configs.py configs/workflow_klambda.yaml \
  --farm config_farm --store_dir $STORE --ray_dir $PSCRATCH/ray
bash config_farm/train-evenet.sh && bash config_farm/predict-evenet.sh

# 6. evaluate: AUC money plot (G0) + ratio-closure gates (G0.5)
python3 scripts/plot_data_efficiency.py --store_dir $STORE --ceiling <ceiling.json>
python3 scripts/eval_closure.py --store_dir $STORE --output-prefix closure
#   gates: |IC| < max(0.01, stat), SC_rms < 0.05 (CLOSURE_TOL); chi2/ndf ~ 1 = noise-only
```

Pre-flight checklist (carried from HANDOFF.md — do BEFORE the 75-job launch):
1. create `options_frozen.yaml` (freeze.type: full) next to the other option files;
2. load-test ONE config inside the shifter image (schema mismatch is the likely snag);
3. confirm the prediction key is `classification/klambda` (else edit plot/eval scripts' key);
4. local smoke tests pass without ROOT: `nanoaod_to_evenet_npz.py --smoke 512`,
   `eval_closure.py --self-test`.

Decision readout (G0): adopt EveNet iff finetune > scratch beyond the compute penalty AND
beats the kinematic ceiling — and (G0.5) closure gates pass wherever scratch passes. A G0
null with passing scratch closure = the objective-mismatch verdict (feasibility note §8).
