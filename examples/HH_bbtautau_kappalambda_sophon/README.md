# HH→bbττ κ_λ Sophon Example (Phase 1)

This example trains a **constituent-level hierarchical density-ratio estimator** for
the HH→bbττ signal-vs-background binary task using the sophon-ak4 Particle Transformer
(ParT) as a jet backbone. The goal of Phase 1 is to demonstrate that foundation-model
pretraining (sophon-ak4) leads to faster convergence and better data efficiency compared
to a from-scratch trained or stub encoder.

Phase 2 (separate plan) will extend this to the full κ_λ morphing fit by consuming the
per-basis ratio arrays produced here inside `sbi_parametric_model`.

---

## Architecture overview

```
Per-AK4-jet constituents (N_JETS_MAX=4, N_PART_MAX=64, F_PART=8)
        │
        ▼
  Jet Encoder (one of: stub / from-scratch ParT / sophon-ak4 ParT)
        │  → 64-d jet embedding per jet
        ▼
  Event Set-Transformer (masked multi-head self-attention)
   over { jet tokens, τ/lepton/MET tokens (N_OBJ_MAX=6, F_OBJ=6) }
        │
        ▼
  Binary density-ratio head  →  per-event score → r(x) = p_sig / p_bkg
```

Per-event features follow the `CloudSpec` conventions defined in
`nsbi_common_utils.lightning_tools.cloud_spec`:

- **Constituent features** (`F_PART=8`): `[log_pt, deta, dphi, charge, is_electron,
  is_muon, is_photon, is_charged_hadron]` (the Task 11 sophon-ak4 integration extends
  this to the full checkpoint schema including `d0/dz`).
- **Object-token features** (`F_OBJ=6`): `[log_pt, eta, sin_phi, cos_phi, mass_or_met,
  type_id]` where `type_id ∈ {0:tau_had, 1:electron, 2:muon, 3:met}`.

---

## Pipeline

### Step 1 — Convert Delphes samples to padded clouds

```bash
python scripts/delphes_to_clouds.py \
    --input /path/to/hh_bbtautau_sm.root --tree Delphes \
    --output saved_clouds/hh_lam1.npz

python scripts/delphes_to_clouds.py \
    --input /path/to/ttbar.root --tree Delphes \
    --output saved_clouds/ttbar.npz
```

Each `.npz` contains the arrays `parts`, `part_mask`, `jet_mask`, `obj`, `obj_mask`,
`w` matching the `WeightedParticleCloudDataset` schema.

### Step 2 — Train the comparison controls

Edit `config_train.yml` to point at your `.npz` files, then run:

```bash
python compare.py
```

Or use the API directly:

```python
import numpy as np
from nsbi_common_utils.training.particle_ratio_estimation import particle_density_ratio_trainer

clouds = dict(np.load("saved_clouds/hh_lam1.npz"))
# merge signal + background, add y labels and w weights, then:
tr = particle_density_ratio_trainer(
    clouds=clouds, sample_name=["signal", "background"],
    output_name="hh_vs_ttbar", path_to_models="models/",
    encoder_kind="sophon-ak4",   # or "stub" / "scratch"
    freeze_backbone=True,
)
tr.train(number_of_epochs=100, batch_size=512, learning_rate=1e-3)
```

The comparison harness `compare.py::run_controls` trains all requested controls in
sequence and returns per-control training histories for convergence plots.

#### Available controls

| Key             | Encoder          | Backbone frozen? | Purpose                        |
|-----------------|------------------|------------------|--------------------------------|
| `stub_scratch`  | StubJetEncoder   | No               | Fast CPU baseline (no ParT)    |
| `stub_frozen`   | StubJetEncoder   | Yes              | Sanity check                   |
| `scratch`       | ParT random init | No               | Architecture control           |
| `sophon_frozen` | sophon-ak4       | Yes              | Foundation model, frozen       |
| `sophon_finetune`| sophon-ak4      | No               | Foundation model, fine-tuned   |

The `stub_*` controls run on CPU/CI without any external checkpoint. The `scratch` and
`sophon_*` controls require the sophon-ak4 backbone (see below).

### Step 3 — Evaluate per-event ratios on the Asimov set

```bash
python scripts/eval_to_ratios.py \
    --clouds saved_clouds/ --model models/model0.onnx \
    --output ratios/
```

This writes `ratio_<process>.npy` and `weights.npy` on the shared Asimov event ordering
that `sbi_parametric_model` expects.

### Step 4 — Phase-2 κ_λ fit (separate plan)

The ratio `.npy` files produced above are the inputs to the Phase-2 morphing fit.
See the companion plan for details.

---

## Sophon-ak4 checkpoint

To run the `sophon_*` controls, download the checkpoint from HuggingFace. The weight
file in `jet-universe/sophon-ak4` is `models/JetClassII_SophonAK4/model.pt` (do this on a
login node — compute nodes may lack internet):

```bash
hf download jet-universe/sophon-ak4 \
    models/JetClassII_SophonAK4/model.pt --local-dir $SCRATCH/sophon-ak4
export SOPHON_AK4_CKPT=$SCRATCH/sophon-ak4/models/JetClassII_SophonAK4/model.pt
```

(`hf` is the current HuggingFace CLI; the old `huggingface-cli download` is deprecated and
no longer works. In the pixi env it's `pixi run -e nsbi-env-gpu hf download ...`.)

Or let `huggingface_hub` fetch it automatically by setting the (repo-relative) filename in
`config_train.yml`:

```yaml
encoder:
  kind: sophon-ak4
  checkpoint: models/JetClassII_SophonAK4/model.pt
```

Set the environment variable so integration tests can find it:

```bash
export SOPHON_AK4_CKPT=$SCRATCH/sophon-ak4/models/JetClassII_SophonAK4/model.pt
```

---

## Running the smoke test

A tiny synthetic end-to-end check (no Delphes files needed, no checkpoint):

```bash
python smoke_test.py
# or via pytest:
pytest tests/test_smoke_end_to_end.py -v
```

Expected output: `{'onnx_exists': True, 'ratios': array([...], dtype=float32)}`.

---

## Running on Perlmutter (NERSC)

Activate the pixi environment, then run any of the scripts above:

```bash
pixi run -e nsbi-env-gpu python compare.py
pixi run -e nsbi-env-gpu pytest -q
```

For the full suite with GPU acceleration, submit via SLURM with `--gpus-per-node=1`.

### Stats data-efficiency ablation (batch job)

`run_ablation.sbatch` trains all three controls at several per-point event counts on
one GPU (`shared` QOS) and writes `ablation.json` + `ablation.png` (best val-loss vs N).
Convert the two points first (`convert_all_kl.py --points kl0 kl5`), then:

```bash
mkdir -p logs
CKPT=$SCRATCH/sophon-ak4/models/JetClassII_SophonAK4/model.pt \
  sbatch -A <NERSC_PROJECT>_g \
    examples/HH_bbtautau_kappalambda_sophon/scripts/run_ablation.sbatch
```

Override `CLOUDS`, `OUTDIR`, `POINT_A`/`POINT_B`, `SIZES`, `REPEATS`, `EPOCHS`, `LR` via
env (see the script header). The default task is `kl0` vs the `kl5` reference, swept over
N = 2k/5k/10k/20k/50k/100k events per point — the low end is the foundation-model stress
test. Low-N single runs are noisy, so set `REPEATS=3` (retrain each point with different
seeds → mean ± std error bands) when you want a trustworthy small-N curve:

```bash
REPEATS=3 CKPT=$SCRATCH/sophon-ak4/models/JetClassII_SophonAK4/model.pt \
  sbatch -A <NERSC_PROJECT>_g \
    examples/HH_bbtautau_kappalambda_sophon/scripts/run_ablation.sbatch
```

`REPEATS` multiplies runtime; if the full sweep × repeats overruns the 6 h `shared`
limit, run the low-N points with repeats separately (e.g. `SIZES="2000 5000 10000"
REPEATS=5`) from the high-N anchor.

---

## File reference

| File | Purpose |
|------|---------|
| `scripts/delphes_to_clouds.py` | Delphes ROOT → padded cloud `.npz` |
| `scripts/eval_to_ratios.py` | Clouds + ONNX → per-event ratio `.npy` on Asimov ordering |
| `compare.py` | 3-control harness + low-statistics data-efficiency ablation |
| `smoke_test.py` | Tiny synthetic end-to-end pipeline check |
| `config_train.yml` | Training hyperparameters and data paths |
