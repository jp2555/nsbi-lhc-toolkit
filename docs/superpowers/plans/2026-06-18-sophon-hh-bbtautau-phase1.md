# Sophon HH→bbττ density-ratio (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a constituent-level (sophon-ak4) hierarchical density-ratio estimator in `nsbi-lhc-toolkit` and a diagnostics harness that shows foundation-model pretraining helps the HH→bbττ signal-vs-background classifier (faster/better convergence + low-statistics data-efficiency).

**Architecture:** Per-AK4-jet constituents → a swappable jet encoder (sophon-ak4 ParT, fine-tuned/frozen; a same-shape from-scratch ParT; or a tiny stub for tests) → 64-d jet embeddings → an event set-transformer over {jet tokens, τ/lepton/MET tokens} → a binary density-ratio head. Trained as a `pl.LightningModule`, exported to ONNX, evaluated to per-event ratio `.npy` on the shared Asimov ordering (the existing JAX fit is untouched in Phase 1).

**Tech Stack:** PyTorch + PyTorch Lightning, ONNX/onnxruntime, uproot/awkward (Delphes I/O), huggingface_hub + weaver-core/particle_transformer (sophon-ak4 checkpoint), pytest, pixi.

**Scope note:** This is Phase 1 only. The κ_λ morphing fit (Phase 2) — extending `sbi_parametric_model` and porting the NSBI-pheno morphing — is a separate plan that consumes the per-basis ratios this phase produces.

**Testability strategy:** Every core module is TDD'd against **synthetic constituent clouds** and a **`StubJetEncoder`** (no external deps, no GPU). The real sophon-ak4 checkpoint and real Delphes conversion are isolated integration tasks (Task 11, Task 7) validated when those resources are available on Perlmutter. Tests run inside the pixi env: prefix commands with `pixi run -e nsbi-env-gpu` (or the project's CPU env) if not already in a pixi shell.

**Shape conventions (module-level constants in `cloud_spec.py`, Task 2):**
- `N_JETS_MAX = 4`, `N_PART_MAX = 64`, `F_PART = 8`, `N_OBJ_MAX = 6`, `F_OBJ = 6`, `EMBED_DIM = 64`.
- Per-constituent features (`F_PART=8`): `[log_pt, deta, dphi, charge, is_electron, is_muon, is_photon, is_charged_hadron]` (the sophon-ak4 integration task extends this to the full checkpoint schema incl. `d0/dz`).
- Object-token features (`F_OBJ=6`): `[log_pt, eta, sin_phi, cos_phi, mass_or_met, type_id]` where `type_id ∈ {0:tau_had, 1:electron, 2:muon, 3:met}`.

---

## File structure

| File | Responsibility |
|------|----------------|
| `pyproject.toml`, `pixi.toml`, `image.def` | add pytest, huggingface_hub, weaver-core/particle_transformer deps |
| `tests/conftest.py` | synthetic-cloud fixtures shared across tests |
| `src/nsbi_common_utils/lightning_tools/cloud_spec.py` | shape constants + `CloudSpec` dataclass |
| `src/nsbi_common_utils/lightning_tools/particle_cloud_dataset.py` | `WeightedParticleCloudDataset` (ragged→padded) |
| `src/nsbi_common_utils/lightning_tools/event_transformer.py` | `EventSetTransformer` + masked attention pool |
| `src/nsbi_common_utils/lightning_tools/jet_encoder.py` | `StubJetEncoder`, `build_jet_encoder()` factory (stub/scratch/sophon-ak4) |
| `src/nsbi_common_utils/lightning_tools/hh_density_ratio_model.py` | `HHDensityRatioLightning(pl.LightningModule)` |
| `src/nsbi_common_utils/training/utils.py` | `+ save_model_constituents`, `+ predict_with_onnx_constituents` |
| `src/nsbi_common_utils/training/particle_ratio_estimation.py` | `particle_density_ratio_trainer` |
| `src/nsbi_common_utils/lightning_tools/__init__.py`, `training/__init__.py` | export new symbols |
| `examples/HH_bbtautau_kappalambda_sophon/scripts/delphes_to_clouds.py` | Delphes ROOT → padded clouds (`.npz`) |
| `examples/HH_bbtautau_kappalambda_sophon/scripts/eval_to_ratios.py` | clouds + ONNX → per-event ratio `.npy` on Asimov order |
| `examples/HH_bbtautau_kappalambda_sophon/compare.py` | 3-control harness + low-stat ablation |
| `examples/HH_bbtautau_kappalambda_sophon/smoke_test.py` | end-to-end tiny pipeline check |
| `examples/HH_bbtautau_kappalambda_sophon/config_train.yml`, `README.md` | example config + docs |
| `docs/basics/sophon_hh_density_ratio.rst` | toolkit docs page |

---

## Task 1: Dev/runtime dependencies + test harness

**Files:**
- Modify: `pyproject.toml`
- Modify: `pixi.toml`
- Create: `tests/conftest.py`
- Create: `tests/test_harness_smoke.py`

- [ ] **Step 1: Add a trivially-true test to prove pytest runs**

Create `tests/test_harness_smoke.py`:
```python
def test_pytest_runs():
    assert True
```

- [ ] **Step 2: Add pytest + deps to pixi**

Dependencies/environments are defined in **`pixi.toml`** (envs `nsbi-env` [cpu] and
`nsbi-env-gpu` [gpu] via `[feature.cpu]`/`[feature.gpu]`; tables `[dependencies]`
[conda] + `[pypi-dependencies]`). Add to `pixi.toml` `[dependencies]` (shared by both
envs):
```toml
pytest = ">=8.0"
huggingface_hub = ">=0.24"
```
Add a pytest config block to `pyproject.toml` (pytest reads it there):
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```
After editing, re-solve the env: `pixi install -e nsbi-env-gpu`.
(weaver-core / particle_transformer are added in Task 11, the integration task, to avoid blocking the testable core.)

- [ ] **Step 3: Run the harness test**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_harness_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 4: Create shared synthetic-cloud fixtures**

Create `tests/conftest.py`:
```python
import numpy as np
import pytest

N_JETS_MAX, N_PART_MAX, F_PART, N_OBJ_MAX, F_OBJ = 4, 64, 8, 6, 6


def _rng(seed):
    return np.random.default_rng(seed)


def make_event(seed, n_jets=2, n_parts=20, n_obj=3):
    """One synthetic event: padded clouds + masks + object tokens."""
    r = _rng(seed)
    parts = np.zeros((N_JETS_MAX, N_PART_MAX, F_PART), dtype=np.float32)
    part_mask = np.zeros((N_JETS_MAX, N_PART_MAX), dtype=np.float32)
    jet_mask = np.zeros((N_JETS_MAX,), dtype=np.float32)
    for j in range(min(n_jets, N_JETS_MAX)):
        jet_mask[j] = 1.0
        for p in range(min(n_parts, N_PART_MAX)):
            part_mask[j, p] = 1.0
            parts[j, p] = r.normal(size=F_PART).astype(np.float32)
    obj = np.zeros((N_OBJ_MAX, F_OBJ), dtype=np.float32)
    obj_mask = np.zeros((N_OBJ_MAX,), dtype=np.float32)
    for o in range(min(n_obj, N_OBJ_MAX)):
        obj_mask[o] = 1.0
        obj[o] = r.normal(size=F_OBJ).astype(np.float32)
        obj[o, -1] = float(o % 4)  # type_id
    return parts, part_mask, jet_mask, obj, obj_mask


@pytest.fixture
def synth_batch():
    """A small labelled, weighted batch as numpy arrays."""
    n = 32
    parts, pm, jm, obj, om, y, w = [], [], [], [], [], [], []
    for i in range(n):
        p, pmask, jmask, o, omask = make_event(i, n_jets=1 + i % 3)
        parts.append(p); pm.append(pmask); jm.append(jmask); obj.append(o); om.append(omask)
        y.append(i % 2)
        w.append(1.0)
    return {
        "parts": np.stack(parts), "part_mask": np.stack(pm), "jet_mask": np.stack(jm),
        "obj": np.stack(obj), "obj_mask": np.stack(om),
        "y": np.array(y, dtype=np.float32), "w": np.array(w, dtype=np.float32),
    }
```

- [ ] **Step 5: Commit**
```bash
git add pyproject.toml pixi.toml tests/
git commit -m "test: set up pytest harness + synthetic-cloud fixtures"
```

---

## Task 2: Cloud spec constants

**Files:**
- Create: `src/nsbi_common_utils/lightning_tools/cloud_spec.py`
- Test: `tests/test_cloud_spec.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cloud_spec.py`:
```python
from nsbi_common_utils.lightning_tools.cloud_spec import CloudSpec, DEFAULT_SPEC


def test_default_spec_shapes():
    s = DEFAULT_SPEC
    assert (s.n_jets_max, s.n_part_max, s.f_part) == (4, 64, 8)
    assert (s.n_obj_max, s.f_obj, s.embed_dim) == (6, 6, 64)


def test_spec_is_overridable():
    s = CloudSpec(n_jets_max=2)
    assert s.n_jets_max == 2 and s.n_part_max == 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_cloud_spec.py -v`
Expected: FAIL with `ModuleNotFoundError: ...cloud_spec`.

- [ ] **Step 3: Implement**

Create `src/nsbi_common_utils/lightning_tools/cloud_spec.py`:
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class CloudSpec:
    """Fixed-size padded representation of an event for the hierarchical model."""
    n_jets_max: int = 4
    n_part_max: int = 64
    f_part: int = 8
    n_obj_max: int = 6
    f_obj: int = 6
    embed_dim: int = 64


DEFAULT_SPEC = CloudSpec()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_cloud_spec.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**
```bash
git add src/nsbi_common_utils/lightning_tools/cloud_spec.py tests/test_cloud_spec.py
git commit -m "feat: add CloudSpec shape constants for constituent clouds"
```

---

## Task 3: Weighted particle-cloud dataset

**Files:**
- Create: `src/nsbi_common_utils/lightning_tools/particle_cloud_dataset.py`
- Test: `tests/test_particle_cloud_dataset.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_particle_cloud_dataset.py`:
```python
import torch
from nsbi_common_utils.lightning_tools.particle_cloud_dataset import WeightedParticleCloudDataset


def test_dataset_yields_tensors(synth_batch):
    ds = WeightedParticleCloudDataset(**synth_batch)
    assert len(ds) == 32
    item = ds[0]
    assert set(item.keys()) == {"parts", "part_mask", "jet_mask", "obj", "obj_mask", "y", "w"}
    assert item["parts"].shape == (4, 64, 8)
    assert item["obj"].shape == (6, 6)
    assert item["y"].dtype == torch.float32 and item["w"].dtype == torch.float32


def test_dataloader_batches(synth_batch):
    from torch.utils.data import DataLoader
    ds = WeightedParticleCloudDataset(**synth_batch)
    batch = next(iter(DataLoader(ds, batch_size=8)))
    assert batch["parts"].shape == (8, 4, 64, 8)
    assert batch["jet_mask"].shape == (8, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_particle_cloud_dataset.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

Create `src/nsbi_common_utils/lightning_tools/particle_cloud_dataset.py`:
```python
import numpy as np
import torch
from torch.utils.data import Dataset


class WeightedParticleCloudDataset(Dataset):
    """Yields padded per-jet constituent clouds + event object tokens + (label, weight).

    All inputs are numpy arrays with a leading event axis:
      parts:     (N, n_jets_max, n_part_max, f_part)
      part_mask: (N, n_jets_max, n_part_max)   1.0 = real particle
      jet_mask:  (N, n_jets_max)               1.0 = real jet
      obj:       (N, n_obj_max, f_obj)
      obj_mask:  (N, n_obj_max)                1.0 = real object token
      y:         (N,)  binary label (1 = numerator / hypothesis A)
      w:         (N,)  per-event weight (already class-normalised by the caller)
    """

    def __init__(self, parts, part_mask, jet_mask, obj, obj_mask, y, w):
        self.parts = torch.as_tensor(np.asarray(parts), dtype=torch.float32)
        self.part_mask = torch.as_tensor(np.asarray(part_mask), dtype=torch.float32)
        self.jet_mask = torch.as_tensor(np.asarray(jet_mask), dtype=torch.float32)
        self.obj = torch.as_tensor(np.asarray(obj), dtype=torch.float32)
        self.obj_mask = torch.as_tensor(np.asarray(obj_mask), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        self.w = torch.as_tensor(np.asarray(w), dtype=torch.float32)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return {
            "parts": self.parts[i], "part_mask": self.part_mask[i],
            "jet_mask": self.jet_mask[i], "obj": self.obj[i],
            "obj_mask": self.obj_mask[i], "y": self.y[i], "w": self.w[i],
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_particle_cloud_dataset.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**
```bash
git add src/nsbi_common_utils/lightning_tools/particle_cloud_dataset.py tests/test_particle_cloud_dataset.py
git commit -m "feat: WeightedParticleCloudDataset for constituent clouds"
```

---

## Task 4: Event set-transformer

**Files:**
- Create: `src/nsbi_common_utils/lightning_tools/event_transformer.py`
- Test: `tests/test_event_transformer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_event_transformer.py`:
```python
import torch
from nsbi_common_utils.lightning_tools.event_transformer import EventSetTransformer


def test_forward_shape_and_masking():
    B, L, D = 5, 10, 64
    tokens = torch.randn(B, L, D)
    mask = torch.ones(B, L)
    mask[:, 7:] = 0.0  # last 3 tokens padded
    net = EventSetTransformer(embed_dim=D, n_heads=8, n_layers=2)
    out = net(tokens, mask)
    assert out.shape == (B, 1)
    # Changing only padded tokens must not change the output (mask works)
    tokens2 = tokens.clone()
    tokens2[:, 7:] = torch.randn(B, 3, D)
    out2 = net(tokens2, mask)
    assert torch.allclose(out, out2, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_event_transformer.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

Create `src/nsbi_common_utils/lightning_tools/event_transformer.py`:
```python
import torch
import torch.nn as nn


class EventSetTransformer(nn.Module):
    """Masked multi-head self-attention over event tokens -> scalar logit.

    Padded tokens (mask==0) are excluded from attention via key_padding_mask and
    from the final masked-mean pool, so their values cannot affect the output.
    """

    def __init__(self, embed_dim=64, n_heads=8, n_layers=2, ff_mult=2, dropout=0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * ff_mult, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1),
        )

    def forward(self, tokens, mask):
        # mask: (B, L) with 1.0 = real. TransformerEncoder wants True = PAD.
        key_padding = mask < 0.5
        # A fully-padded row would make attention nan; guarantee >=1 valid token.
        safe = key_padding.clone()
        safe[:, 0] = False
        z = self.encoder(tokens, src_key_padding_mask=safe)
        m = mask.unsqueeze(-1)
        pooled = (z * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_event_transformer.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**
```bash
git add src/nsbi_common_utils/lightning_tools/event_transformer.py tests/test_event_transformer.py
git commit -m "feat: EventSetTransformer with masked attention + pooling"
```

---

## Task 5: Jet encoder (stub + factory)

**Files:**
- Create: `src/nsbi_common_utils/lightning_tools/jet_encoder.py`
- Test: `tests/test_jet_encoder.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_jet_encoder.py`:
```python
import torch
from nsbi_common_utils.lightning_tools.jet_encoder import StubJetEncoder, build_jet_encoder


def test_stub_encoder_shape_and_mask():
    B, P, F, D = 7, 64, 8, 64
    parts = torch.randn(B, P, F)
    mask = torch.ones(B, P)
    mask[:, 30:] = 0.0
    enc = StubJetEncoder(f_part=F, embed_dim=D)
    emb = enc(parts, mask)
    assert emb.shape == (B, D)
    parts2 = parts.clone(); parts2[:, 30:] = torch.randn(B, 34, F)
    assert torch.allclose(emb, enc(parts2, mask), atol=1e-5)


def test_factory_builds_stub():
    enc = build_jet_encoder(kind="stub", f_part=8, embed_dim=64)
    assert isinstance(enc, StubJetEncoder)
    assert enc(torch.randn(3, 64, 8), torch.ones(3, 64)).shape == (3, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_jet_encoder.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

Create `src/nsbi_common_utils/lightning_tools/jet_encoder.py`:
```python
import torch
import torch.nn as nn


class StubJetEncoder(nn.Module):
    """Lightweight masked DeepSets encoder for tests / a no-pretraining baseline.

    Maps a padded constituent cloud (B, n_part, f_part) + mask (B, n_part) to a
    (B, embed_dim) jet embedding. Permutation-invariant and mask-respecting.
    """

    def __init__(self, f_part=8, embed_dim=64, hidden=128):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(f_part, hidden), nn.GELU(), nn.Linear(hidden, embed_dim), nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, parts, mask):
        h = self.phi(parts)                       # (B, P, D)
        m = mask.unsqueeze(-1)                     # (B, P, 1)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.rho(pooled)


def build_jet_encoder(kind="stub", f_part=8, embed_dim=64, **kwargs):
    """Factory for the per-jet encoder.

    kind:
      - "stub"      : StubJetEncoder (tests, CPU, no external deps)
      - "scratch"   : a ParT of the sophon-ak4 shape, random init (Task 11)
      - "sophon-ak4": ParT loaded from the released checkpoint (Task 11)
    """
    if kind == "stub":
        return StubJetEncoder(f_part=f_part, embed_dim=embed_dim, **kwargs)
    if kind in ("scratch", "sophon-ak4"):
        from nsbi_common_utils.lightning_tools.sophon_ak4_backbone import build_part_encoder
        return build_part_encoder(kind=kind, f_part=f_part, embed_dim=embed_dim, **kwargs)
    raise ValueError(f"unknown jet encoder kind: {kind!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_jet_encoder.py -v`
Expected: 2 passed. (The `scratch`/`sophon-ak4` branch imports a module created in Task 11; not exercised by these tests.)

- [ ] **Step 5: Commit**
```bash
git add src/nsbi_common_utils/lightning_tools/jet_encoder.py tests/test_jet_encoder.py
git commit -m "feat: jet encoder factory + StubJetEncoder (masked DeepSets)"
```

---

## Task 6: Hierarchical density-ratio LightningModule

**Files:**
- Create: `src/nsbi_common_utils/lightning_tools/hh_density_ratio_model.py`
- Test: `tests/test_hh_density_ratio_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hh_density_ratio_model.py`:
```python
import torch
from torch.utils.data import DataLoader
from nsbi_common_utils.lightning_tools.particle_cloud_dataset import WeightedParticleCloudDataset
from nsbi_common_utils.lightning_tools.hh_density_ratio_model import HHDensityRatioLightning


def _batch(synth_batch):
    ds = WeightedParticleCloudDataset(**synth_batch)
    return next(iter(DataLoader(ds, batch_size=8)))


def test_forward_returns_scores(synth_batch):
    model = HHDensityRatioLightning(encoder_kind="stub")
    b = _batch(synth_batch)
    out = model(b)
    assert out.shape == (8, 1)
    assert ((out >= 0) & (out <= 1)).all()  # sigmoid by default


def test_training_step_scalar_loss(synth_batch):
    model = HHDensityRatioLightning(encoder_kind="stub")
    loss = model.training_step(_batch(synth_batch), 0)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_label_flip_handles_negative_weights(synth_batch):
    # Negative weights must not produce nan/inf loss (NLO label-flip)
    sb = dict(synth_batch); sb["w"] = sb["w"].copy(); sb["w"][:4] = -1.0
    model = HHDensityRatioLightning(encoder_kind="stub")
    loss = model.training_step(_batch(sb), 0)
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_hh_density_ratio_model.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

Create `src/nsbi_common_utils/lightning_tools/hh_density_ratio_model.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from nsbi_common_utils.lightning_tools.cloud_spec import DEFAULT_SPEC
from nsbi_common_utils.lightning_tools.jet_encoder import build_jet_encoder
from nsbi_common_utils.lightning_tools.event_transformer import EventSetTransformer

N_OBJ_TYPES = 4  # tau_had, electron, muon, met


class HHDensityRatioLightning(pl.LightningModule):
    """Hierarchical constituent density-ratio model: per-jet encoder -> event transformer.

    forward() takes the dict batch produced by WeightedParticleCloudDataset and
    returns a per-event score in (0, 1) (or a raw logit if use_log_loss=True).

    NLO weights: events with w < 0 are handled by a label-flip (train on |w| with
    the label inverted), which keeps the weighted BCE finite and unbiased for
    signed-weight samples.
    """

    def __init__(self, encoder_kind="stub", spec=DEFAULT_SPEC,
                 n_heads=8, n_layers=2, learning_rate=1e-3,
                 use_log_loss=False, freeze_backbone=False,
                 callback_factor=0.5, callback_patience=20, encoder_kwargs=None):
        super().__init__()
        self.save_hyperparameters(ignore=["spec"])
        self.spec = spec
        self.lr = learning_rate
        self.use_log_loss = use_log_loss

        self.encoder = build_jet_encoder(
            kind=encoder_kind, f_part=spec.f_part, embed_dim=spec.embed_dim,
            **(encoder_kwargs or {}),
        )
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.obj_proj = nn.Linear(spec.f_obj - 1, spec.embed_dim)  # -1: type_id is embedded
        self.obj_type_emb = nn.Embedding(N_OBJ_TYPES, spec.embed_dim)
        self.jet_type_emb = nn.Parameter(torch.zeros(spec.embed_dim))
        self.event = EventSetTransformer(
            embed_dim=spec.embed_dim, n_heads=n_heads, n_layers=n_layers,
        )

    def _encode_jets(self, parts, part_mask, jet_mask):
        B, J, P, Fp = parts.shape
        flat = parts.reshape(B * J, P, Fp)
        flat_mask = part_mask.reshape(B * J, P)
        emb = self.encoder(flat, flat_mask).reshape(B, J, -1)   # (B, J, D)
        emb = emb + self.jet_type_emb
        return emb

    def _object_tokens(self, obj):
        feats, type_id = obj[..., :-1], obj[..., -1].long().clamp(0, N_OBJ_TYPES - 1)
        return self.obj_proj(feats) + self.obj_type_emb(type_id)  # (B, O, D)

    def forward(self, batch):
        jet_tok = self._encode_jets(batch["parts"], batch["part_mask"], batch["jet_mask"])
        obj_tok = self._object_tokens(batch["obj"])
        tokens = torch.cat([jet_tok, obj_tok], dim=1)
        mask = torch.cat([batch["jet_mask"], batch["obj_mask"]], dim=1)
        logit = self.event(tokens, mask)
        return logit if self.use_log_loss else torch.sigmoid(logit)

    def _loss(self, batch):
        out = self(batch)
        y = batch["y"].view(-1, 1)
        w = batch["w"].view(-1, 1)
        # NLO label-flip: train on |w|, invert label where w < 0.
        neg = (w < 0).float()
        y_eff = y * (1 - neg) + (1 - y) * neg
        w_eff = w.abs()
        if self.use_log_loss:
            loss = F.binary_cross_entropy_with_logits(out, y_eff, reduction="none")
        else:
            loss = F.binary_cross_entropy(out, y_eff, reduction="none")
        return (loss * w_eff).sum() / w_eff.sum().clamp_min(1e-12)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", self._loss(batch), prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.NAdam(
            [p for p in self.parameters() if p.requires_grad], lr=self.lr)
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=self.hparams.callback_patience, gamma=self.hparams.callback_factor)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_hh_density_ratio_model.py -v`
Expected: 3 passed.

- [ ] **Step 5: Add an overfit-tiny test (proves it learns)**

Append to `tests/test_hh_density_ratio_model.py`:
```python
def test_overfits_tiny_separable_set():
    import numpy as np
    from tests.conftest import make_event
    # Build a tiny set where label correlates with jet count -> learnable signal.
    parts, pm, jm, obj, om, y, w = [], [], [], [], [], [], []
    for i in range(48):
        lab = i % 2
        p, pmask, jmask, o, omask = make_event(i, n_jets=3 if lab else 1)
        parts.append(p); pm.append(pmask); jm.append(jmask); obj.append(o); om.append(omask)
        y.append(lab); w.append(1.0)
    batch = {
        "parts": torch.tensor(np.stack(parts)), "part_mask": torch.tensor(np.stack(pm)),
        "jet_mask": torch.tensor(np.stack(jm)), "obj": torch.tensor(np.stack(obj)),
        "obj_mask": torch.tensor(np.stack(om)),
        "y": torch.tensor(np.array(y, dtype="float32")),
        "w": torch.tensor(np.array(w, dtype="float32")),
    }
    model = HHDensityRatioLightning(encoder_kind="stub", learning_rate=1e-2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for step in range(150):
        opt.zero_grad(); loss = model._loss(batch); loss.backward(); opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < 0.6 * first  # loss dropped substantially
```

- [ ] **Step 6: Run + commit**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_hh_density_ratio_model.py -v`
Expected: 4 passed.
```bash
git add src/nsbi_common_utils/lightning_tools/hh_density_ratio_model.py tests/test_hh_density_ratio_model.py
git commit -m "feat: HHDensityRatioLightning hierarchical constituent density-ratio model"
```

---

## Task 7: Constituent ONNX export + inference

**Files:**
- Modify: `src/nsbi_common_utils/training/utils.py` (append functions)
- Modify: `src/nsbi_common_utils/training/__init__.py` (export)
- Test: `tests/test_constituent_onnx.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_constituent_onnx.py`:
```python
import numpy as np
import torch
from torch.utils.data import DataLoader
from nsbi_common_utils.lightning_tools.particle_cloud_dataset import WeightedParticleCloudDataset
from nsbi_common_utils.lightning_tools.hh_density_ratio_model import HHDensityRatioLightning
from nsbi_common_utils.training.utils import save_model_constituents, predict_with_onnx_constituents


def test_onnx_roundtrip(tmp_path, synth_batch):
    model = HHDensityRatioLightning(encoder_kind="stub").eval()
    ds = WeightedParticleCloudDataset(**synth_batch)
    sample = next(iter(DataLoader(ds, batch_size=4)))
    onnx_path = tmp_path / "m.onnx"
    save_model_constituents(model, sample, str(onnx_path))
    with torch.no_grad():
        torch_out = model(sample).numpy().reshape(-1)
    onnx_out = predict_with_onnx_constituents(
        {k: sample[k].numpy() for k in ("parts", "part_mask", "jet_mask", "obj", "obj_mask")},
        str(onnx_path))
    assert onnx_out.shape == (4,)
    assert np.allclose(torch_out, onnx_out, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_constituent_onnx.py -v`
Expected: FAIL (functions not defined).

- [ ] **Step 3: Implement (append to `training/utils.py`)**

Append to `src/nsbi_common_utils/training/utils.py`:
```python
_CONSTITUENT_INPUTS = ["parts", "part_mask", "jet_mask", "obj", "obj_mask"]


class _DictToTupleWrapper(torch.nn.Module):
    """Wrap a dict-input model so torch.onnx.export can trace positional args."""

    def __init__(self, model, keys):
        super().__init__()
        self.model = model
        self.keys = keys

    def forward(self, parts, part_mask, jet_mask, obj, obj_mask):
        return self.model({
            "parts": parts, "part_mask": part_mask, "jet_mask": jet_mask,
            "obj": obj, "obj_mask": obj_mask,
        })


def save_model_constituents(lightning_model, sample_batch, path_to_save_model, opset=17):
    """Export a constituent (dict-input) density-ratio model to ONNX.

    sample_batch: a dict of torch tensors (one mini-batch) used to trace shapes.
    Dynamic axis 0 (batch) is set on every input and the output.
    """
    lightning_model.eval()
    wrapper = _DictToTupleWrapper(lightning_model, _CONSTITUENT_INPUTS).eval()
    args = tuple(sample_batch[k] for k in _CONSTITUENT_INPUTS)
    dynamic = {k: {0: "batch"} for k in _CONSTITUENT_INPUTS}
    dynamic["output"] = {0: "batch"}
    torch.onnx.export(
        wrapper, args, str(path_to_save_model), export_params=True, opset_version=opset,
        input_names=_CONSTITUENT_INPUTS, output_names=["output"], dynamic_axes=dynamic,
    )


def predict_with_onnx_constituents(arrays, model, batch_size=4096):
    """Batched ONNX inference for constituent models.

    arrays: dict with the 5 constituent inputs as numpy arrays (leading event axis).
    model: path to .onnx, an onnx.ModelProto, or an onnxruntime.InferenceSession.
    Returns a (N,) float32 array of per-event scores.
    """
    import onnxruntime as rt
    if isinstance(model, str):
        sess = rt.InferenceSession(model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    elif isinstance(model, onnx.ModelProto):
        sess = rt.InferenceSession(model.SerializeToString(),
                                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    else:
        sess = model
    n = len(arrays["parts"])
    out_name = sess.get_outputs()[0].name
    preds = np.empty((n,), dtype=np.float32)
    for i in range(0, n, batch_size):
        sl = slice(i, min(i + batch_size, n))
        feed = {k: np.asarray(arrays[k][sl], dtype=np.float32) for k in _CONSTITUENT_INPUTS}
        preds[sl] = sess.run([out_name], feed)[0].reshape(-1)
    return preds
```

- [ ] **Step 4: Export the new functions**

In `src/nsbi_common_utils/training/__init__.py`, add to the imports from `.utils` and to `__all__`:
```python
from nsbi_common_utils.training.utils import (
    save_model_constituents,
    predict_with_onnx_constituents,
)
```
(add the two names to the `__all__` list as well.)

- [ ] **Step 5: Run test to verify it passes**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_constituent_onnx.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**
```bash
git add src/nsbi_common_utils/training/utils.py src/nsbi_common_utils/training/__init__.py tests/test_constituent_onnx.py
git commit -m "feat: constituent ONNX export + batched onnxruntime inference"
```

---

## Task 8: particle_density_ratio_trainer

**Files:**
- Create: `src/nsbi_common_utils/training/particle_ratio_estimation.py`
- Modify: `src/nsbi_common_utils/training/__init__.py`, `lightning_tools/__init__.py` (exports)
- Test: `tests/test_particle_ratio_estimation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_particle_ratio_estimation.py`:
```python
import os
import numpy as np
from nsbi_common_utils.training.particle_ratio_estimation import particle_density_ratio_trainer


def test_trainer_runs_and_exports(tmp_path, synth_batch):
    tr = particle_density_ratio_trainer(
        clouds=synth_batch,
        sample_name=["signal", "background"],
        output_name="signal_vs_background",
        path_to_models=str(tmp_path) + "/",
        encoder_kind="stub",
    )
    history = tr.train(number_of_epochs=2, batch_size=8, learning_rate=1e-3, holdout_split=0.25)
    assert os.path.exists(f"{tmp_path}/model0.onnx")
    assert "train_loss" in history and len(history["train_loss"]) == 2
    ratios = tr.evaluate_ratios(synth_batch)
    assert ratios.shape == (synth_batch["y"].shape[0],) and np.all(ratios > 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_particle_ratio_estimation.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

Create `src/nsbi_common_utils/training/particle_ratio_estimation.py`:
```python
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from nsbi_common_utils.lightning_tools.particle_cloud_dataset import WeightedParticleCloudDataset
from nsbi_common_utils.lightning_tools.hh_density_ratio_model import HHDensityRatioLightning
from nsbi_common_utils.lightning_tools.callbacks import LossHistory
from nsbi_common_utils.lightning_tools.cloud_spec import DEFAULT_SPEC
from nsbi_common_utils.training.utils import (
    save_model_constituents, predict_with_onnx_constituents, convert_score_to_ratio,
)

_KEYS = ["parts", "part_mask", "jet_mask", "obj", "obj_mask", "y", "w"]


class particle_density_ratio_trainer:
    """Constituent-cloud analog of density_ratio_trainer.

    Trains an HHDensityRatioLightning to estimate p_A/p_B from per-jet constituent
    clouds, exports to ONNX, and evaluates per-event density ratios. Mirrors the
    public surface used by the example scripts (train / evaluate_ratios).
    """

    def __init__(self, clouds, sample_name, output_name, path_to_models,
                 encoder_kind="stub", spec=DEFAULT_SPEC, freeze_backbone=False,
                 use_log_loss=False, encoder_kwargs=None):
        self.clouds = {k: np.asarray(clouds[k]) for k in _KEYS}
        self.sample_name = sample_name
        self.output_name = output_name
        self.path_to_models = path_to_models
        self.encoder_kind = encoder_kind
        self.spec = spec
        self.freeze_backbone = freeze_backbone
        self.use_log_loss = use_log_loss
        self.encoder_kwargs = encoder_kwargs or {}
        os.makedirs(path_to_models, exist_ok=True)
        self.model = None

    def train(self, number_of_epochs, batch_size, learning_rate,
              holdout_split=0.3, ensemble_index=0, n_heads=8, n_layers=2):
        ds = WeightedParticleCloudDataset(**{k: self.clouds[k] for k in _KEYS})
        n_hold = max(1, int(len(ds) * holdout_split))
        n_train = len(ds) - n_hold
        gen = torch.Generator().manual_seed(1000 + ensemble_index)
        train_ds, _ = random_split(ds, [n_train, n_hold], generator=gen)
        n_val = max(1, int(n_train * 0.1))
        train_ds, val_ds = random_split(
            train_ds, [n_train - n_val, n_val], generator=gen)

        self.model = HHDensityRatioLightning(
            encoder_kind=self.encoder_kind, spec=self.spec, n_heads=n_heads,
            n_layers=n_layers, learning_rate=learning_rate,
            use_log_loss=self.use_log_loss, freeze_backbone=self.freeze_backbone,
            encoder_kwargs=self.encoder_kwargs)

        history = LossHistory()
        trainer = pl.Trainer(
            max_epochs=number_of_epochs, accelerator="auto", devices="auto",
            logger=False, enable_checkpointing=False,
            callbacks=[history, EarlyStopping(monitor="val_loss", patience=number_of_epochs)],
            enable_progress_bar=False)
        trainer.fit(
            self.model,
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size))

        self.model.eval()
        sample = next(iter(DataLoader(ds, batch_size=min(batch_size, len(ds)))))
        save_model_constituents(self.model, sample, f"{self.path_to_models}model{ensemble_index}.onnx")
        # LossHistory stores per-epoch train/val loss lists (callbacks.py)
        return {"train_loss": list(history.train_loss), "val_loss": list(history.val_loss)}

    def evaluate_ratios(self, clouds, ensemble_index=0):
        arrays = {k: np.asarray(clouds[k]) for k in ("parts", "part_mask", "jet_mask", "obj", "obj_mask")}
        score = predict_with_onnx_constituents(arrays, f"{self.path_to_models}model{ensemble_index}.onnx")
        if self.use_log_loss:
            score = 1.0 / (1.0 + np.exp(-score))
        score = np.clip(score, 1e-9, 1.0 - 1e-9)
        return convert_score_to_ratio(score)
```

> **Note (verified):** `LossHistory` in `lightning_tools/callbacks.py` exposes
> `.train_loss` / `.val_loss` lists appended per epoch — the `train()` references and
> the test match it.

- [ ] **Step 4: Export the trainer**

In `src/nsbi_common_utils/training/__init__.py` add:
```python
from nsbi_common_utils.training.particle_ratio_estimation import particle_density_ratio_trainer
```
(and add `"particle_density_ratio_trainer"` to `__all__`). In `lightning_tools/__init__.py` add `HHDensityRatioLightning`, `WeightedParticleCloudDataset`, `EventSetTransformer`, `StubJetEncoder`, `build_jet_encoder`, `CloudSpec`, `DEFAULT_SPEC` to imports + `__all__`.

- [ ] **Step 5: Run test to verify it passes**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_particle_ratio_estimation.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**
```bash
git add src/nsbi_common_utils/training/particle_ratio_estimation.py src/nsbi_common_utils/training/__init__.py src/nsbi_common_utils/lightning_tools/__init__.py tests/test_particle_ratio_estimation.py
git commit -m "feat: particle_density_ratio_trainer (constituent-cloud trainer)"
```

---

## Task 9: eval_to_ratios — per-event ratios on the Asimov ordering

**Files:**
- Create: `examples/HH_bbtautau_kappalambda_sophon/scripts/eval_to_ratios.py`
- Test: `tests/test_eval_to_ratios.py`

This is the contract surface for the JAX fit: it must write `ratio_<process>.npy` and the Asimov `weights.npy` on **one shared event ordering** (concatenation of basis-process events in a fixed order).

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_to_ratios.py`:
```python
import numpy as np
from torch.utils.data import DataLoader
from nsbi_common_utils.lightning_tools.particle_cloud_dataset import WeightedParticleCloudDataset
from nsbi_common_utils.lightning_tools.hh_density_ratio_model import HHDensityRatioLightning
from nsbi_common_utils.training.utils import save_model_constituents
from examples_pkg.eval_to_ratios import build_asimov, eval_process_ratio


def test_asimov_ordering_and_ratio(tmp_path, synth_batch):
    # two "processes": split synth events in half, fixed order
    n = synth_batch["y"].shape[0]
    procs = {
        "sig": {k: v[: n // 2] for k, v in synth_batch.items()},
        "bkg": {k: v[n // 2:] for k, v in synth_batch.items()},
    }
    asimov, weights = build_asimov(procs, order=["sig", "bkg"])
    assert weights.shape == (n,)
    assert asimov["parts"].shape[0] == n
    # export a model and evaluate one process ratio on the Asimov set
    model = HHDensityRatioLightning(encoder_kind="stub").eval()
    sample = next(iter(DataLoader(WeightedParticleCloudDataset(**procs["sig"]), batch_size=4)))
    onnx_path = tmp_path / "sig.onnx"
    save_model_constituents(model, sample, str(onnx_path))
    r = eval_process_ratio(asimov, str(onnx_path))
    assert r.shape == (n,) and np.all(r > 0)
```

(The test imports `examples_pkg.eval_to_ratios`; add a `conftest` path shim so the example scripts import cleanly — see Step 3.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_eval_to_ratios.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement + path shim**

Create `examples/HH_bbtautau_kappalambda_sophon/scripts/eval_to_ratios.py`:
```python
"""Constituent analog of data_nn_eval.py: build the Asimov event set, run ONNX
inference per basis process, convert score->ratio, aggregate ensembles, and save
ratio_<process>.npy + the Asimov weights.npy on the SHARED event ordering that the
JAX fit (sbi_parametric_model) loads."""
import numpy as np
from nsbi_common_utils.training.utils import predict_with_onnx_constituents, convert_score_to_ratio

_CLOUD_KEYS = ["parts", "part_mask", "jet_mask", "obj", "obj_mask"]


def build_asimov(processes, order):
    """Concatenate per-process clouds in a fixed order -> (asimov_clouds, weights)."""
    asimov = {k: np.concatenate([processes[p][k] for p in order], axis=0) for k in _CLOUD_KEYS}
    weights = np.concatenate([processes[p]["w"] for p in order], axis=0).astype(np.float64)
    return asimov, weights


def eval_process_ratio(asimov_clouds, onnx_paths, aggregation="mean_ratio", use_log_loss=False):
    """Evaluate one process's density ratio on the Asimov set, ensemble-aggregated.

    onnx_paths: str (single member) or list[str] (ensemble).
    """
    if isinstance(onnx_paths, str):
        onnx_paths = [onnx_paths]
    ratios = []
    for p in onnx_paths:
        score = predict_with_onnx_constituents(asimov_clouds, p)
        if use_log_loss:
            score = 1.0 / (1.0 + np.exp(-score))
        score = np.clip(score, 1e-9, 1.0 - 1e-9)
        ratios.append(convert_score_to_ratio(score))
    ratios = np.stack(ratios, axis=0)
    if aggregation == "median_ratio":
        return np.median(ratios, axis=0)
    return np.mean(ratios, axis=0)
```

Create `examples/HH_bbtautau_kappalambda_sophon/scripts/__init__.py` (empty) and add to `tests/conftest.py`:
```python
import sys, pathlib
_EX = pathlib.Path(__file__).resolve().parents[1] / "examples" / "HH_bbtautau_kappalambda_sophon" / "scripts"
sys.path.insert(0, str(_EX))
import importlib  # noqa: E402
# expose as examples_pkg.* for stable test imports
import types  # noqa: E402
examples_pkg = types.ModuleType("examples_pkg")
examples_pkg.eval_to_ratios = importlib.import_module("eval_to_ratios")
sys.modules["examples_pkg"] = examples_pkg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_eval_to_ratios.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**
```bash
git add examples/HH_bbtautau_kappalambda_sophon/scripts/ tests/test_eval_to_ratios.py tests/conftest.py
git commit -m "feat: eval_to_ratios — Asimov-ordered per-event ratios for the JAX fit"
```

---

## Task 10: Delphes → constituent-cloud converter

**Files:**
- Create: `examples/HH_bbtautau_kappalambda_sophon/scripts/delphes_to_clouds.py`
- Test: `tests/test_delphes_to_clouds.py`

The converter reads the Delphes full-output tree, associates EFlow constituents to AK4 jets, and emits the padded cloud arrays + object tokens + weights as a compressed `.npz`. The test builds a tiny synthetic Delphes-like ROOT with uproot so no real sample is needed.

- [ ] **Step 1: Write the failing test**

Create `tests/test_delphes_to_clouds.py`:
```python
import numpy as np
import awkward as ak
import uproot
from examples_pkg_delphes import convert_tree


def _write_fake_delphes(path):
    # 3 events; jagged jet + constituent structure (minimal Delphes-like schema)
    n_ev = 3
    jet_pt = ak.Array([[120.0, 80.0], [60.0], [200.0, 50.0, 40.0]])
    jet_eta = ak.Array([[0.1, -0.5], [1.0], [0.2, 0.3, -1.1]])
    jet_phi = ak.Array([[0.2, 1.1], [-0.7], [0.0, 2.0, -2.5]])
    # one constituent list per event (flat), with a jet index per constituent
    part_pt = ak.Array([[30.0, 20.0, 10.0], [15.0], [50.0, 25.0]])
    part_eta = ak.Array([[0.1, 0.12, -0.5], [1.0], [0.2, 0.25]])
    part_phi = ak.Array([[0.2, 0.22, 1.1], [-0.7], [0.0, 0.05]])
    part_jetidx = ak.Array([[0, 0, 1], [0], [0, 0]])
    part_charge = ak.Array([[1, -1, 0], [1], [0, -1]])
    met = ak.Array([40.0, 25.0, 90.0])
    met_phi = ak.Array([0.5, -1.2, 2.0])
    weight = ak.Array([1.0, -1.0, 1.0])
    with uproot.recreate(path) as f:
        f["Delphes"] = {
            "Jet.PT": jet_pt, "Jet.Eta": jet_eta, "Jet.Phi": jet_phi,
            "EFlow.PT": part_pt, "EFlow.Eta": part_eta, "EFlow.Phi": part_phi,
            "EFlow.JetIndex": part_jetidx, "EFlow.Charge": part_charge,
            "MissingET.MET": met, "MissingET.Phi": met_phi, "Event.Weight": weight,
        }


def test_convert_tiny_delphes(tmp_path):
    root = tmp_path / "fake.root"
    _write_fake_delphes(str(root))
    out = tmp_path / "clouds.npz"
    convert_tree(str(root), "Delphes", str(out))
    d = np.load(out)
    assert d["parts"].shape == (3, 4, 64, 8)
    assert d["jet_mask"].sum() == 2 + 1 + 3  # jets per event
    assert d["w"].shape == (3,) and d["w"][1] == -1.0  # NLO sign preserved
```

Add to `tests/conftest.py` (extend the example shim):
```python
examples_pkg_delphes = importlib.import_module("delphes_to_clouds")
sys.modules["examples_pkg_delphes"] = examples_pkg_delphes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_delphes_to_clouds.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

Create `examples/HH_bbtautau_kappalambda_sophon/scripts/delphes_to_clouds.py`:
```python
"""Delphes full-output ROOT -> padded per-jet constituent clouds + object tokens.

Reads the Delphes tree (Jet/EFlow*/MissingET/Electron/Muon/Event.Weight), assigns
EFlow constituents to AK4 jets, builds the per-jet padded cloud and per-event object
tokens (taus/leptons/MET), and writes a compressed .npz with arrays matching
WeightedParticleCloudDataset. Feature columns follow cloud_spec (F_PART=8, F_OBJ=6);
the sophon-ak4 integration (Task 11) extends F_PART to the checkpoint's full schema.

This baseline maps EFlow.JetIndex -> jet; for real Delphes without an explicit
constituent->jet index, replace assign_constituents() with a Delta-R association.
"""
import argparse
import numpy as np
import awkward as ak
import uproot

N_JETS_MAX, N_PART_MAX, F_PART, N_OBJ_MAX, F_OBJ = 4, 64, 8, 6, 6


def _logpt(pt):
    return np.log(np.clip(pt, 1e-3, None)).astype(np.float32)


def convert_tree(path, tree_name, out_path):
    arr = uproot.open(f"{path}:{tree_name}").arrays(library="ak")
    n = len(arr["Jet.PT"])
    parts = np.zeros((n, N_JETS_MAX, N_PART_MAX, F_PART), dtype=np.float32)
    part_mask = np.zeros((n, N_JETS_MAX, N_PART_MAX), dtype=np.float32)
    jet_mask = np.zeros((n, N_JETS_MAX), dtype=np.float32)
    obj = np.zeros((n, N_OBJ_MAX, F_OBJ), dtype=np.float32)
    obj_mask = np.zeros((n, N_OBJ_MAX), dtype=np.float32)

    for i in range(n):
        njet = min(len(arr["Jet.PT"][i]), N_JETS_MAX)
        jet_eta = ak.to_numpy(arr["Jet.Eta"][i]); jet_phi = ak.to_numpy(arr["Jet.Phi"][i])
        for j in range(njet):
            jet_mask[i, j] = 1.0
        # assign constituents to jets via EFlow.JetIndex
        cp = ak.to_numpy(arr["EFlow.PT"][i]); ce = ak.to_numpy(arr["EFlow.Eta"][i])
        cph = ak.to_numpy(arr["EFlow.Phi"][i]); cj = ak.to_numpy(arr["EFlow.JetIndex"][i])
        cc = ak.to_numpy(arr["EFlow.Charge"][i])
        counts = np.zeros(N_JETS_MAX, dtype=int)
        for k in range(len(cp)):
            j = int(cj[k])
            if j < 0 or j >= njet or counts[j] >= N_PART_MAX:
                continue
            p = counts[j]; counts[j] += 1
            part_mask[i, j, p] = 1.0
            deta = ce[k] - jet_eta[j]
            dphi = np.arctan2(np.sin(cph[k] - jet_phi[j]), np.cos(cph[k] - jet_phi[j]))
            parts[i, j, p] = [_logpt(cp[k]), deta, dphi, cc[k], 0.0, 0.0, 0.0,
                              1.0 if cc[k] != 0 else 0.0]
        # MET object token (type_id=3)
        obj_mask[i, 0] = 1.0
        met = float(arr["MissingET.MET"][i][0]) if len(arr["MissingET.MET"][i]) else 0.0
        mphi = float(arr["MissingET.Phi"][i][0]) if len(arr["MissingET.Phi"][i]) else 0.0
        obj[i, 0] = [_logpt(met), 0.0, np.sin(mphi), np.cos(mphi), met, 3.0]

    w = ak.to_numpy(arr["Event.Weight"]).astype(np.float64)
    np.savez_compressed(out_path, parts=parts, part_mask=part_mask, jet_mask=jet_mask,
                        obj=obj, obj_mask=obj_mask, w=w)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tree", default="Delphes")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    convert_tree(args.input, args.tree, args.output)
```

> **Integration note (real Delphes):** the synthetic schema (`EFlow.JetIndex`, flat `Jet.*`) is a simplification. Against real Delphes output, map the actual branch names from the card (`EFlowTrack/EFlowPhoton/EFlowNeutralHadron`, `Electron`, `Muon`, `Jet.Flavor`), associate constituents by Delta-R to AK4 jets if no index branch exists, add τ_had/electron/muon object tokens (type_id 0/1/2), and extend `F_PART` to the sophon-ak4 schema (add `d0/dz(+err)`, full PID one-hots). Add b-tag info via `Jet.Flavor` if used downstream. Validate on one real file and assert non-empty jets/constituents before bulk conversion.

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_delphes_to_clouds.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**
```bash
git add examples/HH_bbtautau_kappalambda_sophon/scripts/delphes_to_clouds.py tests/test_delphes_to_clouds.py tests/conftest.py
git commit -m "feat: delphes_to_clouds converter (EFlow constituents -> padded clouds)"
```

---

## Task 11: sophon-ak4 backbone integration (real checkpoint + from-scratch control)

**Files:**
- Create: `src/nsbi_common_utils/lightning_tools/sophon_ak4_backbone.py`
- Modify: `src/nsbi_common_utils/lightning_tools/_part_vendor/ParT.py` (vendored, torch-only)
- Test: `tests/test_sophon_ak4_backbone.py` (marked integration; skipped without the checkpoint)

This is the one task that needs external resources. It provides `build_part_encoder(kind="scratch"|"sophon-ak4", ...)` returning an `nn.Module` with the **same `forward(parts, mask) -> (B, embed_dim)` interface** as `StubJetEncoder`, so everything above is unchanged.

- [ ] **Step 1: Vendor ParT (do NOT add weaver-core as a dependency)**

`weaver-core` pins `uproot<5.2`, which conflicts with this toolkit's `uproot 5.7` — it
will break `pixi install`. We only need the ParT `nn.Module`, which is torch-only.
Vendor it directly:
```bash
git clone https://github.com/hqucms/weaver-core /tmp/weaver-core
cp /tmp/weaver-core/weaver/nn/model/ParticleTransformer.py \
   src/nsbi_common_utils/lightning_tools/_part_vendor/ParT.py
# then replace `from weaver.utils.logger import _logger` with:
#   import logging; _logger = logging.getLogger("ParT")
```
Keep the MIT license header. No `pixi.toml` change is needed (see `_part_vendor/README.md`).

- [ ] **Step 2: Write the integration test (auto-skips without checkpoint)**

Create `tests/test_sophon_ak4_backbone.py`:
```python
import os
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.environ.get("SOPHON_AK4_CKPT"),
    reason="set SOPHON_AK4_CKPT to the downloaded sophon-ak4 checkpoint to run")


def test_scratch_encoder_interface():
    from nsbi_common_utils.lightning_tools.jet_encoder import build_jet_encoder
    enc = build_jet_encoder(kind="scratch", f_part=8, embed_dim=64)
    out = enc(torch.randn(3, 64, 8), torch.ones(3, 64))
    assert out.shape == (3, 64)


def test_sophon_ak4_loads_and_runs():
    from nsbi_common_utils.lightning_tools.jet_encoder import build_jet_encoder
    enc = build_jet_encoder(kind="sophon-ak4", f_part=8, embed_dim=64,
                            checkpoint=os.environ["SOPHON_AK4_CKPT"])
    out = enc(torch.randn(2, 64, 8), torch.ones(2, 64))
    assert out.shape == (2, 64)
```

- [ ] **Step 3: Implement the backbone wrapper**

Create `src/nsbi_common_utils/lightning_tools/sophon_ak4_backbone.py`:
```python
"""sophon-ak4 (ParT) per-jet encoder wrapper.

Wraps the Particle Transformer so it exposes the same interface as StubJetEncoder:
forward(parts: (B, n_part, f_part), mask: (B, n_part)) -> (B, embed_dim).

- kind="scratch": ParT of the sophon-ak4 shape (6 particle + 2 class attn blocks,
  embed 64, 8 heads), random init -> the from-scratch control.
- kind="sophon-ak4": same architecture, weights loaded from the released checkpoint
  (download via huggingface_hub: repo 'jet-universe/sophon-ak4'); the 23-class head
  is dropped and the class-token embedding (dim 64) is used as the jet embedding.

ParT expects features (x) and 4-vectors (v) for pairwise interactions; here we pass
the per-constituent feature block as x and reuse (logpt, deta, dphi) for v. The
exact feature ordering/normalization MUST match the checkpoint's training config
(see the sophon repo's data config); adapt _split_features accordingly.
"""
import torch
import torch.nn as nn


def _load_part(embed_dim, f_part):
    # from the vendored ParT (MIT). Config mirrors sophon-ak4: 6 + 2 blocks, 8 heads.
    from nsbi_common_utils.lightning_tools._part_vendor.ParT import ParticleTransformer
    return ParticleTransformer(
        input_dim=f_part, num_classes=23, embed_dims=[embed_dim, embed_dim, embed_dim],
        num_heads=8, num_layers=6, num_cls_layers=2, fc_params=[], use_amp=False)


class SophonAK4Encoder(nn.Module):
    def __init__(self, f_part=8, embed_dim=64, checkpoint=None):
        super().__init__()
        self.part = _load_part(embed_dim, f_part)
        self.embed_dim = embed_dim
        if checkpoint is not None:
            from huggingface_hub import hf_hub_download
            ckpt = checkpoint
            if not checkpoint.endswith((".pt", ".onnx")):
                ckpt = hf_hub_download(repo_id="jet-universe/sophon-ak4", filename=checkpoint)
            state = torch.load(ckpt, map_location="cpu")
            self.part.load_state_dict(state.get("model", state), strict=False)

    def forward(self, parts, mask):
        # ParT signature: (x=(B,F,P), v=(B,4,P), mask=(B,1,P)); return class-token embedding.
        x = parts.transpose(1, 2)                       # (B, F, P)
        v = x[:, :3, :]                                  # reuse (logpt,deta,dphi) as 4-vec proxy
        m = mask.unsqueeze(1)                            # (B, 1, P)
        emb = self.part(x, v, m, return_embed=True)      # (B, embed_dim) class token
        return emb


def build_part_encoder(kind, f_part=8, embed_dim=64, checkpoint=None, **_):
    if kind == "scratch":
        return SophonAK4Encoder(f_part=f_part, embed_dim=embed_dim, checkpoint=None)
    if kind == "sophon-ak4":
        return SophonAK4Encoder(f_part=f_part, embed_dim=embed_dim,
                                checkpoint=checkpoint or "PARTAK4.pt")
    raise ValueError(kind)
```

> **Implementer must verify against the sophon repo:** (a) the exact `ParticleTransformer` constructor kwargs and whether it supports a `return_embed` path (if not, take the pre-final-FC class token); (b) the checkpoint filename in the HF repo and the `state_dict` key layout; (c) the exact input feature set/order and normalization used at training time, and align `delphes_to_clouds.py`'s `F_PART` columns to it. The `scratch` control must use the identical architecture so the only difference vs `sophon-ak4` is the initial weights.

- [ ] **Step 4: Run the scratch path (no checkpoint needed)**

Run: `pixi run -e nsbi-env-gpu pytest "tests/test_sophon_ak4_backbone.py::test_scratch_encoder_interface" -v`
Expected: PASS (or SKIP if you gate the whole module; if so, temporarily unset the skip for this one). The `sophon-ak4` test runs once `SOPHON_AK4_CKPT` is set on Perlmutter.

- [ ] **Step 5: Commit**
```bash
git add src/nsbi_common_utils/lightning_tools/sophon_ak4_backbone.py src/nsbi_common_utils/lightning_tools/_part_vendor/ pixi.toml tests/test_sophon_ak4_backbone.py
git commit -m "feat: sophon-ak4 ParT encoder wrapper + from-scratch control"
```

---

## Task 12: Comparison harness (the 3 controls + low-stat ablation)

**Files:**
- Create: `examples/HH_bbtautau_kappalambda_sophon/compare.py`
- Test: `tests/test_compare.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_compare.py`:
```python
from examples_pkg_compare import run_controls, low_stat_ablation


def test_run_controls_returns_histories(tmp_path, synth_batch):
    res = run_controls(synth_batch, synth_batch, out_dir=str(tmp_path),
                       controls=["stub_scratch", "stub_frozen"], epochs=2, batch_size=8)
    assert set(res) == {"stub_scratch", "stub_frozen"}
    assert all("train_loss" in v and len(v["train_loss"]) == 2 for v in res.values())


def test_low_stat_ablation_shape(tmp_path, synth_batch):
    curve = low_stat_ablation(synth_batch, synth_batch, out_dir=str(tmp_path),
                              fractions=[1.0, 0.5], epochs=1, batch_size=8)
    assert sorted(curve) == [0.5, 1.0]
    assert all("val_loss_final" in curve[f] for f in curve)
```

Add to `tests/conftest.py` shim:
```python
examples_pkg_compare = importlib.import_module("compare")
sys.modules["examples_pkg_compare"] = examples_pkg_compare
```
(`compare.py` lives in the example root; extend the shim path to include it, or place `compare.py` under `scripts/` for a single shim path.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_compare.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

Create `examples/HH_bbtautau_kappalambda_sophon/compare.py`:
```python
"""Diagnostics harness: train the signal-vs-background density ratio under several
controls and compare convergence + data-efficiency.

Controls (key -> (encoder_kind, freeze)):
  stub_scratch   : StubJetEncoder, trainable     (no-foundation baseline, fast/CPU)
  stub_frozen    : StubJetEncoder, frozen        (sanity)
  scratch        : ParT random init, trainable   (architecture control)
  sophon_frozen  : sophon-ak4, frozen            (foundation, frozen)
  sophon_finetune: sophon-ak4, fine-tuned        (foundation, fine-tuned)
On CPU/CI use the stub_* controls; the real comparison uses scratch + sophon_*.
"""
import numpy as np
from nsbi_common_utils.training.particle_ratio_estimation import particle_density_ratio_trainer

_CONTROL_MAP = {
    "stub_scratch": ("stub", False),
    "stub_frozen": ("stub", True),
    "scratch": ("scratch", False),
    "sophon_frozen": ("sophon-ak4", True),
    "sophon_finetune": ("sophon-ak4", False),
}


def _train_one(clouds, kind, freeze, out_dir, epochs, batch_size, lr=1e-3, encoder_kwargs=None):
    tr = particle_density_ratio_trainer(
        clouds=clouds, sample_name=["signal", "background"],
        output_name=f"sig_vs_bkg_{kind}_{'frozen' if freeze else 'ft'}",
        path_to_models=f"{out_dir}/{kind}_{'frozen' if freeze else 'ft'}/",
        encoder_kind=kind, freeze_backbone=freeze, encoder_kwargs=encoder_kwargs)
    return tr.train(number_of_epochs=epochs, batch_size=batch_size, learning_rate=lr)


def run_controls(train_clouds, val_clouds, out_dir, controls, epochs=50, batch_size=512):
    results = {}
    for c in controls:
        kind, freeze = _CONTROL_MAP[c]
        results[c] = _train_one(train_clouds, kind, freeze, out_dir, epochs, batch_size)
    return results


def _subsample(clouds, frac, seed=0):
    n = clouds["y"].shape[0]
    k = max(2, int(n * frac))
    idx = np.random.default_rng(seed).choice(n, size=k, replace=False)
    return {key: np.asarray(v)[idx] for key, v in clouds.items()}


def low_stat_ablation(train_clouds, val_clouds, out_dir, fractions,
                      control="stub_scratch", epochs=50, batch_size=512):
    kind, freeze = _CONTROL_MAP[control]
    curve = {}
    for f in fractions:
        sub = _subsample(train_clouds, f)
        hist = _train_one(sub, kind, freeze, f"{out_dir}/frac_{f}", epochs, batch_size)
        curve[f] = {"val_loss_final": hist["val_loss"][-1] if hist["val_loss"] else float("nan")}
    return curve
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_compare.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**
```bash
git add examples/HH_bbtautau_kappalambda_sophon/compare.py tests/test_compare.py tests/conftest.py
git commit -m "feat: comparison harness (controls + low-stat ablation)"
```

---

## Task 13: End-to-end smoke test + example config + docs

**Files:**
- Create: `examples/HH_bbtautau_kappalambda_sophon/smoke_test.py`
- Create: `examples/HH_bbtautau_kappalambda_sophon/config_train.yml`
- Create: `examples/HH_bbtautau_kappalambda_sophon/README.md`
- Create: `docs/basics/sophon_hh_density_ratio.rst`
- Test: `tests/test_smoke_end_to_end.py`

- [ ] **Step 1: Write the failing end-to-end test**

Create `tests/test_smoke_end_to_end.py`:
```python
import numpy as np
from examples_pkg_smoke import run_smoke


def test_end_to_end(tmp_path, synth_batch):
    result = run_smoke(synth_batch, out_dir=str(tmp_path))
    assert result["onnx_exists"]
    assert result["ratios"].shape == (synth_batch["y"].shape[0],)
    assert np.all(result["ratios"] > 0)
```

Add the shim line for `smoke_test` to `tests/conftest.py` (as in Task 9/12).

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_smoke_end_to_end.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement the smoke runner**

Create `examples/HH_bbtautau_kappalambda_sophon/smoke_test.py`:
```python
"""Tiny end-to-end check: synthetic clouds -> train 2 epochs -> ONNX -> ratios.
Run directly (`python smoke_test.py`) or via pytest (tests/test_smoke_end_to_end.py)."""
import os
import numpy as np
from nsbi_common_utils.training.particle_ratio_estimation import particle_density_ratio_trainer


def _synthetic(n=64, seed=0):
    from cloud_spec_shapes import N_JETS_MAX, N_PART_MAX, F_PART, N_OBJ_MAX, F_OBJ  # see note
    r = np.random.default_rng(seed)
    parts = r.normal(size=(n, N_JETS_MAX, N_PART_MAX, F_PART)).astype("float32")
    part_mask = (r.random((n, N_JETS_MAX, N_PART_MAX)) > 0.5).astype("float32")
    jet_mask = (r.random((n, N_JETS_MAX)) > 0.3).astype("float32"); jet_mask[:, 0] = 1.0
    obj = r.normal(size=(n, N_OBJ_MAX, F_OBJ)).astype("float32")
    obj_mask = (r.random((n, N_OBJ_MAX)) > 0.5).astype("float32"); obj_mask[:, 0] = 1.0
    y = (np.arange(n) % 2).astype("float32"); w = np.ones(n, dtype="float32")
    return dict(parts=parts, part_mask=part_mask, jet_mask=jet_mask, obj=obj,
               obj_mask=obj_mask, y=y, w=w)


def run_smoke(clouds=None, out_dir="/tmp/sophon_smoke"):
    os.makedirs(out_dir, exist_ok=True)
    clouds = clouds or _synthetic()
    tr = particle_density_ratio_trainer(
        clouds=clouds, sample_name=["signal", "background"],
        output_name="smoke", path_to_models=out_dir + "/", encoder_kind="stub")
    tr.train(number_of_epochs=2, batch_size=16, learning_rate=1e-3)
    ratios = tr.evaluate_ratios(clouds)
    return {"onnx_exists": os.path.exists(f"{out_dir}/model0.onnx"), "ratios": ratios}


if __name__ == "__main__":
    print(run_smoke())
```

> **Note:** for the standalone `_synthetic()` import, either import the constants from `nsbi_common_utils.lightning_tools.cloud_spec` (`DEFAULT_SPEC`) instead of a local module, or hardcode `(4,64,8,6,6)`. Prefer importing `DEFAULT_SPEC` to stay DRY:
> ```python
> from nsbi_common_utils.lightning_tools.cloud_spec import DEFAULT_SPEC as S
> N_JETS_MAX, N_PART_MAX, F_PART, N_OBJ_MAX, F_OBJ = S.n_jets_max, S.n_part_max, S.f_part, S.n_obj_max, S.f_obj
> ```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e nsbi-env-gpu pytest tests/test_smoke_end_to_end.py -v`
Expected: 1 passed.

- [ ] **Step 5: Write the example config**

Create `examples/HH_bbtautau_kappalambda_sophon/config_train.yml`:
```yaml
# Training config for the Sophon HH->bbtautau density-ratio (Phase 1).
cloud:
  n_jets_max: 4
  n_part_max: 64
  f_part: 8         # extend to the sophon-ak4 schema for the real encoder
  n_obj_max: 6
  f_obj: 6
  embed_dim: 64
encoder:
  kind: sophon-ak4          # one of: stub | scratch | sophon-ak4
  checkpoint: PARTAK4.pt    # HF file in jet-universe/sophon-ak4
  freeze_backbone: true     # frozen-first; also run false (fine-tune)
training:
  number_of_epochs: 100
  batch_size: 512
  learning_rate: 0.001
  holdout_split: 0.3
data:
  # produced by scripts/delphes_to_clouds.py; one .npz per process
  signal:      saved_clouds/hh_lam1.npz   # SM (kappa_lambda=1) for Phase-1 binary task
  backgrounds:
    - saved_clouds/ttbar.npz
controls: [stub_scratch, scratch, sophon_frozen, sophon_finetune]
low_stat_fractions: [1.0, 0.25, 0.1, 0.05]
```

- [ ] **Step 6: Write README + docs page**

Create `examples/HH_bbtautau_kappalambda_sophon/README.md` (hypothesis, pipeline order: `delphes_to_clouds.py` → train via `compare.py` → `eval_to_ratios.py` → Phase-2 fit; how to set `SOPHON_AK4_CKPT`; the controls and expected plots). Create `docs/basics/sophon_hh_density_ratio.rst` mirroring `docs/basics/density_ratio_training.rst` (the hierarchical model, the constituent extension pattern, how it plugs into the eval→.npy→JAX-fit contract).

- [ ] **Step 7: Commit**
```bash
git add examples/HH_bbtautau_kappalambda_sophon/ docs/basics/sophon_hh_density_ratio.rst tests/test_smoke_end_to_end.py tests/conftest.py
git commit -m "feat: end-to-end smoke test, example config, README, docs page"
```

---

## Task 14: Full suite + dependency/image finalization

**Files:**
- Modify: `image.def`
- Test: all of `tests/`

- [ ] **Step 1: Run the whole suite**

Run: `pixi run -e nsbi-env-gpu pytest -q`
Expected: all pass except the `sophon-ak4`-checkpoint integration test, which SKIPs without `SOPHON_AK4_CKPT`.

- [ ] **Step 2: Update the HPC image**

In `image.def`, ensure the pixi env build includes the new deps (huggingface_hub, weaver-core/ParT, pytest). Add a build-test line that runs `pytest -q` (allowing skips) so the container is validated at build time.

- [ ] **Step 3: Commit**
```bash
git add image.def
git commit -m "build: add foundation-model deps to HPC image; run tests at build"
```

---

## Self-review

- **Spec coverage:** §5 architecture → Tasks 4–6; §6.1 dataset → Task 3; §6.2 encoder → Tasks 5, 11; §6.3 event transformer → Task 4; §6.4 LightningModule → Task 6; §6.5 trainer → Task 8; §6.6 constituent ONNX → Task 7; §6.8 example (converter/eval/compare/smoke/config/README) → Tasks 9, 10, 12, 13; §6.9 docs → Task 13; §6.10 deps/image → Tasks 1, 11, 14; §7 diagnostics + 3 controls + low-stat ablation → Task 12 (closure/calibration plots reuse the existing trainer diagnostics + the ported `histogram_analysis.py`, wired in the notebooks during execution). Phase 2 (§8) is intentionally a separate plan.
- **Constituent eval/Asimov-ordering contract** (the corrected compatibility point) → Task 9.
- **NLO label-flip lesson** → Task 6 (`_loss`, tested). **No-importance-truncation / calibration / reference-design lessons** belong to Phase 2 / data production and are recorded in the spec's risk register; flagged here for the executor.
- **Type/name consistency:** the 5 constituent input keys `parts/part_mask/jet_mask/obj/obj_mask` are used identically in the dataset, model `forward`, `save_model_constituents`, `predict_with_onnx_constituents`, `eval_to_ratios`, and the converter output. `encoder_kind` values `stub|scratch|sophon-ak4` are consistent across `build_jet_encoder`, the trainer, and `compare.py`. `evaluate_ratios`/`convert_score_to_ratio` reused from the existing utils.
- **Known couplings the executor must verify (called out inline, not placeholders):** `LossHistory.train_loss/.val_loss` attribute names (Task 8 note); the exact ParT constructor/`return_embed`/checkpoint key layout + input feature schema (Task 11 note); real Delphes branch names + Delta-R association (Task 10 note). These are genuine integration unknowns, each with a concrete fallback.

---

## Execution handoff

Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks.
2. **Inline Execution** — execute tasks in this session with checkpoints.
