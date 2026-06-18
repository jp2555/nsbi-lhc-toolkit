# `_part_vendor/` — vendored Particle Transformer (ParT)

`ParT.py` here is the **Particle Transformer** model class, **vendored from
weaver-core** (MIT, © 2020 Huilin Qu):

    https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py

The **only** change from upstream is replacing `from weaver.utils.logger import _logger`
with a stdlib `logging` stub, so the file is **torch-only**. We do not depend on the
`weaver-core` package because it pins `uproot<5.2`, which conflicts with this toolkit's
`uproot 5.7`.

To update it, re-copy that file and re-apply the two-line logger stub (keep the MIT
attribution header).

## How it's used

`sophon_ak4_backbone.py` builds `ParticleTransformer(..., num_classes=None,
fc_params=[])` so `self.fc is None` and the model yields the **class-token
embedding**. The embedding is read via `_forward_encoder` → `_forward_aggregator`
(ParT's public `forward` assumes a classifier head — this is the same pattern
Sophon's own wrapper uses). `trim=False` disables the SequenceTrimmer for
deterministic, ONNX-clean behavior.

## sophon-ak4 input schema (must match to use the pretrained weights)

From the sophon-ak4 HF repo data config
(`jet-universe/sophon-ak4 : data/JetClassII/JetClassII_ak4unscaled_full.yaml`),
the model was trained on **128 constituents** with:

**`pf_features` (input_dim = 17, in this order),** with manual standardization
`(value - subtract) * multiply`, clipped to [-5, 5]:

| # | feature | subtract | multiply |
|---|---------|----------|----------|
| 1 | part_pt_log = log(pt)            | 1.7  | 0.7 |
| 2 | part_e_log = log(E)             | 2.0  | 0.7 |
| 3 | part_logptrel = log(pt/jet_pt)  | -4.7 | 0.7 |
| 4 | part_logerel = log(E/jet_E)     | -4.7 | 0.7 |
| 5 | part_deltaR = hypot(deta,dphi)  | 0.2  | 4.0 |
| 6 | part_charge                     | 0    | 1   |
| 7 | part_isChargedHadron            | 0    | 1   |
| 8 | part_isNeutralHadron            | 0    | 1   |
| 9 | part_isPhoton                   | 0    | 1   |
| 10| part_isElectron                 | 0    | 1   |
| 11| part_isMuon                     | 0    | 1   |
| 12| part_d0 = tanh(d0val)           | 0    | 1   |
| 13| part_d0err  (clip 0..1)         | 0    | 1   |
| 14| part_dz = tanh(dzval)           | 0    | 1   |
| 15| part_dzerr  (clip 0..1)         | 0    | 1   |
| 16| part_deta                       | 0    | 1   |
| 17| part_dphi                       | 0    | 1   |

**`pf_vectors` (pair_input_dim = 4):** `[part_px, part_py, part_pz, part_energy]`
— pass as the `vectors` arg to the encoder's `forward` to enable the pairwise
interaction features the checkpoint was trained with.

**Labels:** 23 classes (`jet_label` 0–22). The classifier head (`fc.*`) is dropped on
load; only the backbone + class-attention weights are kept.

The `delphes_to_clouds.py` converter must be aligned to emit exactly these 17
features (standardized) + the 4-vectors for the sophon-ak4 path. (The `scratch`
control can use any feature set, e.g. the current 8.)

## Checkpoint

- File: `models/JetClassII_SophonAK4/model.pt` in the HF repo `jet-universe/sophon-ak4`
  (there is also a `model.onnx`). `SophonAK4Encoder` downloads it via
  `huggingface_hub.hf_hub_download` if a local path isn't given.
- Loading strips the SophonWrapper `mod.` prefix and the `fc.` head, then
  `load_state_dict(strict=False)`. The `(missing, unexpected)` report is stored on
  `encoder.load_report` and warned if non-trivial.
- **Confirm `embed_dims` / `pair_embed_dims` / `num_layers` against the checkpoint**:
  the defaults in `sophon_ak4_backbone.py` yield a 64-d embedding but the exact
  intermediate dims are a best-guess. On the cluster, inspect shapes:
  ```python
  import torch
  sd = torch.load("model.pt", map_location="cpu")
  sd = sd.get("model", sd)
  for k, v in sd.items():
      if "embed" in k: print(k, tuple(v.shape))
  ```
  and adjust the `embed_dims`/`pair_embed_dims` kwargs until `load_report` shows no
  (or only `fc.*`) mismatches.

## Effect on tests

`ParT.py` being present makes `_PART_AVAILABLE` true, so `tests/test_sophon_ak4_backbone.py`
now runs (instead of skipping). `test_scratch_encoder_interface` and
`test_scratch_encoder_with_pairwise_and_vectors` run anywhere torch is available;
`test_sophon_ak4_loads_and_runs` additionally requires `SOPHON_AK4_CKPT`.
