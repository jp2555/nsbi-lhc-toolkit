# `_part_vendor/` — Particle Transformer vendoring placeholder

This package directory is a placeholder for the MIT-licensed Particle Transformer
model definition from:

    https://github.com/jet-universe/particle_transformer

## What to vendor here

Copy the file `ParT.py` from the above repository into this directory. It must
provide the class `ParticleTransformer` at the top level (i.e. importable as
`from nsbi_common_utils.lightning_tools._part_vendor.ParT import ParticleTransformer`).

**Keep the original MIT license header** at the top of `ParT.py` intact.

## Why this is not committed

`ParT.py` is maintained by the Particle Transformer authors in the repository
above. Rather than fork it, we vendor it manually on the cluster (Perlmutter)
so that the license header is preserved and updates can be pulled cleanly.

## Effect on tests

Until `ParT.py` is present here:
- The `scratch` and `sophon-ak4` encoder kinds in `build_jet_encoder()` are
  unavailable.
- All tests in `tests/test_sophon_ak4_backbone.py` are **automatically skipped**
  (the module-level `pytestmark` guard checks for the presence of
  `nsbi_common_utils.lightning_tools._part_vendor.ParT`).
- All other tests remain unaffected — the `StubJetEncoder` (kind="stub") has
  no dependency on this package.

## Steps to activate on Perlmutter

1. Clone or download `particle_transformer` from
   https://github.com/jet-universe/particle_transformer.
2. Copy `weaver/models/ParT.py` (or equivalent) into this directory as `ParT.py`.
3. Ensure the `ParticleTransformer` class is importable from it.
4. Run `pixi run -e nsbi-env-gpu pytest tests/test_sophon_ak4_backbone.py -v`
   to verify the scratch path passes, then set `SOPHON_AK4_CKPT` for the
   checkpoint integration test.
