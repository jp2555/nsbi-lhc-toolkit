# `_part_vendor/` — Particle Transformer vendoring placeholder

This package directory is a placeholder for the MIT-licensed **Particle Transformer
(ParT)** model class that the Sophon / sophon-ak4 checkpoints are built on.

## Why we vendor instead of depending on `weaver-core`

`weaver-core` (the Sophon training framework) pins **`uproot>=4.2,<5.2`**, which
**conflicts** with this toolkit's `uproot 5.7`. So `weaver-core` is deliberately NOT
a project dependency. We only need the ParT `nn.Module` definition, which is plain
PyTorch — vendor that single file here.

## What to vendor here

Copy the ParticleTransformer model definition into this directory as `ParT.py`,
exposing the class `ParticleTransformer` at the top level (importable as
`from nsbi_common_utils.lightning_tools._part_vendor.ParT import ParticleTransformer`).

The class lives in **weaver-core's repo** (torch-only file, no need to install the
package):

    https://github.com/hqucms/weaver-core  ->  weaver/nn/model/ParticleTransformer.py

(The `jet-universe/particle_transformer` repo's `networks/example_ParticleTransformer.py`
is a thin wrapper around that same class.)

**Keep the original MIT license header** intact.

## One edit you will likely need

`weaver/nn/model/ParticleTransformer.py` imports a logger:

    from weaver.utils.logger import _logger

Since we are not installing `weaver`, replace that line with a stub, e.g.:

    import logging; _logger = logging.getLogger("ParT")

It uses only `torch` otherwise.

## Effect on tests

Until `ParT.py` is present here:
- The `scratch` and `sophon-ak4` encoder kinds in `build_jet_encoder()` are unavailable.
- All tests in `tests/test_sophon_ak4_backbone.py` are **automatically skipped**
  (the module-level `pytestmark` guard checks for
  `nsbi_common_utils.lightning_tools._part_vendor.ParT`).
- All other tests are unaffected — `StubJetEncoder` (kind="stub") has no dependency here.

## Steps to activate on Perlmutter

1. `git clone https://github.com/hqucms/weaver-core /tmp/weaver-core`
2. `cp /tmp/weaver-core/weaver/nn/model/ParticleTransformer.py src/nsbi_common_utils/lightning_tools/_part_vendor/ParT.py`
3. Replace the `from weaver.utils.logger import _logger` import with the stub above.
4. Reconcile `sophon_ak4_backbone.py` with the actual `ParticleTransformer` constructor
   signature and how it returns the class-token embedding (and the sophon-ak4
   checkpoint `state_dict` key layout).
5. `pixi run -e nsbi-env-gpu pytest tests/test_sophon_ak4_backbone.py -v`, then set
   `SOPHON_AK4_CKPT` for the checkpoint integration test.
