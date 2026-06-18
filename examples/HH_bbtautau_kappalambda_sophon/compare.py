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
