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


def test_overfits_tiny_separable_set():
    import numpy as np
    from _synth import make_event
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
