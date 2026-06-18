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
