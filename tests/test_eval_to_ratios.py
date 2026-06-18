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
