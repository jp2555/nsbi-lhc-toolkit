import numpy as np
from examples_pkg_smoke import run_smoke


def test_end_to_end(tmp_path, synth_batch):
    result = run_smoke(synth_batch, out_dir=str(tmp_path))
    assert result["onnx_exists"]
    assert result["ratios"].shape == (synth_batch["y"].shape[0],)
    assert np.all(result["ratios"] > 0)
