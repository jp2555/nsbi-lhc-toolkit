"""Tiny end-to-end check: synthetic clouds -> train 2 epochs -> ONNX -> ratios.
Run directly (`python smoke_test.py`) or via pytest (tests/test_smoke_end_to_end.py)."""
import os
import numpy as np
from nsbi_common_utils.training.particle_ratio_estimation import particle_density_ratio_trainer


def _synthetic(n=64, seed=0):
    from nsbi_common_utils.lightning_tools.cloud_spec import DEFAULT_SPEC as S
    N_JETS_MAX, N_PART_MAX, F_PART, N_OBJ_MAX, F_OBJ = (
        S.n_jets_max, S.n_part_max, S.f_part, S.n_obj_max, S.f_obj)
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
