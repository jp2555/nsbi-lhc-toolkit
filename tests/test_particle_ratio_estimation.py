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
