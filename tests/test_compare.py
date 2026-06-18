from compare import run_controls, low_stat_ablation


def test_run_controls_returns_histories(tmp_path, synth_batch):
    res = run_controls(synth_batch, synth_batch, out_dir=str(tmp_path),
                       controls=["stub_scratch", "stub_frozen"], epochs=2, batch_size=8)
    assert set(res) == {"stub_scratch", "stub_frozen"}
    assert all("train_loss" in v and len(v["train_loss"]) == 2 for v in res.values())


def test_low_stat_ablation_shape(tmp_path, synth_batch):
    curve = low_stat_ablation(synth_batch, synth_batch, out_dir=str(tmp_path),
                              fractions=[1.0, 0.5], epochs=1, batch_size=8)
    assert sorted(curve) == [0.5, 1.0]
    assert all("val_loss_final" in curve[f] for f in curve)
