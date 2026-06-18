import numpy as np
from run_kappa_diagnostic import build_binary_task, load_point, _CLOUD_KEYS


def _fake_point(n, seed):
    r = np.random.default_rng(seed)
    return {
        "parts": r.normal(size=(n, 8, 128, 21)).astype("float32"),
        "part_mask": (r.random((n, 8, 128)) > 0.5).astype("float32"),
        "jet_mask": (r.random((n, 8)) > 0.3).astype("float32"),
        "obj": r.normal(size=(n, 6, 6)).astype("float32"),
        "obj_mask": (r.random((n, 6)) > 0.5).astype("float32"),
        "w": np.ones(n, dtype="float32"),
    }


def test_build_binary_task():
    a, b = _fake_point(5, 0), _fake_point(7, 1)
    task = build_binary_task(a, b)
    assert task["parts"].shape == (12, 8, 128, 21)
    assert task["y"].sum() == 7
    assert (task["y"][:5] == 0).all() and (task["y"][5:] == 1).all()
    # weights normalized per class -> each class sums to 1
    assert np.isclose(task["w"][:5].sum(), 1.0)
    assert np.isclose(task["w"][5:].sum(), 1.0)
    for k in _CLOUD_KEYS:
        assert task[k].shape[0] == 12


def test_load_point_caps(tmp_path):
    p = tmp_path / "kl1.npz"
    np.savez(str(p), **_fake_point(50, 2))
    out = load_point(str(p), num=10)
    assert out["w"].shape[0] == 10
    assert out["parts"].shape == (10, 8, 128, 21)
