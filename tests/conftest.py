import numpy as np
import pytest

N_JETS_MAX, N_PART_MAX, F_PART, N_OBJ_MAX, F_OBJ = 4, 64, 8, 6, 6


def _rng(seed):
    return np.random.default_rng(seed)


def make_event(seed, n_jets=2, n_parts=20, n_obj=3):
    """One synthetic event: padded clouds + masks + object tokens."""
    r = _rng(seed)
    parts = np.zeros((N_JETS_MAX, N_PART_MAX, F_PART), dtype=np.float32)
    part_mask = np.zeros((N_JETS_MAX, N_PART_MAX), dtype=np.float32)
    jet_mask = np.zeros((N_JETS_MAX,), dtype=np.float32)
    for j in range(min(n_jets, N_JETS_MAX)):
        jet_mask[j] = 1.0
        for p in range(min(n_parts, N_PART_MAX)):
            part_mask[j, p] = 1.0
            parts[j, p] = r.normal(size=F_PART).astype(np.float32)
    obj = np.zeros((N_OBJ_MAX, F_OBJ), dtype=np.float32)
    obj_mask = np.zeros((N_OBJ_MAX,), dtype=np.float32)
    for o in range(min(n_obj, N_OBJ_MAX)):
        obj_mask[o] = 1.0
        obj[o] = r.normal(size=F_OBJ).astype(np.float32)
        obj[o, -1] = float(o % 4)  # type_id
    return parts, part_mask, jet_mask, obj, obj_mask


@pytest.fixture
def synth_batch():
    """A small labelled, weighted batch as numpy arrays."""
    n = 32
    parts, pm, jm, obj, om, y, w = [], [], [], [], [], [], []
    for i in range(n):
        p, pmask, jmask, o, omask = make_event(i, n_jets=1 + i % 3)
        parts.append(p); pm.append(pmask); jm.append(jmask); obj.append(o); om.append(omask)
        y.append(i % 2)
        w.append(1.0)
    return {
        "parts": np.stack(parts), "part_mask": np.stack(pm), "jet_mask": np.stack(jm),
        "obj": np.stack(obj), "obj_mask": np.stack(om),
        "y": np.array(y, dtype=np.float32), "w": np.array(w, dtype=np.float32),
    }
