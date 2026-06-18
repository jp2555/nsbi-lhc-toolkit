import sys
import pathlib

import numpy as np
import pytest

# Make the tests helper + the example modules importable by bare module name.
_TESTS_DIR = pathlib.Path(__file__).resolve().parent
_REPO = _TESTS_DIR.parent
_EX_ROOT = _REPO / "examples" / "HH_bbtautau_kappalambda_sophon"
_SCRIPTS = _EX_ROOT / "scripts"
for _p in (_TESTS_DIR, _SCRIPTS, _EX_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _synth import make_event  # noqa: E402


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
