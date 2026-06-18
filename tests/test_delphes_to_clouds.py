import numpy as np
import awkward as ak
import uproot
from examples_pkg_delphes import convert_tree


def _write_fake_delphes(path):
    # 3 events; jagged jet + constituent structure (minimal Delphes-like schema)
    n_ev = 3
    jet_pt = ak.Array([[120.0, 80.0], [60.0], [200.0, 50.0, 40.0]])
    jet_eta = ak.Array([[0.1, -0.5], [1.0], [0.2, 0.3, -1.1]])
    jet_phi = ak.Array([[0.2, 1.1], [-0.7], [0.0, 2.0, -2.5]])
    # one constituent list per event (flat), with a jet index per constituent
    part_pt = ak.Array([[30.0, 20.0, 10.0], [15.0], [50.0, 25.0]])
    part_eta = ak.Array([[0.1, 0.12, -0.5], [1.0], [0.2, 0.25]])
    part_phi = ak.Array([[0.2, 0.22, 1.1], [-0.7], [0.0, 0.05]])
    part_jetidx = ak.Array([[0, 0, 1], [0], [0, 0]])
    part_charge = ak.Array([[1, -1, 0], [1], [0, -1]])
    met = ak.Array([40.0, 25.0, 90.0])
    met_phi = ak.Array([0.5, -1.2, 2.0])
    weight = ak.Array([1.0, -1.0, 1.0])
    with uproot.recreate(path) as f:
        f["Delphes"] = {
            "Jet.PT": jet_pt, "Jet.Eta": jet_eta, "Jet.Phi": jet_phi,
            "EFlow.PT": part_pt, "EFlow.Eta": part_eta, "EFlow.Phi": part_phi,
            "EFlow.JetIndex": part_jetidx, "EFlow.Charge": part_charge,
            "MissingET.MET": met, "MissingET.Phi": met_phi, "Event.Weight": weight,
        }


def test_convert_tiny_delphes(tmp_path):
    root = tmp_path / "fake.root"
    _write_fake_delphes(str(root))
    out = tmp_path / "clouds.npz"
    convert_tree(str(root), "Delphes", str(out))
    d = np.load(out)
    assert d["parts"].shape == (3, 4, 64, 8)
    assert d["jet_mask"].sum() == 2 + 1 + 3  # jets per event
    assert d["w"].shape == (3,) and d["w"][1] == -1.0  # NLO sign preserved
