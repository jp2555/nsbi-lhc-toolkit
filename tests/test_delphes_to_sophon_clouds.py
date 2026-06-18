import numpy as np
import awkward as ak
import uproot
from delphes_to_sophon_clouds import convert_tree, N_JETS_MAX, N_PART_MAX, F_TOTAL


def _write_fake_delphes(path):
    """Two events with the real Delphes branch names from the card's TreeWriter."""
    # Event 0: jet at (eta0,phi0); a charged pion track near it (dR~0.07); a photon far away.
    # Event 1: jet at (eta1,phi1); a neutral hadron near it; a muon + MET.
    def J(v):  # jagged helper
        return ak.Array(v)

    data = {
        "Jet.PT":   J([[100.0], [50.0]]),
        "Jet.Eta":  J([[0.0],   [1.0]]),
        "Jet.Phi":  J([[0.0],   [1.0]]),
        "Jet.Mass": J([[10.0],  [5.0]]),
        "EFlowTrack.PT":      J([[30.0], []]),
        "EFlowTrack.Eta":     J([[0.05], []]),
        "EFlowTrack.Phi":     J([[0.05], []]),
        "EFlowTrack.Charge":  J([[1.0],  []]),
        "EFlowTrack.PID":     J([[211.0], []]),
        "EFlowTrack.D0":      J([[0.1],  []]),
        "EFlowTrack.DZ":      J([[0.2],  []]),
        "EFlowTrack.ErrorD0": J([[0.5],  []]),
        "EFlowTrack.ErrorDZ": J([[0.5],  []]),
        "EFlowPhoton.ET":     J([[20.0], []]),
        "EFlowPhoton.Eta":    J([[2.0],  []]),   # far from the jet -> not associated
        "EFlowPhoton.Phi":    J([[2.0],  []]),
        "EFlowNeutralHadron.ET":  J([[], [25.0]]),
        "EFlowNeutralHadron.Eta": J([[], [1.02]]),
        "EFlowNeutralHadron.Phi": J([[], [1.03]]),
        "Electron.PT":     J([[40.0], []]),
        "Electron.Eta":    J([[1.0],  []]),
        "Electron.Phi":    J([[1.0],  []]),
        "Electron.Charge": J([[-1.0], []]),
        "Muon.PT":     J([[], [30.0]]),
        "Muon.Eta":    J([[], [0.0]]),
        "Muon.Phi":    J([[], [0.0]]),
        "Muon.Charge": J([[], [1.0]]),
        "MissingET.MET": J([[35.0], [25.0]]),
        "MissingET.Phi": J([[0.5],  [-1.0]]),
        "Event.Weight":  ak.Array([1.0, -1.0]),
    }
    with uproot.recreate(path) as f:
        f["Delphes"] = data


def test_convert_real_delphes_schema(tmp_path):
    root = tmp_path / "fake_delphes.root"
    _write_fake_delphes(str(root))
    out = tmp_path / "clouds.npz"
    convert_tree(str(root), "Delphes", str(out), step_size=10)
    d = np.load(out)

    # shapes match the sophon-ak4 schema: parts = 17 features + 4 pf_vectors = 21 cols
    assert d["parts"].shape == (2, N_JETS_MAX, N_PART_MAX, F_TOTAL)
    assert F_TOTAL == 21
    assert d["jet_mask"].sum() == 2          # one selected jet per event

    # event 0, jet 0: only the near charged track is associated (photon is dR>0.4)
    assert d["part_mask"][0, 0].sum() == 1
    feat = d["parts"][0, 0, 0]
    assert feat[6] == 1.0 and feat[9] == 0.0 and feat[8] == 0.0   # is_ch=1, is_e=0, is_ph=0
    # deltaR feature is standardized: (hypot(0.05,0.05) - 0.2) * 4.0
    assert np.isclose(feat[4], (np.hypot(0.05, 0.05) - 0.2) * 4.0, atol=1e-4)
    # 4-vector px lives in parts column 17 (= pt*cos(phi))
    assert np.isclose(feat[17], 30.0 * np.cos(0.05), atol=1e-3)

    # event 1, jet 0: only the near neutral hadron is associated
    assert d["part_mask"][1, 0].sum() == 1
    assert d["parts"][1, 0, 0, 7] == 1.0     # is_nh = 1

    # NLO weight sign preserved; object tokens present (electron+MET ev0, muon+MET ev1)
    assert d["w"][1] == -1.0
    assert d["obj_mask"][0].sum() >= 2 and d["obj_mask"][1].sum() >= 2
