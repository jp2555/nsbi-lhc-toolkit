import json
import os

from convert_all_kl import convert_all, _DSET
from test_delphes_to_sophon_clouds import _write_fake_delphes


def test_convert_all_kl_layout(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "clouds"
    # Recreate the real layout for the SM (kl=1) point:
    #   <raw>/GluGluHHto2B2Tau_...-kl-1p00-..._Delphes/delphes-tree-<hash>/delphes-tree_0.root
    dset = raw / _DSET.format(suf="1p00") / "delphes-tree-abc123"
    dset.mkdir(parents=True)
    _write_fake_delphes(str(dset / "delphes-tree_0.root"))

    manifest = convert_all(str(raw), str(out))

    assert os.path.exists(out / "kl1.npz")
    assert os.path.exists(out / "manifest.json")
    assert manifest["kl1"]["kappa_lambda"] == 1.0
    assert manifest["kl1"]["n_files"] == 1
    m = json.loads((out / "manifest.json").read_text())
    assert m["kl1"]["npz"].endswith("kl1.npz")
    # kappa_lambda points with no files present are skipped
    assert "kl5" not in manifest
