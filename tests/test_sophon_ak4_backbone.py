import os
import importlib.util
import pytest
import torch

_PART_AVAILABLE = importlib.util.find_spec("nsbi_common_utils.lightning_tools._part_vendor.ParT") is not None
pytestmark = pytest.mark.skipif(not _PART_AVAILABLE, reason="vendor ParT into _part_vendor/ (see README) to run")


def test_scratch_encoder_interface():
    from nsbi_common_utils.lightning_tools.jet_encoder import build_jet_encoder
    enc = build_jet_encoder(kind="scratch", f_part=8, embed_dim=64)
    out = enc(torch.randn(3, 64, 8), torch.ones(3, 64))
    assert out.shape == (3, 64)


@pytest.mark.skipif(not os.environ.get("SOPHON_AK4_CKPT"),
                    reason="set SOPHON_AK4_CKPT to the downloaded sophon-ak4 model.pt to run")
def test_sophon_ak4_loads_and_runs():
    # parts = 17 features + 4 pf_vectors = 21 cols; encoder slices 0:17 (x) and 17:21 (v).
    from nsbi_common_utils.lightning_tools.jet_encoder import build_jet_encoder
    enc = build_jet_encoder(kind="sophon-ak4", embed_dim=64,
                            checkpoint=os.environ["SOPHON_AK4_CKPT"])
    out = enc(torch.randn(2, 128, 21), torch.ones(2, 128))
    assert out.shape == (2, 64)


def test_scratch_encoder_with_pairwise():
    # exercise the pairwise path: parts has 17 features + 4 4-vector cols (=21), use_pair=True
    from nsbi_common_utils.lightning_tools.sophon_ak4_backbone import build_part_encoder
    enc = build_part_encoder(kind="scratch", input_dim=17, embed_dim=64, use_pair=True)
    out = enc(torch.randn(2, 32, 21), torch.ones(2, 32))
    assert out.shape == (2, 64)
