import torch
from nsbi_common_utils.lightning_tools.jet_encoder import StubJetEncoder, build_jet_encoder


def test_stub_encoder_shape_and_mask():
    B, P, F, D = 7, 64, 8, 64
    parts = torch.randn(B, P, F)
    mask = torch.ones(B, P)
    mask[:, 30:] = 0.0
    enc = StubJetEncoder(f_part=F, embed_dim=D)
    emb = enc(parts, mask)
    assert emb.shape == (B, D)
    parts2 = parts.clone(); parts2[:, 30:] = torch.randn(B, 34, F)
    assert torch.allclose(emb, enc(parts2, mask), atol=1e-5)


def test_factory_builds_stub():
    enc = build_jet_encoder(kind="stub", f_part=8, embed_dim=64)
    assert isinstance(enc, StubJetEncoder)
    assert enc(torch.randn(3, 64, 8), torch.ones(3, 64)).shape == (3, 64)
