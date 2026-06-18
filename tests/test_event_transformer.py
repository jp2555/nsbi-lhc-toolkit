import torch
from nsbi_common_utils.lightning_tools.event_transformer import EventSetTransformer


def test_forward_shape_and_masking():
    B, L, D = 5, 10, 64
    tokens = torch.randn(B, L, D)
    mask = torch.ones(B, L)
    mask[:, 7:] = 0.0  # last 3 tokens padded
    net = EventSetTransformer(embed_dim=D, n_heads=8, n_layers=2)
    out = net(tokens, mask)
    assert out.shape == (B, 1)
    # Changing only padded tokens must not change the output (mask works)
    tokens2 = tokens.clone()
    tokens2[:, 7:] = torch.randn(B, 3, D)
    out2 = net(tokens2, mask)
    assert torch.allclose(out, out2, atol=1e-5)
