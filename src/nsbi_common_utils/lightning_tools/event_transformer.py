import torch
import torch.nn as nn


class EventSetTransformer(nn.Module):
    """Masked multi-head self-attention over event tokens -> scalar logit.

    Padded tokens (mask==0) are excluded from attention via key_padding_mask and
    from the final masked-mean pool, so their values cannot affect the output.
    """

    def __init__(self, embed_dim=64, n_heads=8, n_layers=2, ff_mult=2, dropout=0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * ff_mult, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        # enable_nested_tensor=False: the nested-tensor fast path does not trace
        # cleanly to ONNX (opset 17); disabling it keeps export deterministic.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1),
        )

    def forward(self, tokens, mask):
        # mask: (B, L) with 1.0 = real. TransformerEncoder wants True = PAD.
        key_padding = mask < 0.5
        # A fully-padded row makes attention's softmax(all -inf) = NaN. Guard ONLY
        # genuinely all-padded rows by un-padding column 0 there; real rows are left
        # untouched (so padded tokens never influence real ones). Done without
        # data-dependent in-place mutation, which keeps the ONNX export clean.
        L = tokens.shape[1]
        is_col0 = (torch.arange(L, device=mask.device) == 0).unsqueeze(0)   # (1, L)
        all_pad = key_padding.all(dim=1, keepdim=True)                       # (B, 1)
        key_padding = key_padding & ~(all_pad & is_col0)
        z = self.encoder(tokens, src_key_padding_mask=key_padding)
        m = mask.unsqueeze(-1)
        pooled = (z * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)
