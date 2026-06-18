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
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1),
        )

    def forward(self, tokens, mask):
        # mask: (B, L) with 1.0 = real. TransformerEncoder wants True = PAD.
        key_padding = mask < 0.5
        # A fully-padded row would make attention nan; guarantee >=1 valid token.
        safe = key_padding.clone()
        safe[:, 0] = False
        z = self.encoder(tokens, src_key_padding_mask=safe)
        m = mask.unsqueeze(-1)
        pooled = (z * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)
