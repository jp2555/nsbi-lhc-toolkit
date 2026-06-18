import torch
import torch.nn as nn


class StubJetEncoder(nn.Module):
    """Lightweight masked DeepSets encoder for tests / a no-pretraining baseline.

    Maps a padded constituent cloud (B, n_part, f_part) + mask (B, n_part) to a
    (B, embed_dim) jet embedding. Permutation-invariant and mask-respecting.
    """

    def __init__(self, f_part=8, embed_dim=64, hidden=128):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(f_part, hidden), nn.GELU(), nn.Linear(hidden, embed_dim), nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, parts, mask):
        h = self.phi(parts)                       # (B, P, D)
        m = mask.unsqueeze(-1)                     # (B, P, 1)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.rho(pooled)


def build_jet_encoder(kind="stub", f_part=8, embed_dim=64, **kwargs):
    """Factory for the per-jet encoder.

    kind:
      - "stub"      : StubJetEncoder (tests, CPU, no external deps)
      - "scratch"   : a ParT of the sophon-ak4 shape, random init (Task 11)
      - "sophon-ak4": ParT loaded from the released checkpoint (Task 11)
    """
    if kind == "stub":
        return StubJetEncoder(f_part=f_part, embed_dim=embed_dim, **kwargs)
    if kind in ("scratch", "sophon-ak4"):
        from nsbi_common_utils.lightning_tools.sophon_ak4_backbone import build_part_encoder
        return build_part_encoder(kind=kind, f_part=f_part, embed_dim=embed_dim, **kwargs)
    raise ValueError(f"unknown jet encoder kind: {kind!r}")
