"""sophon-ak4 (ParT) per-jet encoder wrapper.

Wraps the Particle Transformer so it exposes the same interface as StubJetEncoder:
forward(parts: (B, n_part, f_part), mask: (B, n_part)) -> (B, embed_dim).

- kind="scratch": ParT of the sophon-ak4 shape (6 particle + 2 class attn blocks,
  embed 64, 8 heads), random init -> the from-scratch control.
- kind="sophon-ak4": same architecture, weights loaded from the released checkpoint
  (download via huggingface_hub: repo 'jet-universe/sophon-ak4'); the 23-class head
  is dropped and the class-token embedding (dim 64) is used as the jet embedding.

ParT expects features (x) and 4-vectors (v) for pairwise interactions; here we pass
the per-constituent feature block as x and reuse (logpt, deta, dphi) for v. The
exact feature ordering/normalization MUST match the checkpoint's training config
(see the sophon repo's data config); adapt _split_features accordingly.
"""
import torch
import torch.nn as nn


def _load_part(embed_dim, f_part):
    # from the vendored ParT (MIT). Config mirrors sophon-ak4: 6 + 2 blocks, 8 heads.
    from nsbi_common_utils.lightning_tools._part_vendor.ParT import ParticleTransformer
    return ParticleTransformer(
        input_dim=f_part, num_classes=23, embed_dims=[embed_dim, embed_dim, embed_dim],
        num_heads=8, num_layers=6, num_cls_layers=2, fc_params=[], use_amp=False)


class SophonAK4Encoder(nn.Module):
    def __init__(self, f_part=8, embed_dim=64, checkpoint=None):
        super().__init__()
        self.part = _load_part(embed_dim, f_part)
        self.embed_dim = embed_dim
        if checkpoint is not None:
            from huggingface_hub import hf_hub_download
            ckpt = checkpoint
            if not checkpoint.endswith((".pt", ".onnx")):
                ckpt = hf_hub_download(repo_id="jet-universe/sophon-ak4", filename=checkpoint)
            state = torch.load(ckpt, map_location="cpu")
            self.part.load_state_dict(state.get("model", state), strict=False)

    def forward(self, parts, mask):
        # ParT signature: (x=(B,F,P), v=(B,4,P), mask=(B,1,P)); return class-token embedding.
        x = parts.transpose(1, 2)                       # (B, F, P)
        v = x[:, :3, :]                                  # reuse (logpt,deta,dphi) as 4-vec proxy
        m = mask.unsqueeze(1)                            # (B, 1, P)
        emb = self.part(x, v, m, return_embed=True)      # (B, embed_dim) class token
        return emb


def build_part_encoder(kind, f_part=8, embed_dim=64, checkpoint=None, **_):
    if kind == "scratch":
        return SophonAK4Encoder(f_part=f_part, embed_dim=embed_dim, checkpoint=None)
    if kind == "sophon-ak4":
        return SophonAK4Encoder(f_part=f_part, embed_dim=embed_dim,
                                checkpoint=checkpoint or "PARTAK4.pt")
    raise ValueError(kind)
