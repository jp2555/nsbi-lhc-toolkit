"""sophon-ak4 (Particle Transformer) per-jet encoder wrapper.

Exposes the same interface as ``StubJetEncoder``:
``forward(parts: (B, n_part, f_part), mask: (B, n_part)) -> (B, embed_dim)``,
with an optional ``vectors`` arg carrying the per-particle 4-vectors used by ParT's
pairwise interaction features.

Kinds (via ``build_part_encoder``):
- ``"scratch"``: a ParT of the sophon-ak4 shape (6 particle + 2 class-attention
  blocks, 8 heads, 64-d embedding), random init -> the from-scratch control. By
  default it runs WITHOUT the pairwise term (``use_pair=False``) since the toolkit's
  cloud does not yet carry 4-vectors.
- ``"sophon-ak4"``: the same architecture with weights loaded from the released
  checkpoint (HF ``jet-universe/sophon-ak4`` -> ``models/JetClassII_SophonAK4/model.pt``).
  The 23-class classifier head and the SophonWrapper ``mod.`` prefix are stripped on
  load; the class-token embedding is used as the jet embedding.

Embedding extraction follows Sophon's own wrapper: call ``_forward_encoder`` then
``_forward_aggregator`` (ParT's public ``forward`` assumes a classifier head).

IMPORTANT — to actually benefit from the pretrained weights, the per-constituent
inputs MUST match sophon-ak4's training schema (see ``_part_vendor/README.md``):
17 ``pf_features`` in order [part_pt_log, part_e_log, part_logptrel, part_logerel,
part_deltaR, part_charge, part_isChargedHadron, part_isNeutralHadron, part_isPhoton,
part_isElectron, part_isMuon, part_d0(=tanh(d0val)), part_d0err, part_dz(=tanh(dzval)),
part_dzerr, part_deta, part_dphi] with the config's manual standardization applied,
plus 4 ``pf_vectors`` [part_px, part_py, part_pz, part_energy] passed as ``vectors``.
Aligning the Delphes->cloud converter to this schema is the remaining integration step.
"""
import torch
import torch.nn as nn

# sophon-ak4 architecture (from the HF repo config / paper): 6 particle-attention +
# 2 class-attention blocks, 8 heads, 64-d class-token embedding, 17 input features,
# 4-d pairwise (px,py,pz,E). embed_dims/pair_embed_dims below are a best-guess that
# yields a 64-d embedding; CONFIRM against model.pt key shapes on the cluster (a few
# mismatched keys from load_state_dict(strict=False) means these need adjusting).
SOPHON_AK4_INPUT_DIM = 17
SOPHON_AK4_CKPT_FILE = "models/JetClassII_SophonAK4/model.pt"
# Confirmed from model.pt key shapes (2026-06-18): particle-embed [64,256,64],
# pair-embed [32,32,32] (final -> num_heads=8), 6 particle blocks + 2 class blocks.
SOPHON_AK4_EMBED_DIMS = [64, 256, 64]
SOPHON_AK4_PAIR_EMBED_DIMS = [32, 32, 32]


def _build_part(input_dim, embed_dim=64, num_layers=6, num_cls_layers=2, num_heads=8,
                pair_input_dim=4, use_pair=True, embed_dims=None, pair_embed_dims=None):
    """Construct a ParT configured as an encoder (fc=None -> embedding output)."""
    from nsbi_common_utils.lightning_tools._part_vendor.ParT import ParticleTransformer
    embed_dims = list(embed_dims) if embed_dims else [embed_dim, embed_dim * 4, embed_dim]
    pair_embed_dims = (list(pair_embed_dims) if pair_embed_dims else [64, 64, 64]) if use_pair else None
    return ParticleTransformer(
        input_dim=input_dim,
        num_classes=None,
        pair_input_dim=(pair_input_dim if use_pair else None),
        embed_dims=embed_dims,
        pair_embed_dims=pair_embed_dims,
        num_heads=num_heads,
        num_layers=num_layers,
        num_cls_layers=num_cls_layers,
        # fc_params=None (NOT []) -> self.fc is None -> the model yields the embedding.
        # ParT builds a final nn.Linear(in_dim, num_classes) whenever fc_params is not
        # None, which would be nn.Linear(.., None) and crash. We read the class-token
        # embedding via _forward_encoder/_forward_aggregator regardless.
        fc_params=None,
        trim=False,                       # disable the SequenceTrimmer: deterministic + ONNX-clean
        for_inference=False,
        use_amp=False,
    )


class SophonAK4Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_layers=6, num_cls_layers=2,
                 num_heads=8, pair_input_dim=4, use_pair=True,
                 embed_dims=None, pair_embed_dims=None, checkpoint=None):
        super().__init__()
        self.use_pair = use_pair
        self.part = _build_part(
            input_dim, embed_dim=embed_dim, num_layers=num_layers,
            num_cls_layers=num_cls_layers, num_heads=num_heads,
            pair_input_dim=pair_input_dim, use_pair=use_pair,
            embed_dims=embed_dims, pair_embed_dims=pair_embed_dims)
        self.embed_dim = (list(embed_dims)[-1] if embed_dims else embed_dim)
        self.load_report = None
        if checkpoint is not None:
            self.load_report = self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, checkpoint):
        """Load a sophon-ak4 checkpoint: strip the SophonWrapper 'mod.' prefix and the
        classifier head ('fc.'), keep only keys whose shape matches this model, then load
        non-strictly. NOTE: load_state_dict raises on *shape* mismatch even with
        strict=False (strict only tolerates missing/unexpected keys), so we filter
        mismatched shapes here. Returns a report so the caller can confirm the config."""
        import os
        path = checkpoint
        if not os.path.exists(path):
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id="jet-universe/sophon-ak4", filename=checkpoint)
        state = torch.load(path, map_location="cpu")
        sd = state.get("model", state.get("state_dict", state)) if isinstance(state, dict) else state
        model_sd = self.part.state_dict()
        cleaned, shape_mismatch = {}, []
        for k, v in sd.items():
            kk = k[4:] if k.startswith("mod.") else k
            if kk.startswith("fc."):           # drop the 23-class classifier head
                continue
            if kk in model_sd and tuple(v.shape) != tuple(model_sd[kk].shape):
                shape_mismatch.append((kk, tuple(v.shape), tuple(model_sd[kk].shape)))
                continue
            cleaned[kk] = v
        missing, unexpected = self.part.load_state_dict(cleaned, strict=False)
        report = {"loaded": len(cleaned), "missing": list(missing),
                  "unexpected": list(unexpected), "shape_mismatch": shape_mismatch}
        if missing or unexpected or shape_mismatch:
            _warn_keys(missing, unexpected, shape_mismatch)
        return report

    def forward(self, parts, mask, vectors=None):
        # parts: (B, P, F) -> x: (B, F, P); mask: (B, P) -> (B, 1, P) (trimmer .bool()s it)
        x = parts.transpose(1, 2)
        m = mask.unsqueeze(1)
        v = vectors.transpose(1, 2) if (self.use_pair and vectors is not None) else None
        enc, padding_mask = self.part._forward_encoder(x, v=v, mask=m)
        return self.part._forward_aggregator(enc, padding_mask)   # (B, embed_dim)


def _warn_keys(missing, unexpected, shape_mismatch=()):
    import logging
    log = logging.getLogger("sophon_ak4")
    log.warning(
        "sophon-ak4 checkpoint load: %d missing, %d unexpected, %d shape-mismatch keys. "
        "Shape-mismatched keys are SKIPPED (left at init); if numerous, the ParT config "
        "(embed_dims/pair_embed_dims/num_layers) does not match model.pt.",
        len(missing), len(unexpected), len(shape_mismatch))
    for kk, got, exp in shape_mismatch:
        log.warning("  shape-mismatch %s: ckpt %s vs model %s", kk, got, exp)


def build_part_encoder(kind, f_part=8, embed_dim=64, checkpoint=None,
                       num_layers=6, num_cls_layers=2, num_heads=8, pair_input_dim=4,
                       input_dim=None, use_pair=None, embed_dims=None,
                       pair_embed_dims=None, **_):
    """Factory returning a ParT-based jet encoder with the StubJetEncoder interface.

    kind="scratch": random-init ParT on ``f_part`` features, no pairwise by default.
    kind="sophon-ak4": ParT on the 17-feature schema + 4-vec pairwise, weights from
    the released checkpoint.
    """
    if kind == "scratch":
        return SophonAK4Encoder(
            input_dim=(input_dim or f_part), embed_dim=embed_dim,
            num_layers=num_layers, num_cls_layers=num_cls_layers, num_heads=num_heads,
            pair_input_dim=pair_input_dim,
            use_pair=(False if use_pair is None else use_pair),
            embed_dims=embed_dims, pair_embed_dims=pair_embed_dims, checkpoint=None)
    if kind == "sophon-ak4":
        return SophonAK4Encoder(
            input_dim=(input_dim or SOPHON_AK4_INPUT_DIM), embed_dim=embed_dim,
            num_layers=num_layers, num_cls_layers=num_cls_layers, num_heads=num_heads,
            pair_input_dim=pair_input_dim,
            use_pair=(True if use_pair is None else use_pair),
            embed_dims=(embed_dims or SOPHON_AK4_EMBED_DIMS),
            pair_embed_dims=(pair_embed_dims or SOPHON_AK4_PAIR_EMBED_DIMS),
            checkpoint=(checkpoint or SOPHON_AK4_CKPT_FILE))
    raise ValueError(f"unknown ParT encoder kind: {kind!r}")
