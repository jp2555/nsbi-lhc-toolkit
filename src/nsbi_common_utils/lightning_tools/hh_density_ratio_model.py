import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from nsbi_common_utils.lightning_tools.cloud_spec import DEFAULT_SPEC
from nsbi_common_utils.lightning_tools.jet_encoder import build_jet_encoder
from nsbi_common_utils.lightning_tools.event_transformer import EventSetTransformer

N_OBJ_TYPES = 4  # tau_had, electron, muon, met


class HHDensityRatioLightning(pl.LightningModule):
    """Hierarchical constituent density-ratio model: per-jet encoder -> event transformer.

    forward() takes the dict batch produced by WeightedParticleCloudDataset and
    returns a per-event score in (0, 1) (or a raw logit if use_log_loss=True).

    NLO weights: events with w < 0 are handled by a label-flip (train on |w| with
    the label inverted), which keeps the weighted BCE finite and unbiased for
    signed-weight samples.
    """

    def __init__(self, encoder_kind="stub", spec=DEFAULT_SPEC,
                 n_heads=8, n_layers=2, learning_rate=1e-3,
                 use_log_loss=False, freeze_backbone=False, weight_decay=1e-4,
                 callback_factor=0.5, callback_patience=20, encoder_kwargs=None):
        super().__init__()
        self.save_hyperparameters(ignore=["spec"])
        self.spec = spec
        self.lr = learning_rate
        self.use_log_loss = use_log_loss

        self.encoder = build_jet_encoder(
            kind=encoder_kind, f_part=spec.f_part, embed_dim=spec.embed_dim,
            **(encoder_kwargs or {}),
        )
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.obj_proj = nn.Linear(spec.f_obj - 1, spec.embed_dim)  # -1: type_id is embedded
        self.obj_type_emb = nn.Embedding(N_OBJ_TYPES, spec.embed_dim)
        self.jet_type_emb = nn.Parameter(torch.zeros(spec.embed_dim))
        self.event = EventSetTransformer(
            embed_dim=spec.embed_dim, n_heads=n_heads, n_layers=n_layers,
        )

    def _encode_jets(self, parts, part_mask, jet_mask):
        B, J, P, Fp = parts.shape
        flat = parts.reshape(B * J, P, Fp)
        flat_mask = part_mask.reshape(B * J, P)
        emb = self.encoder(flat, flat_mask).reshape(B, J, -1)   # (B, J, D)
        emb = emb + self.jet_type_emb
        return emb

    def _object_tokens(self, obj):
        feats, type_id = obj[..., :-1], obj[..., -1].long().clamp(0, N_OBJ_TYPES - 1)
        return self.obj_proj(feats) + self.obj_type_emb(type_id)  # (B, O, D)

    def forward(self, batch):
        jet_tok = self._encode_jets(batch["parts"], batch["part_mask"], batch["jet_mask"])
        obj_tok = self._object_tokens(batch["obj"])
        tokens = torch.cat([jet_tok, obj_tok], dim=1)
        mask = torch.cat([batch["jet_mask"], batch["obj_mask"]], dim=1)
        logit = self.event(tokens, mask)
        return logit if self.use_log_loss else torch.sigmoid(logit)

    def _loss(self, batch):
        out = self(batch)
        y = batch["y"].view(-1, 1)
        w = batch["w"].view(-1, 1)
        # NLO label-flip: train on |w|, invert label where w < 0.
        neg = (w < 0).float()
        y_eff = y * (1 - neg) + (1 - y) * neg
        w_eff = w.abs()
        if self.use_log_loss:
            loss = F.binary_cross_entropy_with_logits(out, y_eff, reduction="none")
        else:
            loss = F.binary_cross_entropy(out, y_eff, reduction="none")
        return (loss * w_eff).sum() / w_eff.sum().clamp_min(1e-12)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", self._loss(batch), prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.NAdam(
            [p for p in self.parameters() if p.requires_grad], lr=self.lr,
            weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=self.hparams.callback_patience, gamma=self.hparams.callback_factor)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
