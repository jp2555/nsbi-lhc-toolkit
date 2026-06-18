import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from nsbi_common_utils.lightning_tools.particle_cloud_dataset import WeightedParticleCloudDataset
from nsbi_common_utils.lightning_tools.hh_density_ratio_model import HHDensityRatioLightning
from nsbi_common_utils.lightning_tools.callbacks import LossHistory
from nsbi_common_utils.lightning_tools.cloud_spec import DEFAULT_SPEC
from nsbi_common_utils.training.utils import (
    save_model_constituents, predict_with_onnx_constituents, convert_score_to_ratio,
)

_KEYS = ["parts", "part_mask", "jet_mask", "obj", "obj_mask", "y", "w"]


class particle_density_ratio_trainer:
    """Constituent-cloud analog of density_ratio_trainer.

    Trains an HHDensityRatioLightning to estimate p_A/p_B from per-jet constituent
    clouds, exports to ONNX, and evaluates per-event density ratios. Mirrors the
    public surface used by the example scripts (train / evaluate_ratios).
    """

    def __init__(self, clouds, sample_name, output_name, path_to_models,
                 encoder_kind="stub", spec=DEFAULT_SPEC, freeze_backbone=False,
                 use_log_loss=False, encoder_kwargs=None):
        self.clouds = {k: np.asarray(clouds[k]) for k in _KEYS}
        self.sample_name = sample_name
        self.output_name = output_name
        self.path_to_models = path_to_models
        self.encoder_kind = encoder_kind
        self.spec = spec
        self.freeze_backbone = freeze_backbone
        self.use_log_loss = use_log_loss
        self.encoder_kwargs = encoder_kwargs or {}
        os.makedirs(path_to_models, exist_ok=True)
        self.model = None

    def train(self, number_of_epochs, batch_size, learning_rate,
              holdout_split=0.3, ensemble_index=0, n_heads=8, n_layers=2):
        ds = WeightedParticleCloudDataset(**{k: self.clouds[k] for k in _KEYS})
        n_hold = max(1, int(len(ds) * holdout_split))
        n_train = len(ds) - n_hold
        gen = torch.Generator().manual_seed(1000 + ensemble_index)
        train_ds, _ = random_split(ds, [n_train, n_hold], generator=gen)
        n_val = max(1, int(n_train * 0.1))
        train_ds, val_ds = random_split(
            train_ds, [n_train - n_val, n_val], generator=gen)

        self.model = HHDensityRatioLightning(
            encoder_kind=self.encoder_kind, spec=self.spec, n_heads=n_heads,
            n_layers=n_layers, learning_rate=learning_rate,
            use_log_loss=self.use_log_loss, freeze_backbone=self.freeze_backbone,
            encoder_kwargs=self.encoder_kwargs)

        history = LossHistory()
        trainer = pl.Trainer(
            max_epochs=number_of_epochs, accelerator="auto", devices="auto",
            logger=False, enable_checkpointing=False,
            # num_sanity_val_steps=0: the pre-training sanity pass otherwise fires
            # the validation callback once before epoch 0, leaving val_loss one entry
            # longer than train_loss (misaligned history). Disabling it makes both
            # per-epoch lists have length == number_of_epochs.
            num_sanity_val_steps=0,
            callbacks=[history, EarlyStopping(monitor="val_loss", patience=number_of_epochs)],
            enable_progress_bar=False)
        trainer.fit(
            self.model,
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size))

        self.model.eval()
        sample = next(iter(DataLoader(ds, batch_size=min(batch_size, len(ds)))))
        save_model_constituents(self.model, sample, f"{self.path_to_models}model{ensemble_index}.onnx")
        # LossHistory stores per-epoch train/val loss lists (callbacks.py)
        return {"train_loss": list(history.train_loss), "val_loss": list(history.val_loss)}

    def evaluate_ratios(self, clouds, ensemble_index=0):
        arrays = {k: np.asarray(clouds[k]) for k in ("parts", "part_mask", "jet_mask", "obj", "obj_mask")}
        score = predict_with_onnx_constituents(arrays, f"{self.path_to_models}model{ensemble_index}.onnx")
        if self.use_log_loss:
            score = 1.0 / (1.0 + np.exp(-score))
        score = np.clip(score, 1e-9, 1.0 - 1e-9)
        return convert_score_to_ratio(score)
