import torch
from nsbi_common_utils.lightning_tools.particle_cloud_dataset import WeightedParticleCloudDataset


def test_dataset_yields_tensors(synth_batch):
    ds = WeightedParticleCloudDataset(**synth_batch)
    assert len(ds) == 32
    item = ds[0]
    assert set(item.keys()) == {"parts", "part_mask", "jet_mask", "obj", "obj_mask", "y", "w"}
    assert item["parts"].shape == (4, 64, 8)
    assert item["obj"].shape == (6, 6)
    assert item["y"].dtype == torch.float32 and item["w"].dtype == torch.float32


def test_dataloader_batches(synth_batch):
    from torch.utils.data import DataLoader
    ds = WeightedParticleCloudDataset(**synth_batch)
    batch = next(iter(DataLoader(ds, batch_size=8)))
    assert batch["parts"].shape == (8, 4, 64, 8)
    assert batch["jet_mask"].shape == (8, 4)
