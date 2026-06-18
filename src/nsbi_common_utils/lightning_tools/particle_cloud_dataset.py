import numpy as np
import torch
from torch.utils.data import Dataset


class WeightedParticleCloudDataset(Dataset):
    """Yields padded per-jet constituent clouds + event object tokens + (label, weight).

    All inputs are numpy arrays with a leading event axis:
      parts:     (N, n_jets_max, n_part_max, f_part)
      part_mask: (N, n_jets_max, n_part_max)   1.0 = real particle
      jet_mask:  (N, n_jets_max)               1.0 = real jet
      obj:       (N, n_obj_max, f_obj)
      obj_mask:  (N, n_obj_max)                1.0 = real object token
      y:         (N,)  binary label (1 = numerator / hypothesis A)
      w:         (N,)  per-event weight (already class-normalised by the caller)
    """

    def __init__(self, parts, part_mask, jet_mask, obj, obj_mask, y, w):
        self.parts = torch.as_tensor(np.asarray(parts), dtype=torch.float32)
        self.part_mask = torch.as_tensor(np.asarray(part_mask), dtype=torch.float32)
        self.jet_mask = torch.as_tensor(np.asarray(jet_mask), dtype=torch.float32)
        self.obj = torch.as_tensor(np.asarray(obj), dtype=torch.float32)
        self.obj_mask = torch.as_tensor(np.asarray(obj_mask), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        self.w = torch.as_tensor(np.asarray(w), dtype=torch.float32)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return {
            "parts": self.parts[i], "part_mask": self.part_mask[i],
            "jet_mask": self.jet_mask[i], "obj": self.obj[i],
            "obj_mask": self.obj_mask[i], "y": self.y[i], "w": self.w[i],
        }
