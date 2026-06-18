from dataclasses import dataclass


@dataclass(frozen=True)
class CloudSpec:
    """Fixed-size padded representation of an event for the hierarchical model."""
    n_jets_max: int = 4
    n_part_max: int = 64
    f_part: int = 8
    n_obj_max: int = 6
    f_obj: int = 6
    embed_dim: int = 64


DEFAULT_SPEC = CloudSpec()

# sophon-ak4 layout: 8 AK4 jets x 128 constituents; `parts` columns are the 17 sophon
# pf_features followed by the 4 pf_vectors (px,py,pz,energy) -> f_part = 21. The
# SophonAK4Encoder slices cols 0:17 (x) and 17:21 (v).
SOPHON_SPEC = CloudSpec(n_jets_max=8, n_part_max=128, f_part=21, embed_dim=64)
