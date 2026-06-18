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
