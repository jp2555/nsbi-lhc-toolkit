from nsbi_common_utils.lightning_tools.cloud_spec import CloudSpec, DEFAULT_SPEC


def test_default_spec_shapes():
    s = DEFAULT_SPEC
    assert (s.n_jets_max, s.n_part_max, s.f_part) == (4, 64, 8)
    assert (s.n_obj_max, s.f_obj, s.embed_dim) == (6, 6, 64)


def test_spec_is_overridable():
    s = CloudSpec(n_jets_max=2)
    assert s.n_jets_max == 2 and s.n_part_max == 64
