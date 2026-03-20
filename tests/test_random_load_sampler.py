from lwmg.envs.random_load_sampler import RandomLoadSampler


def test_random_load_sampler_ranges() -> None:
    sample = RandomLoadSampler((1.0, 2.0)).sample()
    assert 1.0 <= sample.payload_mass <= 2.0
    assert sample.payload_com_shift.shape[0] == 3
    assert sample.hand_wrench_lr.shape == (2, 3)
