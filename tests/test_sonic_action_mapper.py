import torch

from lwmg.trackers.sonic_action_mapper import map_sonic_output_to_targets


def test_sonic_action_mapper_pad() -> None:
    out = map_sonic_output_to_targets(torch.ones(4), target_dim=6)
    assert out.shape[0] == 6
    assert out[-1] == 0
