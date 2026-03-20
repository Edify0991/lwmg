import torch

from lwmg.envs.terminations import check_terminations


def test_soft_and_hard_flags() -> None:
    q = torch.tensor([3.0, 0.1])
    dq = torch.ones(2) * 6.0
    flags = check_terminations(q, dq, tracking_error=torch.tensor([0.6]))
    assert flags.growing_tracking_error_warning
    assert flags.hard_failure
