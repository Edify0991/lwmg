from __future__ import annotations

import torch


def task_progress_reward(progress: torch.Tensor) -> torch.Tensor:
    return progress
