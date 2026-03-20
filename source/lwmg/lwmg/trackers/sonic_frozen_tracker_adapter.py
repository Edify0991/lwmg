from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from lwmg.envs.observations import Observation
from lwmg.sonic_io.sonic_config_parser import parse_observation_config
from .base_tracker import BaseTracker
from .sonic_action_mapper import map_sonic_output_to_targets
from .sonic_obs_builder import SonicObservationBuilder
from .sonic_onnx_runner import SonicOnnxRunner


class SonicFrozenTrackerAdapter(BaseTracker):
    def __init__(
        self,
        encoder_path: Path,
        decoder_path: Path,
        observation_config_path: Path,
        target_dim: int,
        obs_mapping_path: Path,
        provider: str = "cpu",
    ) -> None:
        self.obs_cfg = parse_observation_config(observation_config_path)
        self.builder = SonicObservationBuilder(
            mapping_path=obs_mapping_path,
            observation_config_path=observation_config_path,
        )
        self.runner = SonicOnnxRunner(encoder_path=encoder_path, decoder_path=decoder_path, provider=provider)
        self.target_dim = target_dim

    def act_from_structured(
        self, current: Observation, history: Iterable[Observation] | None = None
    ) -> torch.Tensor:
        obs_vec = self.builder.build(current=current, history=history)
        raw = self.runner.infer(obs_vec)
        return map_sonic_output_to_targets(raw, self.target_dim)

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        raw = self.runner.infer(observation)
        return map_sonic_output_to_targets(raw, self.target_dim)
