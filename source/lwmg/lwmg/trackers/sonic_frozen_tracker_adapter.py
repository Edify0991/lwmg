from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from lwmg.envs.observations import Observation
from lwmg.sonic_io.sonic_config_parser import parse_observation_config
from .base_tracker import BaseTracker
from .sonic_action_mapper import map_sonic_output_to_targets
from .sonic_obs_builder import SonicObservationBuilder
from .sonic_obs_mapping import SonicObservationMapping
from .sonic_onnx_runner import SonicOnnxRunner


class SonicFrozenTrackerAdapter(BaseTracker):
    def __init__(
        self,
        encoder_path: Path,
        decoder_path: Path,
        observation_config_path: Path,
        target_dim: int,
    ) -> None:
        self.obs_cfg = parse_observation_config(observation_config_path)
        self.mapping = SonicObservationMapping.default()
        self.builder = SonicObservationBuilder(self.mapping, history_steps=self.obs_cfg.get("history_steps", 1))
        self.runner = SonicOnnxRunner(encoder_path=encoder_path, decoder_path=decoder_path)
        self.target_dim = target_dim

    def act_from_structured(
        self, current: Observation, history: Iterable[Observation] | None = None
    ) -> torch.Tensor:
        obs_vec = self.builder.build(current=current, history=history)
        raw = self.runner.run(obs_vec)
        return map_sonic_output_to_targets(raw, self.target_dim)

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        raw = self.runner.run(observation)
        return map_sonic_output_to_targets(raw, self.target_dim)
