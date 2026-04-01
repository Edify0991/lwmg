from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from lwmg.envs.observations import Observation
from lwmg.references.reference_types import ReferenceTarget
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
        if not bool(self.obs_cfg.get("parse_ok", True)):
            err = str(self.obs_cfg.get("parse_error", "SONIC observation config parsing failed")).strip()
            raise ValueError(f"Invalid SONIC observation config: {err}")

        self.builder = SonicObservationBuilder(
            mapping_path=obs_mapping_path,
            observation_config_path=observation_config_path,
        )
        self.runner = SonicOnnxRunner(encoder_path=encoder_path, decoder_path=decoder_path, provider=provider)
        self.target_dim = target_dim

        self.runner.warmup()

        if self.builder.strict_mode and self.builder.strict_dim_check:
            if self.builder.expected_decoder_dim is not None:
                if self.runner.decoder_input_dim != self.builder.expected_decoder_dim:
                    raise ValueError(
                        f"SONIC decoder input mismatch: builder={self.builder.expected_decoder_dim} "
                        f"onnx={self.runner.decoder_input_dim}"
                    )
            if self.builder.expected_encoder_dim is not None:
                if self.runner.encoder_input_dim != self.builder.expected_encoder_dim:
                    raise ValueError(
                        f"SONIC encoder input mismatch: builder={self.builder.expected_encoder_dim} "
                        f"onnx={self.runner.encoder_input_dim}"
                    )

        self._history_by_env: dict[int, deque[Observation]] = defaultdict(lambda: deque(maxlen=64))

    def reset_env(self, env_id: int) -> None:
        self._history_by_env[env_id].clear()
        self.builder.reset_env(env_id)

    def _act_structured_impl(
        self,
        current: Observation,
        history: Iterable[Observation] | None = None,
        reference: ReferenceTarget | None = None,
        reference_generator: Any | None = None,
        env_id: int = 0,
        return_debug: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        if history is None:
            history_list = list(self._history_by_env[env_id])
        else:
            history_list = list(history)

        pack = self.builder.build_pack(
            current=current,
            history=history_list,
            reference=reference,
            reference_generator=reference_generator,
            env_id=env_id,
        )

        raw = self.runner.infer(
            decoder_obs=pack.decoder_obs,
            encoder_obs=pack.encoder_obs,
            token_slice=pack.token_slice,
        )
        action = map_sonic_output_to_targets(raw, self.target_dim)

        self._history_by_env[env_id].append(current)

        if not return_debug:
            return action, None

        debug: dict[str, Any] = {
            "raw_action": raw.detach().clone(),
            "decoder_obs": pack.decoder_obs.detach().clone(),
            "token_slice": pack.token_slice,
        }
        debug["encoder_obs"] = None if pack.encoder_obs is None else pack.encoder_obs.detach().clone()
        return action, debug

    def act_batch_from_structured(
        self,
        *,
        currents: Sequence[Observation],
        references: Sequence[ReferenceTarget | None] | None = None,
        reference_generator: Any | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        n = len(currents)
        if n <= 0:
            return torch.zeros((0, self.target_dim), dtype=torch.float32)

        if env_ids is None:
            env_ids = list(range(n))
        if len(env_ids) != n:
            raise ValueError(f"env_ids length {len(env_ids)} does not match currents length {n}")

        if references is None:
            ref_list: list[ReferenceTarget | None] = [None] * n
        else:
            if len(references) != n:
                raise ValueError(f"references length {len(references)} does not match currents length {n}")
            ref_list = list(references)

        packs = []
        for i, env_id in enumerate(env_ids):
            history_list = list(self._history_by_env[int(env_id)])
            pack = self.builder.build_pack(
                current=currents[i],
                history=history_list,
                reference=ref_list[i],
                reference_generator=reference_generator,
                env_id=int(env_id),
            )
            packs.append(pack)

        decoder_batch = torch.stack([pack.decoder_obs for pack in packs], dim=0)
        token_slice_ref = packs[0].token_slice
        token_slice_consistent = all(pack.token_slice == token_slice_ref for pack in packs)
        has_encoder_flags = [pack.encoder_obs is not None for pack in packs]

        if all(has_encoder_flags) and token_slice_consistent:
            encoder_batch = torch.stack([pack.encoder_obs for pack in packs if pack.encoder_obs is not None], dim=0)
            raw_batch = self.runner.infer_batch(
                decoder_obs_batch=decoder_batch,
                encoder_obs_batch=encoder_batch,
                token_slice=token_slice_ref,
            )
        elif not any(has_encoder_flags):
            raw_batch = self.runner.infer_batch(decoder_obs_batch=decoder_batch)
        else:
            # Mixed encoder availability or token_slice mismatch: safe fallback to per-sample infer.
            rows = []
            for pack in packs:
                rows.append(
                    self.runner.infer(
                        decoder_obs=pack.decoder_obs,
                        encoder_obs=pack.encoder_obs,
                        token_slice=pack.token_slice,
                    )
                )
            raw_batch = torch.stack(rows, dim=0)

        action_rows = []
        for i, env_id in enumerate(env_ids):
            action_rows.append(map_sonic_output_to_targets(raw_batch[i], self.target_dim))
            self._history_by_env[int(env_id)].append(currents[i])

        return torch.stack(action_rows, dim=0)

    def act_from_structured(
        self,
        current: Observation,
        history: Iterable[Observation] | None = None,
        reference: ReferenceTarget | None = None,
        reference_generator: Any | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        action, _ = self._act_structured_impl(
            current=current,
            history=history,
            reference=reference,
            reference_generator=reference_generator,
            env_id=env_id,
            return_debug=False,
        )
        return action

    def act_from_structured_debug(
        self,
        current: Observation,
        history: Iterable[Observation] | None = None,
        reference: ReferenceTarget | None = None,
        reference_generator: Any | None = None,
        env_id: int = 0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        action, debug = self._act_structured_impl(
            current=current,
            history=history,
            reference=reference,
            reference_generator=reference_generator,
            env_id=env_id,
            return_debug=True,
        )
        assert debug is not None
        return action, debug

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        raw = self.runner.infer(observation)
        return map_sonic_output_to_targets(raw, self.target_dim)

    def track_reference(self, reference_step: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        """Compatibility hook: reference -> SONIC (frozen) -> control target."""
        fused_obs = torch.cat([observation.flatten(), reference_step.flatten()], dim=0)
        raw = self.runner.infer(fused_obs)
        return map_sonic_output_to_targets(raw, self.target_dim)
