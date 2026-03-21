from pathlib import Path

import torch

from lwmg.references.flows.flow_matching_generator import FlowMatchingGenerator
from lwmg.trackers.sonic_frozen_tracker_adapter import SonicFrozenTrackerAdapter


def test_guided_flow_output_with_sonic_stub(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text("mapping:\n  q: q\n  dq: dq\n")
    obs_cfg = tmp_path / "observation_config.yaml"
    obs_cfg.write_text("history_steps: 1\n")

    tracker = SonicFrozenTrackerAdapter(
        encoder_path=tmp_path / "missing_encoder.onnx",
        decoder_path=tmp_path / "missing_decoder.onnx",
        observation_config_path=obs_cfg,
        target_dim=12,
        obs_mapping_path=mapping,
    )
    flow = FlowMatchingGenerator()
    ref = flow.sample_unguided(1, 4, torch.randn(1, 16))
    act = tracker.track_reference(ref[:, 0], torch.zeros(24))
    assert act.shape[0] == 12
