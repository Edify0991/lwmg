from __future__ import annotations

from pathlib import Path


def locate_sonic_checkpoints(root_dir: Path) -> dict[str, Path]:
    expected = ["model_encoder.onnx", "model_decoder.onnx", "planner_sonic.onnx", "observation_config.yaml"]
    return {name: root_dir / name for name in expected}
