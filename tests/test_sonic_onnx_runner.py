from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from lwmg.trackers.sonic_onnx_runner import SonicOnnxRunner


def test_sonic_onnx_runner_mock_mode_when_missing() -> None:
    runner = SonicOnnxRunner(Path("/tmp/missing_encoder.onnx"), Path("/tmp/missing_decoder.onnx"), provider="cpu")
    runner.warmup(obs_dim=32)
    out = runner.infer(torch.ones(32))
    assert runner.mock_mode
    assert out.shape[0] == 29
