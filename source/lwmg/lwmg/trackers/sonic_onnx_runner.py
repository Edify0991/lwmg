from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None


@dataclass
class SonicOnnxRunner:
    encoder_path: Path
    decoder_path: Path
    provider: str = "cpu"
    mock_output_dim: int = 29
    debug: bool = False
    _encoder: Any = field(default=None, init=False, repr=False)
    _decoder: Any = field(default=None, init=False, repr=False)
    _mock_mode: bool = field(default=False, init=False)

    def _provider_list(self) -> list[str]:
        key = self.provider.lower()
        if key == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _inputs_ready(self) -> bool:
        return self.encoder_path.exists() and self.decoder_path.exists() and ort is not None

    def _ensure_sessions(self) -> None:
        if not self._inputs_ready():
            self._mock_mode = True
            return

        providers = self._provider_list()
        if self._encoder is None:
            self._encoder = ort.InferenceSession(str(self.encoder_path), providers=providers)
        if self._decoder is None:
            self._decoder = ort.InferenceSession(str(self.decoder_path), providers=providers)

    @property
    def mock_mode(self) -> bool:
        return self._mock_mode

    def warmup(self, obs_dim: int = 256) -> None:
        self._ensure_sessions()
        dummy = torch.zeros(obs_dim, dtype=torch.float32)
        _ = self.infer(dummy)
        if self.debug:
            print(f"[SonicOnnxRunner] warmup complete, mock_mode={self._mock_mode}, obs_dim={obs_dim}")

    def infer(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        self._ensure_sessions()

        obs_np = obs_tensor.detach().cpu().flatten().numpy().astype(np.float32)[None, :]
        if self._mock_mode:
            out = np.zeros((1, self.mock_output_dim), dtype=np.float32)
            if out.shape[1] > 0:
                out[0, : min(self.mock_output_dim, obs_np.shape[1])] = obs_np[0, : min(self.mock_output_dim, obs_np.shape[1])]
            return torch.from_numpy(out[0])

        encoder_input_name = self._encoder.get_inputs()[0].name
        latent = self._encoder.run(None, {encoder_input_name: obs_np})[0]
        decoder_input_name = self._decoder.get_inputs()[0].name
        decoded = self._decoder.run(None, {decoder_input_name: latent})[0]

        out = torch.from_numpy(np.asarray(decoded[0], dtype=np.float32))
        if self.debug:
            print(f"[SonicOnnxRunner] infer obs_dim={obs_np.shape[1]} out_dim={out.numel()}")
        return out
