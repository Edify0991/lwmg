from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    provider: str = "CPUExecutionProvider"
    _encoder: Optional["ort.InferenceSession"] = None
    _decoder: Optional["ort.InferenceSession"] = None

    def _ensure_sessions(self) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime is required for frozen SONIC tracking")
        if self._encoder is None:
            self._encoder = ort.InferenceSession(str(self.encoder_path), providers=[self.provider])
        if self._decoder is None:
            self._decoder = ort.InferenceSession(str(self.decoder_path), providers=[self.provider])

    def run(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.encoder_path.exists() or not self.decoder_path.exists():
            return obs[: min(obs.shape[0], 12)] * 0.0
        self._ensure_sessions()
        encoder_input = {self._encoder.get_inputs()[0].name: obs.detach().cpu().numpy().astype(np.float32)[None, :]}
        latent = self._encoder.run(None, encoder_input)[0]
        decoder_input = {self._decoder.get_inputs()[0].name: latent}
        out = self._decoder.run(None, decoder_input)[0]
        return torch.from_numpy(out[0])
