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


def _shape_last_dim(shape: list[Any], fallback: int) -> int:
    if not shape:
        return fallback
    last = shape[-1]
    if isinstance(last, int) and last > 0:
        return last
    return fallback


@dataclass
class SonicOnnxRunner:
    encoder_path: Path
    decoder_path: Path
    provider: str = "cpu"
    mock_output_dim: int = 29
    debug: bool = False
    mock_encoder_input_dim: int = 1762
    mock_decoder_input_dim: int = 994
    mock_token_dim: int = 64

    _encoder: Any = field(default=None, init=False, repr=False)
    _decoder: Any = field(default=None, init=False, repr=False)
    _mock_mode: bool = field(default=False, init=False)
    _encoder_input_dim: int | None = field(default=None, init=False)
    _decoder_input_dim: int | None = field(default=None, init=False)
    _token_dim: int | None = field(default=None, init=False)
    _provider_warned: bool = field(default=False, init=False, repr=False)
    _encoder_input_name: str | None = field(default=None, init=False, repr=False)
    _decoder_input_name: str | None = field(default=None, init=False, repr=False)
    _encoder_batch_fallback_warned: bool = field(default=False, init=False, repr=False)
    _decoder_batch_fallback_warned: bool = field(default=False, init=False, repr=False)

    def _provider_list(self) -> list[str]:
        key = self.provider.lower().strip()
        requested = ["CPUExecutionProvider"]
        if key.startswith("cuda") or "cuda" in key or key.startswith("gpu"):
            requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        if ort is None:
            return requested

        try:
            available = list(ort.get_available_providers())
        except Exception:
            return requested

        selected = [p for p in requested if p in available]
        if not selected:
            selected = [p for p in ["CPUExecutionProvider"] if p in available]
        if not selected and available:
            selected = [available[0]]

        if "CUDAExecutionProvider" in requested and "CUDAExecutionProvider" not in selected and not self._provider_warned:
            print(
                "[SonicOnnxRunner] CUDAExecutionProvider unavailable; "
                f"falling back to {selected}. available={available}"
            )
            self._provider_warned = True

        return selected

    def _inputs_ready(self) -> bool:
        return self.encoder_path.exists() and self.decoder_path.exists() and ort is not None

    def _ensure_sessions(self) -> None:
        if not self._inputs_ready():
            self._mock_mode = True
            return

        providers = self._provider_list()
        if self._encoder is None:
            self._encoder = ort.InferenceSession(str(self.encoder_path), providers=providers)
            enc_in = self._encoder.get_inputs()[0]
            enc_out = self._encoder.get_outputs()[0]
            self._encoder_input_name = enc_in.name
            self._encoder_input_dim = _shape_last_dim(enc_in.shape, self.mock_encoder_input_dim)
            self._token_dim = _shape_last_dim(enc_out.shape, self.mock_token_dim)
        if self._decoder is None:
            self._decoder = ort.InferenceSession(str(self.decoder_path), providers=providers)
            dec_in = self._decoder.get_inputs()[0]
            dec_out = self._decoder.get_outputs()[0]
            self._decoder_input_name = dec_in.name
            self._decoder_input_dim = _shape_last_dim(dec_in.shape, self.mock_decoder_input_dim)
            self.mock_output_dim = _shape_last_dim(dec_out.shape, self.mock_output_dim)

    @property
    def mock_mode(self) -> bool:
        return self._mock_mode

    @property
    def encoder_input_dim(self) -> int:
        self._ensure_sessions()
        return self._encoder_input_dim or self.mock_encoder_input_dim

    @property
    def decoder_input_dim(self) -> int:
        self._ensure_sessions()
        return self._decoder_input_dim or self.mock_decoder_input_dim

    @property
    def token_dim(self) -> int:
        self._ensure_sessions()
        return self._token_dim or self.mock_token_dim

    def warmup(self, obs_dim: int = 256) -> None:
        self._ensure_sessions()
        _ = self.decode_batch(
            torch.zeros((1, self.decoder_input_dim if self.decoder_input_dim > 0 else obs_dim), dtype=torch.float32)
        )
        _ = self.encode_batch(
            torch.zeros((1, self.encoder_input_dim if self.encoder_input_dim > 0 else obs_dim), dtype=torch.float32)
        )
        if self.debug:
            print(
                "[SonicOnnxRunner] warmup complete "
                f"mock_mode={self._mock_mode} enc_in={self.encoder_input_dim} dec_in={self.decoder_input_dim}"
            )

    def _as_batch_2d(self, obs: torch.Tensor, *, expected_dim: int, name: str) -> torch.Tensor:
        x = obs.detach().cpu().to(dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(f"{name} must be rank-1 or rank-2, got shape={tuple(x.shape)}")
        if x.shape[1] != expected_dim:
            raise ValueError(f"{name} dim mismatch: expected {expected_dim}, got {x.shape[1]}")
        return x

    def _mock_encode_batch(self, obs_batch: torch.Tensor) -> torch.Tensor:
        n = int(obs_batch.shape[0])
        out = torch.zeros((n, self.token_dim), dtype=torch.float32)
        copy_n = min(out.shape[1], obs_batch.shape[1])
        out[:, :copy_n] = obs_batch[:, :copy_n]
        return out

    def _mock_decode_batch(self, obs_batch: torch.Tensor) -> torch.Tensor:
        n = int(obs_batch.shape[0])
        out = torch.zeros((n, self.mock_output_dim), dtype=torch.float32)
        copy_n = min(out.shape[1], obs_batch.shape[1])
        out[:, :copy_n] = obs_batch[:, :copy_n]
        return out

    def encode_batch(self, encoder_obs_batch: torch.Tensor) -> torch.Tensor:
        self._ensure_sessions()
        batch = self._as_batch_2d(
            encoder_obs_batch,
            expected_dim=self.encoder_input_dim,
            name="encoder_obs_batch",
        )

        if self._mock_mode:
            return self._mock_encode_batch(batch)

        obs_np = np.ascontiguousarray(batch.numpy().astype(np.float32, copy=False))
        assert self._encoder is not None
        assert self._encoder_input_name is not None
        try:
            latent = self._encoder.run(None, {self._encoder_input_name: obs_np})[0]
            return torch.from_numpy(np.asarray(latent, dtype=np.float32))
        except Exception as exc:
            if obs_np.shape[0] <= 1:
                raise
            if not self._encoder_batch_fallback_warned:
                print(
                    "[SonicOnnxRunner] encoder batch inference failed; "
                    f"fallback to per-sample loop. error={exc}"
                )
                self._encoder_batch_fallback_warned = True
            rows = []
            for i in range(obs_np.shape[0]):
                one = self._encoder.run(None, {self._encoder_input_name: obs_np[i : i + 1]})[0]
                rows.append(np.asarray(one[0], dtype=np.float32))
            return torch.from_numpy(np.stack(rows, axis=0))

    def decode_batch(self, decoder_obs_batch: torch.Tensor) -> torch.Tensor:
        self._ensure_sessions()
        batch = self._as_batch_2d(
            decoder_obs_batch,
            expected_dim=self.decoder_input_dim,
            name="decoder_obs_batch",
        )

        if self._mock_mode:
            return self._mock_decode_batch(batch)

        obs_np = np.ascontiguousarray(batch.numpy().astype(np.float32, copy=False))
        assert self._decoder is not None
        assert self._decoder_input_name is not None
        try:
            decoded = self._decoder.run(None, {self._decoder_input_name: obs_np})[0]
            return torch.from_numpy(np.asarray(decoded, dtype=np.float32))
        except Exception as exc:
            if obs_np.shape[0] <= 1:
                raise
            if not self._decoder_batch_fallback_warned:
                print(
                    "[SonicOnnxRunner] decoder batch inference failed; "
                    f"fallback to per-sample loop. error={exc}"
                )
                self._decoder_batch_fallback_warned = True
            rows = []
            for i in range(obs_np.shape[0]):
                one = self._decoder.run(None, {self._decoder_input_name: obs_np[i : i + 1]})[0]
                rows.append(np.asarray(one[0], dtype=np.float32))
            return torch.from_numpy(np.stack(rows, axis=0))

    def encode(self, encoder_obs: torch.Tensor) -> torch.Tensor:
        out = self.encode_batch(encoder_obs)
        return out[0]

    def decode(self, decoder_obs: torch.Tensor) -> torch.Tensor:
        out = self.decode_batch(decoder_obs)
        return out[0]

    def infer_batch(
        self,
        decoder_obs_batch: torch.Tensor,
        encoder_obs_batch: torch.Tensor | None = None,
        token_slice: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        decoder_batch = self._as_batch_2d(
            decoder_obs_batch,
            expected_dim=self.decoder_input_dim,
            name="decoder_obs_batch",
        )

        if encoder_obs_batch is not None:
            token_batch = self.encode_batch(encoder_obs_batch)
            if token_batch.shape[0] != decoder_batch.shape[0]:
                raise ValueError(
                    f"Batch size mismatch between decoder ({decoder_batch.shape[0]}) and encoder ({token_batch.shape[0]})"
                )
            start, end = token_slice or (0, self.token_dim)
            if end < start:
                raise ValueError(f"Invalid token slice {token_slice}")
            if end > decoder_batch.shape[1]:
                raise ValueError(
                    f"Token slice {token_slice} exceeds decoder observation dim {decoder_batch.shape[1]}"
                )
            decoder_batch = decoder_batch.clone()
            token_n = min(token_batch.shape[1], end - start)
            decoder_batch[:, start : start + token_n] = token_batch[:, :token_n]

        out = self.decode_batch(decoder_batch)
        if self.debug:
            print(
                f"[SonicOnnxRunner] infer_batch n={decoder_batch.shape[0]} dec_in={decoder_batch.shape[1]} "
                f"action_dim={out.shape[1]} mock_mode={self._mock_mode}"
            )
        return out

    def infer(
        self,
        decoder_obs: torch.Tensor,
        encoder_obs: torch.Tensor | None = None,
        token_slice: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        decoder_batch = decoder_obs.detach().flatten().to(dtype=torch.float32).unsqueeze(0)
        encoder_batch = None
        if encoder_obs is not None:
            encoder_batch = encoder_obs.detach().flatten().to(dtype=torch.float32).unsqueeze(0)
        out = self.infer_batch(
            decoder_obs_batch=decoder_batch,
            encoder_obs_batch=encoder_batch,
            token_slice=token_slice,
        )
        return out[0]
