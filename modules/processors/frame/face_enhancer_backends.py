"""Face enhancer backend implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Iterable, List, Optional, Sequence

import numpy as np
import onnxruntime as ort
import torch

import gfpgan

from modules.typing import Frame


class FaceEnhancerBackendError(RuntimeError):
    """Base exception for backend errors."""

    def __init__(self, message: str, original: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.original = original


class BackendLoadError(FaceEnhancerBackendError):
    """Raised when a backend fails to initialise."""


class BackendInferenceError(FaceEnhancerBackendError):
    """Raised when a backend fails during inference."""


class FaceEnhancerBackend(ABC):
    """Interface for interchangeable face enhancer implementations."""

    backend_id: str
    display_name: str
    model_urls: Sequence[str] = ()

    def __init__(self, models_dir: str) -> None:
        self.models_dir = models_dir

    @classmethod
    def id(cls) -> str:
        return cls.backend_id

    @classmethod
    def required_model_urls(cls) -> Sequence[str]:
        return cls.model_urls

    @abstractmethod
    def load(
        self,
        *,
        device: Optional[torch.device] = None,
        execution_providers: Iterable[str] = (),
    ) -> None:
        """Load underlying inference runtime."""

    @abstractmethod
    def enhance(self, frame: Frame) -> Frame:
        """Apply enhancement and return a processed frame."""

    @abstractmethod
    def unload(self) -> None:
        """Release any loaded resources."""

    @property
    def device(self) -> Optional[torch.device]:  # pragma: no cover - default implementation
        return None

    @property
    def provider(self) -> Optional[str]:  # pragma: no cover - default implementation
        return None


class GfpganTorchBackend(FaceEnhancerBackend):
    """Torch-based GFPGAN backend."""

    backend_id = "gfpgan-torch"
    display_name = "GFPGAN (PyTorch)"
    model_urls = ("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",)

    def __init__(self, models_dir: str) -> None:
        super().__init__(models_dir)
        self._enhancer: Optional[gfpgan.GFPGANer] = None
        self._device: Optional[torch.device] = None

    def load(
        self,
        *,
        device: Optional[torch.device] = None,
        execution_providers: Iterable[str] = (),
    ) -> None:
        del execution_providers
        if device is None:
            raise BackendLoadError("A torch device is required for GFPGAN.")

        model_path = self._resolve_model_path("GFPGANv1.4.pth")
        try:
            self._enhancer = gfpgan.GFPGANer(model_path=model_path, upscale=1, device=device)
            self._device = device
        except Exception as exc:  # pragma: no cover - runtime specific
            self._enhancer = None
            self._device = None
            raise BackendLoadError("Failed to initialise GFPGAN backend.", exc) from exc

    def enhance(self, frame: Frame) -> Frame:
        if self._enhancer is None:
            raise BackendInferenceError("GFPGAN backend is not loaded.")
        try:
            _, _, enhanced = self._enhancer.enhance(frame, paste_back=True)
            return enhanced
        except Exception as exc:  # pragma: no cover - runtime specific
            raise BackendInferenceError("GFPGAN failed during inference.", exc) from exc

    def unload(self) -> None:
        self._enhancer = None
        self._device = None

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    def _resolve_model_path(self, filename: str) -> str:
        return os.path.join(self.models_dir, filename)


class CodeFormerOnnxBackend(FaceEnhancerBackend):
    """ONNX Runtime backend for CodeFormer restoration."""

    backend_id = "codeformer-onnx"
    display_name = "CodeFormer (ONNX)"
    model_urls = (
        "https://huggingface.co/spaces/sczhou/CodeFormer/resolve/main/weights/codeformer-v0.1.0.onnx",
    )

    def __init__(self, models_dir: str) -> None:
        super().__init__(models_dir)
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._providers: List[str] = []
        self._input_layout: Optional[Sequence[int]] = None

    def load(
        self,
        *,
        device: Optional[torch.device] = None,
        execution_providers: Iterable[str] = (),
    ) -> None:
        del device
        model_path = self._resolve_model_path("codeformer-v0.1.0.onnx")
        providers = self._normalise_providers(execution_providers)
        try:
            session = ort.InferenceSession(model_path, providers=providers)
        except FileNotFoundError as exc:
            raise BackendLoadError("CodeFormer ONNX model not found.", exc) from exc
        except Exception as exc:  # pragma: no cover - runtime specific
            raise BackendLoadError("Failed to initialise CodeFormer ONNX backend.", exc) from exc

        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name
        self._providers = list(session.get_providers())
        self._input_layout = session.get_inputs()[0].shape

    def enhance(self, frame: Frame) -> Frame:
        if self._session is None or self._input_name is None or self._output_name is None:
            raise BackendInferenceError("CodeFormer backend is not loaded.")

        tensor = self._prepare_input(frame)
        try:
            output = self._session.run([self._output_name], {self._input_name: tensor})[0]
        except Exception as exc:  # pragma: no cover - runtime specific
            raise BackendInferenceError("CodeFormer failed during inference.", exc) from exc

        return self._prepare_output(output)

    def unload(self) -> None:
        self._session = None
        self._input_name = None
        self._output_name = None
        self._providers = []
        self._input_layout = None

    @property
    def provider(self) -> Optional[str]:
        return self._providers[0] if self._providers else None

    def _resolve_model_path(self, filename: str) -> str:
        return os.path.join(self.models_dir, filename)

    def _normalise_providers(self, execution_providers: Iterable[str]) -> List[str]:
        requested = list(execution_providers)
        available = ort.get_available_providers()
        if requested:
            providers = [provider for provider in requested if provider in available]
        else:
            providers = []
        if not providers:
            providers = ["CPUExecutionProvider"]
        return providers

    def _prepare_input(self, frame: Frame) -> np.ndarray:
        data = frame.astype(np.float32)
        data = data / 255.0
        if self._input_layout and len(self._input_layout) == 4:
            layout = self._input_layout
            if layout[1] == 3 or layout[1] == "3":
                data = np.transpose(data, (2, 0, 1))
            data = data[np.newaxis, ...]
        else:
            data = data.reshape(1, *data.shape)
        return data.astype(np.float32)

    def _prepare_output(self, output: np.ndarray) -> Frame:
        data = output
        if data.ndim == 4:
            data = data[0]
        if data.shape[0] == 3 and data.ndim == 3:
            data = np.transpose(data, (1, 2, 0))
        data = np.clip(data, 0.0, 1.0)
        data = (data * 255.0).astype(np.uint8)
        return data


AVAILABLE_FACE_ENHANCER_BACKENDS = {
    backend.id(): backend
    for backend in (GfpganTorchBackend, CodeFormerOnnxBackend)
}

DEFAULT_FACE_ENHANCER_BACKEND = GfpganTorchBackend.backend_id
