"""Backend implementations for face enhancement models.

This module provides a small abstraction over the different face enhancement
implementations that Deep Live Cam can use.  While GFPGAN integrates tightly
with PyTorch, other solutions such as CodeFormer rely on ONNX Runtime.  The
``FaceEnhancerBackend`` base class keeps the interface consistent so the
high-level processor can swap implementations transparently.

Two concrete backends are provided:

``GfpganTorchBackend``
    A thin wrapper around :class:`gfpgan.GFPGANer`.  The underlying GFPGAN
    implementation already exposes an ``enhance`` method that performs all
    required logic, therefore the backend simply forwards the call.

``CodeFormerOnnxBackend``
    Runs CodeFormer through ONNX Runtime.  The backend expects a detected
    :class:`~modules.typing.Face` instance so that the face region can be
    cropped, optionally aligned, and replaced in the original frame after the
    model finishes.

The backends return the same tuple structure as GFPGAN (cropped faces, restored
faces, and the final full-resolution frame) so existing callers continue to
operate unchanged.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:  # Optional dependency, only required for CodeFormer.
    import onnxruntime as ort
except Exception:  # pragma: no cover - handled gracefully when unavailable.
    ort = None  # type: ignore[assignment]

from modules.typing import Face, Frame


logger = logging.getLogger(__name__)


class FaceEnhancerBackend(ABC):
    """Abstract base class for all face enhancer backends."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def enhance(
        self,
        frame: Frame,
        face: Optional[Face] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]], Frame]:
        """Enhance ``frame`` and optionally use ``face`` metadata.

        Parameters
        ----------
        frame:
            The BGR frame that should be enhanced.
        face:
            Optional detection metadata describing the face to enhance.  The
            base implementation makes no assumptions about the presence of the
            metadata, so backends that do not require it are free to ignore the
            parameter.
        **kwargs:
            Backend-specific keyword arguments forwarded to the implementation.

        Returns
        -------
        tuple
            A 3-tuple containing the cropped faces, restored faces, and the
            full-resolution frame.  The structure mirrors the values returned
            by :meth:`gfpgan.GFPGANer.enhance` for compatibility.
        """


class GfpganTorchBackend(FaceEnhancerBackend):
    """Adapter that forwards calls to ``gfpgan.GFPGANer``."""

    def __init__(self, enhancer: Any) -> None:
        super().__init__(name="gfpgan")
        self._enhancer = enhancer

    def enhance(
        self,
        frame: Frame,
        face: Optional[Face] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]], Frame]:
        # ``GFPGANer`` does not consume the optional ``face`` parameter.  The
        # signature accepts it for API compatibility with other backends.
        return self._enhancer.enhance(frame, **kwargs)


class CodeFormerOnnxBackend(FaceEnhancerBackend):
    """Face enhancement backend powered by a CodeFormer ONNX model.

    The backend uses a radial mask when compositing the restored face back into
    the frame.  The mask is generated from a normalized distance field so that
    the center of the face keeps a weight close to ``1`` while the values fade
    smoothly towards the borders.  ``mask_blur`` controls the Gaussian kernel
    applied to the mask, producing a soft elliptical feather that works for both
    ROI overlays and affine warps.  Empirically a kernel size around ``45``
    yielded a natural-looking transition while still keeping the restored facial
    details crisp.

    Before the alpha compositing step the backend optionally performs a simple
    colour transfer between the restored face and the destination region.
    Matching the first two Lab channel moments keeps the tonal values close to
    the surrounding frame and avoids sudden luminance jumps.  The behaviour can
    be tuned through ``color_correction_strength`` where ``0`` disables the
    adjustment and ``1`` applies the full transfer.
    """

    def __init__(
        self,
        session: "ort.InferenceSession",
        fidelity: float = 0.7,
        input_size: int = 512,
        face_padding: float = 0.1,
        mask_blur: int = 45,
        color_correction_strength: float = 1.0,
    ) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime is required for CodeFormerOnnxBackend")

        if not isinstance(session, ort.InferenceSession):  # type: ignore[arg-type]
            raise TypeError("session must be an onnxruntime.InferenceSession instance")

        super().__init__(name="codeformer")
        self.session = session
        self.fidelity = float(np.clip(fidelity, 0.0, 1.0))
        self.input_size = int(max(32, input_size))
        self.face_padding = max(0.0, float(face_padding))
        self.mask_blur = int(max(0, mask_blur))  # 45 keeps the mask feather gentle by default.
        self.color_correction_strength = float(np.clip(color_correction_strength, 0.0, 1.0))

        inputs = session.get_inputs()
        if not inputs:
            raise RuntimeError("CodeFormer ONNX model does not expose any inputs")

        self.input_name = inputs[0].name
        self.fidelity_name: Optional[str] = None
        if len(inputs) > 1:
            # Some exports keep the fidelity weight as an explicit input.  It is
            # usually named ``w`` or ``fidelity``.  We only support scalar
            # tensors, so inspect the shape as well.
            for tensor in inputs[1:]:
                if tensor.shape == [] or tensor.shape == [1] or tensor.shape == [1, 1]:
                    self.fidelity_name = tensor.name
                    break

        # Canonical five point template extracted from ArcFace; we only need
        # the first three points for affine alignment.
        template = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
            ],
            dtype=np.float32,
        )
        self._canonical_points = template * (self.input_size / 112.0)

    def set_mask_blur(self, mask_blur: int) -> None:
        """Update the Gaussian blur kernel size used for the composite mask."""

        self.mask_blur = int(max(0, mask_blur))

    def set_color_correction_strength(self, strength: float) -> None:
        """Update the Lab colour transfer strength applied during blending."""

        self.color_correction_strength = float(np.clip(strength, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _prepare_input(self, aligned_face: np.ndarray) -> Dict[str, np.ndarray]:
        face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        face_tensor = face_rgb.astype(np.float32) / 255.0
        face_tensor = (face_tensor - 0.5) / 0.5
        face_tensor = np.transpose(face_tensor, (2, 0, 1))[None, ...]

        inputs: Dict[str, np.ndarray] = {self.input_name: face_tensor}
        if self.fidelity_name:
            inputs[self.fidelity_name] = np.array([self.fidelity], dtype=np.float32)
        return inputs

    def _run_session(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        outputs = self.session.run(None, inputs)
        if not outputs:
            raise RuntimeError("CodeFormer ONNX model produced no outputs")

        restored = outputs[0]
        if restored.ndim != 4:
            raise RuntimeError(
                f"Unexpected output shape from CodeFormer model: {restored.shape}"
            )

        restored_face = restored[0]
        restored_face = np.transpose(restored_face, (1, 2, 0))
        restored_face = (restored_face * 0.5 + 0.5) * 255.0
        restored_face = np.clip(restored_face, 0, 255).astype(np.uint8)
        restored_face = cv2.cvtColor(restored_face, cv2.COLOR_RGB2BGR)
        return restored_face

    @staticmethod
    def _clamp_bbox(
        bbox: Iterable[float],
        frame_width: int,
        frame_height: int,
        padding: float = 0.0,
    ) -> Optional[Tuple[int, int, int, int]]:
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None

        width = x2 - x1
        height = y2 - y1
        if width <= 1 or height <= 1:
            return None

        pad_x = width * padding
        pad_y = height * padding

        x1 -= pad_x
        x2 += pad_x
        y1 -= pad_y
        y2 += pad_y

        x1 = int(np.clip(x1, 0, frame_width - 1))
        y1 = int(np.clip(y1, 0, frame_height - 1))
        x2 = int(np.clip(x2, x1 + 1, frame_width))
        y2 = int(np.clip(y2, y1 + 1, frame_height))
        return x1, y1, x2, y2

    def _create_mask(self, height: int, width: int) -> np.ndarray:
        """Create a smooth elliptical mask that fades towards the edges."""

        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        distance = np.sqrt(xx**2 + yy**2)

        mask = np.clip(1.0 - distance, 0.0, 1.0)
        mask = mask**2

        if self.mask_blur > 0:
            kernel = max(1, self.mask_blur)
            if kernel % 2 == 0:
                kernel += 1
            mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)

        mask = np.clip(mask, 0.0, 1.0)
        max_value = float(mask.max())
        if max_value > 0:
            mask /= max_value

        return mask[..., None]

    def _match_color(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Match ``source`` colour statistics to ``reference`` using Lab moments."""

        if self.color_correction_strength <= 0.0:
            return source

        if mask is None:
            weights = np.ones(source.shape[:2], dtype=np.float32)
        else:
            weights = mask.astype(np.float32)
            if weights.ndim == 3:
                weights = weights[..., 0]

        total_weight = float(weights.sum())
        if total_weight <= 1e-6:
            return source

        src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

        weights_flat = weights.reshape(-1, 1)
        src_flat = src_lab.reshape(-1, 3)
        ref_flat = ref_lab.reshape(-1, 3)

        src_mean = (src_flat * weights_flat).sum(axis=0) / total_weight
        ref_mean = (ref_flat * weights_flat).sum(axis=0) / total_weight

        src_var = (weights_flat * (src_flat - src_mean) ** 2).sum(axis=0) / total_weight
        ref_var = (weights_flat * (ref_flat - ref_mean) ** 2).sum(axis=0) / total_weight

        src_std = np.sqrt(np.maximum(src_var, 1e-6))
        ref_std = np.sqrt(np.maximum(ref_var, 1e-6))

        adjusted_flat = ((src_flat - src_mean) / src_std) * ref_std + ref_mean
        adjusted_lab = adjusted_flat.reshape(src_lab.shape)

        strength = self.color_correction_strength
        corrected_lab = src_lab + strength * (adjusted_lab - src_lab)

        corrected_bgr = cv2.cvtColor(np.clip(corrected_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return corrected_bgr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def enhance(
        self,
        frame: Frame,
        face: Optional[Face] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]], Frame]:
        mask_blur_override = kwargs.pop("mask_blur", None)
        if mask_blur_override is not None:
            self.set_mask_blur(mask_blur_override)

        color_strength_override = kwargs.pop("color_correction_strength", None)
        if color_strength_override is not None:
            self.set_color_correction_strength(color_strength_override)

        if face is None:
            logger.debug("CodeFormerOnnxBackend.enhance called without a face; returning original frame")
            return None, None, frame

        frame_height, frame_width = frame.shape[:2]
        bbox = getattr(face, "bbox", None)
        if bbox is None:
            logger.debug("Face metadata does not include a bounding box; skipping enhancement")
            return None, None, frame

        clamped_bbox = self._clamp_bbox(bbox, frame_width, frame_height, self.face_padding)
        if clamped_bbox is None:
            logger.debug("Invalid bounding box detected %s; skipping enhancement", bbox)
            return None, None, frame

        x1, y1, x2, y2 = clamped_bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            logger.debug("Empty region of interest extracted from frame; skipping enhancement")
            return None, None, frame

        kps = getattr(face, "kps", None)
        use_alignment = kps is not None and len(kps) >= 3
        try:
            if use_alignment:
                src_points = np.asarray(kps[:3], dtype=np.float32)
                affine = cv2.getAffineTransform(src_points, self._canonical_points)
                aligned = cv2.warpAffine(
                    frame,
                    affine,
                    (self.input_size, self.input_size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
                inverse_affine = cv2.invertAffineTransform(affine)
            else:
                inverse_affine = None
                aligned = cv2.resize(roi, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

            model_inputs = self._prepare_input(aligned)
            restored_face = self._run_session(model_inputs)

            output_frame = frame.copy()

            if inverse_affine is not None:
                canvas = cv2.warpAffine(
                    restored_face,
                    inverse_affine,
                    (frame_width, frame_height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
                base_mask = self._create_mask(self.input_size, self.input_size).squeeze(-1)
                mask = cv2.warpAffine(
                    base_mask,
                    inverse_affine,
                    (frame_width, frame_height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
                corrected_canvas = self._match_color(canvas, output_frame, mask)
                mask_3c = mask[..., None]
                inv_mask_3c = np.float32(1.0) - mask_3c
                mask = np.clip(mask, 0.0, 1.0)
                mask = mask[..., None]

                blended = (
                    corrected_canvas.astype(np.float32) * mask_3c
                    + output_frame.astype(np.float32) * inv_mask_3c
                )
                output_frame = np.clip(blended, 0, 255).astype(np.uint8)
            else:
                resized_face = cv2.resize(
                    restored_face,
                    (x2 - x1, y2 - y1),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask = self._create_mask(y2 - y1, x2 - x1)
                base_region = output_frame[y1:y2, x1:x2]
                corrected_face = self._match_color(resized_face, base_region, mask)
                base_region_float = base_region.astype(np.float32)
                mask_float = mask.astype(np.float32)
                inv_mask = np.float32(1.0) - mask_float
                blended = (
                    corrected_face.astype(np.float32) * mask_float
                    + base_region_float * inv_mask
                )
                output_frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

            return None, [restored_face], output_frame
        except Exception as exc:  # pragma: no cover - defensive logging path.
            logger.exception("CodeFormer enhancement failed: %s", exc)
            return None, None, frame

