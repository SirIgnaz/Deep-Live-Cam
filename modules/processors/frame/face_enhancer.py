from typing import Any, List, Optional, Tuple
import cv2
import threading
import os

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
import platform
import torch

from modules.processors.frame.face_enhancer_backends import (
    AVAILABLE_FACE_ENHANCER_BACKENDS,
    DEFAULT_FACE_ENHANCER_BACKEND,
    BackendInferenceError,
    BackendLoadError,
    FaceEnhancerBackend,
    GfpganTorchBackend,
)

TORCH_DIRECTML_AVAILABLE = False
DIRECTML_DEVICE = None
try:
    import torch_directml

    DIRECTML_DEVICE = torch_directml.device()
    TORCH_DIRECTML_AVAILABLE = True
except ImportError:
    pass
from modules.utilities import (
    conditional_download,
    is_image,
    is_video,
)

FACE_ENHANCER: Optional[FaceEnhancerBackend] = None
DIRECTML_FACE_ENHANCER_DISABLED = False
DIRECTML_FACE_ENHANCER_FORCED_CPU = False
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-ENHANCER"
MAX_DIRECTML_ERROR_LENGTH = 200

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)
ALLOW_DIRECTML_FACE_ENHANCER = (
    os.environ.get("DLC_ALLOW_DIRECTML_FACE_ENHANCER", "").strip().lower()
    in ("1", "true", "yes", "on")
)


def _get_configured_backend_id() -> str:
    env_override = os.environ.get("DLC_FACE_ENHANCER_BACKEND", "").strip().lower()
    if env_override:
        if env_override in AVAILABLE_FACE_ENHANCER_BACKENDS:
            return env_override
        update_status(
            (
                f"Unknown face enhancer backend '{env_override}'. "
                "Falling back to default."
            ),
            NAME,
        )

    configured = (modules.globals.face_enhancer_backend or "").strip().lower()
    if configured in AVAILABLE_FACE_ENHANCER_BACKENDS:
        return configured

    if configured:
        update_status(
            (
                f"Face enhancer backend '{configured}' is not available. "
                f"Using '{DEFAULT_FACE_ENHANCER_BACKEND}' instead."
            ),
            NAME,
        )

    return DEFAULT_FACE_ENHANCER_BACKEND


def pre_check() -> bool:
    backend_id = _get_configured_backend_id()
    backend_cls = AVAILABLE_FACE_ENHANCER_BACKENDS.get(
        backend_id, AVAILABLE_FACE_ENHANCER_BACKENDS[DEFAULT_FACE_ENHANCER_BACKEND]
    )
    required_urls = list(backend_cls.required_model_urls())
    if required_urls:
        conditional_download(models_dir, required_urls)
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(
        modules.globals.target_path
    ):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


TENSORRT_AVAILABLE = False
try:
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
except ImportError as im:
    print(f"TensorRT is not available: {im}")
    pass
except Exception as e:
    print(f"TensorRT is not available: {e}")
    pass

def _device_name(device: Optional[torch.device]) -> str:
    if device is None:
        return "Unknown"
    if TORCH_DIRECTML_AVAILABLE and device == DIRECTML_DEVICE:
        return "DirectML"
    return device.type.upper()


def _directml_error_summary(error: Exception) -> str:
    message = str(error).strip()
    if not message:
        return "Unknown error"
    if len(message) > MAX_DIRECTML_ERROR_LENGTH:
        message = message[: MAX_DIRECTML_ERROR_LENGTH - 1].rstrip() + "…"
    return message


def _select_torch_device(force_device: Optional[torch.device]) -> Tuple[torch.device, List[str]]:
    selected_device: Optional[torch.device] = None
    device_priority: List[str] = []

    if force_device is not None:
        if (
            TORCH_DIRECTML_AVAILABLE
            and DIRECTML_FACE_ENHANCER_DISABLED
            and force_device == DIRECTML_DEVICE
        ):
            update_status(
                (
                    "DirectML face enhancement previously failed; "
                    "using CPU fallback instead."
                ),
                NAME,
            )
            selected_device = torch.device("cpu")
            device_priority.append("CPU (fallback)")
        else:
            selected_device = force_device
            device_priority.append(_device_name(selected_device))
    else:
        if (
            'DmlExecutionProvider' in modules.globals.execution_providers
            and TORCH_DIRECTML_AVAILABLE
            and not DIRECTML_FACE_ENHANCER_DISABLED
        ):
            if not ALLOW_DIRECTML_FACE_ENHANCER:
                if not DIRECTML_FACE_ENHANCER_FORCED_CPU:
                    selected_device = _force_cpu_face_enhancer(
                        (
                            "DirectML face enhancement is disabled by default "
                            "because torch-directml cannot execute GFPGAN "
                            "reliably. Using CPU instead. Set "
                            "DLC_ALLOW_DIRECTML_FACE_ENHANCER=1 to override."
                        )
                    )
                else:
                    selected_device = torch.device("cpu")
                device_priority.append("CPU (DirectML disabled)")
            else:
                selected_device = DIRECTML_DEVICE
                device_priority.append("DirectML")
        elif (
            'DmlExecutionProvider' in modules.globals.execution_providers
            and DIRECTML_FACE_ENHANCER_DISABLED
        ):
            update_status(
                (
                    "DirectML face enhancement previously failed; "
                    "using CPU fallback instead."
                ),
                NAME,
            )
            selected_device = torch.device("cpu")
            device_priority.append("CPU (fallback)")
        elif (
            'DmlExecutionProvider' in modules.globals.execution_providers
            and not TORCH_DIRECTML_AVAILABLE
        ):
            update_status(
                "torch-directml not found. Falling back to CPU for face enhancer.",
                NAME,
            )
            selected_device = torch.device("cpu")
            device_priority.append("CPU")
        elif TENSORRT_AVAILABLE and torch.cuda.is_available():
            selected_device = torch.device("cuda")
            device_priority.append("TensorRT+CUDA")
        elif torch.cuda.is_available():
            selected_device = torch.device("cuda")
            device_priority.append("CUDA")
        elif torch.backends.mps.is_available() and platform.system() == "Darwin":
            selected_device = torch.device("mps")
            device_priority.append("MPS")
        else:
            selected_device = torch.device("cpu")
            device_priority.append("CPU")

    if selected_device is None:
        selected_device = torch.device("cpu")
        if not device_priority:
            device_priority.append("CPU")

    return selected_device, device_priority


def _force_cpu_face_enhancer(
    message: Optional[str] = None, mark_disabled: bool = False
) -> torch.device:
    """Return a CPU device and mark DirectML as unusable when requested."""

    global DIRECTML_FACE_ENHANCER_DISABLED, DIRECTML_FACE_ENHANCER_FORCED_CPU

    if mark_disabled:
        DIRECTML_FACE_ENHANCER_DISABLED = True

    DIRECTML_FACE_ENHANCER_FORCED_CPU = True

    if message:
        update_status(
            (
                f"{message}"
                "\nContinuing with CPU fallback; this can be significantly slower, "
                "especially on high-resolution videos. The interface may look"
                " idle while the CPU works through each frame, but progress"
                " updates will resume once the first frame finishes."
            ),
            NAME,
        )

    return torch.device("cpu")


def _initialise_face_enhancer(force_device: Optional[torch.device] = None) -> FaceEnhancerBackend:
    global FACE_ENHANCER, DIRECTML_FACE_ENHANCER_DISABLED
    global DIRECTML_FACE_ENHANCER_FORCED_CPU

    backend_id = _get_configured_backend_id()
    backend_cls = AVAILABLE_FACE_ENHANCER_BACKENDS.get(
        backend_id, AVAILABLE_FACE_ENHANCER_BACKENDS[DEFAULT_FACE_ENHANCER_BACKEND]
    )

    if FACE_ENHANCER is None or not isinstance(FACE_ENHANCER, backend_cls):
        if FACE_ENHANCER is not None:
            FACE_ENHANCER.unload()
        FACE_ENHANCER = backend_cls(models_dir)

    device_priority: List[str] = []
    backend = FACE_ENHANCER

    if isinstance(backend, GfpganTorchBackend):
        selected_device, device_priority = _select_torch_device(force_device)

        try:
            backend.load(device=selected_device)
        except BackendLoadError as directml_error:
            if TORCH_DIRECTML_AVAILABLE and selected_device == DIRECTML_DEVICE:
                DIRECTML_FACE_ENHANCER_DISABLED = True
            if (
                force_device is None
                and TORCH_DIRECTML_AVAILABLE
                and selected_device == DIRECTML_DEVICE
            ):
                update_status(
                    (
                        "DirectML face enhancement failed, switching to CPU. "
                        f"Details: {_directml_error_summary(directml_error.original or directml_error)}"
                    ),
                    NAME,
                )
                print(
                    "DirectML initialisation for GFPGAN failed; "
                    f"falling back to CPU: {directml_error.original or directml_error}"
                )
                backend.unload()
                return _initialise_face_enhancer(torch.device("cpu"))
            raise

        device = backend.device
        device_label = _device_name(device)
        if str(device).upper() != device_label:
            print(
                "Selected device: "
                f"{device_label} ({device}) and device priority: {device_priority}"
            )
        else:
            print(
                f"Selected device: {device_label} and device priority: {device_priority}"
            )
    else:
        providers = modules.globals.execution_providers
        try:
            backend.load(execution_providers=providers)
        except BackendLoadError as error:
            provider_label = ", ".join(providers) if providers else "default providers"
            update_status(
                (
                    f"Failed to initialise {backend.display_name} using {provider_label}. "
                    f"Details: {_directml_error_summary(error.original or error)}"
                ),
                NAME,
            )
            raise

        provider = backend.provider or "CPUExecutionProvider"
        print(
            f"Selected backend: {backend.display_name} via {provider}"
        )

    return backend


def get_face_enhancer(force_device: Optional[torch.device] = None) -> FaceEnhancerBackend:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            FACE_ENHANCER = _initialise_face_enhancer(force_device)
        elif (
            force_device is not None
            and isinstance(FACE_ENHANCER, GfpganTorchBackend)
            and FACE_ENHANCER.device != force_device
        ):
            FACE_ENHANCER.unload()
            FACE_ENHANCER = _initialise_face_enhancer(force_device)

    # FACE_ENHANCER is guaranteed to be set at this point
    assert FACE_ENHANCER is not None
    return FACE_ENHANCER


def enhance_face(temp_frame: Frame) -> Frame:
    with THREAD_SEMAPHORE:
        enhancer = get_face_enhancer()

        while True:
            try:
                temp_frame = enhancer.enhance(temp_frame)
                break
            except BackendInferenceError as backend_error:
                original_error = backend_error.original or backend_error

                if (
                    TORCH_DIRECTML_AVAILABLE
                    and isinstance(enhancer, GfpganTorchBackend)
                    and enhancer.device == DIRECTML_DEVICE
                ):
                    error_message = str(original_error).lower()
                    directml_tensor_mismatch = "privateuseone" in error_message

                    cpu_device = _force_cpu_face_enhancer(
                        (
                            "DirectML face enhancement failed during inference, "
                            "switching to CPU. "
                            f"Details: {_directml_error_summary(original_error)}"
                        ),
                        mark_disabled=True,
                    )

                    if directml_tensor_mismatch:
                        print(
                            "DirectML inference for GFPGAN failed due to tensor type mismatch; "
                            f"falling back to CPU: {original_error}"
                        )
                    else:
                        print(
                            "DirectML inference for GFPGAN failed; "
                            f"falling back to CPU: {original_error}"
                        )

                    enhancer.unload()
                    enhancer = get_face_enhancer(cpu_device)
                    continue

                raise

    return temp_frame


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame)
    return temp_frame


def process_frames(
    source_path: str, temp_frame_paths: List[str], progress: Any = None
) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if progress:
            progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(None, temp_frame_paths, process_frames)


def process_frame_v2(temp_frame: Frame) -> Frame:
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame)
    return temp_frame
