from typing import Any, List, Optional
import cv2
import threading
import gfpgan
import os

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
import platform
import torch

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

FACE_ENHANCER = None
FACE_ENHANCER_DEVICE: Optional[torch.device] = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-ENHANCER"

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)


def pre_check() -> bool:
    download_directory_path = models_dir
    conditional_download(
        download_directory_path,
        [
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
        ],
    )
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


def _initialise_face_enhancer(force_device: Optional[torch.device] = None) -> Any:
    global FACE_ENHANCER, FACE_ENHANCER_DEVICE

    model_path = os.path.join(models_dir, "GFPGANv1.4.pth")

    selected_device: Optional[torch.device] = None
    device_priority: List[str] = []

    if force_device is not None:
        selected_device = force_device
        device_priority.append(_device_name(selected_device))
    else:
        if (
            'DmlExecutionProvider' in modules.globals.execution_providers
            and TORCH_DIRECTML_AVAILABLE
        ):
            selected_device = DIRECTML_DEVICE
            device_priority.append("DirectML")
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

    try:
        FACE_ENHANCER = gfpgan.GFPGANer(
            model_path=model_path, upscale=1, device=selected_device
        )
        FACE_ENHANCER_DEVICE = selected_device
    except Exception as directml_error:
        if (
            force_device is None
            and TORCH_DIRECTML_AVAILABLE
            and selected_device == DIRECTML_DEVICE
        ):
            update_status(
                (
                    "DirectML face enhancement failed, switching to CPU. "
                    "See logs for details."
                ),
                NAME,
            )
            print(
                "DirectML initialisation for GFPGAN failed; "
                f"falling back to CPU: {directml_error}"
            )
            return _initialise_face_enhancer(torch.device("cpu"))
        raise

    print(
        f"Selected device: {selected_device} and device priority: {device_priority}"
    )
    return FACE_ENHANCER


def get_face_enhancer(force_device: Optional[torch.device] = None) -> Any:
    global FACE_ENHANCER, FACE_ENHANCER_DEVICE

    with THREAD_LOCK:
        if (
            FACE_ENHANCER is None
            or (force_device is not None and FACE_ENHANCER_DEVICE != force_device)
        ):
            FACE_ENHANCER = None
            FACE_ENHANCER_DEVICE = None
            return _initialise_face_enhancer(force_device)

    return FACE_ENHANCER


def enhance_face(temp_frame: Frame) -> Frame:
    global FACE_ENHANCER

    with THREAD_SEMAPHORE:
        enhancer = get_face_enhancer()
        try:
            _, _, temp_frame = enhancer.enhance(temp_frame, paste_back=True)
        except RuntimeError as runtime_error:
            error_message = str(runtime_error).lower()
            directml_tensor_mismatch = (
                TORCH_DIRECTML_AVAILABLE
                and FACE_ENHANCER_DEVICE == DIRECTML_DEVICE
                and "privateuseone" in error_message
            )

            if directml_tensor_mismatch:
                update_status(
                    (
                        "DirectML face enhancement failed during inference, "
                        "switching to CPU. See logs for details."
                    ),
                    NAME,
                )
                print(
                    "DirectML inference for GFPGAN failed due to tensor type mismatch; "
                    f"falling back to CPU: {runtime_error}"
                )
                FACE_ENHANCER = None
                enhancer = get_face_enhancer(torch.device("cpu"))
                _, _, temp_frame = enhancer.enhance(temp_frame, paste_back=True)
            else:
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
