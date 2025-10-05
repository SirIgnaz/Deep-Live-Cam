import logging
import os
import shutil

from typing import Any, Optional
import insightface

import cv2
import modules.globals
from tqdm import tqdm

try:
    from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException as OrtRuntimeException
except ImportError:  # pragma: no cover - onnxruntime may not expose the C API package.
    try:
        import onnxruntime  # type: ignore

        class _OrtRuntimeFallback(RuntimeError):
            """Fallback when onnxruntime lacks a RuntimeException attribute."""

            pass

        OrtRuntimeException = getattr(
            onnxruntime,
            "RuntimeException",
            _OrtRuntimeFallback,
        )  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover - onnxruntime is entirely unavailable.
        class OrtRuntimeException(RuntimeError):
            """Fallback runtime exception when onnxruntime is unavailable."""

            pass


LOGGER = logging.getLogger(__name__)

_DIRECTML_PROVIDER = "dmlexecutionprovider"
from modules.typing import Frame
from modules.cluster_analysis import find_cluster_centroids, find_closest_centroid
from modules.utilities import get_temp_directory_path, create_temp, extract_frames, clean_temp, get_temp_frame_paths
from modules.mapping_utils import clone_frame_entries
from pathlib import Path

FACE_ANALYSER = None

MIN_FACE_DET_SCORE = 0.5


def _get_face_attribute(face: Any, attribute: str, default: Any = None) -> Any:
    if isinstance(face, dict):
        return face.get(attribute, default)
    return getattr(face, attribute, default)


def _get_face_det_score(face: Any) -> float:
    score = _get_face_attribute(face, 'det_score')
    return score if score is not None else 0.0


def _log_no_faces(context: str, path: str) -> None:
    LOGGER.warning("No faces detected %s: %s", context, path)


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(
            name='buffalo_l', providers=modules.globals.execution_providers
        )
        try:
            FACE_ANALYSER.prepare(
                ctx_id=0, det_size=modules.globals.face_detector_size
            )
        except Exception as error:
            default_size = (640, 640)
            if modules.globals.face_detector_size != default_size:
                print(
                    "\033[33mFailed to prepare face analyser with det_size="
                    f"{modules.globals.face_detector_size}. Falling back to {default_size}."
                    f" Error: {error}\033[0m"
                )
                modules.globals.face_detector_size = default_size
                FACE_ANALYSER.prepare(ctx_id=0, det_size=default_size)
            else:
                FACE_ANALYSER = None
                raise error
    return FACE_ANALYSER


def reset_face_analyser(det_size: Optional[tuple[int, int]] = None) -> tuple[int, int]:

    """Reset the cached face analyser and optionally set a new detector size."""

    global FACE_ANALYSER

    if det_size is not None:
        modules.globals.face_detector_size = det_size

    FACE_ANALYSER = None

    # Trigger a re-initialisation so det_size fallbacks are handled immediately.
    get_face_analyser()

    return modules.globals.face_detector_size


def _is_directml_preferred() -> bool:
    providers = modules.globals.execution_providers or []
    return bool(providers) and providers[0].lower() == _DIRECTML_PROVIDER


def _providers_without_directml() -> list[str]:
    return [
        provider
        for provider in modules.globals.execution_providers or []
        if provider.lower() != _DIRECTML_PROVIDER
    ]


def _retry_without_directml(original_error: Exception, frame: Frame) -> Any:
    if not _is_directml_preferred():
        raise original_error

    fallback_providers = _providers_without_directml()
    if not fallback_providers:
        raise original_error

    LOGGER.warning(
        "DirectML execution failed for face analysis; retrying with providers: %s",  # noqa: G004 - logging placeholder
        fallback_providers,
    )

    previous_providers = list(modules.globals.execution_providers or [])
    modules.globals.execution_providers = fallback_providers
    reset_face_analyser()

    try:
        return get_face_analyser().get(frame)
    except IndexError:
        return None
    except Exception as fallback_error:
        modules.globals.execution_providers = previous_providers
        try:
            reset_face_analyser()
        except Exception as restore_error:  # pragma: no cover - defensive clean-up only.
            LOGGER.debug(
                "Failed to restore face analyser after fallback: %s",
                restore_error,
            )

        if isinstance(fallback_error, OrtRuntimeException):
            raise original_error from fallback_error
        raise


def get_one_face(frame: Frame) -> Any:
    try:
        face = get_face_analyser().get(frame)
    except IndexError:
        return None
    except OrtRuntimeException as error:
        try:
            face = _retry_without_directml(error, frame)
        except OrtRuntimeException:
            raise
        if face is None:
            return None

    try:
        return min(face, key=lambda x: x.bbox[0])
    except (TypeError, ValueError):
        return None


def get_many_faces(frame: Frame) -> Any:
    try:
        return get_face_analyser().get(frame)
    except IndexError:
        return None
    except OrtRuntimeException as error:
        faces = _retry_without_directml(error, frame)
        if faces is None:
            return None
        return faces

def has_valid_map() -> bool:
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            return True
    return False

def default_source_face() -> Any:
    for map in modules.globals.source_target_map:
        if "source" in map:
            return map['source']['face']
    return None

def simplify_maps() -> Any:
    centroids = []
    faces = []
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            centroids.append(map['target']['face'].normed_embedding)
            faces.append(map['source']['face'])

    modules.globals.simple_map = {'source_faces': faces, 'target_embeddings': centroids}
    return None

def add_blank_map() -> Any:
    try:
        max_id = -1
        if len(modules.globals.source_target_map) > 0:
            max_id = max(modules.globals.source_target_map, key=lambda x: x['id'])['id']

        modules.globals.source_target_map.append({
                'id' : max_id + 1
                })
    except ValueError:
        return None
    
def get_unique_faces_from_target_image() -> Any:
    try:
        modules.globals.source_target_map = []
        target_frame = cv2.imread(modules.globals.target_path)
        if target_frame is None:
            LOGGER.error("Failed to read target image: %s", modules.globals.target_path)
            return None
        many_faces = get_many_faces(target_frame)
        if not many_faces:
            _log_no_faces("in target image", modules.globals.target_path)
            return None
        i = 0

        for face in many_faces:
            x_min, y_min, x_max, y_max = face['bbox']
            modules.globals.source_target_map.append({
                'id' : i, 
                'target' : {
                            'cv2' : target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                            'face' : face
                            }
                })
            i = i + 1
    except ValueError:
        return None
    
    
def get_unique_faces_from_target_video() -> Any:
    try:
        modules.globals.source_target_map = []
        frame_face_embeddings = []
        face_embeddings = []
    
        print('Creating temp resources...')
        clean_temp(modules.globals.target_path)
        create_temp(modules.globals.target_path)
        print('Extracting frames...')
        extract_frames(modules.globals.target_path)

        temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)

        if not temp_frame_paths:
            LOGGER.warning(
                "No frames were extracted from target video: %s", modules.globals.target_path
            )
            return None

        i = 0
        for temp_frame_path in tqdm(temp_frame_paths, desc="Extracting face embeddings from frames"):
            temp_frame = cv2.imread(temp_frame_path)
            if temp_frame is None:
                LOGGER.error("Failed to read extracted frame: %s", temp_frame_path)
                continue
            many_faces = get_many_faces(temp_frame)

            filtered_faces = []
            if many_faces:
                for face in many_faces:
                    if _get_face_det_score(face) >= MIN_FACE_DET_SCORE:
                        face_embeddings.append(face.normed_embedding)
                        filtered_faces.append(face)
            frame_face_embeddings.append({'frame': i, 'faces': filtered_faces, 'location': temp_frame_path})
            i += 1

        centroids = find_cluster_centroids(face_embeddings) if face_embeddings else []

        if not centroids:
            _log_no_faces("across target video frames", modules.globals.target_path)
            return None

        for frame in frame_face_embeddings:
            for face in frame['faces']:
                closest_centroid_index, _ = find_closest_centroid(centroids, face.normed_embedding)
                if isinstance(face, dict):
                    face['target_centroid'] = closest_centroid_index
                else:
                    setattr(face, 'target_centroid', closest_centroid_index)

        for i in range(len(centroids)):
            modules.globals.source_target_map.append({
                'id' : i
            })

            temp = []
            for frame in tqdm(frame_face_embeddings, desc=f"Mapping frame embeddings to centroids-{i}"):
                faces_for_centroid = [
                    face for face in frame['faces']
                    if _get_face_attribute(face, 'target_centroid') == i and _get_face_det_score(face) >= MIN_FACE_DET_SCORE
                ]
                temp.append({'frame': frame['frame'], 'faces': faces_for_centroid, 'location': frame['location']})

            filtered_temp = clone_frame_entries(temp)

            modules.globals.source_target_map[i]['_all_target_faces_in_frame'] = temp
            modules.globals.source_target_map[i]['target_faces_in_frame_filtered'] = filtered_temp
            modules.globals.source_target_map[i]['target_faces_in_frame'] = filtered_temp

        # dump_faces(centroids, frame_face_embeddings)
        default_target_face()
    except ValueError:
        return None
    

def default_target_face():
    for map in modules.globals.source_target_map:
        frames = map.get('target_faces_in_frame_filtered') or map.get('target_faces_in_frame') or []
        best_face = None
        best_frame = None
        for frame in frames:
            if len(frame['faces']) > 0:
                best_face = frame['faces'][0]
                best_frame = frame
                break

        if not best_face or not best_frame:
            continue

        for frame in frames:
            for face in frame['faces']:
                if face['det_score'] > best_face['det_score']:
                    best_face = face
                    best_frame = frame

        if best_face and best_frame:
            x_min, y_min, x_max, y_max = best_face['bbox']

            target_frame = cv2.imread(best_frame['location'])
            map['target'] = {
                            'cv2' : target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                            'face' : best_face
                            }


def dump_faces(centroids: Any, frame_face_embeddings: list):
    temp_directory_path = get_temp_directory_path(modules.globals.target_path)

    for i in range(len(centroids)):
        if os.path.exists(temp_directory_path + f"/{i}") and os.path.isdir(temp_directory_path + f"/{i}"):
            shutil.rmtree(temp_directory_path + f"/{i}")
        Path(temp_directory_path + f"/{i}").mkdir(parents=True, exist_ok=True)

        for frame in tqdm(frame_face_embeddings, desc=f"Copying faces to temp/./{i}"):
            temp_frame = cv2.imread(frame['location'])

            j = 0
            for face in frame['faces']:
                if face['target_centroid'] == i:
                    x_min, y_min, x_max, y_max = face['bbox']

                    if temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)].size > 0:
                        cv2.imwrite(temp_directory_path + f"/{i}/{frame['frame']}_{j}.png", temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)])
                j += 1