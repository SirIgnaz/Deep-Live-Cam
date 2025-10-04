import sys
import types

import pytest

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

if "insightface" not in sys.modules:
    fake_insightface = types.ModuleType("insightface")

    class _DummyFaceAnalysis:
        def __init__(self, *args, **kwargs):
            pass

        def prepare(self, *args, **kwargs):
            pass

    fake_app_module = types.ModuleType("insightface.app")
    fake_app_module.FaceAnalysis = _DummyFaceAnalysis

    fake_common_module = types.ModuleType("insightface.app.common")

    class _DummyFace:  # pragma: no cover - used for type compatibility only.
        pass

    fake_common_module.Face = _DummyFace

    fake_insightface.app = fake_app_module

    sys.modules["insightface"] = fake_insightface
    sys.modules["insightface.app"] = fake_app_module
    sys.modules["insightface.app.common"] = fake_common_module

if "cv2" not in sys.modules:
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.IMREAD_COLOR = 1

    def _not_implemented(*args, **kwargs):  # pragma: no cover - the stub should never be executed.
        raise NotImplementedError

    fake_cv2.imread = _not_implemented
    fake_cv2.imdecode = _not_implemented
    fake_cv2.imencode = lambda *args, **kwargs: (False, None)
    sys.modules["cv2"] = fake_cv2

from modules import globals as global_state  # noqa: E402
from modules import face_analyser  # noqa: E402


def test_get_many_faces_recovers_from_directml_failure(monkeypatch):
    monkeypatch.setattr(
        global_state,
        "execution_providers",
        ["DmlExecutionProvider", "CPUExecutionProvider"],
        raising=False,
    )

    attempts = {"count": 0}

    class _FailingAnalyser:
        def get(self, frame):
            raise face_analyser.OrtRuntimeException("DML failure")

    class _SuccessfulAnalyser:
        def get(self, frame):
            return ["face"]

    def fake_get_face_analyser():
        attempts["count"] += 1
        if attempts["count"] == 1:
            return _FailingAnalyser()
        return _SuccessfulAnalyser()

    reset_calls = {"count": 0}

    def fake_reset_face_analyser():
        reset_calls["count"] += 1
        return global_state.face_detector_size

    monkeypatch.setattr(face_analyser, "get_face_analyser", fake_get_face_analyser)
    monkeypatch.setattr(face_analyser, "reset_face_analyser", fake_reset_face_analyser)

    faces = face_analyser.get_many_faces(frame="frame")

    assert faces == ["face"]
    assert attempts["count"] == 2
    assert reset_calls["count"] == 1
    assert global_state.execution_providers == ["CPUExecutionProvider"]


def test_directml_failure_without_fallback_reraises(monkeypatch):
    monkeypatch.setattr(
        global_state,
        "execution_providers",
        ["DmlExecutionProvider"],
        raising=False,
    )

    class _FailingAnalyser:
        def get(self, frame):
            raise face_analyser.OrtRuntimeException("DML failure")

    def fake_get_face_analyser():
        return _FailingAnalyser()

    def fail_reset(*args, **kwargs):  # pragma: no cover - ensure it is not called.
        raise AssertionError("reset_face_analyser should not be called without fallback providers")

    monkeypatch.setattr(face_analyser, "get_face_analyser", fake_get_face_analyser)
    monkeypatch.setattr(face_analyser, "reset_face_analyser", fail_reset)

    with pytest.raises(face_analyser.OrtRuntimeException):
        face_analyser.get_many_faces(frame="frame")

    assert global_state.execution_providers == ["DmlExecutionProvider"]


def test_failed_fallback_restores_original_providers(monkeypatch):
    monkeypatch.setattr(
        global_state,
        "execution_providers",
        ["DmlExecutionProvider", "CPUExecutionProvider"],
        raising=False,
    )

    reset_history = []

    def fake_reset_face_analyser(det_size=None):
        reset_history.append(list(global_state.execution_providers))
        return global_state.face_detector_size

    class _DynamicAnalyser:
        def get(self, frame):
            if face_analyser._is_directml_preferred():
                raise face_analyser.OrtRuntimeException("DML failure")
            raise face_analyser.OrtRuntimeException("CPU failure")

    monkeypatch.setattr(face_analyser, "get_face_analyser", lambda: _DynamicAnalyser())
    monkeypatch.setattr(face_analyser, "reset_face_analyser", fake_reset_face_analyser)

    with pytest.raises(face_analyser.OrtRuntimeException) as exc_info:
        face_analyser.get_many_faces(frame="frame")

    assert str(exc_info.value) == "DML failure"
    assert global_state.execution_providers == ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert reset_history == [
        ["CPUExecutionProvider"],
        ["DmlExecutionProvider", "CPUExecutionProvider"],
    ]
