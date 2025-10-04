"""Utility helpers for handling mapping-related data structures."""

from typing import Any, Iterable, List, Optional


def clone_frame_entries(frames: Optional[Iterable[Any]]) -> List[Any]:
    """Return a shallow copy of frame entries with duplicated face lists."""

    if frames is None:
        return []

    cloned_entries: List[Any] = []

    for frame_entry in frames:
        if isinstance(frame_entry, dict):
            new_entry = dict(frame_entry)
            faces = frame_entry.get("faces")
            if isinstance(faces, list):
                new_entry["faces"] = list(faces)
            cloned_entries.append(new_entry)
        else:
            cloned_entries.append(frame_entry)

    return cloned_entries

