"""Loading and saving demonstration data."""

from __future__ import annotations

import datetime as _dt
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np


def load_demonstration(path: str | Path) -> dict:
    """Load one pickle demonstration."""

    with Path(path).open("rb") as fd:
        return pickle.load(fd)


def load_demonstrations(
    directory: str | Path = "data/demonstrations",
    pattern: str = "*.pickle",
) -> list[dict]:
    """Load demonstrations matching a glob pattern."""

    root = Path(directory)
    filenames = sorted(root.glob(pattern))
    if not filenames:
        raise FileNotFoundError(f"No demonstrations matched {root / pattern}.")
    return [load_demonstration(path) for path in filenames]


def combine_demonstrations(demonstrations: Iterable[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate observations, actions, and room ids from demonstrations."""

    demos = list(demonstrations)
    if not demos:
        raise ValueError("Cannot combine an empty demonstration list.")

    observations = np.concatenate([demo["observations"] for demo in demos])
    actions = np.concatenate([demo["actions"] for demo in demos]).astype(int)
    room_ids = np.concatenate([demo["time"] for demo in demos]).astype(int)
    return observations, actions, room_ids


def save_demonstration(
    observations: np.ndarray,
    actions: np.ndarray,
    room_id: int,
    directory: str | Path = "data/demonstrations",
    prefix: str = "demonstration",
) -> Path:
    """Save recorded observations and actions in the project data format."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = root / f"{prefix}-{room_id}-{timestamp}.pickle"
    payload = {
        "observations": observations,
        "actions": actions.astype(int),
        "time": room_id * np.ones(observations.shape[0], dtype=int),
    }
    with filename.open("wb") as fd:
        pickle.dump(payload, fd)
    return filename


def demonstration_summary(directory: str | Path = "data/demonstrations") -> dict:
    """Return lightweight counts for local demonstration files."""

    root = Path(directory)
    summary = {
        "files": 0,
        "samples": 0,
        "rooms": {},
        "actions": {},
    }
    for path in sorted(root.glob("*.pickle")):
        demo = load_demonstration(path)
        actions = np.asarray(demo["actions"]).astype(int)
        room_ids = np.asarray(demo["time"]).astype(int)
        summary["files"] += 1
        summary["samples"] += int(actions.shape[0])
        for room_id in np.unique(room_ids):
            summary["rooms"][int(room_id)] = summary["rooms"].get(int(room_id), 0) + int(
                np.sum(room_ids == room_id)
            )
        for action in np.unique(actions):
            summary["actions"][int(action)] = summary["actions"].get(int(action), 0) + int(
                np.sum(actions == action)
            )
    return summary
