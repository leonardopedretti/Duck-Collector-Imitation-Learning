"""Video export helpers for recorded observations."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ACTION_LABELS = {
    0: "left",
    1: "right",
    2: "forward",
}


def save_observations_video(
    observations: np.ndarray,
    output_path: str | Path,
    actions: np.ndarray | None = None,
    fps: float = 20.0,
) -> Path:
    """Write RGB observation frames to an MP4 file."""

    frames = np.asarray(observations)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape (N, H, W, 3), got {frames.shape}.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}.")

    try:
        for i, frame in enumerate(frames):
            bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            if actions is not None and i < len(actions):
                label = ACTION_LABELS.get(int(actions[i]), str(int(actions[i])))
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.28
                thickness = 1
                text_size, baseline = cv2.getTextSize(label, font, scale, thickness)
                text_w, text_h = text_size
                x = 3
                y = max(text_h + 4, height - 5)
                cv2.rectangle(
                    bgr,
                    (x - 2, y - text_h - 3),
                    (x + text_w + 2, y + baseline),
                    (0, 0, 0),
                    thickness=-1,
                )
                cv2.putText(
                    bgr,
                    label,
                    (x, y),
                    font,
                    scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
            writer.write(bgr)
    finally:
        writer.release()

    return output
