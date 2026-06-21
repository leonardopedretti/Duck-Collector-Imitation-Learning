#!/usr/bin/env python3
"""Export a recorded demonstration pickle to MP4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", type=Path, help="Path to a demonstration pickle.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "videos" / "demo.mp4",
        help="MP4 output path.",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from duck_collector.data import load_demonstration
    from duck_collector.video import save_observations_video

    demo = load_demonstration(args.demo)
    output = save_observations_video(
        observations=demo["observations"],
        actions=demo.get("actions"),
        output_path=args.output,
        fps=args.fps,
    )
    print(f"Saved video to {output}.")


if __name__ == "__main__":
    main()
