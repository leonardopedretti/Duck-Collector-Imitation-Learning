#!/usr/bin/env python3
"""Run the pretrained policy once and optionally export a video/demo pickle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "best_model.keras")
    parser.add_argument("--room", type=int, default=3, help="Room id to run.")
    parser.add_argument("--seed", type=int, default=0, help="Simulator seed. Use -1 for random.")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--render", action="store_true", help="Show the MiniWorld window.")
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="Optional output pickle path for the recorded run.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=REPO_ROOT / "outputs" / "videos" / "room3_policy.mp4",
        help="MP4 output path.",
    )
    parser.add_argument("--no-video", action="store_true", help="Do not export an MP4.")
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from duck_collector.data import save_demonstration
    from duck_collector.environment import DuckCollectorEnv
    from duck_collector.policy import KerasFeaturePolicy
    from duck_collector.simulation import run_simulation
    from duck_collector.video import save_observations_video

    video_path = None if args.no_video else args.video
    record_data = bool(video_path or args.record)

    policy = KerasFeaturePolicy(args.model)
    env = DuckCollectorEnv(
        room_id=args.room,
        max_episode_time_step=args.max_steps,
        rendering=args.render,
    )

    result = run_simulation(
        policy,
        env=env,
        max_steps=args.max_steps,
        verbose=1,
        record_data=record_data,
        delay=args.delay,
        seed=args.seed,
        human_control=False,
    )

    if not record_data:
        return

    rewards, observations, actions, truncation = result
    print(f"Recorded {observations.shape[0]} frames; total reward={rewards.sum()}.")
    if truncation:
        print("Run ended by truncation.")

    if args.record:
        output = save_demonstration(
            observations=observations,
            actions=actions,
            room_id=args.room,
            directory=args.record.parent,
            prefix=args.record.stem,
        )
        if output != args.record:
            output.replace(args.record)
            output = args.record
        print(f"Saved recorded run to {output}.")

    if video_path:
        output_video = save_observations_video(
            observations=observations,
            actions=actions,
            output_path=video_path,
            fps=args.fps,
        )
        print(f"Saved video to {output_video}.")


if __name__ == "__main__":
    main()
