#!/usr/bin/env python3
"""Collect interactive human demonstrations with the arrow keys."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("room_id", type=int, choices=[0, 1, 2], help="Training room to record.")
    parser.add_argument(
        "--path",
        type=Path,
        default=REPO_ROOT / "data" / "demonstrations",
        help="Directory where the demonstration pickle will be saved.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Maximum number of simulator steps.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Delay between control steps in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from duck_collector.data import save_demonstration
    from duck_collector.environment import DuckCollectorEnv
    from duck_collector.simulation import policy_human, run_simulation

    env = DuckCollectorEnv(
        room_id=args.room_id,
        max_episode_time_step=args.iterations,
        rendering=True,
    )

    print("Starting new demonstration data collection.")
    rewards, observations, actions, truncation = run_simulation(
        policy_human,
        env=env,
        record_data=True,
        max_steps=args.iterations,
        human_control=True,
        delay=args.delay,
    )

    if truncation:
        print("Recording was truncated; demonstration was not saved.")
        return

    output = save_demonstration(
        observations=observations,
        actions=actions,
        room_id=args.room_id,
        directory=args.path,
    )
    print(f"Saved {observations.shape[0]} frames, total reward {rewards.sum()}, to {output}.")


if __name__ == "__main__":
    main()
