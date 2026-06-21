#!/usr/bin/env python3
"""Evaluate the pretrained policy over deterministic simulator seeds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "best_model.keras")
    parser.add_argument("--room", type=int, default=3, help="Room id to evaluate.")
    parser.add_argument("--runs", type=int, default=10, help="Number of deterministic seeds.")
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for per-run evaluation data.",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not write evaluation files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from duck_collector.environment import DuckCollectorEnv
    from duck_collector.policy import KerasFeaturePolicy
    from duck_collector.simulation import run_simulation

    policy = KerasFeaturePolicy(args.model)
    env = DuckCollectorEnv(room_id=args.room, max_episode_time_step=1000, rendering=False)
    rows = []

    for run_number, seed in enumerate(range(args.runs), start=1):
        print(f"Run {run_number} / {args.runs}", flush=True)
        rewards = run_simulation(
            policy,
            env,
            record_data=False,
            verbose=args.verbose,
            max_steps=220,
            delay=0.0,
            seed=seed,
        )
        total_reward = float(rewards.sum())
        success_rate_run = total_reward / float(env.num_objs) if total_reward > 0 else 0.0
        avg_reward_run = float(rewards.mean()) if rewards.size else 0.0
        rows.append(
            {
                "room": args.room,
                "seed": seed,
                "steps": int(rewards.shape[0]),
                "objects_collected": int(total_reward),
                "success_rate": success_rate_run,
                "average_reward": avg_reward_run,
            }
        )
        print("Success rate:", success_rate_run if success_rate_run > 0 else False, flush=True)

    success_rate = sum(row["success_rate"] for row in rows) / float(args.runs)
    avg_reward = sum(row["average_reward"] for row in rows) / float(args.runs)
    avg_objects = sum(row["objects_collected"] for row in rows) / float(args.runs)

    print(f"\nroom={args.room} runs={args.runs}")
    print(f"success_rate={success_rate:.3f}")
    print(f"average_objects_collected={avg_objects:.2f}/{env.num_objs}")
    print(f"average_reward={avg_reward:.5f}")

    if not args.no_save:
        output = args.output
        if output is None:
            output = (
                REPO_ROOT
                / "outputs"
                / "runs"
                / f"evaluation_room{args.room}_{args.runs}runs.csv"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as fd:
            writer = csv.DictWriter(fd, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        summary_path = output.with_suffix(".summary.json")
        summary = {
            "room": args.room,
            "runs": args.runs,
            "success_rate": success_rate,
            "average_objects_collected": avg_objects,
            "num_objects": env.num_objs,
            "average_reward": avg_reward,
            "csv": str(output),
        }
        with summary_path.open("w") as fd:
            json.dump(summary, fd, indent=2)
            fd.write("\n")

        print(f"saved_run_data={output}")
        print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
