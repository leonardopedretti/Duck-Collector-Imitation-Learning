#!/usr/bin/env python3
"""Retrain the final Keras model from demonstration pickles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "demonstrations",
        help="Directory containing demonstration pickle files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.pickle",
        help="Glob pattern for training demonstrations.",
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "models" / "best_model.keras")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import tensorflow as tf

    from duck_collector.data import combine_demonstrations, load_demonstrations
    from duck_collector.training import TrainingConfig, train_final_model

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    demos = load_demonstrations(args.data, args.pattern)
    observations, actions, room_ids = combine_demonstrations(demos)
    print(
        f"Loaded {len(demos)} demos: observations={observations.shape}, "
        f"actions={actions.shape}, rooms={sorted(set(room_ids.tolist()))}."
    )

    config = TrainingConfig(random_state=args.seed, epochs=args.epochs)
    _model, metrics = train_final_model(
        observations=observations,
        actions=actions,
        room_ids=room_ids,
        output_path=args.output,
        config=config,
    )

    print(f"Saved model to {args.output}.")
    print(f"Validation macro-F1: {metrics['macro_f1']:.3f}")
    print("Confusion matrix:")
    for row in metrics["confusion_matrix"]:
        print(row)
    print("\nClassification report:")
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
