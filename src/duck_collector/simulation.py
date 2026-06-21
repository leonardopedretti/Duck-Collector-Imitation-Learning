"""Simulation helpers shared by training, evaluation, and data collection."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from pyglet.window import key

from .constants import ACTION_NAMES, ACTIONS

Policy = Callable[[np.ndarray], int]

KEY_PRESSED = {key.LEFT: False, key.RIGHT: False, key.UP: False}


def run_simulation(
    policy: Policy,
    env,
    max_steps: int = 500,
    verbose: int = 1,
    record_data: bool = False,
    delay: float = 0.0,
    seed: int = -1,
    human_control: bool = False,
):
    """Run a policy in a DuckCollectorEnv environment."""

    if verbose > 0:
        print(f"Starting simulation for {max_steps} steps.")
        if getattr(env, "rendering", False):
            print("*** Press ESC in the popup window to stop the simulation. ***")
        print()

    if seed >= 0:
        env.reset(seed=seed)
    else:
        env.reset(seed=int(time.time() * 1000.0))

    rewards = []
    observations = []
    actions = []
    truncation = False
    completed_steps = 0

    try:
        # DuckCollectorEnv.step() advances the MiniWorld simulator twice: once
        # for the policy action and once for the automatic pickup action. The
        # runner therefore doubles the internal episode cap so a 220-step
        # policy evaluation is not truncated after roughly 110 policy actions.
        obs = env.reinit(2 * max_steps, wait_for_keypress=human_control)[0]

        for step in range(max_steps):
            time.sleep(delay)

            if human_control:
                if getattr(env, "rendering", False):
                    env.render()

                KEY_PRESSED.update(env.key_pressed)
                if env.stop_simulation:
                    break

                action = policy(obs)
            else:
                action = policy(obs)
                if getattr(env, "rendering", False):
                    env.last_policy_action = int(action)
                    env.render()

                KEY_PRESSED.update(env.key_pressed)
                if env.stop_simulation:
                    break

            assert isinstance(action, (int, np.integer))
            assert int(action) in ACTIONS
            action = int(action)
            if human_control and getattr(env, "rendering", False):
                env.last_policy_action = action

            if verbose > 1:
                print(f"step {step}: action = {ACTION_NAMES[action]}")

            if record_data:
                observations.append(obs)
                actions.append(action)

            obs, reward, environment_done, truncation, _info = env.step(action)
            completed_steps += 1

            if verbose > 1:
                print(f"step {step}: reward = {reward}")

            rewards.append(reward)
            if environment_done or truncation:
                break

    finally:
        if getattr(env, "rendering", False):
            try:
                if getattr(env, "window", None):
                    env.window.remove_handlers()
            except Exception:
                pass
            try:
                if getattr(env, "window", None):
                    env.window.close()
            finally:
                env.window = None

    rewards = np.array(rewards)
    total_reward = float(np.sum(rewards))

    if verbose > 0:
        print(f"total reward after {completed_steps} steps: {total_reward}")
        if completed_steps:
            print(f"average reward: {total_reward / completed_steps}")

    if record_data:
        return rewards, np.array(observations), np.array(actions, dtype=int), truncation
    return rewards


def test_policy(policy: Policy, env, verbose: int = 0, num_runs: int = 10):
    """Evaluate a policy over deterministic seeds."""

    avg_success_rate = 0.0
    avg_reward = 0.0

    for run_count, seed in enumerate(range(num_runs), start=1):
        print(f"Run {run_count} / {num_runs}")
        success = False
        rewards = run_simulation(
            policy,
            env,
            record_data=False,
            verbose=verbose,
            max_steps=220,
            delay=0.0,
            seed=seed,
        )

        avg_reward += float(np.mean(rewards)) if rewards.size else 0.0
        if np.sum(rewards) > 0:
            success = float(np.sum(rewards) / env.num_objs)
            avg_success_rate += success

        print("Success rate:", success)

    avg_success_rate /= float(num_runs)
    avg_reward /= float(num_runs)

    print()
    print(f"average success rate: {avg_success_rate}")
    print(f"average reward: {avg_reward}")
    return avg_success_rate, avg_reward


def policy_human(_observation: np.ndarray) -> int | None:
    """Keyboard policy used for collecting demonstrations."""

    if KEY_PRESSED[key.LEFT]:
        return ACTIONS[0]
    if KEY_PRESSED[key.RIGHT]:
        return ACTIONS[1]
    if KEY_PRESSED[key.UP]:
        return ACTIONS[2]
    return None
