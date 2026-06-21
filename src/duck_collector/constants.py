"""Shared action labels used by the simulator and policies."""

ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_FORWARD = 2

ACTIONS = [ACTION_LEFT, ACTION_RIGHT, ACTION_FORWARD]
ACTION_NAMES = {
    ACTION_LEFT: "turn_left",
    ACTION_RIGHT: "turn_right",
    ACTION_FORWARD: "move_forward",
}
NUM_ACTIONS = len(ACTIONS)
