import matplotlib
import random
from typing import Optional, Tuple

import numpy as np

matplotlib.use("Agg")  # safe in headless sessions

from gymnasium import spaces, utils
from gymnasium.core import ObsType
from pyglet.window import key
import pyglet

from miniworld.entity import Agent, Box, MeshEnt
from miniworld.miniworld import MiniWorldEnv

ACTION_DISPLAY_NAMES = {
    0: "left",
    1: "right",
    2: "forward",
}


class DuckCollectorEnv(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Room with multiple objects. The agent collects +1 reward for picking up
    each object. Objects disappear when picked up.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing an RGB image of what the agents see.

    ## Rewards

    +1 when agent picked up an object

    ## Arguments

    * `size`: size of world
    * `num_objs`: number of objects
    * `room_id`: visual/time variant (0, 1, or 2)
    * `rendering`: enable pyglet windowed rendering helpers
    * `wait_for_keypress`: pause each rendered frame until a key is pressed
    * `min_obj_spacing`: minimum distance between any two objects
    * `wall_margin`: keep objects at least this far away from walls
    * `agent_margin`: keep objects at least this far away from the agent spawn
    """

    def __init__(
        self,
        size: int = 12,
        num_objs: int = 5,
        room_id: int = 0,
        rendering: bool = False,
        wait_for_keypress: bool = False,
        max_episode_time_step=250,
        **kwargs,
    ):
        assert size >= 2
        self.size = size
        self.num_objs = num_objs
        self.room_id = room_id
        self.rendering = rendering

        # Select time/appearance preset
        if room_id == 0:
            self.loc_sky_color = [0.4, 0.5, 1.0]
            self.loc_light_amb = [1.0, 1.0, 1.0]
            self.loc_light_color = [1.0, 1.0, 1.0]
            self.wall_tex = "wood_planks"
            self.floor_tex = "grass"
        elif room_id == 1:
            self.loc_sky_color = [0.7, 0.7, 1.0]
            self.loc_light_amb = [1.0, 1.0, 1.0]
            self.loc_light_color = [1.0, 1.0, 1.0]
            self.wall_tex = "marble"
            self.floor_tex = "water"
        elif room_id == 2:
            self.loc_sky_color = [0.7, 0.7, 0.2]
            self.loc_light_amb = [1.0, 1.0, 1.0]
            self.loc_light_color = [1.0, 1.0, 1.0]
            self.wall_tex = "brick_wall"
            self.floor_tex = "cardboard"
        else:
            self.loc_sky_color = [0.25, 0.0, 0.0]
            self.loc_light_amb = [1.0, 1.0, 1.0]
            self.loc_light_color = [1.0, 1.0, 1.0]
            self.wall_tex = "rock"
            self.floor_tex = "metal_grill"

        # --- Spacing/margin knobs (can be overridden via kwargs) ---
        self.min_obj_spacing = kwargs.get("min_obj_spacing", 4.25)  # min dist between objects
        self.wall_margin = kwargs.get("wall_margin", 1.0)           # keep away from walls
        self.agent_margin = kwargs.get("agent_margin", 2.0)         # keep away from agent spawn

        # Base env init
        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_time_step, **kwargs)
        utils.EzPickle.__init__(
            self, size, num_objs, room_id, rendering, wait_for_keypress, **kwargs
        )

        # Reduce the action space
        self.action_space = spaces.Discrete(self.actions.pickup + 1)

        # Human interface helpers (only used if rendering=True)
        self.stop_simulation = False
        self.last_policy_action = None
        self.key_pressed = {
            key.LEFT: False,
            key.RIGHT: False,
            key.UP: False,
            key.DOWN: False,
            key.P: False,
        }
        self.press_event = False
        self.wait_for_keypress = wait_for_keypress if self.rendering else False

    # ---------- Utility helpers for spaced sampling ----------

    def _xz(self, vec3):
        """Return (x, z) tuple from a 3D vector-like."""
        return float(vec3[0]), float(vec3[2])

    def _too_close(self, p, q, d):
        """Check if (x,z) positions p, q are closer than distance d."""
        dx = p[0] - q[0]
        dz = p[1] - q[1]
        return (dx * dx + dz * dz) < (d * d)

    def _sample_spread_positions(
        self,
        n,
        forbidden_points,
        min_x,
        max_x,
        min_z,
        max_z,
        min_dist,
        max_tries_per_point=200,
        relax_factor=0.9,
    ):
        """
        Sample n (x,z) positions inside the given bounds such that each new point
        is at least min_dist from all previously chosen points, and also at least
        agent_margin from any 'forbidden_points' (e.g., the agent spawn).
        If it's too hard, gradually relax the spacing by relax_factor.
        """
        chosen = []
        cur_min_dist = float(min_dist)

        for _ in range(n):
            success = False
            tries = 0

            while not success:
                # If sampling is too hard, relax the min distance slightly
                if tries >= max_tries_per_point:
                    cur_min_dist *= relax_factor
                    tries = 0

                x = self.np_random.uniform(min_x, max_x)
                z = self.np_random.uniform(min_z, max_z)
                p = (x, z)

                ok = True
                # Against already chosen objects
                for q in chosen:
                    if self._too_close(p, q, cur_min_dist):
                        ok = False
                        break

                # Against forbidden points (e.g., agent)
                if ok:
                    for q in forbidden_points:
                        if self._too_close(p, q, max(cur_min_dist, self.agent_margin)):
                            ok = False
                            break

                if ok:
                    chosen.append(p)
                    success = True
                else:
                    tries += 1

        return chosen

    # ---------- Lifecycle ----------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, dict]:
        """
        Reset the simulation at the start of a new episode.
        """
        super().reset(seed=seed)

        # Step count since episode start
        self.step_count = 0

        # Create the agent and clear world containers
        self.agent = Agent()
        # self.agent.cam_height = 0
        self.entities = []
        self.rooms = []
        self.wall_segs = []

        # Generate the world
        self._gen_world()

        # Domain randomization (then override with room_id-specific values)
        rand = self.np_random if self.domain_rand else None
        self.params.sample_many(
            rand, self, ["sky_color", "light_pos", "light_color", "light_ambient"]
        )

        # Apply room_id-controlled visuals
        self.sky_color = self.loc_sky_color
        self.light_ambient = self.loc_light_amb
        self.light_color = self.loc_light_color

        # Randomize entity parameters
        for ent in self.entities:
            ent.randomize(self.params, rand)

        # Compute world extents
        self.min_x = min(r.min_x for r in self.rooms)
        self.max_x = max(r.max_x for r in self.rooms)
        self.min_z = min(r.min_z for r in self.rooms)
        self.max_z = max(r.max_z for r in self.rooms)

        # Static buffers/geometry
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        self._render_static()

        # First observation
        obs = self.render_obs()
        return obs, {}

    def reinit(self, max_steps: int, wait_for_keypress: bool):
        """
        Convenience for interactive runs: change episode length and keypress behavior,
        re-attach pyglet handlers if needed, and reset.
        """
        self.wait_for_keypress = wait_for_keypress if self.rendering else False
        if self.rendering:
            self.render_mode = "pyglet"
            super().render()
            self.on_key_press = self.unwrapped.window.event(self.on_key_press)
            self.on_key_release = self.unwrapped.window.event(self.on_key_release)
            self.stop_simulation = False
            self.press_event = False
            for k in list(self.key_pressed.keys()):
                self.key_pressed[k] = False

        self.max_episode_steps = max_steps
        return self.reset()

    # ---------- World construction ----------

    def _gen_world(self):
        # Single room whose size is controlled by self.size
        self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex=self.wall_tex,
            floor_tex=self.floor_tex,
            no_ceiling=True,
        )

        # Place the agent first so we can keep objects away from it
        self.place_agent()  # MiniWorld picks a valid spawn
        agent_xz = [self._xz(self.agent.pos)]

        # Object types/colors
        obj_types = ['duckie']
        colorlist = ['yellow']

        # Sampling bounds with wall margins
        xmin = 0 + self.wall_margin
        xmax = self.size - self.wall_margin
        zmin = 0 + self.wall_margin
        zmax = self.size - self.wall_margin

        # Plan all object positions with spacing
        obj_positions_xz = self._sample_spread_positions(
            n=self.num_objs,
            forbidden_points=agent_xz,
            min_x=xmin, max_x=xmax,
            min_z=zmin, max_z=zmax,
            min_dist=self.min_obj_spacing,
        )

        # Create and place objects at the sampled positions
        for (x, z) in obj_positions_xz:
            obj_type = obj_types[self.np_random.choice(len(obj_types))]
            color = colorlist[self.np_random.choice(len(colorlist))]

            if obj_type == 'duckie':
                ent = MeshEnt(mesh_name="duckie", height=random.uniform(0.75, 1.25))
                ent.can_pickup = True
                ent.static = False
                ent.radius = 0.2
                y = 0
                self.place_entity(ent, pos=np.array([float(x), float(y), float(z)]))
            else:
                ent = Box(color=color, size=random.uniform(0.75, 1.25))
                y = 0
                self.place_entity(ent, pos=np.array([float(x), float(y), float(z)]))

        self.num_picked_up = 0

    # ---------- Step / reward ----------

    def step(self, action):
        self.agent.cam_height = 0.75 + random.uniform(-0.025, 0.025)
        obs, reward, termination, truncation, info = super().step(action)

        # Auto-attempt pickup each step (as in your original logic)
        obs, reward, termination, truncation, info = super().step(4)

        if self.agent.carrying:
            # Remove the picked up object
            self.entities.remove(self.agent.carrying)
            self.agent.carrying = None
            self.num_picked_up += 1
            reward = 1

            if self.num_picked_up == self.num_objs:
                termination = True

        return obs, reward, termination, truncation, info

    # ---------- Rendering helpers (interactive) ----------

    def render(self, mode: str = "rgb_array"):
        if self.rendering:
            self.render_mode = "pyglet"
            out = super().render()
            self._draw_policy_action_label()
            self.update_pyglet()
        else:
            self.render_mode = mode
            out = super().render()
        return out

    def _draw_policy_action_label(self):
        """Draw the latest policy action in the live text panel."""

        if self.last_policy_action is None:
            return

        label = ACTION_DISPLAY_NAMES.get(
            int(self.last_policy_action),
            str(int(self.last_policy_action)),
        )
        if not hasattr(self, "policy_action_label"):
            self.policy_action_label = pyglet.text.Label(
                font_name="Arial",
                font_size=20,
                bold=True,
                multiline=False,
                width=400,
                x=self.text_label.x,
                y=self.text_label.y - 90,
            )

        self.policy_action_label.text = f"action: {label}"
        self.policy_action_label.draw()

    def on_key_press(self, symbol, modifier):
        if symbol == key.ESCAPE:
            self.stop_simulation = True
        self.key_pressed[symbol] = True
        self.press_event = True

    def on_key_release(self, symbol, modifier):
        self.key_pressed[symbol] = False
        self.press_event = False

    def update_pyglet(self):
        self.window.flip()
        while True:
            pyglet.clock.tick()
            self.window.dispatch_events()
            if (not self.wait_for_keypress) or self.press_event:
                break

    def close(self):
        super().close()
