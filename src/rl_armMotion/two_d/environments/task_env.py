"""Goal-reaching environment for a planar human arm (Gymnasium API).

The arm is a rigid-body chain with gravity, coupled inertia, human range of
motion, torque limits and soft end-range stiffness (see utils/dynamics.py and
config/arm_config.py for the model and its conventions). It is driven either
by antagonist Hill-type muscle pairs (utils/muscle_model.py) or directly by
joint torques. The task is to visit an ordered list of waypoints and to hold
still at the last one.

Coordinates
-----------
Workspace frame: +x anterior, +y up, shoulder fixed at
``shoulder_base_position`` (default [1.0, 0.0]). For a left arm the forward
kinematics mirror x about the shoulder, so left-arm waypoints are simply given
in the left-arm workspace.

Task (waypoint mode)
--------------------
- Intermediate waypoints are "settle-and-go": the hand must be inside the
  position tolerance and slow for ``intermediate_hold_steps`` consecutive
  steps, then the next waypoint becomes active.
- The final waypoint requires ``hold_steps_required`` consecutive in-goal
  steps. Reaching that count marks the episode a SUCCESS (``info["success"]``)
  and pays a one-time bonus. By default the episode does NOT terminate there:
  the agent keeps earning the in-goal reward for standing still until the time
  limit, which is what makes the learned hold stable. Evaluation and demos may
  pass ``terminate_on_success=True``.
- ``in_goal`` = position error < tolerance AND hand speed < ``ee_speed_tolerance``
  AND joint speed norm < ``joint_speed_tolerance``. Hand orientation is NOT
  part of the criterion for a 2-link arm: it is a function of hand position
  once the elbow branch is fixed by the range of motion. ``require_orientation``
  re-enables it for redundant (3-link) arms.

Observation (OBS_VERSION 2, dimension 3n + 8 for n joints, all bounded)
-----------------------------------------------------------------------
    sin q_j, cos q_j (2n)          [-1, 1]
    qd_j / velocity_limit (n)      [-1, 1]
    (goal_k - hand) / reach (2)    [-2, 2]
    |goal_k - hand| / reach (1)    [0, 2]
    hand velocity / v_ref (2)      [-1, 1]  (v_ref = 2 m/s, clipped)
    k / (N - 1) (1)                [0, 1]   waypoint progress (0 when N = 1)
    in_goal flag (1)               {0, 1}
    hold progress (1)              [0, 1]
plus, when enabled, muscle activations (2n) and the last ``sensory_delay_steps``
actions (action history keeps the delayed observation Markov).

Reward (all distances normalised by reach R)
--------------------------------------------
    Phi(s) = -|goal_k(s) - hand(s)| / R                 potential on (hand, k)
    r = W_SHAPING * (gamma * Phi(s') - Phi(s))           two-sided shaping
      - W_DIST   * |goal_k' - hand'| / R                 dense far-field gradient
      - W_EFFORT * |a|^2                                 effort (no co-contraction penalty)
      - W_SMOOTH * |qd'| / velocity_limit                mild smoothness
      - W_LIMIT  * (joints on a hard stop + end-range spring engagement 0..1)   end-range avoidance
      - W_SPEED_EXCESS * max(0, |hand velocity| - 1.5)^2  human-like speed cap
      - W_ACCEL * min((|hand accel| / 10)^2, 4)           smoothness (min-jerk proxy; 0 unless enabled)
      + R_IN_GOAL      if in_goal at the FINAL waypoint  hold stream
      + R_INTERMEDIATE on completing an intermediate waypoint
      + R_SUCCESS      once, when the final hold completes
Shaping is potential-based on the augmented state (hand position, waypoint
index) and is therefore policy-invariant (Ng, Harada & Russell 1999).
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_armMotion.two_d.config import ArmConfiguration
from rl_armMotion.two_d.utils import dynamics
from rl_armMotion.two_d.utils.arm_kinematics import ArmKinematics
from rl_armMotion.two_d.utils.muscle_model import JointMusclePair


OBS_VERSION = 2
DEFAULT_PRESET = "2dof_human_right"

__all__ = ["ArmTaskEnv", "ActionSmoother", "OBS_VERSION", "DEFAULT_PRESET"]


class ActionSmoother(gym.ActionWrapper):
    """EMA low-pass filter on actions for smoother muscle commands.

    Use at evaluation/visualisation time; it does not affect training.
    """

    def __init__(self, env: gym.Env, alpha: float = 0.3):
        super().__init__(env)
        self.alpha = alpha
        self._prev_action: Optional[np.ndarray] = None

    def action(self, action: np.ndarray) -> np.ndarray:
        if self._prev_action is None:
            self._prev_action = np.array(action, dtype=np.float32)
        else:
            self._prev_action = (
                self.alpha * np.array(action, dtype=np.float32)
                + (1.0 - self.alpha) * self._prev_action
            )
        return self._prev_action

    def reset(self, **kwargs):
        self._prev_action = None
        return self.env.reset(**kwargs)

    # Explicit pass-throughs (Gymnasium 1.x wrappers do not forward attributes).
    def get_state_info(self):
        return self.env.get_state_info()

    def set_waypoints(self, *args, **kwargs):
        return self.env.set_waypoints(*args, **kwargs)

    def describe(self):
        return self.env.describe()

    @property
    def config(self):
        return self.env.config

    @property
    def dt(self):
        return self.env.dt


class ArmTaskEnv(gym.Env):
    """Planar human-arm waypoint-reaching environment."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    # Task defaults (production evaluation standard lives in evaluation.py and
    # imports these so there is a single source).
    DEFAULT_GOAL_TOLERANCE: float = 0.05          # m
    DEFAULT_EE_SPEED_TOLERANCE: float = 0.10      # m/s
    DEFAULT_JOINT_SPEED_TOLERANCE: float = 1.0    # rad/s (norm)
    DEFAULT_HOLD_STEPS: int = 25                  # 0.5 s at 50 Hz
    DEFAULT_HOLD_DECREMENT: int = 5
    DEFAULT_INTERMEDIATE_HOLD_STEPS: int = 5      # 0.1 s
    DEFAULT_MAX_CURRICULUM_TOLERANCE: float = 0.15
    DEFAULT_MAX_EPISODE_STEPS: int = 500          # 10 s at 50 Hz
    DEFAULT_ORIENTATION_TOLERANCE_DEG: float = 12.0
    EE_SPEED_REF: float = 2.0                     # m/s, observation scale

    # Reward weights
    W_SHAPING: float = 2.0
    W_DIST: float = 0.5
    W_EFFORT: float = 0.02
    W_SMOOTH: float = 0.05
    W_LIMIT: float = 0.1     # per joint resting on a hard stop (end-range is avoided by humans)
    # Hand speed above a human-like cap is penalised quadratically. Peak hand
    # speed in human point-to-point reaching is ~1-1.5 m/s for 30-50 cm moves;
    # without this the policy whips the hand at 4-5 m/s (bang-bang start,
    # velocity-profile symmetry ~0.2 instead of ~0.45).
    W_SPEED_EXCESS: float = 0.5
    EE_SPEED_CAP: float = 1.5  # m/s
    # Hand-acceleration penalty (Flash & Hogan 1985 minimum-jerk proxy):
    # -W_ACCEL * (|a_hand| / ACCEL_REF)^2, capped. Off by default; the
    # "clean" training stage enables it.
    W_ACCEL: float = 0.0
    ACCEL_REF: float = 10.0    # m/s^2
    ACCEL_PENALTY_CAP: float = 4.0
    R_IN_GOAL: float = 1.0
    R_INTERMEDIATE: float = 10.0
    R_SUCCESS: float = 50.0

    ACTUATION_MUSCLE: str = "muscle"
    ACTUATION_TORQUE: str = "torque"
    REWARD_WEIGHT_KEYS = ("W_SHAPING", "W_DIST", "W_EFFORT", "W_SMOOTH", "W_LIMIT", "W_SPEED_EXCESS",
                          "EE_SPEED_CAP", "W_ACCEL", "ACCEL_REF", "ACCEL_PENALTY_CAP",
                          "R_IN_GOAL", "R_INTERMEDIATE", "R_SUCCESS")

    def __init__(
        self,
        preset: str = DEFAULT_PRESET,
        actuation_mode: str = ACTUATION_MUSCLE,
        goal_direction: str = "EAST",
        shoulder_base_position: Optional[Sequence[float]] = None,
        goal_tolerance: Optional[float] = None,
        ee_speed_tolerance: Optional[float] = None,
        joint_speed_tolerance: Optional[float] = None,
        hold_steps_required: Optional[int] = None,
        hold_decrement: Optional[int] = None,
        intermediate_hold_steps: Optional[int] = None,
        max_curriculum_tolerance: float = DEFAULT_MAX_CURRICULUM_TOLERANCE,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        terminate_on_success: bool = False,
        require_orientation: bool = False,
        orientation_tolerance_deg: Optional[float] = None,
        discount: float = 0.99,
        randomize_initial_state: bool = True,
        hand_load_range: Tuple[float, float] = (0.0, 0.0),
        obs_noise_std: Optional[Tuple[float, float]] = None,
        sensory_delay_steps: int = 0,
        config: Optional[ArmConfiguration] = None,
        activation_time_constants: Optional[Sequence[float]] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        random_waypoints: Optional[int] = None,
        waypoint_region: Optional[Sequence[float]] = None,
        waypoint_grid: float = 0.02,
        render_mode: Optional[str] = None,
        # Legacy keyword accepted and ignored so old call sites do not break.
        use_2dof: bool = True,
        **legacy_kwargs: Any,
    ):
        super().__init__()
        if "hold_velocity_tolerance" in legacy_kwargs and joint_speed_tolerance is None:
            joint_speed_tolerance = float(legacy_kwargs.pop("hold_velocity_tolerance"))
        legacy_kwargs.pop("muscle_moment_arm", None)
        legacy_kwargs.pop("muscle_params", None)
        if legacy_kwargs:
            raise TypeError(f"unexpected keyword arguments: {sorted(legacy_kwargs)}")

        self.render_mode = render_mode
        self.preset_name = str(preset)
        self.config: ArmConfiguration = config if config is not None else ArmConfiguration.get_preset(preset)
        if activation_time_constants is not None:
            # First-order excitation -> activation dynamics (Zajac 1989):
            # muscles cannot change force instantly, which rules out bang-bang
            # commands and gives bell-shaped hand-speed profiles.
            data = self.config.to_dict()
            data["activation_time_constants"] = [float(activation_time_constants[0]),
                                                 float(activation_time_constants[1])]
            self.config = ArmConfiguration.from_dict(data)
        # Reward weights are class defaults, overridable per env and recorded
        # in describe() so a package reproduces its training reward exactly.
        for key, val in (reward_weights or {}).items():
            if not key.isupper() or not hasattr(type(self), key):
                raise ValueError(f"unknown reward weight {key!r}")
            setattr(self, key, float(val))
        ok, msg = self.config.validate()
        if not ok:
            raise ValueError(f"invalid arm configuration: {msg}")
        self.num_dof = int(self.config.dof)
        self.dt = float(self.config.dt)
        self.reach = float(self.config.reach)

        mode = str(actuation_mode).strip().lower()
        if mode == "velocity":
            raise ValueError(
                "actuation_mode='velocity' was removed: a kinematic velocity command "
                "is incompatible with gravity and reported the action as the joint "
                "velocity. Use 'torque' for the non-muscle baseline."
            )
        if mode not in (self.ACTUATION_MUSCLE, self.ACTUATION_TORQUE):
            raise ValueError(f"unknown actuation_mode {actuation_mode!r}")
        self.actuation_mode = mode

        self.shoulder_base_position = (
            np.asarray(shoulder_base_position, dtype=float)
            if shoulder_base_position is not None
            else np.array([1.0, 0.0], dtype=float)
        )

        # Task parameters
        self.goal_tolerance = float(goal_tolerance if goal_tolerance is not None else self.DEFAULT_GOAL_TOLERANCE)
        self.height_tolerance = self.goal_tolerance  # legacy alias
        self.ee_speed_tolerance = float(ee_speed_tolerance if ee_speed_tolerance is not None else self.DEFAULT_EE_SPEED_TOLERANCE)
        self.joint_speed_tolerance = float(joint_speed_tolerance if joint_speed_tolerance is not None else self.DEFAULT_JOINT_SPEED_TOLERANCE)
        self.hold_steps_required = int(hold_steps_required if hold_steps_required is not None else self.DEFAULT_HOLD_STEPS)
        self.hold_decrement = int(hold_decrement if hold_decrement is not None else self.DEFAULT_HOLD_DECREMENT)
        self.intermediate_hold_steps = int(intermediate_hold_steps if intermediate_hold_steps is not None else self.DEFAULT_INTERMEDIATE_HOLD_STEPS)
        self.max_curriculum_tolerance = float(max_curriculum_tolerance)
        self.max_episode_steps = int(max_episode_steps)
        self.terminate_on_success = bool(terminate_on_success)
        self.require_orientation = bool(require_orientation)
        self.orientation_tolerance = float(np.deg2rad(
            orientation_tolerance_deg if orientation_tolerance_deg is not None
            else self.DEFAULT_ORIENTATION_TOLERANCE_DEG))
        self.discount = float(discount)
        self.randomize_initial_state = bool(randomize_initial_state)
        self.hand_load_range = (float(hand_load_range[0]), float(hand_load_range[1]))
        self.obs_noise_std = None if obs_noise_std is None else (float(obs_noise_std[0]), float(obs_noise_std[1]))
        self.sensory_delay_steps = int(sensory_delay_steps)
        if self.goal_tolerance > self.max_curriculum_tolerance:
            raise ValueError(
                f"goal_tolerance {self.goal_tolerance} exceeds max_curriculum_tolerance "
                f"{self.max_curriculum_tolerance}")

        # Actuators
        self._pairs: List[JointMusclePair] = []
        if self.actuation_mode == self.ACTUATION_MUSCLE:
            self._pairs = [JointMusclePair.from_config(self.config, j) for j in range(self.num_dof)]
        self._act_tau = self.config.activation_time_constants
        self.use_activation_dynamics = self._act_tau is not None
        self._torque_limits = (
            np.asarray(self.config.torque_limits, dtype=float)
            if self.config.torque_limits is not None
            else np.full(self.num_dof, 10.0)
        )

        # Action space
        if self.actuation_mode == self.ACTUATION_MUSCLE:
            self.action_space = spaces.Box(0.0, 1.0, shape=(2 * self.num_dof,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(self.num_dof,), dtype=np.float32)
        self.action_dim = int(self.action_space.shape[0])

        # Observation space
        n = self.num_dof
        low = [-1.0] * (2 * n) + [-1.0] * n + [-2.0, -2.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0]
        high = [1.0] * (2 * n) + [1.0] * n + [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        if self.use_activation_dynamics:
            low += [0.0] * (2 * n)
            high += [1.0] * (2 * n)
        if self.sensory_delay_steps > 0:
            for _ in range(self.sensory_delay_steps):
                low += list(map(float, self.action_space.low))
                high += list(map(float, self.action_space.high))
        self.observation_space = spaces.Box(
            low=np.asarray(low, dtype=np.float32), high=np.asarray(high, dtype=np.float32), dtype=np.float32)
        self.obs_dim = int(self.observation_space.shape[0])

        # Goal state
        self.goal_direction = "EAST"
        self.goal_position = np.zeros(2)
        self.goal_height = 0.0
        self.waypoints: Optional[List[np.ndarray]] = None
        self.current_waypoint_index = 0
        self._waypoint_sampler: Optional[Callable[[np.random.Generator], Sequence[Sequence[float]]]] = None
        # Random-waypoint task: every reset draws `random_waypoints` points from
        # the feasible region (a 2 cm grid of points passing waypoint_feasibility
        # at the coarsest curriculum tolerance, optionally clipped to
        # waypoint_region = [xmin, xmax, ymin, ymax] in the workspace frame).
        self.random_waypoints: Optional[int] = int(random_waypoints) if random_waypoints else None
        self.waypoint_region: Optional[List[float]] = (
            [float(v) for v in waypoint_region] if waypoint_region is not None else None)
        self.waypoint_grid = float(waypoint_grid)
        self._feasible_points: Optional[np.ndarray] = None
        self._configure_goal(goal_direction)
        if self.random_waypoints is not None:
            if self.random_waypoints < 1:
                raise ValueError("random_waypoints must be >= 1")
            self.goal_direction = "WAYPOINTS"

        # Dynamic state
        self.q = np.array(self.config.initial_angles, dtype=float)
        self.qd = np.zeros(self.num_dof)
        self.state: Optional[np.ndarray] = None
        self._activation = np.zeros(2 * self.num_dof)
        self._cfg_dyn: ArmConfiguration = self.config
        self.hand_load_kg = 0.0
        self.step_count = 0
        self.hold_counter = 0
        self.success = False
        self.success_step: Optional[int] = None
        self._success_paid = False
        self._entered_from_outside = False
        self.best_distance = float("inf")
        self._last_action = np.zeros(self.action_dim)
        self._prev_ee_v = np.zeros(2)
        self._action_history: deque = deque(maxlen=max(self.sensory_delay_steps, 1))
        self._obs_buffer: deque = deque(maxlen=self.sensory_delay_steps + 1)
        self._last_reward_terms: Dict[str, float] = {}
        self._last_tau = np.zeros(self.num_dof)
        self._last_at_limit = np.zeros(self.num_dof, dtype=bool)
        self._last_muscle: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Configuration setters (curriculum / evaluation hooks)
    # ------------------------------------------------------------------ #
    def set_goal_tolerance(self, tolerance: float) -> None:
        """Position tolerance of the in-goal test (m). Curriculum-controlled."""
        tolerance = float(tolerance)
        if tolerance <= 0.0:
            raise ValueError(f"goal_tolerance must be positive, got {tolerance}")
        if tolerance > self.max_curriculum_tolerance + 1e-12:
            raise ValueError(
                f"goal_tolerance {tolerance} exceeds max_curriculum_tolerance "
                f"{self.max_curriculum_tolerance}; waypoint spacing is validated "
                f"against that bound (construct the env with a larger value to allow it)")
        self.goal_tolerance = tolerance
        self.height_tolerance = tolerance

    def set_hold_steps_required(self, steps: int) -> None:
        if int(steps) < 1:
            raise ValueError(f"hold_steps_required must be >= 1, got {steps}")
        self.hold_steps_required = int(steps)

    def set_hold_decrement(self, decrement: int) -> None:
        if int(decrement) < 1:
            raise ValueError(f"hold_decrement must be >= 1, got {decrement}")
        self.hold_decrement = int(decrement)

    def set_intermediate_hold_steps(self, steps: int) -> None:
        if int(steps) < 0:
            raise ValueError("intermediate_hold_steps must be >= 0")
        self.intermediate_hold_steps = int(steps)

    def set_orientation_tolerance(self, tolerance_deg: float) -> None:
        if tolerance_deg <= 0.0:
            raise ValueError("orientation tolerance must be positive")
        self.orientation_tolerance = float(np.deg2rad(tolerance_deg))

    def set_terminate_on_success(self, flag: bool) -> None:
        self.terminate_on_success = bool(flag)

    def set_waypoint_sampler(self, sampler: Optional[Callable]) -> None:
        """Random-waypoint hook: ``sampler(np_random) -> list of [x, y]``, called at reset."""
        self._waypoint_sampler = sampler

    def set_hand_load(self, kg: float) -> None:
        """Add a point mass at the hand (end of the last link)."""
        self.hand_load_kg = float(kg)
        self._cfg_dyn = self._config_with_hand_load(self.hand_load_kg)

    def _config_with_hand_load(self, kg: float) -> ArmConfiguration:
        if kg <= 0.0:
            return self.config
        data = self.config.to_dict()
        j = self.num_dof - 1
        L = float(data["link_lengths"][j])
        m = float(data["masses"][j])
        lc = float(data["com_offsets"][j])
        I = float(data["inertias"][j])
        m_new = m + kg
        lc_new = (m * lc + kg * L) / m_new
        # Parallel-axis: both the segment and the load move relative to the new COM.
        I_new = I + m * (lc - lc_new) ** 2 + kg * (L - lc_new) ** 2
        data["masses"][j] = m_new
        data["com_offsets"][j] = lc_new
        data["inertias"][j] = I_new
        return ArmConfiguration.from_dict(data)

    # ------------------------------------------------------------------ #
    # Goals and waypoints
    # ------------------------------------------------------------------ #
    def _configure_goal(self, direction: str) -> None:
        """Directional single goals. All reduce to one explicit 2-D goal."""
        direction = str(direction).strip().upper()
        R = self.reach
        s = self.shoulder_base_position
        if direction == "WAYPOINTS":
            return
        if direction == "EAST":
            goal = s + np.array([R, 0.0])
        elif direction == "WEST":
            goal = s + np.array([-R, 0.0])
        elif direction == "NORTH":
            goal = s + np.array([0.0, R])
        elif direction == "HEIGHT":
            # Legacy "shoulder height" task, now a real 2-D goal in front of the body.
            goal = s + np.array([0.8 * R, 0.0])
        elif direction == "EXPLICIT":
            goal = np.asarray(self.goal_position, dtype=float)
        else:
            direction = "EAST"
            goal = s + np.array([R, 0.0])
        self.goal_direction = direction
        self.waypoints = None
        self.current_waypoint_index = 0
        self._set_goal(goal)

    def _set_goal(self, position: Sequence[float]) -> None:
        self.goal_position = np.asarray(position, dtype=float).reshape(2).copy()
        self.goal_height = float(self.goal_position[1])
        self._entered_from_outside = False

    def set_goal_position(self, position: Sequence[float]) -> None:
        """Single explicit goal (used by the Fitts / power-law validators)."""
        pos = np.asarray(position, dtype=float).reshape(-1)
        if pos.shape != (2,):
            raise ValueError(f"goal position must have shape (2,), got {pos.shape}")
        self.goal_direction = "EXPLICIT"
        self.waypoints = None
        self.current_waypoint_index = 0
        self._set_goal(pos)

    def set_waypoints(self, positions, tolerance: Optional[float] = None) -> None:
        """Ordered list of [x, y] waypoints in the workspace frame.

        Raises ValueError if the sequence could be walked by a motionless arm:
        every pair must be more than 2 * max_curriculum_tolerance apart and the
        first waypoint must be farther than that from the start pose.
        """
        try:
            positions_list = list(positions)
        except TypeError as exc:
            raise TypeError("positions must be an iterable of 2D points") from exc
        if len(positions_list) < 1:
            raise ValueError("waypoints list must contain at least one point")
        wps: List[np.ndarray] = []
        for i, raw in enumerate(positions_list):
            arr = np.asarray(raw, dtype=float).reshape(-1)
            if arr.shape != (2,):
                raise ValueError(f"waypoint {i} must have shape (2,), got {arr.shape}")
            wps.append(arr.copy())
        self._validate_waypoint_spacing(wps)
        for i, wp in enumerate(wps):
            f = self.waypoint_feasibility(wp, self.max_curriculum_tolerance)
            if not f.get("feasible", False):
                warnings.warn(f"waypoint {i} {wp.tolist()} is not robustly feasible: {f.get('reason')}",
                              RuntimeWarning, stacklevel=2)
        if tolerance is not None:
            self.set_goal_tolerance(tolerance)
        self.waypoints = wps
        self.current_waypoint_index = 0
        self.goal_direction = "WAYPOINTS"
        self._apply_waypoint(0)

    def _validate_waypoint_spacing(self, wps: List[np.ndarray]) -> None:
        bound = 2.0 * self.max_curriculum_tolerance
        for i in range(len(wps)):
            for j in range(i + 1, len(wps)):
                d = float(np.linalg.norm(wps[i] - wps[j]))
                if d <= bound:
                    raise ValueError(
                        f"waypoints {i} and {j} are {d:.3f} m apart; must exceed "
                        f"2 * max_curriculum_tolerance = {bound:.3f} m so a motionless "
                        f"arm can never satisfy two of them at once")
        start = self._end_effector_from_angles(np.asarray(self.config.initial_angles, dtype=float))
        d0 = float(np.linalg.norm(wps[0] - start))
        jitter = float(np.max(np.abs(self.config.initial_angle_jitter_max) + np.abs(self.config.initial_angle_jitter_min))) * self.reach
        if d0 <= self.max_curriculum_tolerance + jitter:
            raise ValueError(
                f"waypoint 0 is {d0:.3f} m from the start pose; must exceed "
                f"max_curriculum_tolerance + jitter margin = {self.max_curriculum_tolerance + jitter:.3f} m")

    def _apply_waypoint(self, index: int) -> None:
        if self.waypoints is None or not (0 <= index < len(self.waypoints)):
            raise IndexError(f"waypoint index {index} out of range")
        self._set_goal(self.waypoints[index])

    def clear_waypoints(self) -> None:
        self.waypoints = None
        self.current_waypoint_index = 0
        if self.goal_direction == "WAYPOINTS":
            self._configure_goal("EAST")

    @property
    def num_waypoints(self) -> int:
        return len(self.waypoints) if self.waypoints is not None else 1

    @property
    def is_final_waypoint(self) -> bool:
        return self.waypoints is None or self.current_waypoint_index >= len(self.waypoints) - 1

    # ------------------------------------------------------------------ #
    # Kinematics helpers
    # ------------------------------------------------------------------ #
    def _link_positions(self, q: np.ndarray) -> np.ndarray:
        """(n+1, 2) joint positions in the workspace frame (mirrored for a left arm)."""
        pos = ArmKinematics.forward_kinematics(np.asarray(q, dtype=float), self.config)[:, :2]
        return pos + self.shoulder_base_position

    def _end_effector_from_angles(self, q: np.ndarray) -> np.ndarray:
        return self._link_positions(q)[-1]

    def _get_end_effector_position(self, angles: np.ndarray) -> np.ndarray:  # legacy name
        return self._end_effector_from_angles(angles)

    def _end_effector_velocity(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        v = dynamics.end_effector_jacobian(q, self._cfg_dyn) @ np.asarray(qd, dtype=float)
        v[0] *= self.config.mirror_sign
        return v

    @staticmethod
    def _angle_normalize(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _compute_orientation(self, q: np.ndarray) -> float:
        return self._angle_normalize(float(np.sum(q)))

    def _local(self, position: Sequence[float]) -> np.ndarray:
        """Workspace point -> shoulder-local, un-mirrored coordinates."""
        p = np.asarray(position, dtype=float).reshape(2) - self.shoulder_base_position
        p[0] *= self.config.mirror_sign
        return p

    def solve_ik(self, position: Sequence[float]) -> Optional[np.ndarray]:
        """Joint angles placing the hand at ``position`` (elbow-down branch for
        2 links, damped least squares with a mid-range preference otherwise).
        Returns None if unreachable or outside the range of motion."""
        p = self._local(position)
        L = np.asarray(self.config.link_lengths, dtype=float)
        qmin = np.asarray(self.config.joint_limits_min, dtype=float)
        qmax = np.asarray(self.config.joint_limits_max, dtype=float)
        r = float(np.hypot(p[0], p[1]))
        if r > np.sum(L) + 1e-9 or r < 1e-9:
            return None
        if self.num_dof == 2:
            l0, l1 = L
            if r < abs(l0 - l1) - 1e-9:
                return None
            c = np.clip((r ** 2 - l0 ** 2 - l1 ** 2) / (2.0 * l0 * l1), -1.0, 1.0)
            elbow = float(np.arccos(c))
            shoulder = self._angle_normalize(
                float(np.arctan2(p[1], p[0])) - float(np.arctan2(l1 * np.sin(elbow), l0 + l1 * np.cos(elbow))))
            q = np.array([shoulder, elbow])
        else:
            q = self.config.joint_mid_angles.copy()
            for _ in range(200):
                pos = ArmKinematics.forward_kinematics(q, self.config)[:, :2]
                ee = pos[-1].copy()
                ee[0] *= self.config.mirror_sign
                err = p - ee
                if np.linalg.norm(err) < 1e-6:
                    break
                J = dynamics.end_effector_jacobian(q, self.config)
                lam = 1e-3
                dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(2), err)
                # Null-space pull toward mid-range keeps redundant joints neutral.
                N = np.eye(self.num_dof) - np.linalg.pinv(J) @ J
                dq += 0.1 * N @ (self.config.joint_mid_angles - q)
                q = np.clip(q + dq, qmin, qmax)
            pos = ArmKinematics.forward_kinematics(q, self.config)[:, :2]
            ee = pos[-1].copy(); ee[0] *= self.config.mirror_sign
            if np.linalg.norm(p - ee) > 1e-3:
                return None
        if np.any(q < qmin - 1e-9) or np.any(q > qmax + 1e-9):
            return None
        return q

    def _achievable_orientation(self, position: Sequence[float]) -> Optional[float]:
        q = self.solve_ik(position)
        return None if q is None else self._compute_orientation(q)

    def waypoint_feasibility(self, position: Optional[Sequence[float]] = None,
                             tolerance: Optional[float] = None) -> Dict[str, Any]:
        """Geometric + static feasibility of holding the hand at ``position``."""
        pos = np.asarray(self.goal_position if position is None else position, dtype=float).reshape(2)
        tol = float(self.goal_tolerance if tolerance is None else tolerance)
        p = self._local(pos)
        r = float(np.hypot(p[0], p[1]))
        L = np.asarray(self.config.link_lengths, dtype=float)
        out: Dict[str, Any] = {"position": pos.tolist(), "tolerance": tol, "radius_from_shoulder": r}
        r_min = abs(L[0] - L[1]) if self.num_dof == 2 else 0.0
        # A goal closer to the reach circle than the coarsest curriculum
        # tolerance can be "reached" by a straight arm resting on the elbow's
        # end-range spring -- a degenerate hold that policies learn at the
        # coarse rungs and cannot unlearn (Stage B, v1 waypoints).
        reach_margin = max(tol, float(self.max_curriculum_tolerance)) + 0.03
        out["reach_margin_required"] = reach_margin
        out["reach_margin"] = float(np.sum(L)) - r
        if not (r_min + tol < r < float(np.sum(L)) - reach_margin):
            out.update(reachable=r < float(np.sum(L)), within_rom=False, feasible=False,
                       reason=(f"radius {r:.3f} m is within {reach_margin:.3f} m of the reach circle "
                               f"({np.sum(L):.3f} m): a straight-arm rest would count as in-goal at the "
                               f"coarse curriculum tolerance") if r >= float(np.sum(L)) - reach_margin else
                              f"radius {r:.3f} m is inside the inner workspace limit {r_min + tol:.3f} m")
            return out
        q = self.solve_ik(pos)
        out["reachable"] = True
        if q is None:
            out.update(within_rom=False, feasible=False, reason="inverse kinematics violates the range of motion")
            return out
        qmin = np.asarray(self.config.joint_limits_min, dtype=float)
        qmax = np.asarray(self.config.joint_limits_max, dtype=float)
        margin = float(min(np.min(q - qmin), np.min(qmax - q)))
        G = dynamics.gravity_vector(q, self._cfg_dyn)
        ratio = np.abs(G) / self._torque_limits
        # The pose that holds this target must sit OUTSIDE the passive end-range
        # zone. Inside it the spring pushes the joint back out and the reward
        # charges the end-range term of the joint-limit penalty (W_LIMIT), so a
        # target whose only pose lies in there asks the policy to hold something
        # the same MDP penalises. This is the joint-space twin of the reach
        # margin above: both exclude targets that can only be held degenerately.
        # A joint with no passive margin (legacy gravity-free presets) is exempt.
        m_min = np.asarray(self.config.passive_limit_margins_min, dtype=float)
        m_max = np.asarray(self.config.passive_limit_margins_max, dtype=float)
        clear_lo = (q - qmin) - np.where(m_min > 0, m_min, 0.0)
        clear_hi = (qmax - q) - np.where(m_max > 0, m_max, 0.0)
        end_range_clearance = float(min(np.min(clear_lo), np.min(clear_hi)))
        out.update(
            within_rom=True,
            joint_angles_deg=np.degrees(q).tolist(),
            rom_margin_deg=float(np.degrees(margin)),
            end_range_clearance_deg=float(np.degrees(end_range_clearance)),
            achievable_orientation_deg=float(np.degrees(self._compute_orientation(q))),
            gravity_torque=G.tolist(),
            torque_ratio=ratio.tolist(),
        )
        if end_range_clearance < 0.0:
            joint = int(np.argmin(np.minimum(clear_lo, clear_hi)))
            out.update(feasible=False,
                       reason=(f"holding here puts joint {joint} "
                               f"{np.degrees(-end_range_clearance):.1f} deg inside its passive end-range "
                               f"zone, which the spring resists and the joint-limit penalty charges for"))
            return out
        if np.any(ratio > 0.8):
            out.update(feasible=False, reason="holding here needs more than 80% of a torque limit")
            return out
        out.update(feasible=True, reason="ok")
        return out

    def hold_feasibility(self, position=None) -> Dict[str, Any]:  # legacy name
        return self.waypoint_feasibility(position)

    # ------------------------------------------------------------------ #
    # Random waypoints
    # ------------------------------------------------------------------ #
    def feasible_points(self) -> np.ndarray:
        """Grid of workspace points where a waypoint is robustly holdable
        (reach margin, range of motion, static torque), cached per env."""
        if self._feasible_points is None:
            S = self.shoulder_base_position
            R = self.reach
            g = self.waypoint_grid
            if self.waypoint_region is not None:
                xmin, xmax, ymin, ymax = self.waypoint_region
            else:
                xmin, xmax, ymin, ymax = S[0] - R, S[0] + R, S[1] - R, S[1] + R
            xs = np.arange(xmin, xmax + 1e-9, g)
            ys = np.arange(ymin, ymax + 1e-9, g)
            pts = []
            tol = self.max_curriculum_tolerance
            for x in xs:
                for y in ys:
                    if self.waypoint_feasibility([x, y], tol).get("feasible", False):
                        pts.append((float(x), float(y)))
            self._feasible_points = np.asarray(pts, dtype=float).reshape(-1, 2)
            if self._feasible_points.shape[0] == 0:
                raise ValueError("no feasible waypoint positions in the requested region")
        return self._feasible_points

    def sample_waypoints(self, rng: np.random.Generator, n: Optional[int] = None,
                         attempts: int = 500) -> List[List[float]]:
        """Draw ``n`` feasible waypoints that also satisfy the spacing rules."""
        n = int(self.random_waypoints if n is None else n)
        pts = self.feasible_points()
        half = 0.5 * self.waypoint_grid
        for _ in range(attempts):
            idx = rng.choice(pts.shape[0], size=n, replace=False)
            cand = pts[idx] + rng.uniform(-half, half, size=(n, 2))
            wps = [np.asarray(w, dtype=float) for w in cand]
            try:
                self._validate_waypoint_spacing(wps)
            except ValueError:
                continue
            return [w.tolist() for w in wps]
        raise RuntimeError(f"could not sample {n} waypoints satisfying the spacing rules in {attempts} attempts")

    def _set_waypoints_unchecked(self, wps: Sequence[Sequence[float]]) -> None:
        self.waypoints = [np.asarray(w, dtype=float).reshape(2).copy() for w in wps]
        self.current_waypoint_index = 0
        self.goal_direction = "WAYPOINTS"
        self._apply_waypoint(0)

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        rng = self.np_random

        if "waypoints" in options and options["waypoints"] is not None:
            self._set_waypoints_unchecked(options["waypoints"])       # one-off override (tests, demos)
        elif self._waypoint_sampler is not None and self.goal_direction == "WAYPOINTS":
            self.set_waypoints(self._waypoint_sampler(rng))
        elif self.random_waypoints is not None:
            self._set_waypoints_unchecked(self.sample_waypoints(rng))

        randomize = bool(options.get("randomize", self.randomize_initial_state))
        nominal = np.asarray(self.config.initial_angles, dtype=float)
        if "state" in options:
            q0, qd0 = options["state"]
            q = np.asarray(q0, dtype=float).copy()
            qd = np.asarray(qd0, dtype=float).copy()
        else:
            q = nominal.copy()
            qd = np.zeros(self.num_dof)
            if randomize:
                lo = np.asarray(self.config.initial_angle_jitter_min, dtype=float)
                hi = np.asarray(self.config.initial_angle_jitter_max, dtype=float)
                for _ in range(20):
                    q = nominal + rng.uniform(lo, hi)
                    ee = self._end_effector_from_angles(q)
                    if np.linalg.norm(ee - self.goal_position) > self.goal_tolerance:
                        break
        qmin = np.asarray(self.config.joint_limits_min, dtype=float)
        qmax = np.asarray(self.config.joint_limits_max, dtype=float)
        self.q = np.clip(q, qmin, qmax)
        self.qd = qd

        lo_kg, hi_kg = self.hand_load_range
        if "hand_load_kg" in options:
            self.set_hand_load(float(options["hand_load_kg"]))
        elif hi_kg > lo_kg:
            self.set_hand_load(float(rng.uniform(lo_kg, hi_kg)))
        elif self.hand_load_kg != 0.0 and hi_kg == lo_kg == 0.0:
            pass  # a load set explicitly via set_hand_load persists

        if self.goal_direction == "WAYPOINTS" and self.waypoints is not None:
            self.current_waypoint_index = 0
            self._apply_waypoint(0)
        else:
            self._entered_from_outside = False

        self.step_count = 0
        self.hold_counter = 0
        self.success = False
        self.success_step = None
        self._success_paid = False
        self.best_distance = float("inf")
        self._activation = np.zeros(2 * self.num_dof)
        self._last_action = np.zeros(self.action_dim)
        self._prev_ee_v = self._end_effector_velocity(self.q, self.qd)
        self._action_history.clear()
        for _ in range(self.sensory_delay_steps):
            self._action_history.append(np.zeros(self.action_dim))
        self._obs_buffer.clear()
        self._last_tau = np.zeros(self.num_dof)
        self._last_at_limit = np.zeros(self.num_dof, dtype=bool)
        self._last_muscle = {}
        self._last_reward_terms = {}
        self._sync_state()
        obs = self._observe()
        return obs, self.get_state_info()

    def _sync_state(self) -> None:
        self.state = np.concatenate([self.q, self.qd]).astype(np.float64)

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def _end_range_engagement(self, q: np.ndarray) -> np.ndarray:
        """Per joint, how far into the passive end-range margin it sits (0..1).

        Humans avoid end-range postures (ligament strain); leaning on the
        end-range spring is also a free way to hold a straight arm, which the
        joint-limit penalty should charge for even before the hard stop.
        """
        cfg = self.config
        qmin = np.asarray(cfg.joint_limits_min, dtype=float)
        qmax = np.asarray(cfg.joint_limits_max, dtype=float)
        m_min = np.asarray(cfg.passive_limit_margins_min, dtype=float)
        m_max = np.asarray(cfg.passive_limit_margins_max, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            low = np.where(m_min > 0, ((qmin + m_min) - q) / m_min, 0.0)
            high = np.where(m_max > 0, (q - (qmax - m_max)) / m_max, 0.0)
        return np.clip(np.maximum(low, high), 0.0, 1.0)

    def _in_goal(self, distance: float, ee_speed: float, joint_speed: float, q: np.ndarray) -> bool:
        ok = (distance < self.goal_tolerance
              and ee_speed < self.ee_speed_tolerance
              and joint_speed < self.joint_speed_tolerance)
        if ok and self.require_orientation:
            target = self._achievable_orientation(self.goal_position)
            if target is not None:
                err = abs(self._angle_normalize(self._compute_orientation(q) - target))
                ok = err < self.orientation_tolerance
        return bool(ok)

    def _raw_observation(self) -> np.ndarray:
        q = self.q.copy()
        qd = self.qd.copy()
        if self.obs_noise_std is not None:
            q = q + self.np_random.normal(0.0, self.obs_noise_std[0], size=q.shape)
            qd = qd + self.np_random.normal(0.0, self.obs_noise_std[1], size=qd.shape)
        ee = self._end_effector_from_angles(self.q)
        ee_v = self._end_effector_velocity(self.q, self.qd)
        rel = (self.goal_position - ee) / self.reach
        dist = float(np.linalg.norm(rel))
        in_goal = self._in_goal(dist * self.reach, float(np.linalg.norm(ee_v)), float(np.linalg.norm(self.qd)), self.q)
        n_wp = self.num_waypoints
        progress = self.current_waypoint_index / (n_wp - 1) if n_wp > 1 else 0.0
        hold_progress = min(1.0, self.hold_counter / float(self.hold_steps_required))
        parts = [np.sin(q), np.cos(q), np.clip(qd / self.config.velocity_limits, -1.0, 1.0)]
        parts.append(np.clip(rel, -2.0, 2.0))
        parts.append([min(dist, 2.0)])
        parts.append(np.clip(ee_v / self.EE_SPEED_REF, -1.0, 1.0))
        parts.append([progress, 1.0 if in_goal else 0.0, hold_progress])
        if self.use_activation_dynamics:
            parts.append(self._activation)
        obs = np.concatenate([np.asarray(p, dtype=float).reshape(-1) for p in parts])
        return obs

    def _observe(self) -> np.ndarray:
        raw = self._raw_observation()
        if self.sensory_delay_steps > 0:
            self._obs_buffer.append(raw)
            delayed = self._obs_buffer[0]
            raw = np.concatenate([delayed] + [np.asarray(a, dtype=float) for a in self._action_history])
        obs = np.asarray(raw, dtype=np.float32)
        if not np.all(np.isfinite(obs)):
            raise FloatingPointError("non-finite observation")
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def get_observation(self) -> np.ndarray:
        """Current observation without stepping (e.g. after set_goal_position)."""
        return self._observe() if self.sensory_delay_steps == 0 else np.asarray(
            np.concatenate([self._raw_observation()] + [np.asarray(a) for a in self._action_history]), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Actuation
    # ------------------------------------------------------------------ #
    def _actuator_torque(self, action: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        if self.actuation_mode == self.ACTUATION_TORQUE:
            self._last_muscle = {}
            return np.asarray(action, dtype=float) * self._torque_limits
        u = np.asarray(action, dtype=float)
        if self.use_activation_dynamics:
            tau_rise, tau_fall = float(self._act_tau[0]), float(self._act_tau[1])
            a = self._activation
            rate = np.where(u > a, 1.0 / max(tau_rise, 1e-6), 1.0 / max(tau_fall, 1e-6))
            a = a + self.dt * rate * (u - a)
            self._activation = np.clip(a, 0.0, 1.0)
        else:
            self._activation = np.clip(u, 0.0, 1.0)
        tau = np.zeros(self.num_dof)
        forces = []
        for j, pair in enumerate(self._pairs):
            t, diag = pair.torque(self._activation[2 * j], self._activation[2 * j + 1], float(q[j]), float(qd[j]))
            tau[j] = t
            forces.append((diag["force_pos"], diag["force_neg"]))
        self._last_muscle = {"forces": forces, "activations": self._activation.copy()}
        return tau

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before stepping")
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), self.action_space.low, self.action_space.high)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"action must have shape ({self.action_dim},), got {action.shape}")
        a = action.astype(float)
        self._last_action = a.copy()

        # Potential before the step, on the current goal.
        ee_before = self._end_effector_from_angles(self.q)
        phi_before = -float(np.linalg.norm(self.goal_position - ee_before)) / self.reach

        # Physics
        tau = self._actuator_torque(a, self.q, self.qd)
        tau = np.clip(tau, -self._torque_limits, self._torque_limits)
        q, qd, at_limit = dynamics.integrate(self.q, self.qd, tau, self._cfg_dyn)
        self.q, self.qd = q, qd
        self._last_tau = tau
        self._last_at_limit = at_limit
        self._sync_state()

        ee = self._end_effector_from_angles(q)
        ee_v = self._end_effector_velocity(q, qd)
        ee_speed = float(np.linalg.norm(ee_v))
        ee_accel = float(np.linalg.norm(ee_v - self._prev_ee_v)) / self.dt
        self._prev_ee_v = ee_v.copy()
        joint_speed = float(np.linalg.norm(qd))
        distance = float(np.linalg.norm(self.goal_position - ee))
        if distance > self.goal_tolerance:
            self._entered_from_outside = True
        in_goal = self._in_goal(distance, ee_speed, joint_speed, q)
        if distance < self.best_distance:
            self.best_distance = distance

        # Hold bookkeeping
        if in_goal and self._entered_from_outside:
            self.hold_counter = min(self.hold_counter + 1, self.max_episode_steps)
        else:
            self.hold_counter = max(0, self.hold_counter - self.hold_decrement)

        waypoint_advanced = False
        intermediate_bonus = 0.0
        success_bonus = 0.0
        if not self.is_final_waypoint:
            if self.hold_counter >= self.intermediate_hold_steps and in_goal:
                self.current_waypoint_index += 1
                self._apply_waypoint(self.current_waypoint_index)
                self.hold_counter = 0
                waypoint_advanced = True
                intermediate_bonus = self.R_INTERMEDIATE
                distance = float(np.linalg.norm(self.goal_position - ee))
                if distance > self.goal_tolerance:
                    self._entered_from_outside = True
                in_goal = self._in_goal(distance, ee_speed, joint_speed, q)
                if in_goal and self._entered_from_outside:
                    self.hold_counter = 1
        if self.is_final_waypoint and self.hold_counter >= self.hold_steps_required and not self.success:
            self.success = True
            self.success_step = self.step_count
        if self.success and not self._success_paid:
            self._success_paid = True
            success_bonus = self.R_SUCCESS

        # Reward
        phi_after = -distance / self.reach
        shaping = self.W_SHAPING * (self.discount * phi_after - phi_before)
        dist_term = -self.W_DIST * distance / self.reach
        effort = -self.W_EFFORT * float(np.dot(a, a))
        smooth = -self.W_SMOOTH * joint_speed / float(self.config.velocity_limits)
        end_range = self._end_range_engagement(q)
        limit_term = -self.W_LIMIT * (float(np.sum(at_limit)) + float(np.sum(end_range)))
        speed_excess = -self.W_SPEED_EXCESS * max(0.0, ee_speed - self.EE_SPEED_CAP) ** 2
        accel_term = -self.W_ACCEL * min((ee_accel / self.ACCEL_REF) ** 2, self.ACCEL_PENALTY_CAP)
        in_goal_term = self.R_IN_GOAL if (in_goal and self.is_final_waypoint) else 0.0
        reward = (shaping + dist_term + effort + smooth + limit_term + speed_excess + accel_term
                  + in_goal_term + intermediate_bonus + success_bonus)
        self._last_reward_terms = {
            "shaping": shaping, "distance": dist_term, "effort": effort, "smoothness": smooth,
            "joint_limit": limit_term, "speed_excess": speed_excess, "hand_accel": accel_term,
            "in_goal": in_goal_term, "intermediate": intermediate_bonus, "success": success_bonus,
        }
        if not np.isfinite(reward):
            raise FloatingPointError("non-finite reward")

        terminated = bool(self.success and self.terminate_on_success)
        truncated = bool(self.step_count >= self.max_episode_steps)

        if self.sensory_delay_steps > 0:
            self._action_history.append(a.copy())
        obs = self._observe()
        info = self._info(ee, ee_v, distance, in_goal, waypoint_advanced)
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Info / state
    # ------------------------------------------------------------------ #
    def _info(self, ee: np.ndarray, ee_v: np.ndarray, distance: float, in_goal: bool,
              waypoint_advanced: bool) -> Dict[str, Any]:
        q = self.q
        target_orientation = self._achievable_orientation(self.goal_position)
        orientation_error = (
            self._angle_normalize(self._compute_orientation(q) - target_orientation)
            if target_orientation is not None else 0.0)
        hold_progress = min(1.0, self.hold_counter / float(self.hold_steps_required))
        return {
            "goal_distance": float(distance),
            "height_error": float(ee[1] - self.goal_position[1]),
            "orientation_error": float(orientation_error),
            "orientation_error_abs": float(abs(orientation_error)),
            "target_orientation": target_orientation,
            "ee_speed": float(np.linalg.norm(ee_v)),
            "muscle_activation_dynamics": bool(self.use_activation_dynamics),
            "joint_speed": float(np.linalg.norm(self.qd)),
            "hold_counter": int(self.hold_counter),
            "hold_steps_required": int(self.hold_steps_required),
            "hold_progress": float(hold_progress),
            "in_goal_region": bool(in_goal),
            "success": bool(self.success),
            "success_step": self.success_step,
            "goal_reached": bool(self.success),
            "end_effector_position": ee.copy(),
            "ee_velocity": ee_v.copy(),
            "joint_angles": q.copy(),
            "joint_velocities": self.qd.copy(),
            "link_positions": self._link_positions(q),
            "shoulder_position": self.shoulder_base_position.copy(),
            "goal_height": float(self.goal_height),
            "goal_position": self.goal_position.copy(),
            "goal_direction": self.goal_direction,
            "goal_tolerance": float(self.goal_tolerance),
            "step": int(self.step_count),
            "best_distance": float(self.best_distance),
            "waypoints": [w.tolist() for w in self.waypoints] if self.waypoints is not None else None,
            "current_waypoint_index": int(self.current_waypoint_index),
            "num_waypoints": int(self.num_waypoints),
            "waypoints_reached": int(self.current_waypoint_index + (1 if self.success else 0)),
            "waypoint_advanced": bool(waypoint_advanced),
            "actuation_mode": self.actuation_mode,
            "actuator_torque": self._last_tau.copy(),
            "gravity_torque": dynamics.gravity_vector(q, self._cfg_dyn),
            "at_joint_limit": self._last_at_limit.copy(),
            "end_range_engagement": self._end_range_engagement(q),
            "muscle_forces": self._last_muscle.get("forces"),
            "muscle_activations": (self._last_muscle.get("activations").copy()
                                   if self._last_muscle.get("activations") is not None else None),
            "hand_load_kg": float(self.hand_load_kg),
            "reward_terms": dict(self._last_reward_terms),
        }

    def get_state_info(self) -> Dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Environment state not initialized. Call reset() first.")
        ee = self._end_effector_from_angles(self.q)
        ee_v = self._end_effector_velocity(self.q, self.qd)
        distance = float(np.linalg.norm(self.goal_position - ee))
        in_goal = self._in_goal(distance, float(np.linalg.norm(ee_v)), float(np.linalg.norm(self.qd)), self.q)
        info = self._info(ee, ee_v, distance, in_goal, False)
        info["distance_to_goal"] = float(distance)
        info["max_steps"] = int(self.max_episode_steps)
        info["workspace_origin"] = np.zeros(2)
        return info

    def describe(self) -> Dict[str, Any]:
        """Everything needed to rebuild this env identically (run manifest)."""
        return {
            "obs_version": OBS_VERSION,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "preset": self.preset_name,
            "config": self.config.to_dict(),
            "actuation_mode": self.actuation_mode,
            "goal_direction": self.goal_direction,
            "goal_position": self.goal_position.tolist(),
            "waypoints": [w.tolist() for w in self.waypoints] if self.waypoints is not None else None,
            "shoulder_base_position": self.shoulder_base_position.tolist(),
            "goal_tolerance": self.goal_tolerance,
            "ee_speed_tolerance": self.ee_speed_tolerance,
            "joint_speed_tolerance": self.joint_speed_tolerance,
            "hold_steps_required": self.hold_steps_required,
            "hold_decrement": self.hold_decrement,
            "intermediate_hold_steps": self.intermediate_hold_steps,
            "max_curriculum_tolerance": self.max_curriculum_tolerance,
            "max_episode_steps": self.max_episode_steps,
            "terminate_on_success": self.terminate_on_success,
            "require_orientation": self.require_orientation,
            "orientation_tolerance_deg": float(np.degrees(self.orientation_tolerance)),
            "discount": self.discount,
            "randomize_initial_state": self.randomize_initial_state,
            "hand_load_range": list(self.hand_load_range),
            "obs_noise_std": list(self.obs_noise_std) if self.obs_noise_std is not None else None,
            "sensory_delay_steps": self.sensory_delay_steps,
            "dt": self.dt,
            "physics_substeps": int(self.config.physics_substeps),
            "activation_time_constants": self.config.activation_time_constants,
            "reward_weights": {k: float(getattr(self, k)) for k in self.REWARD_WEIGHT_KEYS},
            "random_waypoints": self.random_waypoints,
            "waypoint_region": self.waypoint_region,
            "waypoint_grid": self.waypoint_grid,
        }

    @classmethod
    def from_description(cls, desc: Dict[str, Any], **overrides: Any) -> "ArmTaskEnv":
        """Inverse of describe(). ``overrides`` replace individual task settings."""
        if int(desc.get("obs_version", -1)) != OBS_VERSION:
            raise ValueError(
                f"manifest obs_version {desc.get('obs_version')} does not match this code's "
                f"OBS_VERSION {OBS_VERSION}; the model was trained on a different observation")
        cfg = ArmConfiguration.from_dict(desc["config"]) if "config" in desc else None
        kwargs = dict(
            preset=desc.get("preset", DEFAULT_PRESET),
            actuation_mode=desc.get("actuation_mode", cls.ACTUATION_MUSCLE),
            goal_direction=desc.get("goal_direction", "EAST") if desc.get("goal_direction") != "WAYPOINTS" else "EAST",
            shoulder_base_position=desc.get("shoulder_base_position"),
            goal_tolerance=desc.get("goal_tolerance"),
            ee_speed_tolerance=desc.get("ee_speed_tolerance"),
            joint_speed_tolerance=desc.get("joint_speed_tolerance"),
            hold_steps_required=desc.get("hold_steps_required"),
            hold_decrement=desc.get("hold_decrement"),
            intermediate_hold_steps=desc.get("intermediate_hold_steps"),
            max_curriculum_tolerance=desc.get("max_curriculum_tolerance", cls.DEFAULT_MAX_CURRICULUM_TOLERANCE),
            max_episode_steps=desc.get("max_episode_steps", cls.DEFAULT_MAX_EPISODE_STEPS),
            terminate_on_success=desc.get("terminate_on_success", False),
            require_orientation=desc.get("require_orientation", False),
            orientation_tolerance_deg=desc.get("orientation_tolerance_deg"),
            discount=desc.get("discount", 0.99),
            randomize_initial_state=desc.get("randomize_initial_state", True),
            hand_load_range=tuple(desc.get("hand_load_range", (0.0, 0.0))),
            obs_noise_std=tuple(desc["obs_noise_std"]) if desc.get("obs_noise_std") else None,
            sensory_delay_steps=desc.get("sensory_delay_steps", 0),
            config=cfg,
            reward_weights={k: v for k, v in (desc.get("reward_weights") or {}).items()
                            if k in cls.REWARD_WEIGHT_KEYS},
            random_waypoints=desc.get("random_waypoints"),
            waypoint_region=desc.get("waypoint_region"),
            waypoint_grid=desc.get("waypoint_grid", 0.02),
        )
        if "waypoints" in overrides:
            # Explicit waypoints pin the task (e.g. GUI-placed points on a
            # random-waypoint model): sampling is switched off.
            waypoints = overrides.pop("waypoints")
            kwargs["random_waypoints"] = None
        elif desc.get("random_waypoints"):
            waypoints = None                       # the env samples its own at reset
        else:
            waypoints = desc.get("waypoints")
        kwargs.update(overrides)
        env = cls(**kwargs)
        if waypoints:
            env.set_waypoints(waypoints)
        elif desc.get("goal_direction") == "EXPLICIT" and desc.get("goal_position") is not None:
            env.set_goal_position(desc["goal_position"])
        if env.obs_dim != int(desc.get("obs_dim", env.obs_dim)):
            raise ValueError(f"rebuilt env has obs_dim {env.obs_dim}, manifest says {desc.get('obs_dim')}")
        return env

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode == "human" and self.state is not None:
            ee = self._end_effector_from_angles(self.q)
            d = float(np.linalg.norm(self.goal_position - ee))
            print(f"Step {self.step_count:3d} | q={np.round(self.q, 3)} | hand={np.round(ee, 3)} | "
                  f"dist={d:.3f} | wp {self.current_waypoint_index + 1}/{self.num_waypoints} | "
                  f"hold {self.hold_counter}/{self.hold_steps_required}")

    def close(self):
        pass
