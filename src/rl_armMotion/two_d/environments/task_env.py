"""Task-specific environment for the 2-DOF robotic arm reinforcement learning project.

This module is the heart of the project — it defines the Gymnasium-compatible
environment in which the RL policy is trained and evaluated. Every training
run, every validator harness, and every GUI simulation steps through this env.

Physical setup
--------------
- Shoulder joint is fixed at workspace coordinates [1.0, 0] meters
- Two links: upper arm (1.0 m) + forearm (0.8 m), max reach 1.8 m from shoulder
- Initial pose: arm hangs straight down (shoulder = -90 deg, elbow = 0 deg)
- Workspace is the 2-D plane; the origin is at [0, 0]

Goal modes (the agent's task)
-----------------------------
The env supports several mutually exclusive goal modes, selected via
`goal_direction` or via calls to `set_goal_position` / `set_waypoints`:

  - "HEIGHT"      legacy: hold the end-effector at the shoulder height line
  - "EAST"        reach the rightmost edge of the workspace and hold
  - "WEST"        reach the leftmost edge
  - "NORTH"       reach straight up
  - "EXPLICIT"    user supplied an arbitrary (x, y) goal via set_goal_position
                  (used by the Fitts' Law validator to place targets at any
                  point in the workspace)
  - "WAYPOINTS"   user supplied an ordered list of (x, y) waypoints via
                  set_waypoints; intermediate waypoints advance on touch,
                  the final waypoint requires the full hold criterion

Actuation modes (how the policy's actions are interpreted)
----------------------------------------------------------
Two actuation modes, selected via the `actuation_mode` constructor parameter:

  - "velocity" (default, legacy)
      action[j] in [-1, 1] is interpreted as a normalised joint-velocity
      command. Step physics: angle <- angle + action * velocity_limit * dt
      followed by joint-limit clipping. Simple kinematic Euler.

  - "muscle"   (Phase 6 of the Fischer 2021 integration)
      action[2j], action[2j+1] in [0, 1] are interpreted as the (extensor,
      flexor) activations of an antagonist muscle pair at joint j. Step
      physics: Hill-type force per muscle -> net torque via configurable
      moment arm -> Euler integration against joint inertia with damping.
      Smoothness emerges from the muscle force-velocity curve, not from
      reward shaping. This is the actuation model Fischer et al. (2021)
      used in their MuJoCo musculoskeletal arm.

Observation space (11-dim, both actuation modes)
------------------------------------------------
[sin(theta_0), cos(theta_0),     joint 0 angle as unit-circle coords (continuous)
 sin(theta_1), cos(theta_1),     joint 1 angle as unit-circle coords
 vel_0_norm, vel_1_norm,         joint velocities normalised to velocity_limits
 signed_goal_error,              signed scalar error along goal axis (metres)
 signed_orientation_error,       signed end-effector orientation error (rad)
 gradient_norm,                  rate-of-change of total error in [0, 1]
 in_goal_region_flag,            1 if currently in goal region, else 0
 hold_progress]                  fraction of required hold steps completed

Reward (computed in step(), see commentary inline)
--------------------------------------------------
A 10-term shaped reward: five continuous penalties (distance, orientation,
velocity-norm, gradient-norm, action-norm) + four shaping bonuses (progress,
proximity ramp, in-goal constant, hold-growth) + a +150 terminal bonus on
successful hold completion. See docs/Reward_System_Report.pdf for the
complete formulation and rationale.

Fischer 2021 integration
------------------------
The full Fischer 2021 RL methodology is implemented around this env:
SAC with the curriculum callback (see training/curriculum_callback.py)
adjusts the goal tolerance during training via set_goal_tolerance();
the Fitts' Law and 2/3 Power Law harnesses (see validation/) evaluate
trained policies against the canonical biological motion benchmarks.
"""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_armMotion.two_d.config import ArmConfiguration
from rl_armMotion.two_d.utils.arm_kinematics import ArmController, ArmKinematics


class ActionSmoother(gym.ActionWrapper):
    """EMA low-pass filter on actions for smoother muscle commands.

    Wraps any env and applies exponential moving average to the action
    before passing it to the underlying env. Use at evaluation/visualization
    time — does not affect training.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    alpha : float
        Smoothing factor in (0, 1]. Lower = smoother but more lag.
        0.3 is a good default for muscle actuation.
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


class ArmTaskEnv(gym.Env):
    """
    2-DOF Robotic Arm Task Environment.

    Task objective:
    1. Reach the selected target (height line or directional far point)
    2. Align end-effector orientation to direction-specific target orientation
    3. Hold the pose for a required number of consecutive steps

    Observation (11-dim):
    [sin(theta1), cos(theta1), sin(theta2), cos(theta2),
     vel1_norm, vel2_norm,
     signed_height_error,
     signed_orientation_error,
     gradient_norm,
     in_goal_region_flag,
     hold_progress]

    Action (2-dim):
    Normalized joint velocity commands in [-1, 1], scaled by velocity limits.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # Default goal tolerance, in metres. Matches the historical hardcoded value
    # of 10 cm. The adaptive curriculum following Fischer et al. (2021) starts
    # near 0.60 m (their reported initial radius) and decays toward this value
    # as the success rate improves; see ArmTaskEnv.set_goal_tolerance().
    DEFAULT_GOAL_TOLERANCE: float = 0.10
    DEFAULT_ORIENTATION_TOLERANCE_DEG: float = 12.0
    DEFAULT_HOLD_VELOCITY_TOLERANCE: float = 0.50
    EMA_VELOCITY_ALPHA: float = 0.15

    # Actuation-mode constants. Phase 6 of the Fischer integration adds an
    # opt-in muscle-driven actuation path; the historical default is the
    # direct velocity-command actuation that the existing 49 regression tests
    # were written against.
    ACTUATION_VELOCITY: str = "velocity"
    ACTUATION_MUSCLE: str = "muscle"

    def __init__(
        self,
        render_mode: Optional[str] = None,
        shoulder_base_position: Optional[np.ndarray] = None,
        use_2dof: bool = True,
        goal_direction: str = "HEIGHT",
        goal_tolerance: Optional[float] = None,
        orientation_tolerance_deg: Optional[float] = None,
        hold_velocity_tolerance: Optional[float] = None,
        actuation_mode: str = "velocity",
        muscle_moment_arm: float = 0.05,
        muscle_params=None,
    ):
        """Construct the 2-DOF goal-reaching environment.

        Parameters
        ----------
        goal_tolerance : float, optional
            Position tolerance in metres for the goal-reached check. Defaults to
            ``DEFAULT_GOAL_TOLERANCE`` (0.10 m). Pass a larger value for early
            curriculum stages and a smaller value for late stages, following
            Fischer et al. (2021), Sci. Rep. 11:14445.
        orientation_tolerance_deg : float, optional
            End-effector orientation tolerance in degrees. Defaults to 10°.
        hold_velocity_tolerance : float, optional
            Maximum joint-velocity norm permitted while satisfying the hold
            condition. Defaults to 0.30 rad/s.
        actuation_mode : str, default "velocity"
            Selects how the policy's action vector is interpreted.

            ``"velocity"`` (default, backward-compatible) — actions are
            normalised joint-velocity commands in [-1, 1], applied
            directly via Euler integration. This is the original
            behaviour against which the existing 49 regression tests
            were written.

            ``"muscle"`` — actions are antagonist-pair muscle activations
            in [0, 1] (two muscles per joint: extensor and flexor). On
            each step, each pair's net torque is computed from the
            Hill-type force-length-velocity-activation product of
            :class:`rl_armMotion.two_d.utils.muscle_model.HillTypeMuscle`,
            and Euler-integrated against the joint inertia. This is the
            actuation model used by Fischer et al. (2021).
        muscle_moment_arm : float, default 0.05
            Effective moment arm in metres at which each muscle force acts
            on its joint. Used only in muscle mode. The default 5 cm is a
            reasonable order of magnitude for upper-extremity flexor
            muscles (Murray, Buchanan, & Delp, 1995, J. Biomech. 28:513).
        muscle_params : MuscleParameters, optional
            Override the default Hill-type muscle parameters for every
            muscle in the environment. Used only in muscle mode. Defaults
            to the canonical literature values (F_max=100 N, L_opt=0.10 m,
            v_max=10 L/s).
        """
        self.render_mode = render_mode
        self.use_2dof = use_2dof
        self.goal_direction = str(goal_direction).strip().upper()
        if self.goal_direction not in {"HEIGHT", "EAST", "WEST", "NORTH"}:
            self.goal_direction = "HEIGHT"

        # Actuation-mode validation. Anything other than the two known
        # strings falls back to the historical velocity-mode default so the
        # constructor never raises for a typo from a Gymnasium wrapper.
        mode = str(actuation_mode).strip().lower()
        if mode not in {self.ACTUATION_VELOCITY, self.ACTUATION_MUSCLE}:
            mode = self.ACTUATION_VELOCITY
        self.actuation_mode = mode

        # Load 2-DOF arm configuration with constraints
        self.config = ArmConfiguration.get_preset("2dof_simple")
        assert self.config.dof == 2, "This environment requires 2-DOF arm"

        self.num_dof = self.config.dof

        # Workspace setup
        self.workspace_origin = np.array([0.0, 0.0], dtype=np.float32)
        self.shoulder_base_position = (
            np.asarray(shoulder_base_position, dtype=np.float32)
            if shoulder_base_position is not None
            else np.array([1.0, 0.0], dtype=np.float32)
        )

        # Arm controller for dynamics
        self.controller = ArmController(self.config)

        # Muscle-mode set-up. Each joint gets an antagonist pair of Hill-
        # type muscles (an extensor and a flexor) so the agent has
        # independent control over positive and negative torque. The
        # action vector in muscle mode has length 2 * num_dof and is
        # bounded to [0, 1] (activation).
        self.muscle_moment_arm = float(muscle_moment_arm)
        self._muscles = None
        if self.actuation_mode == self.ACTUATION_MUSCLE:
            from rl_armMotion.two_d.utils.muscle_model import HillTypeMuscle
            # Two muscles per joint (extensor, flexor). All share the same
            # parameters by default; users wanting heterogeneous muscles
            # can subclass or post-construct.
            self._muscles = [
                (HillTypeMuscle(muscle_params), HillTypeMuscle(muscle_params))
                for _ in range(self.num_dof)
            ]

        # Action space depends on actuation mode.
        #   velocity mode : action[j]   in [-1, 1] (signed velocity command)
        #   muscle   mode : action[2*j], action[2*j+1] in [0, 1] (extensor,
        #                   flexor activations for joint j)
        if self.actuation_mode == self.ACTUATION_MUSCLE:
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2 * self.num_dof,),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_dof,),
                dtype=np.float32,
            )

        # Observation space
        max_reach = float(np.sum(self.config.link_lengths))
        obs_low = np.array(
            [
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -max_reach,
                -np.pi,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                max_reach,
                np.pi,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Simulation parameters
        self.dt = float(self.config.dt)
        self.max_episode_steps = 1000
        self.step_count = 0

        # State: [angles, velocities]
        self.state: Optional[np.ndarray] = None

        # Goal specification (legacy height mode or directional far-point mode).
        self.goal_height = float(self.shoulder_base_position[1])
        self.goal_position = np.array(
            [self.shoulder_base_position[0], self.goal_height],
            dtype=np.float32,
        )
        self.target_orientation = 0.0
        self.goal_axis = np.array([0.0, 1.0], dtype=np.float32)
        self._configure_goal(self.goal_direction)

        # Tolerances and hold constraints. Defaults preserve the historical
        # behaviour (10 cm, 10°, 0.30 rad/s); curriculum learning uses
        # set_goal_tolerance() to vary the position tolerance during training.
        self.height_tolerance = float(
            goal_tolerance if goal_tolerance is not None else self.DEFAULT_GOAL_TOLERANCE
        )
        self.orientation_tolerance = float(np.deg2rad(
            orientation_tolerance_deg
            if orientation_tolerance_deg is not None
            else self.DEFAULT_ORIENTATION_TOLERANCE_DEG
        ))
        self.hold_velocity_tolerance = float(
            hold_velocity_tolerance
            if hold_velocity_tolerance is not None
            else self.DEFAULT_HOLD_VELOCITY_TOLERANCE
        )
        self.hold_steps_required = 20                      # ~0.2 s hold — achievable in one episode
        self.gradient_scale = 5.0

        # Compatibility alias with previous code/tests
        self.goal_tolerance = self.height_tolerance

        # Track performance and hold state
        self.best_distance = float("inf")
        self.best_total_error = float("inf")
        self.previous_total_error = float("inf")
        self.hold_counter = 0
        self.last_gradient = 0.0
        self._ema_velocity_norm = 0.0

        # Waypoint state. When the user calls set_waypoints() the goal_direction
        # is switched to "WAYPOINTS" and self.waypoints holds an ordered list of
        # 2D points. self.current_waypoint_index advances through this list
        # whenever the active waypoint is reached; the episode terminates when
        # the final waypoint's standard hold criterion is satisfied.
        self.waypoints: Optional[List[np.ndarray]] = None
        self.current_waypoint_index: int = 0

    def set_goal_tolerance(self, tolerance: float) -> None:
        """Update the position tolerance used for the goal-reached check.

        This is the entry point used by the adaptive curriculum scheduler
        (Fischer et al., 2021, Sci. Rep. 11:14445), which begins training with
        a wide tolerance (~0.6 m) and shrinks it toward the production value
        (~0.02 m) as the agent's success rate over the recent episode window
        exceeds a configured threshold (Fischer used 80 %).

        The orientation and hold-velocity tolerances are intentionally NOT
        scheduled by this method; only the position tolerance is curriculum-
        controlled, matching the protocol in the source paper.
        """
        if tolerance <= 0.0:
            raise ValueError(f"goal_tolerance must be positive, got {tolerance}")
        self.height_tolerance = float(tolerance)
        self.goal_tolerance = self.height_tolerance

    def set_waypoints(self, positions, tolerance: Optional[float] = None) -> None:
        """Configure a sequence of 2D waypoints to be visited in order.

        Each waypoint is an [x, y] point in the workspace frame. During an
        episode the arm must reach the first waypoint, then the second, and
        so on. Intermediate waypoints are considered reached when the
        end-effector enters the position tolerance radius (touch-and-go);
        only the **final** waypoint requires the full hold criterion
        (position + orientation + velocity for hold_steps_required
        consecutive steps). Episode termination occurs when the final
        waypoint's hold criterion is met.

        Calling this method switches the environment to waypoint mode by
        setting ``goal_direction = "WAYPOINTS"`` and resets the current
        waypoint index to zero. The first waypoint is applied as the active
        goal immediately. The change persists across ``reset()``: a fresh
        episode begins again at the first waypoint.

        Parameters
        ----------
        positions : sequence of array-like, each of shape (2,)
            Ordered list of waypoints. Must contain at least one waypoint.
        tolerance : float, optional
            If supplied, ``set_goal_tolerance(tolerance)`` is called so the
            same tolerance applies to every waypoint. The adaptive
            curriculum scheduler may subsequently shrink this value.
        """
        try:
            positions_list = list(positions)
        except TypeError as exc:
            raise TypeError(
                f"positions must be an iterable of 2D points, got {type(positions).__name__}"
            ) from exc

        if len(positions_list) < 1:
            raise ValueError("waypoints list must contain at least one point")

        waypoints: List[np.ndarray] = []
        for i, raw_pos in enumerate(positions_list):
            arr = np.asarray(raw_pos, dtype=np.float32).reshape(-1)
            if arr.shape != (2,):
                raise ValueError(
                    f"waypoint {i} must have shape (2,), got shape {arr.shape}"
                )
            waypoints.append(arr.copy())

        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.goal_direction = "WAYPOINTS"

        if tolerance is not None:
            self.set_goal_tolerance(tolerance)

        self._apply_waypoint(0)

    def _apply_waypoint(self, index: int) -> None:
        """Activate the waypoint at ``index`` as the current goal.

        Updates the goal-related attributes (position, height, axis, target
        orientation) so that the existing reward and observation pipeline
        seamlessly treats the new waypoint as the active goal. Does not
        modify ``current_waypoint_index`` itself; the caller is responsible
        for that bookkeeping so the relationship between index and applied
        waypoint stays explicit.
        """
        if self.waypoints is None or not (0 <= index < len(self.waypoints)):
            raise IndexError(
                f"waypoint index {index} out of range for "
                f"{0 if self.waypoints is None else len(self.waypoints)} waypoints"
            )
        self._set_explicit_goal_state(self.waypoints[index])

    def _set_explicit_goal_state(self, position: np.ndarray) -> None:
        """Update goal_position / goal_height / goal_axis / target_orientation.

        Shared by ``set_goal_position`` (EXPLICIT mode) and ``_apply_waypoint``
        (WAYPOINTS mode). Does NOT touch ``goal_direction`` — the caller
        owns that, because the two modes mean different things downstream
        (see _compute_goal_distance and the waypoint advancement block in
        step()).

        The position is copied so external mutation of the caller's array
        cannot leak in. ``goal_axis`` is recomputed as the unit vector from
        the shoulder base to the goal; if the goal coincides with the
        shoulder it falls back to the +y axis to avoid divide-by-zero, and
        ``target_orientation`` is computed from the resulting axis so the
        orientation-error term in the reward remains well-defined.
        """
        pos = np.asarray(position, dtype=np.float32).reshape(-1)
        self.goal_position = pos.copy()
        self.goal_height = float(pos[1])

        delta = pos - self.shoulder_base_position
        norm = float(np.linalg.norm(delta))
        if norm > 1e-9:
            self.goal_axis = (delta / norm).astype(np.float32)
        else:
            self.goal_axis = np.array([0.0, 1.0], dtype=np.float32)

        self.target_orientation = float(
            np.arctan2(self.goal_axis[1], self.goal_axis[0])
        )

    def clear_waypoints(self) -> None:
        """Discard any waypoint sequence and return to the previous goal mode.

        After this call ``waypoints`` is ``None``, ``current_waypoint_index``
        is reset to 0, and ``goal_direction`` is set back to ``"HEIGHT"``.
        Callers wishing to restore a specific direction or explicit-position
        goal should call the corresponding setter (``_configure_goal``,
        ``set_goal_position``) after this.
        """
        self.waypoints = None
        self.current_waypoint_index = 0
        if self.goal_direction == "WAYPOINTS":
            self._configure_goal("HEIGHT")

    def set_goal_position(self, position) -> None:
        """Place the goal at an arbitrary 2D point in the workspace frame.

        This is the entry point used by the Fitts' Law validation harness
        (Fischer et al., 2021, Sci. Rep. 11:14445), which sweeps a grid of
        (distance, tolerance) conditions to test whether the trained policy's
        movement-time response obeys MT = a + b * log2(2D/W) with a high R^2.

        The goal_direction is set to "EXPLICIT" so that the Euclidean-distance
        branch of _compute_goal_distance is taken (the legacy "HEIGHT" branch
        only measures vertical error, which is not appropriate for arbitrary
        2D targets). The goal_axis is recomputed to point from the shoulder
        toward the new goal so that the signed-error projection used by the
        reward and observation code remains well-defined.

        Parameters
        ----------
        position : array-like of shape (2,)
            Target [x, y] coordinates in the workspace frame (the same frame
            in which shoulder_base_position is expressed).
        """
        pos = np.asarray(position, dtype=np.float32).reshape(-1)
        if pos.shape != (2,):
            raise ValueError(
                f"goal position must have shape (2,), got shape {pos.shape}"
            )

        self.goal_direction = "EXPLICIT"
        self._set_explicit_goal_state(pos)

    @staticmethod
    def _angle_normalize(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _compute_orientation(self, angles: np.ndarray) -> float:
        """Compute end-effector orientation for planar serial chain."""
        return self._angle_normalize(float(np.sum(angles)))

    def _compute_orientation_error(self, angles: np.ndarray) -> Tuple[float, float]:
        """Return signed and absolute orientation error from target orientation."""
        orientation = self._compute_orientation(angles)
        signed_error = self._angle_normalize(orientation - self.target_orientation)
        return signed_error, abs(signed_error)

    def _configure_goal(self, direction: str) -> None:
        """Configure directional goal target and orientation."""
        max_reach = float(np.sum(self.config.link_lengths))
        direction = str(direction).strip().upper()
        if direction not in {"HEIGHT", "EAST", "WEST", "NORTH"}:
            direction = "HEIGHT"
        self.goal_direction = direction

        if direction == "EAST":
            self.goal_axis = np.array([1.0, 0.0], dtype=np.float32)
            self.goal_position = self.shoulder_base_position + self.goal_axis * max_reach
            self.target_orientation = 0.0
        elif direction == "WEST":
            self.goal_axis = np.array([-1.0, 0.0], dtype=np.float32)
            self.goal_position = self.shoulder_base_position + self.goal_axis * max_reach
            self.target_orientation = np.pi
        elif direction == "NORTH":
            self.goal_axis = np.array([0.0, 1.0], dtype=np.float32)
            self.goal_position = self.shoulder_base_position + self.goal_axis * max_reach
            self.target_orientation = np.pi / 2.0
        else:
            # Legacy mode: target line at shoulder height.
            self.goal_axis = np.array([0.0, 1.0], dtype=np.float32)
            self.goal_position = np.array(
                [self.shoulder_base_position[0], self.shoulder_base_position[1]],
                dtype=np.float32,
            )
            self.target_orientation = 0.0

        self.goal_height = float(self.goal_position[1])

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state (arm vertical downward)."""
        super().reset(seed=seed)

        angles = np.array(self.config.initial_angles, dtype=np.float32)
        velocities = np.zeros(self.num_dof, dtype=np.float32)

        self.state = np.concatenate([angles, velocities]).astype(np.float32)
        self.step_count = 0
        self.best_distance = float("inf")
        self.best_total_error = float("inf")
        self.hold_counter = 0
        self.last_gradient = 0.0
        self._ema_velocity_norm = 0.0

        # In waypoint mode, rewind to the first waypoint at the start of every
        # episode so the agent always begins by attempting waypoint 0. The
        # waypoint list itself persists across resets.
        if self.goal_direction == "WAYPOINTS" and self.waypoints is not None:
            self.current_waypoint_index = 0
            self._apply_waypoint(0)

        self.controller.angles = angles.copy()

        end_effector = self._get_end_effector_position(angles)
        signed_height_error = self._compute_signed_goal_error(end_effector)
        signed_orientation_error, abs_orientation_error = self._compute_orientation_error(angles)
        self.previous_total_error = 2.0 * abs(signed_height_error) + abs_orientation_error

        obs = self._get_observation(
            angles=angles,
            velocities=velocities,
            signed_height_error=signed_height_error,
            signed_orientation_error=signed_orientation_error,
            gradient_norm=0.0,
            in_goal_region=False,
        )
        return obs, self.get_state_info()

    def _get_end_effector_position(self, angles: np.ndarray) -> np.ndarray:
        """Compute end-effector position in workspace frame."""
        positions = ArmKinematics.forward_kinematics(angles, self.config)
        end_effector_shoulder_frame = positions[-1, :2]
        return end_effector_shoulder_frame + self.shoulder_base_position

    def _compute_goal_distance(self, end_effector_pos: np.ndarray) -> float:
        """Compute distance to current goal definition."""
        if self.goal_direction == "HEIGHT":
            return abs(float(end_effector_pos[1] - self.goal_height))
        return float(np.linalg.norm(np.asarray(end_effector_pos, dtype=float) - self.goal_position))

    def _compute_signed_goal_error(self, end_effector_pos: np.ndarray) -> float:
        """Compute signed error projected onto goal axis."""
        if self.goal_direction == "HEIGHT":
            return float(end_effector_pos[1] - self.goal_height)
        delta = np.asarray(end_effector_pos, dtype=float) - self.goal_position
        return float(np.dot(delta, self.goal_axis))

    def _is_goal_reached(
        self,
        height_error_abs: float,
        orientation_error_abs: float,
        velocity_norm: float,
    ) -> bool:
        """Check if arm is in stable goal region for this step."""
        return (
            height_error_abs < self.height_tolerance
            and orientation_error_abs < self.orientation_tolerance
            and velocity_norm < self.hold_velocity_tolerance
        )

    def _get_observation(
        self,
        angles: np.ndarray,
        velocities: np.ndarray,
        signed_height_error: float,
        signed_orientation_error: float,
        gradient_norm: float,
        in_goal_region: bool,
    ) -> np.ndarray:
        """Build observation vector for policy."""
        velocities_norm = np.asarray(velocities, dtype=np.float32) / np.asarray(
            self.config.velocity_limits, dtype=np.float32
        )
        velocities_norm = np.clip(velocities_norm, -1.0, 1.0)

        hold_progress = min(1.0, self.hold_counter / float(self.hold_steps_required))

        obs = np.array(
            [
                np.sin(angles[0]),
                np.cos(angles[0]),
                np.sin(angles[1]),
                np.cos(angles[1]),
                velocities_norm[0],
                velocities_norm[1],
                signed_height_error,
                signed_orientation_error,
                gradient_norm,
                1.0 if in_goal_region else 0.0,
                hold_progress,
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        self.step_count += 1

        if self.state is None:
            raise RuntimeError("Environment must be reset before stepping")

        angles = self.state[: self.num_dof].copy()
        current_velocities = self.state[self.num_dof :].copy()

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.actuation_mode == self.ACTUATION_MUSCLE:
            # ---------- Muscle-driven actuation (Fischer 2021 style) ----------
            # The action vector has length 2 * num_dof. For joint j:
            #   action[2*j]     = extensor activation in [0, 1]
            #   action[2*j + 1] = flexor   activation in [0, 1]
            # Net joint torque is computed from a Hill-type antagonist
            # pair and Euler-integrated against joint inertia (plus joint
            # damping) to produce the next angular velocity and angle.
            new_velocities = np.zeros_like(current_velocities)
            new_angles = np.zeros_like(angles)
            for j in range(self.num_dof):
                ext, flex = self._muscles[j]
                a_ext = float(action[2 * j])
                a_flex = float(action[2 * j + 1])
                omega_j = float(current_velocities[j])
                theta_j = float(angles[j])

                ma = self.muscle_moment_arm
                # Both muscles operate near optimal length; velocity in
                # muscle frame is shortening = positive. As the joint
                # extends (theta_j increasing), the extensor shortens
                # (negative velocity in muscle frame, concentric weak)
                # and the flexor lengthens (positive velocity, eccentric
                # strong). The Hill model handles the f_V curve for us.
                L_opt_ext = ext.params.optimal_length
                L_opt_flex = flex.params.optimal_length
                ext_velocity = -ma * omega_j / max(L_opt_ext, 1e-9)
                flex_velocity = +ma * omega_j / max(L_opt_flex, 1e-9)

                f_ext = ext.force(a_ext, L_opt_ext, ext_velocity)
                f_flex = flex.force(a_flex, L_opt_flex, flex_velocity)

                # Extensor torque is +ve (drives theta upward), flexor is -ve.
                muscle_torque = ma * (f_ext - f_flex)
                # Add joint damping opposing motion (same as the velocity-
                # mode controller's implicit damping via velocity limits).
                damping_torque = -float(self.config.damping) * omega_j
                net_torque = muscle_torque + damping_torque

                inertia = float(self.config.inertias[j])
                if inertia <= 0.0:
                    inertia = 1e-3  # safety; should not occur with valid configs

                ang_accel = net_torque / inertia
                new_omega = omega_j + ang_accel * self.dt
                new_omega = float(np.clip(
                    new_omega,
                    -float(self.config.velocity_limits),
                    +float(self.config.velocity_limits),
                ))
                new_velocities[j] = new_omega
                new_angles[j] = theta_j + new_omega * self.dt

            new_angles = np.clip(
                new_angles,
                self.config.joint_limits_min,
                self.config.joint_limits_max,
            ).astype(np.float32)
            new_velocities = new_velocities.astype(np.float32)

        else:
            # ---------- Velocity-command actuation (legacy default) ----------
            # Convert normalized action to physical joint velocity command.
            joint_velocity_cmd = action * np.asarray(
                self.config.velocity_limits, dtype=np.float32
            )
            new_angles = angles + joint_velocity_cmd * self.dt
            new_angles = np.clip(
                new_angles,
                self.config.joint_limits_min,
                self.config.joint_limits_max,
            ).astype(np.float32)
            new_velocities = joint_velocity_cmd

        self.state = np.concatenate([new_angles, new_velocities])  # preserves float32
        self.controller.angles = new_angles

        end_effector_pos = self._get_end_effector_position(new_angles)
        signed_height_error = self._compute_signed_goal_error(end_effector_pos)
        goal_distance = self._compute_goal_distance(end_effector_pos)

        signed_orientation_error, orientation_error = self._compute_orientation_error(new_angles)
        velocity_norm = float(np.linalg.norm(new_velocities))

        alpha = self.EMA_VELOCITY_ALPHA
        self._ema_velocity_norm = alpha * velocity_norm + (1.0 - alpha) * self._ema_velocity_norm

        in_goal_region = self._is_goal_reached(goal_distance, orientation_error, velocity_norm)

        if in_goal_region:
            self.hold_counter += 1
        else:
            self.hold_counter = max(0, self.hold_counter - 5)

        total_error = 2.0 * goal_distance + orientation_error

        if goal_distance < self.best_distance:
            self.best_distance = goal_distance
        if total_error < self.best_total_error:
            self.best_total_error = total_error

        gradient = abs(total_error - self.previous_total_error) / max(self.dt, 1e-8)
        gradient_norm = float(np.clip(gradient / self.gradient_scale, 0.0, 1.0))
        self.last_gradient = gradient_norm

        progress = self.previous_total_error - total_error
        self.previous_total_error = total_error

        # ------------------------------------------------------------------
        # REWARD COMPUTATION
        # ------------------------------------------------------------------
        # The reward is a 10-term shaped reward designed to give SAC a strong
        # learning gradient on this goal-reaching task. The five continuous
        # penalty terms below are paid every step; the four shaping bonuses
        # that follow only fire under specific conditions; the terminal +150
        # bonus fires once on successful hold completion (further down).
        #
        # The full theoretical breakdown of every term is in
        # docs/Reward_System_Report.pdf; the labels P1..P5 / B1..B4 below
        # match the labels used in that document.
        # ------------------------------------------------------------------
        action_arr = np.asarray(action, dtype=np.float32)

        cocontraction = 0.0
        if self.actuation_mode == self.ACTUATION_MUSCLE:
            for j in range(self.num_dof):
                a_ext = float(action_arr[2 * j])
                a_flex = float(action_arr[2 * j + 1])
                cocontraction += a_ext * a_flex
            cocontraction /= max(self.num_dof, 1)

        reward = (
            -2.0  * goal_distance                       # P1 distance penalty (primary signal)
            -1.0  * orientation_error                   # P2 orientation-error penalty
            -0.15 * velocity_norm                       # P3 velocity-norm penalty (smoothness)
            -0.20 * gradient_norm                       # P4 error-gradient penalty (anti-overshoot)
            -0.01 * float(np.linalg.norm(action))       # P5 action-norm penalty (small effort cost)
            -0.08 * cocontraction                       # P7 co-contraction penalty (reduce muscle tremor)
        )

        # B1 — Progress bonus. Standard potential-based reward shaping (Ng,
        # Harada & Russell, 1999): rewarding step-to-step improvement leaves
        # the optimal policy invariant but accelerates convergence.
        if progress > 0:
            reward += 1.5 * progress

        # B2 — Proximity bonus. Smoothly ramps up as the arm enters a three-
        # times-tolerance radius around the goal. Makes the reward landscape
        # locally convex near the goal so the policy can find the last few
        # centimetres of approach without depending on a stochastic visit to
        # the sparse success region.
        proximity_threshold = 3.0 * self.height_tolerance
        if goal_distance < proximity_threshold:
            proximity_bonus = 4.0 * (1.0 - goal_distance / proximity_threshold)
            reward += proximity_bonus

        # B3 + B4 — In-goal constant and hold-growth bonuses. These fire
        # together every step the arm satisfies the full in-goal criterion
        # (position + orientation + velocity all within tolerance). The
        # constant B3 pulls the policy to stay in the goal region; the
        # linearly-growing B4 makes it strictly better to remain than to
        # leave and re-enter, encouraging the policy to settle and hold
        # rather than dither at the boundary.
        if in_goal_region:
            reward += 10.0                              # B3 in-goal constant
            reward += 2.0 * float(self.hold_counter)    # B4 hold-growth bonus

        # --- Waypoint sequence handling ---
        # In waypoint mode, intermediate waypoints use a touch-and-go criterion:
        # entering the position tolerance radius is enough to advance to the
        # next waypoint. Orientation and velocity criteria apply only to the
        # final waypoint, via the standard hold mechanism. On a touch, award
        # a transition bonus and update goal-dependent state to reflect the
        # new active waypoint so the returned observation already targets it.
        waypoint_advanced = False
        if (
            self.goal_direction == "WAYPOINTS"
            and self.waypoints is not None
            and self.current_waypoint_index < len(self.waypoints) - 1
            and goal_distance < self.height_tolerance
        ):
            self.current_waypoint_index += 1
            self._apply_waypoint(self.current_waypoint_index)
            self.hold_counter = 0
            reward += 25.0                              # waypoint transition bonus
            waypoint_advanced = True

        if waypoint_advanced:
            # Recompute goal-dependent quantities against the new waypoint so
            # the returned observation and info reflect the active goal rather
            # than the one the agent just visited.
            signed_height_error = self._compute_signed_goal_error(end_effector_pos)
            goal_distance = self._compute_goal_distance(end_effector_pos)
            signed_orientation_error, orientation_error = self._compute_orientation_error(new_angles)
            in_goal_region = self._is_goal_reached(goal_distance, orientation_error, velocity_norm)
            total_error = 2.0 * goal_distance + orientation_error
            # Avoid a spurious progress jump on the transition step.
            self.previous_total_error = total_error

        # --- Termination ---
        # In waypoint mode the success bonus and termination only fire on the
        # final waypoint's hold criterion. Intermediate waypoints contribute
        # the transition bonus above and do not terminate the episode.
        if self.goal_direction == "WAYPOINTS" and self.waypoints is not None:
            is_final_waypoint = (
                self.current_waypoint_index == len(self.waypoints) - 1
            )
            terminated = (
                is_final_waypoint
                and self.hold_counter >= self.hold_steps_required
            )
        else:
            terminated = self.hold_counter >= self.hold_steps_required

        # Terminal +150 success bonus. Large relative to the per-step
        # penalties (which rarely exceed about |-5| at peak goal distance),
        # so successful hold completion is the dominant trajectory-level
        # signal. Fires exactly once per episode, on the step that triggers
        # termination.
        if terminated:
            reward += 150.0

        truncated = self.step_count >= self.max_episode_steps

        hold_progress = min(1.0, self.hold_counter / float(self.hold_steps_required))

        info = {
            "goal_distance": float(goal_distance),
            "height_error": float(signed_height_error),
            "orientation_error": float(signed_orientation_error),
            "orientation_error_abs": float(orientation_error),
            "gradient": float(gradient),
            "gradient_norm": float(gradient_norm),
            "total_error": float(total_error),
            "hold_counter": int(self.hold_counter),
            "hold_steps_required": int(self.hold_steps_required),
            "hold_progress": float(hold_progress),
            "in_goal_region": bool(in_goal_region),
            "end_effector_position": end_effector_pos.copy(),
            "joint_angles": new_angles.copy(),
            "joint_velocities": new_velocities.copy(),
            "shoulder_position": self.shoulder_base_position.copy(),
            "goal_height": self.goal_height,
            "goal_position": self.goal_position.copy(),
            "goal_direction": self.goal_direction,
            "target_orientation": self.target_orientation,
            "goal_reached": bool(terminated),
            "step": self.step_count,
            "best_distance": float(self.best_distance),
            "waypoints": (
                [w.tolist() for w in self.waypoints]
                if self.waypoints is not None else None
            ),
            "current_waypoint_index": int(self.current_waypoint_index),
            "num_waypoints": (
                len(self.waypoints) if self.waypoints is not None else 0
            ),
            "waypoint_advanced": bool(waypoint_advanced),
            "actuation_mode": self.actuation_mode,
            "ema_velocity_norm": float(self._ema_velocity_norm),
            "cocontraction": float(cocontraction),
        }

        obs = self._get_observation(
            angles=new_angles,
            velocities=new_velocities,
            signed_height_error=signed_height_error,
            signed_orientation_error=signed_orientation_error,
            gradient_norm=gradient_norm,
            in_goal_region=in_goal_region,
        )

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        """Render environment information to console."""
        if self.render_mode == "human" and self.state is not None:
            angles = self.state[: self.num_dof]
            end_effector = self._get_end_effector_position(angles)
            goal_distance = self._compute_goal_distance(end_effector)
            _, orientation_error = self._compute_orientation_error(angles)
            print(
                f"Step {self.step_count:3d} | "
                f"Angles: [{angles[0]:6.3f}, {angles[1]:6.3f}] | "
                f"EE: [{end_effector[0]:6.3f}, {end_effector[1]:6.3f}] | "
                f"GoalDist: {goal_distance:6.3f} | "
                f"Orient Err: {orientation_error:6.3f} | "
                f"GoalDir: {self.goal_direction:>6s} | "
                f"Hold: {self.hold_counter}/{self.hold_steps_required}"
            )

    def close(self):
        """Close the environment."""
        pass

    def get_state_info(self) -> Dict[str, Any]:
        """Get detailed information about current environment state."""
        if self.state is None:
            raise RuntimeError("Environment state not initialized. Call reset() first.")

        angles = self.state[: self.num_dof]
        velocities = self.state[self.num_dof :]
        end_effector = self._get_end_effector_position(angles)
        signed_height_error = self._compute_signed_goal_error(end_effector)
        signed_orientation_error, orientation_error_abs = self._compute_orientation_error(angles)
        velocity_norm = float(np.linalg.norm(velocities))
        goal_distance = self._compute_goal_distance(end_effector)
        in_goal_region = self._is_goal_reached(goal_distance, orientation_error_abs, velocity_norm)

        return {
            "joint_angles": angles.copy(),
            "joint_velocities": velocities.copy(),
            "end_effector_position": end_effector.copy(),
            "shoulder_position": self.shoulder_base_position.copy(),
            "workspace_origin": self.workspace_origin.copy(),
            "goal_height": self.goal_height,
            "goal_position": self.goal_position.copy(),
            "goal_direction": self.goal_direction,
            "target_orientation": self.target_orientation,
            "distance_to_goal": float(goal_distance),
            "height_error": float(signed_height_error),
            "orientation_error": float(signed_orientation_error),
            "orientation_error_abs": float(orientation_error_abs),
            "gradient_norm": float(self.last_gradient),
            "hold_counter": int(self.hold_counter),
            "hold_steps_required": int(self.hold_steps_required),
            "hold_progress": float(min(1.0, self.hold_counter / float(self.hold_steps_required))),
            "in_goal_region": bool(in_goal_region),
            "goal_reached": bool(self.hold_counter >= self.hold_steps_required),
            "step": self.step_count,
            "max_steps": self.max_episode_steps,
            "waypoints": (
                [w.tolist() for w in self.waypoints]
                if self.waypoints is not None else None
            ),
            "current_waypoint_index": int(self.current_waypoint_index),
            "num_waypoints": (
                len(self.waypoints) if self.waypoints is not None else 0
            ),
            "actuation_mode": self.actuation_mode,
        }


__all__ = ["ArmTaskEnv"]
