"""Task-specific environment for 2-DOF robotic arm with workspace setup.

This environment defines a task where:
- Shoulder joint is fixed at position [1.0, 0] meters from workspace origin
- Initial configuration: Arm pointing vertically downward (shoulder -90°, elbow 0°)
- Goal configuration can be:
  - legacy height goal (reach shoulder height + orientation hold)
  - directional far-point goal (EAST, WEST, NORTH)
- Workspace: 2D plane with origin at [0, 0] and shoulder base at [1.0, 0]
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_armMotion.two_d.config import ArmConfiguration
from rl_armMotion.two_d.utils.arm_kinematics import ArmController, ArmKinematics


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

    def __init__(
        self,
        render_mode: Optional[str] = None,
        shoulder_base_position: Optional[np.ndarray] = None,
        use_2dof: bool = True,
        goal_direction: str = "HEIGHT",
    ):
        self.render_mode = render_mode
        self.use_2dof = use_2dof
        self.goal_direction = str(goal_direction).strip().upper()
        if self.goal_direction not in {"HEIGHT", "EAST", "WEST", "NORTH"}:
            self.goal_direction = "HEIGHT"

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

        # Action space: normalized velocity command for each joint
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
        self.max_episode_steps = 800
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

        # Tolerances and hold constraints
        self.height_tolerance = 0.03
        self.orientation_tolerance = float(np.deg2rad(5.0))
        self.hold_velocity_tolerance = 0.15
        self.hold_steps_required = 60
        self.gradient_scale = 5.0

        # Compatibility alias with previous code/tests
        self.goal_tolerance = self.height_tolerance

        # Track performance and hold state
        self.best_distance = float("inf")
        self.best_total_error = float("inf")
        self.previous_total_error = float("inf")
        self.hold_counter = 0
        self.last_gradient = 0.0

        # ── Physics constants ──────────────────────────────────────────────────
        # Standard gravitational acceleration (NIST, 2018 CODATA value).
        self.gravity: float = 9.81  # m/s²

        # Precomputed physical arrays in float64 for torque/energy precision.
        self._link_lengths = np.asarray(self.config.link_lengths, dtype=np.float64)
        self._masses       = np.asarray(self.config.masses,       dtype=np.float64)

        # ── A: Work done against gravity ───────────────────────────────────────
        # Physical quantity: change in gravitational potential energy per step.
        #   ΔPE = PE_new − PE_prev   where   PE = Σ m_k · g · y_com_k
        #
        # Scientific basis (Work-energy theorem):
        #   When ΔPE > 0 the arm has lifted mass against gravity and the
        #   actuators must supply that energy.  ΔPE ≤ 0 means the arm
        #   descended; gravity did the work — no actuator energy cost.
        #
        # Key design property: penalty is ZERO when the arm is stationary
        # at ANY pose, including the goal (horizontal), so it never
        # discourages the agent from holding the target configuration.
        # This contrasts with penalising Σ|τ_i|, which is MAXIMUM at the
        # horizontal goal and would actively work against task completion.
        #
        # Literature support:
        #   - Goldstein, Poole & Safko, "Classical Mechanics" (3rd ed.),
        #     §1.4 — gravitational PE of a system of particles.
        #   - Zhang & Gao (2018), "Kuka Youbot Arm Path Planning Based on
        #     Gravity," MEEES-18, Atlantis Press, pp. 426-429 — uses mgh
        #     (gravitational PE) as a trajectory planning objective.
        #   - "Optimization of energy consumption in industrial robots,
        #     a review," ScienceDirect 2023 — notes PE-based cost functions
        #     are valid secondary shaping terms (not sole energy proxies).
        #   Note: ΔPE is used here as a secondary shaping term alongside
        #   the mechanical energy budget (J) below, consistent with the
        #   literature recommendation.
        #
        # Coefficient sizing: full vertical→horizontal swing ΔPE ≈ 30 J
        # over ~150 steps → ~0.20 J/step peak.  Coefficient 0.10 →
        # max reward impact ≈ 0.02/step (< 1% of primary distance reward).
        self.gravity_work_coeff: float = 0.10
        self.prev_gravitational_pe: float = 0.0   # initialised properly in reset()

        # ── C: Joint acceleration limit ────────────────────────────────────────
        # Physical basis: finite motor torque → finite joint acceleration.
        #
        # Real robot reference data (from published specs and literature):
        #   • UR5 (5 kg payload):       ~5.24 rad/s²  (300 deg/s²)
        #     Source: Universal Robots UR5 Technical Spec Sheet (Item 110105).
        #   • KUKA LBR iiwa 14 R820 (14 kg payload):  2.0–5.0 rad/s² per joint
        #     Source: ETA-IK, arXiv:2411.14381 (Fraunhofer, 2024).
        #
        # Our arm: link masses [2.0, 1.5] kg — significantly lighter than both
        # reference robots.  Shoulder inertia estimate (uniform-rod + distal mass):
        #   I ≈ m₁·l₁²/3 + m₂·l₁² = 2.0·1.0²/3 + 1.5·1.0² ≈ 2.17 kg·m²
        # Physical max with 100 Nm motor: α_max = τ/I ≈ 100/2.17 ≈ 46 rad/s².
        # We use 8.0 rad/s² (~1/6 of physical max), which is above the iiwa's
        # 5 rad/s² but consistent with its lighter mass and a moderately fast
        # collaborative operating mode.
        #
        # ISO/TS 15066:2016 governs collaborative robot contact force/pressure
        # limits rather than acceleration directly; the 8.0 rad/s² bound is
        # therefore application-appropriate for this arm class.
        self.max_joint_accel: float = 8.0   # rad/s²
        self.max_delta_vel:   float = self.max_joint_accel * self.dt  # rad/s per step

        # Penalty: normalised mean per-joint acceleration effort in [0, 1].
        # 1.0 = every joint saturating its acceleration limit this step.
        self.accel_penalty_coeff: float = 0.05
        self.prev_velocities: np.ndarray = np.zeros(self.num_dof, dtype=np.float32)

        # ── J: Mechanical energy budget ────────────────────────────────────────
        # Physical quantity: instantaneous mechanical power × dt per step (J).
        #   E_step = Σ |τ_gravity_i| · |ω_i| · dt
        #
        # Validation:
        #   Petrichenko et al. (2024), "Energy Consumption in Robotics: A
        #   Simplified Modeling Approach," arXiv:2411.03194 (Fraunhofer IPK):
        #   validates P = τᵀ·q̇ to within 3.5–4% of measured electrical power
        #   on a Franka Emika Panda robot.
        #
        # Absolute-value justification (non-regenerative actuator assumption):
        #   Peri et al. (2025), "Non-conflicting Energy Minimization in RL-
        #   based Robot Control," arXiv:2509.01765: explicitly discusses taking
        #   |τ·ω| to prevent negative power credits in actuators that cannot
        #   recover braking energy (standard DC servo drives dissipate it as
        #   heat).  Validated as the correct assumption for simulation.
        #
        # RL precedent:
        #   Zhang et al. (2023), "Multi-Objective Optimal Trajectory Planning
        #   for Robotic Arms Using Deep RL," Sensors 23(13):5974, DOI:
        #   10.3390/s23135974: uses r_et = −w_e·Σ(Δθ_k·τ_k)², a discrete
        #   approximation of ∫τ·ω·dt, as an energy reward term.
        #
        # Coefficient sizing: max |τ| ≈ 30 Nm, max |ω| = 2 rad/s, dt = 0.01 s
        # → max E_step ≈ 0.6 J.  Coefficient 0.01 → max impact ≈ 0.006/step.
        self.energy_penalty_coeff: float = 0.01
        self.episode_energy:       float = 0.0

        # ── K: Jerk penalty ────────────────────────────────────────────────────
        # Physical quantity: rate of change of acceleration, normalised to [0, 1].
        #   jerk_norm = |Δdelta_vel| / (2 · max_delta_vel)
        # where Δdelta_vel = delta_vel_t − delta_vel_{t−1}.
        # Normalisation: 2·max_delta_vel is the maximum possible |Δdelta_vel|
        # (full reversal from +max to −max acceleration in one step).
        # 0 = constant acceleration this step; 1 = maximum direction reversal.
        #
        # Scientific basis — minimum-jerk criterion:
        #   Flash & Hogan (1985), "The coordination of arm movements: an
        #   experimentally confirmed mathematical model," J. Neuroscience
        #   5(7):1688-1703.  Proves that human arm trajectories minimise the
        #   integral of squared jerk: C = ½∫₀ᵀ(d³x/dt³)² dt.
        #   Minimising jerk produces straight-line paths with bell-shaped
        #   velocity profiles — the hallmark of natural arm motion.
        #   Review: Todorov (2004), "Optimality principles in sensorimotor
        #   control," Nature Neuroscience 7(9):907-915.
        #
        # RL application:
        #   Kim et al. (2024), "Not Only Rewards But Also Constraints:
        #   Applications on Legged Robot Locomotion," arXiv:2308.12517:
        #   uses the second-order discrete action difference
        #   ‖q_t − 2q_{t-1} + q_{t-2}‖² as a jerk-equivalent smoothness
        #   term, citing prevention of motor vibration in real deployment.
        self.jerk_penalty_coeff: float = 0.02
        self.prev_delta_vel: np.ndarray = np.zeros(self.num_dof, dtype=np.float32)

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

    def _compute_gravity_torques(self, angles: np.ndarray) -> np.ndarray:
        """Compute gravity torques (Nm) at each joint for a planar 2D serial arm.

        Uses standard recursive formula for a uniform-rod serial chain:
            τ_i = -g * Σ_{k=i}^{n-1}  m_k * (x_com_k − x_joint_i)

        Sign convention: negative torque = gravity creates CW rotation (arm falls).

        Args:
            angles: Joint angles in radians, shape (num_dof,).

        Returns:
            torques: Gravity torque at each joint, shape (num_dof,), in Nm.
        """
        l = self._link_lengths          # link lengths [l0, l1]
        m = self._masses                # link masses  [m0, m1]
        n = self.num_dof

        cum_angles = np.cumsum(np.asarray(angles, dtype=np.float64))  # [θ1, θ1+θ2]
        cos_a = np.cos(cum_angles)

        # x-coordinate of each joint (shoulder at origin)
        joint_x = np.zeros(n + 1)
        joint_x[1:] = np.cumsum(l * cos_a)

        # x-coordinate of each link's center of mass (mid-point of uniform rod)
        com_x = joint_x[:n] + 0.5 * l * cos_a

        # τ_i = -g * Σ_{k≥i} m_k * (com_x_k − joint_x_i)
        torques = np.empty(n)
        for i in range(n):
            torques[i] = -self.gravity * float(np.dot(m[i:], com_x[i:] - joint_x[i]))

        return torques

    def _compute_gravitational_pe(self, angles: np.ndarray) -> float:
        """Gravitational potential energy of the arm (Joules), referenced at shoulder height.

        PE = Σ_{k=0}^{n-1}  m_k · g · y_com_k

        where y_com_k is the vertical (y) position of link k's centre of mass
        above the shoulder joint origin.

        Physical basis:
            Work-energy theorem: W = ΔPE.  When ΔPE > 0 between two consecutive
            steps the actuators have done positive work against gravity.  When
            ΔPE ≤ 0 the arm has descended and no energy cost is incurred.
            Using ΔPE rather than static torque magnitude means the penalty is
            zero when the arm is stationary (including at the goal configuration),
            correctly reflecting that holding a pose costs no energy in an ideal
            system with static friction / no gravity compensation required.

        Reference:
            Goldstein, H., Poole, C., Safko, J. (2002). "Classical Mechanics"
            (3rd ed.), §1.4 — potential energy of a system of particles.

        Args:
            angles: Joint angles in radians, shape (num_dof,).

        Returns:
            Gravitational potential energy in Joules.
        """
        l = self._link_lengths
        m = self._masses
        n = self.num_dof

        cum_angles = np.cumsum(np.asarray(angles, dtype=np.float64))
        sin_a = np.sin(cum_angles)

        # y-coordinate of each joint (shoulder at y=0)
        joint_y = np.zeros(n + 1)
        joint_y[1:] = np.cumsum(l * sin_a)

        # y-coordinate of each link's centre of mass (mid-point of uniform rod)
        com_y = joint_y[:n] + 0.5 * l * sin_a

        return float(self.gravity * np.dot(m, com_y))

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
        self.prev_velocities = np.zeros(self.num_dof, dtype=np.float32)
        self.prev_delta_vel  = np.zeros(self.num_dof, dtype=np.float32)
        self.episode_energy  = 0.0
        # Initialise gravitational PE so the first step's ΔPE is computed correctly.
        self.prev_gravitational_pe = self._compute_gravitational_pe(angles)

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

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Convert normalized action to physical joint velocity command.
        joint_velocity_cmd = action * np.asarray(self.config.velocity_limits, dtype=np.float32)

        # --- Joint acceleration limit ---
        # Velocity can change by at most max_delta_vel per step (C: jerk constraint).
        delta_vel = joint_velocity_cmd - self.prev_velocities
        delta_vel = np.clip(delta_vel, -self.max_delta_vel, self.max_delta_vel)
        actual_velocities = (self.prev_velocities + delta_vel).astype(np.float32)

        new_angles = angles + actual_velocities * self.dt
        new_angles = np.clip(
            new_angles,
            self.config.joint_limits_min,
            self.config.joint_limits_max,
        ).astype(np.float32)

        new_velocities = actual_velocities
        self.prev_velocities = new_velocities
        self.state = np.concatenate([new_angles, new_velocities])  # preserves float32
        self.controller.angles = new_angles

        end_effector_pos = self._get_end_effector_position(new_angles)
        signed_height_error = self._compute_signed_goal_error(end_effector_pos)
        goal_distance = self._compute_goal_distance(end_effector_pos)

        signed_orientation_error, orientation_error = self._compute_orientation_error(new_angles)
        velocity_norm = float(np.linalg.norm(new_velocities))

        in_goal_region = self._is_goal_reached(goal_distance, orientation_error, velocity_norm)

        if in_goal_region:
            self.hold_counter += 1
        else:
            self.hold_counter = 0

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

        # ── A: Gravity torques & work against gravity ──────────────────────────
        # Torque formula: τ_i = -g · Σ_{k≥i} m_k · (x_com_k − x_joint_i)
        # Ref: Spong, Hutchinson & Vidyasagar, "Robot Modeling and Control"
        #      (2006), §6.5 — planar gravity torque for serial chain.
        gravity_torques = self._compute_gravity_torques(new_angles)

        # Work against gravity this step = max(0, ΔPE)  [Joules].
        # ΔPE > 0: arm lifted → actuators spent energy.
        # ΔPE ≤ 0: arm descended → no energy penalty (gravity does the work).
        # This is the only physically correct formulation: it is zero when the
        # arm is stationary at ANY configuration, including the goal, so it
        # never penalises the agent for holding the target pose.
        pe_new = self._compute_gravitational_pe(new_angles)
        work_against_gravity = max(0.0, pe_new - self.prev_gravitational_pe)
        self.prev_gravitational_pe = pe_new

        # ── C: Acceleration effort (penalty for saturating acceleration limit) ─
        # Normalised mean per-joint effort in [0, 1].
        # 1.0 = every joint commanded at maximum allowed acceleration this step.
        accel_effort = float(np.mean(np.abs(delta_vel) / max(self.max_delta_vel, 1e-8)))

        # ── J: Mechanical energy budget ────────────────────────────────────────
        # E_step = Σ |τ_gravity_i| · |ω_i| · dt   [Joules]
        # Absolute-value product = non-regenerative actuator assumption:
        # both lifting AND braking cost energy (braking energy dissipated as heat).
        # Ref: Sciavicco & Siciliano, "Modelling and Control of Robot
        #      Manipulators" (2000), §8.3.
        step_energy = float(
            np.dot(np.abs(gravity_torques), np.abs(new_velocities.astype(np.float64)))
        ) * self.dt
        self.episode_energy += step_energy

        # ── K: Jerk (rate of change of acceleration) ──────────────────────────
        # Normalised to [0, 1]:
        #   0 = acceleration held constant this step  (smooth motion)
        #   1 = acceleration reversed from +max to -max in one step (max jerk)
        # Derivation: max |Δdelta_vel| = 2 · max_delta_vel (full reversal),
        # so jerk_norm = |delta_vel_t − delta_vel_{t-1}| / (2 · max_delta_vel).
        # Ref: Flash & Hogan, J. Neuroscience 5(7):1688-1703 (1985) —
        #      minimum-jerk model of human arm trajectories.
        jerk_norm = float(np.clip(
            np.mean(np.abs(delta_vel - self.prev_delta_vel))
            / (2.0 * max(self.max_delta_vel, 1e-8)),
            0.0, 1.0,
        ))
        self.prev_delta_vel = delta_vel.copy()

        # ── Reward ─────────────────────────────────────────────────────────────
        reward = (
            -2.0  * goal_distance       # primary: minimise distance to goal
            -1.0  * orientation_error   # secondary: align end-effector orientation
            -0.15 * velocity_norm       # damping: decelerate near goal
            -0.20 * gradient_norm       # stability: smooth approach trajectory
            -0.01 * float(np.linalg.norm(action))  # action regularisation
            # A: penalise work done AGAINST gravity (zero when stationary/descending)
            -self.gravity_work_coeff   * work_against_gravity
            # C: penalise saturating the acceleration limit (motor current proxy)
            -self.accel_penalty_coeff  * accel_effort
            # J: penalise total mechanical energy exchange each step
            -self.energy_penalty_coeff * step_energy
            # K: penalise jerk — encourages minimum-jerk arc trajectories
            -self.jerk_penalty_coeff   * jerk_norm
        )

        if progress > 0:
            reward += 1.5 * progress

        if in_goal_region:
            reward += 2.0 * float(self.hold_counter)

        terminated = self.hold_counter >= self.hold_steps_required
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
            # Physics metrics
            "gravity_torques":      gravity_torques.tolist(),          # Nm per joint
            "work_against_gravity": float(work_against_gravity),       # J this step (A)
            "gravitational_pe":     float(pe_new),                     # J absolute PE
            "accel_effort":         float(accel_effort),               # [0,1] (C)
            "step_energy":          float(step_energy),                # J this step (J)
            "episode_energy":       float(self.episode_energy),        # J cumulative (J)
            "jerk_norm":            float(jerk_norm),                  # [0,1] (K)
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
        }


__all__ = ["ArmTaskEnv"]
