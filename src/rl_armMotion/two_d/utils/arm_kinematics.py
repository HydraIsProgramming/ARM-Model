"""Arm kinematics, control, and motion recording utilities"""

import numpy as np
import json
from typing import Dict, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from rl_armMotion.two_d.config import ArmConfiguration


@dataclass
class ArmState:
    """Current state of the arm"""

    angles: np.ndarray  # Joint angles (radians)
    velocities: np.ndarray  # Joint velocities (rad/s)
    positions: np.ndarray  # Link positions (x, y, z) for each joint + base
    timestamp: float = 0.0

    def to_dict(self) -> Dict:
        """Convert state to dictionary for serialization"""
        return {
            "angles": self.angles.tolist(),
            "velocities": self.velocities.tolist(),
            "positions": self.positions.tolist(),
            "timestamp": float(self.timestamp),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ArmState":
        """Create state from dictionary"""
        return cls(
            angles=np.array(data["angles"]),
            velocities=np.array(data["velocities"]),
            positions=np.array(data["positions"]),
            timestamp=data.get("timestamp", 0.0),
        )


class ArmKinematics:
    """Forward kinematics for robotic arm"""

    @staticmethod
    def forward_kinematics(
        angles: np.ndarray, config: ArmConfiguration
    ) -> np.ndarray:
        """
        Compute forward kinematics (simplified 2D serial chain).

        Args:
            angles: Joint angles in radians (shape: dof,)
            config: Arm configuration with link lengths

        Returns:
            Link positions (shape: dof+1, 3) with base at origin
        """
        dof = len(angles)
        positions = np.zeros((dof + 1, 3))

        # Base position at origin
        positions[0] = [0, 0, 0]

        # 2D chain kinematics - accumulate positions step by step
        x = 0
        y = 0
        cumulative_angle = 0
        for i in range(dof):
            cumulative_angle += angles[i]
            # Each link contributes its length in its own direction
            x += config.link_lengths[i] * np.cos(cumulative_angle)
            y += config.link_lengths[i] * np.sin(cumulative_angle)
            positions[i + 1] = [x, y, 0]

        # A left arm is the same sagittal-plane chain seen from the other
        # side: mirror x. Dynamics are side-independent.
        positions[:, 0] *= float(getattr(config, "mirror_sign", 1.0))
        return positions

    @staticmethod
    def compute_link_positions(
        angles: np.ndarray, config: ArmConfiguration
    ) -> List[Tuple[float, float, float]]:
        """
        Compute all link positions for visualization.

        Args:
            angles: Joint angles
            config: Arm configuration

        Returns:
            List of (x, y, z) positions for each link + base
        """
        positions = ArmKinematics.forward_kinematics(angles, config)
        return [tuple(pos) for pos in positions]

    @staticmethod
    def end_effector_position(
        angles: np.ndarray, config: ArmConfiguration
    ) -> Tuple[float, float, float]:
        """Get end-effector position"""
        positions = ArmKinematics.forward_kinematics(angles, config)
        return tuple(positions[-1])


class ArmController:
    """Real-time arm state controller"""

    def __init__(self, config: ArmConfiguration):
        """
        Initialize arm controller.

        Args:
            config: Arm configuration
        """
        self.config = config
        self.dof = config.dof

        # Current state
        self.angles = np.array(self.config.initial_angles, dtype=float)
        self.velocities = np.zeros(self.dof)
        self.positions = self._compute_positions()

        # Control parameters
        self.max_delta = 0.05  # Max angular change per step (radians)
        self.velocity_ramp_rate = 0.1  # Velocity ramping factor

    def _compute_positions(self) -> np.ndarray:
        """Compute current link positions from angles"""
        return ArmKinematics.forward_kinematics(self.angles, self.config)

    def update_joint_angle(self, joint_id: int, target_angle: float) -> None:
        """
        Set joint to target angle (with limits).

        Args:
            joint_id: Joint index (0 to dof-1)
            target_angle: Target angle in radians
        """
        if not 0 <= joint_id < self.dof:
            raise IndexError(f"Joint {joint_id} out of range [0, {self.dof-1}]")

        # Apply joint limits
        min_limit = self.config.joint_limits_min[joint_id]
        max_limit = self.config.joint_limits_max[joint_id]
        clamped_angle = np.clip(target_angle, min_limit, max_limit)

        # Update angle and recompute positions
        self.angles[joint_id] = clamped_angle
        self.positions = self._compute_positions()

    def increment_joint(self, joint_id: int, delta: float) -> None:
        """
        Increment joint by delta (smooth motion).

        Args:
            joint_id: Joint index
            delta: Angular increment in radians
        """
        if not 0 <= joint_id < self.dof:
            raise IndexError(f"Joint {joint_id} out of range [0, {self.dof-1}]")

        # Calculate new angle with ramp
        new_angle = self.angles[joint_id] + delta

        # Apply velocity limits
        max_vel_delta = self.config.velocity_limits * self.config.dt
        delta_clamped = np.clip(delta, -max_vel_delta, max_vel_delta)

        # Apply joint limits
        min_limit = self.config.joint_limits_min[joint_id]
        max_limit = self.config.joint_limits_max[joint_id]
        new_angle = np.clip(new_angle, min_limit, max_limit)

        self.angles[joint_id] = new_angle
        self.velocities[joint_id] = delta_clamped / self.config.dt
        self.positions = self._compute_positions()

    def set_home_position(self) -> None:
        """Reset all joints to home (initial configured) position"""
        self.angles = np.array(self.config.initial_angles, dtype=float)
        self.velocities = np.zeros(self.dof)
        self.positions = self._compute_positions()

    def get_state(self, timestamp: float = 0.0) -> ArmState:
        """
        Get current arm state.

        Args:
            timestamp: Optional timestamp

        Returns:
            ArmState object
        """
        return ArmState(
            angles=self.angles.copy(),
            velocities=self.velocities.copy(),
            positions=self.positions.copy(),
            timestamp=timestamp,
        )

    def apply_state(self, state: ArmState) -> None:
        """Apply state (restore from recording or other source)"""
        self.angles = state.angles.copy()
        self.velocities = state.velocities.copy()
        self.positions = state.positions.copy()

    def get_end_effector_position(self) -> Tuple[float, float, float]:
        """Get current end-effector position"""
        return tuple(self.positions[-1])


class MotionRecorder:
    """Record and playback motion sequences"""

    def __init__(self):
        """Initialize motion recorder"""
        self.frames: List[ArmState] = []
        self.is_recording = False

    def start_recording(self) -> None:
        """Start recording motion"""
        self.frames = []
        self.is_recording = True

    def stop_recording(self) -> None:
        """Stop recording motion"""
        self.is_recording = False

    def record_frame(self, state: ArmState) -> None:
        """Record a frame of motion"""
        if self.is_recording:
            self.frames.append(state)

    def clear_frames(self) -> None:
        """Clear all recorded frames"""
        self.frames = []

    def get_num_frames(self) -> int:
        """Get number of recorded frames"""
        return len(self.frames)

    def get_frames(self) -> List[ArmState]:
        """Get all recorded frames"""
        return self.frames.copy()

    def playback(self, speed_multiplier: float = 1.0) -> List[ArmState]:
        """
        Get frames for playback with speed control.

        Args:
            speed_multiplier: Playback speed (1.0 = normal, 2.0 = 2x speed)

        Returns:
            List of frames (potentially interpolated for speed)
        """
        if speed_multiplier == 1.0:
            return self.frames.copy()

        # TODO: Implement frame interpolation for smooth speed changes
        # For now, just return frames as-is
        return self.frames.copy()

    def save_to_json(self, filepath: str) -> None:
        """
        Save recorded motion to JSON file.

        Args:
            filepath: Path to save file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "num_frames": len(self.frames),
            "frames": [frame.to_dict() for frame in self.frames],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[OK] Recorded {len(self.frames)} frames to {filepath}")

    @classmethod
    def load_from_json(cls, filepath: str) -> "MotionRecorder":
        """
        Load recorded motion from JSON file.

        Args:
            filepath: Path to load from

        Returns:
            MotionRecorder with loaded frames
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        recorder = cls()
        recorder.frames = [ArmState.from_dict(frame_data) for frame_data in data["frames"]]

        print(f"[OK] Loaded {len(recorder.frames)} frames from {filepath}")
        return recorder


__all__ = ["ArmState", "ArmKinematics", "ArmController", "MotionRecorder"]
