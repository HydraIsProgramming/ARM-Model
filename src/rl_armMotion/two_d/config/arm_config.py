"""Configuration system for robotic arm properties and parameters"""

import json
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple
from pathlib import Path


@dataclass
class ArmConfiguration:
    """Configuration for robotic arm physical and kinematic properties"""

    # Arm topology
    dof: int = 2  # Degrees of freedom (2-DOF: shoulder + elbow)
    name: str = "2DOF_Standard"

    # Initial joint angles for home position (in radians)
    initial_angles: List[float] = field(default_factory=lambda: [-np.pi/2, 0])  # Shoulder: -90° (down), Elbow: 0° (neutral)

    # Physical properties per joint (lists of length dof)
    link_lengths: List[float] = field(default_factory=lambda: [1.0, 0.8])
    masses: List[float] = field(default_factory=lambda: [2.0, 1.5])
    inertias: List[float] = field(default_factory=lambda: [0.1, 0.08])

    # Global damping
    damping: float = 0.1

    # Joint limits (per joint, in radians)
    joint_limits_min: List[float] = field(
        default_factory=lambda: [-np.pi, -np.pi/2]  # Shoulder: -180°, Elbow: -90° (relative)
    )
    joint_limits_max: List[float] = field(
        default_factory=lambda: [np.pi, np.pi/2]  # Shoulder: +180°, Elbow: +90° (relative)
    )

    # Dynamics
    dt: float = 0.01  # Time step
    velocity_limits: float = 2.0

    def __post_init__(self):
        """Validate configuration after initialization"""
        if len(self.initial_angles) != self.dof:
            self.initial_angles = self.initial_angles[:self.dof] or [-np.pi/2] * self.dof
        if len(self.link_lengths) != self.dof:
            self.link_lengths = self.link_lengths[:self.dof] or [1.0] * self.dof
        if len(self.masses) != self.dof:
            self.masses = self.masses[:self.dof] or [1.0] * self.dof
        if len(self.inertias) != self.dof:
            self.inertias = self.inertias[:self.dof] or [0.1] * self.dof
        if len(self.joint_limits_min) != self.dof:
            self.joint_limits_min = (
                self.joint_limits_min[:self.dof] or [-2.96] * self.dof
            )
        if len(self.joint_limits_max) != self.dof:
            self.joint_limits_max = (
                self.joint_limits_max[:self.dof] or [2.96] * self.dof
            )

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ArmConfiguration":
        """Create configuration from dictionary"""
        return cls(**data)

    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✓ Configuration saved to {filepath}")

    @classmethod
    def from_json(cls, filepath: str) -> "ArmConfiguration":
        """Load configuration from JSON file"""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def get_preset(cls, preset_name: str) -> "ArmConfiguration":
        """Get predefined configuration preset

        Args:
            preset_name: Name of preset ('7dof_industrial', 'simple_planar', etc.)

        Returns:
            ArmConfiguration instance
        """
        presets = {
            "2dof_simple": cls(
                dof=2,
                name="2DOF_Simple_Arm",
                link_lengths=[1.0, 0.8],  # Upper arm, forearm
                masses=[2.0, 1.5],        # Upper arm, forearm
                inertias=[0.1, 0.08],     # Upper arm, forearm
                damping=0.1,
                initial_angles=[-np.pi/2, 0],  # Shoulder: -90° (pointing down), Elbow: 0° (neutral)
                joint_limits_min=[-np.pi, 0],  # Shoulder: full rotation, Elbow: 0° (no backward bend)
                joint_limits_max=[np.pi, 2.094],  # Shoulder: full rotation, Elbow: 120° (forward only)
                velocity_limits=2.0,
            ),
            "7dof_industrial": cls(
                dof=7,
                name="7DOF_Industrial",
                link_lengths=[0.4, 0.4, 0.4, 0.3, 0.3, 0.2, 0.1],
                masses=[50, 40, 30, 20, 15, 10, 5],
                inertias=[5.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.2],
                damping=0.1,
            ),
            "simple_planar": cls(
                dof=3,
                name="3DOF_Planar",
                link_lengths=[1.0, 0.8, 0.6],
                masses=[2.0, 1.5, 1.0],
                inertias=[0.2, 0.15, 0.1],
                damping=0.05,
                joint_limits_min=[-3.14, -3.14, -3.14],
                joint_limits_max=[3.14, 3.14, 3.14],
            ),
            "light_arm": cls(
                dof=7,
                name="7DOF_Light",
                link_lengths=[0.5] * 7,
                masses=[1.0] * 7,
                inertias=[0.1] * 7,
                damping=0.05,
            ),
            "heavy_arm": cls(
                dof=7,
                name="7DOF_Heavy",
                link_lengths=[0.3] * 7,
                masses=[10.0] * 7,
                inertias=[1.0] * 7,
                damping=0.2,
            ),
            # ----------------------------------------------------------
            # 7-DOF biomechanically-proportioned upper-extremity arm,
            # matching the kinematic structure used by Fischer, Hoinville,
            # Eickhoff, and Lilienthal (2021), Sci. Rep. 11:14445, in
            # their MuJoCo musculoskeletal model. The seven joints are:
            #   0: shoulder flex/extension          (sagittal plane)
            #   1: shoulder abduction/adduction     (frontal plane)
            #   2: shoulder internal/external rot   (humeral long axis)
            #   3: elbow flex/extension
            #   4: forearm pronation/supination
            #   5: wrist flex/extension
            #   6: wrist radial/ulnar deviation
            # Segment lengths follow the standard cadaveric proportions
            # of Winter (2009), "Biomechanics and Motor Control of Human
            # Movement", 4th ed.: humerus 30 cm, ulna+radius 27 cm, and
            # hand+wrist 18 cm. Masses are anthropometric estimates for
            # an average adult upper extremity.
            #
            # Note: this configuration carries the geometric and inertial
            # parameters of Fischer's arm. Driving it through a true
            # musculoskeletal simulation (with anatomically correct moment
            # arms and Hill-type muscles per joint) requires the muscle
            # model in rl_armMotion.two_d.utils.muscle_model and additional
            # work on the environment side. As of this commit, this preset
            # is data-ready but not yet wired into the 2-DOF training env.
            # ----------------------------------------------------------
            "7dof_fischer": cls(
                dof=7,
                name="7DOF_Fischer_Upper_Extremity",
                link_lengths=[
                    0.30,   # shoulder flex/ext   (humerus length)
                    0.00,   # shoulder abd/add    (zero — shares axis w/ flex)
                    0.00,   # shoulder int/ext rot(zero — purely rotational)
                    0.27,   # elbow flex/ext      (forearm length)
                    0.00,   # forearm prono/sup   (zero — purely rotational)
                    0.18,   # wrist flex/ext      (hand length)
                    0.00,   # wrist rad/uln dev   (zero — purely rotational)
                ],
                masses=[2.10, 0.10, 0.10, 1.20, 0.10, 0.40, 0.10],
                inertias=[0.0210, 0.0010, 0.0008, 0.0080, 0.0006,
                          0.0015, 0.0005],
                damping=0.05,
                # Joint limits per Fischer 2021 / typical anatomy (radians).
                joint_limits_min=[
                    np.deg2rad(-60.0),    # shoulder flex (-60)
                    np.deg2rad(-45.0),    # shoulder abd  (-45)
                    np.deg2rad(-90.0),    # shoulder rot  (-90)
                    np.deg2rad(0.0),      # elbow         (0)
                    np.deg2rad(-90.0),    # forearm pro   (-90)
                    np.deg2rad(-70.0),    # wrist flex    (-70)
                    np.deg2rad(-30.0),    # wrist dev     (-30)
                ],
                joint_limits_max=[
                    np.deg2rad(180.0),    # shoulder flex (+180)
                    np.deg2rad(180.0),    # shoulder abd  (+180)
                    np.deg2rad(90.0),     # shoulder rot  (+90)
                    np.deg2rad(150.0),    # elbow         (+150)
                    np.deg2rad(90.0),     # forearm pro   (+90)
                    np.deg2rad(70.0),     # wrist flex    (+70)
                    np.deg2rad(30.0),     # wrist dev     (+30)
                ],
                velocity_limits=2.0,
            ),
            "default": cls(),
        }

        if preset_name not in presets:
            print(
                f"⚠ Unknown preset '{preset_name}', returning default. "
                f"Available: {list(presets.keys())}"
            )
            return presets["default"]

        return presets[preset_name]

    @staticmethod
    def list_presets() -> List[str]:
        """Get list of available preset names"""
        return [
            "2dof_simple",
            "7dof_industrial",
            "7dof_fischer",
            "simple_planar",
            "light_arm",
            "heavy_arm",
            "default",
        ]

    def get_joint_limits(self) -> np.ndarray:
        """Get joint limits as (dof, 2) array"""
        return np.column_stack([self.joint_limits_min, self.joint_limits_max])

    def validate(self) -> Tuple[bool, str]:
        """Validate configuration integrity

        Returns:
            (bool, str): (is_valid, error_message)
        """
        if not self.dof > 0:
            return False, "DOF must be positive"

        if len(self.link_lengths) != self.dof:
            return False, f"link_lengths length {len(self.link_lengths)} != DOF {self.dof}"

        if len(self.masses) != self.dof:
            return False, f"masses length {len(self.masses)} != DOF {self.dof}"

        if len(self.inertias) != self.dof:
            return False, f"inertias length {len(self.inertias)} != DOF {self.dof}"

        if any(m <= 0 for m in self.masses):
            return False, "All masses must be positive"

        if any(i <= 0 for i in self.inertias):
            return False, "All inertias must be positive"

        if any(l <= 0 for l in self.link_lengths):
            return False, "All link lengths must be positive"

        if self.damping < 0:
            return False, "Damping must be non-negative"

        if self.dt <= 0:
            return False, "Time step (dt) must be positive"

        if self.velocity_limits <= 0:
            return False, "Velocity limits must be positive"

        return True, "Configuration is valid"

    def __repr__(self) -> str:
        """String representation of configuration"""
        return (
            f"ArmConfiguration("
            f"name='{self.name}', dof={self.dof}, "
            f"link_lengths={self.link_lengths[:3]}..., "
            f"masses={self.masses[:3]}..., "
            f"damping={self.damping})"
        )


__all__ = ["ArmConfiguration"]
