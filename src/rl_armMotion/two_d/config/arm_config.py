"""Configuration system for robotic arm properties and parameters.

Conventions (used by utils/dynamics.py, utils/muscle_model.py and the env)
--------------------------------------------------------------------------
- Planar chain in the sagittal plane of a RIGHT arm viewed from the right:
  +x is anterior (in front of the body), +y is up, gravity acts along -y.
- Joint angle 0 is measured counter-clockwise from +x; the arm hangs
  straight down at theta0 = -90 deg. Elbow angle theta1 is the relative
  flexion angle (0 = straight, positive = flexed, hand swings forward/up).
- ``inertias`` are the segment moments of inertia ABOUT THE SEGMENT CENTRE
  OF MASS (kg m^2). ``com_offsets`` are the centre-of-mass distances from
  the proximal joint along the link (m). Older presets that predate the
  dynamics model keep their legacy numbers; only ``2dof_human_right`` and
  ``3dof_human_right`` are physically consistent.
"""

import json
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Winter (2009), "Biomechanics and Motor Control of Human Movement", 4th ed.,
# Table 4.1, for a 75 kg / 1.75 m adult. Upper arm: 0.028 M, COM 0.436 L from
# the shoulder, radius of gyration about COM 0.322 L. Forearm + hand: 0.022 M,
# COM 0.682 L_forearm from the elbow, radius of gyration 0.468 L_forearm.
# The end-effector is the hand centre (elbow -> palm = 0.35 m).
_HUMAN_MASS_KG = 75.0
_HUMAN_HEIGHT_M = 1.75
_UPPER_ARM_LEN = 0.30
_FOREARM_LEN = 0.256           # elbow axis -> ulnar styloid (wrist)
_ELBOW_TO_HAND_CENTRE = 0.35   # end-effector for the 2-DOF model
_HAND_LEN = 0.19               # wrist -> fingertip (3-DOF model)
_UPPER_ARM_MASS = 0.028 * _HUMAN_MASS_KG          # 2.10 kg
_FOREARM_HAND_MASS = 0.022 * _HUMAN_MASS_KG       # 1.65 kg
_FOREARM_MASS = 0.016 * _HUMAN_MASS_KG            # 1.20 kg
_HAND_MASS = 0.006 * _HUMAN_MASS_KG               # 0.45 kg


@dataclass
class ArmConfiguration:
    """Configuration for robotic arm physical and kinematic properties"""

    # Arm topology
    dof: int = 2  # Degrees of freedom (2-DOF: shoulder + elbow)
    name: str = "2DOF_Standard"

    # Initial joint angles for home position (in radians)
    initial_angles: List[float] = field(default_factory=lambda: [-np.pi/2, 0])  # Shoulder: -90 deg (down), Elbow: 0 deg

    # Physical properties per joint (lists of length dof)
    link_lengths: List[float] = field(default_factory=lambda: [1.0, 0.8])
    masses: List[float] = field(default_factory=lambda: [2.0, 1.5])
    inertias: List[float] = field(default_factory=lambda: [0.1, 0.08])  # about segment COM

    # Global damping (N*m*s/rad). joint_damping overrides it per joint.
    damping: float = 0.1
    joint_damping: Optional[List[float]] = None

    # Joint limits (per joint, in radians)
    joint_limits_min: List[float] = field(
        default_factory=lambda: [-np.pi, -np.pi/2]  # Shoulder: -180 deg, Elbow: -90 deg (relative)
    )
    joint_limits_max: List[float] = field(
        default_factory=lambda: [np.pi, np.pi/2]  # Shoulder: +180 deg, Elbow: +90 deg (relative)
    )

    # Dynamics
    dt: float = 0.01  # Control time step (s)
    velocity_limits: float = 2.0  # Hard safety clamp on |joint velocity| (rad/s)
    physics_substeps: int = 1     # Integrator substeps per control step
    max_acceleration: float = 2000.0  # Numerical guard on |joint acceleration| (rad/s^2)

    # Rigid-body parameters (None = derived: COM at mid-link, no gravity).
    com_offsets: Optional[List[float]] = None   # m, from the proximal joint
    gravity: float = 0.0                        # m/s^2, acts along -y
    torque_limits: Optional[List[float]] = None  # N*m per joint; None = unlimited

    # Soft end-of-range stiffness: a spring engages within ``passive_limit_margin``
    # of each joint limit so the controller feels the limit before the hard stop.
    passive_limit_stiffness: float = 0.0   # N*m/rad
    passive_limit_margin: float = 0.0      # rad, default for both sides of every joint
    # Per-joint, per-side overrides (None = use passive_limit_margin). A zero
    # entry is a hard bony stop with no soft spring (e.g. full elbow extension).
    passive_limit_margins_min: Optional[List[float]] = None
    passive_limit_margins_max: Optional[List[float]] = None

    # Muscle parameters per joint (used by actuation_mode="muscle").
    muscle_moment_arms: Optional[List[float]] = None  # m
    muscle_f_max: Optional[List[float]] = None        # N, per direction
    muscle_l_opt: Optional[List[float]] = None        # m, optimal fibre length
    muscle_v_max: float = 6.0                          # L_opt / s
    # Optional first-order excitation->activation dynamics (Zajac 1989).
    # None = instantaneous activation (moderate-realism default).
    activation_time_constants: Optional[List[float]] = None  # [tau_rise, tau_fall] s

    # Initial-state randomisation applied by the env at reset (rad, per joint).
    initial_angle_jitter_min: Optional[List[float]] = None
    initial_angle_jitter_max: Optional[List[float]] = None

    # Which arm this is. "left" mirrors the x axis in forward kinematics so the
    # same sagittal-plane dynamics draw and reach as a left arm.
    side: str = "right"

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
        if self.com_offsets is None:
            self.com_offsets = [0.5 * float(l) for l in self.link_lengths]
        if self.joint_damping is None:
            self.joint_damping = [float(self.damping)] * self.dof
        if self.passive_limit_margins_min is None:
            self.passive_limit_margins_min = [float(self.passive_limit_margin)] * self.dof
        if self.passive_limit_margins_max is None:
            self.passive_limit_margins_max = [float(self.passive_limit_margin)] * self.dof
        if self.initial_angle_jitter_min is None:
            self.initial_angle_jitter_min = [0.0] * self.dof
        if self.initial_angle_jitter_max is None:
            self.initial_angle_jitter_max = [0.0] * self.dof
        for attr in ("com_offsets", "joint_damping", "torque_limits",
                     "muscle_moment_arms", "muscle_f_max", "muscle_l_opt",
                     "passive_limit_margins_min", "passive_limit_margins_max",
                     "initial_angle_jitter_min", "initial_angle_jitter_max"):
            val = getattr(self, attr)
            if val is not None and len(val) != self.dof:
                raise ValueError(
                    f"{attr} must have {self.dof} entries, got {len(val)}"
                )
        self.side = str(self.side).lower()
        if self.side not in ("right", "left"):
            raise ValueError(f"side must be 'right' or 'left', got {self.side!r}")
        if self.physics_substeps < 1:
            raise ValueError("physics_substeps must be >= 1")

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #
    @property
    def mirror_sign(self) -> float:
        """+1 for the right arm, -1 for the left arm (x axis mirrored)."""
        return -1.0 if self.side == "left" else 1.0

    @property
    def reach(self) -> float:
        """Maximum distance from the base joint to the end-effector (m)."""
        return float(np.sum(self.link_lengths))

    @property
    def joint_mid_angles(self) -> np.ndarray:
        return 0.5 * (np.asarray(self.joint_limits_min, dtype=float)
                      + np.asarray(self.joint_limits_max, dtype=float))

    def for_side(self, side: str) -> "ArmConfiguration":
        """Return a copy of this configuration for the given arm side."""
        data = self.to_dict()
        data["side"] = str(side).lower()
        data["name"] = data["name"].replace("_Right", "").replace("_Left", "") + (
            "_Left" if data["side"] == "left" else "_Right"
        )
        return ArmConfiguration.from_dict(data)

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ArmConfiguration":
        """Create configuration from dictionary (unknown keys are ignored)."""
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[OK] Configuration saved to {filepath}")

    @classmethod
    def from_json(cls, filepath: str) -> "ArmConfiguration":
        """Load configuration from JSON file"""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ------------------------------------------------------------------ #
    # Presets
    # ------------------------------------------------------------------ #
    @classmethod
    def _human_right_2dof(cls) -> "ArmConfiguration":
        # Peak gravity torque at the horizontal pose:
        #   G1 = g*(m1*lc1 + m2*L1 + m2*lc2) = 9.81*(0.275 + 0.495 + 0.289) = 10.4 N*m
        #   G2 = g*m2*lc2                     = 9.81*1.65*0.175          =  2.8 N*m
        # Torque limits 25 / 12 N*m are 2.4x / 4.3x those peaks and well under
        # adult maxima (~60 N*m), so the arm can hold anywhere it can reach
        # without being whip-fast. Muscle F_max = tau_max / moment arm.
        upper_com = 0.436 * _UPPER_ARM_LEN                         # 0.131
        fore_com = 0.682 * _FOREARM_LEN                            # 0.175
        upper_I = _UPPER_ARM_MASS * (0.322 * _UPPER_ARM_LEN) ** 2  # 0.0196
        fore_I = _FOREARM_HAND_MASS * (0.468 * _FOREARM_LEN) ** 2  # 0.0237
        torque_limits = [25.0, 12.0]
        moment_arms = [0.04, 0.03]
        return cls(
            dof=2,
            name="2DOF_Human_Right",
            link_lengths=[_UPPER_ARM_LEN, _ELBOW_TO_HAND_CENTRE],
            masses=[round(_UPPER_ARM_MASS, 3), round(_FOREARM_HAND_MASS, 3)],
            inertias=[round(upper_I, 4), round(fore_I, 4)],
            com_offsets=[round(upper_com, 3), round(fore_com, 3)],
            damping=0.30,
            joint_damping=[0.30, 0.30],
            # Relaxed human elbow rests ~10 deg flexed (passive tissue keeps it
            # off full extension); starting straight would put the hand on
            # the reach circle, a kinematic singularity where elbow flexion
            # has no first-order effect on hand distance.
            initial_angles=[-np.pi / 2, float(np.deg2rad(10.0))],
            joint_limits_min=[float(np.deg2rad(-150.0)), 0.0],          # flexion -60 deg
            joint_limits_max=[float(np.deg2rad(90.0)), float(np.deg2rad(145.0))],  # flexion 180 deg
            velocity_limits=8.0,
            dt=0.02,
            physics_substeps=4,
            gravity=9.81,
            torque_limits=torque_limits,
            passive_limit_stiffness=20.0,
            passive_limit_margin=float(np.deg2rad(10.0)),
            muscle_moment_arms=moment_arms,
            muscle_f_max=[t / r for t, r in zip(torque_limits, moment_arms)],  # 625, 400 N
            muscle_l_opt=[0.20, 0.10],
            muscle_v_max=6.0,
            initial_angle_jitter_min=[float(np.deg2rad(-5.0)), float(np.deg2rad(-5.0))],
            initial_angle_jitter_max=[float(np.deg2rad(5.0)), float(np.deg2rad(10.0))],
            side="right",
        )

    @classmethod
    def _human_right_3dof(cls) -> "ArmConfiguration":
        """Stage-D preset: shoulder + elbow + wrist (flexion/extension)."""
        upper_com = 0.436 * _UPPER_ARM_LEN
        fore_com = 0.430 * _FOREARM_LEN                       # forearm only, Winter
        hand_com = 0.506 * _HAND_LEN
        upper_I = _UPPER_ARM_MASS * (0.322 * _UPPER_ARM_LEN) ** 2
        fore_I = _FOREARM_MASS * (0.303 * _FOREARM_LEN) ** 2
        hand_I = _HAND_MASS * (0.297 * _HAND_LEN) ** 2
        torque_limits = [25.0, 12.0, 5.0]
        moment_arms = [0.04, 0.03, 0.015]
        return cls(
            dof=3,
            name="3DOF_Human_Right",
            link_lengths=[_UPPER_ARM_LEN, _FOREARM_LEN, _HAND_LEN],
            masses=[round(_UPPER_ARM_MASS, 3), round(_FOREARM_MASS, 3), round(_HAND_MASS, 3)],
            inertias=[round(upper_I, 4), round(fore_I, 4), round(hand_I, 5)],
            com_offsets=[round(upper_com, 3), round(fore_com, 3), round(hand_com, 3)],
            damping=0.30,
            joint_damping=[0.30, 0.30, 0.10],
            initial_angles=[-np.pi / 2, float(np.deg2rad(10.0)), 0.0],
            joint_limits_min=[float(np.deg2rad(-150.0)), 0.0, float(np.deg2rad(-70.0))],
            joint_limits_max=[float(np.deg2rad(90.0)), float(np.deg2rad(145.0)),
                              float(np.deg2rad(70.0))],
            velocity_limits=8.0,
            dt=0.02,
            physics_substeps=4,
            gravity=9.81,
            torque_limits=torque_limits,
            passive_limit_stiffness=20.0,
            passive_limit_margin=float(np.deg2rad(10.0)),
            muscle_moment_arms=moment_arms,
            muscle_f_max=[t / r for t, r in zip(torque_limits, moment_arms)],
            muscle_l_opt=[0.20, 0.10, 0.05],
            muscle_v_max=6.0,
            initial_angle_jitter_min=[float(np.deg2rad(-5.0)), float(np.deg2rad(-5.0)), 0.0],
            initial_angle_jitter_max=[float(np.deg2rad(5.0)), float(np.deg2rad(10.0)), 0.0],
            side="right",
        )

    @classmethod
    def get_preset(cls, preset_name: str) -> "ArmConfiguration":
        """Get predefined configuration preset

        Args:
            preset_name: Name of preset ('2dof_human_right', '2dof_simple', ...)

        Returns:
            ArmConfiguration instance
        """
        presets = {
            # Physically consistent human-scale presets (gravity, torque
            # limits, Winter 2009 anthropometry). These are what the env uses.
            "2dof_human_right": cls._human_right_2dof,
            "2dof_human_left": lambda: cls._human_right_2dof().for_side("left"),
            "3dof_human_right": cls._human_right_3dof,
            "3dof_human_left": lambda: cls._human_right_3dof().for_side("left"),
            # Legacy gravity-free preset the project trained on before the
            # human-arm rebuild. Kept so old configs remain constructible.
            "2dof_simple": lambda: cls(
                dof=2,
                name="2DOF_Simple_Arm",
                link_lengths=[1.0, 0.8],  # Upper arm, forearm
                masses=[2.0, 1.5],        # Upper arm, forearm
                inertias=[0.1, 0.08],     # Upper arm, forearm
                damping=0.1,
                initial_angles=[-np.pi/2, 0],  # Shoulder: -90 deg (pointing down), Elbow: 0 deg
                joint_limits_min=[-np.pi, 0],  # Shoulder: full rotation, Elbow: 0 deg (no backward bend)
                joint_limits_max=[np.pi, 2.094],  # Shoulder: full rotation, Elbow: 120 deg (forward only)
                velocity_limits=2.0,
            ),
            "7dof_industrial": lambda: cls(
                dof=7,
                name="7DOF_Industrial",
                link_lengths=[0.4, 0.4, 0.4, 0.3, 0.3, 0.2, 0.1],
                masses=[50, 40, 30, 20, 15, 10, 5],
                inertias=[5.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.2],
                damping=0.1,
            ),
            "simple_planar": lambda: cls(
                dof=3,
                name="3DOF_Planar",
                link_lengths=[1.0, 0.8, 0.6],
                masses=[2.0, 1.5, 1.0],
                inertias=[0.2, 0.15, 0.1],
                damping=0.05,
                joint_limits_min=[-3.14, -3.14, -3.14],
                joint_limits_max=[3.14, 3.14, 3.14],
            ),
            "light_arm": lambda: cls(
                dof=7,
                name="7DOF_Light",
                link_lengths=[0.5] * 7,
                masses=[1.0] * 7,
                inertias=[0.1] * 7,
                damping=0.05,
            ),
            "heavy_arm": lambda: cls(
                dof=7,
                name="7DOF_Heavy",
                link_lengths=[0.3] * 7,
                masses=[10.0] * 7,
                inertias=[1.0] * 7,
                damping=0.2,
            ),
            # 7-DOF biomechanically-proportioned upper-extremity arm matching
            # the kinematic structure of Fischer et al. (2021). Data-ready,
            # not wired into a planar env (five of its joints are out of plane).
            "7dof_fischer": lambda: cls(
                dof=7,
                name="7DOF_Fischer_Upper_Extremity",
                link_lengths=[0.30, 0.00, 0.00, 0.27, 0.00, 0.18, 0.00],
                masses=[2.10, 0.10, 0.10, 1.20, 0.10, 0.40, 0.10],
                inertias=[0.0210, 0.0010, 0.0008, 0.0080, 0.0006, 0.0015, 0.0005],
                damping=0.05,
                joint_limits_min=[float(np.deg2rad(v)) for v in (-60.0, -45.0, -90.0, 0.0, -90.0, -70.0, -30.0)],
                joint_limits_max=[float(np.deg2rad(v)) for v in (180.0, 180.0, 90.0, 150.0, 90.0, 70.0, 30.0)],
                velocity_limits=2.0,
            ),
            "default": lambda: cls(),
        }

        if preset_name not in presets:
            print(
                f"[WARN] Unknown preset '{preset_name}', returning default. "
                f"Available: {list(presets.keys())}"
            )
            return presets["default"]()

        return presets[preset_name]()

    @staticmethod
    def list_presets() -> List[str]:
        """Get list of available preset names"""
        return [
            "2dof_human_right",
            "2dof_human_left",
            "3dof_human_right",
            "3dof_human_left",
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

        if any(l < 0 for l in self.link_lengths):
            return False, "All link lengths must be non-negative"

        if any(c < 0 or c > max(l, 1e-9) for c, l in zip(self.com_offsets, self.link_lengths)):
            return False, "com_offsets must lie within [0, link_length]"

        if self.damping < 0 or any(d < 0 for d in self.joint_damping):
            return False, "Damping must be non-negative"

        if self.dt <= 0:
            return False, "Time step (dt) must be positive"

        if self.velocity_limits <= 0:
            return False, "Velocity limits must be positive"

        if self.torque_limits is not None and any(t <= 0 for t in self.torque_limits):
            return False, "Torque limits must be positive"

        if any(lo > hi for lo, hi in zip(self.joint_limits_min, self.joint_limits_max)):
            return False, "joint_limits_min must not exceed joint_limits_max"

        return True, "Configuration is valid"

    def __repr__(self) -> str:
        """String representation of configuration"""
        return (
            f"ArmConfiguration("
            f"name='{self.name}', dof={self.dof}, side={self.side}, "
            f"link_lengths={self.link_lengths[:3]}..., "
            f"masses={self.masses[:3]}..., "
            f"gravity={self.gravity}, damping={self.damping})"
        )


__all__ = ["ArmConfiguration"]
