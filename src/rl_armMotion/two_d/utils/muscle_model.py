"""Hill-type muscle dynamics for biomechanically informed actuation.

This module implements the canonical Hill-type muscle model used by Fischer,
Hoinville, Eickhoff, and Lilienthal (2021), "Reinforcement learning control
of a biomechanical model of the upper extremity", Scientific Reports
11:14445, https://doi.org/10.1038/s41598-021-93760-1, and by virtually all
contemporary musculoskeletal simulators (OpenSim, MuJoCo, MyoSuite,
Anybody, SCONE). The model dates back to A.V. Hill's classical 1938 paper,
"The heat of shortening and the dynamic constants of muscle", Proceedings
of the Royal Society of London. Series B, 126:136--195, and is the
de facto standard for converting a neural-activation command into a
joint-level torque.

The model expresses the contractile-element force as the product of three
dimensionless factors and the maximum isometric force:

    F = a * F_max * f_L(L_norm) * f_V(V_norm)

where

    a in [0, 1]              is the muscle activation level (the input
                             from the controller, equivalent to a
                             normalised neural drive),
    F_max                    is the maximum isometric force (N),
    f_L(L_norm)              is the force-length curve, dimensionless,
                             peaking at L_norm = 1.0 and decaying for
                             both shorter and longer fibre lengths, and
    f_V(V_norm)              is the force-velocity curve, dimensionless,
                             monotonically decreasing in shortening
                             (concentric) velocity and increasing
                             toward an eccentric plateau in lengthening.

The force-length curve used here follows the Gaussian approximation
introduced by Thelen (2003), "Adjustment of muscle mechanics model
parameters to simulate dynamic contractions in older adults", Journal
of Biomechanical Engineering 125:70--77,

    f_L(L_norm) = exp(-((L_norm - 1) / sigma_L) ** 2),

with the half-width parameter ``sigma_L`` set to the standard 0.45
(unitless) so that the curve drops to ~10% of peak at L_norm = 0.4 and
L_norm = 1.6. This matches the active force-length curves reported in
Zajac (1989), "Muscle and tendon: Properties, models, scaling, and
application to biomechanics and motor control", Critical Reviews in
Biomedical Engineering 17:359--411.

The force-velocity curve is the standard Hill hyperbolic form
(eccentric branch piecewise-extended for stability), with a maximum
shortening velocity ``V_max`` (in fibre lengths per second) and an
eccentric plateau ``F_ecc_max`` (typically 1.5 x F_max). At V_norm = 0
the curve gives 1.0 (isometric); at V_norm = +1 it gives 0
(unloaded shortening); for negative V_norm (eccentric / lengthening
contraction) the curve smoothly approaches the eccentric plateau.

Although Fischer et al. (2021) ran their experiments inside MuJoCo with
its native musculoskeletal model file, the underlying mathematics is
identical to the formulae in this module. This module therefore allows
the existing 2-DOF kinematic arm in this project to be driven by the
same biomechanical actuation law as Fischer's 7-DOF arm, without the
MuJoCo dependency. Wiring the model into a true musculoskeletal
multi-body simulation (with anatomically correct moment arms, fibre
geometry, and tendon dynamics) is a separate piece of work and is not
addressed by this module.

References
----------
Hill, A. V. (1938). The heat of shortening and the dynamic constants of
    muscle. Proc. R. Soc. Lond. B, 126, 136--195.
Zajac, F. E. (1989). Muscle and tendon: properties, models, scaling, and
    application to biomechanics and motor control. CRC Crit. Rev. Biomed.
    Eng., 17, 359--411.
Thelen, D. G. (2003). Adjustment of muscle mechanics model parameters to
    simulate dynamic contractions in older adults. J. Biomech. Eng., 125,
    70--77.
Fischer, M., Hoinville, T., Eickhoff, S. B., & Lilienthal, A. J. (2021).
    Reinforcement learning control of a biomechanical model of the upper
    extremity. Sci. Rep., 11, 14445.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


__all__ = ["HillTypeMuscle", "MuscleParameters", "JointMusclePair"]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class MuscleParameters:
    """Parameters of a Hill-type muscle.

    Attributes
    ----------
    f_max : float
        Maximum isometric force in Newtons. Determines the absolute scale
        of the torque this muscle can produce when fully activated at the
        optimal fibre length and zero velocity. Default 100 N is a
        reasonable order of magnitude for a single mid-sized upper-limb
        muscle (e.g., long head of biceps brachii reports ~600 N at peak,
        so a single-muscle torque actuator at ~100 N corresponds to a
        modest contributor).
    optimal_length : float
        Optimal fibre length L_opt in metres. Force-length is normalised
        as L_norm = L / L_opt, peaking at 1.0. Default 0.10 m.
    v_max : float
        Maximum shortening velocity, in optimal fibre lengths per second.
        At ``v_max`` the muscle produces zero force in concentric
        contraction. Mammalian skeletal muscle is typically ~10
        L_opt / s; default 10.0.
    sigma_L : float
        Half-width of the Gaussian force-length curve. Default 0.45
        (unitless) following Thelen (2003), giving a curve that drops to
        ~10% of peak at L_norm = 0.4 and 1.6.
    a_hill : float
        Curvature constant of the Hill hyperbola (a/F0). 0.25 is the
        canonical value for mammalian skeletal muscle (Zajac 1989).
    f_ecc_max : float
        Eccentric force plateau as a multiple of F_max. Lengthening
        contractions can transiently produce more force than isometric;
        Fischer et al. (2021) and the MuJoCo musculoskeletal model both
        use 1.5. Default 1.5.
    activation_min : float
        Minimum allowed activation. Default 0.0 (fully relaxed).
    activation_max : float
        Maximum allowed activation. Default 1.0 (fully recruited).
    """

    f_max: float = 100.0
    optimal_length: float = 0.10
    v_max: float = 10.0
    sigma_L: float = 0.45
    f_ecc_max: float = 1.5
    a_hill: float = 0.25
    activation_min: float = 0.0
    activation_max: float = 1.0

    def __post_init__(self) -> None:
        if self.a_hill <= 0.0:
            raise ValueError(f"a_hill must be positive, got {self.a_hill}")
        if self.f_max <= 0.0:
            raise ValueError(f"f_max must be positive, got {self.f_max}")
        if self.optimal_length <= 0.0:
            raise ValueError(
                f"optimal_length must be positive, got {self.optimal_length}"
            )
        if self.v_max <= 0.0:
            raise ValueError(f"v_max must be positive, got {self.v_max}")
        if self.sigma_L <= 0.0:
            raise ValueError(f"sigma_L must be positive, got {self.sigma_L}")
        if self.f_ecc_max < 1.0:
            raise ValueError(
                f"f_ecc_max should be >= 1.0 (eccentric > isometric), "
                f"got {self.f_ecc_max}"
            )
        if not 0.0 <= self.activation_min <= self.activation_max <= 1.0:
            raise ValueError(
                f"activation bounds must satisfy 0 <= min <= max <= 1, "
                f"got [{self.activation_min}, {self.activation_max}]"
            )


# ---------------------------------------------------------------------------
# Hill-type muscle
# ---------------------------------------------------------------------------
class HillTypeMuscle:
    """Hill-type muscle producing force from activation, length, and velocity.

    Parameters
    ----------
    params : MuscleParameters, optional
        Muscle parameters. Defaults to a generic upper-limb muscle.

    Notes
    -----
    The model is stateless: it has no internal activation dynamics
    integrator. The activation passed to :meth:`force` is treated as the
    instantaneous active state. To add first-order activation dynamics
    a la Zajac (1989), wrap an activation low-pass on the controller side
    or extend this class to track a per-call activation state.
    """

    def __init__(self, params: Optional[MuscleParameters] = None):
        self.params = params if params is not None else MuscleParameters()

    # ------------------------------------------------------------------ #
    # Activation
    # ------------------------------------------------------------------ #
    def clip_activation(self, activation: float) -> float:
        """Clamp activation to [activation_min, activation_max]."""
        return float(
            max(self.params.activation_min,
                min(self.params.activation_max, activation))
        )

    # ------------------------------------------------------------------ #
    # Force-length and force-velocity factors
    # ------------------------------------------------------------------ #
    def force_length(self, length: float) -> float:
        """Return f_L(L) for fibre length ``length`` (metres).

        Uses the Thelen (2003) Gaussian approximation:

            f_L(L_norm) = exp(-((L_norm - 1) / sigma_L) ** 2)

        which is unimodal at the optimal length, smooth, and bounded in
        [0, 1]. Identical at L = L_opt to the polynomial Gordon et al.
        active force-length curve used in OpenSim and MuJoCo.
        """
        if length < 0.0:
            return 0.0
        l_norm = length / self.params.optimal_length
        return math.exp(-((l_norm - 1.0) / self.params.sigma_L) ** 2)

    def force_velocity(self, velocity: float) -> float:
        """Return f_V(V) for fibre velocity ``velocity`` (L_opt/sec).

        Concentric (positive ``velocity`` = shortening) is the Hill hyperbola

            f_V(v) = (1 - v) / (1 + v / a),   v = V / v_max in [0, 1)

        normalised so f_V(0) = 1 and f_V(v_max) = 0, with curvature a = 0.25
        (Zajac 1989, Fig. 4): at half v_max the force is already down to
        0.5 / 3 = 0.167. Eccentric (``velocity`` < 0 = lengthening) rises
        smoothly toward the plateau ``f_ecc_max``,

            f_V(v) = f_ecc - (f_ecc - 1) * exp(k v),   k = (1 + 1/a) / (f_ecc - 1)

        with k chosen so the slope is continuous at v = 0 (the concentric
        slope there is -(1 + 1/a)).
        """
        v_norm = velocity / self.params.v_max
        a = self.params.a_hill
        if v_norm >= 0.0:
            if v_norm >= 1.0:
                return 0.0
            return float((1.0 - v_norm) / (1.0 + v_norm / a))
        f_ecc = self.params.f_ecc_max
        if f_ecc <= 1.0:
            return 1.0
        k = (1.0 + 1.0 / a) / (f_ecc - 1.0)
        return float(f_ecc - (f_ecc - 1.0) * math.exp(k * v_norm))

    # ------------------------------------------------------------------ #
    # Combined force
    # ------------------------------------------------------------------ #
    def force(
        self,
        activation: float,
        length: float,
        velocity: float,
    ) -> float:
        """Return the contractile-element force in Newtons.

        Parameters
        ----------
        activation : float
            Neural drive in [0, 1] (clipped internally).
        length : float
            Current fibre length in metres.
        velocity : float
            Fibre velocity in L_opt/sec. Positive = shortening,
            negative = lengthening.

        Returns
        -------
        float
            Force F = a * F_max * f_L(L) * f_V(V), in Newtons.
        """
        a = self.clip_activation(activation)
        return (
            a
            * self.params.f_max
            * self.force_length(length)
            * self.force_velocity(velocity)
        )

    # ------------------------------------------------------------------ #
    # Vector convenience
    # ------------------------------------------------------------------ #
    def force_vector(
        self,
        activations: Iterable[float],
        lengths: Iterable[float],
        velocities: Iterable[float],
    ) -> np.ndarray:
        """Vectorised force computation across multiple muscles.

        All three inputs must have the same length. Returns an ndarray
        of forces in Newtons.
        """
        a_arr = np.asarray(list(activations), dtype=float)
        l_arr = np.asarray(list(lengths), dtype=float)
        v_arr = np.asarray(list(velocities), dtype=float)
        if not (a_arr.shape == l_arr.shape == v_arr.shape):
            raise ValueError(
                f"activations, lengths, velocities must have matching shapes, "
                f"got {a_arr.shape}, {l_arr.shape}, {v_arr.shape}"
            )
        out = np.empty_like(a_arr)
        for i in range(a_arr.size):
            out[i] = self.force(float(a_arr[i]), float(l_arr[i]), float(v_arr[i]))
        return out

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        p = self.params
        return (
            f"HillTypeMuscle(f_max={p.f_max:.1f}N, L_opt={p.optimal_length*100:.1f}cm, "
            f"v_max={p.v_max:.1f}L/s, sigma_L={p.sigma_L:.2f}, "
            f"f_ecc_max={p.f_ecc_max:.2f})"
        )


# ---------------------------------------------------------------------------
# Antagonist pair acting on one revolute joint
# ---------------------------------------------------------------------------
class JointMusclePair:
    """Two lumped Hill-type muscles pulling a joint in opposite directions.

    ``pos`` produces positive joint torque (raises theta), ``neg`` produces
    negative torque. Anatomically, for the sagittal-plane arm used here:
    shoulder ``pos`` = anterior deltoid / pectoralis (flexion raises theta0
    from the hanging pose), ``neg`` = posterior deltoid / latissimus;
    elbow ``pos`` = biceps / brachialis, ``neg`` = triceps.

    Geometry (constant moment arm ``r``, fibre length linear in angle):

        s     = +1 for pos, -1 for neg
        L_m   = L_opt - s * r * (theta - theta_ref)     # pos SHORTENS as theta rises
        v_m   = s * r * omega / L_opt                   # shortening positive, L_opt/s
        tau   = r * (F_pos - F_neg)

    ``theta_ref`` is the joint's mid-range angle, so both muscles sit at
    optimal length in the middle of the range and weaken toward the ends.
    """

    def __init__(
        self,
        pos: HillTypeMuscle,
        neg: HillTypeMuscle,
        moment_arm: float,
        theta_ref: float = 0.0,
        torque_limit: Optional[float] = None,
    ):
        if moment_arm <= 0.0:
            raise ValueError(f"moment_arm must be positive, got {moment_arm}")
        self.pos = pos
        self.neg = neg
        self.moment_arm = float(moment_arm)
        self.theta_ref = float(theta_ref)
        self.torque_limit = None if torque_limit is None else float(torque_limit)

    @classmethod
    def from_config(cls, cfg, joint: int) -> "JointMusclePair":
        """Build the pair for joint ``joint`` of an ArmConfiguration."""
        r = (cfg.muscle_moment_arms or [0.05] * cfg.dof)[joint]
        f_max = (cfg.muscle_f_max or [100.0] * cfg.dof)[joint]
        l_opt = (cfg.muscle_l_opt or [0.10] * cfg.dof)[joint]
        params = MuscleParameters(
            f_max=float(f_max), optimal_length=float(l_opt),
            v_max=float(getattr(cfg, "muscle_v_max", 6.0)),
        )
        theta_ref = float(cfg.joint_mid_angles[joint])
        tl = None if cfg.torque_limits is None else float(cfg.torque_limits[joint])
        return cls(HillTypeMuscle(params), HillTypeMuscle(params), r, theta_ref, tl)

    def fibre_state(self, theta: float, omega: float):
        """(L_pos, v_pos, L_neg, v_neg) for the current joint state."""
        r = self.moment_arm
        d = float(theta) - self.theta_ref
        l_pos = self.pos.params.optimal_length - r * d
        l_neg = self.neg.params.optimal_length + r * d
        v_pos = r * float(omega) / self.pos.params.optimal_length
        v_neg = -r * float(omega) / self.neg.params.optimal_length
        return l_pos, v_pos, l_neg, v_neg

    def torque(self, a_pos: float, a_neg: float, theta: float, omega: float):
        """Net joint torque (N m) and a diagnostics dict."""
        l_pos, v_pos, l_neg, v_neg = self.fibre_state(theta, omega)
        f_pos = self.pos.force(a_pos, l_pos, v_pos)
        f_neg = self.neg.force(a_neg, l_neg, v_neg)
        tau = self.moment_arm * (f_pos - f_neg)
        if self.torque_limit is not None:
            tau = float(max(-self.torque_limit, min(self.torque_limit, tau)))
        return float(tau), {
            "force_pos": float(f_pos), "force_neg": float(f_neg),
            "length_pos": float(l_pos), "length_neg": float(l_neg),
            "velocity_pos": float(v_pos), "velocity_neg": float(v_neg),
        }

    @property
    def max_isometric_torque(self) -> float:
        return self.moment_arm * self.pos.params.f_max
