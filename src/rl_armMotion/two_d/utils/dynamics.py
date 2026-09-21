"""Rigid-body dynamics of a planar revolute chain (N links, gravity, limits).

Pure NumPy functions on an :class:`ArmConfiguration`; no environment state.
Everything is float64.

Model
-----
    M(q) q'' + C(q, q') q' + G(q) = tau_act + tau_passive(q, q')

- ``mass_matrix``    M(q) from the link Jacobians:
                     M = sum_i [ m_i J_vi^T J_vi + I_i J_wi^T J_wi ]
- ``bias_vector``    C(q, q') q' + G(q) from a planar recursive Newton-Euler
                     pass with q'' = 0 (gravity enters as a base acceleration)
- ``gravity_vector`` G(q) alone (bias with q' = 0)
- ``passive_torque`` -B q'  minus a soft spring that engages within
                     ``passive_limit_margin`` of each joint limit
- ``integrate``      second-order midpoint (RK2) in ``substeps`` sub-steps with
                     hard joint stops that lock a joint pushed into them and
                     ZERO the velocity component pointing into the stop (an arm
                     resting on a limit is at rest)

Conventions: +x anterior, +y up, gravity along -y, joint angles counter-
clockwise from +x (link 0) and relative to the previous link (links >= 1).
For a 2-link chain the closed-form matrices are (l_c = com_offsets, I about
COM):

    M11 = I1 + m1 lc1^2 + I2 + m2 (L1^2 + lc2^2 + 2 L1 lc2 cos t1)
    M12 = M21 = I2 + m2 (lc2^2 + L1 lc2 cos t1)
    M22 = I2 + m2 lc2^2
    h   = m2 L1 lc2 sin t1
    C q' = [ -h (2 t0' t1' + t1'^2),  h t0'^2 ]
    G    = g [ (m1 lc1 + m2 L1) cos t0 + m2 lc2 cos(t0+t1),  m2 lc2 cos(t0+t1) ]

``two_link_closed_form`` returns exactly these and is the test oracle for the
generic code.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from rl_armMotion.two_d.config import ArmConfiguration


__all__ = [
    "link_frames",
    "end_effector_jacobian",
    "mass_matrix",
    "bias_vector",
    "gravity_vector",
    "passive_torque",
    "forward_dynamics",
    "inverse_dynamics",
    "integrate",
    "two_link_closed_form",
    "kinetic_energy",
    "potential_energy",
]


def _params(cfg: ArmConfiguration):
    L = np.asarray(cfg.link_lengths, dtype=float)
    m = np.asarray(cfg.masses, dtype=float)
    lc = np.asarray(cfg.com_offsets, dtype=float)
    I = np.asarray(cfg.inertias, dtype=float)
    return L, m, lc, I


def link_frames(q: np.ndarray, cfg: ArmConfiguration) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Absolute link angles, joint origins and COM positions (base at origin).

    Returns (phi (n,), joints (n+1, 2), coms (n, 2)); x is NOT mirrored here
    (dynamics are side-independent; mirroring is a rendering/goal transform).
    """
    q = np.asarray(q, dtype=float)
    L, _, lc, _ = _params(cfg)
    n = q.shape[0]
    phi = np.cumsum(q)
    joints = np.zeros((n + 1, 2))
    coms = np.zeros((n, 2))
    for i in range(n):
        u = np.array([np.cos(phi[i]), np.sin(phi[i])])
        coms[i] = joints[i] + lc[i] * u
        joints[i + 1] = joints[i] + L[i] * u
    return phi, joints, coms


def end_effector_jacobian(q: np.ndarray, cfg: ArmConfiguration) -> np.ndarray:
    """Linear Jacobian (2 x n) of the end-effector position w.r.t. q."""
    q = np.asarray(q, dtype=float)
    n = q.shape[0]
    _, joints, _ = link_frames(q, cfg)
    ee = joints[-1]
    J = np.zeros((2, n))
    for j in range(n):
        r = ee - joints[j]
        J[:, j] = [-r[1], r[0]]
    return J


def _com_jacobians(q: np.ndarray, cfg: ArmConfiguration) -> Tuple[np.ndarray, np.ndarray]:
    """Linear Jacobians of every COM (n, 2, n) and angular Jacobians (n, n)."""
    q = np.asarray(q, dtype=float)
    n = q.shape[0]
    _, joints, coms = link_frames(q, cfg)
    Jv = np.zeros((n, 2, n))
    Jw = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            r = coms[i] - joints[j]
            Jv[i, :, j] = [-r[1], r[0]]
            Jw[i, j] = 1.0
    return Jv, Jw


def mass_matrix(q: np.ndarray, cfg: ArmConfiguration) -> np.ndarray:
    """Joint-space inertia matrix M(q), (n, n), symmetric positive definite."""
    _, m, _, I = _params(cfg)
    Jv, Jw = _com_jacobians(q, cfg)
    n = Jw.shape[0]
    M = np.zeros((n, n))
    for i in range(n):
        M += m[i] * Jv[i].T @ Jv[i] + I[i] * np.outer(Jw[i], Jw[i])
    return 0.5 * (M + M.T)


def inverse_dynamics(q: np.ndarray, qd: np.ndarray, qdd: np.ndarray,
                     cfg: ArmConfiguration, gravity: float | None = None) -> np.ndarray:
    """tau = M(q) q'' + C(q, q') q' + G(q) via planar recursive Newton-Euler.

    Gravity enters as an upward base acceleration; passing ``gravity=0``
    returns the purely inertial torque.
    """
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    qdd = np.asarray(qdd, dtype=float)
    L, m, lc, I = _params(cfg)
    g = float(cfg.gravity if gravity is None else gravity)
    n = q.shape[0]
    phi = np.cumsum(q)
    omega = np.cumsum(qd)
    alpha = np.cumsum(qdd)

    # Forward pass: joint-origin and COM accelerations.
    a_joint = np.array([0.0, g])          # base acceleration = -gravity vector
    F = np.zeros((n, 2))
    N = np.zeros(n)
    r_com = np.zeros((n, 2))
    r_next = np.zeros((n, 2))
    for i in range(n):
        u = np.array([np.cos(phi[i]), np.sin(phi[i])])
        nrm = np.array([-np.sin(phi[i]), np.cos(phi[i])])
        a_com = a_joint + alpha[i] * lc[i] * nrm - omega[i] ** 2 * lc[i] * u
        F[i] = m[i] * a_com
        N[i] = I[i] * alpha[i]
        r_com[i] = lc[i] * u
        r_next[i] = L[i] * u
        a_joint = a_joint + alpha[i] * L[i] * nrm - omega[i] ** 2 * L[i] * u

    # Backward pass: joint forces and moments.
    tau = np.zeros(n)
    f_next = np.zeros(2)
    tau_next = 0.0
    for i in range(n - 1, -1, -1):
        cross_com = r_com[i, 0] * F[i, 1] - r_com[i, 1] * F[i, 0]
        cross_next = r_next[i, 0] * f_next[1] - r_next[i, 1] * f_next[0]
        tau[i] = N[i] + tau_next + cross_com + cross_next
        f_next = F[i] + f_next
        tau_next = tau[i]
    return tau


def bias_vector(q: np.ndarray, qd: np.ndarray, cfg: ArmConfiguration) -> np.ndarray:
    """C(q, q') q' + G(q)."""
    return inverse_dynamics(q, qd, np.zeros_like(np.asarray(q, dtype=float)), cfg)


def gravity_vector(q: np.ndarray, cfg: ArmConfiguration) -> np.ndarray:
    """G(q): the joint torque an actuator must supply to hold pose q."""
    z = np.zeros_like(np.asarray(q, dtype=float))
    return inverse_dynamics(q, z, z, cfg)


def coriolis_vector(q: np.ndarray, qd: np.ndarray, cfg: ArmConfiguration) -> np.ndarray:
    """C(q, q') q' alone (no gravity)."""
    return inverse_dynamics(q, qd, np.zeros_like(np.asarray(q, dtype=float)), cfg, gravity=0.0)


def passive_torque(q: np.ndarray, qd: np.ndarray, cfg: ArmConfiguration) -> np.ndarray:
    """Viscous damping plus a soft spring near each joint limit."""
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    B = np.asarray(cfg.joint_damping, dtype=float)
    tau = -B * qd
    K = float(cfg.passive_limit_stiffness)
    if K > 0.0:
        qmin = np.asarray(cfg.joint_limits_min, dtype=float)
        qmax = np.asarray(cfg.joint_limits_max, dtype=float)
        m_min = np.asarray(cfg.passive_limit_margins_min, dtype=float)
        m_max = np.asarray(cfg.passive_limit_margins_max, dtype=float)
        over = np.where(m_max > 0.0, np.maximum(0.0, q - (qmax - m_max)), 0.0)
        under = np.where(m_min > 0.0, np.maximum(0.0, (qmin + m_min) - q), 0.0)
        tau += -K * over + K * under
    return tau


def forward_dynamics(q: np.ndarray, qd: np.ndarray, tau: np.ndarray,
                     cfg: ArmConfiguration, locked: np.ndarray | None = None) -> np.ndarray:
    """q'' = M^-1 (tau + tau_passive - C q' - G).

    ``locked`` marks joints held by a hard stop this instant: their
    acceleration is constrained to zero and the reduced system is solved for
    the free joints, so a torque pushing into a stop is absorbed by the stop
    instead of being transmitted up the chain.
    """
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    tau = np.asarray(tau, dtype=float)
    M = mass_matrix(q, cfg)
    rhs = tau + passive_torque(q, qd, cfg) - bias_vector(q, qd, cfg)
    qdd = np.zeros_like(q)
    if locked is None or not np.any(locked):
        qdd = np.linalg.solve(M, rhs)
    else:
        free = ~np.asarray(locked, dtype=bool)
        if np.any(free):
            qdd[free] = np.linalg.solve(M[np.ix_(free, free)], rhs[free])
    a_max = float(cfg.max_acceleration)
    if a_max > 0.0:
        qdd = np.clip(qdd, -a_max, a_max)
    return qdd


def _locked_joints(q, qd, tau, cfg, qmin, qmax, eps=1e-12):
    """Joints resting on a stop whose free acceleration pushes into it."""
    on_min = (q <= qmin + eps) & (qd <= 0.0)
    on_max = (q >= qmax - eps) & (qd >= 0.0)
    if not (np.any(on_min) or np.any(on_max)):
        return None
    qdd = forward_dynamics(q, qd, tau, cfg)
    locked = (on_min & (qdd < 0.0)) | (on_max & (qdd > 0.0))
    if not np.any(locked):
        return None
    qdd = forward_dynamics(q, qd, tau, cfg, locked=locked)
    locked2 = locked | (on_min & (qdd < 0.0)) | (on_max & (qdd > 0.0))
    return locked2


def integrate(q: np.ndarray, qd: np.ndarray, tau: np.ndarray, cfg: ArmConfiguration,
              dt: float | None = None, substeps: int | None = None
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance (q, q') by one control step under constant actuator torque.

    Second-order (midpoint / RK2) integration in ``substeps`` sub-steps. Joints
    resting on a hard stop with the net generalised force pushing into it are
    locked for the sub-step (q'' = 0, reduced system solved), so a torque into
    a stop is absorbed there instead of being transmitted up the chain. After
    each sub-step the angle is clamped to the range of motion and the velocity
    component into the stop is zeroed, so a stationary arm on a limit reports
    zero velocity. Returns (q, q', at_limit).
    """
    q = np.array(q, dtype=float)
    qd = np.array(qd, dtype=float)
    tau = np.asarray(tau, dtype=float)
    dt = float(cfg.dt if dt is None else dt)
    n_sub = int(cfg.physics_substeps if substeps is None else substeps)
    h = dt / max(n_sub, 1)
    qmin = np.asarray(cfg.joint_limits_min, dtype=float)
    qmax = np.asarray(cfg.joint_limits_max, dtype=float)
    v_max = float(cfg.velocity_limits)
    at_limit = np.zeros(q.shape[0], dtype=bool)
    for _ in range(max(n_sub, 1)):
        locked = _locked_joints(q, qd, tau, cfg, qmin, qmax)
        k1 = forward_dynamics(q, qd, tau, cfg, locked=locked)
        q_mid = q + 0.5 * h * qd
        qd_mid = qd + 0.5 * h * k1
        k2 = forward_dynamics(q_mid, qd_mid, tau, cfg, locked=locked)
        q = q + h * qd_mid
        qd = qd + h * k2
        if v_max > 0.0:
            qd = np.clip(qd, -v_max, v_max)
        low = q <= qmin
        high = q >= qmax
        q = np.where(low, qmin, q)
        q = np.where(high, qmax, q)
        qd = np.where(low, np.maximum(qd, 0.0), qd)
        qd = np.where(high, np.minimum(qd, 0.0), qd)
        at_limit = low | high
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qd))):
        raise FloatingPointError("dynamics.integrate produced a non-finite state")
    return q, qd, at_limit


def kinetic_energy(q: np.ndarray, qd: np.ndarray, cfg: ArmConfiguration) -> float:
    qd = np.asarray(qd, dtype=float)
    return float(0.5 * qd @ mass_matrix(q, cfg) @ qd)


def potential_energy(q: np.ndarray, cfg: ArmConfiguration) -> float:
    """Gravitational plus end-range spring potential energy."""
    q = np.asarray(q, dtype=float)
    _, m, _, _ = _params(cfg)
    _, _, coms = link_frames(q, cfg)
    E = float(cfg.gravity * np.sum(m * coms[:, 1]))
    K = float(cfg.passive_limit_stiffness)
    if K > 0.0:
        qmin = np.asarray(cfg.joint_limits_min, dtype=float)
        qmax = np.asarray(cfg.joint_limits_max, dtype=float)
        m_min = np.asarray(cfg.passive_limit_margins_min, dtype=float)
        m_max = np.asarray(cfg.passive_limit_margins_max, dtype=float)
        over = np.where(m_max > 0.0, np.maximum(0.0, q - (qmax - m_max)), 0.0)
        under = np.where(m_min > 0.0, np.maximum(0.0, (qmin + m_min) - q), 0.0)
        E += float(0.5 * K * np.sum(over ** 2 + under ** 2))
    return E


def two_link_closed_form(q: np.ndarray, qd: np.ndarray, cfg: ArmConfiguration
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form (M, C q', G) for a 2-link chain; the test oracle."""
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    L, m, lc, I = _params(cfg)
    g = float(cfg.gravity)
    L1, lc1, lc2 = L[0], lc[0], lc[1]
    m1, m2, I1, I2 = m[0], m[1], I[0], I[1]
    t0, t1 = q
    c1 = np.cos(t1)
    M = np.array([
        [I1 + m1 * lc1 ** 2 + I2 + m2 * (L1 ** 2 + lc2 ** 2 + 2 * L1 * lc2 * c1),
         I2 + m2 * (lc2 ** 2 + L1 * lc2 * c1)],
        [I2 + m2 * (lc2 ** 2 + L1 * lc2 * c1), I2 + m2 * lc2 ** 2],
    ])
    hh = m2 * L1 * lc2 * np.sin(t1)
    Cqd = np.array([-hh * (2 * qd[0] * qd[1] + qd[1] ** 2), hh * qd[0] ** 2])
    G = g * np.array([
        (m1 * lc1 + m2 * L1) * np.cos(t0) + m2 * lc2 * np.cos(t0 + t1),
        m2 * lc2 * np.cos(t0 + t1),
    ])
    return M, Cqd, G
