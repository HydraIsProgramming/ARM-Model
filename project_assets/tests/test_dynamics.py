"""Rigid-body dynamics of the human arm (Phase 1)."""

import numpy as np
import pytest

from rl_armMotion.two_d.config import ArmConfiguration
from rl_armMotion.two_d.utils import dynamics as D


@pytest.fixture
def cfg():
    return ArmConfiguration.get_preset("2dof_human_right")


def test_preset_is_human_scale(cfg):
    assert cfg.validate()[0]
    assert cfg.reach == pytest.approx(0.65)
    assert cfg.gravity == pytest.approx(9.81)
    assert cfg.torque_limits == [25.0, 12.0]
    assert np.degrees(cfg.joint_limits_min[0]) == pytest.approx(-150.0)
    assert np.degrees(cfg.joint_limits_max[0]) == pytest.approx(90.0)
    assert np.degrees(cfg.joint_limits_max[1]) == pytest.approx(145.0)


def test_crba_rnea_match_two_link_closed_form(cfg):
    rng = np.random.default_rng(0)
    for _ in range(100):
        q = rng.uniform(-3, 3, 2)
        qd = rng.uniform(-5, 5, 2)
        M, Cqd, G = D.two_link_closed_form(q, qd, cfg)
        assert np.allclose(D.mass_matrix(q, cfg), M, atol=1e-10)
        assert np.allclose(D.bias_vector(q, qd, cfg), Cqd + G, atol=1e-10)
        assert np.allclose(D.gravity_vector(q, cfg), G, atol=1e-10)
        assert np.allclose(D.coriolis_vector(q, qd, cfg), Cqd, atol=1e-10)


def test_mass_matrix_spd(cfg):
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.uniform(-3, 3, 2)
        M = D.mass_matrix(q, cfg)
        assert np.allclose(M, M.T)
        assert np.all(np.linalg.eigvalsh(M) > 0)


def test_gravity_zero_when_hanging_and_peak_matches_analytic(cfg):
    assert np.allclose(D.gravity_vector([-np.pi / 2, 0.0], cfg), 0.0, atol=1e-12)
    G = D.gravity_vector([0.0, 0.0], cfg)
    m1, m2 = cfg.masses
    lc1, lc2 = cfg.com_offsets
    L1 = cfg.link_lengths[0]
    assert G[0] == pytest.approx(9.81 * (m1 * lc1 + m2 * L1 + m2 * lc2), rel=1e-9)
    assert G[1] == pytest.approx(9.81 * m2 * lc2, rel=1e-9)
    assert 10.0 < G[0] < 11.0 and 2.5 < G[1] < 3.0


def test_gravity_compensation_holds_pose(cfg):
    rng = np.random.default_rng(2)
    for _ in range(5):
        q0 = rng.uniform([-2.0, 0.3], [1.0, 2.0], 2)
        q, qd = q0.copy(), np.zeros(2)
        for _ in range(50):  # 1 s
            q, qd, _ = D.integrate(q, qd, D.gravity_vector(q, cfg), cfg)
        assert np.allclose(q, q0, atol=0.01)
        assert np.allclose(qd, 0.0, atol=0.05)


def test_arm_falls_under_gravity_and_settles_hanging(cfg):
    q, qd = np.array([0.0, 0.3]), np.zeros(2)
    q, qd, _ = D.integrate(q, qd, np.zeros(2), cfg)
    assert qd[0] < 0.0  # starts falling
    for _ in range(1000):  # 20 s with joint damping
        q, qd, _ = D.integrate(q, qd, np.zeros(2), cfg)
    # Passive rest: arm hanging, elbow a few degrees flexed by the end-range
    # spring (a relaxed human elbow is never fully straight).
    assert abs(q[0] + np.pi / 2) < 0.15
    assert 0.0 <= q[1] < np.deg2rad(12.0)
    assert np.all(np.abs(qd) < 0.05)


def test_free_pendulum_conserves_energy():
    cfg = ArmConfiguration.get_preset("2dof_human_right")
    data = cfg.to_dict()
    data["joint_damping"] = [0.0, 0.0]
    data["damping"] = 0.0
    data["passive_limit_stiffness"] = 0.0
    data["joint_limits_min"] = [-6.3, -6.3]   # no hard stops: a free double pendulum
    data["joint_limits_max"] = [6.3, 6.3]
    data["velocity_limits"] = 50.0
    data["physics_substeps"] = 16
    cfg0 = ArmConfiguration.from_dict(data)
    q, qd = np.array([-0.3, 0.8]), np.zeros(2)
    E0 = D.kinetic_energy(q, qd, cfg0) + D.potential_energy(q, cfg0)
    drift = 0.0
    for _ in range(100):  # 2 s
        q, qd, _ = D.integrate(q, qd, np.zeros(2), cfg0)
        E = D.kinetic_energy(q, qd, cfg0) + D.potential_energy(q, cfg0)
        drift = max(drift, abs(E - E0))
    assert drift < 0.02 * abs(E0) + 0.01


def test_damped_energy_nonincreasing(cfg):
    q, qd = np.array([-0.3, 0.8]), np.zeros(2)
    E_prev = D.kinetic_energy(q, qd, cfg) + D.potential_energy(q, cfg)
    for _ in range(200):
        q, qd, _ = D.integrate(q, qd, np.zeros(2), cfg)
        E = D.kinetic_energy(q, qd, cfg) + D.potential_energy(q, cfg)
        assert E <= E_prev + 1e-6
        E_prev = E


def test_limit_contact_zeroes_velocity_and_absorbs_torque(cfg):
    q, qd = np.array([-np.pi / 2, 0.0]), np.zeros(2)
    for _ in range(100):
        q, qd, at_limit = D.integrate(q, qd, np.array([0.0, -12.0]), cfg)
    assert q[1] == pytest.approx(0.0)
    assert qd[1] == 0.0
    assert at_limit[1]
    # The elbow torque into its stop must not swing the shoulder.
    assert abs(q[0] + np.pi / 2) < 1e-6
    assert abs(qd[0]) < 1e-6


def test_state_never_exceeds_limits(cfg):
    rng = np.random.default_rng(3)
    q, qd = np.array(cfg.initial_angles), np.zeros(2)
    lo, hi = np.asarray(cfg.joint_limits_min), np.asarray(cfg.joint_limits_max)
    for _ in range(500):
        tau = rng.uniform(-1, 1, 2) * np.asarray(cfg.torque_limits)
        q, qd, _ = D.integrate(q, qd, tau, cfg)
        assert np.all(q >= lo - 1e-12) and np.all(q <= hi + 1e-12)
        assert np.all(np.abs(qd) <= cfg.velocity_limits + 1e-12)
        assert np.all(np.isfinite(q)) and np.all(np.isfinite(qd))


def test_three_link_mass_matrix_consistent():
    cfg3 = ArmConfiguration.get_preset("3dof_human_right")
    rng = np.random.default_rng(4)
    for _ in range(20):
        q = rng.uniform(-1, 1, 3)
        Mj = D.mass_matrix(q, cfg3)
        Mr = np.column_stack([D.inverse_dynamics(q, np.zeros(3), np.eye(3)[j], cfg3, gravity=0.0) for j in range(3)])
        assert np.allclose(Mj, Mr, atol=1e-10)


def test_left_side_mirrors_x():
    from rl_armMotion.two_d.utils.arm_kinematics import ArmKinematics
    right = ArmConfiguration.get_preset("2dof_human_right")
    left = ArmConfiguration.get_preset("2dof_human_left")
    q = np.array([-0.5, 1.0])
    pr = ArmKinematics.forward_kinematics(q, right)
    pl = ArmKinematics.forward_kinematics(q, left)
    assert np.allclose(pl[:, 0], -pr[:, 0]) and np.allclose(pl[:, 1], pr[:, 1])
    assert np.allclose(D.gravity_vector(q, left), D.gravity_vector(q, right))
