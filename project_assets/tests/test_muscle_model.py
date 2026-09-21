"""Hill-type muscle model and antagonist pairs (Phase 1.3)."""

import numpy as np
import pytest

from rl_armMotion.two_d.config import ArmConfiguration
from rl_armMotion.two_d.utils import dynamics as D
from rl_armMotion.two_d.utils.muscle_model import HillTypeMuscle, JointMusclePair, MuscleParameters


@pytest.fixture
def muscle():
    return HillTypeMuscle(MuscleParameters(v_max=6.0))


def test_fv_isometric_one_and_vmax_zero(muscle):
    assert muscle.force_velocity(0.0) == pytest.approx(1.0)
    assert muscle.force_velocity(6.0) == 0.0
    assert muscle.force_velocity(100.0) == 0.0


def test_fv_half_vmax_hyperbolic(muscle):
    assert muscle.force_velocity(3.0) == pytest.approx(0.5 / 3.0)


def test_fv_eccentric_gt_one_and_plateau(muscle):
    assert muscle.force_velocity(-1.0) > 1.0
    assert muscle.force_velocity(-1000.0) == pytest.approx(1.5)


def test_fv_slope_continuous_at_zero(muscle):
    h = 1e-6
    left = (muscle.force_velocity(0.0) - muscle.force_velocity(-h)) / (h / 6.0)
    right = (muscle.force_velocity(h) - muscle.force_velocity(0.0)) / (h / 6.0)
    assert left == pytest.approx(right, rel=1e-3)
    assert left == pytest.approx(-5.0, rel=1e-3)


def test_fv_monotone_decreasing(muscle):
    vs = np.linspace(-6, 6, 200)
    f = [muscle.force_velocity(v) for v in vs]
    assert all(b <= a + 1e-12 for a, b in zip(f, f[1:]))


@pytest.fixture
def cfg():
    return ArmConfiguration.get_preset("2dof_human_right")


def test_fl_at_rom_ends_ge_0p4(cfg):
    for j in range(2):
        pair = JointMusclePair.from_config(cfg, j)
        iso = pair.max_isometric_torque
        t_min, _ = pair.torque(1.0, 0.0, cfg.joint_limits_min[j], 0.0)
        t_max, _ = pair.torque(1.0, 0.0, cfg.joint_limits_max[j], 0.0)
        assert t_min >= 0.4 * iso and t_max >= 0.4 * iso
        t_mid, _ = pair.torque(1.0, 0.0, cfg.joint_mid_angles[j], 0.0)
        assert t_mid == pytest.approx(iso)


def test_extensor_force_drops_with_shortening_speed(cfg):
    pair = JointMusclePair.from_config(cfg, 0)
    mid = cfg.joint_mid_angles[0]
    t0, d0 = pair.torque(1.0, 0.0, mid, 0.0)
    t1, d1 = pair.torque(1.0, 0.0, mid, 3.0)     # joint rising: pos muscle shortens
    assert d1["velocity_pos"] > 0.0
    assert d1["force_pos"] < d0["force_pos"]
    # Lengthening (joint falling) gives eccentric force above isometric.
    _, d2 = pair.torque(1.0, 0.0, mid, -3.0)
    assert d2["force_pos"] > d0["force_pos"]


def test_pair_torque_within_limits(cfg):
    pair = JointMusclePair.from_config(cfg, 0)
    for w in (-6.0, -3.0, 0.0, 3.0, 6.0):
        for th in np.linspace(cfg.joint_limits_min[0], cfg.joint_limits_max[0], 7):
            t, _ = pair.torque(1.0, 0.0, th, w)
            assert -25.0 - 1e-9 <= t <= 25.0 + 1e-9
            t, _ = pair.torque(0.0, 1.0, th, w)
            assert -25.0 - 1e-9 <= t <= 25.0 + 1e-9


def test_cocontraction_zero_net_torque_but_damps_motion(cfg):
    pair = JointMusclePair.from_config(cfg, 1)
    mid = cfg.joint_mid_angles[1]
    t, _ = pair.torque(0.7, 0.7, mid, 0.0)
    assert t == pytest.approx(0.0, abs=1e-9)
    t_moving, _ = pair.torque(0.7, 0.7, mid, 2.0)
    assert t_moving < 0.0  # opposes the motion
    t_low, _ = pair.torque(0.2, 0.2, mid, 2.0)
    assert t_moving < t_low  # more co-contraction = more damping


def test_max_muscle_torque_ge_2x_gravity(cfg):
    G = D.gravity_vector([0.0, 0.0], cfg)
    for j in range(2):
        pair = JointMusclePair.from_config(cfg, j)
        assert pair.max_isometric_torque >= 2.0 * abs(G[j])


def test_extensor_activation_raises_theta():
    from rl_armMotion.two_d.evaluation import make_env
    env = make_env(actuation_mode="muscle", waypoints=2)
    env.reset(seed=0, options={"randomize": False})
    q0 = env.q.copy()
    for _ in range(25):
        env.step(np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))
    assert env.q[0] > q0[0] and env.q[1] > q0[1]
