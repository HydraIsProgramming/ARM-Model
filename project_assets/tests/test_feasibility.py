"""Inverse kinematics and waypoint feasibility."""

import numpy as np
import pytest

from rl_armMotion.two_d.environments.task_env import ArmTaskEnv
from rl_armMotion.two_d.evaluation import CANONICAL_WAYPOINTS, CANONICAL_WAYPOINTS_V1, WAYPOINTS_SPREAD, EVAL_STANDARD


@pytest.fixture
def env():
    return ArmTaskEnv()


def test_canonical_waypoints_feasible(env):
    for wp in CANONICAL_WAYPOINTS[3]:
        f = env.waypoint_feasibility(wp, env.max_curriculum_tolerance)   # robust at the coarsest rung
        assert f["feasible"], f
        assert f["rom_margin_deg"] > 5.0
        assert max(f["torque_ratio"]) < 0.5
        assert f["reach_margin"] >= 0.18


def test_v1_final_waypoint_rejected_near_reach_circle(env):
    """The Stage-B failure mode: a goal 9 cm inside the reach circle lets a
    straight arm resting on the elbow spring count as in-goal at 15 cm."""
    f = env.waypoint_feasibility(CANONICAL_WAYPOINTS_V1[3][-1], env.max_curriculum_tolerance)
    assert f["reachable"] and not f["feasible"]
    assert "reach circle" in f["reason"]
    # With a 5 cm coarsest tolerance the same point is fine: the rule scales with the curriculum.
    tight = ArmTaskEnv(max_curriculum_tolerance=0.05)
    assert tight.waypoint_feasibility(CANONICAL_WAYPOINTS_V1[3][-1], 0.05)["feasible"]
    with pytest.warns(RuntimeWarning):
        env.set_waypoints(CANONICAL_WAYPOINTS_V1[3])   # loads (old packages must still open) but warns


def test_spread_waypoints_reachable_but_near_limit(env):
    f = env.waypoint_feasibility(WAYPOINTS_SPREAD[0], EVAL_STANDARD.tolerance)
    assert f["reachable"] and not f["feasible"]          # too close to the reach circle for the curriculum
    tight = ArmTaskEnv(max_curriculum_tolerance=0.05)
    g = tight.waypoint_feasibility(WAYPOINTS_SPREAD[0], 0.05)
    assert g["reachable"] and g["within_rom"]
    assert g["rom_margin_deg"] < 10.0                     # and its shoulder sits near the ROM limit


def test_rejects_unreachable(env):
    f = env.waypoint_feasibility([1.0 + 0.7, 0.0], 0.05)
    assert not f["feasible"] and not f["reachable"]
    f = env.waypoint_feasibility([1.0, 0.0], 0.05)   # at the shoulder
    assert not f["feasible"]


def test_rejects_rom_violation(env):
    # Directly behind the shoulder at shoulder height needs theta0 = 180 deg (outside -150..90).
    f = env.waypoint_feasibility([1.0 - 0.6, 0.0], 0.02)
    assert f["reachable"] and not f["within_rom"] and not f["feasible"]


def test_fk_ik_roundtrip(env):
    rng = np.random.default_rng(0)
    lo = np.asarray(env.config.joint_limits_min) + 0.05
    hi = np.asarray(env.config.joint_limits_max) - 0.05
    for _ in range(50):
        q = rng.uniform(lo, hi)
        if q[1] < 0.2:
            continue
        p = env._end_effector_from_angles(q)
        q_ik = env.solve_ik(p)
        assert q_ik is not None
        assert np.allclose(env._end_effector_from_angles(q_ik), p, atol=1e-9)
        assert np.allclose(q_ik, q, atol=1e-6)


def test_left_arm_ik_mirrors_right():
    right = ArmTaskEnv(preset="2dof_human_right")
    left = ArmTaskEnv(preset="2dof_human_left")
    target = np.asarray(CANONICAL_WAYPOINTS[1][0])
    mirrored = np.array([2 * right.shoulder_base_position[0] - target[0], target[1]])
    assert np.allclose(left.solve_ik(mirrored), right.solve_ik(target))


def test_three_link_ik_dls():
    env3 = ArmTaskEnv(preset="3dof_human_right", actuation_mode="torque")
    q = env3.solve_ik([1.35, 0.2])
    assert q is not None and q.shape == (3,)
    assert np.allclose(env3._end_effector_from_angles(q), [1.35, 0.2], atol=1e-3)
