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


def test_rejects_pose_inside_the_passive_end_range_zone(env):
    """A target whose only pose lies inside the elbow's end-range spring is not
    holdable: the spring pushes the joint out and W_LIMIT charges for staying.

    This is the joint-space twin of the reach-circle rule above.
    """
    qmax = np.asarray(env.config.joint_limits_max, dtype=float)
    margin = np.asarray(env.config.passive_limit_margins_max, dtype=float)
    assert margin[1] > 0, "this preset has no elbow end-range spring to test"
    # A pose 2 deg inside the elbow's end-range zone, comfortable at the shoulder.
    q = np.array([np.radians(-30.0), qmax[1] - margin[1] + np.radians(2.0)])
    target = env._end_effector_from_angles(q)
    f = env.waypoint_feasibility(target, EVAL_STANDARD.tolerance)
    assert f["reachable"] and f["within_rom"]
    assert not f["feasible"]
    assert "end-range" in f["reason"]
    assert f["end_range_clearance_deg"] == pytest.approx(-2.0, abs=0.1)

    # Just outside the zone the same geometry is accepted.
    q_ok = np.array([np.radians(-30.0), qmax[1] - margin[1] - np.radians(2.0)])
    g = env.waypoint_feasibility(env._end_effector_from_angles(q_ok), EVAL_STANDARD.tolerance)
    assert g["feasible"], g
    assert g["end_range_clearance_deg"] == pytest.approx(2.0, abs=0.1)


def test_end_range_rule_shrinks_the_sampler_grid_without_emptying_it(env):
    """The random-waypoint sampler must lose the unholdable targets and keep a
    workspace that still reaches well inside the arm."""
    pts = env.feasible_points()
    assert len(pts) > 500, "the end-range rule must not gut the feasible region"
    shoulder = np.asarray(env.shoulder_base_position, dtype=float)
    radius = np.linalg.norm(pts - shoulder, axis=1)
    assert radius.min() < 0.30, "close-in targets must survive the rule"
    for p in pts:
        f = env.waypoint_feasibility(p, env.goal_tolerance)
        assert f["end_range_clearance_deg"] >= 0.0


def test_legacy_preset_without_springs_is_unaffected():
    """A joint with no passive margin is exempt, so gravity-free presets keep
    their old feasible set."""
    legacy = ArmTaskEnv(preset="2dof_human_right")
    legacy.config.passive_limit_margins_min = [0.0, 0.0]
    legacy.config.passive_limit_margins_max = [0.0, 0.0]
    qmax = np.asarray(legacy.config.joint_limits_max, dtype=float)
    q = np.array([np.radians(-30.0), qmax[1] - np.radians(3.0)])
    f = legacy.waypoint_feasibility(legacy._end_effector_from_angles(q), EVAL_STANDARD.tolerance)
    assert f["feasible"], f


# ---------------------------------------------------------------------------
# Full human workspace (Stage R7 on): only the joint ranges and strength limit
# where a target may be.

def _human():
    return ArmTaskEnv(random_waypoints=3, joint_margin_deg=5.0, reach_margin=0.0)


def test_human_limits_workspace_is_far_larger_than_the_legacy_one():
    legacy, human = ArmTaskEnv(random_waypoints=3), _human()
    n_legacy, n_human = len(legacy.feasible_points()), len(human.feasible_points())
    assert n_human > 2 * n_legacy                        # 1850 vs 744 grid points
    S = np.asarray(human.shoulder_base_position)
    r = np.hypot(*(human.feasible_points() - S).T)
    # Out to (almost) full reach, which the legacy 18 cm ring forbade ...
    assert r.max() > human.reach - 0.01
    assert np.hypot(*(legacy.feasible_points() - S).T).max() < human.reach - 0.17
    # ... and in to what the elbow's range allows (140 deg -> 0.23 m).
    assert r.min() < 0.24
    # Behind the body: the shoulder's 55 deg of usable extension.
    assert (human.feasible_points() - S)[:, 0].min() < -0.40


def test_human_limits_rule_keeps_every_joint_clear_of_its_hard_stop():
    env = _human()
    qmin = np.degrees(env.config.joint_limits_min)
    qmax = np.degrees(env.config.joint_limits_max)
    for p in env.feasible_points()[::25]:
        f = env.waypoint_feasibility(p, env.max_curriculum_tolerance)
        q = np.asarray(f["joint_angles_deg"])
        assert np.all(q - qmin >= 5.0 - 1e-6) and np.all(qmax - q >= 5.0 - 1e-6)
        assert max(f["torque_ratio"]) <= 0.8
    # A straight arm (elbow on its stop) is outside the human range, so full reach itself is not a target.
    S = np.asarray(env.shoulder_base_position)
    f = env.waypoint_feasibility((S + [env.reach - 1e-4, 0.0]).tolist(), env.max_curriculum_tolerance)
    assert not f["feasible"]


def test_human_limits_rules_round_trip_and_old_manifests_keep_theirs():
    env = _human()
    rebuilt = ArmTaskEnv.from_description(env.describe())
    assert (rebuilt.joint_margin_deg, rebuilt.reach_margin) == (5.0, 0.0)
    assert len(rebuilt.feasible_points()) == len(env.feasible_points())
    # A manifest written before the keys existed (every model up to R6) rebuilds with the legacy rules.
    old = ArmTaskEnv(random_waypoints=3).describe()
    old.pop("joint_margin_deg")
    old.pop("reach_margin")
    legacy = ArmTaskEnv.from_description(old)
    assert legacy.joint_margin_deg is None and legacy.reach_margin is None
    assert len(legacy.feasible_points()) == len(ArmTaskEnv(random_waypoints=3).feasible_points())
