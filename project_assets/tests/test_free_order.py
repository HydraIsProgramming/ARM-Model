"""Free-order waypoint task: the policy sees every point and picks its own route.

The ordered task hands the policy one active goal at a time and an index saying
how far through the list it is. That index is a guide rail: the route is the
environment's choice, not the policy's. In free-order mode the observation
carries every waypoint and a visited flag for each, nothing says which to go to,
and any outstanding waypoint can be visited at any time.
"""

import itertools

import numpy as np
import pytest

from rl_armMotion.two_d.environments.task_env import (ArmTaskEnv, OBS_VERSION,
                                                      OBS_VERSION_FREE_ORDER)
from helpers import OraclePolicy, RouteOraclePolicy

WPS = [[1.40, 0.15], [1.20, 0.40], [1.38, -0.20]]


def make(**kw):
    kw.setdefault("actuation_mode", "torque")
    kw.setdefault("goal_tolerance", 0.05)
    kw.setdefault("hold_steps_required", 25)
    kw.setdefault("terminate_on_success", True)
    env = ArmTaskEnv(free_order=True, free_order_slots=len(WPS), **kw)
    env.set_waypoints(WPS)
    return env


def run(env, policy, seed=0, max_steps=900):
    obs, info = env.reset(seed=seed)
    for _ in range(max_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return info


# --------------------------------------------------------------------------- #
# Observation: every point visible, no ordering signal
# --------------------------------------------------------------------------- #

def test_observation_carries_every_waypoint_and_no_active_goal():
    env = make()
    n = env.num_dof
    assert env.obs_dim == 3 * n + 2 + 4 * len(WPS) + 2
    assert env.obs_version == OBS_VERSION_FREE_ORDER != OBS_VERSION
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    ee = env._end_effector_from_angles(env.q)
    base = 3 * n + 2
    for k, wp in enumerate(WPS):
        block = obs[base + 4 * k: base + 4 * k + 4]
        expected = (np.asarray(wp) - ee) / env.reach
        assert block[:2] == pytest.approx(expected, abs=1e-5)
        assert block[2] == pytest.approx(np.linalg.norm(expected), abs=1e-5)
        assert block[3] == 0.0                      # nothing visited yet


def test_observation_does_not_change_when_only_the_internal_target_changes():
    """The bookkeeping target must not leak into the observation - if it did,
    the env would still be telling the policy where to go."""
    env = make()
    obs_a, _ = env.reset(seed=3)
    env.current_waypoint_index = 2
    env._set_goal(env.waypoints[2])
    obs_b = env.get_observation()
    assert np.array_equal(obs_a, obs_b)


def test_visited_flag_is_the_only_thing_that_changes_per_waypoint():
    env = make()
    obs, _ = env.reset(seed=4)
    base = 3 * env.num_dof + 2
    env.visited[1] = True
    after = env.get_observation()
    changed = np.flatnonzero(~np.isclose(obs, after))
    assert changed.tolist() == [base + 4 * 1 + 3]


def test_observation_bounds_hold_over_random_rollouts():
    env = make(actuation_mode="muscle", activation_time_constants=(0.04, 0.06))   # muscle-only feature
    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        for _ in range(300):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
            assert np.isfinite(reward)
            if terminated or truncated:
                break


# --------------------------------------------------------------------------- #
# The route is the policy's choice
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("route", [(0, 1, 2), (2, 1, 0), (1, 2, 0), (2, 0, 1)])
def test_any_order_completes_the_task(route):
    env = make()
    info = run(env, RouteOraclePolicy(env, route))
    assert info["success"], f"route {route} failed: {info['visited']}"
    assert all(info["visited"])
    assert info["waypoints_reached"] == 3


def test_order_of_visits_follows_the_policy_not_the_list():
    env = make()
    order = []
    obs, _ = env.reset(seed=0)
    policy = RouteOraclePolicy(env, (2, 0, 1))
    for _ in range(900):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        if info["waypoint_advanced"]:
            order.append(int(info["waypoint_completed"]))
        if terminated or truncated:
            break
    assert order == [2, 0, 1], "the policy's route, not the list order"
    assert info["success"]


def test_success_needs_every_waypoint():
    env = make()
    info = run(env, RouteOraclePolicy(env, (0, 1)), max_steps=600)
    assert info["visited"] == [True, True, False]
    assert not info["success"]
    assert info["waypoints_reached"] == 2


def test_last_waypoint_needs_the_full_hold():
    """An intermediate point needs only the settle; the one the policy leaves
    until last needs the whole hold, whichever it turns out to be."""
    env = make(hold_steps_required=25, intermediate_hold_steps=5)
    obs, _ = env.reset(seed=0)
    policy = RouteOraclePolicy(env, (1, 2, 0))
    holds = {}
    counter = 0
    for _ in range(900):
        action, _ = policy.predict(obs, deterministic=True)
        prev = int(env.hold_counter)
        obs, _, terminated, truncated, info = env.step(action)
        counter = prev
        if info["waypoint_advanced"]:
            holds[int(info["waypoint_completed"])] = counter + 1
        if terminated or truncated:
            break
    assert info["success"]
    assert holds[1] <= 6 and holds[2] <= 6        # settle only
    assert holds[0] >= 25                          # the last one held in full


def test_hold_credit_does_not_transfer_between_waypoints():
    env = make()
    env.reset(seed=0)
    env.q = env.solve_ik(np.asarray(WPS[0]))
    env.qd = np.zeros(env.num_dof)
    env._sync_state()
    env._refresh_free_order_target()
    env._entered_from_outside = True
    env.hold_counter = 20
    assert env.current_waypoint_index == 0
    # Teleport the hand to another waypoint: the partial hold must not count there.
    env.q = env.solve_ik(np.asarray(WPS[2]))
    env.qd = np.zeros(env.num_dof)
    env._sync_state()
    env.step(np.zeros(env.action_dim))
    assert env.current_waypoint_index == 2
    assert env.hold_counter <= 1
    assert not env.visited[2]


# --------------------------------------------------------------------------- #
# Shaping potential
# --------------------------------------------------------------------------- #

def test_tour_length_is_the_shortest_route():
    env = make()
    env.reset(seed=0)
    start = np.array([1.0, -0.4])
    pts = [np.asarray(w) for w in WPS]
    brute = min(
        sum(float(np.linalg.norm(p - q)) for p, q in zip([start] + [pts[i] for i in perm], [pts[i] for i in perm]))
        for perm in itertools.permutations(range(3)))
    assert env._tour_length(start, [0, 1, 2]) == pytest.approx(brute, rel=1e-9)


def test_potential_depends_only_on_state():
    """Same hand position and same visited set - same potential, whatever the
    history. That is what keeps potential-based shaping policy-invariant."""
    env = make()
    env.reset(seed=0)
    ee = np.array([1.1, 0.1])
    env.visited[:] = [False, True, False]
    a = env._free_order_potential(ee)
    env.reset(seed=7)
    env.visited[:] = [False, True, False]
    b = env._free_order_potential(ee)
    assert a == pytest.approx(b)


def test_potential_rises_as_waypoints_are_completed():
    env = make()
    env.reset(seed=0)
    ee = np.array([1.0, -0.4])
    env.visited[:] = [False, False, False]
    none_done = env._free_order_potential(ee)
    env.visited[:] = [True, False, False]
    one_done = env._free_order_potential(ee)
    env.visited[:] = [True, True, False]
    two_done = env._free_order_potential(ee)
    assert none_done < one_done < two_done < 0.0


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #

def test_describe_roundtrip_keeps_free_order():
    env = make(actuation_mode="muscle", activation_time_constants=(0.04, 0.06))   # muscle-only feature
    desc = env.describe()
    assert desc["free_order"] and desc["free_order_slots"] == 3
    assert desc["obs_version"] == OBS_VERSION_FREE_ORDER
    rebuilt = ArmTaskEnv.from_description(desc)
    assert rebuilt.free_order and rebuilt.obs_dim == env.obs_dim


def test_ordered_description_is_rejected_by_a_free_order_env_and_vice_versa():
    free_desc = make().describe()
    free_desc["free_order"] = False
    with pytest.raises(ValueError, match="obs_version"):
        ArmTaskEnv.from_description(free_desc)

    ordered = ArmTaskEnv(actuation_mode="torque")
    ordered.set_waypoints(WPS)
    ordered_desc = ordered.describe()
    ordered_desc["free_order"] = True
    with pytest.raises(ValueError, match="obs_version"):
        ArmTaskEnv.from_description(ordered_desc)


def test_wrong_number_of_waypoints_is_refused():
    env = make()
    with pytest.raises(ValueError, match="waypoint slots"):
        env.set_waypoints(WPS[:2])


def test_ordered_task_is_untouched():
    """The ordered env must behave exactly as before: same layout, same version."""
    env = ArmTaskEnv(actuation_mode="torque", goal_tolerance=0.05, terminate_on_success=True)
    env.set_waypoints(WPS)
    assert env.obs_version == OBS_VERSION
    assert env.obs_dim == 3 * env.num_dof + 8
    assert not env.free_order
    info = run(env, OraclePolicy(env))
    assert info["success"] and info["waypoints_reached"] == 3


def test_completing_a_waypoint_does_not_cut_the_per_step_reward():
    """The regression that killed the first free-order run.

    With the dense term keyed to the nearest outstanding waypoint, finishing one
    made that term jump from ~0 to -0.25 and worse, so the step after a
    completion paid far less than the step before it and the policy learned to
    park on the first point it reached and never leave. Keyed to the remaining
    tour, completing a waypoint leaves the term where it was.
    """
    env = make()
    policy = RouteOraclePolicy(env, (0, 1, 2))
    obs, _ = env.reset(seed=0)
    prev_terms = None
    checked = 0
    for _ in range(900):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        terms = dict(env._last_reward_terms)
        if info["waypoint_completed"] is not None and prev_terms is not None:
            # the one-off bonus aside, the recurring part must not fall off a cliff
            before = prev_terms["distance"] + prev_terms["shaping"]
            after = terms["distance"] + terms["shaping"]
            assert after > before - 0.05, (
                f"completing waypoint {info['waypoint_completed']} cut the recurring reward "
                f"from {before:.3f} to {after:.3f}")
            checked += 1
        prev_terms = terms
        if terminated or truncated:
            break
    assert checked >= 2, "expected at least two completions to check"


def test_dense_term_improves_monotonically_as_waypoints_are_completed():
    env = make()
    env.reset(seed=0)
    ee = np.array([1.30, 0.05])
    values = []
    for visited in ([False, False, False], [True, False, False], [True, True, False]):
        env.visited[:] = visited
        values.append(env.W_DIST * env._free_order_potential(ee))
    assert values[0] < values[1] < values[2] <= 0.0
