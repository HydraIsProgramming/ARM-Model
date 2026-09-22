"""Environment contract: observation, reward, waypoints, hold, seeding (Phase 2)."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_armMotion.two_d.environments.task_env import ArmTaskEnv, OBS_VERSION
from rl_armMotion.two_d.evaluation import CANONICAL_WAYPOINTS, make_env
from helpers import OraclePolicy


def test_default_env_is_human_muscle_arm():
    env = ArmTaskEnv()
    assert env.preset_name == "2dof_human_right"
    assert env.actuation_mode == "muscle"
    assert env.action_space.shape == (4,)
    assert env.obs_dim == 14 and env.observation_space.shape == (14,)
    assert env.dt == pytest.approx(0.02)
    assert env.max_episode_steps == 500
    assert env.hold_decrement == 5 and env.hold_steps_required == 25


def test_velocity_mode_rejected():
    with pytest.raises(ValueError):
        ArmTaskEnv(actuation_mode="velocity")


def test_action_spaces_by_mode():
    assert make_env(actuation_mode="torque").action_space.shape == (2,)
    assert float(make_env(actuation_mode="torque").action_space.low.min()) == -1.0
    assert float(make_env(actuation_mode="muscle").action_space.low.min()) == 0.0


def test_gymnasium_check_env(muscle_env):
    check_env(muscle_env, skip_render_check=True)


def test_obs_within_declared_bounds_random_rollout(muscle_env):
    env = muscle_env
    lo, hi = env.observation_space.low, env.observation_space.high
    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        for _ in range(500):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
            assert np.isfinite(r)
            if term or trunc:
                break
        assert trunc and not term  # time limit is a truncation, never a termination


def test_obs_goal_error_tracks_goal(torque_env):
    env = torque_env
    obs, info = env.reset(seed=0)
    rel = (env.goal_position - info["end_effector_position"]) / env.reach
    assert np.allclose(obs[6:8], rel, atol=1e-5)
    assert obs[8] == pytest.approx(np.linalg.norm(rel), abs=1e-5)
    assert obs[11] == pytest.approx(0.0)  # waypoint progress at k=0
    env.set_goal_position([1.2, 0.2])
    obs2 = env.get_observation()
    assert not np.allclose(obs2[6:8], obs[6:8])


def test_reset_seed_reproducible_and_unseeded_varies():
    env = make_env(waypoints=2)
    a, _ = env.reset(seed=123)
    b, _ = env.reset(seed=123)
    c, _ = env.reset(seed=124)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    d, _ = env.reset(options={"randomize": False})
    e, _ = env.reset(options={"randomize": False})
    assert np.array_equal(d, e)


def test_zero_action_negative_return_no_advance(muscle_env):
    env = muscle_env
    env.reset(seed=0)
    total = 0.0
    for _ in range(500):
        _, r, term, trunc, info = env.step(np.zeros(4, dtype=np.float32))
        total += r
    assert total < -100.0
    assert info["current_waypoint_index"] == 0 and not info["success"]


def test_waypoint_spacing_assertion():
    env = ArmTaskEnv()
    with pytest.raises(ValueError):
        env.set_waypoints([[1.4, 0.1], [1.5, 0.1]])
    with pytest.raises(ValueError):
        env.set_waypoints([[1.0, -0.6]])  # too close to the start pose
    with pytest.raises(ValueError):
        env.set_goal_tolerance(0.5)  # above max_curriculum_tolerance


def test_oracle_completes_three_waypoints_and_reward_terms(torque_env):
    env = torque_env
    pol = OraclePolicy(env)
    obs, info = env.reset(seed=0)
    total = 0.0
    intermediate = 0
    for _ in range(500):
        a, _ = pol.predict(obs)
        obs, r, term, trunc, info = env.step(a)
        total += r
        intermediate += int(info["waypoint_advanced"])
        if term or trunc:
            break
    assert info["success"] and term
    assert intermediate == 2
    assert info["waypoints_reached"] == 3
    assert total > 30.0
    assert info["reward_terms"]["success"] == pytest.approx(env.R_SUCCESS)


def test_transition_step_consistent(torque_env):
    """On the advance step the reward is paid, the index increments and the
    returned observation already describes the NEW goal."""
    env = torque_env
    pol = OraclePolicy(env)
    obs, _ = env.reset(seed=0)
    for _ in range(500):
        a, _ = pol.predict(obs)
        obs, r, term, trunc, info = env.step(a)
        if info["waypoint_advanced"]:
            assert info["current_waypoint_index"] == 1
            assert info["reward_terms"]["intermediate"] == pytest.approx(env.R_INTERMEDIATE)
            rel = (np.asarray(env.waypoints[1]) - info["end_effector_position"]) / env.reach
            assert np.allclose(obs[6:8], rel, atol=1e-5)
            assert obs[11] == pytest.approx(0.5)
            break
    else:
        pytest.fail("oracle never advanced")


def test_potential_shaping_two_sided(torque_env):
    env = torque_env
    env.reset(seed=0, options={"randomize": False})
    # Full backward shoulder torque for 0.5 s swings the hand away from the
    # goal in front of the body: the shaping term must go negative.
    shaping = []
    for _ in range(25):
        _, _, _, _, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
        shaping.append(info["reward_terms"]["shaping"])
    assert min(shaping) < 0.0
    # Telescoping: sum of shaping over a trajectory = W * sum(gamma*phi_{t+1} - phi_t).
    pol = OraclePolicy(env)
    obs, info = env.reset(seed=1)
    phis = [-np.linalg.norm(env.goal_position - info["end_effector_position"]) / env.reach]
    total = 0.0
    for _ in range(40):
        a, _ = pol.predict(obs)
        obs, r, term, trunc, info = env.step(a)
        total += info["reward_terms"]["shaping"]
        phis.append(-info["goal_distance"] / env.reach)
        if info["waypoint_advanced"]:
            break
    expected = env.W_SHAPING * sum(env.discount * phis[i + 1] - phis[i] for i in range(len(phis) - 1))
    assert total == pytest.approx(expected, abs=1e-6)


def test_hold_counter_decrement_and_in_goal_stream_only_at_final(torque_env):
    env = torque_env
    env.set_terminate_on_success(False)
    pol = OraclePolicy(env)
    obs, _ = env.reset(seed=0)
    seen_stream_at_intermediate = False
    for _ in range(500):
        a, _ = pol.predict(obs)
        obs, r, term, trunc, info = env.step(a)
        if info["in_goal_region"] and info["current_waypoint_index"] < info["num_waypoints"] - 1:
            seen_stream_at_intermediate |= info["reward_terms"]["in_goal"] != 0.0
        if info["success"]:
            break
    assert not seen_stream_at_intermediate
    assert info["reward_terms"]["in_goal"] == pytest.approx(env.R_IN_GOAL)
    # Now knock the arm off the goal: the counter drops by hold_decrement per step.
    before = info["hold_counter"]
    _, _, _, _, info2 = env.step(np.array([1.0, 1.0], dtype=np.float32))
    _, _, _, _, info3 = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert info3["hold_counter"] <= max(0, before - 2 * env.hold_decrement) + env.hold_decrement


def test_hold_complete_bonus_fires_once_and_episode_continues(torque_env):
    env = torque_env
    env.set_terminate_on_success(False)
    pol = OraclePolicy(env)
    obs, _ = env.reset(seed=0)
    bonuses = 0
    for _ in range(500):
        a, _ = pol.predict(obs)
        obs, r, term, trunc, info = env.step(a)
        bonuses += int(info["reward_terms"]["success"] > 0)
        assert not term
        if trunc:
            break
    assert info["success"] and bonuses == 1
    assert info["hold_counter"] > env.hold_steps_required  # kept holding after success


def test_hovering_return_lower_than_holding_return(torque_env):
    """Cycling in/out of the goal to avoid completing the hold must earn less
    than simply staying still (the failure mode of a terminating design)."""
    env = torque_env
    env.set_terminate_on_success(False)
    env.set_waypoints(CANONICAL_WAYPOINTS[1])

    def run(offset_fn):
        pol = OraclePolicy(env)
        obs, _ = env.reset(seed=0)
        total = 0.0
        for t in range(500):
            pol.offset = offset_fn(t)
            a, _ = pol.predict(obs)
            obs, r, term, trunc, info = env.step(a)
            total += r
        return total

    hold = run(lambda t: (0.0, 0.0))
    hover = run(lambda t: (0.0, 0.0) if (t % 30) < 24 else (0.12, 0.0))
    assert hold > hover


def test_terminate_on_success_option_ends_episode():
    env = make_env(actuation_mode="torque", waypoints=1, terminate_on_success=True)
    pol = OraclePolicy(env)
    obs, _ = env.reset(seed=0)
    for _ in range(500):
        a, _ = pol.predict(obs)
        obs, r, term, trunc, info = env.step(a)
        if term:
            break
    assert term and info["success"] and info["success_step"] == info["step"]


def test_intermediate_hold_required(torque_env):
    env = torque_env
    env.set_intermediate_hold_steps(0)
    pol = OraclePolicy(env)
    obs, _ = env.reset(seed=0)
    steps_touch = None
    for t in range(500):
        a, _ = pol.predict(obs)
        obs, r, term, trunc, info = env.step(a)
        if info["waypoint_advanced"]:
            steps_touch = t
            break
    env.set_intermediate_hold_steps(5)
    obs, _ = env.reset(seed=0)
    steps_settle = None
    for t in range(500):
        a, _ = pol.predict(obs)
        obs, r, term, trunc, info = env.step(a)
        if info["waypoint_advanced"]:
            steps_settle = t
            break
    assert steps_touch is not None and steps_settle is not None
    assert steps_settle >= steps_touch + 4


def test_hold_satisfiable_at_rom_limit():
    """An arm resting on a joint stop reports zero velocity, so a goal placed
    at the fully-extended reach is holdable."""
    env = make_env(actuation_mode="torque", waypoints=None, goal_direction="EAST", terminate_on_success=True)
    pol = OraclePolicy(env)
    obs, _ = env.reset(seed=0)
    for _ in range(500):
        a, _ = pol.predict(obs)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    assert info["success"]


def test_describe_roundtrips_and_rejects_wrong_obs_version(muscle_env):
    desc = muscle_env.describe()
    env2 = ArmTaskEnv.from_description(desc)
    assert env2.obs_dim == muscle_env.obs_dim
    assert env2.waypoints is not None and len(env2.waypoints) == 2
    assert env2.goal_tolerance == muscle_env.goal_tolerance
    bad = dict(desc, obs_version=OBS_VERSION + 1)
    with pytest.raises(ValueError):
        ArmTaskEnv.from_description(bad)


def test_max_and_min_actions_stay_finite(muscle_env):
    env = muscle_env
    for action in (np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32)):
        obs, _ = env.reset(seed=0)
        for _ in range(500):
            obs, r, term, trunc, info = env.step(action)
            assert np.all(np.isfinite(obs)) and np.isfinite(r)


def test_sensory_delay_shifts_obs_and_noise_is_seeded():
    env = make_env(actuation_mode="torque", waypoints=2, sensory_delay_steps=2)
    assert env.obs_dim == 14 + 2 * 2
    obs0, _ = env.reset(seed=0)
    obs1, *_ = env.step(np.array([1.0, 0.0], dtype=np.float32))
    obs2, *_ = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert np.allclose(obs2[:14], obs0[:14])  # two steps late
    noisy = make_env(actuation_mode="torque", waypoints=2, obs_noise_std=(0.01, 0.05))
    a, _ = noisy.reset(seed=5)
    b, _ = noisy.reset(seed=5)
    c, _ = noisy.reset(seed=6)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_hand_load_changes_dynamics_config():
    env = make_env(actuation_mode="torque", waypoints=2)
    env.reset(seed=0, options={"hand_load_kg": 1.0})
    assert env.hand_load_kg == pytest.approx(1.0)
    assert env._cfg_dyn.masses[1] == pytest.approx(env.config.masses[1] + 1.0)
    assert env._cfg_dyn.com_offsets[1] > env.config.com_offsets[1]


def test_activation_dynamics_lags_command_and_extends_obs():
    env = make_env(actuation_mode="muscle", waypoints=2, activation_time_constants=(0.04, 0.06))
    assert env.obs_dim == 18 and env.observation_space.shape == (18,)
    obs, _ = env.reset(seed=0, options={"randomize": False})
    assert np.allclose(obs[14:], 0.0)
    obs, _, _, _, info = env.step(np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))
    act = info["muscle_activations"]
    assert 0.3 < act[0] < 0.7 and act[1] == 0.0          # rose half-way in one 20 ms step (tau 40 ms)
    assert np.allclose(obs[14:], act, atol=1e-6)
    for _ in range(20):
        obs, _, _, _, info = env.step(np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))
    assert info["muscle_activations"][0] > 0.99
    for _ in range(3):
        obs, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
    assert 0.0 < info["muscle_activations"][0] < 1.0     # decays, does not drop to zero at once
    desc = env.describe()
    env2 = ArmTaskEnv.from_description(desc)
    assert env2.obs_dim == 18 and env2.config.activation_time_constants == [0.04, 0.06]


def test_reward_weight_override_roundtrips_and_accel_term():
    env = make_env(actuation_mode="torque", waypoints=2, reward_weights={"W_ACCEL": 0.02})
    assert env.W_ACCEL == 0.02 and env.describe()["reward_weights"]["W_ACCEL"] == 0.02
    env2 = ArmTaskEnv.from_description(env.describe())
    assert env2.W_ACCEL == 0.02
    env.reset(seed=0, options={"randomize": False})
    _, _, _, _, info = env.step(np.array([1.0, 1.0], dtype=np.float32))   # hard jerk from rest
    assert info["reward_terms"]["hand_accel"] < 0.0
    with pytest.raises(ValueError):
        make_env(actuation_mode="torque", reward_weights={"W_BOGUS": 1.0})
    assert make_env(actuation_mode="torque").W_ACCEL == 0.0                 # default off


def test_end_range_engagement_penalised_before_hard_stop():
    env = make_env(actuation_mode="torque", waypoints=2)
    env.reset(seed=0, options={"randomize": False})
    q_straight = np.array([-np.pi / 2, np.deg2rad(3.0)])        # elbow 3 deg: inside the 10 deg spring margin
    eng = env._end_range_engagement(q_straight)
    assert 0.6 < eng[1] < 0.8 and eng[0] == 0.0
    env.reset(seed=0, options={"state": (q_straight, np.zeros(2))})
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["reward_terms"]["joint_limit"] < 0.0
    assert not info["at_joint_limit"][1]
    mid = env.config.joint_mid_angles
    assert np.all(env._end_range_engagement(mid) == 0.0)


def test_random_waypoints_sampled_per_episode_and_recorded():
    env = make_env(actuation_mode="torque", random_waypoints=3)
    pts = env.feasible_points()
    assert pts.shape[0] > 500
    for p in pts[::97]:
        assert env.waypoint_feasibility(p, env.max_curriculum_tolerance)["feasible"]
    o1, _ = env.reset(seed=1); w1 = [w.tolist() for w in env.waypoints]
    o2, _ = env.reset(seed=1); w2 = [w.tolist() for w in env.waypoints]
    o3, _ = env.reset(seed=2); w3 = [w.tolist() for w in env.waypoints]
    assert w1 == w2 and w1 != w3 and len(w3) == 3
    env._validate_waypoint_spacing([np.asarray(w) for w in w3])       # spacing rules hold
    desc = env.describe()
    assert desc["random_waypoints"] == 3
    again = ArmTaskEnv.from_description(desc)
    assert again.random_waypoints == 3
    pinned = ArmTaskEnv.from_description(desc, waypoints=CANONICAL_WAYPOINTS[3])
    assert pinned.random_waypoints is None and len(pinned.waypoints) == 3
    pinned.reset(seed=0)
    assert [w.tolist() for w in pinned.waypoints] == CANONICAL_WAYPOINTS[3]
    # a one-off override at reset does not change the sampler
    env.reset(seed=3, options={"waypoints": CANONICAL_WAYPOINTS[2]})
    assert len(env.waypoints) == 2
    env.reset(seed=4)
    assert len(env.waypoints) == 3
