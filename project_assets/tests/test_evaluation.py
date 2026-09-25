"""The shared evaluator, packages and the trainer defaults merge (Phase 0)."""

import json

import numpy as np
import pytest

from rl_armMotion.two_d.evaluation import (
    CANONICAL_WAYPOINTS, EVAL_STANDARD, EvalResult, EpisodeResult, detect_actuation_mode, evaluate_policy,
    make_env, resolve_waypoints,
)
from helpers import OraclePolicy


def test_resolve_waypoints_forms():
    assert resolve_waypoints(2) == CANONICAL_WAYPOINTS[2]
    assert resolve_waypoints("3") == CANONICAL_WAYPOINTS[3]
    assert resolve_waypoints("1.0,2.0;3.0,4.0") == [[1.0, 2.0], [3.0, 4.0]]
    assert resolve_waypoints(None) is None and resolve_waypoints("none") is None
    assert len(resolve_waypoints("spread")) == 3


def test_make_env_applies_standard():
    env = make_env(actuation_mode="torque", waypoints=3)
    assert env.goal_tolerance == EVAL_STANDARD.tolerance
    assert env.hold_steps_required == EVAL_STANDARD.hold_steps
    assert env.hold_decrement == EVAL_STANDARD.hold_decrement
    assert env.max_episode_steps == EVAL_STANDARD.max_steps
    assert len(env.waypoints) == 3


def test_detect_actuation_mode():
    assert detect_actuation_mode(make_env(actuation_mode="torque")) == "torque"
    assert detect_actuation_mode(make_env(actuation_mode="muscle")) == "muscle"


def test_evaluate_policy_reproducible_and_episodes_distinct(torque_env):
    pol = OraclePolicy(torque_env)
    r1 = evaluate_policy(pol, torque_env, n_episodes=3, seed=11)
    r2 = evaluate_policy(pol, torque_env, n_episodes=3, seed=11)
    assert [e.success_step for e in r1.episodes] == [e.success_step for e in r2.episodes]
    assert [e.episode_return for e in r1.episodes] == [e.episode_return for e in r2.episodes]
    assert len({e.seed for e in r1.episodes}) == 3
    assert r1.success_rate == 1.0
    assert all(e.waypoints_reached == 3 for e in r1.episodes)
    assert r1.mean_steps_to_success < 300
    hl = r1.human_likeness_summary()
    assert np.isfinite(hl["straightness"]) and hl["straightness"] >= 1.0


def test_waypoints_reached_counts_final_hold(torque_env):
    partial = OraclePolicy(torque_env, offset=(0.09, 0.0))   # never inside the 5 cm tolerance
    res = evaluate_policy(partial, torque_env, n_episodes=2, seed=3)
    assert res.success_rate == 0.0
    assert all(e.waypoints_reached == 0 for e in res.episodes)
    assert res.mean_final_distance < 0.2


def test_score_orders_success_over_partials():
    def mk(success, wp, hold, dist, steps):
        return EvalResult([EpisodeResult(seed=0, success=success, success_step=steps if success else None,
                                         waypoints_reached=wp, num_waypoints=3, max_hold_progress=hold,
                                         final_distance=dist, min_final_distance=dist, steps=500,
                                         episode_return=0.0)])
    assert mk(True, 3, 1.0, 0.01, 200).score() > mk(False, 2, 0.9, 0.01, 0).score()
    assert mk(False, 2, 0.5, 0.05, 0).score() > mk(False, 2, 0.2, 0.01, 0).score()
    assert mk(False, 2, 0.2, 0.01, 0).score() > mk(False, 2, 0.2, 0.10, 0).score()
    assert mk(True, 3, 1.0, 0.01, 150).score() > mk(True, 3, 1.0, 0.01, 200).score()


def test_result_save_and_trajectories(tmp_path, torque_env):
    pol = OraclePolicy(torque_env)
    res = evaluate_policy(pol, torque_env, n_episodes=1, seed=0, log_trajectories=True)
    out = res.save(tmp_path / "eval.json", trajectories=True)
    data = json.loads(out.read_text())
    assert data["summary"]["successes"] == 1
    assert (tmp_path / "eval.trajectories.npz").exists()
    arr = np.load(tmp_path / "eval.trajectories.npz")
    assert arr["ep0_hand"].shape[1] == 2 and arr["ep0_q"].shape[1] == 2


def test_project_sac_defaults_merge_with_overrides():
    from rl_armMotion.two_d.models.trainers import RLTrainer
    env = make_env(actuation_mode="torque", waypoints=2)
    tr = RLTrainer(env=env, algorithm="SAC", hyperparams={"batch_size": 64, "policy_kwargs": {"net_arch": [16, 16]}})
    assert tr.model.buffer_size == 300_000
    assert tr.model.learning_starts == 5_000
    assert tr.model.batch_size == 64
    assert tr.model.gamma == pytest.approx(0.99)
    rh = tr.resolved_hyperparams
    assert rh["net_arch"] == [16, 16] and rh["buffer_size"] == 300_000


def test_rebuilt_env_uses_the_reward_the_model_was_trained_with():
    """A reward key missing from an old manifest means the term did not exist
    yet, so it must be OFF - not today's default.

    Every stage before R6 predates at least one reward key. Rebuilding the
    deployed R3 model used to switch on a linear speed charge it never trained
    against, which scored one of its episodes -42 for speed.
    """
    import math
    from rl_armMotion.two_d.environments.task_env import ArmTaskEnv

    env = ArmTaskEnv(actuation_mode="torque")
    desc = env.describe()
    # an R3-era manifest: speed penalty recorded, the newer keys absent
    desc["reward_weights"] = {k: v for k, v in desc["reward_weights"].items()
                              if k not in ("W_SPEED_LINEAR", "SPEED_PENALTY_CAP")}
    desc["reward_weights"].update({"W_SPEED_EXCESS": 0.5, "EE_SPEED_CAP": 1.5})
    old = ArmTaskEnv.from_description(desc)
    assert old.W_SPEED_LINEAR == 0.0
    assert math.isinf(old.SPEED_PENALTY_CAP)
    assert old.W_SPEED_EXCESS == 0.5 and old.EE_SPEED_CAP == 1.5

    # a Stage-A-era manifest: no speed or acceleration penalty existed at all
    desc["reward_weights"] = {k: v for k, v in desc["reward_weights"].items()
                              if k not in ("W_SPEED_EXCESS", "EE_SPEED_CAP", "W_ACCEL",
                                           "ACCEL_REF", "ACCEL_PENALTY_CAP")}
    stage_a = ArmTaskEnv.from_description(desc)
    assert stage_a.W_SPEED_EXCESS == 0.0 and stage_a.W_SPEED_LINEAR == 0.0 and stage_a.W_ACCEL == 0.0


def test_a_current_manifest_round_trips_every_reward_weight():
    """New manifests record every key, so nothing falls back to a legacy value."""
    from rl_armMotion.two_d.environments.task_env import ArmTaskEnv
    env = ArmTaskEnv(actuation_mode="torque")
    rebuilt = ArmTaskEnv.from_description(env.describe())
    for k in ArmTaskEnv.REWARD_WEIGHT_KEYS:
        assert getattr(rebuilt, k) == getattr(env, k), k
