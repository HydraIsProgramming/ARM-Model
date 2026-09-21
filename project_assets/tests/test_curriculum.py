"""Evaluation-driven curriculum and checkpoint selection (Phase 3)."""

import json

import numpy as np
import pytest

from rl_armMotion.two_d.evaluation import PACKAGE_FILES, make_env
from rl_armMotion.two_d.training.curriculum_callback import EvalCheckpointCurriculumCallback
from helpers import OraclePolicy


def _make_callback(tmp_path, env_fn, model, train_env, **kw):
    kw.setdefault("advance_consecutive", 1)
    cb = EvalCheckpointCurriculumCallback(run_dir=str(tmp_path), eval_env_fn=env_fn, eval_freq=100,
                                          verbose=0, save_replay_buffer=False, **kw)
    model.get_env = lambda: train_env   # BaseCallback.training_env reads model.get_env()
    cb.model = model
    cb.num_timesteps = 0
    cb._on_training_start()
    return cb


def test_advances_on_training_eval_not_production(tmp_path):
    train_env = make_env(actuation_mode="torque", waypoints=2, tolerance=0.15, hold_steps=10)
    eval_env = make_env(actuation_mode="torque", waypoints=2, terminate_on_success=True)
    # Succeeds only when the tolerance is >= 0.10 m (hand parks 8 cm off target).
    model = OraclePolicy(eval_env, offset=(0.08, 0.0))
    cb = _make_callback(tmp_path, lambda: eval_env, model, train_env,
                        tolerance_ladder=(0.15, 0.10, 0.07, 0.05), hold_ladder=(10, 25))
    assert train_env.goal_tolerance == pytest.approx(0.15)
    for step in range(100, 900, 100):
        cb.num_timesteps = step
        cb._run_round()
    # 0.15 -> 0.10 (round 1), cooldown, 0.10 -> 0.07 (round 3), then the
    # training eval fails at 0.07 forever: the ladder stalls there and the
    # hold ladder never moves.
    assert cb.current_tolerance == pytest.approx(0.07)
    assert train_env.goal_tolerance == pytest.approx(0.07)
    assert cb.hold_idx == 0
    assert cb.history[-1]["train_success_rate"] == 0.0
    assert cb.history[0]["train_success_rate"] == 1.0
    # Production success never happened, so the score's success component is 0.
    assert cb.best_score[0] == 0.0
    assert (tmp_path / PACKAGE_FILES["curriculum"]).exists()


def test_one_change_per_round_and_hold_after_tolerance(tmp_path):
    train_env = make_env(actuation_mode="torque", waypoints=2, tolerance=0.15, hold_steps=10)
    eval_env = make_env(actuation_mode="torque", waypoints=2, terminate_on_success=True)
    model = OraclePolicy(eval_env)   # perfect
    cb = _make_callback(tmp_path, lambda: eval_env, model, train_env,
                        tolerance_ladder=(0.15, 0.05), hold_ladder=(10, 25))
    stages = []
    for step in range(100, 900, 100):
        cb.num_timesteps = step
        cb._run_round()
        stages.append((cb.tolerance_idx, cb.hold_idx))
    # Never two changes at once, tolerance before hold, cooldown between.
    for a, b in zip(stages, stages[1:]):
        assert (b[0] - a[0]) + (b[1] - a[1]) <= 1
    assert stages[-1] == (1, 1)
    assert cb.best_score[0] == 1.0


def test_best_checkpoint_uses_continuous_score(tmp_path):
    train_env = make_env(actuation_mode="torque", waypoints=2, tolerance=0.15, hold_steps=10)
    eval_env = make_env(actuation_mode="torque", waypoints=2, terminate_on_success=True)
    far = OraclePolicy(eval_env, offset=(0.20, 0.0))
    near = OraclePolicy(eval_env, offset=(0.08, 0.0))
    cb = _make_callback(tmp_path, lambda: eval_env, far, train_env)
    cb.num_timesteps = 100
    cb._run_round()
    s_far = list(cb.best_score)
    near.get_env = far.get_env
    cb.model = near
    cb.num_timesteps = 200
    cb._run_round()
    s_near = list(cb.best_score)
    assert s_near > s_far and cb.best_saved_at == 200


def test_state_persisted_and_resumed(tmp_path):
    train_env = make_env(actuation_mode="torque", waypoints=2, tolerance=0.15, hold_steps=10)
    eval_env = make_env(actuation_mode="torque", waypoints=2, terminate_on_success=True)
    model = OraclePolicy(eval_env)
    cb = _make_callback(tmp_path, lambda: eval_env, model, train_env, tolerance_ladder=(0.15, 0.05))
    cb.num_timesteps = 100
    cb._run_round()
    state = json.loads((tmp_path / PACKAGE_FILES["curriculum"]).read_text())
    assert state["tolerance_idx"] == 1 and state["best_score"][0] == 1.0
    train_env2 = make_env(actuation_mode="torque", waypoints=2, tolerance=0.15, hold_steps=10)
    cb2 = _make_callback(tmp_path / "resumed", lambda: eval_env, model, train_env2,
                         tolerance_ladder=(0.15, 0.05), resume_state=state)
    assert cb2.current_tolerance == pytest.approx(0.05)
    assert train_env2.goal_tolerance == pytest.approx(0.05)
    assert cb2.best_score == state["best_score"]


def test_advance_requires_consecutive_successes(tmp_path):
    train_env = make_env(actuation_mode="torque", waypoints=2, tolerance=0.15, hold_steps=10)
    eval_env = make_env(actuation_mode="torque", waypoints=2, terminate_on_success=True)
    model = OraclePolicy(eval_env)
    cb = _make_callback(tmp_path, lambda: eval_env, model, train_env,
                        tolerance_ladder=(0.15, 0.05), advance_consecutive=2, cooldown_rounds=1)
    cb.num_timesteps = 100
    cb._run_round()
    assert cb.tolerance_idx == 0 and cb.history[-1]["train_streak"] == 1   # one good round is not enough
    cb.num_timesteps = 200
    cb._run_round()
    assert cb.tolerance_idx == 1 and cb.history[-1]["advanced"]            # second good round advances
