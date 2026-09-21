"""Validators run in-distribution; the demo script draws what the env reports."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from rl_armMotion.two_d.evaluation import make_env
from rl_armMotion.two_d.validation import FittsLawValidator, PowerLawValidator
from rl_armMotion.two_d.utils.arm_kinematics import ArmKinematics
from helpers import OraclePolicy


def test_fitts_goals_all_feasible_and_obs_refreshed():
    env = make_env(actuation_mode="torque", waypoints=None, terminate_on_success=True)
    pol = OraclePolicy(env)
    fl = FittsLawValidator(model=pol, env=env)
    res = fl.run(distances=(0.15, 0.30), widths=(0.05, 0.10), n_trials_per_condition=3,
                 max_steps_per_trial=150, seed=1)
    ran = [t for t in res.trials if t.steps_taken > 0]
    assert ran, "no trial ran"
    for t in ran:
        assert t.success, t
    assert np.isfinite(res.slope)


def test_power_law_runs_on_human_arm():
    env = make_env(actuation_mode="torque", waypoints=None, terminate_on_success=True)
    pol = OraclePolicy(env)
    pl = PowerLawValidator(model=pol, env=env)
    res = pl.run(n_trials=4, max_steps_per_trial=150, seed=2)
    assert res.n_successful_trials >= 3


def test_show_waypoints_frames_match_fk(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import show_waypoints as sw
    env = make_env(actuation_mode="torque", waypoints=2, terminate_on_success=True)
    pol = OraclePolicy(env)
    frames, info = sw.run_episode(pol, env, seed=0)
    assert info["success"]
    for f in frames[::20]:
        assert f["links"].shape == (3, 2)
        assert np.allclose(f["links"][-1], f["hand"])
    last = frames[-1]
    assert last["reached"] == 2
    gif = tmp_path / "demo.gif"
    sw.animate([frames[:60]], env, "test", fps=10, gif=str(gif), show=False)
    assert gif.exists() and gif.stat().st_size > 0
