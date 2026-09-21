"""Seeding contract, manifests, packages, continuation (Phase 0). Slow: trains SAC."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_armMotion.two_d.evaluation import PACKAGE_FILES, evaluate_policy, load_package, make_env
from rl_armMotion.two_d.training.ppo_trainer_wrapper import RLTrainerWithMetrics

pytestmark = pytest.mark.slow

HP = {"learning_starts": 200, "policy_kwargs": {"net_arch": [16, 16]}, "batch_size": 32}


def _train(run_dir, seed, steps=800, continue_from=None):
    torch.set_num_threads(1)
    env = make_env(waypoints=2, tolerance=0.15, hold_steps=10)
    tr = RLTrainerWithMetrics(env=env, total_timesteps=steps, algorithm="SAC", hyperparams=HP,
                              run_dir=str(run_dir), seed=seed, eval_freq=400, checkpoint_every=400,
                              continue_from=continue_from, cli_args={"test": True})
    tr.train()
    tr.save_model_and_results(str(run_dir))
    return tr


def test_same_seed_same_weights_different_seed_differs(tmp_path):
    a = _train(tmp_path / "a", 7)
    b = _train(tmp_path / "b", 7)
    c = _train(tmp_path / "c", 8)
    sa, sb, sc = (t.trainer.model.policy.state_dict() for t in (a, b, c))
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(not torch.equal(sa[k], sc[k]) for k in sa)


def test_manifest_and_package_roundtrip(tmp_path):
    _train(tmp_path / "run", 3)
    names = {p.name for p in (tmp_path / "run").iterdir()}
    for key in ("manifest", "best", "last", "final", "replay_buffer", "curriculum", "best_eval", "checkpoints"):
        assert PACKAGE_FILES[key] in names
    man = json.loads((tmp_path / "run" / PACKAGE_FILES["manifest"]).read_text())
    for key in ("seed", "versions", "env", "hyperparameters", "eval_standard", "curriculum", "lineage", "git_commit"):
        assert key in man
    assert man["seed"] == 3 and man["versions"]["torch_num_threads"] == "1"
    assert man["hyperparameters"]["buffer_size"] == 300_000
    model, env, m = load_package(tmp_path / "run", which="best", terminate_on_success=True)
    assert env.obs_dim == man["env"]["obs_dim"]
    # The manifest holds the training-START difficulty; a reloaded model is
    # measured at the production standard unless asked otherwise.
    assert man["env"]["goal_tolerance"] == pytest.approx(0.15) and man["env"]["hold_steps_required"] == 10
    assert env.goal_tolerance == pytest.approx(0.05) and env.hold_steps_required == 25
    raw = load_package(tmp_path / "run", which="best", standard=None)[1]
    assert raw.goal_tolerance == pytest.approx(0.15)
    res = evaluate_policy(model, env, n_episodes=2, seed=0)
    assert res.n == 2


def test_load_package_rejects_obs_version_mismatch(tmp_path):
    _train(tmp_path / "run", 4)
    mp = tmp_path / "run" / PACKAGE_FILES["manifest"]
    man = json.loads(mp.read_text())
    man["env"]["obs_version"] = 99
    mp.write_text(json.dumps(man))
    with pytest.raises(ValueError):
        load_package(tmp_path / "run")


def test_continue_from_resumes_and_leaves_source_untouched(tmp_path):
    _train(tmp_path / "src", 5, steps=800)
    before = {p.name: p.stat().st_mtime for p in (tmp_path / "src").iterdir()}
    tr = _train(tmp_path / "cont", 5, steps=400, continue_from=tmp_path / "src")
    assert tr.trainer.model.num_timesteps == 1200
    assert tr.lineage[-1]["parent_timesteps"] == 800
    assert tr.trainer.model.learning_starts == 0
    man = json.loads((tmp_path / "cont" / PACKAGE_FILES["manifest"]).read_text())
    assert man["lineage"][0]["parent_run_dir"].endswith("src")
    after = {p.name: p.stat().st_mtime for p in (tmp_path / "src").iterdir()}
    assert before == after


def test_cli_refuses_to_overwrite(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import train_fischer_session as cli
    (tmp_path / "sac_model_best.zip").write_bytes(b"x")
    rc = cli.main(["--timesteps", "10", "--save-dir", str(tmp_path), "--skip-validators"])
    assert rc == 2
