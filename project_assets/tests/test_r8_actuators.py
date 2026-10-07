"""R8: torque actuators, the action-rate penalty, the previous-action input, T80 probes,
non-destructive promotion and TQC packages."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from rl_armMotion.two_d.environments.task_env import ArmTaskEnv

ROOT = Path(__file__).resolve().parents[2]


def _torque_env(**kw):
    kw.setdefault("actuation_mode", "torque")
    return ArmTaskEnv(**kw)


# --------------------------------------------------------------------------- env

def test_torque_actuators_have_no_activation_dynamics():
    """Muscle activation dynamics in torque mode used to add four always-zero inputs."""
    with pytest.raises(ValueError, match="activation"):
        _torque_env(activation_time_constants=(0.04, 0.06))
    env = _torque_env()
    assert env.obs_dim == 14 and env.action_dim == 2
    obs, info = env.reset(seed=0)
    obs, r, *_ , info = env.step(np.array([0.5, -0.5], dtype=np.float32))
    assert np.allclose(info["actuator_torque"], [0.5 * 25.0, -0.5 * 12.0])
    assert info["muscle_activations"] is None


def test_previous_action_input_is_the_last_command():
    env = _torque_env(prev_action_obs=True)
    assert env.obs_dim == 16
    obs, _ = env.reset(seed=1)
    assert np.allclose(obs[-2:], 0.0)                       # nothing commanded yet
    a = np.array([0.3, -0.7], dtype=np.float32)
    obs, *_ = env.step(a)
    assert np.allclose(obs[-2:], a, atol=1e-6)
    assert env.observation_space.contains(obs)
    rebuilt = ArmTaskEnv.from_description(env.describe())
    assert rebuilt.prev_action_obs and rebuilt.obs_dim == 16
    old = env.describe()
    old.pop("prev_action_obs")
    old["obs_dim"] = 14
    assert not ArmTaskEnv.from_description(old).prev_action_obs   # older manifests: off


def test_action_rate_penalty_charges_changes_only():
    env = _torque_env(reward_weights={"W_ACTION_RATE": 0.02})
    env.reset(seed=2)
    a = np.array([0.4, 0.4], dtype=np.float32)
    env.step(a)                                              # first step pays for leaving zero
    env.step(a)
    assert env._last_reward_terms["action_rate"] == pytest.approx(0.0)
    env.step(-a)                                             # a full flip of both commands
    assert env._last_reward_terms["action_rate"] == pytest.approx(-0.02 * float(np.sum((2 * a) ** 2)))
    assert ArmTaskEnv().W_ACTION_RATE == 0.0                 # off unless a stage sets it


def test_old_manifests_rebuild_with_the_action_rate_penalty_off():
    desc = _torque_env(reward_weights={"W_ACTION_RATE": 0.02}).describe()
    desc["reward_weights"].pop("W_ACTION_RATE")
    assert ArmTaskEnv.from_description(desc).W_ACTION_RATE == 0.0


def test_accel_penalty_saturates_at_10_with_accel_ref_5():
    """The user's change: same maximum (0.08 per step at W_ACCEL 0.02), reached at 10 m/s^2 instead of 20."""
    for ref, sat in ((10.0, 20.0), (5.0, 10.0)):
        env = _torque_env(reward_weights={"W_ACCEL": 0.02, "ACCEL_REF": ref})
        assert ref * math.sqrt(env.ACCEL_PENALTY_CAP) == pytest.approx(sat)
        assert 0.02 * env.ACCEL_PENALTY_CAP == pytest.approx(0.08)


# --------------------------------------------------------------------------- campaign

def test_r8_stages_build_torque_envs_with_r7s_task():
    import sys
    sys.path.insert(0, str(ROOT))
    from run_campaign import STAGES
    for stage in ("R8", "R8E", "R8N", "R8U", "R8T"):
        extra = STAGES[stage]["extra"]
        assert extra[extra.index("--actuation-mode") + 1] == "torque"
        assert "--activation-dynamics" not in extra and "--prev-action-obs" in extra
        assert extra[extra.index("--joint-margin-deg") + 1] == "5"
    r7 = [x for x in STAGES["R7"]["extra"] if x.startswith(("W_", "EE_", "SPEED_"))]
    r8 = [x for x in STAGES["R8"]["extra"] if x.startswith(("W_", "EE_", "SPEED_"))]
    assert r7 == r8                                           # same reward as R7


def test_promotion_never_deletes_an_existing_champion(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(ROOT))
    import run_campaign
    monkeypatch.setattr(run_campaign, "ROOT", tmp_path)
    outputs = tmp_path / "project_assets" / "outputs"
    old = outputs / "CHAMPION_HUMAN_V1_ZZ"
    old.mkdir(parents=True)
    (old / "sac_model_best.zip").write_text("old")
    new = tmp_path / "run" / "seed_1"
    new.mkdir(parents=True)
    (new / "sac_model_best.zip").write_text("new")
    run_campaign.promote(new, "ZZ", force=False)              # refuses, keeps the old one
    assert (old / "sac_model_best.zip").read_text() == "old"
    run_campaign.promote(new, "ZZ", force=True)               # supersedes, still keeps the old one
    assert (old / "sac_model_best.zip").read_text() == "new"
    kept = list((outputs / "_superseded").glob("CHAMPION_HUMAN_V1_ZZ_*"))
    assert len(kept) == 1 and (kept[0] / "sac_model_best.zip").read_text() == "old"


# --------------------------------------------------------------------------- TQC and T80

@pytest.mark.slow
def test_tqc_package_round_trips_with_standard_file_names(tmp_path):
    pytest.importorskip("sb3_contrib")
    from rl_armMotion.two_d.evaluation import load_package, make_env
    from rl_armMotion.two_d.training.ppo_trainer_wrapper import RLTrainerWithMetrics
    env = make_env(actuation_mode="torque", waypoints=2, prev_action_obs=True)
    trainer = RLTrainerWithMetrics(env=env, total_timesteps=1200, algorithm="TQC", run_dir=str(tmp_path), seed=3,
                                   hyperparams={"learning_starts": 200, "batch_size": 64,
                                                "policy_kwargs": {"net_arch": [32, 32], "n_quantiles": 5,
                                                                  "n_critics": 2}},
                                   eval_freq=600, checkpoint_every=10_000,
                                   curriculum_kwargs={"probe_episodes": 2})
    trainer.train()
    trainer.save_model_and_results(str(tmp_path))
    assert (tmp_path / "sac_model_best.zip").exists()
    model, env2, manifest = load_package(tmp_path)
    assert type(model).__name__ == "TQC" and manifest["algorithm"] == "TQC"
    hist = json.loads((tmp_path / "curriculum_state.json").read_text())["history"]
    assert all("probe_success_rate" in h for h in hist)


def _write_checkpoint(d, steps):
    """The files a resume reads, intact (resume refuses zero-filled ones)."""
    import zipfile
    with zipfile.ZipFile(d / "sac_model_last.zip", "w") as z:
        z.writestr("data", "weights")
    (d / "replay_buffer.pkl").write_bytes(bytes([0x80, 0x05]) + b" buffer .")
    (d / "run_manifest.json").write_text("{}")
    (d / "curriculum_state.json").write_text(json.dumps({"num_timesteps": steps}))


def test_a_crashed_run_resumes_from_its_last_checkpoint(tmp_path, monkeypatch):
    """Native crashes (access violations) must not end a seed: the crashed folder is kept and the
    run continues from its last checkpoint with the remaining steps."""
    import sys
    sys.path.insert(0, str(ROOT))
    import run_campaign
    calls = []

    def fake_call(cmd, stdout=None, stderr=None, cwd=None):
        calls.append(list(cmd))
        d = Path(cmd[cmd.index("--save-dir") + 1])
        if len(calls) == 1:                                   # dies after its 100K checkpoint
            _write_checkpoint(d, 100_000)
            return 3221225477
        (d / "final_eval.json").write_text("{}")
        return 0

    monkeypatch.setattr(run_campaign.subprocess, "call", fake_call)
    seed, rc, note = run_campaign.train_one((tmp_path, 1, "R8", 1_000_000, 2, "muscle", [], 50_000, 20_000, None, ""))
    assert rc == 0 and "resumed from 100,000" in note
    assert (tmp_path / "seed_1_crash1" / "replay_buffer.pkl").exists()
    second = calls[1]
    assert second[second.index("--continue-from") + 1] == str(tmp_path / "seed_1_crash1")
    assert second[second.index("--timesteps") + 1] == "900000"


def test_a_relaunched_campaign_picks_up_runs_a_blue_screen_left_unfinished(tmp_path, monkeypatch):
    """A blue screen kills the campaign itself, so nothing recovers in-process. Launching the same
    campaign again must continue each unfinished seed from its checkpoint, not overwrite it."""
    import sys
    sys.path.insert(0, str(ROOT))
    import run_campaign
    d = tmp_path / "seed_2"
    d.mkdir()
    _write_checkpoint(d, 300_000)
    (tmp_path / "seed_2_crash1").mkdir()                     # an older crash keeps its name
    calls = []

    def fake_call(cmd, stdout=None, stderr=None, cwd=None):
        calls.append(list(cmd))
        (Path(cmd[cmd.index("--save-dir") + 1]) / "final_eval.json").write_text("{}")
        return 0

    monkeypatch.setattr(run_campaign.subprocess, "call", fake_call)
    seed, rc, note = run_campaign.train_one((tmp_path, 2, "R8", 1_000_000, 1, "muscle", [], 50_000, 20_000, None, ""))
    assert rc == 0 and len(calls) == 1 and "resumed from 300,000" in note
    assert calls[0][calls[0].index("--continue-from") + 1] == str(tmp_path / "seed_2_crash2")
    assert calls[0][calls[0].index("--timesteps") + 1] == "700000"
    assert (tmp_path / "seed_2_crash2" / "replay_buffer.pkl").exists()
