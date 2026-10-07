"""Training on a machine that blue-screens: crash-safe checkpoint writes, refusing damaged
checkpoints on resume, the gentle-CPU options, and the hardware watchdog's pause/resume logic."""

import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rl_armMotion.two_d.training.durable import PARTIAL_PREFIX, durable_text, durable_write  # noqa: E402


# --------------------------------------------------------------------------- durable writes

def test_durable_write_replaces_the_file_and_leaves_no_partial(tmp_path):
    target = tmp_path / "curriculum_state.json"
    target.write_text('{"num_timesteps": 50000}')
    durable_text(target, '{"num_timesteps": 100000}')
    assert json.loads(target.read_text())["num_timesteps"] == 100000
    assert not list(tmp_path.glob(PARTIAL_PREFIX + "*"))


def test_a_write_that_dies_midway_leaves_the_previous_file_intact(tmp_path):
    target = tmp_path / "replay_buffer.pkl"
    target.write_bytes(b"\x80 old complete buffer .")

    def dies(p):
        Path(p).write_bytes(b"\x00" * 64)      # what a crash used to leave: the right size, all zeros
        raise OSError("machine went down")

    with pytest.raises(OSError):
        durable_write(target, dies)
    assert target.read_bytes() == b"\x80 old complete buffer ."


def test_partial_files_are_invisible_to_package_globs(tmp_path):
    durable_write(tmp_path / "step_0050000.zip", lambda p: Path(p).write_bytes(b"x"))
    (tmp_path / (PARTIAL_PREFIX + "step_0100000.zip")).write_bytes(b"left by a crash")
    assert [p.name for p in tmp_path.glob("step_*.zip")] == ["step_0050000.zip"]


def test_sb3_saves_keep_their_file_names_through_durable_write(tmp_path):
    """SB3 adds '.zip' / '.pkl' to names without them; the temporary name must keep the extension."""
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import SAC
    from rl_armMotion.two_d.environments.task_env import ArmTaskEnv
    model = SAC("MlpPolicy", ArmTaskEnv(actuation_mode="torque"), learning_starts=10, buffer_size=200,
                policy_kwargs={"net_arch": [8, 8]}, verbose=0)
    durable_write(tmp_path / "sac_model_last.zip", lambda p: model.save(str(p)))
    durable_write(tmp_path / "replay_buffer.pkl", lambda p: model.save_replay_buffer(str(p)))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["replay_buffer.pkl", "sac_model_last.zip"]
    assert zipfile.ZipFile(tmp_path / "sac_model_last.zip").testzip() is None


# --------------------------------------------------------------------------- resume refuses damage

def _checkpoint(d: Path, zero=None):
    d.mkdir(parents=True)
    with zipfile.ZipFile(d / "sac_model_last.zip", "w") as z:
        z.writestr("data", "weights")
    (d / "replay_buffer.pkl").write_bytes(b"\x80\x05 buffer .")
    (d / "curriculum_state.json").write_text(json.dumps({"num_timesteps": 50_000}))
    (d / "run_manifest.json").write_text("{}")
    if zero:
        n = (d / zero).stat().st_size
        (d / zero).write_bytes(b"\x00" * n)


@pytest.mark.parametrize("zero", [None, "sac_model_last.zip", "replay_buffer.pkl", "curriculum_state.json"])
def test_resume_only_trusts_intact_checkpoints(tmp_path, zero):
    import run_campaign
    _checkpoint(tmp_path / "seed_1", zero)
    assert run_campaign._resumable(tmp_path / "seed_1") is (zero is None)


def test_an_unfinished_run_with_a_damaged_checkpoint_is_kept_and_restarted(tmp_path, monkeypatch):
    import run_campaign
    _checkpoint(tmp_path / "seed_3", "curriculum_state.json")
    calls = []

    def fake_call(cmd, stdout=None, stderr=None, cwd=None):
        calls.append(list(cmd))
        (Path(cmd[cmd.index("--save-dir") + 1]) / "final_eval.json").write_text("{}")
        return 0

    monkeypatch.setattr(run_campaign.subprocess, "call", fake_call)
    _, rc, note = run_campaign.train_one((tmp_path, 3, "R8N", 1_000_000, 1, "muscle", [], 50_000, 20_000, None, ""))
    assert rc == 0 and len(calls) == 1 and "--continue-from" not in calls[0]
    assert calls[0][calls[0].index("--timesteps") + 1] == "1000000"
    assert (tmp_path / "seed_3_crash1" / "curriculum_state.json").exists()      # kept, not deleted
    assert "no intact checkpoint" in note


def test_queue_spec_and_stagger(tmp_path, monkeypatch):
    import run_campaign
    assert run_campaign.parse_queue("R8:1-3;R8E:2") == [("R8", [1, 2, 3]), ("R8E", [2])]
    with pytest.raises(SystemExit):
        run_campaign.parse_queue("NOPE:1")
    slept = []
    monkeypatch.setattr(run_campaign.time, "sleep", slept.append)
    monkeypatch.setattr(run_campaign.subprocess, "call", lambda cmd, **k: (
        (Path(cmd[cmd.index("--save-dir") + 1]) / "final_eval.json").write_text("{}"), 0)[1])
    run_campaign.train_one((tmp_path, 1, "R8", 10, 1, "torque", [], 5, 5, None, "", 90.0))
    assert slept == [90.0]


def test_cpu_spec_parsing():
    from rl_armMotion.two_d.training.process_control import parse_cpus
    assert parse_cpus("16-23") == list(range(16, 24))
    assert parse_cpus("0-3, 8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert parse_cpus("") == []


# --------------------------------------------------------------------------- watchdog

class _Proc:
    def __init__(self):
        self.calls = []

    def suspend(self):
        self.calls.append("suspend")

    def resume(self):
        self.calls.append("resume")


def _run_watchdog(monkeypatch, tmp_path, errors_by_poll, argv=(), polls=None):
    """Drive the watchdog loop on a fake clock: errors_by_poll[i] is how many WHEA events the
    window query returns at poll i (the cooldown query sees the same number)."""
    import hw_watchdog as w
    proc = _Proc()
    clock = {"t": 0.0, "i": 0}
    awake = []
    monkeypatch.setattr(w, "training_processes", lambda: [proc])
    monkeypatch.setattr(w, "whea_events", lambda minutes: errors_by_poll[min(clock["i"], len(errors_by_poll) - 1)])
    monkeypatch.setattr(w, "keep_awake", awake.append)
    monkeypatch.setattr(w.time, "time", lambda: clock["t"])

    def sleep(s):
        clock["t"] += s
        clock["i"] += 1
        if clock["i"] >= (polls or len(errors_by_poll)):
            raise KeyboardInterrupt

    monkeypatch.setattr(w.time, "sleep", sleep)
    monkeypatch.setattr(sys, "argv", ["hw_watchdog.py", "--log", str(tmp_path / "wd.log"), *argv])
    with pytest.raises(KeyboardInterrupt):
        w.main()
    return proc.calls, awake, (tmp_path / "wd.log").read_text()


def test_watchdog_pauses_on_hardware_errors_and_resumes_when_quiet(monkeypatch, tmp_path):
    calls, awake, log = _run_watchdog(monkeypatch, tmp_path, [0, 0, 4, 4, 0, 0])
    assert calls == ["suspend", "resume"]
    assert awake[0] is True and awake[-1] is False            # kept awake while training, released at exit
    assert "PAUSED 1" in log and "RESUMED 1" in log


def test_watchdog_stays_paused_when_errors_keep_returning(monkeypatch, tmp_path):
    seq = [5, 0, 5, 0, 5, 0, 0, 0]
    calls, _, log = _run_watchdog(monkeypatch, tmp_path, seq, argv=["--max-pauses", "3"])
    assert calls == ["suspend", "resume", "suspend", "resume", "suspend"]
    assert "STAYING PAUSED" in log


def test_watchdog_never_leaves_runs_frozen_when_it_stops(monkeypatch, tmp_path):
    calls, _, _ = _run_watchdog(monkeypatch, tmp_path, [6, 6, 6])
    assert calls == ["suspend", "resume"]                     # the final resume comes from the exit path
