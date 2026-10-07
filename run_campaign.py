"""
Staged training campaign on the human-arm waypoint task.

    python run_campaign.py --stage A                # 2 waypoints, 5 seeds, 300K steps each
    python run_campaign.py --stage B                # 3 waypoints, 5 seeds, 600K steps each
    python run_campaign.py --stage A --report       # re-read the packages, print the table, promote

On the training PC (its CPU logs hardware errors and has blue-screened under load), train gently and
alongside the watchdog - several stages through one pool, E-cores only, efficiency mode, staggered:

    python scripts/hw_watchdog.py
    python run_campaign.py --queue "R8:1-5;R8E:1-3;R8N:1-3" --parallel 8 --threads 1 --cpus 16-23 \\
        --efficiency --stagger 90

Checkpoints are written crash-safely; after a crash or reboot, the same command resumes every
unfinished run from its last intact checkpoint.

Pass criterion per stage: >= 3 of 5 seeds reach >= 80% success at the
evaluation standard (20 randomised episodes, tolerance 0.05 m, 0.5 s hold).
If a stage fails, the fix goes into the environment / reward / curriculum,
never into more seeds. The champion package (best continuous score) is copied
whole to project_assets/outputs/CHAMPION_HUMAN_V1_<stage>/ and never
overwritten without --force.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts" / "train_fischer_session.py"
PY = sys.executable

# R7's arguments with the muscles replaced by joint-torque actuators (see "R8" below). The
# observation also carries the previous torque command (--prev-action-obs): muscles expose their
# activation state, actuators have none, and the action-rate penalty variant needs it to be Markov.
R8_EXTRA = ["--actuation-mode", "torque", "--prev-action-obs",
            "--random-waypoints", "3", "--joint-margin-deg", "5", "--reach-margin", "0",
            "--reward-weight", "W_ACCEL=0.02", "--reward-weight", "EE_SPEED_CAP=1.5",
            "--reward-weight", "W_SPEED_EXCESS=0.5", "--reward-weight", "W_SPEED_LINEAR=0",
            "--reward-weight", "SPEED_PENALTY_CAP=inf"]
# Screening runs stop 200K steps after T80 (first round where the mean of the last 3 fresh
# 10-episode probes is >= 80%) - long enough to see whether success holds.
SCREEN = ["--stop-after-t80", "200000"]

STAGES = {
    "A": dict(waypoints="2v1", timesteps=300_000, description="two waypoints (v1 set), hold at the second"),
    "B": dict(waypoints="3v1", timesteps=600_000, description="three waypoints (v1 set), hold at the third"),
    "A2": dict(waypoints="2", timesteps=300_000, description="two waypoints (v2 set), CLEAN config",
               extra=["--activation-dynamics", "0.04,0.06", "--reward-weight", "W_ACCEL=0.02"]),
    "B2": dict(waypoints="3v1", timesteps=600_000,
               description="three waypoints (v1 set), CLEAN: muscle activation dynamics + hand-acceleration penalty",
               extra=["--activation-dynamics", "0.04,0.06", "--reward-weight", "W_ACCEL=0.02"]),
    "B3": dict(waypoints="3", timesteps=600_000,
               description="three waypoints (v2 set, >=0.19 m inside the reach circle), CLEAN config",
               extra=["--activation-dynamics", "0.04,0.06", "--reward-weight", "W_ACCEL=0.02"]),
    "R3": dict(waypoints="none", timesteps=1_000_000,
               description="THREE RANDOM waypoints per episode from the feasible region, CLEAN config (generalising policy)",
               extra=["--random-waypoints", "3", "--activation-dynamics", "0.04,0.06", "--reward-weight", "W_ACCEL=0.02"]),
    "R4": dict(waypoints="none", timesteps=1_000_000,
               description=("THREE RANDOM waypoints, CLEAN config, with the end-range feasibility rule: targets "
                            "whose holding pose falls inside a joint's passive end-range zone are no longer sampled"),
               extra=["--random-waypoints", "3", "--activation-dynamics", "0.04,0.06", "--reward-weight", "W_ACCEL=0.02"]),
    "F3": dict(waypoints="none", timesteps=1_000_000,
               description=("THREE RANDOM waypoints, FREE ORDER: the policy sees all three at once and "
                            "chooses its own route; no active-goal channel, no ordering signal"),
               extra=["--random-waypoints", "3", "--free-order", "--activation-dynamics", "0.04,0.06",
                      "--reward-weight", "W_ACCEL=0.02"]),
    "R5": dict(waypoints="none", timesteps=1_000_000,
               description=("THREE RANDOM waypoints, end-range rule, HUMAN SPEED COST: hand-speed cap 1.2 m/s "
                            "at weight 10 (was 1.5 at 0.5, which the champion simply paid to exceed)"),
               extra=["--random-waypoints", "3", "--activation-dynamics", "0.04,0.06",
                      "--reward-weight", "W_ACCEL=0.02"]),
    "R6": dict(waypoints="none", timesteps=1_000_000,
               description=("THREE RANDOM waypoints, end-range rule, BINDING speed cost: the hand-speed "
                            "penalty gains a linear term so it has a real slope at the cap"),
               extra=["--random-waypoints", "3", "--activation-dynamics", "0.04,0.06",
                      "--reward-weight", "W_ACCEL=0.02"]),
    "R7": dict(waypoints="none", timesteps=1_000_000,
               description=("THREE RANDOM waypoints over the FULL HUMAN WORKSPACE: targets limited only by the "
                            "joint ranges (5 deg clear of each hard stop) and strength, not the 18 cm reach-margin "
                            "ring; otherwise R3's exact config and reward"),
               extra=["--random-waypoints", "3", "--joint-margin-deg", "5", "--reach-margin", "0",
                      "--activation-dynamics", "0.04,0.06",
                      "--reward-weight", "W_ACCEL=0.02", "--reward-weight", "EE_SPEED_CAP=1.5",
                      "--reward-weight", "W_SPEED_EXCESS=0.5", "--reward-weight", "W_SPEED_LINEAR=0",
                      "--reward-weight", "SPEED_PENALTY_CAP=inf"]),
    # --- R8: the same task and reward as R7, but the arm is driven by direct joint-torque
    # actuators (shoulder, elbow; action in [-1, 1] x 25 / 12 N.m) instead of muscles. No
    # --activation-dynamics: that is a muscle property, and in torque mode it would only add
    # four always-zero observation inputs. "--actuation-mode torque" sits inside `extra` so a
    # resumed run (scripts/resume_stage.py rebuilds from `extra`) stays a torque run.
    "R8": dict(waypoints="none", timesteps=1_000_000,
               description=("THREE RANDOM waypoints over the FULL HUMAN WORKSPACE with TORQUE ACTUATORS at the "
                            "shoulder and elbow (no muscles); otherwise R7's exact config and reward"),
               extra=R8_EXTRA),
    # Step-reduction variants of R8: one change each, seeds 1-3 (paired with R8's), capped at 1M and
    # stopped 200K after T80. Compared on T80 and on held-out success.
    "R8E": dict(waypoints="none", timesteps=1_000_000,
                description=("R8 with 250-step TRAINING episodes (5 s; evaluation stays 10 s): a successful "
                             "episode is ~70% holding, so this doubles the fresh targets per million steps"),
                extra=R8_EXTRA + ["--train-max-steps", "250"] + SCREEN),
    "R8N": dict(waypoints="none", timesteps=1_000_000,
                description="R8 with 3-step returns (SB3 n_steps=3): rewards reach earlier actions faster",
                extra=R8_EXTRA + ["--n-steps", "3"] + SCREEN),
    "R8U": dict(waypoints="none", timesteps=1_000_000,
                description="R8 with 2 gradient updates per environment step (--utd 2; ~2x compute per step)",
                extra=R8_EXTRA + ["--utd", "2"] + SCREEN),
    "R8T": dict(waypoints="none", timesteps=1_000_000,
                description="R8 trained with TQC (sb3-contrib) instead of SAC",
                extra=R8_EXTRA + ["--algorithm", "TQC"] + SCREEN),
    "F2": dict(waypoints="none", timesteps=500_000,
               description=("PILOT: TWO random waypoints, FREE ORDER - isolates chaining, which is where the "
                            "three-point free-order runs stalled (they reached one point and stopped)"),
               extra=["--random-waypoints", "2", "--free-order", "--activation-dynamics", "0.04,0.06",
                      "--reward-weight", "W_ACCEL=0.02"]),
    "C": dict(waypoints="3", timesteps=800_000, description="three waypoints + hand load 0-1 kg + sensory noise/delay",
              extra=["--hand-load-range", "0,1", "--obs-noise", "0.01,0.05", "--sensory-delay", "2"]),
}
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
PROTECTED = {"CHAMPION", "fischer_waypoints_v2_500k", "CHAMPION_V2"}


def parse_args():
    p = argparse.ArgumentParser(description="staged training campaign")
    p.add_argument("--stage", choices=sorted(STAGES), help="the stage to train (or use --queue)")
    p.add_argument("--queue", default="",
                   help=("several stages through ONE pool of --parallel runs, e.g. \"R8:1-5;R8E:1-3;R8N:1-3\". "
                         "Each stage is reported at the end; nothing is promoted (use --stage X --report --promote)."))
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--parallel", type=int, default=5)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default=None, help="default: $ARM_MODEL_OUTPUT_DIR/stage<X> or C:/arm-model-runs/stage<X>")
    p.add_argument("--actuation-mode", default="muscle", choices=["muscle", "torque"])
    p.add_argument("--report", action="store_true", help="only re-evaluate/report existing packages")
    p.add_argument("--promote", action="store_true", help="copy the champion package into project_assets/outputs")
    p.add_argument("--no-promote", action="store_true",
                   help="never copy into project_assets/outputs (screening runs stay in the runs folder)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--pass-success", type=float, default=0.8)
    p.add_argument("--pass-seeds", type=int, default=3)
    p.add_argument("--checkpoint-every", type=int, default=50_000)
    p.add_argument("--eval-freq", type=int, default=20_000)
    p.add_argument("--warm-start-from", default=None,
                   help="stage directory whose seed_<k> packages seed the same-numbered runs (weights only)")
    p.add_argument("--suffix", default="", help="run-dir suffix, e.g. _warm")
    p.add_argument("--extra", default="", help="extra CLI flags for every run, e.g. \"--net-arch 128,128\"")
    p.add_argument("--cpu-exclude", default="",
                   help="logical processors to keep training off, e.g. \"12-15\" (Windows; every run inherits it)")
    p.add_argument("--cpus", default="",
                   help="logical processors training may use, e.g. \"16-23\" for the E-cores only (Windows)")
    p.add_argument("--efficiency", action="store_true",
                   help="Windows efficiency mode for every run: EcoQoS (efficient clocks) + below-normal priority")
    p.add_argument("--stagger", type=float, default=0.0,
                   help="seconds between starting runs, so their checkpoints are not all written at once")
    args = p.parse_args()
    if not args.stage and not args.queue:
        p.error("give --stage or --queue")
    return args


def parse_queue(spec: str):
    """"R8:1-5;R8E:1-3" -> [("R8", [1, 2, 3, 4, 5]), ("R8E", [1, 2, 3])]"""
    out = []
    for item in spec.split(";"):
        if not item.strip():
            continue
        stage, _, seeds = item.partition(":")
        stage = stage.strip()
        if stage not in STAGES:
            raise SystemExit(f"unknown stage {stage!r} in --queue")
        nums = []
        for part in (seeds or ",".join(map(str, DEFAULT_SEEDS))).split(","):
            lo, _, hi = part.strip().partition("-")
            nums += list(range(int(lo), int(hi or lo) + 1))
        out.append((stage, nums))
    return out


def gentle_cpu(args) -> None:
    """--cpus / --efficiency: applied to this process; every run it starts inherits the processor set
    and the priority, and turns on EcoQoS for itself (train_fischer_session reads the env flag)."""
    if not (args.cpus or args.efficiency):
        return
    sys.path.insert(0, str(ROOT / "src"))
    from rl_armMotion.two_d.training.process_control import ECO_ENV, efficiency_mode, parse_cpus, set_cpus
    if args.cpus:
        cpus = parse_cpus(args.cpus)
        mask = set_cpus(cpus)
        print(f"[cpu] training only on logical processors {cpus} (affinity mask {mask:#x})", flush=True)
    if args.efficiency:
        os.environ[ECO_ENV] = "1"
        efficiency_mode()
        print("[cpu] efficiency mode: EcoQoS + below-normal priority for every run", flush=True)


def exclude_cpus(spec: str) -> None:
    """Keep this process - and every training run it starts, which inherit the mask - off the given
    logical processors. Used to avoid CPU cores that log corrected machine-check errors."""
    if not spec or os.name != "nt":
        return
    import ctypes
    bad = set()
    for part in spec.split(","):
        lo, _, hi = part.strip().partition("-")
        bad.update(range(int(lo), int(hi or lo) + 1))
    n = os.cpu_count() or 1
    mask = sum(1 << i for i in range(n) if i not in bad)
    if not mask:
        raise SystemExit("--cpu-exclude leaves no processors")
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE          # a 64-bit pseudo-handle; the default int truncates it
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k32.SetProcessAffinityMask.restype = wintypes.BOOL
    if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
        raise SystemExit(f"could not set the processor affinity (error {ctypes.get_last_error()})")
    print(f"[cpu] training kept off logical processors {sorted(bad)} (affinity mask {mask:#x})", flush=True)


def run_dir_for(out: Path, seed: int, suffix: str = "") -> Path:
    return out / f"seed_{seed}{suffix}"


MAX_CRASH_RECOVERIES = 6


def _resumable(run_dir: Path) -> bool:
    """A crashed run can continue if a checkpoint left its model, replay buffer and curriculum state -
    and they are intact: before checkpoints were written crash-safely, a blue screen left the files
    being written at that moment full of zeros."""
    files = ("sac_model_last.zip", "replay_buffer.pkl", "curriculum_state.json", "run_manifest.json")
    return all((run_dir / f).exists() for f in files) and _intact(run_dir)


def _intact(run_dir: Path) -> bool:
    import zipfile
    try:
        with zipfile.ZipFile(run_dir / "sac_model_last.zip") as z:
            if z.testzip() is not None:
                return False
        for f in ("curriculum_state.json", "run_manifest.json"):
            json.loads((run_dir / f).read_text(encoding="utf-8"))
        with open(run_dir / "replay_buffer.pkl", "rb") as fh:       # a pickle starts with PROTO, ends with STOP
            if fh.read(1) != b"\x80":
                return False
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b".":
                return False
    except (OSError, ValueError, zipfile.BadZipFile):
        return False
    return True


def _crash_dir(d: Path) -> Path:
    """The first unused seed_k_crashN name, so crashes across several campaign launches never collide."""
    n = 1
    while d.with_name(f"{d.name}_crash{n}").exists():
        n += 1
    return d.with_name(f"{d.name}_crash{n}")


def _continue_cmd(base, timesteps, crashed: Path):
    with open(crashed / "curriculum_state.json", encoding="utf-8") as fh:
        done = int(json.load(fh).get("num_timesteps", 0))
    return base + ["--timesteps", str(max(1, timesteps - done)), "--continue-from", str(crashed)], done


def train_one(args_tuple):
    """Train one seed. A run that dies (on this machine: native access violations from a CPU that
    logs corrected machine-check errors) is recovered automatically: the crashed folder is kept as
    seed_k_crashN, and the run continues from its last checkpoint (model + replay buffer +
    curriculum state, written every `ckpt` steps) into seed_k - or restarts if it died before its
    first checkpoint. The final seed_k folder carries the whole curriculum history and its lineage.
    A run left unfinished by a campaign that died with the machine (a blue screen) is picked up the
    same way when the campaign is launched again with the same arguments."""
    out, seed, stage, timesteps, threads, actuation, extra, ckpt, evalf, warm, suffix = args_tuple[:11]
    delay = args_tuple[11] if len(args_tuple) > 11 else 0.0
    d = run_dir_for(out, seed, suffix)
    if (d / "run_manifest.json").exists() and (d / "final_eval.json").exists():
        return seed, 0, "already finished"
    if delay:
        time.sleep(delay)
    base = [PY, str(SCRIPT), "--seed", str(seed), "--num-threads", str(threads),
            "--waypoints", STAGES[stage]["waypoints"], "--actuation-mode", actuation, "--save-dir", str(d),
            "--skip-validators", "--overwrite", "--checkpoint-every", str(ckpt), "--eval-freq", str(evalf),
            ] + list(STAGES[stage].get("extra", [])) + extra
    cmd = base + ["--timesteps", str(timesteps)]
    if warm:
        src = Path(warm) / f"seed_{seed}"
        if (src / "run_manifest.json").exists():
            cmd += ["--continue-from", str(src), "--drop-buffer"]
    t0 = time.perf_counter()
    notes = []
    if _resumable(d):
        crashed = _crash_dir(d)
        shutil.move(str(d), str(crashed))
        cmd, done = _continue_cmd(base, timesteps, crashed)
        notes.append(f"unfinished run from an earlier launch, resumed from {done:,}")
        print(f"  seed {seed}: {notes[-1]}", flush=True)
    elif (d / "run_manifest.json").exists():
        crashed = _crash_dir(d)                    # unfinished, nothing intact to resume: keep it, start over
        shutil.move(str(d), str(crashed))
        notes.append(f"unfinished run from an earlier launch had no intact checkpoint (kept as {crashed.name}); "
                     f"restarted")
        print(f"  seed {seed}: {notes[-1]}", flush=True)
    for attempt in range(MAX_CRASH_RECOVERIES + 1):
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "stdout.log", "w", encoding="utf-8") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        if rc == 0 or attempt == MAX_CRASH_RECOVERIES:
            break
        crashed = _crash_dir(d)
        shutil.move(str(d), str(crashed))
        if _resumable(crashed):
            cmd, done = _continue_cmd(base, timesteps, crashed)
            notes.append(f"crash rc={rc}, resumed from {done:,}")
        else:
            cmd = base + ["--timesteps", str(timesteps)]
            notes.append(f"crash rc={rc} before the first checkpoint, restarted")
        print(f"  seed {seed}: {notes[-1]}", flush=True)
    return seed, rc, f"{(time.perf_counter() - t0) / 60:.1f} min" + (f"; {'; '.join(notes)}" if notes else "")


def evaluate_package(d: Path, n_episodes: int = 20):
    sys.path.insert(0, str(ROOT / "src"))
    from rl_armMotion.two_d.evaluation import EVAL_STANDARD, evaluate_policy, load_package
    try:
        model, env, manifest = load_package(d, which="best", terminate_on_success=True)
    except Exception as exc:
        return None, f"load failed: {exc}"
    res = evaluate_policy(model, env, n_episodes=n_episodes, seed=EVAL_STANDARD.seed + 900, log_trajectories=True)
    res.save(d / "campaign_eval.json", trajectories=True)
    return res, manifest


def report(out: Path, seeds, args):
    rows = []
    print(f"\n{'=' * 78}\nSTAGE {args.stage}: {STAGES[args.stage]['description']}  ({out})\n{'=' * 78}")
    print(f"{'seed':>5} {'success':>8} {'wp':>5} {'hold':>5} {'dist(m)':>8} {'steps':>6} {'straight':>8} {'sym':>5} {'best@':>8}")
    for s in seeds:
        d = run_dir_for(out, s, args.suffix)
        if not d.exists():
            print(f"{s:>5}  (missing)")
            continue
        res, manifest = evaluate_package(d)
        if res is None:
            print(f"{s:>5}  {manifest}")
            continue
        hl = res.human_likeness_summary()
        steps = res.mean_steps_to_success
        rows.append((s, res, manifest))
        print(f"{s:>5} {res.success_rate:>8.0%} {res.mean_waypoints:>5.2f} {res.mean_max_hold:>5.2f} "
              f"{res.mean_final_distance:>8.3f} {steps if steps != float('inf') else float('nan'):>6.0f} "
              f"{hl['straightness']:>8.2f} {hl['symmetry']:>5.2f} {str(manifest.get('best_saved_at')):>8}")
    if not rows:
        return None
    passed = [r for r in rows if r[1].success_rate >= args.pass_success]
    ok = len(passed) >= args.pass_seeds
    import numpy as np
    sr = [r[1].success_rate for r in rows]
    print(f"\nsuccess rate over seeds: mean {np.mean(sr):.2f} +/- {np.std(sr, ddof=1) if len(sr) > 1 else 0:.2f}  "
          f"({len(passed)}/{len(rows)} seeds >= {args.pass_success:.0%})  ->  {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("Stage FAILED: fix the MDP (env / reward / curriculum), do not add seeds. "
              "Inspect curriculum_state.json history per seed to see which ladder stalled.")
    champion = max(rows, key=lambda r: r[1].score())
    print(f"champion: seed {champion[0]} score {champion[1].score()}")
    summary = {
        "stage": args.stage, "passed": ok, "pass_success": args.pass_success, "pass_seeds": args.pass_seeds,
        "seeds": {r[0]: r[1].summary() for r in rows},
        "champion_seed": champion[0], "champion_dir": str(run_dir_for(out, champion[0], args.suffix)),
    }
    with open(out / f"campaign_report{args.suffix}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return champion


def promote(champion_dir: Path, stage: str, force: bool):
    """Copy the champion run into project_assets/outputs/CHAMPION_HUMAN_V1_<stage>.

    Nothing is ever deleted. With --force an existing package is first MOVED to
    project_assets/outputs/_superseded/CHAMPION_HUMAN_V1_<stage>_<timestamp>/ (outside the
    GUI's model list, still in the repo), so an older champion can always be brought back.
    """
    outputs = ROOT / "project_assets" / "outputs"
    target = outputs / f"CHAMPION_HUMAN_V1_{stage}"
    if target.name in PROTECTED:
        raise SystemExit(f"refusing to write protected directory {target}")
    if target.exists():
        if not force:
            print(f"[promote] {target} exists; use --force to supersede it (the old one is kept)")
            return
        keep = outputs / "_superseded" / f"{target.name}_{time.strftime('%Y%m%d-%H%M')}"
        keep.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(keep))
        print(f"[promote] previous {target.name} moved to {keep}")
    ignore = shutil.ignore_patterns("replay_buffer.pkl", "checkpoints", "stdout.log", "*.npz")
    shutil.copytree(champion_dir, target, ignore=ignore)
    print(f"[promote] champion package copied to {target} (replay buffer and checkpoints left in {champion_dir})")


def run_queue(args) -> None:
    """Several stages through one pool, so the number of runs at once is capped across all of them."""
    plan = parse_queue(args.queue)
    root = Path(args.out) if args.out else Path(os.environ.get("ARM_MODEL_OUTPUT_DIR", "C:/arm-model-runs"))
    extra = args.extra.split() if args.extra else []
    jobs = []
    for stage, seeds in plan:
        base = root / f"stage{stage}"
        base.mkdir(parents=True, exist_ok=True)
        steps = args.timesteps or STAGES[stage]["timesteps"]
        jobs += [(base, s, stage, steps, args.threads, args.actuation_mode, extra, args.checkpoint_every,
                  args.eval_freq, args.warm_start_from, args.suffix) for s in seeds]
    jobs = [j + (args.stagger * i if i < args.parallel else 0.0,) for i, j in enumerate(jobs)]
    print(f"Queue: {', '.join(f'{st} seeds {sd}' for st, sd in plan)} -> {len(jobs)} runs, {args.parallel} at a time, "
          f"{args.threads} threads each, started {args.stagger:g} s apart -> {root}", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        futs = {ex.submit(train_one, j): (j[2], j[1]) for j in jobs}
        for f in as_completed(futs):
            stage, seed = futs[f]
            try:
                _, rc, note = f.result()
            except Exception as exc:                     # a worker lost with the machine: report, keep going
                rc, note = -1, f"worker error: {exc}"
            print(f"  {stage} seed {seed}: {'ok' if rc == 0 else f'FAILED rc={rc}'} ({note})", flush=True)
    print(f"queue wall time {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
    for stage, seeds in plan:
        args.stage = stage
        report(root / f"stage{stage}", seeds, args)


def main():
    args = parse_args()
    exclude_cpus(args.cpu_exclude)
    gentle_cpu(args)
    if args.queue:
        return run_queue(args)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    base = Path(args.out) if args.out else Path(os.environ.get("ARM_MODEL_OUTPUT_DIR", "C:/arm-model-runs")) / f"stage{args.stage}"
    base.mkdir(parents=True, exist_ok=True)
    timesteps = args.timesteps or STAGES[args.stage]["timesteps"]
    extra = args.extra.split() if args.extra else []
    if not args.report:
        print(f"Stage {args.stage}: {len(seeds)} seeds x {timesteps:,} steps, {args.parallel} parallel, "
              f"{args.threads} threads each -> {base}", flush=True)
        t0 = time.perf_counter()
        jobs = [(base, s, args.stage, timesteps, args.threads, args.actuation_mode, extra,
                 args.checkpoint_every, args.eval_freq, args.warm_start_from, args.suffix,
                 args.stagger * i if i < args.parallel else 0.0) for i, s in enumerate(seeds)]
        with ProcessPoolExecutor(max_workers=max(1, args.parallel)) as ex:
            futs = {ex.submit(train_one, j): j[1] for j in jobs}
            for f in as_completed(futs):
                seed, rc, note = f.result()
                print(f"  seed {seed}: {'ok' if rc == 0 else f'FAILED rc={rc}'} ({note})", flush=True)
        print(f"campaign wall time {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
    champion = report(base, seeds, args)
    if champion is not None and not args.no_promote and (args.promote or not args.report):
        promote(run_dir_for(base, champion[0], args.suffix), args.stage + args.suffix, args.force)


if __name__ == "__main__":
    main()
