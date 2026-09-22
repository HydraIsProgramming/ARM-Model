"""
Staged training campaign on the human-arm waypoint task.

    python run_campaign.py --stage A                # 2 waypoints, 5 seeds, 300K steps each
    python run_campaign.py --stage B                # 3 waypoints, 5 seeds, 600K steps each
    python run_campaign.py --stage A --report       # re-read the packages, print the table, promote

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

STAGES = {
    "A": dict(waypoints="2", timesteps=300_000, description="two waypoints, hold at the second"),
    "B": dict(waypoints="3", timesteps=600_000, description="three waypoints, hold at the third"),
    "B2": dict(waypoints="3", timesteps=600_000,
               description="three waypoints, CLEAN: muscle activation dynamics + hand-acceleration penalty",
               extra=["--activation-dynamics", "0.04,0.06", "--reward-weight", "W_ACCEL=0.02"]),
    "C": dict(waypoints="3", timesteps=800_000, description="three waypoints + hand load 0-1 kg + sensory noise/delay",
              extra=["--hand-load-range", "0,1", "--obs-noise", "0.01,0.05", "--sensory-delay", "2"]),
}
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
PROTECTED = {"CHAMPION", "fischer_waypoints_v2_500k", "CHAMPION_V2"}


def parse_args():
    p = argparse.ArgumentParser(description="staged training campaign")
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--parallel", type=int, default=5)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default=None, help="default: $ARM_MODEL_OUTPUT_DIR/stage<X> or C:/arm-model-runs/stage<X>")
    p.add_argument("--actuation-mode", default="muscle", choices=["muscle", "torque"])
    p.add_argument("--report", action="store_true", help="only re-evaluate/report existing packages")
    p.add_argument("--promote", action="store_true", help="copy the champion package into project_assets/outputs")
    p.add_argument("--force", action="store_true")
    p.add_argument("--pass-success", type=float, default=0.8)
    p.add_argument("--pass-seeds", type=int, default=3)
    p.add_argument("--checkpoint-every", type=int, default=50_000)
    p.add_argument("--eval-freq", type=int, default=20_000)
    p.add_argument("--warm-start-from", default=None,
                   help="stage directory whose seed_<k> packages seed the same-numbered runs (weights only)")
    p.add_argument("--suffix", default="", help="run-dir suffix, e.g. _warm")
    p.add_argument("--extra", default="", help="extra CLI flags for every run, e.g. \"--net-arch 128,128\"")
    return p.parse_args()


def run_dir_for(out: Path, seed: int, suffix: str = "") -> Path:
    return out / f"seed_{seed}{suffix}"


def train_one(args_tuple):
    out, seed, stage, timesteps, threads, actuation, extra, ckpt, evalf, warm, suffix = args_tuple
    d = run_dir_for(out, seed, suffix)
    if (d / "run_manifest.json").exists() and (d / "final_eval.json").exists():
        return seed, 0, "already finished"
    d.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(SCRIPT), "--timesteps", str(timesteps), "--seed", str(seed), "--num-threads", str(threads),
           "--waypoints", STAGES[stage]["waypoints"], "--actuation-mode", actuation, "--save-dir", str(d),
           "--skip-validators", "--overwrite", "--checkpoint-every", str(ckpt), "--eval-freq", str(evalf),
           ] + list(STAGES[stage].get("extra", [])) + extra
    if warm:
        src = Path(warm) / f"seed_{seed}"
        if (src / "run_manifest.json").exists():
            cmd += ["--continue-from", str(src), "--drop-buffer"]
    t0 = time.perf_counter()
    with open(d / "stdout.log", "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    return seed, rc, f"{(time.perf_counter() - t0) / 60:.1f} min"


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
    target = ROOT / "project_assets" / "outputs" / f"CHAMPION_HUMAN_V1_{stage}"
    if target.name in PROTECTED:
        raise SystemExit(f"refusing to write protected directory {target}")
    if target.exists():
        if not force:
            print(f"[promote] {target} exists; use --force to replace it")
            return
        shutil.rmtree(target)
    ignore = shutil.ignore_patterns("replay_buffer.pkl", "checkpoints", "stdout.log", "*.npz")
    shutil.copytree(champion_dir, target, ignore=ignore)
    print(f"[promote] champion package copied to {target} (replay buffer and checkpoints left in {champion_dir})")


def main():
    args = parse_args()
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
                 args.checkpoint_every, args.eval_freq, args.warm_start_from, args.suffix) for s in seeds]
        with ProcessPoolExecutor(max_workers=max(1, args.parallel)) as ex:
            futs = {ex.submit(train_one, j): j[1] for j in jobs}
            for f in as_completed(futs):
                seed, rc, note = f.result()
                print(f"  seed {seed}: {'ok' if rc == 0 else f'FAILED rc={rc}'} ({note})", flush=True)
        print(f"campaign wall time {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
    champion = report(base, seeds, args)
    if champion is not None and (args.promote or not args.report):
        promote(run_dir_for(base, champion[0], args.suffix), args.stage + args.suffix, args.force)


if __name__ == "__main__":
    main()
