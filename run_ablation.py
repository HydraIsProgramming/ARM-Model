"""
Ablation study for the four terminal-hold mechanisms.

The paper claims four mechanisms address the terminal-stabilisation failure
on the three-waypoint task. This harness measures whether they actually do,
by disabling one at a time and comparing against the full configuration.

Conditions (SEEDS_PER_CONDITION runs each, matched seeds across conditions):

  full           all mechanisms enabled (the tuned configuration)
  no_hold_curr   hold requirement pinned at 20 steps from the start
  decrement_5    hold progress decays by 5 per out-of-region step, not 3
  stochastic     curricula driven by training-policy success rate, and only
                 the final model is kept (no deterministic eval, no
                 best-model checkpointing)

The fourth mechanism, best-model checkpointing, is measured for free inside
every 'eval'-mode condition: each run saves both a best checkpoint and a
final model, and analyse() scores both, so the checkpointing benefit is the
paired difference between them.

Seeds are matched across conditions (seed 101 appears in every condition), so
differences are not confounded by initialisation.

Run from the project root:
    python run_ablation.py            # run the study
    python run_ablation.py --analyse  # re-analyse existing results only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
# SEEDS_PER_CONDITION: matched seeds per condition. 3 is the minimum that
#   gives any spread; raise for tighter error bars at linear time cost.
SEEDS_PER_CONDITION = 3
BASE_SEED = 101

TIMESTEPS = 500_000
WAYPOINTS = "0.2,1.2;2.2,0.5;1.0,-1.6"
WAYPOINTS_LIST = [[0.2, 1.2], [2.2, 0.5], [1.0, -1.6]]
EVAL_TOLERANCE = 0.6
EVAL_EPISODES = 5

# MAX_PARALLEL: concurrent training processes. One per condition keeps each
#   batch balanced so conditions finish together.
MAX_PARALLEL = 4
THREADS_PER_RUN = max(1, os.cpu_count() // MAX_PARALLEL - 1)

CONDITIONS = {
    "full": [],
    "no_hold_curr": ["--no-hold-curriculum"],
    "decrement_5": ["--hold-decrement", "5"],
    "stochastic": ["--curriculum-mode", "stochastic"],
}

PYTHON = sys.executable
ROOT = Path(__file__).parent
SCRIPT = str(ROOT / "scripts" / "train_fischer_session.py")
OUT_BASE = ROOT / "project_assets" / "outputs" / "Ablation"
OUT_BASE.mkdir(parents=True, exist_ok=True)
# ──────────────────────────────────────────────────────────────────────────────


def run_one(condition: str, seed: int) -> tuple:
    """Train one (condition, seed) pair. Skips if already complete."""
    save_dir = OUT_BASE / condition / f"seed_{seed:03d}"
    if (save_dir / "sac_model.zip").exists():
        print(f"  [{condition}/{seed}] already done, skipping", flush=True)
        return condition, seed, 0
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, SCRIPT,
        "--timesteps", str(TIMESTEPS),
        "--actuation-mode", "muscle",
        "--goal-direction", "EAST",
        "--waypoints", WAYPOINTS,
        "--seed", str(seed),
        "--save-dir", str(save_dir),
        "--num-threads", str(THREADS_PER_RUN),
        # Validators add ~15s and are not used by this study.
        "--fitts-trials", "1",
        "--power-trials", "1",
    ] + CONDITIONS[condition]

    print(f"  [{condition}/{seed}] starting", flush=True)
    t0 = time.time()
    with open(save_dir / "stdout.log", "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=f)
    mins = (time.time() - t0) / 60.0
    print(f"  [{condition}/{seed}] {'ok' if rc == 0 else 'FAIL'} ({mins:.0f} min)",
          flush=True)
    return condition, seed, rc


def evaluate(model_path: Path) -> dict:
    """Deterministically evaluate one saved model on the waypoint task.

    Returns mean waypoints reached, success count, and mean hold progress
    over EVAL_EPISODES episodes at the production standard.
    """
    if not model_path.with_suffix(".zip").exists():
        return {}
    script = f"""
import sys, json; sys.path.insert(0, 'src')
from stable_baselines3 import SAC
from rl_armMotion.two_d.environments.task_env import ArmTaskEnv
model = SAC.load(r'{model_path}')
env = ArmTaskEnv(actuation_mode='muscle', goal_direction='EAST')
env.set_waypoints({WAYPOINTS_LIST}, tolerance={EVAL_TOLERANCE})
wps, holds, succ = [], [], 0
for _ in range({EVAL_EPISODES}):
    obs, _ = env.reset()
    info = {{}}
    for _ in range(800):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    wps.append(info.get('current_waypoint_index', 0))
    holds.append(info.get('hold_progress', 0.0))
    if info.get('goal_reached'):
        succ += 1
print(json.dumps({{'mean_wp': sum(wps)/len(wps), 'max_wp': max(wps),
                  'mean_hold': sum(holds)/len(holds), 'max_hold': max(holds),
                  'successes': succ, 'n': {EVAL_EPISODES}}}))
"""
    r = subprocess.run([PYTHON, "-c", script], capture_output=True, text=True, cwd=ROOT)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {}


def analyse() -> None:
    """Score every completed run and write the ablation report."""
    print(f"\n{'='*72}")
    print("ABLATION ANALYSIS")
    print(f"{'='*72}", flush=True)

    results = {}
    for condition in CONDITIONS:
        cond_dir = OUT_BASE / condition
        if not cond_dir.exists():
            continue
        rows = []
        for seed_dir in sorted(cond_dir.glob("seed_*")):
            stats_p = seed_dir / "training_stats.json"
            stats = json.load(open(stats_p)) if stats_p.exists() else {}
            row = {
                "seed": seed_dir.name,
                "mean_reward": stats.get("mean_reward"),
                "train_hold": stats.get("hold_progress"),
                "final": evaluate(seed_dir / "sac_model"),
                "best": evaluate(seed_dir / "sac_model_best"),
            }
            rows.append(row)
            fin, bst = row["final"], row["best"]
            print(f"  {condition:14s} {row['seed']}  "
                  f"final: wp {fin.get('mean_wp', '-')} hold {fin.get('max_hold', '-')}  "
                  f"best: wp {bst.get('mean_wp', '-')} hold {bst.get('max_hold', '-')}",
                  flush=True)
        if rows:
            results[condition] = rows

    def agg(rows, key, field):
        vals = [r[key].get(field) for r in rows if r[key].get(field) is not None]
        return sum(vals) / len(vals) if vals else None

    report = [
        "# Ablation Study — Terminal-Hold Mechanisms",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"- Seeds per condition: {SEEDS_PER_CONDITION} (matched across conditions)",
        f"- Timesteps per run: {TIMESTEPS:,}",
        f"- Waypoints: {WAYPOINTS_LIST}",
        f"- Evaluation: {EVAL_EPISODES} deterministic episodes at "
        f"{EVAL_TOLERANCE} m tolerance, full 20-step hold",
        "",
        "## Results (best checkpoint, mean over seeds)",
        "",
        "| Condition | Mean waypoints | Max hold progress | Successes | Mean train reward |",
        "|-----------|----------------|-------------------|-----------|-------------------|",
    ]
    for condition, rows in results.items():
        mw = agg(rows, "best", "mean_wp") or agg(rows, "final", "mean_wp")
        mh = agg(rows, "best", "max_hold") or agg(rows, "final", "max_hold")
        sc = sum((r["best"] or r["final"]).get("successes", 0) for r in rows)
        mr = [r["mean_reward"] for r in rows if r["mean_reward"] is not None]
        report.append(
            f"| {condition} | {mw:.2f} | {mh:.2f} | {sc} | "
            f"{(sum(mr)/len(mr)):,.0f} |" if mw is not None and mh is not None
            else f"| {condition} | - | - | {sc} | - |"
        )

    report += [
        "",
        "## Best-checkpoint vs final-model (checkpointing benefit)",
        "",
        "| Condition | Final hold | Best hold | Delta |",
        "|-----------|-----------|-----------|-------|",
    ]
    for condition, rows in results.items():
        fh, bh = agg(rows, "final", "max_hold"), agg(rows, "best", "max_hold")
        if fh is None or bh is None:
            report.append(f"| {condition} | - | - | n/a (no checkpoint) |")
        else:
            report.append(f"| {condition} | {fh:.2f} | {bh:.2f} | {bh-fh:+.2f} |")

    report += ["", "## Per-run detail", "", "```",
               json.dumps(results, indent=2), "```"]

    out = OUT_BASE / "ABLATION_REPORT.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"\nReport written: {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true",
                    help="Skip training; analyse existing results only.")
    args = ap.parse_args()

    if args.analyse:
        analyse()
        return

    jobs = [(c, BASE_SEED + i)
            for i in range(SEEDS_PER_CONDITION)
            for c in CONDITIONS]

    print("=" * 72)
    print("ABLATION STUDY")
    print(f"  Conditions        : {list(CONDITIONS)}")
    print(f"  Seeds/condition   : {SEEDS_PER_CONDITION} (matched)")
    print(f"  Total runs        : {len(jobs)}")
    print(f"  Timesteps each    : {TIMESTEPS:,}")
    print(f"  Parallel          : {MAX_PARALLEL} ({THREADS_PER_RUN} threads each)")
    print(f"  Output            : {OUT_BASE}")
    print("=" * 72, flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futures = {ex.submit(run_one, c, s): (c, s) for c, s in jobs}
        for fut in as_completed(futures):
            fut.result()
    print(f"\nAll runs complete in {(time.time()-t0)/3600:.1f} h", flush=True)

    analyse()


if __name__ == "__main__":
    main()
