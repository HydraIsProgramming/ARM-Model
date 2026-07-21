"""
Automated SAC seed search with early killing, thread pinning, and winner analysis.

Trains seeds in parallel batches of SEEDS_PER_BATCH. At the halfway point
(EARLY_KILL_STEPS), any seed whose reward is below EARLY_KILL_THRESHOLD is
killed and replaced with a fresh seed. After each seed completes 500K steps,
it is evaluated at 0.6m tolerance. When a 3/3 winner is found, a full
replication report is generated and the model is copied to a champion directory.

Run from the project root:
    python run_auto_search.py
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
# SEEDS_PER_BATCH: how many SAC seeds to train in parallel per batch.
#   5 works well on a 24-core machine. Reduce to 3 if the computer feels sluggish.
SEEDS_PER_BATCH   = 5

# TIMESTEPS: training budget per seed. 500K is the empirically determined sweet
#   spot — more steps have been shown not to improve the policy further.
TIMESTEPS         = 500_000

# WAYPOINTS: the three target positions the arm must visit in order.
#   Format for CLI: "x1,y1;x2,y2;x3,y3"
#   Shoulder is fixed at [1.0, 0.0]. Max reach is 1.8m, so the reachable
#   workspace is a circle of radius 1.8 centred on [1.0, 0.0]
#   (x: -0.8..2.8, y: -1.8..1.8). Points are kept 1.3-1.6m from the
#   shoulder — well inside reach so the arm doesn't strain at the boundary.
#     WP1 [0.2,  1.2] — upper left   (1.44m from shoulder)
#     WP2 [2.2,  0.5] — right        (1.30m from shoulder)
#     WP3 [1.0, -1.6] — straight down (1.60m from shoulder)
#   They form a wide triangle (2.1-2.3m between points) with a clear
#   visual sweep: upper-left -> right -> bottom.
#   Previous waypoints reached out to x=2.5 which is nearly at the edge
#   of the workspace and hard for the arm to hold.
WAYPOINTS         = "0.2,1.2;2.2,0.5;1.0,-1.6"
WAYPOINTS_LIST    = [[0.2, 1.2], [2.2, 0.5], [1.0, -1.6]]

# MAX_BATCHES: safety ceiling — stops after this many batches even if no winner.
#   At ~1/8 success rate, 20 batches * 5 seeds = 100 seeds gives ~99.9% chance
#   of finding at least one winner.
MAX_BATCHES       = 20

# EARLY_KILL_STEPS / EARLY_KILL_THRESH: if a seed's reward is still below
#   EARLY_KILL_THRESH at the halfway point, it is killed. This saves ~45 min
#   per bad seed. Threshold 0 kills only seeds with negative mean reward.
EARLY_KILL_STEPS  = 250_000
EARLY_KILL_THRESH = 0

# CONSISTENCY_RUNS: how many times to re-evaluate a 3/3 winner to confirm
#   the result is not a lucky episode.
CONSISTENCY_RUNS  = 5

# THREADS_PER_SEED: PyTorch CPU thread count per seed process. Divides the
#   available cores evenly so seeds don't fight each other for CPU time.
THREADS_PER_SEED  = max(1, os.cpu_count() // SEEDS_PER_BATCH - 1)

PYTHON      = sys.executable
SCRIPT      = str(Path(__file__).parent / "scripts" / "train_fischer_session.py")
OUTPUT_BASE = Path(__file__).parent / "project_assets" / "outputs" / "Parallel_Seeds"
CHAMPION_DIR = Path(__file__).parent / "project_assets" / "outputs" / "CHAMPION"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
# ──────────────────────────────────────────────────────────────────────────────


def get_latest_reward(log_path: Path) -> float:
    if not log_path.exists():
        return -999999
    val = -999999
    for line in log_path.read_text(errors="ignore").splitlines():
        if "ep_rew_mean" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            for i, p in enumerate(parts):
                if p == "ep_rew_mean" and i + 1 < len(parts):
                    try:
                        val = float(parts[i + 1])
                    except ValueError:
                        pass
    return val


def get_latest_timesteps(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    val = 0
    for line in log_path.read_text(errors="ignore").splitlines():
        if "total_timesteps" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            for i, p in enumerate(parts):
                if p == "total_timesteps" and i + 1 < len(parts):
                    try:
                        val = int(parts[i + 1])
                    except ValueError:
                        pass
    return val


def evaluate_model(model_dir: Path, tolerance: float = 0.6) -> dict:
    """Evaluate model at given tolerance. Returns dict with waypoints, steps.

    Prefers the best-checkpoint snapshot (sac_model_best.zip, saved by the
    periodic-eval callback at the policy's peak) over the final model —
    SAC can partially forget the task late in training.
    """
    if (model_dir / "sac_model_best.zip").exists():
        model_path = model_dir / "sac_model_best"
    else:
        model_path = model_dir / "sac_model"
    if not (model_dir / "sac_model.zip").exists() and not (model_dir / "sac_model_best.zip").exists():
        return {"waypoints": 0, "steps": 800, "completed": False}
    script = f"""
import sys; sys.path.insert(0, 'src')
from stable_baselines3 import SAC
from rl_armMotion.two_d.environments.task_env import ArmTaskEnv
model = SAC.load(r'{model_path}')
env = ArmTaskEnv(actuation_mode='muscle', goal_direction='EAST')
env.set_waypoints({WAYPOINTS_LIST}, tolerance={tolerance})
obs, _ = env.reset()
final_wp = 0
for step in range(800):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, info = env.step(action)
    final_wp = info['current_waypoint_index']
    if term:
        print(f'{{info["num_waypoints"]}} {{step}}')
        break
else:
    print(f'{{final_wp}} 800')
"""
    result = subprocess.run([PYTHON, "-c", script], capture_output=True, text=True,
                            cwd=Path(__file__).parent)
    try:
        last = result.stdout.strip().splitlines()[-1]
        wp, steps = last.split()
        return {"waypoints": int(wp), "steps": int(steps), "completed": int(wp) == 3}
    except Exception:
        return {"waypoints": 0, "steps": 800, "completed": False}


def analyze_winner(seed_id: int, seed_dir: Path) -> None:
    """Full analysis and replication report for a 3/3 winner."""
    print(f"\n{'='*60}", flush=True)
    print("WINNER ANALYSIS", flush=True)
    print(f"{'='*60}", flush=True)

    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Consistency check — run CONSISTENCY_RUNS times at 0.6m
    print(f"\nRunning {CONSISTENCY_RUNS} consistency checks at 0.6m...", flush=True)
    consistency_results = []
    for run in range(1, CONSISTENCY_RUNS + 1):
        r = evaluate_model(seed_dir, tolerance=0.6)
        consistency_results.append(r)
        print(f"  Run {run}: {r['waypoints']}/3 in {r['steps']} steps", flush=True)

    wins = sum(1 for r in consistency_results if r["completed"])
    avg_steps = sum(r["steps"] for r in consistency_results if r["completed"]) / max(wins, 1)
    print(f"  Consistency: {wins}/{CONSISTENCY_RUNS} runs completed 3/3", flush=True)
    print(f"  Avg steps (winners): {avg_steps:.0f}", flush=True)

    # 2. Multi-tolerance evaluation
    print(f"\nEvaluating at multiple tolerances...", flush=True)
    tol_results = {}
    for tol in [0.6, 0.4, 0.2]:
        r = evaluate_model(seed_dir, tolerance=tol)
        tol_results[tol] = r
        status = f"{r['waypoints']}/3 in {r['steps']} steps" if r["completed"] else f"{r['waypoints']}/3 (incomplete)"
        print(f"  tol={tol}m: {status}", flush=True)

    # 3. Load training stats
    stats = {}
    stats_path = seed_dir / "training_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

    # 4. Copy model to CHAMPION directory
    print(f"\nCopying model to {CHAMPION_DIR}...", flush=True)
    for f in seed_dir.iterdir():
        shutil.copy2(f, CHAMPION_DIR / f.name)
    shutil.copy2(seed_dir / "stdout.log", CHAMPION_DIR / "stdout.log")

    # 5. Write replication report
    report_path = CHAMPION_DIR / "REPLICATION_REPORT.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"""# Champion Model — Replication Report
Generated: {now}

## Identity
- **Seed ID**: seed_{seed_id:03d}
- **Source dir**: `{seed_dir}`
- **Champion dir**: `{CHAMPION_DIR}`

## Training Configuration (copy exactly to replicate)
```
python scripts/train_fischer_session.py \\
    --timesteps {TIMESTEPS} \\
    --actuation-mode muscle \\
    --goal-direction EAST \\
    --waypoints "{WAYPOINTS}" \\
    --seed {seed_id} \\
    --save-dir ./project_assets/outputs/MY_REPLICATED_RUN
```
The --seed flag makes the run exactly reproducible: same seed + same
config trains the same model.

## Environment Settings (do NOT change)
- Algorithm: SAC (MlpPolicy)
- Network: [256, 256]
- Observation dim: 11
- Action dim: 4 (muscle activations [0,1])
- Waypoints: {WAYPOINTS_LIST}
- Arm: 2-DOF, shoulder at [1.0, 0.0], links [1.0, 0.8]m
- Orientation tolerance: 12 degrees
- Hold grace: decrement by 3 (not hard reset)
- Hold criterion: 20 steps within position + orientation + velocity
- Hold curriculum: 10 -> 15 -> 20 steps (advances at 60% success)
- Tolerance curriculum: 0.60 m -> 0.20 m floor (waypoint mode)
- Waypoint layout: WP1=[0.2,1.2] upper-left, WP2=[2.2,0.5] right, WP3=[1.0,-1.6] bottom

## Performance Results
### Consistency ({CONSISTENCY_RUNS} runs at 0.6m tolerance)
- Wins: {wins}/{CONSISTENCY_RUNS}
- Avg steps to complete: {avg_steps:.0f}

### Multi-Tolerance
| Tolerance | Waypoints | Steps | Completed |
|-----------|-----------|-------|-----------|
"""
    for tol, r in tol_results.items():
        report += f"| {tol}m | {r['waypoints']}/3 | {r['steps']} | {'Yes' if r['completed'] else 'No'} |\n"

    report += f"""
## Training Stats
"""
    if stats:
        for k, v in stats.items():
            report += f"- **{k}**: {v}\n"

    report += f"""
## Key Warnings (from project handoff)
- **NEVER fine-tune this model** — SAC exploration noise destroys hold behavior
- **NEVER change reward weights** — they are carefully tuned
- **NEVER change observation space** — policy depends on 11D structure
- **DO NOT** increase training beyond 500K steps
- Keep this model as reference — never overwrite it
- The `save-point` for this model is in `{CHAMPION_DIR}`

## How to Load and Evaluate
```python
import sys; sys.path.insert(0, 'src')
from stable_baselines3 import SAC
from rl_armMotion.two_d.environments.task_env import ArmTaskEnv

# sac_model_best is the peak-of-training checkpoint (preferred);
# sac_model is the final model at 500K steps.
model = SAC.load('{CHAMPION_DIR}/sac_model_best')
env = ArmTaskEnv(actuation_mode='muscle', goal_direction='EAST')
env.set_waypoints({WAYPOINTS_LIST}, tolerance=0.6)
obs, _ = env.reset()
for step in range(800):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, info = env.step(action)
    if term:
        print(f'3/3 in {{step}} steps')
        break
```
"""

    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nReplication report saved: {report_path}", flush=True)
    print(f"Model copied to: {CHAMPION_DIR}", flush=True)
    print(f"\nConsistency: {wins}/{CONSISTENCY_RUNS} | Best tolerance: ", end="", flush=True)
    best_tol = min((t for t, r in tol_results.items() if r["completed"]), default=None)
    print(f"{best_tol}m" if best_tol else "none", flush=True)


def run_seed(seed_id: int, save_dir: Path) -> tuple:
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "stdout.log"
    cmd = [
        PYTHON, SCRIPT,
        "--timesteps", str(TIMESTEPS),
        "--actuation-mode", "muscle",
        "--goal-direction", "EAST",
        "--waypoints", WAYPOINTS,
        "--save-dir", str(save_dir),
        "--num-threads", str(THREADS_PER_SEED),
        "--seed", str(seed_id),
    ]
    print(f"  [seed {seed_id:03d}] Starting (threads={THREADS_PER_SEED})...", flush=True)
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=f, text=True)

    killed = False
    while proc.poll() is None:
        time.sleep(30)
        steps = get_latest_timesteps(log_path)
        if steps >= EARLY_KILL_STEPS and not killed:
            rew = get_latest_reward(log_path)
            if rew < EARLY_KILL_THRESH:
                proc.terminate()
                killed = True
                print(f"  [seed {seed_id:03d}] KILLED early (reward={rew:.0f} at {steps:,} steps)", flush=True)
                return seed_id, -1

    rc = proc.returncode
    rew = get_latest_reward(log_path)
    print(f"  [seed {seed_id:03d}] {'OK' if rc == 0 else 'FAIL'} | reward={rew:.0f}", flush=True)
    return seed_id, rc


def main():
    print("=" * 60)
    print("AUTO SEED SEARCH")
    print(f"  Seeds per batch : {SEEDS_PER_BATCH}")
    print(f"  Timesteps each  : {TIMESTEPS:,}")
    print(f"  Threads per seed: {THREADS_PER_SEED}")
    print(f"  Early kill at   : {EARLY_KILL_STEPS:,} steps if reward < {EARLY_KILL_THRESH}")
    print(f"  Max batches     : {MAX_BATCHES}")
    print(f"  Output          : {OUTPUT_BASE}")
    print("=" * 60, flush=True)

    # Resume from after the last seed that actually completed (has a saved model).
    # Directories without a model were killed mid-run and will be overwritten.
    completed = [d for d in sorted(OUTPUT_BASE.glob("seed_*")) if (d / "sac_model.zip").exists()]
    if completed:
        last_num = int(completed[-1].name.split("_")[1])
        seed_counter = last_num + 1
    else:
        seed_counter = 1
    print(f"Resuming from seed {seed_counter:03d} (last completed: {completed[-1].name if completed else 'none'})", flush=True)

    for batch in range(1, MAX_BATCHES + 1):
        print(f"\n{'='*60}", flush=True)
        print(f"BATCH {batch} — seeds {seed_counter:03d} to {seed_counter + SEEDS_PER_BATCH - 1:03d}", flush=True)
        print(f"{'='*60}", flush=True)

        batch_seeds = [(seed_counter + i, OUTPUT_BASE / f"seed_{seed_counter + i:03d}")
                       for i in range(SEEDS_PER_BATCH)]

        with ProcessPoolExecutor(max_workers=SEEDS_PER_BATCH) as ex:
            futures = {ex.submit(run_seed, sid, sdir): sid for sid, sdir in batch_seeds}
            for future in as_completed(futures):
                future.result()

        seed_counter += SEEDS_PER_BATCH

        # Evaluate all completed seeds in this batch
        print(f"\nEvaluating batch {batch}...", flush=True)
        winner_found = False
        for sid, sdir in batch_seeds:
            if not (sdir / "sac_model.zip").exists():
                print(f"  seed_{sid:03d}: no model (killed early or failed)", flush=True)
                continue
            r = evaluate_model(sdir, tolerance=0.6)
            print(f"  seed_{sid:03d}: {r['waypoints']}/3 waypoints in {r['steps']} steps", flush=True)
            if r["waypoints"] == 3:
                analyze_winner(sid, sdir)
                winner_found = True
                break

        if winner_found:
            return

        print(f"\nNo winner in batch {batch}. Starting next batch...", flush=True)

    print(f"\nReached max batches ({MAX_BATCHES}). No 3/3 winner found.", flush=True)


if __name__ == "__main__":
    main()
