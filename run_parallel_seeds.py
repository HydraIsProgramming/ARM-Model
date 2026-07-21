"""
Multi-seed parallel SAC training launcher.
Runs NUM_SEEDS independent training jobs in parallel, then prints a summary.
After this completes, run: python scripts/pick_best_seed.py
"""

import subprocess
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

NUM_SEEDS = 5
TIMESTEPS = 500_000
WAYPOINTS = "2.3,-1.0;1.0,1.5;2.5,0.0"
PYTHON = sys.executable
SCRIPT = str(Path(__file__).parent / "scripts" / "train_fischer_session.py")
OUTPUT_BASE = Path(__file__).parent / "project_assets" / "outputs" / "Parallel_Seeds"


def run_seed(seed_id: int):
    save_dir = str(OUTPUT_BASE / f"seed_{seed_id:02d}")
    cmd = [
        PYTHON, SCRIPT,
        "--timesteps", str(TIMESTEPS),
        "--actuation-mode", "muscle",
        "--goal-direction", "EAST",
        "--waypoints", WAYPOINTS,
        "--save-dir", save_dir,
    ]
    log_path = OUTPUT_BASE / f"seed_{seed_id:02d}" / "stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[seed {seed_id:02d}] Starting... (tail -f {log_path})", flush=True)
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=log_file, text=True)
    status = "OK" if result.returncode == 0 else "FAIL"
    print(f"[seed {seed_id:02d}] {status}", flush=True)
    return seed_id, result.returncode


if __name__ == "__main__":
    max_workers = min(NUM_SEEDS, os.cpu_count() - 2)
    print(f"Launching {NUM_SEEDS} seeds with {max_workers} parallel workers")
    print(f"Waypoints: {WAYPOINTS}")
    print(f"Steps per seed: {TIMESTEPS:,}")
    print(f"Output: {OUTPUT_BASE}")
    print("-" * 50, flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_seed, i): i for i in range(1, NUM_SEEDS + 1)}
        for future in as_completed(futures):
            seed_id, rc = future.result()
            results.append((seed_id, rc))

    results.sort()
    ok = sum(1 for _, rc in results if rc == 0)
    fail = NUM_SEEDS - ok
    print("\n" + "=" * 50)
    print(f"Done: {ok}/{NUM_SEEDS} completed OK, {fail} failed")
    print("Next step: python scripts/pick_best_seed.py")
    print("=" * 50)
