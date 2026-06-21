"""Evaluate all multi-seed training runs and rank them.

Run after multi_seed_train.sh completes:

    /Users/ranjotsandhu/Documents/Project/venv/bin/python scripts/pick_best_seed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def evaluate_model(model_path: str, tolerance: float = 0.6) -> dict:
    from stable_baselines3 import SAC
    from rl_armMotion.two_d.environments.task_env import ArmTaskEnv

    model = SAC.load(model_path)
    waypoints = [[2.3, -1.0], [1.0, 1.5], [2.5, 0.0]]

    results = {}
    for tol in [0.6, 0.4, 0.2]:
        env = ArmTaskEnv(actuation_mode="muscle", goal_direction="EAST")
        env.set_waypoints(waypoints, tolerance=tol)
        obs, _ = env.reset()
        completed_steps = None
        final_wp = 0
        for step in range(800):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            final_wp = info["current_waypoint_index"]
            if terminated:
                completed_steps = step
                final_wp = info["num_waypoints"]
                break
        env.close()
        results[f"tol_{tol}"] = {
            "waypoints_completed": final_wp,
            "total_waypoints": 3,
            "steps": completed_steps if completed_steps else 800,
            "completed": completed_steps is not None,
        }
    return results


def main() -> int:
    base_dir = PROJECT_ROOT / "project_assets/outputs"
    seed_dirs = sorted(base_dir.glob("multiseed_run_*"))

    if not seed_dirs:
        print("No multiseed_run_* directories found. Run multi_seed_train.sh first.")
        return 1

    print("=" * 70)
    print("MULTI-SEED EVALUATION")
    print("=" * 70)

    all_results = []
    for seed_dir in seed_dirs:
        model_path = seed_dir / "sac_model.zip"
        if not model_path.exists():
            print(f"\n{seed_dir.name}: NO MODEL (training may still be running)")
            continue

        print(f"\n{seed_dir.name}:")
        try:
            results = evaluate_model(str(seed_dir / "sac_model"))
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        score = 0
        for tol_key, r in results.items():
            wp = r["waypoints_completed"]
            steps = r["steps"]
            done = "DONE" if r["completed"] else f"stuck at {wp}/3"
            print(f"  {tol_key}: {wp}/3 in {steps} steps ({done})")
            if r["completed"]:
                score += 100 - steps * 0.1
            else:
                score += wp * 10

        all_results.append((seed_dir.name, score, results))
        print(f"  Score: {score:.1f}")

    if not all_results:
        print("\nNo models to evaluate.")
        return 1

    all_results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 70)
    print("RANKING (higher score = better)")
    print("=" * 70)
    for rank, (name, score, _) in enumerate(all_results, 1):
        marker = " <-- BEST" if rank == 1 else ""
        print(f"  #{rank}: {name} (score={score:.1f}){marker}")

    best_name = all_results[0][0]
    best_path = base_dir / best_name / "sac_model.zip"
    print(f"\nBest model: {best_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
