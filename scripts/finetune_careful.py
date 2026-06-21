"""Careful fine-tune of fischer_v2 with tiny learning rate + checkpoints.

Key differences from the failed fine-tune:
  - Learning rate 1e-5 (30x smaller) to avoid catastrophic forgetting
  - Checkpoints every 25K steps so we can pick the best snapshot
  - Co-contraction penalty active in env for smoother motion
  - Eval-based curriculum to attempt tolerance tightening
  - Evaluates every checkpoint, not just the final model

Run:
    /Users/ranjotsandhu/Documents/Project/venv/bin/python scripts/finetune_careful.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def evaluate_model(model, waypoints, tolerance, max_steps=800):
    """Run one deterministic evaluation episode, return results dict."""
    from rl_armMotion.two_d.environments.task_env import ArmTaskEnv

    env = ArmTaskEnv(actuation_mode="muscle", goal_direction="EAST")
    env.set_waypoints(waypoints, tolerance=tolerance)
    obs, _ = env.reset()
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            env.close()
            return {
                "completed": True,
                "waypoints": info["num_waypoints"],
                "steps": step,
                "final_dist": info["goal_distance"],
            }
    env.close()
    return {
        "completed": False,
        "waypoints": info["current_waypoint_index"],
        "steps": max_steps,
        "final_dist": info["goal_distance"],
    }


def main() -> int:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from rl_armMotion.two_d.environments.task_env import ArmTaskEnv
    from rl_armMotion.two_d.training.curriculum_callback import EvalBasedCurriculumCallback

    BASE_MODEL = PROJECT_ROOT / "project_assets/outputs/fischer_waypoints_v2_500k/sac_model.zip"
    SAVE_DIR = PROJECT_ROOT / "project_assets/outputs/fischer_v2_careful_finetune"
    FINETUNE_STEPS = 100_000
    WAYPOINTS = [[2.3, -1.0], [1.0, 1.5], [2.5, 0.0]]
    INIT_TOL = 0.60
    LEARNING_RATE = 1e-5

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = SAVE_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    log_path = SAVE_DIR / "training_log.txt"

    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    log("=" * 60)
    log("CAREFUL FINE-TUNE FROM FISCHER V2")
    log("=" * 60)
    log(f"Base model: {BASE_MODEL}")
    log(f"Learning rate: {LEARNING_RATE} (30x smaller than default)")
    log(f"Fine-tune steps: {FINETUNE_STEPS:,}")
    log(f"Checkpoints every 25K steps")
    log(f"Waypoints: {WAYPOINTS}")
    log(f"Curriculum: eval-based, init_tol={INIT_TOL}")

    env = ArmTaskEnv(actuation_mode="muscle", goal_direction="EAST")
    env.set_waypoints(WAYPOINTS, tolerance=INIT_TOL)

    def make_eval_env():
        e = ArmTaskEnv(actuation_mode="muscle", goal_direction="EAST")
        e.set_waypoints(WAYPOINTS, tolerance=INIT_TOL)
        return e

    eval_curriculum = EvalBasedCurriculumCallback(
        eval_env_fn=make_eval_env,
        initial_tolerance=INIT_TOL,
        min_tolerance=0.05,
        success_rate_threshold=0.70,
        decay_factor=0.90,
        eval_freq=10_000,
        n_eval_episodes=20,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=str(checkpoint_dir),
        name_prefix="finetune",
    )

    log("Loading base model...")
    model = SAC.load(BASE_MODEL, env=env)
    model.learning_rate = LEARNING_RATE
    log("Base model loaded, lr set to 1e-5")

    # Evaluate baseline before fine-tuning
    log("")
    log("BASELINE (before fine-tune):")
    for tol in [0.6, 0.4, 0.2]:
        r = evaluate_model(model, WAYPOINTS, tol)
        status = f"3/3 in {r['steps']} steps" if r["completed"] else f"{r['waypoints']}/3 stuck (dist={r['final_dist']:.3f})"
        log(f"  tol={tol}m: {status}")

    t0 = time.time()
    log("")
    log("Starting fine-tuning...")
    model.learn(
        total_timesteps=FINETUNE_STEPS,
        callback=[eval_curriculum, checkpoint_cb],
        reset_num_timesteps=True,
    )
    train_secs = time.time() - t0
    log(f"Fine-tuning complete in {train_secs:.1f}s ({train_secs/60:.1f} min)")

    curriculum_info = eval_curriculum.get_progress()
    log(f"Final tolerance: {curriculum_info['current_tolerance']:.3f}m")
    log(f"Curriculum stage: {curriculum_info['curriculum_stage']}")

    # Save final model
    final_path = str(SAVE_DIR / "sac_model")
    model.save(final_path)
    log(f"Final model saved: {final_path}.zip")

    # Evaluate final model
    log("")
    log("FINAL MODEL (after fine-tune):")
    for tol in [0.6, 0.4, 0.2]:
        r = evaluate_model(model, WAYPOINTS, tol)
        status = f"3/3 in {r['steps']} steps" if r["completed"] else f"{r['waypoints']}/3 stuck (dist={r['final_dist']:.3f})"
        log(f"  tol={tol}m: {status}")

    # Evaluate all checkpoints
    log("")
    log("CHECKPOINT EVALUATION:")
    checkpoint_files = sorted(checkpoint_dir.glob("finetune_*.zip"))
    best_score = -1
    best_checkpoint = None

    for ckpt in checkpoint_files:
        ckpt_model = SAC.load(str(ckpt), env=env)
        ckpt_name = ckpt.stem
        score = 0
        results_str = []
        for tol in [0.6, 0.4, 0.2]:
            r = evaluate_model(ckpt_model, WAYPOINTS, tol)
            if r["completed"]:
                score += 100 - r["steps"] * 0.1
                results_str.append(f"{tol}m:3/3@{r['steps']}")
            else:
                score += r["waypoints"] * 10
                results_str.append(f"{tol}m:{r['waypoints']}/3")
        log(f"  {ckpt_name}: {', '.join(results_str)}  (score={score:.1f})")
        if score > best_score:
            best_score = score
            best_checkpoint = ckpt

    # Also score the original baseline
    orig_model = SAC.load(str(BASE_MODEL), env=env)
    orig_score = 0
    for tol in [0.6, 0.4, 0.2]:
        r = evaluate_model(orig_model, WAYPOINTS, tol)
        if r["completed"]:
            orig_score += 100 - r["steps"] * 0.1
        else:
            orig_score += r["waypoints"] * 10
    log(f"  ORIGINAL baseline: score={orig_score:.1f}")

    # Also score the final model
    final_score = 0
    for tol in [0.6, 0.4, 0.2]:
        r = evaluate_model(model, WAYPOINTS, tol)
        if r["completed"]:
            final_score += 100 - r["steps"] * 0.1
        else:
            final_score += r["waypoints"] * 10
    log(f"  FINAL model: score={final_score:.1f}")

    log("")
    if best_checkpoint and best_score > orig_score:
        log(f"BEST: {best_checkpoint.name} (score={best_score:.1f} vs original {orig_score:.1f})")
        # Copy best checkpoint as the recommended model
        import shutil
        best_out = SAVE_DIR / "best_model.zip"
        shutil.copy2(str(best_checkpoint), str(best_out))
        log(f"Copied to: {best_out}")
    else:
        log(f"Original model is still the best (score={orig_score:.1f})")

    log("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
