"""Fine-tune from the proven fischer_waypoints_v2_500k model.

Loads the original model that achieves 3/3 waypoints in ~181 steps at 0.6m
tolerance, then continues training with eval-based curriculum to progressively
tighten tolerance. The model already succeeds deterministically at 0.6m, so
the curriculum should detect that and begin decaying.

Run from the project root:

    /Users/ranjotsandhu/Documents/Project/venv/bin/python scripts/finetune_from_v2.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    from stable_baselines3 import SAC
    from rl_armMotion.two_d.environments.task_env import ArmTaskEnv
    from rl_armMotion.two_d.training.curriculum_callback import EvalBasedCurriculumCallback

    BASE_MODEL = PROJECT_ROOT / "project_assets/outputs/fischer_waypoints_v2_500k/sac_model.zip"
    SAVE_DIR = PROJECT_ROOT / "project_assets/outputs/fischer_v2_finetuned_v2"
    FINETUNE_STEPS = 200_000
    WAYPOINTS = [[2.3, -1.0], [1.0, 1.5], [2.5, 0.0]]
    INIT_TOL = 0.60

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SAVE_DIR / "training_log.txt"

    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    log("=" * 60)
    log("FINE-TUNE FROM FISCHER V2 BASELINE")
    log("=" * 60)
    log(f"Base model: {BASE_MODEL}")
    log(f"Fine-tune steps: {FINETUNE_STEPS:,}")
    log(f"Waypoints: {WAYPOINTS}")
    log(f"Curriculum: eval-based, init_tol={INIT_TOL}, min_tol=0.05, decay=0.85, threshold=70%")

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
        decay_factor=0.85,
        eval_freq=10_000,
        n_eval_episodes=20,
        verbose=1,
    )

    log("Loading base model...")
    model = SAC.load(BASE_MODEL, env=env)
    log("Base model loaded successfully")

    t0 = time.time()
    log("Starting fine-tuning...")
    model.learn(total_timesteps=FINETUNE_STEPS, callback=eval_curriculum, reset_num_timesteps=False)
    train_secs = time.time() - t0
    log(f"Fine-tuning complete in {train_secs:.1f}s ({train_secs/60:.1f} min)")

    curriculum_info = eval_curriculum.get_progress()
    log(f"Final tolerance: {curriculum_info['current_tolerance']:.3f}m")
    log(f"Curriculum stage: {curriculum_info['curriculum_stage']}")
    log(f"Last eval success rate: {curriculum_info['recent_success_rate']}")

    model_path = str(SAVE_DIR / "sac_model")
    model.save(model_path)
    log(f"Model saved: {model_path}.zip")

    import pickle
    metadata = {
        "algorithm": "SAC",
        "base_model": str(BASE_MODEL),
        "finetune_timesteps": FINETUNE_STEPS,
        "training_time_seconds": train_secs,
        "waypoints": WAYPOINTS,
        "curriculum_final": curriculum_info,
        "timestamp": datetime.now().isoformat(),
    }
    with open(f"{model_path}_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    # Evaluation at multiple tolerances
    for tol in [0.6, 0.4, 0.2, 0.1]:
        log(f"")
        log(f"EVAL at tolerance={tol}m (800 steps, deterministic)")
        eval_env = make_eval_env()
        eval_env.set_goal_tolerance(tol)
        obs, _ = eval_env.reset()
        for step in range(800):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if step % 50 == 0 or terminated:
                ee = info["end_effector_position"]
                wp_idx = info["current_waypoint_index"]
                gd = info["goal_distance"]
                hc = info["hold_counter"]
                log(
                    f"  step={step:4d}  ee=({ee[0]:.2f},{ee[1]:.2f})  "
                    f"wp={wp_idx}/{info['num_waypoints']}  dist={gd:.3f}  "
                    f"hold={hc}/{info['hold_steps_required']}"
                )
            if terminated:
                log(f"  COMPLETED at step {step} -- all waypoints done!")
                break
        else:
            log(f"  Truncated at 800 steps, WPs={info['current_waypoint_index']}/{info['num_waypoints']}")
        eval_env.close()

    log("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
