"""Train improved waypoint model with EMA velocity, eval-based curriculum, and co-contraction penalty.

Three improvements over the baseline fischer_waypoints_v2_500k:
  1. EMA velocity smoothing: hold criterion uses exponential moving average
     of joint velocity instead of instantaneous, filtering out Hill-type
     muscle tremor (~0.44 rad/s oscillation from antagonist co-contraction).
  2. Eval-based curriculum: periodic deterministic eval episodes decide when
     to tighten position tolerance (0.6m → 0.02m), instead of stochastic
     training policy success rate which systematically underestimates capability.
  3. Co-contraction penalty (P7=-0.08): discourages simultaneous agonist/
     antagonist activation, reducing the root cause of muscle tremor.

Run from the project root:

    /Users/ranjotsandhu/Documents/Project/venv/bin/python scripts/train_improved_waypoints.py
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
    from rl_armMotion.two_d.training.ppo_trainer_wrapper import RLTrainerWithMetrics

    SAVE_DIR = PROJECT_ROOT / "project_assets/outputs/improved_waypoints_500k_v3"
    TOTAL_STEPS = 500_000
    WAYPOINTS = [[2.3, -1.0], [1.0, 1.5], [2.5, 0.0]]
    INIT_TOL = 0.80

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SAVE_DIR / "training_log.txt"

    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    log("=" * 60)
    log("IMPROVED WAYPOINT TRAINING")
    log("=" * 60)
    log(f"Timesteps: {TOTAL_STEPS:,}")
    log(f"Waypoints: {WAYPOINTS}")
    log("Improvements:")
    log("  1. EMA velocity (alpha=0.15) for hold criterion")
    log("  2. Eval-based curriculum (20 deterministic episodes every 10k steps)")
    log("  3. Co-contraction penalty P7=-0.08")
    log(f"Curriculum: eval-based, init_tol={INIT_TOL}, min_tol=0.02, decay=0.85, threshold=70%")

    env = ArmTaskEnv(actuation_mode="muscle", goal_direction="EAST")
    env.set_waypoints(WAYPOINTS, tolerance=INIT_TOL)

    def make_eval_env():
        e = ArmTaskEnv(actuation_mode="muscle", goal_direction="EAST")
        e.set_waypoints(WAYPOINTS, tolerance=INIT_TOL)
        return e

    eval_curriculum = EvalBasedCurriculumCallback(
        eval_env_fn=make_eval_env,
        initial_tolerance=INIT_TOL,
        min_tolerance=0.02,
        success_rate_threshold=0.70,
        decay_factor=0.85,
        eval_freq=10_000,
        n_eval_episodes=20,
        verbose=1,
    )

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        learning_starts=1_000,
        gamma=0.99,
        tau=0.005,
        verbose=1,
    )

    t0 = time.time()
    log("Starting training...")
    model.learn(total_timesteps=TOTAL_STEPS, callback=eval_curriculum)
    train_secs = time.time() - t0
    log(f"Training complete in {train_secs:.1f}s ({train_secs/60:.1f} min)")

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
        "total_timesteps": TOTAL_STEPS,
        "training_time_seconds": train_secs,
        "waypoints": WAYPOINTS,
        "improvements": [
            "EMA velocity (alpha=0.15)",
            "Eval-based curriculum",
            "Co-contraction penalty P7=-0.08",
        ],
        "curriculum_final": curriculum_info,
        "timestamp": datetime.now().isoformat(),
    }
    with open(f"{model_path}_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    log(f"Metadata saved: {model_path}_metadata.pkl")

    log("")
    log("EVALUATION (deterministic, 800 steps, tol=current curriculum)")
    eval_env = make_eval_env()
    eval_env.set_goal_tolerance(curriculum_info["current_tolerance"])
    obs, _ = eval_env.reset()
    total_reward = 0.0
    for step in range(800):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        if step % 50 == 0 or terminated:
            ee = info["end_effector_position"]
            wp_idx = info["current_waypoint_index"]
            gd = info["goal_distance"]
            hc = info["hold_counter"]
            ema_v = info["ema_velocity_norm"]
            cc = info["cocontraction"]
            log(
                f"  step={step:4d}  ee=({ee[0]:.2f},{ee[1]:.2f})  "
                f"wp={wp_idx}/{info['num_waypoints']}  dist={gd:.3f}  "
                f"hold={hc}/{info['hold_steps_required']}  ema_v={ema_v:.3f}  "
                f"cc={cc:.3f}"
            )
        if terminated:
            log(f"  TERMINATED at step {step} — all waypoints complete!")
            break
    else:
        log(f"  Truncated at 800 steps, WPs={info['current_waypoint_index']}/{info['num_waypoints']}")

    log(f"  Total reward: {total_reward:.1f}")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
