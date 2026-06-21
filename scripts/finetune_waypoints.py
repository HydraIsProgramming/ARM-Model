"""Fine-tune the best waypoint model with curriculum tightening.

Loads the existing fischer_waypoints_v2_500k model and continues training
for additional timesteps. The corrected velocity tolerance (0.50 rad/s)
means the agent now achieves successful holds, so the adaptive curriculum
can finally tighten the position tolerance from 0.6 m toward 0.02 m.

Run from the project root:

    python scripts/finetune_waypoints.py
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
    from rl_armMotion.two_d.training.curriculum_callback import AdaptiveCurriculumCallback
    from rl_armMotion.two_d.training.ppo_trainer_wrapper import TrainingGUICallback, RLTrainerWithMetrics
    from rl_armMotion.two_d.validation import FittsLawValidator, PowerLawValidator

    BASE_MODEL = PROJECT_ROOT / "project_assets/outputs/fischer_waypoints_v2_500k/sac_model"
    SAVE_DIR = PROJECT_ROOT / "project_assets/outputs/fischer_waypoints_finetuned"
    FINETUNE_STEPS = 300_000
    WAYPOINTS = [[2.3, -1.0], [1.0, 1.5], [2.5, 0.0]]

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SAVE_DIR / "training_log.txt"

    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    log(f"Fine-tuning from: {BASE_MODEL}")
    log(f"Additional timesteps: {FINETUNE_STEPS:,}")
    log(f"Waypoints: {WAYPOINTS}")
    log(f"Velocity tolerance: 0.50 rad/s (corrected for muscle dynamics)")
    log(f"Curriculum: active, initial_tol=0.60, min_tol=0.02, decay=0.80, threshold=80%")

    env = ArmTaskEnv(actuation_mode="muscle", goal_direction="EAST")
    env.set_waypoints(WAYPOINTS)

    trainer = RLTrainerWithMetrics(
        env=env,
        total_timesteps=FINETUNE_STEPS,
        algorithm="SAC",
        preload_model_path=str(BASE_MODEL),
        use_curriculum=True,
        curriculum_kwargs={
            "initial_tolerance": 0.60,
            "min_tolerance": 0.02,
            "success_rate_threshold": 0.80,
            "decay_factor": 0.80,
            "window_size": 50,
            "min_episodes_before_decay": 20,
            "verbose": 1,
        },
    )

    t0 = time.time()
    log("Starting fine-tuning...")
    result = trainer.train()
    train_secs = time.time() - t0

    log(f"Fine-tuning complete in {train_secs:.1f}s")
    log(f"Episodes: {result.get('final_metrics', {}).get('episodes', 0):,}")
    log(f"Best reward: {result.get('final_metrics', {}).get('best_reward', 0):.2f}")
    log(f"Mean reward (100ep): {result.get('final_metrics', {}).get('mean_reward', 0):.2f}")

    curriculum_info = result.get("final_metrics", {}).get("curriculum", {})
    log(f"Final goal tolerance: {curriculum_info.get('current_tolerance', 'n/a')}")
    log(f"Curriculum stage: {curriculum_info.get('curriculum_stage', 'n/a')}")
    log(f"Recent success rate: {curriculum_info.get('recent_success_rate', 'n/a')}")

    save_paths = trainer.save_model_and_results(str(SAVE_DIR))
    log(f"Saved: {save_paths}")

    log(f"Running Fitts' Law sweep (10 trials per condition)...")
    fl = FittsLawValidator(
        model=trainer.trainer.model,
        env=ArmTaskEnv(actuation_mode="muscle"),
    )
    t0 = time.time()
    fl_result = fl.run(n_trials_per_condition=10, max_steps_per_trial=200, seed=42)
    log(f"Fitts sweep complete in {time.time() - t0:.1f}s")
    log(f"Fitts: {fl_result.regression_summary()}")
    fl_result.save_json(str(SAVE_DIR / "fitts_law.json"))
    fl_result.plot(save_path=str(SAVE_DIR / "fitts_law.png"), show=False,
                   title="Fitts' Law - SAC Fine-tuned")
    log(f"Saved: {SAVE_DIR / 'fitts_law.json'}")
    log(f"Saved: {SAVE_DIR / 'fitts_law.png'}")

    log(f"Running 2/3 Power Law sweep (20 trials)...")
    pl = PowerLawValidator(
        model=trainer.trainer.model,
        env=ArmTaskEnv(actuation_mode="muscle"),
    )
    t0 = time.time()
    pl_result = pl.run(n_trials=20, max_steps_per_trial=200, seed=42)
    log(f"Power Law sweep complete in {time.time() - t0:.1f}s")
    log(f"Power Law: {pl_result.regression_summary()}")
    pl_result.save_json(str(SAVE_DIR / "power_law.json"))
    pl_result.plot(save_path=str(SAVE_DIR / "power_law.png"), show=False,
                   title="2/3 Power Law - SAC Fine-tuned")
    log(f"Saved: {SAVE_DIR / 'power_law.json'}")
    log(f"Saved: {SAVE_DIR / 'power_law.png'}")

    log(f"Session complete. Total wall time: {datetime.now() - datetime.fromisoformat(log_path.read_text().split(']')[0][1:])}")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
