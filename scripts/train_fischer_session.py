"""
Long Fischer-style training run script.

Trains SAC with the full Fischer 2021 protocol (adaptive curriculum,
motor babbling) on the 2-DOF arm for the configured number of timesteps,
saves the model, and runs both validation harnesses (Fitts' Law,
2/3 Power Law) on the trained policy, writing JSON and PNG outputs.

Run from the project root with the venv activated:

    python scripts/train_fischer_session.py --timesteps 300000 \
        --save-dir ./project_assets/outputs/fischer_300k_session \
        --goal-direction EAST

The script is deliberately CPU-only and self-contained; it does not
depend on the GUIs.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure the package import works whether the script is invoked from
# the repo root or from anywhere else.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fischer 2021 SAC training session (no GUI)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=300_000,
        help="Total SAC training timesteps (default: 300000).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./project_assets/outputs/fischer_300k_session",
        help="Output directory for model, history, JSON results, and PNG plots.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="SAC",
        choices=["SAC", "PPO", "A2C"],
        help="RL algorithm (default: SAC).",
    )
    parser.add_argument(
        "--goal-direction",
        type=str,
        default="EAST",
        choices=["EAST", "WEST", "NORTH", "HEIGHT"],
        help="Goal direction for the training task (default: EAST).",
    )
    parser.add_argument(
        "--actuation-mode",
        type=str,
        default="velocity",
        choices=["velocity", "muscle"],
        help=(
            "Actuation mode for the training environment. velocity (default) "
            "issues direct joint-velocity commands in [-1, 1] per joint; "
            "muscle issues Hill-type antagonist-pair activations in [0, 1] "
            "per joint, following the Fischer et al. (2021) biomechanical "
            "actuation model."
        ),
    )
    parser.add_argument(
        "--waypoints",
        type=str,
        default=None,
        help=(
            'Optional waypoint sequence as "x1,y1;x2,y2;x3,y3". If supplied, '
            'overrides --goal-direction and trains in waypoint mode. The '
            'agent must visit the waypoints in order; intermediate waypoints '
            'advance on touch (position within tolerance), the final '
            'waypoint requires the full hold criterion. Example: '
            '"2.0,-0.5;1.5,0.8;2.4,0.3" trains a reach-pull-place sequence.'
        ),
    )
    parser.add_argument(
        "--fitts-trials",
        type=int,
        default=10,
        help="Trials per condition for the Fitts' Law sweep (default: 10).",
    )
    parser.add_argument(
        "--power-trials",
        type=int,
        default=20,
        help="Trials for the 2/3 Power Law sweep (default: 20).",
    )
    parser.add_argument(
        "--validator-seed",
        type=int,
        default=42,
        help="RNG seed for both validators (default: 42).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from rl_armMotion.two_d.environments.task_env import ArmTaskEnv
    from rl_armMotion.two_d.training.ppo_trainer_wrapper import RLTrainerWithMetrics
    from rl_armMotion.two_d.validation import FittsLawValidator, PowerLawValidator

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    log_path = save_dir / "training_log.txt"

    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    # Parse waypoints (if any) before logging so the log records them.
    waypoints = None
    if args.waypoints:
        try:
            waypoints = [
                [float(x) for x in pair.split(",")]
                for pair in args.waypoints.split(";")
            ]
            assert all(len(wp) == 2 for wp in waypoints), "each waypoint must have x,y"
            assert len(waypoints) >= 1, "need at least one waypoint"
        except Exception as exc:
            raise ValueError(
                f"could not parse --waypoints {args.waypoints!r}: {exc}. "
                f"Expected format: 'x1,y1;x2,y2;x3,y3'"
            ) from exc

    log(f"Starting Fischer session: algorithm={args.algorithm}, "
        f"timesteps={args.timesteps:,}, direction={args.goal_direction}, "
        f"actuation={args.actuation_mode}"
        + (f", waypoints={waypoints}" if waypoints else ""))
    log(f"Save dir: {save_dir}")

    # --- TRAIN ---------------------------------------------------------
    env = ArmTaskEnv(
        goal_direction=args.goal_direction,
        actuation_mode=args.actuation_mode,
    )
    if waypoints is not None:
        env.set_waypoints(waypoints)
        log(f"Waypoint mode active: {len(waypoints)} waypoints, "
            f"current_waypoint_index=0 at every episode reset")
    trainer = RLTrainerWithMetrics(
        env=env,
        total_timesteps=args.timesteps,
        algorithm=args.algorithm,
        # default: curriculum auto-enabled for SAC, motor babbling via SAC defaults
    )

    t0 = time.time()
    result = trainer.train()
    train_secs = time.time() - t0
    log(f"Training complete in {train_secs:.1f}s")
    log(f"Episodes: {result['final_metrics'].get('episodes', 0):,}")
    log(f"Best reward: {result['final_metrics'].get('best_reward', 0):.2f}")
    log(f"Mean reward (100ep): {result['final_metrics'].get('mean_reward', 0):.2f}")
    log(f"Final goal tolerance: "
        f"{result['final_metrics'].get('curriculum', {}).get('current_tolerance', 'n/a')}")
    log(f"Curriculum stage: "
        f"{result['final_metrics'].get('curriculum', {}).get('curriculum_stage', 'n/a')}")

    # --- SAVE MODEL ----------------------------------------------------
    save_paths = trainer.save_model_and_results(str(save_dir))
    log(f"Saved: {save_paths}")

    # --- VALIDATE: FITTS' LAW -----------------------------------------
    log(f"Running Fitts' Law sweep ({args.fitts_trials} trials per condition)...")
    fl = FittsLawValidator(
        model=trainer.trainer.model,
        env=ArmTaskEnv(actuation_mode=args.actuation_mode),
    )
    t0 = time.time()
    fl_result = fl.run(
        n_trials_per_condition=args.fitts_trials,
        max_steps_per_trial=200,
        seed=args.validator_seed,
    )
    log(f"Fitts sweep complete in {time.time() - t0:.1f}s")
    log(f"Fitts: {fl_result.regression_summary()}")
    fl_json = save_dir / "fitts_law.json"
    fl_png = save_dir / "fitts_law.png"
    fl_result.save_json(str(fl_json))
    fl_result.plot(
        save_path=str(fl_png),
        show=False,
        title=f"Fitts' Law - {args.algorithm} ({args.timesteps:,} steps)",
    )
    log(f"Saved: {fl_json}")
    log(f"Saved: {fl_png}")

    # --- VALIDATE: 2/3 POWER LAW --------------------------------------
    log(f"Running 2/3 Power Law sweep ({args.power_trials} trials)...")
    pl = PowerLawValidator(
        model=trainer.trainer.model,
        env=ArmTaskEnv(actuation_mode=args.actuation_mode),
    )
    t0 = time.time()
    pl_result = pl.run(
        n_trials=args.power_trials,
        max_steps_per_trial=200,
        seed=args.validator_seed,
    )
    log(f"Power Law sweep complete in {time.time() - t0:.1f}s")
    log(f"Power Law: {pl_result.regression_summary()}")
    pl_json = save_dir / "power_law.json"
    pl_png = save_dir / "power_law.png"
    pl_result.save_json(str(pl_json))
    pl_result.plot(
        save_path=str(pl_png),
        show=False,
        title=f"2/3 Power Law - {args.algorithm} ({args.timesteps:,} steps)",
    )
    log(f"Saved: {pl_json}")
    log(f"Saved: {pl_png}")

    elapsed = datetime.now() - start_time
    log(f"Session complete. Total wall time: {elapsed}")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
