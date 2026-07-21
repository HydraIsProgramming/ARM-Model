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
        "--num-threads",
        type=int,
        default=0,
        help="PyTorch CPU thread count (0 = use all available, set to cores/num_seeds for parallel runs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "RNG seed for torch/numpy/random. When set, the run is exactly "
            "reproducible: the same seed + same config always trains the same "
            "model. Leave unset for uncontrolled (random) initialisation."
        ),
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
    """End-to-end Fischer 2021 training session.

    Order of operations:
      1. Parse CLI flags (algorithm, timestep budget, actuation mode,
         goal direction or explicit waypoints, validator parameters,
         save directory).
      2. Construct the Gymnasium environment with the requested
         actuation and goal mode.
      3. Wrap the env in the RLTrainerWithMetrics, which auto-attaches
         the AdaptiveCurriculumCallback for SAC (so motor babbling +
         curriculum decay both happen during training without any
         further setup).
      4. Train.
      5. Save the trained model, training history CSV, and stats JSON.
      6. Run the Fitts' Law validator against the trained policy.
      7. Run the 2/3 Power Law validator against the trained policy.
      8. Save JSON + PNG outputs for both validators.
      9. Log the regression summary lines so the saved training_log.txt
         is self-documenting.

    All output paths are absolute and reported in the log so the run is
    fully reproducible.
    """
    args = parse_args()

    # Local imports so a quick --help does not require torch / SB3 to be on
    # the path (useful when triaging a misconfigured environment).
    import torch
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    # Seed everything for reproducibility. This is what makes "seed_007" a
    # real seed: rerunning with --seed 7 reproduces the exact same model.
    if args.seed is not None:
        import random
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    from rl_armMotion.two_d.environments.task_env import ArmTaskEnv
    from rl_armMotion.two_d.training.ppo_trainer_wrapper import RLTrainerWithMetrics
    from rl_armMotion.two_d.validation import FittsLawValidator, PowerLawValidator

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    log_path = save_dir / "training_log.txt"

    def log(msg: str) -> None:
        """Append a timestamped line to both stdout and the training log file.

        Using both makes the script equally usable interactively (you see the
        log scroll on stdout) and as a background task (the log file persists
        if the process is detached, and is the only artefact if the parent
        terminal goes away).
        """
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
        + (f", waypoints={waypoints}" if waypoints else "")
        + (f", seed={args.seed}" if args.seed is not None else ", seed=unset"))
    log(f"Save dir: {save_dir}")

    # --- TRAIN ---------------------------------------------------------
    # The env is the central object — both the trainer and the validators
    # need a copy of it. We construct it once here with the requested
    # actuation mode (velocity or muscle) and goal mode (direction-based
    # via goal_direction, or arbitrary waypoint sequence via set_waypoints).
    env = ArmTaskEnv(
        goal_direction=args.goal_direction,
        actuation_mode=args.actuation_mode,
    )
    if waypoints is not None:
        # Override the directional goal with a sequence of explicit waypoints.
        # The agent must visit them in order; intermediate waypoints advance
        # on touch, the final waypoint requires the full hold criterion.
        env.set_waypoints(waypoints)
        log(f"Waypoint mode active: {len(waypoints)} waypoints, "
            f"current_waypoint_index=0 at every episode reset")

    # RLTrainerWithMetrics wraps Stable-Baselines3 with two things:
    #   1. A live metrics stream usable by the training GUI (we ignore it
    #      here because we are headless, but it is also the same data
    #      that ends up in the saved training_history.csv).
    #   2. Auto-attach of the AdaptiveCurriculumCallback when the algorithm
    #      is SAC — which is the default. The callback shrinks the goal
    #      tolerance during training following Fischer's protocol; see
    #      curriculum_callback.py for details.
    # In waypoint mode, stop the tolerance curriculum at 0.20 m instead of
    # Fischer's 0.02 m precision floor. Winners are evaluated at 0.6 m, so
    # training below ~0.2 m buys no evaluated precision while making even
    # the touch-and-go intermediate waypoints brutally hard — seeds burn
    # their remaining budget failing at a precision the task never tests.
    curriculum_kwargs = {"min_tolerance": 0.20} if waypoints is not None else None
    if curriculum_kwargs:
        log(f"Waypoint mode: tolerance curriculum floor raised to "
            f"{curriculum_kwargs['min_tolerance']} m (eval standard is 0.6 m)")

    # Eval env factory for the checkpoint/curriculum callback: a fresh env
    # configured exactly like the training env. Periodic deterministic
    # evaluation on this env (every 25K steps, production standard: 0.6 m
    # tolerance + full 20-step hold) drives best-model checkpointing
    # (sac_model_best.zip) and both curricula. Only used in waypoint mode —
    # directional goals keep the classic stochastic-success curriculum.
    eval_env_fn = None
    if waypoints is not None:
        _wp = [list(w) for w in waypoints]

        def eval_env_fn():
            e = ArmTaskEnv(
                goal_direction=args.goal_direction,
                actuation_mode=args.actuation_mode,
            )
            e.set_waypoints(_wp, tolerance=0.6)
            return e

        log("Best-model checkpointing active: deterministic eval every 25K "
            "steps, best snapshot saved to sac_model_best.zip")

    trainer = RLTrainerWithMetrics(
        env=env,
        total_timesteps=args.timesteps,
        algorithm=args.algorithm,
        curriculum_kwargs=curriculum_kwargs,
        eval_env_fn=eval_env_fn,
        eval_save_dir=str(save_dir) if eval_env_fn is not None else None,
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
    if trainer.eval_checkpoint_callback is not None:
        cp = trainer.eval_checkpoint_callback.get_progress()
        log(f"Best checkpoint: score={cp.get('best_score')} "
            f"at step {cp.get('best_saved_at')} -> "
            f"{save_dir / (args.algorithm.lower() + '_model_best.zip')}")

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
