"""
Train one SAC policy on the human-arm waypoint task and write a model package.

    python scripts/train_fischer_session.py --timesteps 300000 --waypoints 2 --seed 1 \
        --save-dir project_assets/outputs/stageA/seed_1

The run directory becomes a self-contained package (run_manifest.json,
sac_model_best.zip, sac_model_last.zip, replay_buffer.pkl,
curriculum_state.json, best_eval.json, checkpoints/) that
rl_armMotion.two_d.evaluation.load_package can reload, evaluate or demo, and
that ``--continue-from`` can train further into a NEW package.

Reproducibility contract: same --seed, --num-threads, library versions and
manifest => identical weights (tests/test_reproducibility.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAC training run on the human-arm waypoint task")
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=1, help="Seeds SAC, the env, the action space and torch.")
    p.add_argument("--num-threads", type=int, default=4, help="torch CPU threads (part of the reproducibility contract).")
    p.add_argument("--save-dir", type=str, default=None,
                   help="Package directory (default: $ARM_MODEL_OUTPUT_DIR/run_<timestamp>).")
    p.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty package directory.")
    p.add_argument("--continue-from", type=str, default=None,
                   help="Existing package to continue (model + replay buffer + curriculum); writes a NEW package.")
    p.add_argument("--drop-buffer", action="store_true",
                   help="With --continue-from: keep the weights only (warm start on a new task), refill the buffer.")
    p.add_argument("--algorithm", type=str, default="SAC", choices=["SAC", "PPO", "A2C"])
    # Task
    p.add_argument("--preset", type=str, default="2dof_human_right")
    p.add_argument("--actuation-mode", type=str, default="muscle", choices=["muscle", "torque"])
    p.add_argument("--waypoints", type=str, default="2",
                   help='2 | 3 | spread | "x,y;x,y;..." | none (single goal from --goal-direction).')
    p.add_argument("--goal-direction", type=str, default="EAST", choices=["EAST", "NORTH", "HEIGHT", "WEST"])
    p.add_argument("--hold-decrement", type=int, default=None, help="Default: evaluation standard (5).")
    p.add_argument("--intermediate-hold-steps", type=int, default=None)
    p.add_argument("--hand-load-range", type=str, default="0,0", help="kg range randomised at reset, e.g. 0,1")
    p.add_argument("--obs-noise", type=str, default=None, help="sigma_angle_rad,sigma_velocity e.g. 0.01,0.05")
    p.add_argument("--sensory-delay", type=int, default=0, help="Observation delay in control steps.")
    p.add_argument("--activation-dynamics", type=str, default=None,
                   help='Muscle excitation->activation time constants "tau_rise,tau_fall" in s, e.g. "0.04,0.06". '
                        "Adds the 4 activations to the observation.")
    p.add_argument("--reward-weight", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a reward weight, e.g. --reward-weight W_ACCEL=0.02 (repeatable).")
    # Curriculum
    p.add_argument("--tolerance-ladder", type=str, default="0.15,0.10,0.07,0.05")
    p.add_argument("--hold-ladder", type=str, default="10,15,25")
    p.add_argument("--eval-freq", type=int, default=20_000)
    p.add_argument("--advance-consecutive", type=int, default=2,
                   help="Successive >=80%% rounds needed before a curriculum rung tightens.")
    p.add_argument("--cooldown-rounds", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=100_000)
    p.add_argument("--no-curriculum", action="store_true")
    # SAC
    p.add_argument("--net-arch", type=str, default="256,256")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--utd", type=float, default=1.0, help="gradient updates per env step; 1/utd must be an integer if < 1")
    p.add_argument("--gamma", type=float, default=None, help="Default: project default 0.99 (matches env.discount).")
    p.add_argument("--buffer-size", type=int, default=None)
    p.add_argument("--learning-starts", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--device", type=str, default="cpu", help="cpu (the GPU measured 3.5-5x slower for this SAC).")
    p.add_argument("--n-envs", type=int, default=1, help="DummyVecEnv workers; diversifies the buffer, no speed-up.")
    # Validators
    p.add_argument("--skip-validators", action="store_true")
    p.add_argument("--fitts-trials", type=int, default=8)
    p.add_argument("--power-trials", type=int, default=16)
    p.add_argument("--validator-seed", type=int, default=42)
    return p.parse_args(argv)


def _parse_floats(s):
    return [float(x) for x in str(s).split(",") if x.strip()]


def main(argv=None) -> int:
    args = parse_args(argv)

    import numpy as np
    import torch
    torch.set_num_threads(max(1, int(args.num_threads)))

    from rl_armMotion.two_d.environments.task_env import ArmTaskEnv
    from rl_armMotion.two_d.evaluation import (
        EVAL_STANDARD, PACKAGE_FILES, evaluate_policy, make_env, resolve_waypoints,
    )
    from rl_armMotion.two_d.training.ppo_trainer_wrapper import RLTrainerWithMetrics
    from rl_armMotion.two_d.training.reproducibility import set_global_seed

    # --- output directory --------------------------------------------------
    if args.save_dir:
        save_dir = Path(args.save_dir).resolve()
    else:
        base = Path(os.environ.get("ARM_MODEL_OUTPUT_DIR", PROJECT_ROOT / "project_assets" / "outputs"))
        save_dir = (base / f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}_seed{args.seed}").resolve()
    existing = [n for n in (PACKAGE_FILES["best"], PACKAGE_FILES["last"], PACKAGE_FILES["final"], "training_log.txt")
                if (save_dir / n).exists()]
    if existing and not args.overwrite:
        print(f"[ABORT] {save_dir} already holds {existing}. Use --overwrite, a new --save-dir, "
              f"or --continue-from to train it further into a new directory.")
        return 2
    if args.continue_from and Path(args.continue_from).resolve() == save_dir:
        print("[ABORT] --continue-from must differ from --save-dir (the source package is never modified).")
        return 2
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "training_log.txt"

    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    set_global_seed(args.seed)
    log(f"Run: {args.algorithm} {args.timesteps:,} steps, seed {args.seed}, threads {torch.get_num_threads()}, "
        f"device {args.device}, preset {args.preset}, actuation {args.actuation_mode}, waypoints {args.waypoints}")
    log(f"Package dir: {save_dir}")

    # --- environment -------------------------------------------------------
    waypoints = resolve_waypoints(None if args.waypoints.lower() == "none" else args.waypoints)
    hand_load = tuple(_parse_floats(args.hand_load_range)) if args.hand_load_range else (0.0, 0.0)
    obs_noise = tuple(_parse_floats(args.obs_noise)) if args.obs_noise else None
    reward_weights = {}
    for item in args.reward_weight:
        key, _, val = item.partition("=")
        reward_weights[key.strip()] = float(val)
    env_kwargs = dict(
        preset=args.preset,
        actuation_mode=args.actuation_mode,
        goal_direction=args.goal_direction,
        terminate_on_success=False,
        hand_load_range=hand_load,
        obs_noise_std=obs_noise,
        sensory_delay_steps=args.sensory_delay,
        activation_time_constants=_parse_floats(args.activation_dynamics) if args.activation_dynamics else None,
        reward_weights=reward_weights or None,
    )
    tolerance_ladder = _parse_floats(args.tolerance_ladder)
    hold_ladder = [int(x) for x in _parse_floats(args.hold_ladder)]
    env = make_env(waypoints=waypoints, tolerance=tolerance_ladder[0], hold_steps=hold_ladder[0], **env_kwargs)
    if args.hold_decrement is not None:
        env.set_hold_decrement(args.hold_decrement)
    if args.intermediate_hold_steps is not None:
        env.set_intermediate_hold_steps(args.intermediate_hold_steps)
    if waypoints:
        for i, wp in enumerate(waypoints):
            f = env.waypoint_feasibility(wp, EVAL_STANDARD.tolerance)
            log(f"  waypoint {i} {wp}: {'feasible' if f['feasible'] else 'INFEASIBLE'} - {f.get('reason')}"
                + (f", joints {np.round(f['joint_angles_deg'], 1).tolist()} deg" if f.get('joint_angles_deg') else ""))
            if not f["feasible"]:
                log("[ABORT] infeasible waypoint; fix the waypoint set before spending compute.")
                return 3
    log(f"Env: obs_dim {env.obs_dim}, action_dim {env.action_dim}, dt {env.dt}, "
        f"train tol {env.goal_tolerance} m, hold {env.hold_steps_required}, decrement {env.hold_decrement}, "
        f"activation dynamics {env.config.activation_time_constants}, reward weights {env.describe()['reward_weights']}")

    # --- hyperparameters ---------------------------------------------------
    net_arch = [int(x) for x in args.net_arch.split(",") if x.strip()]
    hyperparams = {"batch_size": args.batch_size, "policy_kwargs": dict(net_arch=net_arch), "device": args.device}
    if args.gamma is not None:
        hyperparams["gamma"] = args.gamma
    if args.buffer_size is not None:
        hyperparams["buffer_size"] = args.buffer_size
    if args.learning_starts is not None:
        hyperparams["learning_starts"] = args.learning_starts
    if args.learning_rate is not None:
        hyperparams["learning_rate"] = args.learning_rate
    if args.utd != 1.0:
        if args.utd <= 0:
            raise ValueError("--utd must be positive")
        if args.utd < 1.0:
            inv = 1.0 / args.utd
            if abs(inv - round(inv)) > 1e-9:
                raise ValueError(f"--utd {args.utd}: 1/utd must be an integer (got {inv:.3f})")
            hyperparams["train_freq"] = int(round(inv))
            hyperparams["gradient_steps"] = 1
        else:
            if abs(args.utd - round(args.utd)) > 1e-9:
                raise ValueError("--utd > 1 must be an integer")
            hyperparams["train_freq"] = 1
            hyperparams["gradient_steps"] = int(round(args.utd))
    gamma_for_env = hyperparams.get("gamma", 0.99)
    env.discount = float(gamma_for_env)

    trainer = RLTrainerWithMetrics(
        env=env,
        total_timesteps=args.timesteps,
        algorithm=args.algorithm,
        hyperparams=hyperparams,
        run_dir=str(save_dir),
        seed=args.seed,
        n_envs=args.n_envs,
        tolerance_ladder=tolerance_ladder,
        hold_ladder=hold_ladder,
        eval_freq=args.eval_freq,
        checkpoint_every=args.checkpoint_every,
        use_curriculum=not args.no_curriculum,
        continue_from=args.continue_from,
        continue_drop_buffer=args.drop_buffer,
        advance_consecutive=args.advance_consecutive,
        cooldown_rounds=args.cooldown_rounds,
        cli_args=vars(args),
    )
    log(f"Resolved hyperparameters: {json.dumps(trainer.trainer.resolved_hyperparams, default=str)}")

    t0 = time.time()
    result = trainer.train()
    train_secs = time.time() - t0
    log(f"Training complete in {train_secs:.1f}s ({trainer.trainer.model.num_timesteps:,} total steps)")
    log(f"Episodes: {result['final_metrics'].get('episodes', 0):,}; mean reward (100 ep): "
        f"{result['final_metrics'].get('mean_reward', 0):.2f}")
    cur = result["final_metrics"].get("curriculum", {})
    log(f"Curriculum: tolerance {cur.get('current_tolerance')} m, hold {cur.get('current_hold_steps')}, "
        f"best score {cur.get('best_score')} at step {cur.get('best_saved_at')}")

    save_paths = trainer.save_model_and_results(str(save_dir))
    log(f"Saved package: {save_paths}")

    # --- final production evaluation of the best checkpoint -----------------
    best_path = save_dir / PACKAGE_FILES["best"]
    from rl_armMotion.two_d.evaluation import load_package
    if best_path.exists():
        model, eval_env, _ = load_package(save_dir, which="best", terminate_on_success=True)
        res = evaluate_policy(model, eval_env, n_episodes=EVAL_STANDARD.n_episodes * 2, seed=EVAL_STANDARD.seed + 500)
        res.save(save_dir / "final_eval.json")
        log(f"FINAL (best checkpoint, {res.n} episodes): {res.summary_line()}")
        hl = res.human_likeness_summary()
        log(f"Human-likeness: straightness {hl['straightness']:.2f}, symmetry {hl['symmetry']:.2f}, "
            f"submovements {hl['submovements']:.1f}, jerk {hl['dimensionless_jerk']:.0f}")

    if args.skip_validators:
        log("Validators skipped (--skip-validators).")
        log("=" * 60)
        return 0

    # --- validators (single-goal env at the same standard) -----------------
    from rl_armMotion.two_d.validation import FittsLawValidator, PowerLawValidator
    val_env = make_env(waypoints=None, terminate_on_success=True, **{k: v for k, v in env_kwargs.items() if k != "terminate_on_success"})
    model = trainer.trainer.model
    try:
        fl = FittsLawValidator(model=model, env=val_env)
        t0 = time.time()
        fl_result = fl.run(n_trials_per_condition=args.fitts_trials, max_steps_per_trial=200, seed=args.validator_seed)
        log(f"Fitts sweep {time.time() - t0:.1f}s: {fl_result.regression_summary()}")
        fl_result.save_json(str(save_dir / "fitts_law.json"))
        fl_result.plot(save_path=str(save_dir / "fitts_law.png"), show=False,
                       title=f"Fitts' Law - {args.algorithm} ({args.timesteps:,} steps)")
        pl = PowerLawValidator(model=model, env=val_env)
        t0 = time.time()
        pl_result = pl.run(n_trials=args.power_trials, max_steps_per_trial=200, seed=args.validator_seed)
        log(f"Power law {time.time() - t0:.1f}s: {pl_result.regression_summary()}")
        pl_result.save_json(str(save_dir / "power_law.json"))
        pl_result.plot(save_path=str(save_dir / "power_law.png"), show=False,
                       title=f"2/3 Power Law - {args.algorithm} ({args.timesteps:,} steps)")
    except Exception as exc:
        log(f"Validators failed: {exc!r}")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
