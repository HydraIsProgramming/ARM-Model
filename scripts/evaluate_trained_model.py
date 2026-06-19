"""
Comprehensive evaluation of a trained Fischer model.

Measures three things the casual reader needs to see to believe the policy
is "working":

  1. Goal-reaching success rate: does the policy actually complete the task?
     Reported as fraction of episodes that reach the goal tolerance, plus
     mean final distance and mean episode length.

  2. Motion smoothness: quantitative jerk metrics on the end-effector
     trajectory. Lower = smoother. Compared between the trained policy and
     a uniformly-random baseline.

  3. Sample trajectories: plotted in workspace coordinates plus per-joint
     velocity-and-action time series so a human can visually confirm that
     the motion is purposeful and not jerky.

Run from the project root with the venv activated:

    python scripts/evaluate_trained_model.py \\
        --model-zip ./project_assets/outputs/fischer_muscle_200k/sac_model.zip \\
        --actuation-mode muscle \\
        --goal-direction EAST \\
        --n-episodes 30 \\
        --output-dir ./project_assets/outputs/fischer_muscle_200k/evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure src/ is importable
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Fischer model")
    parser.add_argument("--model-zip", type=str, required=True)
    parser.add_argument(
        "--actuation-mode", type=str, default="velocity",
        choices=["velocity", "muscle"],
    )
    parser.add_argument(
        "--goal-direction", type=str, default="EAST",
        choices=["EAST", "WEST", "NORTH", "HEIGHT"],
    )
    parser.add_argument("--algorithm", type=str, default="SAC",
                        choices=["SAC", "PPO", "A2C"])
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def jerk_metrics(positions: np.ndarray, dt: float) -> dict:
    """Compute jerk metrics from a position trajectory.

    Jerk is the third time derivative of position (the rate of change of
    acceleration). For biological reaching motions, jerk should be small
    in magnitude and continuous in time — humans do not flick their hands
    suddenly. We report three metrics:

      rms_jerk
          Root-mean-square jerk magnitude over the trajectory, in m/s^3.
          Intuitive units; lower means smoother.

      peak_jerk
          Maximum instantaneous jerk magnitude in the trajectory.
          Helpful for spotting transient discontinuities.

      log_dimensionless_jerk
          The unit-free smoothness measure of Hogan and Sternad (2009),
          widely used in biomechanics. Higher (less negative) values mean
          smoother motion. Formula (our sign convention):
              LDJ = -log( integral(|jerk|^2 dt) * T^5 / vpeak^2 )
          where T is the trajectory duration and vpeak is the peak
          tangential speed. The normalisation by T^5 / vpeak^2 makes the
          metric invariant to the trajectory's duration and amplitude,
          so two reaches of different distances are still comparable.

    Derivatives are computed by np.gradient (centred differences in the
    interior, forward / backward at the boundaries). For very short
    trajectories (< 5 samples) we return NaNs rather than risk numerical
    nonsense.
    """
    if positions.shape[0] < 5:
        return {"rms_jerk": float("nan"), "peak_jerk": float("nan"),
                "log_dimensionless_jerk": float("nan")}
    x = positions[:, 0]
    y = positions[:, 1]
    # Three successive numerical derivatives: position -> velocity ->
    # acceleration -> jerk. np.gradient handles the boundaries cleanly.
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    jx = np.gradient(ax, dt)
    jy = np.gradient(ay, dt)
    jerk_mag = np.sqrt(jx ** 2 + jy ** 2)
    v_mag = np.sqrt(vx ** 2 + vy ** 2)
    T = positions.shape[0] * dt
    vpeak = float(np.max(v_mag))
    int_jerk_sq = float(np.sum(jerk_mag ** 2) * dt)
    if vpeak > 1e-9 and int_jerk_sq > 1e-12:
        ldj = -np.log((int_jerk_sq * T ** 5) / (vpeak ** 2))
    else:
        ldj = float("nan")
    return {
        "rms_jerk": float(np.sqrt(np.mean(jerk_mag ** 2))),
        "peak_jerk": float(np.max(jerk_mag)),
        "log_dimensionless_jerk": float(ldj),
    }


def run_one_episode(env, model, max_steps: int, deterministic: bool = True):
    """Run one episode, returning a trial summary dict."""
    obs, info = env.reset()
    positions = [info["end_effector_position"].copy()]
    velocities = [info["joint_velocities"].copy()]
    actions_log = []
    final_distance = float("inf")
    goal_reached = False
    for step in range(max_steps):
        if model is None:
            action = env.action_space.sample()  # random baseline
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        actions_log.append(np.asarray(action, dtype=float))
        positions.append(info["end_effector_position"].copy())
        velocities.append(info["joint_velocities"].copy())
        final_distance = float(info["goal_distance"])
        if terminated:
            goal_reached = True
            break
        if truncated:
            break
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    actions_arr = np.asarray(actions_log, dtype=float) if actions_log else np.zeros((0,))
    return {
        "positions": positions,
        "velocities": velocities,
        "actions": actions_arr,
        "n_steps": int(positions.shape[0] - 1),
        "final_distance": float(final_distance),
        "goal_reached": bool(goal_reached),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from rl_armMotion.two_d.environments.task_env import ArmTaskEnv
    from stable_baselines3 import SAC, PPO, A2C
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(args.seed)

    # --- Load model -------------------------------------------------
    AlgClass = {"SAC": SAC, "PPO": PPO, "A2C": A2C}[args.algorithm]
    model = AlgClass.load(args.model_zip)
    env = ArmTaskEnv(
        goal_direction=args.goal_direction,
        actuation_mode=args.actuation_mode,
    )
    dt = float(env.dt)

    # --- Trained policy: run N episodes ----------------------------
    trained_results = []
    for ep in range(args.n_episodes):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ep_data = run_one_episode(env, model, args.max_steps)
        trained_results.append(ep_data)

    # --- Random baseline: same N episodes --------------------------
    rand_env = ArmTaskEnv(
        goal_direction=args.goal_direction,
        actuation_mode=args.actuation_mode,
    )
    random_results = []
    for ep in range(args.n_episodes):
        rand_env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ep_data = run_one_episode(rand_env, None, args.max_steps)
        random_results.append(ep_data)

    # --- Aggregate task metrics ------------------------------------
    def summarise(results):
        n_reached = sum(1 for r in results if r["goal_reached"])
        final_distances = [r["final_distance"] for r in results]
        n_steps = [r["n_steps"] for r in results]
        return {
            "n_episodes": len(results),
            "success_rate": n_reached / len(results),
            "n_reached": int(n_reached),
            "mean_final_distance": float(np.mean(final_distances)),
            "median_final_distance": float(np.median(final_distances)),
            "min_final_distance": float(np.min(final_distances)),
            "mean_episode_length": float(np.mean(n_steps)),
        }

    trained_summary = summarise(trained_results)
    random_summary = summarise(random_results)

    # --- Aggregate jerk metrics ------------------------------------
    def jerk_summary(results):
        rmsj = []
        ldj = []
        for r in results:
            jm = jerk_metrics(r["positions"], dt)
            if not np.isnan(jm["rms_jerk"]):
                rmsj.append(jm["rms_jerk"])
            if not np.isnan(jm["log_dimensionless_jerk"]):
                ldj.append(jm["log_dimensionless_jerk"])
        return {
            "mean_rms_jerk": float(np.mean(rmsj)) if rmsj else float("nan"),
            "median_rms_jerk": float(np.median(rmsj)) if rmsj else float("nan"),
            "mean_log_dimensionless_jerk": float(np.mean(ldj)) if ldj else float("nan"),
            "n_valid": len(rmsj),
        }

    trained_jerk = jerk_summary(trained_results)
    random_jerk = jerk_summary(random_results)

    # --- Report to stdout ------------------------------------------
    def report(label, summary, jerk):
        print(f"\n{label}")
        print("-" * len(label))
        print(f"  Episodes:                 {summary['n_episodes']}")
        print(f"  Goal reached (success):   {summary['n_reached']}/{summary['n_episodes']}"
              f" = {summary['success_rate']*100:.1f}%")
        print(f"  Mean final distance:      {summary['mean_final_distance']:.4f} m")
        print(f"  Median final distance:    {summary['median_final_distance']:.4f} m")
        print(f"  Closest approach:         {summary['min_final_distance']:.4f} m")
        print(f"  Mean episode length:      {summary['mean_episode_length']:.1f} steps"
              f" ({summary['mean_episode_length']*dt:.2f} s)")
        print(f"  Mean RMS jerk:            {jerk['mean_rms_jerk']:.3f} m/s^3")
        print(f"  Mean log dimensionless jerk: {jerk['mean_log_dimensionless_jerk']:.2f}"
              f"  (higher / less negative = smoother)")

    print(f"Evaluation run: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Model:           {args.model_zip}")
    print(f"Actuation mode:  {args.actuation_mode}")
    print(f"Goal direction:  {args.goal_direction}")
    print(f"Episodes:        {args.n_episodes}")
    print(f"Max steps:       {args.max_steps}")
    print(f"dt:              {dt} s")
    report("TRAINED POLICY", trained_summary, trained_jerk)
    report("RANDOM BASELINE", random_summary, random_jerk)

    # --- Save aggregate JSON ---------------------------------------
    summary_path = output_dir / "evaluation_summary.json"
    with summary_path.open("w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": str(args.model_zip),
            "actuation_mode": args.actuation_mode,
            "goal_direction": args.goal_direction,
            "n_episodes": args.n_episodes,
            "max_steps": args.max_steps,
            "dt": dt,
            "trained_policy": {
                "task_metrics": trained_summary,
                "smoothness_metrics": trained_jerk,
            },
            "random_baseline": {
                "task_metrics": random_summary,
                "smoothness_metrics": random_jerk,
            },
        }, f, indent=2)
    print(f"\nSaved: {summary_path}")

    # --- Plot 1: trajectories in workspace -------------------------
    shoulder = env.shoulder_base_position.copy()
    goal_pos = env.goal_position.copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, (results, label) in enumerate([
        (trained_results, "Trained policy"),
        (random_results, "Random baseline"),
    ]):
        ax = axes[ax_idx]
        for r in results[:10]:
            traj = r["positions"]
            color = "tab:green" if r["goal_reached"] else "tab:red"
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.5, linewidth=1)
        ax.scatter(shoulder[0], shoulder[1], c="black", s=80, marker="s",
                   label=f"Shoulder ({shoulder[0]:.1f}, {shoulder[1]:.1f})", zorder=10)
        ax.scatter(goal_pos[0], goal_pos[1], c="tab:orange", s=160, marker="X",
                   edgecolors="black", linewidths=1.2, label=f"Goal ({goal_pos[0]:.1f}, {goal_pos[1]:.1f})",
                   zorder=10)
        reach = float(np.sum(env.config.link_lengths))
        circle = plt.Circle((shoulder[0], shoulder[1]), reach, fill=False,
                            color="gray", linestyle="--", linewidth=0.8)
        ax.add_patch(circle)
        ax.set_xlim(shoulder[0] - reach - 0.3, shoulder[0] + reach + 0.3)
        ax.set_ylim(shoulder[1] - reach - 0.3, shoulder[1] + reach + 0.3)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"{label}: end-effector trajectories\n"
                     f"(green = reached goal, red = did not)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    traj_path = output_dir / "trajectories.png"
    fig.savefig(traj_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {traj_path}")

    # --- Plot 2: per-trial smoothness comparison -------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    trained_ldjs = [jerk_metrics(r["positions"], dt)["log_dimensionless_jerk"]
                    for r in trained_results]
    random_ldjs = [jerk_metrics(r["positions"], dt)["log_dimensionless_jerk"]
                   for r in random_results]
    trained_ldjs = [v for v in trained_ldjs if not np.isnan(v)]
    random_ldjs = [v for v in random_ldjs if not np.isnan(v)]
    bp = ax.boxplot([trained_ldjs, random_ldjs],
                    labels=["Trained policy", "Random baseline"],
                    showmeans=True, meanline=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], ["tab:blue", "tab:red"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.set_ylabel("Log dimensionless jerk\n(higher / less negative = smoother)")
    ax.set_title("Motion smoothness — trained policy vs random baseline\n"
                 "(Hogan & Sternad 2009 metric)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    smooth_path = output_dir / "smoothness_comparison.png"
    fig.savefig(smooth_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {smooth_path}")

    # --- Plot 3: example velocity / action time series -------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    example = trained_results[0]
    t = np.arange(example["velocities"].shape[0]) * dt
    for j in range(example["velocities"].shape[1]):
        axes[0].plot(t, example["velocities"][:, j], label=f"joint {j}")
    axes[0].set_ylabel("Joint angular velocity (rad/s)")
    axes[0].set_title("Sample trajectory — trained policy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)
    if example["actions"].size > 0:
        t_act = np.arange(example["actions"].shape[0]) * dt
        for c in range(example["actions"].shape[1]):
            axes[1].plot(t_act, example["actions"][:, c], label=f"a[{c}]",
                         alpha=0.7, linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Action component")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    series_path = output_dir / "sample_trajectory_series.png"
    fig.savefig(series_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {series_path}")

    print("\nEvaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
