"""
Evaluate a saved model package at the evaluation standard.

    python scripts/evaluate_trained_model.py <package_dir> [--which best|last|final] [--episodes 20]
                                             [--n-waypoints 2|3] [--trajectories] [--out eval.json]

Rebuilds the training env from the package's run_manifest.json (so the model
is measured in the env it was trained in), runs N seeded episodes with
randomised start poses and prints success rate, waypoints reached, hold
progress, closest final distance, steps to success and the human-likeness
metrics (path straightness, velocity-profile symmetry, submovements, jerk).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rl_armMotion.two_d.evaluation import EVAL_STANDARD, evaluate_policy, load_package, resolve_waypoints  # noqa: E402
from rl_armMotion.two_d.kinematics_metrics import HUMAN_REFERENCE  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a model package")
    p.add_argument("package")
    p.add_argument("--which", default="best")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=EVAL_STANDARD.seed)
    p.add_argument("--n-waypoints", type=int, default=None)
    p.add_argument("--waypoints", default=None)
    p.add_argument("--tolerance", type=float, default=None)
    p.add_argument("--trajectories", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    overrides = {"terminate_on_success": True}
    if args.waypoints:
        overrides["waypoints"] = resolve_waypoints(args.waypoints)
    elif args.n_waypoints:
        overrides["waypoints"] = resolve_waypoints(args.n_waypoints)
    if args.tolerance is not None:
        overrides["goal_tolerance"] = args.tolerance
    model, env, manifest = load_package(args.package, which=args.which, **overrides)
    print(f"package: {args.package}  ({args.which}, seed {manifest.get('seed')}, "
          f"{manifest.get('final_timesteps', manifest.get('total_timesteps'))} steps)")
    print(f"env: {env.preset_name}, {env.actuation_mode}, {env.num_waypoints} waypoints, "
          f"tolerance {env.goal_tolerance} m, hold {env.hold_steps_required} steps, decrement {env.hold_decrement}")

    res = evaluate_policy(model, env, n_episodes=args.episodes, seed=args.seed, log_trajectories=args.trajectories)
    s = res.summary()
    print(f"\n{res.summary_line()}")
    print(f"return {s['mean_return']:.1f} +/- {s['sd_return']:.1f}; waypoints {s['mean_waypoints_reached']:.2f} "
          f"+/- {s['sd_waypoints_reached']:.2f}")
    hl = s["human_likeness"]
    print("human-likeness (reference ranges in brackets):")
    for k in ("straightness", "symmetry", "submovements", "dimensionless_jerk"):
        lo, hi = HUMAN_REFERENCE[k]
        print(f"  {k:20s} {hl[k]:8.3f}   [{lo}, {hi}]")
    print("\nper-episode:")
    for e in res.episodes:
        print(f"  seed {e.seed}: success={e.success} step={e.success_step} reached={e.waypoints_reached}/{e.num_waypoints} "
              f"hold={e.max_hold_progress:.2f} min_final_dist={e.min_final_distance:.3f} return={e.episode_return:.1f}")
    out = Path(args.out) if args.out else Path(args.package) / f"eval_{args.which}.json"
    res.save(out, trajectories=args.trajectories)
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
