"""
End-to-end test of the thing a user actually does: click points, watch the arm go.

    python scripts/test_click_points.py project_assets/outputs/CHAMPION_HUMAN_V1_R3
    python scripts/test_click_points.py <pkg> --trials 40 --points 3

The campaign evaluates on waypoints drawn by the sampler. A person clicking on
the panel does something different: they pick points anywhere they like, in an
order that suits them, and they are not drawing from the feasible grid. This
drives the SAME code path the GUI uses - the click handler's feasibility and
spacing checks, then set_waypoints, then the policy - on randomly clicked
points, and reports how often the arm actually gets to all of them.

It also reports what the click handler rejects, because a point the GUI refuses
is a different user experience from a point the arm fails to reach, and the two
should not be confused when judging whether the model "works".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rl_armMotion.two_d.evaluation import load_package  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("package")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--points", type=int, default=3)
    p.add_argument("--which", default="best")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--anywhere", action="store_true",
                   help="click uniformly in the reach disc instead of on feasible points "
                        "(measures what the GUI rejects, as a real user would experience)")
    return p.parse_args()


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def click_points(env, rng, n, anywhere):
    """Reproduce the GUI click handler: propose a point, keep it only if the
    feasibility and spacing checks that the handler runs both pass."""
    shoulder = np.asarray(env.shoulder_base_position, dtype=float)
    chosen, rejected = [], 0
    pool = None if anywhere else env.feasible_points()
    for _ in range(4000):
        if len(chosen) == n:
            break
        if anywhere:
            r, th = env.reach * np.sqrt(rng.uniform()), rng.uniform(-np.pi, np.pi)
            wp = [float(shoulder[0] + r * np.cos(th)), float(shoulder[1] + r * np.sin(th))]
        else:
            wp = [float(v) for v in pool[rng.integers(len(pool))]]
        if not env.waypoint_feasibility(wp, env.max_curriculum_tolerance).get("feasible", False):
            rejected += 1
            continue
        trial = chosen + [wp]
        try:
            env._validate_waypoint_spacing([np.asarray(w, dtype=float) for w in trial])
        except ValueError:
            rejected += 1
            continue
        chosen = trial
    return (chosen if len(chosen) == n else None), rejected


def main():
    args = parse_args()
    model, env, _ = load_package(args.package, which=args.which, terminate_on_success=True)
    base = getattr(env, "unwrapped", env)
    # A clicked set is a fixed set: turn off the sampler that would otherwise
    # replace it on reset (the bug that made an earlier probe measure nothing).
    was_random = base.random_waypoints
    base.random_waypoints = None
    rng = np.random.default_rng(args.seed)

    print(f"package {args.package} ({args.which})")
    print(f"{args.trials} trials of {args.points} clicked points, "
          f"{'anywhere in reach' if args.anywhere else 'on feasible points'}; "
          f"tolerance {base.goal_tolerance:.2f} m, hold {base.hold_steps_required} steps\n")

    ok = 0
    n_done = 0
    rejects = 0
    reached_frac, times, misses = [], [], []
    for i in range(args.trials):
        pts, rej = click_points(base, rng, args.points, args.anywhere)
        rejects += rej
        if pts is None:
            continue
        base.set_waypoints(pts)
        obs, info = base.reset(seed=args.seed + i)
        closest = np.full(len(pts), np.inf)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = base.step(action)
            ee = np.asarray(info["end_effector_position"], float)
            for j, wp in enumerate(pts):
                closest[j] = min(closest[j], float(np.linalg.norm(np.asarray(wp) - ee)))
            if terminated or truncated:
                break
        n_done += 1
        ok += bool(info["success"])
        reached_frac.append(info["waypoints_reached"] / len(pts))
        if info["success"]:
            times.append(info["step"] * base.dt)
        else:
            misses.append(float(np.max(closest)))
    base.random_waypoints = was_random

    lo, hi = wilson(ok, max(n_done, 1))
    print(f"reached every clicked point : {ok}/{n_done} = {ok / max(n_done,1):.0%}  (95% {lo:.0%}-{hi:.0%})")
    print(f"points reached per trial    : {np.mean(reached_frac):.0%}")
    if times:
        print(f"time when it succeeds       : median {np.median(times):.1f} s of "
              f"{base.max_episode_steps * base.dt:.0f} s")
    if misses:
        print(f"worst point when it fails   : median {np.median(misses):.3f} m from the target "
              f"(tolerance {base.goal_tolerance:.2f})")
    if args.anywhere:
        print(f"\nclicks the GUI refused      : {rejects} proposals rejected for "
              f"{args.trials * args.points} accepted points")
        print("  (a refused click is the GUI saying 'the arm cannot hold there', not a failure to reach)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
