"""
Why does a random-waypoint policy fail? Per-waypoint failure analysis.

    python scripts/diagnose_random_reach.py C:/arm-model-runs/stageR3_cont/seed_4
    python scripts/diagnose_random_reach.py <pkg> --episodes 50 --which best

The campaign report gives one number per seed (episode success = all N waypoints
plus the final hold). That number cannot say *why* a seed fails, so this script
rolls out the same production standard and records, for every waypoint of every
episode:

  * whether the hand ever came within the tolerance of it   (reach)
  * whether it was ever "in goal" there, i.e. inside the tolerance AND slow
    enough in both hand and joint speed                     (settle)
  * whether the waypoint was completed                      (advance / hold)
  * how close it got, and how long it spent there

Failures are then classified per episode:

  NEVER_REACHED   the hand never entered the tolerance ball of the waypoint
                  it stalled on  ->  a control/precision problem
  NEVER_SETTLED   it entered the ball but never slowly enough to count
                  ->  a speed / damping problem (overshoot, orbiting)
  HOLD_BROKE      it settled but the hold counter never reached the required
                  count  ->  a stability problem
  TIMEOUT_EARLY   it was still working on an earlier waypoint when the clock
                  ran out  ->  too slow overall

and summarised by where in the workspace the stalling waypoint was, so a
systematic blind spot (near the reach limit, high shoulder torque, low points)
shows up as a region rather than as noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rl_armMotion.two_d.evaluation import EVAL_STANDARD, load_package  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("package", help="run directory written by train_fischer_session.py")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--which", default="best", help="best | last | final | step_XXXXXX")
    p.add_argument("--seed", type=int, default=EVAL_STANDARD.seed + 900)
    p.add_argument("--json", default=None, help="also write the per-waypoint rows here")
    return p.parse_args()


def rollout(model, env, ep_seed):
    """One production episode; returns per-waypoint records and the outcome."""
    obs, info = env.reset(seed=ep_seed)
    base = getattr(env, "unwrapped", env)
    n = int(base.num_waypoints)
    wps = [np.asarray(w, dtype=float) for w in base.waypoints]
    rec = [dict(index=k, position=wps[k].tolist(), reached=False, settled=False, completed=False,
                min_distance=float("inf"), steps=0, max_hold=0.0) for k in range(n)]
    tol = float(base.goal_tolerance)
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        k = int(info["current_waypoint_index"])
        r = rec[k]
        r["steps"] += 1
        r["min_distance"] = min(r["min_distance"], float(info["goal_distance"]))
        r["reached"] = r["reached"] or float(info["goal_distance"]) <= tol
        r["settled"] = r["settled"] or bool(info["in_goal_region"])
        r["max_hold"] = max(r["max_hold"], float(info["hold_progress"]))
        if info.get("waypoint_advanced"):
            # the index has already moved on, so the completed waypoint is the previous one
            rec[max(0, k - 1)]["completed"] = True
        if info["success"]:
            rec[n - 1]["completed"] = True
        if terminated or truncated:
            break
    success = bool(info["success"])
    stalled = n - 1 if success else int(info["current_waypoint_index"])
    if success:
        mode = "SUCCESS"
    elif steps >= int(base.max_episode_steps) and stalled < n - 1:
        mode = "TIMEOUT_EARLY"
    elif not rec[stalled]["reached"]:
        mode = "NEVER_REACHED"
    elif not rec[stalled]["settled"]:
        mode = "NEVER_SETTLED"
    else:
        mode = "HOLD_BROKE"
    return dict(seed=ep_seed, success=success, mode=mode, stalled_at=stalled, steps=steps,
                waypoints=rec, tolerance=tol)


MIN_REGION_N = 10       # below this, a per-region success rate is not worth printing as a number


def wilson(successes, n, z=1.96):
    """95% Wilson score interval - honest about small samples, unlike k/n."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def region(shoulder, p):
    """Coarse workspace bucket of a target: radial band and elevation sector."""
    d = np.asarray(p, dtype=float) - np.asarray(shoulder, dtype=float)
    r = float(np.linalg.norm(d))
    ang = float(np.degrees(np.arctan2(d[1], d[0])))
    band = "near" if r < 0.35 else ("mid" if r < 0.50 else "far")
    sector = "low" if ang < -20 else ("level" if ang < 20 else "high")
    return f"{band}/{sector}", r, ang


def main():
    args = parse_args()
    pkg = Path(args.package)
    model, env, manifest = load_package(pkg, which=args.which, terminate_on_success=True)
    base = getattr(env, "unwrapped", env)
    base.reset(seed=args.seed)   # sample a task so num_waypoints is the episode value
    shoulder = np.asarray(base.shoulder_base_position, dtype=float)
    print(f"package {pkg}  ({args.which}, best saved at {manifest.get('best_saved_at')})")
    print(f"standard: tolerance {base.goal_tolerance:.3f} m, hold {base.hold_steps_required} steps, "
          f"hand speed < {base.ee_speed_tolerance:.2f} m/s, {base.max_episode_steps} steps max, "
          f"{base.num_waypoints} random waypoints/episode\n")

    eps = [rollout(model, env, args.seed + i) for i in range(args.episodes)]

    n_wp = int(base.num_waypoints)
    modes = Counter(e["mode"] for e in eps)
    n = len(eps)
    succ = sum(e["success"] for e in eps)
    print(f"episode success  {succ}/{n} = {succ / n:.0%}")
    print("failure modes   ", ", ".join(f"{k} {v}" for k, v in modes.most_common()))

    # Per-waypoint funnel: of the waypoints the policy actually worked on,
    # how many were reached / settled / completed.
    print("\nper-waypoint funnel (counting only the episodes that got that far)")
    print(f"{'waypoint':>9} {'attempted':>10} {'reached':>8} {'settled':>8} {'completed':>10} "
          f"{'median min-dist':>16} {'median steps':>13}")
    for k in range(n_wp):
        rows = [e["waypoints"][k] for e in eps if e["waypoints"][k]["steps"] > 0]
        if not rows:
            print(f"{k + 1:>9} {0:>10}")
            continue
        md = float(np.median([r["min_distance"] for r in rows]))
        ms = float(np.median([r['steps'] for r in rows]))
        print(f"{k + 1:>9} {len(rows):>10} {sum(r['reached'] for r in rows):>8} "
              f"{sum(r['settled'] for r in rows):>8} {sum(r['completed'] for r in rows):>10} "
              f"{md:>16.3f} {ms:>13.0f}")

    # Where the stalling waypoints are. Spreading N episodes over ~8 regions
    # leaves a handful each, and a rate over a handful is noise: reading
    # "near/level 50%" off two episodes once sent a whole day's work chasing a
    # weakness that a 60-episode test showed did not exist. So every rate now
    # carries a 95% Wilson interval, and regions below MIN_REGION_N are pooled
    # rather than printed as a number worth acting on.
    per_region = defaultdict(lambda: [0, 0])
    for e in eps:
        wp = e["waypoints"][e["stalled_at"]]["position"]
        key, _, _ = region(shoulder, wp)
        per_region[key][0] += 1
        per_region[key][1] += int(e["success"])
    print("\nstalling waypoint by workspace region (radius from the shoulder / elevation)")
    print(f"{'region':>12} {'episodes':>9} {'success':>8} {'95% interval':>16}")
    thin = [0, 0]
    for key in sorted(per_region):
        tot, ok = per_region[key]
        if tot < MIN_REGION_N:
            thin[0] += tot
            thin[1] += ok
            continue
        lo, hi = wilson(ok, tot)
        print(f"{key:>12} {tot:>9} {ok / tot:>8.0%} {f'{lo:.0%}-{hi:.0%}':>16}")
    if thin[0]:
        lo, hi = wilson(thin[1], thin[0])
        print(f"{'(thin)':>12} {thin[0]:>9} {thin[1] / thin[0]:>8.0%} {f'{lo:.0%}-{hi:.0%}':>16}"
              f"   pooled: regions with fewer than {MIN_REGION_N} episodes")
    print("     -- to compare two regions, sample each one deliberately (>= 50 episodes);\n"
          "        do not read a conclusion off a bucket of this table.")

    # Distance of every failed waypoint from the tolerance ball.
    misses = [w["min_distance"] for e in eps if not e["success"]
              for w in [e["waypoints"][e["stalled_at"]]]]
    if misses:
        q = np.percentile(misses, [50, 75, 90])
        print(f"\nfailed waypoints: closest approach median {q[0]:.3f} m, p75 {q[1]:.3f} m, p90 {q[2]:.3f} m "
              f"(tolerance {base.goal_tolerance:.3f} m)")
        near = sum(1 for m in misses if m <= base.goal_tolerance * 1.5)
        print(f"{near}/{len(misses)} failures came within 1.5x the tolerance -> precision/settling, not reaching")

    done = [e["steps"] for e in eps if e["success"]]
    if done:
        print(f"\nsuccessful episodes: median {np.median(done):.0f} steps = {np.median(done) * base.dt:.1f} s "
              f"of the {base.max_episode_steps * base.dt:.0f} s budget")

    if args.json:
        Path(args.json).write_text(json.dumps(eps, indent=2, default=float), encoding="utf-8")
        print(f"\nper-episode rows -> {args.json}")


if __name__ == "__main__":
    main()
