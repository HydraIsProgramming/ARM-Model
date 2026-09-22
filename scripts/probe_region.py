"""
Is the policy actually worse in some part of the workspace?

    python scripts/probe_region.py project_assets/outputs/CHAMPION_HUMAN_V1_R3 --episodes 60

`diagnose_random_reach.py` bins whatever waypoints the random sampler happened to
draw, so each region gets a handful of episodes and the rates are mostly noise.
This script does the opposite: it *chooses* the targets. Every episode is the same
shape - one easy mid-range waypoint, then a test point drawn from the band under
test - so the only thing that differs between bands is the test point, and each
band gets its own full sample.

Bands are radial distance from the shoulder; `--split elbow` instead compares
targets whose inverse-kinematics pose sits inside the elbow's passive end-range
spring zone against targets that clear it (the environment penalises end-range
postures through W_LIMIT, so it is fair to ask whether it can still hold there).

Every rate is printed with a 95% Wilson interval. Two bands differ only if their
intervals separate; with 60 episodes a rate is worth about +/- 10 points, so
smaller gaps than that are not findings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rl_armMotion.two_d.evaluation import detect_actuation_mode, load_package  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("package")
    p.add_argument("--episodes", type=int, default=60, help="episodes PER band (default 60)")
    p.add_argument("--which", default="best")
    p.add_argument("--split", default="radius", choices=["radius", "elbow"])
    p.add_argument("--seed", type=int, default=9000)
    return p.parse_args()


def wilson(successes, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def grid_with_margins(env):
    """Every feasible target, with its radius and its per-joint ROM margin."""
    pts = env.feasible_points()
    qmin = np.degrees(np.asarray(env.config.joint_limits_min, dtype=float))
    qmax = np.degrees(np.asarray(env.config.joint_limits_max, dtype=float))
    shoulder = np.asarray(env.shoulder_base_position, dtype=float)
    rows = []
    for p in pts:
        f = env.waypoint_feasibility(p, env.goal_tolerance)
        q = np.asarray(f["joint_angles_deg"], dtype=float)
        rows.append((float(np.hypot(*(p - shoulder))),
                     float(min(q[0] - qmin[0], qmax[0] - q[0])),
                     float(min(q[1] - qmin[1], qmax[1] - q[1]))))
    r, sh, el = np.asarray(rows).T
    return pts, r, sh, el


def main():
    args = parse_args()
    model, env, _ = load_package(args.package, which=args.which, terminate_on_success=True)
    base = getattr(env, "unwrapped", env)
    pts, radius, _sh_margin, el_margin = grid_with_margins(base)
    # The package was trained on the random-waypoint task, so the env it rebuilds
    # re-samples targets on every reset() and would silently discard the test
    # point this script chose - every band would then measure the same thing.
    base.random_waypoints = None

    _, info0 = base.reset(seed=1)
    start = np.asarray(info0["end_effector_position"], dtype=float)
    min_spacing = 2 * base.max_curriculum_tolerance
    min_start = base.max_curriculum_tolerance + 0.17
    mid_pool = pts[(radius >= 0.35) & (radius < 0.50)]

    def trial(test_pt, ep_seed):
        """An easy mid-range waypoint, then the test point; both spacing rules hold."""
        d_test = np.linalg.norm(mid_pool - test_pt, axis=1)
        d_start = np.linalg.norm(mid_pool - start, axis=1)
        ok = (d_test > min_spacing) & (d_start > min_start)
        if not ok.any():
            return None
        base.set_waypoints([mid_pool[ok][np.argmax(d_test[ok])].tolist(), test_pt.tolist()])
        obs, info = base.reset(seed=ep_seed)
        closest = np.inf
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = base.step(action)
            if int(info["current_waypoint_index"]) == base.num_waypoints - 1:
                closest = min(closest, float(info["goal_distance"]))
            if terminated or truncated:
                return bool(info["success"]), closest, int(info["step"])

    if args.split == "radius":
        bands = [("near  (r < 0.35 m)", radius < 0.35),
                 ("mid   (0.35-0.50 m)", (radius >= 0.35) & (radius < 0.50)),
                 ("far   (r >= 0.50 m)", radius >= 0.50)]
    else:
        margin = float(np.degrees(np.max(env.config.passive_limit_margins_max)))
        bands = [(f"elbow inside the spring zone (< {margin:.0f} deg)", el_margin < margin),
                 (f"elbow clear (>= {margin + 5:.0f} deg)", el_margin >= margin + 5)]

    print(f"package {args.package} ({args.which}), actuation {detect_actuation_mode(model)}")
    print(f"standard: {base.goal_tolerance:.2f} m tolerance, {base.hold_steps_required}-step hold, "
          f"{args.episodes} episodes per band\n")
    print(f"{'band':>38} {'n':>4} {'success':>8} {'95% interval':>15} {'closest':>9} {'time':>6}")
    rng = np.random.default_rng(args.seed)
    for label, mask in bands:
        group = pts[mask]
        if len(group) == 0:
            print(f"{label:>38}    -   (no feasible targets in this band)")
            continue
        idx = rng.choice(len(group), size=min(args.episodes, len(group)), replace=False)
        ok, n, dists, steps = 0, 0, [], []
        for i, j in enumerate(idx):
            res = trial(group[j], args.seed + i)
            if res is None:
                continue
            success, closest, step = res
            n += 1
            ok += success
            if np.isfinite(closest):
                dists.append(closest)
            if success:
                steps.append(step)
        lo, hi = wilson(ok, n)
        print(f"{label:>38} {n:>4} {ok / n:>8.0%} {f'{lo:.0%}-{hi:.0%}':>15} "
              f"{np.median(dists):>9.3f} {np.median(steps) * base.dt:>5.1f}s")
    print("\nBands differ only where the intervals do not overlap.")


if __name__ == "__main__":
    main()
