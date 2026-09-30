"""
How well does a model reach targets in each part of the human workspace?

    python scripts/eval_workspace_regions.py project_assets/outputs/CHAMPION_HUMAN_V1_R3
    python scripts/eval_workspace_regions.py <pkgA> <pkgB> --episodes 200 --json out.json

Every model is run on the SAME episodes: three random targets per episode drawn
from the full human workspace (every joint >= 5 deg inside its hard stop,
holdable at <= 80% torque - the region Stage R7 trains on), at the production
standard (5 cm, 0.5 s hold at the last target). Each target is then scored by
where it lies:

  old area      inside the region every model up to R6 was trained on
  outer ring    more than 0.47 m from the shoulder (the legacy 18 cm ring)
  overhead      more than 0.45 m above the shoulder
  behind body   more than 0.15 m behind the shoulder
  hanging low   more than 0.45 m below the shoulder
  close in      less than 0.30 m from the shoulder

(a target can be in several). A target counts as "attempted" once the arm has
finished the one before it, and "completed" when it is finished itself (the last
one needs the full hold). Completion given attempted is reported per region with
a 95% Wilson interval, so a failure early in an episode is not charged to the
targets it never got to.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rl_armMotion.two_d.environments.task_env import ArmTaskEnv  # noqa: E402
from rl_armMotion.two_d.evaluation import EVAL_STANDARD, load_package  # noqa: E402

REGIONS = ["old area", "outer ring", "overhead", "behind body", "hanging low", "close in"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("packages", nargs="+")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--which", default="best")
    p.add_argument("--json", default=None)
    return p.parse_args()


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def episodes(n, seed):
    """Target sets from the full human workspace, identical for every model."""
    human = ArmTaskEnv(random_waypoints=3, joint_margin_deg=5.0, reach_margin=0.0)
    legacy = ArmTaskEnv(random_waypoints=3)
    rng = np.random.default_rng(seed)
    sets = [human.sample_waypoints(rng, 3) for _ in range(n)]
    return sets, np.asarray(human.shoulder_base_position, dtype=float), legacy


def regions_of(p, shoulder, old):
    x, y = np.asarray(p, dtype=float) - shoulder
    r = float(np.hypot(x, y))
    out = []
    if old.waypoint_feasibility(list(p), old.max_curriculum_tolerance).get("feasible", False):
        out.append("old area")
    if r > 0.47:
        out.append("outer ring")
    if y > 0.45:
        out.append("overhead")
    if x < -0.15:
        out.append("behind body")
    if y < -0.45:
        out.append("hanging low")
    if r < 0.30:
        out.append("close in")
    return out


def run(package, sets, which):
    # Explicit waypoints switch the model's own random sampling off, so every
    # episode uses the target set given here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, env, _ = load_package(package, which=which, terminate_on_success=True, waypoints=sets[0])
    assert env.random_waypoints is None
    reached = []
    for i, wps in enumerate(sets):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        # legacy models warn about targets outside their region
            env.set_waypoints(wps)
        obs, info = env.reset(seed=EVAL_STANDARD.seed + i)
        assert np.allclose(np.asarray(info["waypoints"], dtype=float), np.asarray(wps, dtype=float))
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            done = term or trunc
        reached.append(int(info["waypoints_reached"]))
    return np.asarray(reached)


def main():
    args = parse_args()
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    sets, shoulder, old = episodes(args.episodes, args.seed)
    tags = [[regions_of(p, shoulder, old) for p in wps] for wps in sets]
    results = {}
    for pkg in args.packages:
        reached = run(pkg, sets, args.which)
        stats = defaultdict(lambda: [0, 0])          # region -> [completed, attempted]
        for n_done, wps_tags in zip(reached, tags):
            for k, t in enumerate(wps_tags):
                if k > n_done:
                    continue                          # never got to this one
                for region in t + ["all targets"]:
                    stats[region][1] += 1
                    stats[region][0] += int(k < n_done)
        name = Path(pkg).name
        results[name] = {"episode_success": float(np.mean(reached == 3)),
                         "regions": {k: {"completed": v[0], "attempted": v[1]} for k, v in stats.items()}}
        print(f"\n{name}: all three targets + hold in {np.mean(reached == 3):.0%} of {len(sets)} episodes")
        for region in ["all targets"] + REGIONS:
            k, n = stats.get(region, [0, 0])
            lo, hi = wilson(k, n)
            print(f"  {region:12s} {k:4d}/{n:<4d} = {k / max(n, 1):5.0%}   (95% CI {lo:.0%}-{hi:.0%})")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
