"""
Does the trained arm actually respect human limits, and does it do what we said?

    python scripts/verify_human_limits.py project_assets/outputs/CHAMPION_HUMAN_V1_R3
    python scripts/verify_human_limits.py <pkg> --episodes 40 --json report.json

Success rate says the arm reaches the points. It says nothing about *how*. This
walks every step of every episode and checks two different kinds of claim.

HARD LIMITS are constraints the physics and the configuration are supposed to
make impossible to break. Any violation is a bug in the model, not a training
outcome, so these are pass/fail:

  * joint angles inside the range of motion (shoulder -150..+90, elbow 0..145)
  * joint speeds inside the configured velocity limit
  * actuator torque never above the joint's torque limit (25 / 12 N.m)
  * muscle activations inside [0, 1]
  * every state, observation and reward finite

HUMAN-LIKENESS BANDS are empirical ranges from the motor-control literature.
Falling outside one is not a crash, it is a finding - a real arm that moves
like a machine still moves like a machine - so these are reported with their
reference range and flagged, not failed:

  * path straightness 1.00-1.15   (Morasso 1981: reaching paths are near-straight)
  * velocity-profile symmetry 0.35-0.55 (bell-shaped, peak near the middle)
  * submovements 1-2 per reach    (single smooth stroke, not a series of jabs)
  * dimensionless jerk < 60       (smoothness; Hogan & Sternad 2009)
  * peak hand speed <= ~1.5 m/s   (the configured cap for 30-50 cm reaches)
  * reach duration 0.4-1.2 s      (human point-to-point at this amplitude)

CAPABILITIES are the things the plan promised the arm would be able to do, each
measured rather than asserted: reach arbitrary feasible points, hold a pose
against gravity without drifting, stay sub-maximal in torque, avoid end-range
postures, and stiffen by co-contraction when holding rather than while moving.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rl_armMotion.two_d.evaluation import EVAL_STANDARD, load_package  # noqa: E402
from rl_armMotion.two_d.kinematics_metrics import HUMAN_REFERENCE, episode_metrics  # noqa: E402

BANDS = {
    "straightness": (1.00, 1.15, "path length / straight-line distance"),
    "symmetry": (0.35, 0.55, "time to peak speed / movement time"),
    "submovements": (1.0, 2.0, "speed peaks per reach"),
    "dimensionless_jerk": (0.0, 60.0, "smoothness"),
    "peak_speed": (0.0, 1.5, "m/s, the configured human cap"),
    "movement_time": (0.4, 1.2, "s per reach"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("package")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--which", default="best")
    p.add_argument("--seed", type=int, default=EVAL_STANDARD.seed + 1234)
    p.add_argument("--json", default=None, help="write the full report here")
    return p.parse_args()


def rollout(model, env, ep_seed):
    obs, info = env.reset(seed=ep_seed)
    rec = {k: [] for k in ("q", "qd", "tau", "act", "ee", "ee_speed", "end_range", "at_limit",
                           "in_goal", "hold", "reward", "wp_index")}
    order = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rec["q"].append(np.asarray(info["joint_angles"], float))
        rec["qd"].append(np.asarray(info["joint_velocities"], float))
        rec["tau"].append(np.asarray(info["actuator_torque"], float))
        rec["act"].append(np.asarray(info.get("muscle_activations", []), float))
        rec["ee"].append(np.asarray(info["end_effector_position"], float))
        rec["ee_speed"].append(float(info["ee_speed"]))
        rec["end_range"].append(np.asarray(info["end_range_engagement"], float))
        rec["at_limit"].append(np.asarray(info["at_joint_limit"], bool))
        rec["in_goal"].append(bool(info["in_goal_region"]))
        rec["hold"].append(float(info["hold_progress"]))
        rec["reward"].append(float(reward))
        rec["wp_index"].append(int(info["current_waypoint_index"]))
        if info.get("waypoint_completed") is not None:
            order.append(int(info["waypoint_completed"]))
        if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
            raise FloatingPointError(f"non-finite step at {info['step']}")
        if terminated or truncated:
            break
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["success"] = bool(info["success"])
    out["reached"] = int(info["waypoints_reached"])
    out["n_waypoints"] = int(info["num_waypoints"])
    out["steps"] = int(info["step"])
    out["success_step"] = info.get("success_step")
    out["order"] = order
    return out


def main():
    args = parse_args()
    model, env, manifest = load_package(args.package, which=args.which, terminate_on_success=True)
    base = getattr(env, "unwrapped", env)
    cfg = base.config
    qmin = np.asarray(cfg.joint_limits_min, float)
    qmax = np.asarray(cfg.joint_limits_max, float)
    tau_max = np.asarray(cfg.torque_limits, float)
    vel_max = float(cfg.velocity_limits)

    print(f"package {args.package} ({args.which})")
    print(f"preset {base.preset_name}, {base.actuation_mode} actuation, "
          f"{'free order' if base.free_order else 'ordered'}, {args.episodes} episodes\n")

    eps = [rollout(model, env, args.seed + i) for i in range(args.episodes)]

    # ---------------- hard limits ----------------
    q = np.concatenate([e["q"] for e in eps])
    qd = np.concatenate([e["qd"] for e in eps])
    tau = np.concatenate([e["tau"] for e in eps])
    act = np.concatenate([e["act"] for e in eps]) if eps[0]["act"].size else np.zeros((1, 1))
    rom_lo = float(np.degrees(np.min(q - qmin)))
    rom_hi = float(np.degrees(np.min(qmax - q)))
    checks = [
        ("joint angles inside the range of motion", rom_lo >= -1e-6 and rom_hi >= -1e-6,
         f"closest approach: {min(rom_lo, rom_hi):.2f} deg of margin"),
        ("joint speeds inside the velocity limit", float(np.max(np.abs(qd))) <= vel_max + 1e-6,
         f"peak {float(np.max(np.abs(qd))):.2f} of {vel_max:.1f} rad/s"),
        ("actuator torque inside the torque limits", bool(np.all(np.abs(tau) <= tau_max + 1e-6)),
         f"peak {np.round(np.max(np.abs(tau), axis=0), 2).tolist()} of {tau_max.tolist()} N.m"),
        ("muscle activations inside [0, 1]", bool(act.min() >= -1e-9 and act.max() <= 1 + 1e-9),
         f"range {act.min():.3f}..{act.max():.3f}"),
        ("every state, observation and reward finite", True, "no non-finite step raised"),
    ]
    print("HARD LIMITS (a violation is a model bug, not a training outcome)")
    worst = True
    for name, ok, detail in checks:
        worst &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<48} {detail}")

    # ---------------- human-likeness ----------------
    print("\nHUMAN-LIKENESS (reference ranges from the motor-control literature)")
    print(f"  {'metric':<22} {'measured':>18} {'human range':>16}   verdict")
    mets = {}
    for e in eps:
        m = episode_metrics(e["ee"], e["wp_index"], float(base.dt), e.get("success_step"))
        for k, v in (m or {}).items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                mets.setdefault(k, []).append(float(v))
    flagged = []
    for key, (lo, hi, note) in BANDS.items():
        if key not in mets:
            continue
        vals = np.asarray(mets[key])
        med = float(np.median(vals))
        inside = lo <= med <= hi
        if not inside:
            flagged.append((key, med, lo, hi, note))
        print(f"  {key:<22} {med:>10.2f} (p90 {np.percentile(vals, 90):>5.2f}) "
              f"{f'{lo:.2f}-{hi:.2f}':>16}   {'in band' if inside else 'OUTSIDE'}")

    # ---------------- promised capabilities ----------------
    print("\nCAPABILITIES (each measured, not asserted)")
    succ = sum(e["success"] for e in eps)
    reached = np.mean([e["reached"] / max(e["n_waypoints"], 1) for e in eps])
    print(f"  reaches arbitrary feasible points      {succ}/{len(eps)} episodes complete, "
          f"{reached:.0%} of waypoints visited")

    drift, hold_speed = [], []
    for e in eps:
        if not e["success"]:
            continue
        held = np.flatnonzero(e["hold"] >= 0.99)
        if held.size:
            seg = e["ee"][held[0]:]
            drift.append(float(np.max(np.linalg.norm(seg - seg[0], axis=1))) if len(seg) > 1 else 0.0)
            hold_speed.append(float(np.median(e["ee_speed"][held[0]:])))
    if drift:
        print(f"  holds a pose against gravity           drift {np.median(drift) * 1000:.0f} mm, "
              f"hand speed {np.median(hold_speed):.3f} m/s while holding")
    print(f"  stays sub-maximal in torque            peak "
          f"{np.round(100 * np.max(np.abs(tau), axis=0) / tau_max, 0).tolist()} % of the limits")
    er = np.concatenate([e["end_range"] for e in eps])
    at = np.concatenate([e["at_limit"] for e in eps])
    print(f"  avoids end-range postures              {np.mean(np.any(er > 0, axis=1)):.1%} of steps in a "
          f"passive end-range zone, {np.mean(np.any(at, axis=1)):.2%} resting on a hard stop")
    if act.shape[1] >= 4:
        pairs = act.reshape(act.shape[0], -1, 2)
        cocon = np.min(pairs, axis=2).mean(axis=1)
        moving = np.concatenate([e["ee_speed"] for e in eps]) > 0.2
        holding = np.concatenate([np.asarray(e["in_goal"]) for e in eps])
        print(f"  stiffens by co-contraction to hold     co-contraction "
              f"{cocon[holding].mean():.3f} while holding vs {cocon[moving].mean():.3f} while moving")
    if base.free_order:
        orders = [tuple(e["order"]) for e in eps if len(e["order"]) == e["n_waypoints"]]
        uniq = {o: orders.count(o) for o in set(orders)}
        print(f"  chooses its own route                  {len(uniq)} distinct orders over "
              f"{len(orders)} complete episodes: {uniq}")

    print(f"\n{'ALL HARD LIMITS RESPECTED' if worst else 'HARD LIMIT VIOLATED - investigate before using this model'}"
          + (f"; {len(flagged)} human-likeness metric(s) outside the reference band" if flagged else
             "; every human-likeness metric inside its band"))
    for key, med, lo, hi, note in flagged:
        print(f"   - {key} {med:.2f} vs {lo:.2f}-{hi:.2f} ({note})")

    if args.json:
        payload = {
            "package": str(args.package), "episodes": len(eps),
            "hard_limits": {name: bool(ok) for name, ok, _ in checks},
            "human_likeness": {k: float(np.median(v)) for k, v in mets.items()},
            "flagged": [k for k, *_ in flagged],
            "success_rate": succ / len(eps),
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nreport -> {args.json}")
    return 0 if worst else 1


if __name__ == "__main__":
    sys.exit(main())
