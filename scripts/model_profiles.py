"""
Hand velocity, hand acceleration, joint angles and joint torques for every model on the SAME three points.

    python scripts/model_profiles.py                        # every CHAMPION_HUMAN_V1_* package
    python scripts/model_profiles.py --extra R8E=C:/arm-model-runs/stageR8E/seed_2 --extra ...

Every model starts from the same pose (start-pose jitter off) and is given the same targets,
CANONICAL_WAYPOINTS[3] = (1.40, 0.15), (1.20, 0.40), (1.40, -0.20) in the workspace frame (shoulder at
(1.0, 0.0)) - B3's training set, inside every random-point model's region. Models built for two
targets (A: 2 fixed points; F2: free order with exactly 2 slots) get the first and last of the three.
Each rollout is deterministic and lasts 5 s (250 steps), so every profile shares one time axis.

Outputs in project_assets/report/profiles/:
  <model>.csv              one row per 20 ms step
  profiles_all.csv         all models stacked (column `model`)
  profiles_summary.csv     one row per model: success, timing, peaks, RMS values, smoothness
  <model>.png              4 panels: hand velocity, hand acceleration, joint angles, joint torques
  overlay_*.png            each quantity, every model on one plot

Hand acceleration is not in the env's info: it is the change in the simulator's hand velocity over
one 20 ms step, the same quantity the acceleration penalty uses.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rl_armMotion.two_d.evaluation import CANONICAL_WAYPOINTS, load_package, read_manifest  # noqa: E402

OUT = ROOT / "project_assets" / "report" / "profiles"
POINTS = [list(p) for p in CANONICAL_WAYPOINTS[3]]
DURATION_STEPS = 250
ORDER = ["A", "B", "B3", "R3", "R4", "R5", "R6", "R7", "F2", "R8", "R8F", "R8A", "R8P", "R8E", "R8N", "R8U", "R8T"]
SHOULDER_C, ELBOW_C, INK, MUTED = "#2a78d6", "#e8743b", "#1b1d21", "#5d6470"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extra", action="append", default=[], metavar="NAME=RUN_DIR",
                   help="also profile a run directory under this name (e.g. a variant's best seed)")
    p.add_argument("--only", default=None, help="comma-separated model names to include")
    p.add_argument("--out", default=str(OUT))
    return p.parse_args()


def discover(extra, only):
    models = {}
    for d in sorted((ROOT / "project_assets" / "outputs").glob("CHAMPION_HUMAN_V1_*")):
        if (d / "run_manifest.json").exists():
            models[d.name.rsplit("_", 1)[-1]] = d
    for item in extra:
        name, _, path = item.partition("=")
        models[name.strip()] = Path(path.strip())
    if only:
        keep = {s.strip() for s in only.split(",")}
        models = {k: v for k, v in models.items() if k in keep}
    rank = {k: i for i, k in enumerate(ORDER)}
    return dict(sorted(models.items(), key=lambda kv: (rank.get(kv[0], len(ORDER)), kv[0])))


def points_for(manifest):
    env = manifest.get("env", {})
    two_point = env.get("free_order") and int(env.get("free_order_slots") or 0) == 2
    fixed = env.get("waypoints") if not env.get("random_waypoints") else None
    if two_point or (fixed is not None and len(fixed) == 2):
        return [POINTS[0], POINTS[2]], "given points 1 and 3 (built for two targets)"
    note = ""
    if fixed is not None and not np.allclose(np.asarray(fixed, float), np.asarray(POINTS, float)):
        note = "trained on other fixed points; not expected to reach these"
    return POINTS, note


def rollout(name, pkg):
    manifest = read_manifest(pkg)
    pts, note = points_for(manifest)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, env, _ = load_package(pkg, waypoints=pts, terminate_on_success=False)
    base = getattr(env, "unwrapped", env)
    obs, info = env.reset(seed=0, options={"randomize": False})
    shoulder = np.asarray(info["shoulder_position"], float)
    dt = float(base.dt)
    prev_v = np.zeros(2)
    rows, completions = [], []
    success_step = None
    for k in range(DURATION_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)
        v = np.asarray(info["ee_velocity"], float)[:2]
        a = (v - prev_v) / dt
        prev_v = v
        q = np.degrees(np.asarray(info["joint_angles"], float))
        qd = np.degrees(np.asarray(info["joint_velocities"], float))
        tau = np.asarray(info["actuator_torque"], float)
        grav = np.asarray(info["gravity_torque"], float)
        acts = info.get("muscle_activations")
        acts = np.asarray(acts, float) if acts is not None else np.full(4, np.nan)
        ee = np.asarray(info["end_effector_position"], float) - shoulder
        if info.get("waypoint_completed") is not None:
            completions.append((k + 1) * dt)
        if info.get("success") and success_step is None:
            success_step = int(info["success_step"])
            completions.append(success_step * dt)
        rows.append({
            "model": name, "t": (k + 1) * dt, "hand_x": ee[0], "hand_y": ee[1],
            "vx": v[0], "vy": v[1], "speed": float(np.linalg.norm(v)),
            "ax": a[0], "ay": a[1], "accel": float(np.linalg.norm(a)),
            "shoulder_deg": q[0], "elbow_deg": q[1], "shoulder_dps": qd[0], "elbow_dps": qd[1],
            "tau_shoulder": tau[0], "tau_elbow": tau[1], "gravity_shoulder": grav[0], "gravity_elbow": grav[1],
            "act_sh_flex": acts[0], "act_sh_ext": acts[1], "act_el_flex": acts[2], "act_el_ext": acts[3],
            "target_index": int(info["current_waypoint_index"]), "in_goal": bool(info["in_goal_region"]),
            "hold": float(info["hold_progress"]), "success": bool(info["success"]),
        })
    df = pd.DataFrame(rows)
    env_desc = manifest.get("env", {})
    meta = {"model": name, "actuation": env_desc.get("actuation_mode", "muscle"),
            "algorithm": str(manifest.get("algorithm", "SAC")).upper(), "n_points": len(pts),
            "reached": int(info["waypoints_reached"]), "success": bool(info["success"]),
            "time_to_success_s": None if success_step is None else success_step * dt, "note": note,
            "completions": completions}
    return df, meta


def summarise(df, meta, dt=0.02):
    moving = df[df.t <= (meta["time_to_success_s"] or df.t.max())]
    jerk = np.gradient(np.gradient(np.gradient(moving[["hand_x", "hand_y"]].to_numpy(), dt, axis=0), dt, axis=0),
                       dt, axis=0)
    path = float(np.sum(np.linalg.norm(np.diff(moving[["hand_x", "hand_y"]].to_numpy(), axis=0), axis=1)))
    T = float(moving.t.max() - moving.t.min()) if len(moving) > 1 else 0.0
    peak_v = float(moving.speed.max())
    djerk = float(np.sum(np.sum(jerk ** 2, axis=1)) * dt * T ** 5 / max(path, 1e-6) ** 2) if T > 0 else float("nan")
    tau_rate = np.abs(np.diff(df[["tau_shoulder", "tau_elbow"]].to_numpy(), axis=0)) / dt
    out = {k: meta[k] for k in ("model", "actuation", "algorithm", "n_points", "reached", "success",
                                "time_to_success_s", "note")}
    out.update({
        "peak_speed_mps": peak_v, "peak_accel_mps2": float(moving.accel.max()),
        "rms_accel_mps2": float(np.sqrt(np.mean(moving.accel ** 2))),
        "path_length_m": path, "dimensionless_jerk": djerk,
        "peak_tau_shoulder": float(df.tau_shoulder.abs().max()), "peak_tau_elbow": float(df.tau_elbow.abs().max()),
        "rms_tau_shoulder": float(np.sqrt(np.mean(df.tau_shoulder ** 2))),
        "rms_tau_elbow": float(np.sqrt(np.mean(df.tau_elbow ** 2))),
        "rms_torque_rate_shoulder": float(np.sqrt(np.mean(tau_rate[:, 0] ** 2))),
        "rms_torque_rate_elbow": float(np.sqrt(np.mean(tau_rate[:, 1] ** 2))),
        "accel_steps_over_10": float(np.mean(moving.accel > 10)), "accel_steps_over_20": float(np.mean(moving.accel > 20)),
    })
    return out


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("time (s)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def model_figure(df, meta, out):
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True)
    t = df.t
    a = axs[0, 0]
    a.plot(t, df.vx, color=SHOULDER_C, lw=1.1, label="Vx")
    a.plot(t, df.vy, color=ELBOW_C, lw=1.1, label="Vy")
    a.plot(t, df.speed, color=INK, lw=1.6, label="speed")
    style(a, "Hand velocity", "m/s")
    a = axs[0, 1]
    a.plot(t, df.ax, color=SHOULDER_C, lw=1.0, label="Ax")
    a.plot(t, df.ay, color=ELBOW_C, lw=1.0, label="Ay")
    a.plot(t, df.accel, color=INK, lw=1.4, label="|a|")
    style(a, "Hand acceleration", "m/s²")
    a = axs[1, 0]
    a.plot(t, df.shoulder_deg, color=SHOULDER_C, lw=1.5, label="shoulder")
    a.plot(t, df.elbow_deg, color=ELBOW_C, lw=1.5, label="elbow")
    style(a, "Joint angles", "degrees")
    a = axs[1, 1]
    a.plot(t, df.tau_shoulder, color=SHOULDER_C, lw=1.4, label="shoulder")
    a.plot(t, df.tau_elbow, color=ELBOW_C, lw=1.4, label="elbow")
    a.plot(t, df.gravity_shoulder, color=SHOULDER_C, lw=0.9, ls="--", label="gravity load")
    a.plot(t, df.gravity_elbow, color=ELBOW_C, lw=0.9, ls="--")
    style(a, "Joint torque" + (" (actuators)" if meta["actuation"] == "torque" else " (muscles)"), "N·m")
    for ax in axs.flat:
        for c in meta["completions"]:
            ax.axvline(c, color=MUTED, lw=0.7, ls=":")
        ax.legend(fontsize=7, loc="upper right")
    status = (f"all {meta['n_points']} targets + hold in {meta['time_to_success_s']:.2f} s" if meta["success"]
              else f"reached {meta['reached']} of {meta['n_points']} targets")
    fig.suptitle(f"{meta['model']} - {meta['actuation']} actuation, {meta['algorithm']}: {status}"
                 + (f"  ({meta['note']})" if meta["note"] else ""), fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / f"{meta['model']}.png", dpi=140)
    plt.close(fig)


def overlay_figures(all_df, metas, out):
    names = list(metas)
    cmap = plt.get_cmap("tab20")
    colours = {n: cmap(i % 20) for i, n in enumerate(names)}
    panels = [("speed", "Hand speed", "m/s", None), ("accel", "Hand acceleration magnitude", "m/s²", None),
              ("shoulder_deg", "Shoulder angle", "degrees", None), ("elbow_deg", "Elbow angle", "degrees", None),
              ("tau_shoulder", "Shoulder torque", "N·m", None), ("tau_elbow", "Elbow torque", "N·m", None)]
    for col, title, unit, _ in panels:
        fig, ax = plt.subplots(figsize=(11, 4.2))
        for n in names:
            d = all_df[all_df.model == n]
            ls = "-" if metas[n]["actuation"] == "torque" else "--"
            ax.plot(d.t, d[col], color=colours[n], lw=1.3, ls=ls,
                    label=f"{n} ({'actuators' if metas[n]['actuation'] == 'torque' else 'muscles'})")
        style(ax, f"{title} - every model, same 3 points", unit)
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        fig.tight_layout()
        fig.savefig(out / f"overlay_{col}.png", dpi=140)
        plt.close(fig)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = discover(args.extra, args.only)
    frames, metas, summary = [], {}, []
    for name, pkg in models.items():
        try:
            df, meta = rollout(name, pkg)
        except Exception as exc:
            print(f"{name}: skipped ({exc})")
            continue
        df.to_csv(out / f"{name}.csv", index=False)
        model_figure(df, meta, out)
        frames.append(df)
        metas[name] = meta
        summary.append(summarise(df, meta))
        s = summary[-1]
        print(f"{name:5s} {meta['actuation']:6s} {'OK ' if meta['success'] else '-- '} reached {meta['reached']}/"
              f"{meta['n_points']}  t={s['time_to_success_s'] if s['time_to_success_s'] else float('nan'):5.2f}s  "
              f"peak speed {s['peak_speed_mps']:.2f} m/s  peak accel {s['peak_accel_mps2']:5.1f} m/s²  "
              f"RMS torque rate {s['rms_torque_rate_shoulder']:6.1f}/{s['rms_torque_rate_elbow']:6.1f} N·m/s"
              + (f"  [{meta['note']}]" if meta["note"] else ""))
    if not frames:
        return 1
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out / "profiles_all.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "profiles_summary.csv", index=False)
    overlay_figures(all_df, metas, out)
    (out / "points.json").write_text(json.dumps({"points_workspace_frame": POINTS, "duration_s": DURATION_STEPS * 0.02,
                                                 "start_pose": "preset initial angles, jitter off"}, indent=2))
    print(f"wrote {len(frames)} models to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
