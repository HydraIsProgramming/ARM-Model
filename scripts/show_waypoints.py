"""
Visualize a trained SAC model navigating 3 waypoints.
Shows an animated matplotlib window with the arm path and waypoint markers.

Usage:
    python scripts/show_waypoints.py [model_zip_path] [--waypoints "x1,y1;x2,y2;x3,y3"]

Defaults to seed_001 with its original training waypoints.
"""

import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import SAC
from rl_armMotion.two_d.environments.task_env import ArmTaskEnv

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL = str(Path(__file__).parent.parent /
                    "project_assets/outputs/Parallel_Seeds/seed_001/sac_model")
DEFAULT_WAYPOINTS = [[2.3, -1.0], [1.0, 1.5], [2.5, 0.0]]   # seed_001 training waypoints
TOLERANCE = 0.6

WP_COLORS  = ["#2ecc71", "#3498db", "#e74c3c"]   # green, blue, red
WP_LABELS  = ["WP1", "WP2", "WP3"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    p.add_argument("--waypoints", default=None,
                   help='e.g. "2.3,-1.0;1.0,1.5;2.5,0.0"')
    p.add_argument("--tolerance", type=float, default=TOLERANCE)
    return p.parse_args()


def run_episode(model, env):
    obs, _ = env.reset()
    positions, waypoint_times = [], {}
    for step in range(800):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        # record end-effector position via controller
        ee = env.controller.get_end_effector_position()
        x, y = ee[0], ee[1]
        positions.append((x, y, info.get("current_waypoint_index", 0)))

        wp = info.get("current_waypoint_index", 0)
        if wp not in waypoint_times:
            waypoint_times[wp] = step

        if term or trunc:
            break

    return positions, waypoint_times, info


def main():
    args = parse_args()

    waypoints = DEFAULT_WAYPOINTS
    if args.waypoints:
        waypoints = [[float(v) for v in seg.split(",")] for seg in args.waypoints.split(";")]

    print(f"Loading model: {args.model}")
    model = SAC.load(args.model)

    env = ArmTaskEnv(actuation_mode="muscle", goal_direction="EAST")
    env.set_waypoints(waypoints, tolerance=args.tolerance)

    print("Running episode...")
    positions, wp_times, final_info = run_episode(model, env)
    reached = final_info.get("num_waypoints", 0)
    print(f"Result: {reached}/3 waypoints reached")
    print(f"Waypoint arrival steps: {wp_times}")

    # ── Build animation ───────────────────────────────────────────────────────
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    wps = [p[2] for p in positions]

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    ax.set_xlim(-2.5, 3.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Y (m)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # workspace boundary
    circle = plt.Circle((1.0, 0.0), 1.8, color="#ffffff22", fill=False, linestyle="--", linewidth=0.8)
    ax.add_patch(circle)
    ax.plot(1.0, 0.0, "o", color="#aaaaaa", markersize=5)   # shoulder

    # waypoint markers
    for i, (wx, wy) in enumerate(waypoints):
        c = plt.Circle((wx, wy), args.tolerance, color=WP_COLORS[i], alpha=0.18, fill=True)
        ax.add_patch(c)
        ax.plot(wx, wy, "*", color=WP_COLORS[i], markersize=18, zorder=5)
        ax.text(wx+0.08, wy+0.12, WP_LABELS[i], color=WP_COLORS[i], fontsize=12, fontweight="bold")

    # path trail (will be filled in)
    trail_line, = ax.plot([], [], "-", color="#ffffff55", linewidth=1.2, zorder=3)
    dot, = ax.plot([], [], "o", color="white", markersize=9, zorder=6)

    title = ax.set_title("", color="white", fontsize=13)
    status_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                          color="white", fontsize=10, va="top",
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="#00000088"))

    patches = [mpatches.Patch(color=WP_COLORS[i], label=WP_LABELS[i]) for i in range(len(waypoints))]
    ax.legend(handles=patches, loc="lower right", facecolor="#222", edgecolor="#555", labelcolor="white")

    skip = max(1, len(positions) // 300)  # cap animation frames for speed

    def init():
        trail_line.set_data([], [])
        dot.set_data([], [])
        return trail_line, dot, title, status_text

    def update(frame):
        idx = min(frame * skip, len(positions) - 1)
        trail_line.set_data(xs[:idx+1], ys[:idx+1])
        dot.set_data([xs[idx]], [ys[idx]])
        wp_now = wps[idx]
        color = WP_COLORS[min(wp_now, 2)]
        dot.set_color(color)
        pct = int(idx / max(len(positions)-1, 1) * 100)
        title.set_text(f"SAC — 2-DOF Arm  |  Step {idx}/{len(positions)-1}  |  Heading → {WP_LABELS[min(wp_now, 2)]}")
        status_text.set_text(
            f"Waypoints reached: {wp_now}/3\n"
            f"End-effector: ({xs[idx]:.2f}, {ys[idx]:.2f})\n"
            f"Final result: {reached}/3"
        )
        return trail_line, dot, title, status_text

    n_frames = len(positions) // skip + 1
    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        interval=30, blit=True, repeat=True)

    plt.tight_layout()
    print("Showing animation... close the window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
