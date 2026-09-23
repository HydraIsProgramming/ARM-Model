"""
Animate a trained policy moving the human arm through its waypoints.

    python scripts/show_waypoints.py <package_dir_or_model.zip> [--n-waypoints 2|3 | --waypoints "x,y;x,y"]
                                     [--episodes 3] [--seed 0] [--gif out.gif] [--no-smooth]

The env is rebuilt from the package's run_manifest.json (same preset,
tolerances, waypoints as training). Links are drawn from the joint angles the
env reports each step; the hand trail, tolerance circles, active-waypoint
highlight and hold progress are all taken from ``info``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rl_armMotion.two_d.environments.task_env import ActionSmoother  # noqa: E402
from rl_armMotion.two_d.evaluation import (  # noqa: E402
    CANONICAL_WAYPOINTS, EVAL_STANDARD, detect_actuation_mode, find_manifest, load_package, make_env,
    resolve_waypoints,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Animate a trained policy through its waypoints")
    p.add_argument("model", help="package directory, or a model .zip inside one")
    p.add_argument("--which", default="best", help="best | last | final | step_XXXXXXX")
    p.add_argument("--n-waypoints", type=int, default=None, choices=[1, 2, 3])
    p.add_argument("--waypoints", default=None, help='"x,y;x,y;..." or "spread"')
    p.add_argument("--tolerance", type=float, default=None, help="default: evaluation standard")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-smooth", action="store_true", help="disable the demo action smoother")
    p.add_argument("--smooth-alpha", type=float, default=0.3)
    p.add_argument("--stop-on-success", action="store_true", help="end the episode when the hold completes")
    p.add_argument("--gif", default=None, help="write the animation to this GIF (PillowWriter)")
    p.add_argument("--mp4", default=None, help="write the animation to this MP4 (needs ffmpeg)")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args(argv)


def build(args) -> Tuple[Any, Any]:
    """(model, env) from a package; falls back to a bare zip + canonical env."""
    overrides: Dict[str, Any] = {"terminate_on_success": bool(args.stop_on_success)}
    wps = None
    if args.waypoints:
        wps = resolve_waypoints(args.waypoints)
    elif args.n_waypoints:
        wps = resolve_waypoints(args.n_waypoints)
    if wps is not None:
        overrides["waypoints"] = wps
    if args.tolerance is not None:
        overrides["goal_tolerance"] = args.tolerance
    if find_manifest(args.model) is not None:
        model, env, _ = load_package(args.model, which=args.which, **overrides)
    else:
        from stable_baselines3 import SAC
        model = SAC.load(str(Path(args.model).with_suffix("")), device="cpu")
        env = make_env(actuation_mode=detect_actuation_mode(model),
                       waypoints=wps if wps is not None else 2,
                       tolerance=args.tolerance, terminate_on_success=bool(args.stop_on_success))
        model.set_env(env)
    if not args.no_smooth and env.actuation_mode == "muscle":
        env = ActionSmoother(env, alpha=args.smooth_alpha)
    return model, env


def run_episode(model, env, seed: int, max_steps: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Roll out one episode; returns per-step frames (from ``info``) and the final info."""
    base = getattr(env, "unwrapped", env)
    max_steps = int(base.max_episode_steps if max_steps is None else max_steps)
    obs, info = env.reset(seed=seed)
    frames = [_frame(info)]
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(action)
        frames.append(_frame(info))
        if term or trunc:
            break
    return frames, info


def _frame(info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "links": np.asarray(info["link_positions"], dtype=float),
        "hand": np.asarray(info["end_effector_position"], dtype=float),
        "wp_index": int(info["current_waypoint_index"]),
        "hold": float(info["hold_progress"]),
        "in_goal": bool(info["in_goal_region"]),
        "success": bool(info["success"]),
        "reached": int(info["waypoints_reached"]),
        "visited": list(info.get("visited") or []),
        "step": int(info["step"]),
        "speed": float(info.get("ee_speed", 0.0)),
    }


def animate(frames_by_episode: List[List[Dict[str, Any]]], env, title: str, fps: int = 25,
            gif: Optional[str] = None, mp4: Optional[str] = None, show: bool = True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    base = getattr(env, "unwrapped", env)
    wps = [np.asarray(w, dtype=float) for w in (base.waypoints or [base.goal_position])]
    tol = float(base.goal_tolerance)
    shoulder = np.asarray(base.shoulder_base_position, dtype=float)
    reach = float(base.reach)
    n_wp = len(wps)
    colours = plt.cm.tab10(np.arange(max(n_wp, 1)) % 10)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.set_xlim(shoulder[0] - 1.1 * reach, shoulder[0] + 1.1 * reach)
    ax.set_ylim(shoulder[1] - 1.1 * reach, shoulder[1] + 1.1 * reach)
    ax.set_aspect("equal")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.set_xlabel("x (m)  [+x = in front of the body]", color="white")
    ax.set_ylabel("y (m)", color="white")

    # Torso / head silhouette (side view) so the audience reads a human arm.
    torso_w, torso_h = 0.22 * reach, 0.9 * reach
    ax.add_patch(plt.Rectangle((shoulder[0] - torso_w, shoulder[1] - torso_h), torso_w, torso_h,
                               color="#ffffff14", zorder=0))
    ax.add_patch(plt.Circle((shoulder[0] - 0.5 * torso_w, shoulder[1] + 0.32 * reach), 0.17 * reach,
                            color="#ffffff14", zorder=0))
    ax.add_patch(plt.Circle(tuple(shoulder), reach, color="#ffffff22", fill=False, linestyle="--", linewidth=0.8))

    # In free-order mode the arm chooses its own route, so the picture must not
    # imply one: no line joining the points in list order and no WP1/WP2/WP3
    # labels. The points are just points; they turn green as they are visited.
    free_order = bool(getattr(base, "free_order", False))
    PENDING, DONE = "#9aa0a6", "#2ecc71"
    wp_fills, wp_rings, wp_dots = [], [], []
    for i, (wp, c) in enumerate(zip(wps, colours)):
        c = PENDING if free_order else c
        fill = plt.Circle(tuple(wp), tol, color=c, alpha=0.18)
        ring = plt.Circle(tuple(wp), tol, color=c, fill=False, linewidth=1.0)
        ax.add_patch(fill)
        ax.add_patch(ring)
        dot, = ax.plot([wp[0]], [wp[1]], "*", color=c, markersize=16, zorder=5)
        if not free_order:
            ax.text(wp[0] + 0.02, wp[1] + 0.03, f"WP{i + 1}", color=c, fontsize=11, fontweight="bold")
        wp_fills.append(fill)
        wp_rings.append(ring)
        wp_dots.append(dot)
    if n_wp >= 2 and not free_order:
        arr = np.asarray(wps)
        ax.plot(arr[:, 0], arr[:, 1], "--", color="#ffffff55", linewidth=0.8)

    trail, = ax.plot([], [], "-", color="#ffffff66", linewidth=1.2, zorder=3)
    links, = ax.plot([], [], "-o", color="#f5f5f5", linewidth=4, markersize=7, zorder=4)
    hand, = ax.plot([], [], "o", color="white", markersize=10, zorder=6)
    active, = ax.plot([], [], "o", color="none", markeredgecolor="yellow", markersize=34, markeredgewidth=2, zorder=5)
    ttl = ax.set_title(title, color="white", fontsize=12)
    status = ax.text(0.02, 0.97, "", transform=ax.transAxes, color="white", fontsize=10, va="top",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#00000088"))
    bar_bg = ax.add_patch(plt.Rectangle((0.02, 0.02), 0.30, 0.03, transform=ax.transAxes, color="#ffffff33"))
    bar = ax.add_patch(plt.Rectangle((0.02, 0.02), 0.0, 0.03, transform=ax.transAxes, color="#2ecc71"))

    seq = [(e, j) for e, fr in enumerate(frames_by_episode) for j in range(len(fr))]
    dt = float(base.dt)
    stride = max(1, int(round(1.0 / (fps * dt))))
    idxs = list(range(0, len(seq), stride))

    def update(k):
        e, j = seq[idxs[k]]
        ep_frames = frames_by_episode[e]
        f = ep_frames[j]
        L = f["links"]
        links.set_data(L[:, 0], L[:, 1])
        hand.set_data([f["hand"][0]], [f["hand"][1]])
        xs = [fr["hand"][0] for fr in ep_frames[:j + 1]]
        ys = [fr["hand"][1] for fr in ep_frames[:j + 1]]
        trail.set_data(xs, ys)
        if free_order:
            vis = f.get("visited") or []
            for i in range(n_wp):
                c = DONE if (i < len(vis) and vis[i]) else PENDING
                wp_fills[i].set_color(c)
                wp_fills[i].set_alpha(0.18)
                wp_rings[i].set_edgecolor(c)
                wp_dots[i].set_color(c)
            active.set_data([], [])          # no "heading" ring: the route is the arm's
            hand.set_color("white")
        else:
            wp = wps[min(f["wp_index"], n_wp - 1)]
            active.set_data([wp[0]], [wp[1]])
            hand.set_color(colours[min(f["wp_index"], n_wp - 1)])
        bar.set_width(0.30 * f["hold"])
        bar.set_color("#2ecc71" if f["in_goal"] else "#e67e22")
        state = "HOLD COMPLETE" if f["success"] else ("settling" if f["in_goal"] else "moving")
        second = (f"visited: {f['reached']}/{n_wp}   (the arm chose the order)" if free_order
                  else f"waypoints reached: {f['reached']}/{n_wp}   heading: WP{min(f['wp_index'] + 1, n_wp)}")
        status.set_text(f"episode {e + 1}/{len(frames_by_episode)}   t = {f['step'] * dt:4.2f} s\n"
                        f"{second}\n"
                        f"hand speed: {f['speed']:.2f} m/s   hold: {int(round(100 * f['hold']))}%   {state}")
        return links, hand, trail, active, bar, status

    ani = FuncAnimation(fig, update, frames=len(idxs), interval=1000.0 / fps, blit=False, repeat=True)
    plt.tight_layout()
    if gif:
        ani.save(gif, writer=PillowWriter(fps=fps))
        print(f"wrote {gif}")
    if mp4:
        try:
            ani.save(mp4, writer="ffmpeg", fps=fps)
            print(f"wrote {mp4}")
        except Exception as exc:
            print(f"mp4 export failed ({exc}); is ffmpeg installed?")
    if show:
        plt.show()
    plt.close(fig)
    return ani


def main(argv=None) -> int:
    args = parse_args(argv)
    model, env = build(args)
    base = getattr(env, "unwrapped", env)
    # num_waypoints is 1 until the first reset draws a random set, so report the
    # count the task actually uses.
    n_task = int(base.random_waypoints or base.free_order_slots or base.num_waypoints)
    print(f"env: preset {base.preset_name}, {base.actuation_mode} actuation, {n_task} waypoints"
          f"{' (free order: the arm chooses the route)' if base.free_order else ''}, "
          f"tolerance {base.goal_tolerance} m, hold {base.hold_steps_required} steps")
    episodes = []
    for e in range(args.episodes):
        frames, info = run_episode(model, env, seed=args.seed + e)
        episodes.append(frames)
        print(f"episode {e + 1}: reached {info['waypoints_reached']}/{info['num_waypoints']} waypoints, "
              f"success={info['success']} at step {info['success_step']}, final distance {info['goal_distance']:.3f} m")
    title = f"{Path(args.model).name}  -  {base.actuation_mode} arm, {base.num_waypoints} waypoints"
    animate(episodes, env, title, fps=args.fps, gif=args.gif, mp4=args.mp4, show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
