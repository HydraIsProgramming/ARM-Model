"""
A demo window for the trained arm: one big picture and controls you can read.

    python -m rl_armMotion.two_d.gui.demo_app --package project_assets/outputs/CHAMPION_HUMAN_V1_R3
    python -m rl_armMotion.two_d.gui.demo_app --package <dir> --points "1.27,0.27;1.39,-0.01;1.21,-0.29"

The full controller (``gui.app``) carries everything needed to train and inspect
a model, in three columns of panes. That density is why its buttons kept being
clipped: on a short pane, rows below the fold are cut off with no way to reach
them. This window does one job instead - show a trained policy reaching points
you choose - so every control fits at once and can be read across a room.

Layout: the arm fills the left, all controls sit in a single right-hand column
grouped under plain headings, and a status strip runs along the bottom. Nothing
is nested, nothing is hidden behind a mode, and the window has a minimum size
below which it will not shrink controls out of view.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg    # noqa: E402

import tkinter as tk                                               # noqa: E402
from tkinter import filedialog, ttk                                # noqa: E402

from rl_armMotion.two_d.evaluation import (CANONICAL_WAYPOINTS, load_package,   # noqa: E402
                                           make_env, resolve_waypoints)
from rl_armMotion.two_d.gui.waypoint_editing import index_near, nearest_placeable  # noqa: E402

DEMO_POINTS = "1.270,0.270;1.390,-0.010;1.210,-0.290"

BG = "#f4f6f8"
INK = "#1d2733"
MUTED = "#6b7785"
GOOD = "#1e9e5a"
BUSY = "#d1495b"
PENDING = "#b9c2cc"
ARM = "#2b4c7e"


class DemoWindow:
    """One picture, one column of controls, one status strip."""

    def __init__(self, package: Optional[str] = None, points: Optional[str] = None,
                 which: str = "best", autostart: bool = False):
        self.package = package
        self.which = which
        self.autostart = autostart
        self.initial_points = points

        self.model = None
        self.env = None
        self.obs = None
        self.info = None
        self.running = False
        self.loop = True
        self.speed = 1.0
        self.episode_seed = 0
        self.episodes = 0
        self.successes = 0
        self.trail: List[np.ndarray] = []
        self.points: List[List[float]] = []
        self.requested: List[List[float]] = []      # where the click actually landed
        self.mode = "off"                            # off | add | remove
        self._step_credit = 0.0

    # ---------------------------------------------------------------- window

    def build(self):
        self.root = tk.Tk()
        self.root.title("Trained arm - reach any points you choose")
        self.root.configure(bg=BG)
        self.root.minsize(1120, 700)                 # below this, controls would crowd

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, foreground=INK)
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=INK, font=("Segoe UI", 10))
        style.configure("H.TLabel", font=("Segoe UI", 11, "bold"), foreground=INK)
        style.configure("Hint.TLabel", font=("Segoe UI", 9), foreground=MUTED)
        style.configure("Big.TLabel", font=("Segoe UI", 15, "bold"), foreground=INK)
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 7))
        style.configure("Go.TButton", font=("Segoe UI", 11, "bold"), padding=(10, 9))
        style.configure("Toolbutton", font=("Segoe UI", 10), padding=(10, 7))
        style.configure("TCheckbutton", background=BG, font=("Segoe UI", 10))
        style.configure("TRadiobutton", background=BG, font=("Segoe UI", 10))

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, minsize=330)
        outer.rowconfigure(0, weight=1)

        self._build_canvas(outer)
        self._build_controls(outer)
        self._build_status(outer)

        self.root.protocol("WM_DELETE_WINDOW", self._close)
        if self.package:
            self._load(self.package)
        if self.initial_points:
            self.points = [list(map(float, p)) for p in resolve_waypoints(self.initial_points)]
            self.requested = [list(p) for p in self.points]
            self._refresh_points_ui()
        if self.autostart and self.model is not None:
            self.root.after(400, self.start)
        self.root.after(50, self._tick)
        return self.root

    def _build_canvas(self, parent):
        holder = ttk.Frame(parent)
        holder.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.fig = plt.Figure(figsize=(7.4, 7.0), dpi=100, facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=holder)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._draw()

    def _build_controls(self, parent):
        col = ttk.Frame(parent)
        col.grid(row=0, column=1, sticky="nsew")

        # --- model -------------------------------------------------------
        ttk.Label(col, text="MODEL", style="H.TLabel").pack(anchor="w")
        self.model_label = ttk.Label(col, text="no model loaded", style="Hint.TLabel", wraplength=310)
        self.model_label.pack(anchor="w", pady=(2, 6))
        ttk.Button(col, text="Load a model...", command=self._choose).pack(fill="x")
        ttk.Separator(col).pack(fill="x", pady=12)

        # --- points ------------------------------------------------------
        ttk.Label(col, text="TARGET POINTS", style="H.TLabel").pack(anchor="w")
        self.set_var = tk.StringVar(value="demo")
        for text, value in (("Demo set (3 points)", "demo"), ("My own points", "mine")):
            ttk.Radiobutton(col, text=text, value=value, variable=self.set_var,
                            command=self._on_set_choice).pack(anchor="w", pady=1)

        self.mode_frame = ttk.Frame(col)
        self.mode_frame.pack(fill="x", pady=(8, 2))
        self.mode_var = tk.StringVar(value="off")
        self.add_btn = ttk.Radiobutton(self.mode_frame, text="+ Add point", value="add",
                                       variable=self.mode_var, style="Toolbutton",
                                       command=self._on_mode)
        self.add_btn.pack(side="left", expand=True, fill="x")
        self.del_btn = ttk.Radiobutton(self.mode_frame, text="- Remove point", value="remove",
                                       variable=self.mode_var, style="Toolbutton",
                                       command=self._on_mode)
        self.del_btn.pack(side="left", expand=True, fill="x", padx=(6, 0))

        edit_row = ttk.Frame(col)
        edit_row.pack(fill="x", pady=(6, 2))
        ttk.Button(edit_row, text="Undo", command=self._undo).pack(side="left", expand=True, fill="x")
        ttk.Button(edit_row, text="Clear", command=self._clear).pack(side="left", expand=True,
                                                                     fill="x", padx=(6, 0))
        self.points_hint = ttk.Label(col, text="", style="Hint.TLabel", wraplength=310)
        self.points_hint.pack(anchor="w", pady=(4, 0))
        ttk.Separator(col).pack(fill="x", pady=12)

        # --- run ---------------------------------------------------------
        ttk.Label(col, text="RUN", style="H.TLabel").pack(anchor="w")
        run_row = ttk.Frame(col)
        run_row.pack(fill="x", pady=(4, 2))
        self.start_btn = ttk.Button(run_row, text="Start", style="Go.TButton", command=self.start)
        self.start_btn.pack(side="left", expand=True, fill="x")
        self.stop_btn = ttk.Button(run_row, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=(6, 0))

        self.loop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col, text="Repeat automatically", variable=self.loop_var,
                        command=lambda: setattr(self, "loop", bool(self.loop_var.get()))
                        ).pack(anchor="w", pady=(6, 0))

        speed_row = ttk.Frame(col)
        speed_row.pack(fill="x", pady=(8, 0))
        ttk.Label(speed_row, text="Speed").pack(side="left")
        self.speed_label = ttk.Label(speed_row, text="1.0x", style="Hint.TLabel")
        self.speed_label.pack(side="right")
        self.speed_scale = ttk.Scale(col, from_=0.25, to=2.0, value=1.0, command=self._on_speed)
        self.speed_scale.pack(fill="x")
        ttk.Separator(col).pack(fill="x", pady=12)

        # --- live numbers -------------------------------------------------
        ttk.Label(col, text="THIS RUN", style="H.TLabel").pack(anchor="w")
        self.reached_label = ttk.Label(col, text="Points reached:  -", style="Big.TLabel")
        self.reached_label.pack(anchor="w", pady=(4, 2))
        hold_row = ttk.Frame(col)
        hold_row.pack(fill="x", pady=(2, 2))
        ttk.Label(hold_row, text="Holding").pack(side="left")
        self.hold_pct = ttk.Label(hold_row, text="0%", style="Hint.TLabel")
        self.hold_pct.pack(side="right")
        self.hold_bar = ttk.Progressbar(col, maximum=100)
        self.hold_bar.pack(fill="x")
        self.speed_now = ttk.Label(col, text="Hand speed:  -", style="Hint.TLabel")
        self.speed_now.pack(anchor="w", pady=(6, 0))
        self.tally = ttk.Label(col, text="Completed:  0 of 0 runs", style="Hint.TLabel")
        self.tally.pack(anchor="w")

    def _build_status(self, parent):
        bar = ttk.Frame(parent, padding=(2, 8, 2, 0))
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.status = ttk.Label(bar, text="Load a model, choose points, press Start.",
                                style="Hint.TLabel")
        self.status.pack(side="left")

    def _say(self, text):
        self.status.config(text=text)
        print(f"[demo] {text}")

    # ----------------------------------------------------------------- model

    def _choose(self):
        path = filedialog.askdirectory(title="Choose a model package folder")
        if path:
            self._load(path)

    def _load(self, package):
        try:
            self.model, self.env, manifest = load_package(package, which=self.which,
                                                          terminate_on_success=True)
        except Exception as exc:                      # a clear message beats a traceback
            self._say(f"Could not load that model: {exc}")
            return
        base = getattr(self.env, "unwrapped", self.env)
        self.random_waypoints = base.random_waypoints
        base.random_waypoints = None                  # a chosen set must not be resampled
        self.model_label.config(text=f"{Path(package).name}   ({base.actuation_mode}, "
                                     f"hold {base.hold_steps_required} steps at "
                                     f"{base.goal_tolerance * 100:.0f} cm)")
        self._say(f"Loaded {Path(package).name}.")
        self._draw()

    # ---------------------------------------------------------------- points

    @property
    def base(self):
        return getattr(self.env, "unwrapped", self.env) if self.env is not None else None

    def _feasibility_env(self):
        return self.base if self.base is not None else make_env(actuation_mode="muscle", waypoints=None)

    def _on_set_choice(self):
        if self.set_var.get() == "demo":
            self.mode_var.set("off")
            self._on_mode()
            self.points = [list(map(float, p)) for p in resolve_waypoints(DEMO_POINTS)]
            self.requested = [list(p) for p in self.points]
        else:
            self.points, self.requested = [], []
            self.mode_var.set("add")
            self._on_mode()
        self._refresh_points_ui()

    def _on_mode(self):
        self.mode = self.mode_var.get()
        if self.mode in ("add", "remove"):
            self.set_var.set("mine")
            if self.running:
                self.stop()
        self._refresh_points_ui()

    def _refresh_points_ui(self):
        n = len(self.points)
        if self.mode == "add":
            hint = f"Click the picture to drop a point.  ({n} placed)"
        elif self.mode == "remove":
            hint = f"Click a point to delete it.  ({n} placed)" if n else "No points to remove."
        elif n:
            hint = f"{n} point(s) ready.  Press Start."
        else:
            hint = "Choose 'My own points', then press '+ Add point'."
        self.points_hint.config(text=hint)
        self._draw()

    def _on_click(self, event):
        if self.mode == "off" or event.inaxes is not self.ax or event.xdata is None:
            return
        env = self._feasibility_env()
        wp = [float(event.xdata), float(event.ydata)]
        if self.mode == "remove" or getattr(event, "button", 1) in (2, 3):
            i = index_near(self.points, wp)
            if i is None:
                self._say("Click closer to one of your points to remove it.")
                return
            gone = self.points.pop(i)
            if i < len(self.requested):
                self.requested.pop(i)
            self._say(f"Removed the point at ({gone[0]:.2f}, {gone[1]:.2f}).")
            self._refresh_points_ui()
            return

        ok = env.waypoint_feasibility(wp, env.max_curriculum_tolerance).get("feasible", False)
        if ok:
            try:
                env._validate_waypoint_spacing([np.asarray(w, float) for w in self.points + [wp]])
            except ValueError:
                ok = False
        asked = list(wp)
        if not ok:
            snapped, moved = nearest_placeable(env, wp, self.points)
            if snapped is None:
                self._say("The arm cannot hold anywhere near there - try inside the green area.")
                return
            self._say(f"Moved your point {moved * 100:.0f} cm to the nearest spot the arm can hold.")
            wp = snapped
        else:
            self._say(f"Point {len(self.points) + 1} placed.")
        self.points.append(list(wp))
        self.requested.append(asked)
        self.set_var.set("mine")
        self._refresh_points_ui()

    def _undo(self):
        if not self.points:
            self._say("Nothing to undo.")
            return
        self.points.pop()
        if self.requested:
            self.requested.pop()
        self._say("Removed the last point.")
        self._refresh_points_ui()

    def _clear(self):
        self.points, self.requested = [], []
        self._say("Cleared all points.")
        self._refresh_points_ui()

    # ------------------------------------------------------------------- run

    def _on_speed(self, value):
        self.speed = float(value)
        self.speed_label.config(text=f"{self.speed:.2f}x")

    def start(self):
        if self.model is None:
            self._say("Load a model first.")
            return
        pts = self.points or [list(map(float, p)) for p in resolve_waypoints(DEMO_POINTS)]
        if len(pts) < 1:
            self._say("Place at least one point first.")
            return
        try:
            self.base.set_waypoints(pts)
        except ValueError as exc:
            self._say(f"Those points will not work: {exc}")
            return
        self.points = [list(map(float, p)) for p in pts]
        self.obs, self.info = self.base.reset(seed=self.episode_seed)
        self.trail = [np.asarray(self.info["end_effector_position"], float)]
        self.running = True
        self.mode_var.set("off")
        self.mode = "off"
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._say(f"Running {len(pts)} point(s).")

    def stop(self):
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _tick(self):
        if self.running and self.model is not None:
            self._step_credit += self.speed
            for _ in range(int(self._step_credit)):
                self._step_credit -= 1
                action, _ = self.model.predict(self.obs, deterministic=True)
                self.obs, _, terminated, truncated, self.info = self.base.step(action)
                self.trail.append(np.asarray(self.info["end_effector_position"], float))
                if terminated or truncated:
                    self.episodes += 1
                    self.successes += int(bool(self.info["success"]))
                    ok = "reached every point" if self.info["success"] else "did not finish"
                    self._say(f"Run {self.episodes}: {ok} "
                              f"({self.info['waypoints_reached']}/{self.info['num_waypoints']}).")
                    if self.loop:
                        self.episode_seed += 1
                        self.obs, self.info = self.base.reset(seed=self.episode_seed)
                        self.trail = [np.asarray(self.info["end_effector_position"], float)]
                    else:
                        self.stop()
                    break
            self._draw()
            self._refresh_numbers()
        elif self.mode in ("add", "remove"):
            self._draw()
        self.root.after(50, self._tick)

    def _refresh_numbers(self):
        if not self.info:
            return
        n = int(self.info["num_waypoints"])
        self.reached_label.config(text=f"Points reached:  {self.info['waypoints_reached']} of {n}")
        pct = int(round(100 * float(self.info["hold_progress"])))
        self.hold_bar["value"] = pct
        self.hold_pct.config(text=f"{pct}%")
        self.speed_now.config(text=f"Hand speed:  {float(self.info['ee_speed']):.2f} m/s")
        self.tally.config(text=f"Completed:  {self.successes} of {self.episodes} runs")

    # ---------------------------------------------------------------- drawing

    def _draw(self):
        ax = self.ax
        ax.clear()
        env = self.base
        reach = float(env.reach) if env is not None else 0.65
        shoulder = np.asarray(env.shoulder_base_position, float) if env is not None else np.array([1.0, 0.0])

        ax.set_xlim(shoulder[0] - 1.15 * reach, shoulder[0] + 1.15 * reach)
        ax.set_ylim(shoulder[1] - 1.15 * reach, shoulder[1] + 1.15 * reach)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlabel("metres in front of the body", color=MUTED, fontsize=10)

        # body, so the picture reads as an arm
        torso_w, torso_h = 0.22 * reach, 0.95 * reach
        ax.add_patch(plt.Rectangle((shoulder[0] - torso_w, shoulder[1] - torso_h), torso_w, torso_h,
                                   color="#00000010", zorder=0))
        ax.add_patch(plt.Circle((shoulder[0] - 0.5 * torso_w, shoulder[1] + 0.34 * reach),
                                0.17 * reach, color="#00000010", zorder=0))
        ax.add_patch(plt.Circle(tuple(shoulder), reach, fill=False, linestyle=":",
                                color="#00000022", linewidth=1.0))

        if env is not None and self.mode in ("add", "remove"):
            try:
                cloud = env.feasible_points()
                ax.scatter(cloud[:, 0], cloud[:, 1], s=7, c=GOOD, alpha=0.16, linewidths=0, zorder=1)
            except Exception:
                pass

        tol = float(env.goal_tolerance) if env is not None else 0.05
        visited = list((self.info or {}).get("visited") or [])
        active = int((self.info or {}).get("current_waypoint_index", -1)) if self.running else -1
        done_count = int((self.info or {}).get("waypoints_reached", 0)) if self.running else 0
        for i, wp in enumerate(self.points):
            p = np.asarray(wp, float)
            if visited:
                done = bool(visited[i]) if i < len(visited) else False
            else:
                done = self.running and i < done_count
            colour = GOOD if done else (BUSY if i == active else PENDING)
            ax.add_patch(plt.Circle(tuple(p), tol, color=colour, alpha=0.18, zorder=2))
            ax.add_patch(plt.Circle(tuple(p), tol, color=colour, fill=False, linewidth=1.6, zorder=2))
            ax.plot([p[0]], [p[1]], "X", color=colour, markersize=13, markeredgecolor="white",
                    markeredgewidth=1.4, zorder=6)
            # show where the click actually was, when it had to be moved
            if i < len(self.requested):
                asked = np.asarray(self.requested[i], float)
                if float(np.linalg.norm(asked - p)) > 1e-6:
                    ax.plot([asked[0], p[0]], [asked[1], p[1]], ":", color=MUTED, linewidth=1.0, zorder=3)
                    ax.plot([asked[0]], [asked[1]], "o", markerfacecolor="none",
                            markeredgecolor=MUTED, markersize=8, zorder=3)

        if self.info:
            if len(self.trail) > 1:
                t = np.asarray(self.trail)
                ax.plot(t[:, 0], t[:, 1], "-", color="#00000030", linewidth=1.4, zorder=3)
            links = np.asarray(self.info["link_positions"], float)
            ax.plot(links[:, 0], links[:, 1], "-o", color=ARM, linewidth=6, markersize=9,
                    solid_capstyle="round", zorder=7)
            hand = np.asarray(self.info["end_effector_position"], float)
            ax.plot([hand[0]], [hand[1]], "o", color="white", markersize=11,
                    markeredgecolor=ARM, markeredgewidth=3, zorder=8)

        if self.running and self.info:
            title = (f"{self.info['waypoints_reached']} of {self.info['num_waypoints']} reached"
                     + ("   -   HOLDING" if self.info.get("in_goal_region") else ""))
            if self.info.get("success"):
                title = "All points reached and held"
        elif self.points:
            title = f"{len(self.points)} point(s) ready - press Start"
        else:
            title = "Choose points, then press Start"
        ax.set_title(title, color=INK, fontsize=14, fontweight="bold", pad=12)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _close(self):
        self.running = False
        self.root.destroy()


def main(argv=None):
    p = argparse.ArgumentParser(description="Demo window for a trained arm")
    p.add_argument("--package", default=None, help="model package directory")
    p.add_argument("--which", default="best", help="best | last | final")
    p.add_argument("--points", default=None, help='"x,y;x,y;x,y" (default: the demo set)')
    p.add_argument("--no-autostart", action="store_true")
    args = p.parse_args(argv)

    win = DemoWindow(package=args.package, points=args.points or DEMO_POINTS,
                     which=args.which, autostart=not args.no_autostart)
    root = win.build()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
