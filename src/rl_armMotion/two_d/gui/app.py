"""Interactive arm controller GUI (Tkinter + Matplotlib).

One window, three columns:

  controls   a scrollable column of numbered sections - 1 model, 2 target
             points, 3 manual joint control, 4 arm properties - so no control
             can be pushed out of view on a short screen;
  arm view   the arm, its targets and the run's progress, with the live
             torque / hand-velocity / hand-acceleration plots underneath;
  readouts   live status, the four muscles' activation, every reading, and
             the loaded model's details.

A header says what the window is doing (idle, running a model, recording,
playing back) and a status line along the bottom shows the last message.

    python -m rl_armMotion.two_d.gui.app [--package DIR|MODEL.zip] [--which best]
                                         [--waypoints 1|2|3|spread|'x,y;x,y'] [--autostart] [--no-loop]
"""

import json
import os
import pickle
import sys
from collections import deque
from pathlib import Path

# Ensure src directory is in path for imports.
# File lives in `.../src/rl_armMotion/two_d/gui/app.py`, so parents[3] is `src/`.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from time import time

from rl_armMotion.two_d.config import ArmConfiguration
from rl_armMotion.two_d.environments.task_env import ArmTaskEnv, ActionSmoother, DEFAULT_PRESET, OBS_VERSION
from rl_armMotion.two_d.evaluation import (
    CANONICAL_WAYPOINTS, EVAL_STANDARD, WAYPOINTS_SPREAD, detect_actuation_mode, env_from_manifest, find_manifest,
    load_package, make_env, read_manifest, resolve_waypoints,
)
from rl_armMotion.two_d.models.trainers import RLTrainer
from rl_armMotion.two_d.utils import ArmKinematics, ArmController, MotionRecorder, ArmVisualizer, dynamics

PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUTS_DIR = PROJECT_ROOT / "project_assets" / "outputs"

# ----------------------------------------------------------------- palette
BG = "#eef0f3"          # window background, between cards
CARD = "#ffffff"
BORDER = "#d9dde3"
INK = "#1b1d21"
MUTED = "#5d6470"
ACCENT = "#2a78d6"
GOOD = "#1f9d55"
WARN = "#c77700"
BAD = "#d64545"
TRACK = "#e6e9ee"
SHOULDER_C = "#2a78d6"
ELBOW_C = "#e8743b"
ARM_C = "#262a31"

# What each trained package is, for the quick-pick list. Anything not listed
# is described from its own manifest.
KNOWN_PACKAGES = {
    "R7": "any 3 points, full human range",
    "R3": "reaches any 3 points (deployed)",
    "R4": "any 3 points, end-range rule",
    "R5": "any 3 points, human speed cost",
    "R6": "any 3 points, binding speed cap",
    "F2": "2 points, picks its own order",
    "B3": "3 fixed points only",
    "B": "3 fixed points only (v1)",
    "A": "2 fixed points only",
}
PACKAGE_ORDER = ["R7", "R3", "R4", "R5", "R6", "F2", "B3", "B", "A"]

WAYPOINT_CHOICES = [                 # (internal value, label on screen)
    ("from package", "Model's own points"),
    ("1 waypoint", "1 demo point"),
    ("2 waypoints", "2 demo points"),
    ("3 waypoints", "3 demo points"),
    ("spread", "Spread-out set"),
    ("custom", "My own points"),
]

# Order of info["muscle_activations"]: (positive, negative) muscle per joint.
MUSCLE_NAMES = [
    ("Shoulder flexors", "front deltoid, pectoralis", SHOULDER_C),
    ("Shoulder extensors", "back deltoid, latissimus", "#7fb0ea"),
    ("Elbow flexors", "biceps, brachialis", ELBOW_C),
    ("Elbow extensors", "triceps", "#f2ae8a"),
]


class ArmControllerGUI:
    """Main GUI application for interactive arm control using Tkinter"""

    def __init__(self, config: ArmConfiguration = None):
        """
        Initialize GUI application.

        Args:
            config: ArmConfiguration object (uses default if None)
        """
        self.config = config or ArmConfiguration.get_preset(DEFAULT_PRESET)
        self.controller = ArmController(self.config)
        self.recorder = MotionRecorder()
        self.visualizer = ArmVisualizer(dof=self.config.dof, link_lengths=self.config.link_lengths)

        # GUI state
        self.running = True
        self.recording = False
        self.playing_back = False
        self.show_trajectory = False
        self.trajectory_points = []
        self.playback_frames = []
        self.playback_index = 0
        self.last_frame_time = time()
        self.frame_count = 0
        self.fps_counter = 0
        self.selected_joint = 0  # Track selected joint for keyboard control

        # Policy simulation state
        self.simulation_active = False
        self.sim_paused = False
        self.sim_env = None
        self.sim_obs = None
        self.sim_model = None
        self.sim_algorithm = None
        self.sim_model_path = ""
        self.sim_model_kind = "-"
        self.sim_step_count = 0
        self.sim_episode_count = 0
        self.sim_episode_reward = 0.0
        self.sim_speed_multiplier = 1.0
        self.sim_steps_per_frame = max(1, int(round(0.05 / float(self.config.dt))))  # 50 ms frames = real time
        self.sim_step_accumulator = 0.0
        # Waypoint demo state
        self.sim_waypoint_choice = "from package"
        self.sim_custom_waypoints = ""
        self.sim_loop = True
        self.sim_last_info = None
        self.sim_successes = 0
        self.sim_episode_seed = 0
        self.sim_package_dir = None
        self.sim_manifest = None
        self.sim_hold_bar = None
        self.sim_progress_label = None
        self.sim_success_banner_until = 0.0
        self.sim_toggle_button = None
        self.sim_model_label = None
        self.sim_model_details_text = None
        self.sim_time = 0.0
        self.action_smoothing = False   # models with muscle activation dynamics must NOT be smoothed again
        self.clicked_waypoints = []
        self.clicked_requests = []
        self._place_mode = False
        self._feasible_cloud = None
        self._feasible_cloud_key = None
        self.action_smooth_alpha = 0.3
        self.prev_sim_velocities = np.zeros(self.config.dof, dtype=float)
        self.latest_shoulder_torque = 0.0
        self.latest_elbow_torque = 0.0
        self._initial_model = None
        self._pending_autostart = False
        self._last_slow_update = 0.0
        self._last_step_wall = None

        # Torque history for live plotting: the muscles' torque and the torque
        # gravity demands at each joint, both straight from the simulator.
        self.torque_time_hist = deque(maxlen=2000)
        self.shoulder_torque_hist = deque(maxlen=2000)
        self.elbow_torque_hist = deque(maxlen=2000)
        self.shoulder_gravity_hist = deque(maxlen=2000)
        self.elbow_gravity_hist = deque(maxlen=2000)

        # Hand velocity/acceleration history, sampled every simulated step on
        # simulated time (not wall-clock), so it is right at any playback speed.
        self.ee_kin_time_hist = deque(maxlen=2000)
        self.ee_vx_hist = deque(maxlen=2000)
        self.ee_vy_hist = deque(maxlen=2000)
        self.ee_speed_hist = deque(maxlen=2000)
        self.ee_ax_hist = deque(maxlen=2000)
        self.ee_ay_hist = deque(maxlen=2000)
        self._ee_prev_vel_xy = None
        self._hand_trail = deque(maxlen=250)

        # Create main window
        self.root = tk.Tk()
        self.root.title("Human Arm RL - Interactive Controller")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.configure(bg=BG)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = min(1760, int(sw * 0.92)), min(1040, int(sh * 0.88))
        self.root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        self.root.minsize(1240, 740)

        # Matplotlib figure
        self.fig = None
        self.ax = None
        self.canvas = None
        self.canvas_agg = None

        # GUI elements (for updates)
        self.slider_vars = {}
        self.angle_labels = {}
        self.metrics_text = None
        self.torque_fig = None
        self.torque_ax = None
        self.torque_canvas = None
        self.velacc_fig = None
        self.velacc_ax_vel = None
        self.velacc_ax_acc = None
        self.velacc_canvas = None
        self.record_button = None
        self.playback_button = None
        self._tiles = {}
        self._muscle_rows = []
        self._package_choices = {}
        self._scroll_columns = []

    # ================================================================ helpers

    @staticmethod
    def _normalize_model_base(filepath: str) -> str:
        """Normalize selected model path to Stable-Baselines3 base path."""
        path = Path(filepath)
        if path.suffix == ".zip":
            return str(path.with_suffix(""))
        return str(path)

    def _get_algorithm_from_metadata(self, model_base_path: str):
        """Infer algorithm from metadata when available."""
        metadata_path = Path(f"{model_base_path}_metadata.pkl")
        if metadata_path.exists():
            try:
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                algorithm = metadata.get("algorithm")
                if isinstance(algorithm, str):
                    return algorithm.upper()
            except Exception:
                pass

        name = Path(model_base_path).name.lower()
        for algo in ("ppo", "sac", "a2c", "dqn"):
            if algo in name:
                return algo.upper()
        return None

    def _read_model_metadata(self, model_base_path: str):
        """Load optional metadata saved alongside the model."""
        metadata_path = Path(f"{model_base_path}_metadata.pkl")
        if not metadata_path.exists():
            return {}
        try:
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _log(self, message: str):
        """Print a message and show it on the status line."""
        print(message)
        bar = getattr(self, "_status_label", None)
        if bar is None:
            return
        text = message.replace("[OK] ", "").replace("[WARN] ", "")
        colour = WARN if message.startswith("[WARN]") else (BAD if message.startswith("[ERR") else INK)
        try:
            bar.config(text=text, fg=colour)
        except tk.TclError:
            pass

    def _set_model_details_text(self, text: str):
        """Update the model-details text box."""
        if self.sim_model_details_text is None:
            return
        self.sim_model_details_text.config(state="normal")
        self.sim_model_details_text.delete(1.0, tk.END)
        self.sim_model_details_text.insert(1.0, text)
        self.sim_model_details_text.config(state="disabled")

    def _refresh_joint_labels(self):
        for i in range(self.config.dof):
            label = self.angle_labels.get(i)
            if label is not None:
                a = float(self.controller.angles[i])
                label.config(text=f"{a:+.2f} rad  ({np.degrees(a):+.0f}°)")

    def _set_controller_state_from_arrays(self, angles, velocities):
        """Apply angle/velocity arrays to controller and refresh labels."""
        self.controller.angles = np.asarray(angles, dtype=float).copy()
        self.controller.velocities = np.asarray(velocities, dtype=float).copy()
        self._compute_positions()
        self._refresh_joint_labels()

    def _estimate_joint_torques(self, angles, velocities, accelerations):
        """Joint torques from the arm's rigid-body inverse dynamics
        (M(q) q'' + C(q, q') q' + G(q)), i.e. what the actuators must supply.
        Used only when the simulator does not report its own torque."""
        q = np.asarray(angles, dtype=float)
        qd = np.asarray(velocities, dtype=float)
        qdd = np.asarray(accelerations, dtype=float)
        try:
            tau = dynamics.inverse_dynamics(q, qd, qdd, self.config)
        except Exception:
            tau = np.zeros(max(q.shape[0], 2))
        return float(tau[0]), float(tau[1]) if tau.shape[0] > 1 else 0.0

    def _append_torque_sample(self, shoulder_torque, elbow_torque, current_time,
                              shoulder_gravity=np.nan, elbow_gravity=np.nan):
        """Append a new torque sample to history buffers."""
        self.torque_time_hist.append(float(current_time))
        self.shoulder_torque_hist.append(float(shoulder_torque))
        self.elbow_torque_hist.append(float(elbow_torque))
        self.shoulder_gravity_hist.append(float(shoulder_gravity))
        self.elbow_gravity_hist.append(float(elbow_gravity))
        self.latest_shoulder_torque = float(shoulder_torque)
        self.latest_elbow_torque = float(elbow_torque)

    def _clear_torque_history(self):
        for d in (self.torque_time_hist, self.shoulder_torque_hist, self.elbow_torque_hist,
                  self.shoulder_gravity_hist, self.elbow_gravity_hist):
            d.clear()
        self.latest_shoulder_torque = 0.0
        self.latest_elbow_torque = 0.0

    def _reset_ee_kinematics(self):
        """Reset hand velocity/acceleration histories."""
        for d in (self.ee_kin_time_hist, self.ee_vx_hist, self.ee_vy_hist, self.ee_speed_hist,
                  self.ee_ax_hist, self.ee_ay_hist):
            d.clear()
        self._ee_prev_vel_xy = None

    def _append_hand_sample(self, info, dt):
        """One hand velocity/acceleration sample per simulated step, from the
        simulator's own hand velocity, on simulated time."""
        v = np.asarray(info.get("ee_velocity", (0.0, 0.0)), dtype=float)[:2]
        if self._ee_prev_vel_xy is None:
            a = np.zeros(2)
        else:
            a = (v - self._ee_prev_vel_xy) / max(dt, 1e-8)
        self._ee_prev_vel_xy = v.copy()
        self.ee_kin_time_hist.append(float(self.sim_time))
        self.ee_vx_hist.append(float(v[0]))
        self.ee_vy_hist.append(float(v[1]))
        self.ee_speed_hist.append(float(np.linalg.norm(v)))
        self.ee_ax_hist.append(float(a[0]))
        self.ee_ay_hist.append(float(a[1]))

    # ============================================================== styling

    def _setup_style(self):
        families = set(tkfont.families(self.root))
        ui = "Segoe UI" if "Segoe UI" in families else "Arial"
        mono = "Consolas" if "Consolas" in families else "Courier"
        self.f_body = (ui, 10)
        self.f_small = (ui, 9)
        self.f_bold = (ui, 10, "bold")
        self.f_head = (ui, 11, "bold")
        self.f_title = (ui, 15, "bold")
        self.f_value = (ui, 14, "bold")
        self.f_big = (ui, 12, "bold")
        self.f_mono = (mono, 9)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=CARD, foreground=INK, font=self.f_body)
        style.configure("TFrame", background=CARD)
        style.configure("Bg.TFrame", background=BG)
        style.configure("TLabel", background=CARD, foreground=INK)
        style.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=self.f_small)
        style.configure("TCheckbutton", background=CARD)
        style.configure("TRadiobutton", background=CARD)
        style.map("TCheckbutton", background=[("active", CARD)])
        style.map("TRadiobutton", background=[("active", CARD)])
        style.configure("TButton", padding=(10, 5), background="#f3f4f6", bordercolor=BORDER)
        style.map("TButton", background=[("active", "#e4e7eb"), ("disabled", "#f5f6f7")])
        style.configure("Accent.TButton", background=ACCENT, foreground="white", bordercolor=ACCENT,
                        font=self.f_bold, padding=(12, 7))
        style.map("Accent.TButton", background=[("active", "#1f64b8"), ("disabled", "#9dbbe3")],
                  foreground=[("disabled", "#eef3fb")])
        style.configure("Stop.TButton", background=BAD, foreground="white", bordercolor=BAD,
                        font=self.f_bold, padding=(12, 7))
        style.map("Stop.TButton", background=[("active", "#b83434")])
        style.configure("Toggle.TButton", padding=(10, 6))
        style.configure("On.Toggle.TButton", background="#e2f4ea", bordercolor=GOOD, foreground=GOOD,
                        font=self.f_bold)
        style.map("On.Toggle.TButton", background=[("active", "#d2eedd")])
        style.configure("Hold.Horizontal.TProgressbar", troughcolor=TRACK, background=GOOD,
                        bordercolor=TRACK, lightcolor=GOOD, darkcolor=GOOD, thickness=14)
        style.configure("TCombobox", fieldbackground="white", padding=4)
        style.configure("TEntry", fieldbackground="white", padding=4)
        style.configure("Horizontal.TScale", background=CARD, troughcolor=TRACK)

    @staticmethod
    def _autowrap(label, pad=6):
        """Wrap a label to the width it is given, so narrow columns never cut text off."""
        label.bind("<Configure>", lambda e: label.config(wraplength=max(60, e.width - pad)))
        return label

    def _card(self, parent, title, number=None, note=None):
        """A white panel with a heading; returns (outer frame, body frame)."""
        outer = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        head = tk.Frame(outer, bg=CARD)
        head.pack(fill="x", padx=12, pady=(10, 2))
        if number is not None:
            tk.Label(head, text=str(number), bg=ACCENT, fg="white", font=self.f_bold, width=2).pack(
                side="left", padx=(0, 8))
        tk.Label(head, text=title, bg=CARD, fg=INK, font=self.f_head).pack(side="left")
        if note:
            self._autowrap(tk.Label(outer, text=note, bg=CARD, fg=MUTED, font=self.f_small, justify="left",
                                    anchor="w", wraplength=330)).pack(fill="x", padx=12, pady=(0, 4))
        body = tk.Frame(outer, bg=CARD)
        body.pack(fill="both", expand=True, padx=12, pady=(4, 12))
        return outer, body

    def _slider_row(self, parent, key, label, lo, hi, value, on_change, on_entry=None, unit=""):
        """label | slider | value | entry, stored under slider_vars[key...]."""
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=2)
        row.columnconfigure(1, weight=1)
        tk.Label(row, text=label, bg=CARD, fg=INK, font=self.f_body, width=15, anchor="w").grid(
            row=0, column=0, sticky="w")
        var = tk.DoubleVar(value=value)
        self.slider_vars[key] = var
        ttk.Scale(row, from_=lo, to=hi, variable=var, orient="horizontal", command=on_change).grid(
            row=0, column=1, sticky="ew", padx=6)
        val_label = tk.Label(row, text=f"{value:.2f}{unit}", bg=CARD, fg=MUTED, font=self.f_small, width=7,
                             anchor="e")
        val_label.grid(row=0, column=2, sticky="e")
        self.slider_vars[f"{key}_label"] = val_label
        if on_entry is not None:
            entry_var = tk.StringVar(value=f"{value:.2f}")
            self.slider_vars[f"{key}_entry"] = entry_var
            entry = ttk.Entry(row, textvariable=entry_var, width=6)
            entry.grid(row=0, column=3, sticky="e", padx=(4, 0))
            entry.bind("<Return>", lambda _e: on_entry())
            entry.bind("<FocusOut>", lambda _e: on_entry())
        return row

    # ======================================================= control column

    def _create_model_selection_frame(self, parent):
        """Section 1: choose a trained model and run it."""
        frame, body = self._card(parent, "Model", number=1)

        # Quick pick of the trained packages in project_assets/outputs, plus Browse.
        pick = tk.Frame(body, bg=CARD)
        pick.pack(fill="x")
        pick.columnconfigure(0, weight=1)
        self._package_choices = self._discover_packages()
        self._package_var = tk.StringVar(value="")
        self._package_box = ttk.Combobox(pick, textvariable=self._package_var, state="readonly",
                                         values=list(self._package_choices.keys()))
        self._package_box.grid(row=0, column=0, sticky="ew")
        self._package_box.bind("<<ComboboxSelected>>", lambda _e: self._on_pick_package())
        ttk.Button(pick, text="Browse...", command=self._on_select_model).grid(row=0, column=1, padx=(6, 0))

        self.sim_model_label = tk.Label(body, text="Model: not selected", bg=CARD, fg=INK, font=self.f_bold,
                                        anchor="w", justify="left", wraplength=330)
        self._autowrap(self.sim_model_label).pack(fill="x", pady=(8, 0))
        self._model_kind_label = tk.Label(body, text="Pick a trained model above, or Browse for a .zip.",
                                          bg=CARD, fg=MUTED, font=self.f_small, anchor="w", justify="left",
                                          wraplength=330)
        self._autowrap(self._model_kind_label).pack(fill="x")

        run_row = tk.Frame(body, bg=CARD)
        run_row.pack(fill="x", pady=(10, 4))
        run_row.columnconfigure(0, weight=1, uniform="run")
        run_row.columnconfigure(1, weight=1, uniform="run")
        self.sim_toggle_button = ttk.Button(run_row, text="▶  Run simulation", style="Accent.TButton",
                                            command=self._on_toggle_simulation, state="disabled")
        self.sim_toggle_button.grid(row=0, column=0, columnspan=2, sticky="ew")
        self._pause_button = ttk.Button(run_row, text="Pause", command=self._on_pause, state="disabled")
        self._pause_button.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._next_button = ttk.Button(run_row, text="New episode", command=self._on_next_episode,
                                       state="disabled")
        self._next_button.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(6, 0))

        # Simulation speed: slider 1..20, 5 = real time.
        speed_row = tk.Frame(body, bg=CARD)
        speed_row.pack(fill="x", pady=(6, 2))
        speed_row.columnconfigure(1, weight=1)
        tk.Label(speed_row, text="Speed", bg=CARD, fg=INK, font=self.f_body, width=8, anchor="w").grid(
            row=0, column=0, sticky="w")
        self._sim_speed_var = tk.IntVar(value=5)
        self._sim_speed_label = tk.Label(speed_row, text="1.0x real time", bg=CARD, fg=MUTED,
                                         font=self.f_small, width=13, anchor="e")

        def _on_speed_change(_value=None):
            steps = int(round(float(self._sim_speed_var.get())))
            self.sim_speed_multiplier = max(1, steps) / 5.0  # slider 5 = real time
            self._sim_speed_label.config(text=f"{self.sim_speed_multiplier:.1f}x real time")

        ttk.Scale(speed_row, from_=1, to=20, orient="horizontal", variable=self._sim_speed_var,
                  command=_on_speed_change).grid(row=0, column=1, sticky="ew", padx=6)
        self._sim_speed_var.trace_add("write", lambda *_: _on_speed_change())
        self._sim_speed_label.grid(row=0, column=2, sticky="e")

        opts = tk.Frame(body, bg=CARD)
        opts.pack(fill="x", pady=(4, 0))
        self._loop_var = tk.BooleanVar(value=self.sim_loop)
        ttk.Checkbutton(opts, text="Loop episodes", variable=self._loop_var,
                        command=lambda: setattr(self, "sim_loop", bool(self._loop_var.get()))).pack(
            side="left")
        self._smooth_var = tk.BooleanVar(value=self.action_smoothing)
        ttk.Checkbutton(opts, text="Smooth motion", variable=self._smooth_var,
                        command=self._on_smooth_toggle).pack(side="left", padx=(14, 0))

        alpha_row = tk.Frame(body, bg=CARD)
        alpha_row.pack(fill="x", pady=(2, 0))
        alpha_row.columnconfigure(1, weight=1)
        tk.Label(alpha_row, text="Smoothing α", bg=CARD, fg=MUTED, font=self.f_small, width=11,
                 anchor="w").grid(row=0, column=0, sticky="w")
        self._smooth_alpha_var = tk.DoubleVar(value=self.action_smooth_alpha)
        self._smooth_alpha_label = tk.Label(alpha_row, text=f"{self.action_smooth_alpha:.2f}", bg=CARD,
                                            fg=MUTED, font=self.f_small, width=5, anchor="e")

        def _on_alpha_change(_value=None):
            val = round(float(self._smooth_alpha_var.get()), 2)
            self.action_smooth_alpha = val
            self._smooth_alpha_label.config(text=f"{val:.2f}")

        ttk.Scale(alpha_row, from_=0.1, to=0.9, orient="horizontal", variable=self._smooth_alpha_var,
                  command=_on_alpha_change).grid(row=0, column=1, sticky="ew", padx=6)
        self._smooth_alpha_var.trace_add("write", lambda *_: _on_alpha_change())
        self._smooth_alpha_label.grid(row=0, column=2, sticky="e")
        return frame

    def _create_targets_frame(self, parent):
        """Section 2: which points the arm is asked to reach."""
        frame, body = self._card(parent, "Target points", number=2)

        self._wp_choice_var = tk.StringVar(value=self.sim_waypoint_choice)
        grid = tk.Frame(body, bg=CARD)
        grid.pack(fill="x")
        for i, (value, label) in enumerate(WAYPOINT_CHOICES):
            ttk.Radiobutton(grid, text=label, value=value, variable=self._wp_choice_var,
                            command=self._on_waypoint_choice).grid(row=i // 2, column=i % 2, sticky="w",
                                                                   padx=(0, 12), pady=1)

        tk.Label(body, text="My own points  (x, y in metres; one pair per point)", bg=CARD, fg=MUTED,
                 font=self.f_small, anchor="w").pack(fill="x", pady=(8, 0))
        self._wp_custom_var = tk.StringVar(value=self.sim_custom_waypoints)
        custom_entry = ttk.Entry(body, textvariable=self._wp_custom_var, font=self.f_mono)
        custom_entry.pack(fill="x")
        custom_entry.bind("<KeyRelease>", lambda e: self._on_custom_typed())
        custom_entry.bind("<Return>", lambda e: self._on_custom_committed())
        custom_entry.bind("<FocusOut>", lambda e: self._on_custom_committed())

        self._place_var = tk.BooleanVar(value=False)
        self._place_button = ttk.Button(body, text="✚  Click the arm view to add points: OFF",
                                        style="Toggle.TButton", command=self._on_place_button)
        self._place_button.pack(fill="x", pady=(8, 4))

        edit_row = tk.Frame(body, bg=CARD)
        edit_row.pack(fill="x")
        edit_row.columnconfigure(0, weight=1)
        edit_row.columnconfigure(1, weight=1)
        ttk.Button(edit_row, text="Remove last point", command=self._on_remove_last_point).grid(
            row=0, column=0, sticky="ew")
        ttk.Button(edit_row, text="Clear all points", command=self._on_clear_points).grid(
            row=0, column=1, sticky="ew", padx=(6, 0))
        ttk.Button(body, text="▶  Run with these points", style="Accent.TButton",
                   command=self._on_restart_with_points).pack(fill="x", pady=(6, 0))

        self._points_hint = tk.Label(body, text="", bg=CARD, fg=MUTED, font=self.f_small, anchor="w",
                                     justify="left", wraplength=330)
        self._autowrap(self._points_hint).pack(fill="x", pady=(6, 0))
        self._refresh_points_hint()
        return frame

    def _create_control_frame(self, parent):
        """Section 3: move the joints by hand, record and play back."""
        frame, body = self._card(parent, "Manual joint control", number=3,
                                 note="Arrow keys: Up/Down move the selected joint, Left/Right select a joint, "
                                      "T shows the hand's path. Buttons do the same.")
        names = ["Shoulder", "Elbow", "Wrist"]
        for i in range(self.config.dof):
            row = tk.Frame(body, bg=CARD)
            row.pack(fill="x", pady=2)
            name = names[i] if i < len(names) else f"Joint {i + 1}"
            tk.Label(row, text=f"J{i + 1}  {name}", bg=CARD, fg=INK, font=self.f_bold, width=12,
                     anchor="w").pack(side="left")
            ttk.Button(row, text="−", width=3, command=lambda idx=i: self._increment_joint(idx, -0.05)).pack(
                side="left")
            angle_label = tk.Label(row, text="", bg=CARD, fg=INK, font=self.f_mono, width=18)
            angle_label.pack(side="left", padx=4)
            self.angle_labels[i] = angle_label
            ttk.Button(row, text="+", width=3, command=lambda idx=i: self._increment_joint(idx, +0.05)).pack(
                side="left")
        self._refresh_joint_labels()

        buttons = tk.Frame(body, bg=CARD)
        buttons.pack(fill="x", pady=(8, 0))
        for col in range(3):
            buttons.columnconfigure(col, weight=1, uniform="joint_btns")
        self.record_button = ttk.Button(buttons, text="Record", command=self._on_record)
        self.record_button.grid(row=0, column=0, sticky="ew", padx=1, pady=1)
        ttk.Button(buttons, text="Clear", command=self._on_clear_recording).grid(
            row=0, column=1, sticky="ew", padx=1, pady=1)
        self.playback_button = ttk.Button(buttons, text="Playback", command=self._on_playback)
        self.playback_button.grid(row=0, column=2, sticky="ew", padx=1, pady=1)
        ttk.Button(buttons, text="Save motion", command=self._on_save_motion).grid(
            row=1, column=0, sticky="ew", padx=1, pady=1)
        ttk.Button(buttons, text="Load motion", command=self._on_load_motion).grid(
            row=1, column=1, sticky="ew", padx=1, pady=1)
        ttk.Button(buttons, text="Reset arm", command=self._on_reset_arm).grid(
            row=1, column=2, sticky="ew", padx=1, pady=1)

        self._trail_var = tk.BooleanVar(value=self.show_trajectory)
        ttk.Checkbutton(body, text="Show the hand's path (T)", variable=self._trail_var,
                        command=self._on_trail_toggle).pack(anchor="w", pady=(8, 0))
        return frame

    def _create_properties_frame(self, parent):
        """Section 4: the arm's dimensions (manual mode)."""
        frame, body = self._card(parent, "Arm properties", number=4,
                                 note="These apply to manual control. A trained model always runs on the arm it "
                                      "was trained on, and the sliders switch to that arm when it starts.")
        names = ["Upper arm", "Forearm + hand", "Hand"]
        tk.Label(body, text="Link lengths (m)", bg=CARD, fg=INK, font=self.f_bold, anchor="w").pack(fill="x")
        for i in range(self.config.dof):
            self._slider_row(body, f"link_len_{i}", names[i] if i < len(names) else f"Link {i + 1}", 0.1, 2.0,
                             self.config.link_lengths[i],
                             lambda val, idx=i: self._on_link_length_change(idx, val),
                             lambda idx=i: self._on_link_length_entry(idx))
        tk.Label(body, text="Segment masses (kg)", bg=CARD, fg=INK, font=self.f_bold, anchor="w").pack(
            fill="x", pady=(8, 0))
        for i in range(min(3, self.config.dof)):
            self._slider_row(body, f"mass_{i}", names[i] if i < len(names) else f"Mass {i + 1}", 0.1, 10.0,
                             self.config.masses[i],
                             lambda val, idx=i: self._on_mass_change(idx, val),
                             lambda idx=i: self._on_mass_entry(idx))
        tk.Label(body, text="Joint damping (N·m·s/rad)", bg=CARD, fg=INK, font=self.f_bold, anchor="w").pack(
            fill="x", pady=(8, 0))
        self._slider_row(body, "damping", "Damping", 0.0, 1.0, self.config.damping, self._on_damping_change)

        buttons = tk.Frame(body, bg=CARD)
        buttons.pack(fill="x", pady=(8, 0))
        for col in range(3):
            buttons.columnconfigure(col, weight=1, uniform="props_btns")
        ttk.Button(buttons, text="Reset", command=self._on_reset_defaults).grid(row=0, column=0, sticky="ew",
                                                                               padx=1)
        ttk.Button(buttons, text="Save", command=self._on_save_config).grid(row=0, column=1, sticky="ew", padx=1)
        ttk.Button(buttons, text="Load", command=self._on_load_config).grid(row=0, column=2, sticky="ew", padx=1)
        return frame

    # ============================================================ centre

    def _create_visualization_frame(self, parent):
        """The arm view, with the run's progress strip under it."""
        frame = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)

        self.fig = Figure(figsize=(7, 6), dpi=100, facecolor=CARD)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal", adjustable="box")
        self.fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.93)
        strip = tk.Frame(frame, bg=CARD)
        strip.pack(fill="x", side="bottom", padx=12, pady=(4, 10))

        self.canvas_agg = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas_agg.mpl_connect("button_press_event", self._on_canvas_click)
        self.canvas_agg.mpl_connect("draw_event", self._on_arm_draw)
        self.canvas_agg.mpl_connect("resize_event", self._on_arm_resize)
        self._build_arm_scene()
        self.canvas_agg.draw()
        self.canvas_agg.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=(2, 0))

        # Row 1: target, state, progress. Row 2: the hold bar across the full width.
        strip.columnconfigure(2, weight=1)          # spacer: the only column that gives way
        self._target_label = tk.Label(strip, text="No simulation running", bg=CARD, fg=INK, font=self.f_big,
                                      anchor="w")
        self._target_label.grid(row=0, column=0, sticky="w")
        self._state_badge = tk.Label(strip, text="IDLE", bg=TRACK, fg=MUTED, font=self.f_bold, padx=10, pady=2)
        self._state_badge.grid(row=0, column=1, sticky="w", padx=(10, 12))
        self.sim_progress_label = tk.Label(strip, text="", bg=CARD, fg=MUTED, font=self.f_body, anchor="e")
        self.sim_progress_label.grid(row=0, column=3, sticky="e")
        hold_row = tk.Frame(strip, bg=CARD)
        hold_row.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        hold_row.columnconfigure(1, weight=1)
        tk.Label(hold_row, text="Hold", bg=CARD, fg=MUTED, font=self.f_small, width=5, anchor="w").grid(
            row=0, column=0, sticky="w")
        self.sim_hold_bar = ttk.Progressbar(hold_row, orient="horizontal", length=80, mode="determinate",
                                            maximum=100, style="Hold.Horizontal.TProgressbar")
        self.sim_hold_bar.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self._hold_pct_label = tk.Label(hold_row, text="0%", bg=CARD, fg=INK, font=self.f_bold, width=5,
                                        anchor="e")
        self._hold_pct_label.grid(row=0, column=2, sticky="e")
        return frame

    def _create_plots_frame(self, parent):
        """Muscle torque, hand velocity and hand acceleration against time."""
        frame = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        fig = Figure(figsize=(9, 2.4), dpi=100, facecolor=CARD)
        self.torque_ax, self.velacc_ax_vel, self.velacc_ax_acc = fig.subplots(1, 3)
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.2, top=0.86, wspace=0.28)
        self.torque_fig = self.velacc_fig = fig

        ax = self.torque_ax
        ax.set_title("Muscle torque (N·m)", fontsize=9, loc="left")
        self._ln_tau = [ax.plot([], [], color=SHOULDER_C, lw=1.4, label="shoulder")[0],
                        ax.plot([], [], color=ELBOW_C, lw=1.4, label="elbow")[0]]
        self._ln_grav = [ax.plot([], [], color=SHOULDER_C, lw=0.9, ls="--", alpha=0.8)[0],
                         ax.plot([], [], color=ELBOW_C, lw=0.9, ls="--", alpha=0.8)[0]]
        ax.legend(handles=self._ln_tau + [Line2D([], [], color=MUTED, lw=0.9, ls="--",
                                                 label="gravity load")], fontsize=7, loc="upper right")

        ax = self.velacc_ax_vel
        ax.set_title("Hand velocity (m/s)", fontsize=9, loc="left")
        self._ln_vel = [ax.plot([], [], color=SHOULDER_C, lw=1.1, label="Vx")[0],
                        ax.plot([], [], color=ELBOW_C, lw=1.1, label="Vy")[0],
                        ax.plot([], [], color=INK, lw=1.4, label="speed")[0]]
        ax.legend(fontsize=7, loc="upper right")

        ax = self.velacc_ax_acc
        ax.set_title("Hand acceleration (m/s²)", fontsize=9, loc="left")
        self._ln_acc = [ax.plot([], [], color=SHOULDER_C, lw=1.1, label="Ax")[0],
                        ax.plot([], [], color=ELBOW_C, lw=1.1, label="Ay")[0]]
        ax.legend(fontsize=7, loc="upper right")

        self._plot_empty_text = []
        for ax in (self.torque_ax, self.velacc_ax_vel, self.velacc_ax_acc):
            ax.set_xlabel("time (s)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            self._plot_empty_text.append(ax.text(0.5, 0.5, "Simulation not running", ha="center", va="center",
                                                 transform=ax.transAxes, fontsize=8, color=MUTED))

        self.torque_canvas = self.velacc_canvas = FigureCanvasTkAgg(fig, master=frame)
        self.torque_canvas.draw()
        self.torque_canvas.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=2)
        return frame

    # ============================================================ readouts

    def _create_status_frame(self, parent):
        frame, body = self._card(parent, "Live status")
        tiles = [("state", "State"), ("episode", "Episode"),
                 ("time", "Episode time"), ("target", "Target"),
                 ("reached", "Targets reached"), ("hold", "Hold progress"),
                 ("distance", "Distance to target"), ("speed", "Hand speed"),
                 ("reward", "Episode reward"), ("holds", "Holds completed")]
        body.columnconfigure(0, weight=1, uniform="tiles")
        body.columnconfigure(1, weight=1, uniform="tiles")
        for i, (key, caption) in enumerate(tiles):
            cell = tk.Frame(body, bg=CARD)
            cell.grid(row=i // 2, column=i % 2, sticky="nsew", pady=3)
            tk.Label(cell, text=caption, bg=CARD, fg=MUTED, font=self.f_small, anchor="w").pack(fill="x")
            value = tk.Label(cell, text="–", bg=CARD, fg=INK, font=self.f_value, anchor="w")
            value.pack(fill="x")
            self._tiles[key] = value
        return frame

    def _create_muscles_frame(self, parent):
        frame, body = self._card(parent, "Muscle activation")
        body.columnconfigure(1, weight=1)
        self._muscle_rows = []
        for j, (name, detail, colour) in enumerate(MUSCLE_NAMES):
            label = tk.Label(body, text=name, bg=CARD, fg=INK, font=self.f_body, anchor="w", width=17)
            label.grid(row=j, column=0, sticky="w", pady=2)
            canvas = tk.Canvas(body, height=14, bg=TRACK, highlightthickness=0, borderwidth=0)
            canvas.grid(row=j, column=1, sticky="ew", pady=2)
            rect = canvas.create_rectangle(0, 0, 0, 14, fill=colour, width=0)
            pct = tk.Label(body, text="0%", bg=CARD, fg=MUTED, font=self.f_small, width=5, anchor="e")
            pct.grid(row=j, column=2, sticky="e")
            self._muscle_rows.append((canvas, rect, pct))
        self._muscle_note = tk.Label(body, text="Squeezing opposing muscles together stiffens a joint; the arm "
                                                "does this to hold still.", bg=CARD, fg=MUTED, font=self.f_small,
                                     anchor="w", justify="left", wraplength=380)
        self._autowrap(self._muscle_note).grid(row=len(MUSCLE_NAMES), column=0, columnspan=3, sticky="ew",
                                               pady=(4, 0))
        return frame

    def _create_metrics_frame(self, parent):
        """Every reading, as text."""
        frame, body = self._card(parent, "All readings")
        self.metrics_text = tk.Text(body, height=7, font=self.f_mono, bg="#f7f8fa", fg=INK, relief="flat",
                                    wrap="none", state="disabled", highlightthickness=0, padx=6, pady=4)
        bar = ttk.Scrollbar(body, orient="vertical", command=self.metrics_text.yview)
        self.metrics_text.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.metrics_text.pack(side="left", fill="both", expand=True)
        return frame

    def _create_details_frame(self, parent):
        frame, body = self._card(parent, "Model details")
        self.sim_model_details_text = tk.Text(body, height=6, font=self.f_mono, bg="#f7f8fa", fg=INK,
                                              relief="flat", wrap="word", state="disabled",
                                              highlightthickness=0, padx=6, pady=4)
        bar = ttk.Scrollbar(body, orient="vertical", command=self.sim_model_details_text.yview)
        self.sim_model_details_text.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.sim_model_details_text.pack(side="left", fill="both", expand=True)
        self._set_model_details_text(
            "Name: -\nAlgorithm: -\nPolicy: -\nPath: -\nStatus: Select a trained model to view details")
        return frame

    # ============================================================== window

    def _sync_ui_to_config(self):
        """Synchronize UI sliders to current config after config change"""
        for i, val in enumerate(self.config.link_lengths):
            if f"link_len_{i}" not in self.slider_vars:
                continue
            self.slider_vars[f"link_len_{i}"].set(val)
            self.slider_vars[f"link_len_{i}_label"].config(text=f"{val:.2f}")
            self.slider_vars[f"link_len_{i}_entry"].set(f"{val:.2f}")
        for i in range(min(3, self.config.dof)):
            if f"mass_{i}" not in self.slider_vars:
                continue
            self.slider_vars[f"mass_{i}"].set(self.config.masses[i])
            self.slider_vars[f"mass_{i}_label"].config(text=f"{self.config.masses[i]:.2f}")
            self.slider_vars[f"mass_{i}_entry"].set(f"{self.config.masses[i]:.2f}")
        if "damping" in self.slider_vars:
            self.slider_vars["damping"].set(self.config.damping)
            self.slider_vars["damping_label"].config(text=f"{self.config.damping:.2f}")

    def _compute_positions(self):
        """Recompute arm positions after config change"""
        self.controller.positions = ArmKinematics.forward_kinematics(
            self.controller.angles, self.config
        )

    def _calculate_axis_limits(self):
        """Calculate axis limits based on current arm configuration"""
        total_reach = sum(self.config.link_lengths)
        margin = total_reach * 0.2  # 20% margin
        limit = total_reach + margin
        return -limit, limit

    def create_window(self):
        """Create the main GUI window layout"""
        self._setup_style()

        # Header: title, what the window is doing, Exit.
        header = tk.Frame(self.root, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        header.pack(fill="x", side="top")
        titles = tk.Frame(header, bg=CARD)
        titles.pack(side="left", padx=16, pady=8)
        tk.Label(titles, text="Human Arm - Reinforcement Learning Controller", bg=CARD, fg=INK,
                 font=self.f_title).pack(anchor="w")
        tk.Label(titles, text="Two-joint, muscle-driven right arm trained with Soft Actor-Critic",
                 bg=CARD, fg=MUTED, font=self.f_small).pack(anchor="w")
        ttk.Button(header, text="Exit", command=self.on_closing).pack(side="right", padx=16)
        self._mode_pill = tk.Label(header, text="IDLE", bg=TRACK, fg=MUTED, font=self.f_bold, padx=12, pady=4)
        self._mode_pill.pack(side="right", padx=(0, 10))

        # Status line along the bottom.
        status = tk.Frame(self.root, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        status.pack(fill="x", side="bottom")
        self._status_label = tk.Label(status, text="Ready. Pick a model in section 1 and press Run simulation.",
                                      bg=CARD, fg=INK, font=self.f_small, anchor="w")
        self._status_label.pack(side="left", fill="x", expand=True, padx=12, pady=4)
        self._fps_label = tk.Label(status, text="", bg=CARD, fg=MUTED, font=self.f_small)
        self._fps_label.pack(side="right", padx=12)

        # Three resizable columns.
        body = tk.PanedWindow(self.root, orient="horizontal", bg=BG, sashwidth=8, borderwidth=0,
                              sashrelief="flat")
        body.pack(fill="both", expand=True, padx=8, pady=8)

        # Left: controls, scrollable so nothing is ever cut off.
        left_outer, self._left_inner = self._scroll_column(body)
        for build in (self._create_model_selection_frame, self._create_targets_frame,
                      self._create_control_frame, self._create_properties_frame):
            build(self._left_inner).pack(fill="x", pady=(0, 8), padx=(0, 4))

        # Centre: arm view over the live plots.
        centre = tk.PanedWindow(body, orient="vertical", bg=BG, sashwidth=8, borderwidth=0, sashrelief="flat")
        centre.add(self._create_visualization_frame(centre), stretch="always", minsize=360)
        centre.add(self._create_plots_frame(centre), stretch="never", minsize=170, height=250)

        # Right: readouts; also scrollable, with "All readings" taking any spare height.
        right_outer, right = self._scroll_column(body, fill_height=True)
        self._create_status_frame(right).pack(fill="x", pady=(0, 8), padx=(0, 4))
        self._create_muscles_frame(right).pack(fill="x", pady=(0, 8), padx=(0, 4))
        self._create_details_frame(right).pack(fill="x", side="bottom", padx=(0, 4))
        self._create_metrics_frame(right).pack(fill="both", expand=True, pady=(0, 8), padx=(0, 4))

        body.add(left_outer, stretch="never", minsize=330, width=400)
        body.add(centre, stretch="always", minsize=520)
        body.add(right_outer, stretch="never", minsize=340, width=440)

        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self._update_plots()

    def _scroll_column(self, parent, fill_height=False):
        """A column that scrolls when its content is taller than the window.

        With fill_height, content that is shorter than the column is stretched
        to fill it, so a widget packed with expand=True takes the spare room.
        """
        outer = tk.Frame(parent, bg=BG)
        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0, borderwidth=0)
        bar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        def fit(_event=None):
            width = canvas.winfo_width()
            if fill_height:
                height = max(canvas.winfo_height(), inner.winfo_reqheight())
                canvas.itemconfigure(win, width=width, height=height)
            else:
                canvas.itemconfigure(win, width=width)
            canvas.configure(scrollregion=canvas.bbox("all"))

        canvas.bind("<Configure>", lambda e: (fit(), canvas.after_idle(fit)))
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        self._scroll_columns.append((canvas, inner))
        return outer, inner

    def _on_mousewheel(self, event):
        """Scroll a side column when the pointer is over it (text boxes scroll themselves)."""
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            if isinstance(widget, tk.Text):
                return
            for canvas, inner in self._scroll_columns:
                if widget is canvas:
                    if inner.winfo_height() > canvas.winfo_height() + 1:
                        canvas.yview_scroll(int(-event.delta / 120), "units")
                    return
            widget = getattr(widget, "master", None)

    # ============================================================ handlers

    def _on_link_length_change(self, index, value):
        """Handle link length slider change"""
        if self.simulation_active:
            self._stop_simulation()
        self.config.link_lengths[index] = float(value)
        self.slider_vars[f"link_len_{index}_label"].config(text=f"{float(value):.2f}")
        self.slider_vars[f"link_len_{index}_entry"].set(f"{float(value):.2f}")
        self._compute_positions()  # Recompute arm positions with new link length

    def _on_mass_change(self, index, value):
        """Handle mass slider change"""
        if self.simulation_active:
            self._stop_simulation()
        self.config.masses[index] = float(value)
        self.slider_vars[f"mass_{index}_label"].config(text=f"{float(value):.2f}")
        self.slider_vars[f"mass_{index}_entry"].set(f"{float(value):.2f}")
        self._compute_positions()  # Recompute arm positions

    def _on_link_length_entry(self, index):
        """Handle manual link-length numeric input."""
        key = f"link_len_{index}_entry"
        current = float(self.config.link_lengths[index])
        raw = str(self.slider_vars[key].get()).strip()
        try:
            value = float(raw)
        except ValueError:
            value = current

        value = max(0.1, min(2.0, value))
        self.slider_vars[f"link_len_{index}"].set(value)
        self.slider_vars[key].set(f"{value:.2f}")
        if abs(value - current) > 1e-12:
            self._on_link_length_change(index, value)

    def _on_mass_entry(self, index):
        """Handle manual mass numeric input."""
        key = f"mass_{index}_entry"
        current = float(self.config.masses[index])
        raw = str(self.slider_vars[key].get()).strip()
        try:
            value = float(raw)
        except ValueError:
            value = current

        value = max(0.1, min(10.0, value))
        self.slider_vars[f"mass_{index}"].set(value)
        self.slider_vars[key].set(f"{value:.2f}")
        if abs(value - current) > 1e-12:
            self._on_mass_change(index, value)

    def _on_damping_change(self, value):
        """Handle damping slider change"""
        if self.simulation_active:
            self._stop_simulation()
        self.config.damping = float(value)
        self.slider_vars["damping_label"].config(text=f"{float(value):.2f}")
        self._compute_positions()  # Recompute arm positions

    def _increment_joint(self, joint_id, delta):
        """Increment joint angle"""
        if self.simulation_active:
            self._stop_simulation()
        self.controller.increment_joint(joint_id, delta)
        self._refresh_joint_labels()

    def _on_trail_toggle(self):
        self.show_trajectory = bool(self._trail_var.get())
        self._log(f"[OK] Hand path {'shown' if self.show_trajectory else 'hidden'}")

    def _on_record(self):
        """Handle record button"""
        if self.simulation_active:
            self._stop_simulation()
        self.recording = not self.recording
        if self.recording:
            self.recorder.start_recording()
            self.record_button.config(text="Stop recording")
            self._log("[OK] Recording started")
        else:
            self.recorder.stop_recording()
            self.record_button.config(text="Record")
            self._log(f"[OK] Recording stopped ({self.recorder.get_num_frames()} frames)")

    def _on_clear_recording(self):
        """Clear recording"""
        if self.simulation_active:
            self._stop_simulation()
        self.recorder.clear_frames()
        self._log("[OK] Recording cleared")

    def _on_save_motion(self):
        """Save motion recording"""
        if self.simulation_active:
            self._stop_simulation()
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.recorder.save_to_json(filepath)
                messagebox.showinfo("Success", f"Motion saved to {filepath}")
                self._log(f"[OK] Motion saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving motion: {e}")

    def _on_load_motion(self):
        """Load motion recording"""
        if self.simulation_active:
            self._stop_simulation()
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.recorder = MotionRecorder.load_from_json(filepath)
                messagebox.showinfo("Success", f"Loaded {self.recorder.get_num_frames()} frames")
                self._log(f"[OK] Loaded {self.recorder.get_num_frames()} frames")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading motion: {e}")

    def _on_playback(self):
        """Handle playback button"""
        if self.recorder.get_num_frames() > 0:
            if self.simulation_active:
                self._stop_simulation()
            self.playing_back = not self.playing_back
            if self.playing_back:
                self.playback_frames = self.recorder.get_frames()
                self.playback_index = 0
                self.playback_button.config(text="Stop playback")
                self._log("[OK] Playback started")
            else:
                self.playback_button.config(text="Playback")
                self._log("[OK] Playback stopped")
        else:
            messagebox.showwarning("Warning", "No frames recorded to playback")

    # ------------------------------------------------------------ models

    def _discover_packages(self):
        """Trained packages under project_assets/outputs, best first: {label: model zip}."""
        found = {}
        if not OUTPUTS_DIR.is_dir():
            return found
        entries = []
        for d in OUTPUTS_DIR.glob("CHAMPION_HUMAN_V*_*"):
            if not (d / "run_manifest.json").exists():
                continue
            model = d / "sac_model_best.zip"
            if not model.exists():
                model = d / "sac_model.zip"
            if not model.exists():
                continue
            stage = d.name.rsplit("_", 1)[-1]
            rank = PACKAGE_ORDER.index(stage) if stage in PACKAGE_ORDER else len(PACKAGE_ORDER)
            what = KNOWN_PACKAGES.get(stage) or self._describe_manifest(read_manifest(d / "run_manifest.json"))
            entries.append((rank, stage, f"{stage}  -  {what}", model))
        for _rank, _stage, label, model in sorted(entries):
            found[label] = model
        return found

    @staticmethod
    def _describe_manifest(manifest):
        env = (manifest or {}).get("env", {})
        n = env.get("random_waypoints")
        if env.get("free_order"):
            return f"{n} points, picks its own order"
        if n:
            return f"reaches any {n} points"
        wps = env.get("waypoints") or []
        return f"{len(wps)} fixed points only" if wps else "trained model"

    def _on_pick_package(self):
        model = self._package_choices.get(self._package_var.get())
        if model is not None:
            self._load_model(str(model))

    def _on_select_model(self):
        """Load a trained model for policy simulation."""
        filepath = filedialog.askopenfilename(
            title="Select trained model (.zip)",
            initialdir=str(OUTPUTS_DIR) if OUTPUTS_DIR.is_dir() else None,
            filetypes=[("Model files", "*.zip"), ("All files", "*.*")],
        )
        if not filepath:
            return
        self._load_model(filepath)

    def _load_model(self, filepath):
        """Select a model (package zip or bare zip) for the next run."""
        if self.simulation_active:
            self._stop_simulation()
        model_base = self._normalize_model_base(filepath)
        algorithm = self._get_algorithm_from_metadata(model_base)
        manifest_path = find_manifest(filepath)
        self.sim_package_dir = str(manifest_path.parent) if manifest_path else None
        self.sim_manifest = read_manifest(manifest_path) if manifest_path else None
        if algorithm is None and self.sim_manifest is not None:
            algorithm = str(self.sim_manifest.get("algorithm", "SAC")).upper()
        if algorithm is None:
            messagebox.showerror(
                "Unsupported Model",
                "Could not infer algorithm from metadata/name.\n"
                "Expected PPO/SAC/A2C/DQN model artifact.",
            )
            return

        self.sim_model_path = model_base
        self.sim_algorithm = algorithm
        self.sim_model = None
        self._manifest_env = None          # rebuilt from the new manifest on demand
        self._feasible_cloud_key = None
        metadata = self._read_model_metadata(model_base)
        episodes = 0
        best_reward = "-"
        timestamp = "-"
        policy_name = "MlpPolicy"
        if isinstance(metadata, dict):
            history = metadata.get("training_history", {})
            if isinstance(history, dict):
                rewards = history.get("episode_rewards", [])
                if isinstance(rewards, list):
                    episodes = len(rewards)
            best_reward = metadata.get("best_reward", "-")
            timestamp = metadata.get("timestamp", "-")
            policy_name = str(metadata.get("policy", "MlpPolicy"))

        package_name = Path(self.sim_package_dir).name if self.sim_package_dir else Path(model_base).name
        self.sim_model_label.config(text=f"{package_name}  ({algorithm})")
        env_desc = (self.sim_manifest or {}).get("env", {})
        if self.sim_manifest is None:
            kind = "No package manifest: runs on the default arm and the demo points."
        elif env_desc.get("free_order"):
            kind = (f"Free order: sees all {env_desc.get('random_waypoints')} points at once and picks its own "
                    f"route. Place exactly {env_desc.get('random_waypoints')} points.")
        elif env_desc.get("random_waypoints"):
            kind = (f"Trained on random points: reaches any {env_desc.get('random_waypoints')} points you "
                    f"choose, in order, and holds the last one.")
        else:
            kind = ("Trained on fixed points only: it will not reach other points. Use R3 for points "
                    "you choose.")
        self._model_kind_label.config(text=kind)
        self.sim_toggle_button.config(state="normal", text="▶  Run simulation", style="Accent.TButton")
        for label, path in self._package_choices.items():
            if Path(path).with_suffix("") == Path(model_base):
                self._package_var.set(label)
                break
        else:
            self._package_var.set("")

        details_text = (
            f"Name: {Path(model_base).name}\n"
            f"Algorithm: {algorithm}\n"
            f"Policy: {policy_name}\n"
            f"Path: {model_base}.zip\n"
            f"Episodes: {episodes}\n"
            f"Best Reward: {best_reward}\n"
            f"Saved: {timestamp}"
        )
        if self.sim_manifest is not None:
            details_text += (
                f"\nPackage: {self.sim_package_dir}"
                f"\nPreset: {env_desc.get('preset')}  obs v{env_desc.get('obs_version')}  "
                f"{env_desc.get('actuation_mode')}  waypoints "
                f"{env_desc.get('waypoints') if not env_desc.get('random_waypoints') else 'random'}"
                f"\nSeed {self.sim_manifest.get('seed')}  trained {self.sim_manifest.get('final_timesteps')} steps"
                f"\nBest score {self.sim_manifest.get('best_score')}"
            )
            summary = self._package_eval_summary(self.sim_package_dir)
            if summary:
                details_text += f"\n{summary}"
        self._set_model_details_text(details_text)
        self._refresh_points_hint()
        self._log(f"[OK] Loaded model: {package_name} ({algorithm}). Press Run simulation.")

    @staticmethod
    def _package_eval_summary(package_dir):
        """One line from the package's final evaluation, if it has one."""
        path = Path(package_dir) / "final_eval.json"
        try:
            s = json.loads(path.read_text()).get("summary", {})
            return (f"Final evaluation: {s['successes']}/{s['n_episodes']} fresh episodes held the last point "
                    f"({100 * s['success_rate']:.0f}%)")
        except Exception:
            return ""

    def _on_smooth_toggle(self):
        """Toggle action smoothing on/off."""
        self.action_smoothing = self._smooth_var.get()
        state = "ON" if self.action_smoothing else "OFF"
        extra = " - restart the simulation to apply" if self.simulation_active else ""
        self._log(f"[OK] Action smoothing {state} (alpha={self.action_smooth_alpha:.2f}){extra}")

    def _on_toggle_simulation(self):
        """Start/stop policy simulation."""
        if self.simulation_active:
            self._stop_simulation()
            return
        self._start_simulation()

    def _on_pause(self):
        if not self.simulation_active:
            return
        self.sim_paused = not self.sim_paused
        self._pause_button.config(text="Resume" if self.sim_paused else "Pause")
        self._last_slow_update = 0.0
        self._log(f"[OK] Simulation {'paused' if self.sim_paused else 'resumed'}")

    def _on_next_episode(self):
        """Start the next episode now (new random targets for a random-point model)."""
        if not self.simulation_active or self.sim_env is None:
            return
        self._begin_next_episode()
        self._log(f"[OK] Episode {self.sim_episode_count} started")

    # ------------------------------------------------------------ targets

    def _on_waypoint_choice(self):
        self.sim_waypoint_choice = self._wp_choice_var.get()
        if self.sim_waypoint_choice == "custom" and not self._place_mode:
            self._set_place_mode(True)
        elif self.sim_waypoint_choice != "custom" and self._place_mode:
            self._set_place_mode(False)
        self._refresh_points_hint()

    def _on_place_button(self):
        self._set_place_mode(not self._place_mode)

    def _on_place_toggle(self):
        self._place_mode = bool(self._place_var.get())
        self._log(f"[OK] Waypoint placement {'ON: click the arm panel to add points' if self._place_mode else 'OFF'}")

    def _set_place_mode(self, on):
        self._place_var.set(bool(on))
        self._on_place_toggle()
        if on:
            self.sim_waypoint_choice = "custom"
            self._wp_choice_var.set("custom")
        self._place_button.config(
            text=f"✚  Click the arm view to add points: {'ON' if on else 'OFF'}",
            style="On.Toggle.TButton" if on else "Toggle.TButton")
        try:
            self.canvas_agg.get_tk_widget().config(cursor="crosshair" if on else "")
        except Exception:
            pass
        self._refresh_points_hint()

    def _sync_custom_text(self):
        self.sim_custom_waypoints = ";".join(f"{x:.3f},{y:.3f}" for x, y in self.clicked_waypoints)
        if hasattr(self, "_wp_custom_var"):
            self._wp_custom_var.set(self.sim_custom_waypoints)

    def _on_custom_typed(self):
        self.sim_custom_waypoints = self._wp_custom_var.get()
        if self.sim_custom_waypoints.strip() and self.sim_waypoint_choice != "custom":
            self.sim_waypoint_choice = "custom"
            self._wp_choice_var.set("custom")

    def _on_custom_committed(self):
        """Typed coordinates become placed points (so they are drawn)."""
        text = self._wp_custom_var.get().strip()
        current = ";".join(f"{x:.3f},{y:.3f}" for x, y in self.clicked_waypoints)
        if text == current:
            return
        if not text:
            self.clicked_waypoints, self.clicked_requests = [], []
            self._refresh_points_hint()
            return
        try:
            pts = resolve_waypoints(text)
        except ValueError:
            self._log("[WARN] Could not read those points. Write them as x,y; x,y  (e.g. 1.40,0.15; 1.20,0.40)")
            return
        self.clicked_waypoints = [[float(p[0]), float(p[1])] for p in pts]
        self.clicked_requests = [(list(p), list(p)) for p in self.clicked_waypoints]
        self.sim_custom_waypoints = text
        self._log(f"[OK] {len(self.clicked_waypoints)} typed point(s) set; press 'Run with these points'")
        self._refresh_points_hint()

    def _on_remove_last_point(self):
        if not self.clicked_waypoints:
            self._log("[WARN] There are no placed points to remove")
            return
        removed = self.clicked_waypoints.pop()
        if self.clicked_requests:
            self.clicked_requests.pop()
        self._sync_custom_text()
        self._log(f"[OK] Removed point {len(self.clicked_waypoints) + 1} at ({removed[0]:.2f}, {removed[1]:.2f})")
        self._refresh_points_hint()

    def _on_clear_points(self):
        self.clicked_waypoints = []
        self.clicked_requests = []
        self.sim_custom_waypoints = ""
        if hasattr(self, "_wp_custom_var"):
            self._wp_custom_var.set("")
        self._log("[OK] Placed waypoints cleared")
        self._refresh_points_hint()

    def _on_restart_with_points(self):
        if not self.clicked_waypoints:
            messagebox.showwarning("Waypoints", "Place at least one point first: turn on 'Click the arm view to "
                                                "add points' and click inside the green area.")
            return
        if not self.sim_model_path:
            messagebox.showwarning("Warning", "Please select a trained model first (section 1)")
            return
        self.sim_waypoint_choice = "custom"
        self._wp_choice_var.set("custom")
        if self.simulation_active:
            self._stop_simulation()
        self._start_simulation()

    def _refresh_points_hint(self):
        hint = getattr(self, "_points_hint", None)
        if hint is None:
            return
        n = len(self.clicked_waypoints)
        env_desc = (self.sim_manifest or {}).get("env", {})
        if self._place_mode:
            text = (f"{n} point(s) placed. Click inside the green area of the arm view to add one; a click the "
                    f"arm cannot hold is moved to the nearest point it can. Right-click a point to remove it.")
        elif self.sim_waypoint_choice == "custom":
            text = (f"{n} point(s) placed. Turn on 'Click the arm view' to add more, or type them above."
                    if n else "No points yet: turn on 'Click the arm view' and click inside the green area.")
        elif self.sim_waypoint_choice == "from package":
            text = "The model's own points: random each episode for R-models, fixed for A/B models."
        else:
            text = "A fixed demonstration set that every model can be shown on."
        if self.sim_manifest is not None and self.sim_waypoint_choice == "custom":
            if env_desc.get("free_order"):
                text += f"  This model needs exactly {env_desc.get('random_waypoints')} points."
            elif not env_desc.get("random_waypoints"):
                text += "  Note: the loaded model was trained on fixed points only."
        hint.config(text=text)

    def _feasibility_env(self):
        """The env whose rules decide where points can go (and what is shaded green).

        The running simulation's env if there is one; otherwise the loaded
        model's, rebuilt from its manifest - so the green area is that model's
        own training region before a run starts, not some default; otherwise the
        default arm's. Models trained on the full human workspace (R7 on) and
        older ones (the 18 cm reach-margin rules) shade different regions.
        """
        base = getattr(self.sim_env, "unwrapped", None) if getattr(self, "sim_env", None) is not None else None
        if base is not None:
            return base
        if getattr(self, "sim_manifest", None) is not None:
            if getattr(self, "_manifest_env", None) is None:
                try:
                    self._manifest_env = env_from_manifest(self.sim_manifest)
                except Exception:
                    self._manifest_env = None
            if self._manifest_env is not None:
                return self._manifest_env
        key = (self.config.name, tuple(self.config.link_lengths))
        if getattr(self, "_default_env_key", None) != key:
            self._default_env = ArmTaskEnv(actuation_mode="torque", config=self.config)
            self._default_env_key = key
        return self._default_env

    def _nearest_placeable(self, env, wp, existing):
        """Nearest point the arm can actually hold that also clears the spacing
        rules, or None if nothing within reach of the click qualifies."""
        pts = env.feasible_points()
        target = np.asarray(wp, dtype=float)
        for idx in np.argsort(np.linalg.norm(pts - target, axis=1))[:400]:
            cand = [float(v) for v in pts[idx]]
            try:
                env._validate_waypoint_spacing([np.asarray(w, dtype=float) for w in existing + [cand]])
            except ValueError:
                continue
            return cand, float(np.linalg.norm(np.asarray(cand) - target))
        return None, 0.0

    def _on_canvas_click(self, event):
        """Add a waypoint at the clicked workspace point (plot frame is shoulder-local).

        A click the arm cannot hold is moved to the nearest point it can, rather
        than being dropped. Measured on uniformly random clicks, only 22% land
        somewhere holdable - 48% are outside the reach margin alone - so
        refusing them outright means roughly seven clicks in eight do nothing.
        The limits themselves are real (the policy reaches 24% of targets in the
        0.47-0.52 m ring and 0% beyond, so accepting those clicks would only
        trade a silent refusal for a visible failure); what was wrong was
        discarding the user's intent instead of honouring it as closely as the
        arm allows.

        Right-click removes the placed point nearest the click.
        """
        if event.inaxes is not self.ax or event.xdata is None:
            return
        env = self._feasibility_env()
        shoulder = np.asarray(env.shoulder_base_position, dtype=float)
        wp = [float(event.xdata) + shoulder[0], float(event.ydata) + shoulder[1]]
        if getattr(event, "button", 1) == 3:
            if self.clicked_waypoints:
                d = [np.linalg.norm(np.asarray(p) - np.asarray(wp)) for p in self.clicked_waypoints]
                i = int(np.argmin(d))
                if d[i] < 0.10:
                    self.clicked_waypoints.pop(i)
                    if i < len(self.clicked_requests):
                        self.clicked_requests.pop(i)
                    self._sync_custom_text()
                    self._log(f"[OK] Removed point {i + 1}")
                    self._refresh_points_hint()
            return
        if not self._place_mode:
            if not self.simulation_active:
                self._log("[OK] To add a point, choose 'My own points' or turn on 'Click the arm view to add "
                          "points' (section 2)")
            return
        f = env.waypoint_feasibility(wp, env.max_curriculum_tolerance)
        placeable = f.get("feasible", False)
        if placeable:
            try:
                env._validate_waypoint_spacing(
                    [np.asarray(w, dtype=float) for w in self.clicked_waypoints + [wp]])
            except ValueError:
                placeable = False
        if not placeable:
            snapped, moved = self._nearest_placeable(env, wp, self.clicked_waypoints)
            if snapped is None:
                self._log(f"[WARN] no point near ({wp[0]:.2f}, {wp[1]:.2f}) can be held and kept "
                          f"{2 * env.max_curriculum_tolerance:.2f} m clear of the others; "
                          f"try somewhere else in the shaded region")
                return
            self._log(f"[OK] ({wp[0]:.2f}, {wp[1]:.2f}) moved {moved * 100:.0f} cm to "
                      f"({snapped[0]:.2f}, {snapped[1]:.2f}), the nearest point the arm can reach and hold")
            self.clicked_requests.append((list(wp), list(snapped)))
            wp = snapped
        else:
            self.clicked_requests.append((list(wp), list(wp)))
            self._log(f"[OK] waypoint {len(self.clicked_waypoints) + 1} placed at ({wp[0]:.3f}, {wp[1]:.3f}); "
                      f"press 'Run with these points' to run")
        self.clicked_waypoints = self.clicked_waypoints + [wp]
        self.sim_waypoint_choice = "custom"
        self._sync_custom_text()
        if hasattr(self, "_wp_choice_var"):
            self._wp_choice_var.set("custom")
        self._refresh_points_hint()

    def _feasible_cloud_points(self):
        """Shoulder-local scatter of feasible waypoint positions (cached per arm and rule set)."""
        env = self._feasibility_env()
        key = (self.config.name, tuple(self.config.link_lengths), getattr(env, "reach_margin", None),
               getattr(env, "joint_margin_deg", None))
        if self._feasible_cloud_key != key:
            pts = env.feasible_points() - np.asarray(env.shoulder_base_position, dtype=float)
            self._feasible_cloud = pts
            self._feasible_cloud_key = key
        return self._feasible_cloud

    def _selected_waypoints(self):
        """Waypoint override from the demo controls (None = package/default)."""
        choice = self.sim_waypoint_choice
        if choice == "from package":
            return None
        if choice.endswith("waypoint") or choice.endswith("waypoints"):
            return resolve_waypoints(int(choice.split()[0]))
        if choice == "spread":
            return list(WAYPOINTS_SPREAD)
        if choice == "custom":
            return resolve_waypoints(self.sim_custom_waypoints) if self.sim_custom_waypoints.strip() else None
        return None

    # ---------------------------------------------------------- simulation

    def _start_simulation(self):
        """Initialize simulation environment and model from a package (or bare zip)."""
        if not self.sim_model_path or not self.sim_algorithm:
            messagebox.showwarning("Warning", "Please select a trained model first")
            return

        try:
            if self.playing_back:
                self.playing_back = False
                self.playback_button.config(text="Playback")
            if self.sim_env is not None:
                self.sim_env.close()

            overrides = {"terminate_on_success": False}
            wps = self._selected_waypoints()
            if wps is not None:
                overrides["waypoints"] = wps
            if self.sim_package_dir is not None:
                model_zip = self.sim_model_path if self.sim_model_path.endswith(".zip") else self.sim_model_path + ".zip"
                model, raw_env, _manifest = load_package(model_zip, **overrides)
            else:
                from stable_baselines3 import SAC, PPO, A2C
                alg_class = {"SAC": SAC, "PPO": PPO, "A2C": A2C}.get(self.sim_algorithm, SAC)
                model = alg_class.load(self.sim_model_path, device="cpu")
                raw_env = make_env(actuation_mode=detect_actuation_mode(model),
                                   waypoints=wps if wps is not None else 2, terminate_on_success=False)
                if tuple(model.observation_space.shape) != tuple(raw_env.observation_space.shape):
                    raise ValueError(
                        f"model observation shape {model.observation_space.shape} does not match the env "
                        f"({raw_env.observation_space.shape}, OBS_VERSION {OBS_VERSION}). This model was "
                        f"trained on an older observation layout and cannot be run here.")
                model.set_env(raw_env)

            # The drawn arm must be the env's arm.
            self.config = raw_env.config
            self.controller = ArmController(self.config)
            self.visualizer = ArmVisualizer(dof=self.config.dof, link_lengths=self.config.link_lengths)
            try:
                self._sync_ui_to_config()
            except Exception:
                pass
            self.sim_steps_per_frame = max(1, int(round(0.05 / float(raw_env.dt))))

            if self.action_smoothing and raw_env.actuation_mode == "muscle" and not raw_env.use_activation_dynamics:
                self.sim_env = ActionSmoother(raw_env, alpha=self.action_smooth_alpha)
                self._log(f"[OK] Action smoothing ON (alpha={self.action_smooth_alpha})")
            else:
                if self.action_smoothing and raw_env.use_activation_dynamics:
                    self._log("[OK] Action smoothing skipped: this model has muscle activation dynamics "
                              "(an extra smoother puts it out of distribution)")
                self.sim_env = raw_env
            self._feasible_cloud_key = None
            self.sim_model = model
            trained_random = bool((self.sim_manifest or {}).get("env", {}).get("random_waypoints"))
            free_order = bool(getattr(raw_env, "free_order", False))
            if free_order:
                self.sim_model_kind = "free-order (sees all points, picks its own route)"
            elif trained_random:
                self.sim_model_kind = "random-waypoint (generalising)"
            else:
                self.sim_model_kind = "fixed-waypoint"
            if free_order and wps is not None and len(wps) != raw_env.free_order_slots:
                raise ValueError(
                    f"this free-order model reads exactly {raw_env.free_order_slots} points at once; "
                    f"{len(wps)} were placed. Place {raw_env.free_order_slots} and restart.")
            n_points = "random" if (trained_random and wps is None) else str(len(wps) if wps else
                                                                               raw_env.num_waypoints)
            message = (f"[OK] {raw_env.actuation_mode}-mode model in {raw_env.preset_name}: "
                       f"{n_points} waypoints, tolerance {raw_env.goal_tolerance} m, "
                       f"hold {raw_env.hold_steps_required} steps")
            if trained_random and wps is None:
                message += " (new random points every episode)"
            if free_order:
                message += (f". Free-order model: it sees all {raw_env.free_order_slots} points every step "
                            f"and chooses the order itself.")
            if wps is not None and not trained_random:
                trained_wps = (self.sim_manifest or {}).get("env", {}).get("waypoints")
                message = (f"[WARN] This is a FIXED-waypoint model (trained only on {trained_wps}); "
                           f"it is not expected to reach other points. Use a random-waypoint (R3) package "
                           f"for arbitrary targets.")
                if self.sim_model_label is not None:
                    self.sim_model_label.config(
                        text=f"{Path(self.sim_package_dir or self.sim_model_path).name}  "
                             f"[FIXED-waypoint model: custom points unsupported]")

            self.sim_episode_seed = 0
            self.sim_obs, info = self.sim_env.reset(seed=self.sim_episode_seed)
            self.sim_last_info = info
            self._set_controller_state_from_arrays(info["joint_angles"], info["joint_velocities"])

            self.simulation_active = True
            self.sim_paused = False
            self.sim_step_count = 0
            self.sim_episode_count = 1
            self.sim_successes = 0
            self.sim_episode_reward = 0.0
            self.sim_time = 0.0
            self.sim_step_accumulator = 0.0
            self.sim_success_banner_until = 0.0
            self.prev_sim_velocities = np.array(info["joint_velocities"], dtype=float)
            self._clear_torque_history()
            self._reset_ee_kinematics()
            self._hand_trail.clear()
            self.sim_toggle_button.config(text="■  Stop simulation", style="Stop.TButton")
            self._pause_button.config(state="normal", text="Pause")
            self._next_button.config(state="normal")
            if self._place_mode:
                self._set_place_mode(False)
            self._last_slow_update = 0.0     # refresh the readouts on the next frame
            self._log(message)
            print("[OK] Policy simulation started")
        except Exception as e:
            self.simulation_active = False
            self.sim_model = None
            messagebox.showerror("Simulation Error", f"Failed to start simulation:\n{e}")

    def _stop_simulation(self):
        """Stop policy simulation and cleanup. The plots keep the last run's trace."""
        self.simulation_active = False
        self.sim_paused = False
        self.sim_obs = None
        self.sim_model = None
        if self.sim_env is not None:
            try:
                self.sim_env.close()
            except Exception:
                pass
            self.sim_env = None
        if self.sim_toggle_button is not None:
            self.sim_toggle_button.config(text="▶  Run simulation", style="Accent.TButton")
        if getattr(self, "_pause_button", None) is not None:
            self._pause_button.config(state="disabled", text="Pause")
            self._next_button.config(state="disabled")
        self.prev_sim_velocities = np.zeros(self.config.dof, dtype=float)
        self._last_slow_update = 0.0         # refresh the readouts on the next frame
        self._log("[OK] Policy simulation stopped")

    def _begin_next_episode(self):
        self.sim_episode_count += 1
        self.sim_episode_seed += 1
        self.sim_episode_reward = 0.0
        self.sim_success_banner_until = 0.0
        self.sim_obs, info = self.sim_env.reset(seed=self.sim_episode_seed)
        self.sim_last_info = info
        self.prev_sim_velocities = np.array(info["joint_velocities"], dtype=float)
        self._ee_prev_vel_xy = None
        self._hand_trail.clear()
        self._set_controller_state_from_arrays(info["joint_angles"], info["joint_velocities"])

    def _simulation_step(self):
        """Run a single simulation step if active."""
        if not self.simulation_active or self.sim_model is None or self.sim_env is None or self.sim_obs is None:
            return
        base_env = getattr(self.sim_env, "unwrapped", self.sim_env)
        if self.sim_success_banner_until and time() < self.sim_success_banner_until:
            return  # brief pause showing the completed hold

        action, _ = self.sim_model.predict(self.sim_obs, deterministic=True)
        self.sim_obs, reward, terminated, truncated, info = self.sim_env.step(action)
        self.sim_last_info = info
        self.sim_step_count += 1
        self.sim_episode_reward += float(reward)

        angles = np.asarray(info["joint_angles"], dtype=float)
        velocities_arr = np.asarray(info["joint_velocities"], dtype=float)
        dt = float(base_env.dt)
        self.sim_time += dt
        tau = info.get("actuator_torque")
        if tau is not None:
            tau = np.asarray(tau, dtype=float)
            grav = np.asarray(info.get("gravity_torque", np.full(2, np.nan)), dtype=float)
            self._append_torque_sample(tau[0], tau[1] if tau.shape[0] > 1 else 0.0, self.sim_time,
                                       grav[0], grav[1] if grav.shape[0] > 1 else np.nan)
        else:
            accelerations = (velocities_arr - self.prev_sim_velocities) / max(dt, 1e-8)
            shoulder_tau, elbow_tau = self._estimate_joint_torques(angles, velocities_arr, accelerations)
            self._append_torque_sample(shoulder_tau, elbow_tau, self.sim_time)
        self._append_hand_sample(info, dt)
        self.prev_sim_velocities = velocities_arr.copy()
        self._set_controller_state_from_arrays(angles, velocities_arr)
        self._hand_trail.append(np.asarray(self.controller.positions[-1][:2], dtype=float))

        if info.get("success") and info.get("success_step") == info.get("step"):
            self.sim_successes += 1
            self.sim_success_banner_until = time() + 1.0
            self._log(f"[OK] HOLD COMPLETE at step {info['step']} "
                      f"({info['waypoints_reached']}/{info['num_waypoints']} waypoints)")

        if terminated or truncated:
            if not self.sim_loop:
                self._stop_simulation()
                return
            self._begin_next_episode()

    def _on_reset_arm(self):
        """Reset arm to home position"""
        if self.simulation_active:
            self._stop_simulation()
        self.controller.set_home_position()
        self.trajectory_points = []
        self._hand_trail.clear()
        self._refresh_joint_labels()
        self._log("[OK] Arm reset to home position")

    def _on_reset_defaults(self):
        """Reset to default configuration"""
        if self.simulation_active:
            self._stop_simulation()
        self.config = ArmConfiguration.get_preset(DEFAULT_PRESET)
        self.controller = ArmController(self.config)
        self.visualizer = ArmVisualizer(dof=self.config.dof, link_lengths=self.config.link_lengths)
        self._sync_ui_to_config()  # Update all UI sliders
        self._compute_positions()  # Recompute positions
        self._refresh_joint_labels()
        self._log("[OK] Reset to default configuration")
        messagebox.showinfo("Success", "Reset to default configuration")

    def _on_save_config(self):
        """Save configuration"""
        if self.simulation_active:
            self._stop_simulation()
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.config.to_json(filepath)
                messagebox.showinfo("Success", f"Configuration saved to {filepath}")
                self._log(f"[OK] Configuration saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving config: {e}")

    def _on_load_config(self):
        """Load configuration"""
        if self.simulation_active:
            self._stop_simulation()
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.config = ArmConfiguration.from_json(filepath)
                self.controller = ArmController(self.config)
                self._sync_ui_to_config()  # Update all UI sliders to loaded config
                self._compute_positions()  # Recompute positions
                self._refresh_joint_labels()
                messagebox.showinfo("Success", f"Loaded configuration: {self.config.name}")
                self._log(f"[OK] Loaded configuration: {self.config.name}")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading config: {e}")

    # ============================================================ drawing
    #
    # The arm view is drawn in two layers. The background - silhouette, grid,
    # the green reachable area, targets and placed points, legend - changes only
    # when a target is reached, a point is placed or the arm is resized, and is
    # rendered in full then. The moving parts - the arm, the hand, its path and
    # the title - are drawn over a cached copy of that background every frame.
    # Rebuilding and re-rendering the whole view each frame cost ~50 ms, which
    # held the window to ~7 frames a second.

    def _build_arm_scene(self):
        """Static parts of the arm view; rebuilt only when the arm's size changes."""
        ax = self.ax
        ax.clear()
        reach = float(sum(self.config.link_lengths))
        # Torso / head silhouette (side view, +x in front of the body) and full reach.
        torso_w, torso_h = 0.22 * reach, 0.9 * reach
        ax.add_patch(plt.Rectangle((-torso_w, -torso_h), torso_w, torso_h, color="#00000012", zorder=0))
        ax.add_patch(plt.Circle((-0.5 * torso_w, 0.32 * reach), 0.17 * reach, color="#00000012", zorder=0))
        ax.add_patch(plt.Circle((0.0, 0.0), reach, color="#00000030", fill=False, linestyle="--", linewidth=0.8))
        ax.set_xlabel("X (m)  [+x = in front of the body]", fontsize=9)
        ax.set_ylabel("Y (m)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        lo, hi = self._calculate_axis_limits()
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        self._a_cloud = None
        self._a_cloud_key = None
        self._a_title = getattr(self, "_a_title", None)
        self._a_trail, = ax.plot([], [], "--", color=GOOD, alpha=0.6, linewidth=1.2, label="Hand path", zorder=5)
        self._a_arm, = ax.plot([], [], "-", color=ARM_C, linewidth=6, solid_capstyle="round", zorder=6)
        self._a_joints, = ax.plot([], [], "o", linestyle="none", markersize=8, markerfacecolor="white",
                                  markeredgecolor=ARM_C, markeredgewidth=2, zorder=7)
        self._a_shoulder, = ax.plot([], [], "o", color=GOOD, markersize=12, markeredgecolor="white",
                                    markeredgewidth=1.5, label="Shoulder (base)", zorder=8)
        self._a_hand, = ax.plot([], [], "*", color=BAD, markersize=20, markeredgecolor="white",
                                markeredgewidth=0.8, label="Hand (end-effector)", zorder=9)
        if getattr(self, "_a_title", None) is None:
            self._a_title = self.fig.text(0.012, 0.985, "", ha="left", va="top", fontsize=11,
                                          fontweight="bold", color=INK)
        self._dynamic_artists = [self._a_trail, self._a_arm, self._a_joints, self._a_shoulder, self._a_hand,
                                 self._a_title]
        for artist in self._dynamic_artists:
            artist.set_animated(True)
        self._targets_artists = []
        self._placed_artists = []
        self._static_key = None
        self._dynamic_key = None
        self._arm_bg = None
        self._arm_scene_key = tuple(self.config.link_lengths)

    def _on_arm_resize(self, _event=None):
        """Fixed pixel margins, so tick labels and the axis label fit at any window size."""
        w, h = float(self.fig.bbox.width), float(self.fig.bbox.height)
        if w < 120 or h < 120:
            return
        self.fig.subplots_adjust(left=min(0.4, 62.0 / w), right=1.0 - 14.0 / w,
                                 bottom=min(0.4, 50.0 / h), top=1.0 - 36.0 / h)
        self._arm_bg = None

    def _on_arm_draw(self, _event=None):
        """After every full render: keep the background, then draw the moving parts on it."""
        if not getattr(self, "_dynamic_artists", None):
            return
        self._arm_bg = self.canvas_agg.copy_from_bbox(self.fig.bbox)
        self._arm_bg_bounds = tuple(self.fig.bbox.bounds)
        for artist in self._dynamic_artists:
            self.fig.draw_artist(artist)

    def _rebuild_static_layers(self, show_cloud, info, placed_visible, pending):
        """Green area, placed points, targets and legend."""
        ax = self.ax
        for artist in self._targets_artists + self._placed_artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._targets_artists, self._placed_artists = [], []

        if show_cloud:
            try:
                cloud = self._feasible_cloud_points()
                if self._a_cloud is None or self._a_cloud_key != self._feasible_cloud_key:
                    if self._a_cloud is not None:
                        self._a_cloud.remove()
                    self._a_cloud = ax.scatter(cloud[:, 0], cloud[:, 1], s=6, c="#2ecc71", alpha=0.25, zorder=1,
                                               linewidths=0, label="Reachable & holdable")
                    self._a_cloud_key = self._feasible_cloud_key
                self._a_cloud.set_visible(True)
            except Exception:
                pass
        elif self._a_cloud is not None:
            self._a_cloud.set_visible(False)

        # Points the user has placed (drawn until a run uses them).
        if placed_visible:
            shoulder = np.asarray(self._feasibility_env().shoulder_base_position, dtype=float)
            colour = "#8e44ad" if pending else "red"
            for i, wp in enumerate(self.clicked_waypoints):
                p = np.asarray(wp, dtype=float) - shoulder
                # If the click was moved to somewhere holdable, show where it
                # was asked for and the move, so nothing is relocated silently.
                if i < len(self.clicked_requests):
                    asked, used = (np.asarray(v, dtype=float) for v in self.clicked_requests[i])
                    if float(np.linalg.norm(asked - used)) > 1e-6:
                        a = asked - shoulder
                        self._placed_artists += ax.plot([a[0], p[0]], [a[1], p[1]], color="#888888",
                                                        linestyle=":", linewidth=1.0, zorder=3)
                        self._placed_artists.append(ax.scatter(a[0], a[1], s=60, facecolors="none",
                                                               edgecolors="#888888", linewidths=1.2, zorder=3))
                self._placed_artists.append(ax.add_patch(plt.Circle((p[0], p[1]), 0.05, color=colour, fill=False,
                                                                    linewidth=1.2, zorder=4)))
                self._placed_artists.append(ax.scatter(p[0], p[1], c="white" if pending else colour, s=160,
                                                       marker="X", edgecolors=colour if pending else "black",
                                                       linewidths=1.5, zorder=10))
                self._placed_artists.append(ax.annotate(f"{i + 1}", (p[0], p[1]), textcoords="offset points",
                                                        xytext=(8, 8), fontsize=10, fontweight="bold",
                                                        color=colour if pending else INK))

        if info is not None:
            # Waypoints are in the workspace frame; the plot is shoulder-local.
            shoulder = np.asarray(info.get("shoulder_position", [0.0, 0.0]), dtype=float)
            wps = info.get("waypoints") or [info.get("goal_position")]
            k = int(info.get("current_waypoint_index", 0))
            tol = float(info.get("goal_tolerance", 0.05))
            # A free-order policy picks its own route, so the panel must not
            # draw one: no numbers implying a sequence. Points are grey until
            # the arm has visited them, green afterwards.
            free_order = bool(info.get("free_order"))
            visited = list(info.get("visited") or [])
            # No line is drawn between the waypoints. It would read as a track
            # the arm is being steered along, which is the opposite of what
            # happens: the policy is given a target position and works out the
            # movement itself, and the path it takes is the trail it leaves.
            for i, wp in enumerate(wps):
                p = np.asarray(wp, dtype=float) - shoulder
                if free_order:
                    done = bool(visited[i]) if i < len(visited) else False
                    colour = "green" if done else "lightgray"
                else:
                    colour = "green" if i < k or info.get("success") else ("red" if i == k else "lightgray")
                self._targets_artists.append(ax.add_patch(plt.Circle((p[0], p[1]), tol, color=colour, alpha=0.15,
                                                                     zorder=4)))
                self._targets_artists.append(ax.add_patch(plt.Circle((p[0], p[1]), tol, color=colour, fill=False,
                                                                     linewidth=1.2, zorder=4)))
                self._targets_artists.append(ax.scatter(p[0], p[1], c=colour, s=160, marker="X", edgecolors="black",
                                                        linewidths=1.5, zorder=10))
                if not free_order:
                    self._targets_artists.append(ax.annotate(f"{i + 1}", (p[0], p[1]), textcoords="offset points",
                                                             xytext=(8, 8), fontsize=10, fontweight="bold"))

        handles = [h for h in (self._a_cloud, self._a_trail, self._a_shoulder, self._a_hand)
                   if h is not None and h.get_visible()]
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    def _update_visualization(self):
        """Update the arm visualization"""
        if getattr(self, "_arm_scene_key", None) != tuple(self.config.link_lengths):
            self._build_arm_scene()
        positions = np.asarray(self.controller.positions, dtype=float)
        info = self.sim_last_info if self.simulation_active else None

        # Moving parts.
        self._a_arm.set_data(positions[:, 0], positions[:, 1])
        self._a_joints.set_data(positions[1:-1, 0], positions[1:-1, 1])
        self._a_shoulder.set_data([positions[0, 0]], [positions[0, 1]])
        self._a_hand.set_data([positions[-1, 0]], [positions[-1, 1]])
        # The hand's path: the run's recent trail, or the recorded trajectory.
        trail = None
        if self.show_trajectory:
            if self.simulation_active and len(self._hand_trail) > 1:
                trail = np.asarray(self._hand_trail)
            elif self.trajectory_points:
                trail = np.asarray(self.trajectory_points)[:, :2]
        if trail is not None:
            self._a_trail.set_data(trail[:, 0], trail[:, 1])
        self._a_trail.set_visible(trail is not None)

        # Background layers, rebuilt only when what they show changes.
        show_cloud = bool(self._place_mode or (self.clicked_waypoints and not self.simulation_active))
        placed_visible = bool(self.clicked_waypoints and (not self.simulation_active or self._place_mode))
        pending = bool(self.simulation_active)
        placed_key = (tuple(tuple(w) for w in self.clicked_waypoints),
                      tuple((tuple(a), tuple(u)) for a, u in self.clicked_requests), placed_visible, pending)
        targets_key = None
        if info is not None:
            wps = info.get("waypoints") or [info.get("goal_position")]
            targets_key = (tuple(tuple(np.round(np.asarray(w, dtype=float), 6)) for w in wps),
                           int(info.get("current_waypoint_index", 0)), bool(info.get("success")),
                           float(info.get("goal_tolerance", 0.05)), bool(info.get("free_order")),
                           tuple(bool(v) for v in (info.get("visited") or [])))
        static_key = (show_cloud, self._feasible_cloud_key if show_cloud else None, placed_key, targets_key,
                      trail is not None)
        full = static_key != self._static_key
        if full:
            self._rebuild_static_layers(show_cloud, info, placed_visible, pending)
            self._static_key = static_key

        title = "Arm configuration"
        if placed_visible and not self.simulation_active:
            title = f"{len(self.clicked_waypoints)} point(s) placed - press 'Run with these points'"
        elif self._place_mode and not self.simulation_active:
            title = "Click inside the green area to place a point"
        if info is not None:
            wps = info.get("waypoints") or [info.get("goal_position")]
            n_wp = len(wps)
            k = int(info.get("current_waypoint_index", 0))
            hold_pct = int(round(100.0 * float(info.get("hold_progress", 0.0))))
            reached = int(info.get("waypoints_reached", 0))
            state = "HOLD COMPLETE" if info.get("success") else ("settling" if info.get("in_goal_region") else "moving")
            if info.get("free_order"):
                title = f"{reached}/{n_wp} visited (arm's own order) | hold {hold_pct}% | {state}"
            else:
                title = f"WP {min(k + 1, n_wp)}/{n_wp} | hold {hold_pct}% | {state} | {reached}/{n_wp} reached"
            if self.sim_paused:
                title += "  (paused)"
        self._a_title.set_text(title)

        dynamic_key = (positions.tobytes(), title, None if trail is None else (len(trail), trail[-1].tobytes()))
        stale = self._arm_bg is None or getattr(self, "_arm_bg_bounds", None) != tuple(self.fig.bbox.bounds)
        if full or stale:
            self.canvas_agg.draw()
        elif dynamic_key != self._dynamic_key:
            self.canvas_agg.restore_region(self._arm_bg)
            for artist in self._dynamic_artists:
                self.fig.draw_artist(artist)
            self.canvas_agg.blit(self.fig.bbox)
        self._dynamic_key = dynamic_key

    def _update_progress_strip(self):
        info = self.sim_last_info if self.simulation_active else None
        if info is None:
            self._target_label.config(text="No simulation running")
            self._state_badge.config(text="IDLE", bg=TRACK, fg=MUTED)
            self.sim_hold_bar["value"] = 0
            self._hold_pct_label.config(text="0%")
            last = self.sim_last_info
            if last is not None and self.sim_episode_count:
                n_wp = len(last.get("waypoints") or [last.get("goal_position")])
                self.sim_progress_label.config(
                    text=f"last run: reached {int(last.get('waypoints_reached', 0))}/{n_wp}  |  "
                         f"holds {self.sim_successes}/{max(self.sim_episode_count, 1)}")
            else:
                self.sim_progress_label.config(text="")
            return
        wps = info.get("waypoints") or [info.get("goal_position")]
        n_wp = len(wps)
        k = int(info.get("current_waypoint_index", 0))
        reached = int(info.get("waypoints_reached", 0))
        hold_pct = int(round(100.0 * float(info.get("hold_progress", 0.0))))
        if info.get("free_order"):
            self._target_label.config(text=f"{reached} of {n_wp} visited")
        else:
            self._target_label.config(text=f"Target {min(k + 1, n_wp)} of {n_wp}")
        if info.get("success"):
            self._state_badge.config(text="HOLD COMPLETE ✓", bg="#e2f4ea", fg=GOOD)
        elif info.get("in_goal_region"):
            self._state_badge.config(text="SETTLING", bg="#fdf0db", fg=WARN)
        else:
            self._state_badge.config(text="MOVING", bg="#e3eefb", fg=ACCENT)
        self.sim_hold_bar["value"] = hold_pct
        self._hold_pct_label.config(text=f"{hold_pct}%")
        self.sim_progress_label.config(
            text=f"reached {reached}/{n_wp}  |  episodes {self.sim_successes}/{max(self.sim_episode_count, 1)}")

    def _update_plots(self):
        """Refresh the three live plots from the history buffers."""
        if self.torque_canvas is None:
            return
        have_torque = len(self.torque_time_hist) > 0
        have_hand = len(self.ee_kin_time_hist) > 0
        t = np.asarray(self.torque_time_hist, dtype=float)
        for line, data in zip(self._ln_tau, (self.shoulder_torque_hist, self.elbow_torque_hist)):
            line.set_data(t, np.asarray(data, dtype=float))
        for line, data in zip(self._ln_grav, (self.shoulder_gravity_hist, self.elbow_gravity_hist)):
            line.set_data(t, np.asarray(data, dtype=float))
        th = np.asarray(self.ee_kin_time_hist, dtype=float)
        for line, data in zip(self._ln_vel, (self.ee_vx_hist, self.ee_vy_hist, self.ee_speed_hist)):
            line.set_data(th, np.asarray(data, dtype=float))
        for line, data in zip(self._ln_acc, (self.ee_ax_hist, self.ee_ay_hist)):
            line.set_data(th, np.asarray(data, dtype=float))
        panels = ((self.torque_ax, have_torque, t, self._ln_tau + self._ln_grav),
                  (self.velacc_ax_vel, have_hand, th, self._ln_vel),
                  (self.velacc_ax_acc, have_hand, th, self._ln_acc))
        for (ax, has, times, lines), text in zip(panels, self._plot_empty_text):
            text.set_visible(not has)
            if not has:
                ax.set_xlim(0.0, 10.0)
                ax.set_ylim(-1.0, 1.0)
                continue
            # Keep the recent window readable for long runs.
            end = float(times[-1])
            lo, hi = max(0.0, end - 20.0), max(end, 1.0)
            ax.set_xlim(lo, hi)
            # Limits are set from the data on screen every refresh. (Autoscaling
            # cannot be relied on: setting the empty plot's -1..1 turns it off,
            # which left every trace clipped at +/-1.) Extra room at the top
            # keeps the legend clear of the traces.
            shown = [np.asarray(line.get_ydata(), dtype=float)[times >= lo] for line in lines]
            shown = np.concatenate([v[np.isfinite(v)] for v in shown]) if shown else np.array([])
            if shown.size == 0:
                continue
            y0, y1 = float(shown.min()), float(shown.max())
            span = max(y1 - y0, 0.05 * max(abs(y0), abs(y1)), 0.1)
            ax.set_ylim(y0 - 0.08 * span, y1 + 0.35 * span)
        self.torque_canvas.draw_idle()

    def _update_vel_acc_plot(self):
        """Kept for callers of the old two-panel layout: all plots share one canvas."""
        self._update_plots()

    def _update_torque_plot(self):
        """Kept for callers of the old two-panel layout: all plots share one canvas."""
        self._update_plots()

    def _update_readouts(self):
        """Live-status tiles, muscle bars and the header pill."""
        info = self.sim_last_info if self.simulation_active else None
        if self.simulation_active:
            mode, bg, fg = ("PAUSED", "#fdf0db", WARN) if self.sim_paused else ("RUNNING MODEL", "#e2f4ea", GOOD)
        elif self.recording:
            mode, bg, fg = "RECORDING", "#fbe3e3", BAD
        elif self.playing_back:
            mode, bg, fg = "PLAYBACK", "#fdf0db", WARN
        else:
            mode, bg, fg = "IDLE", TRACK, MUTED
        self._mode_pill.config(text=mode, bg=bg, fg=fg)
        self._fps_label.config(text=f"{self.fps_counter:.0f} fps")

        tiles = self._tiles
        if info is None:
            tiles["state"].config(text="Idle" if not self.simulation_active else "Paused", fg=MUTED)
            for key in ("episode", "time", "target", "reached", "hold", "distance", "speed", "reward"):
                tiles[key].config(text="–", fg=INK)
            tiles["holds"].config(text=f"{self.sim_successes}" if self.sim_episode_count else "–", fg=INK)
        else:
            wps = info.get("waypoints") or [info.get("goal_position")]
            n_wp = len(wps)
            k = int(info.get("current_waypoint_index", 0))
            dt = float(getattr(getattr(self.sim_env, "unwrapped", self.sim_env), "dt", 0.02) or 0.02)
            if info.get("success"):
                tiles["state"].config(text="Hold complete", fg=GOOD)
            elif info.get("in_goal_region"):
                tiles["state"].config(text="Settling", fg=WARN)
            else:
                tiles["state"].config(text="Moving", fg=ACCENT)
            tiles["episode"].config(text=f"{self.sim_episode_count}", fg=INK)
            tiles["time"].config(text=f"{int(info.get('step', 0)) * dt:.2f} s", fg=INK)
            tiles["target"].config(text="own order" if info.get("free_order") else f"{min(k + 1, n_wp)} of {n_wp}",
                                   fg=INK)
            tiles["reached"].config(text=f"{int(info.get('waypoints_reached', 0))} of {n_wp}", fg=INK)
            tiles["hold"].config(text=f"{100 * float(info.get('hold_progress', 0.0)):.0f}%", fg=INK)
            tiles["distance"].config(text=f"{100 * float(info.get('goal_distance', 0.0)):.1f} cm", fg=INK)
            tiles["speed"].config(text=f"{float(info.get('ee_speed', 0.0)):.2f} m/s", fg=INK)
            tiles["reward"].config(text=f"{self.sim_episode_reward:+.1f}", fg=INK)
            tiles["holds"].config(text=f"{self.sim_successes} of {max(self.sim_episode_count, 1)}", fg=INK)

        acts = None
        if info is not None and info.get("muscle_activations") is not None:
            acts = np.asarray(info["muscle_activations"], dtype=float)
        for j, (canvas, rect, pct) in enumerate(self._muscle_rows):
            a = float(acts[j]) if acts is not None and j < acts.shape[0] else 0.0
            width = max(1, canvas.winfo_width())
            canvas.coords(rect, 0, 0, width * min(max(a, 0.0), 1.0), 14)
            pct.config(text=f"{100 * a:.0f}%" if acts is not None else "–")

    def _update_metrics_display(self):
        """Update the 'All readings' text."""
        pos = self.controller.get_end_effector_position()
        info = self.sim_last_info
        angles = self.controller.angles
        wp_line = (f"{(int(info.get('current_waypoint_index', 0)) + 1)}/{int(info.get('num_waypoints', 0))}"
                   if info else "-/-")
        tau_note = "muscle torque" if len(self.torque_time_hist) and not np.isnan(
            self.shoulder_gravity_hist[-1]) else "inverse-dynamics estimate"

        metrics_text = f"""CURRENT STATE
{'─' * 34}
Hand position (m, from shoulder):
  ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})
Joint angles (rad):  {', '.join(f'{a:.3f}' for a in angles)}
Joint angles (deg):  {', '.join(f'{np.degrees(a):.1f}' for a in angles)}
Joint velocities (rad/s):
  {', '.join(f'{v:.3f}' for v in self.controller.velocities)}

RECORDING
{'─' * 34}
Status: {'REC' if self.recording else 'STOPPED'}
Frames: {self.recorder.get_num_frames()}
Config: {self.config.name}

PLAYBACK
{'─' * 34}
Status: {'PLAYING' if self.playing_back else 'IDLE'}
Progress: {self.playback_index}/{len(self.playback_frames)}

SIMULATION
{'─' * 34}
Status: {'PAUSED' if self.sim_paused else ('RUNNING' if self.simulation_active else 'IDLE')}
Model: {Path(self.sim_model_path).name if self.sim_model_path else 'None'}
Kind: {self.sim_model_kind}
Algo: {self.sim_algorithm if self.sim_algorithm else '-'}
Episode: {self.sim_episode_count}
Steps: {self.sim_step_count}
Sim time: {self.sim_time:.2f} s
Ep Reward: {self.sim_episode_reward:.3f}
Shoulder τ: {self.latest_shoulder_torque:.3f} Nm ({tau_note})
Elbow τ: {self.latest_elbow_torque:.3f} Nm
Hand speed: {float(info.get('ee_speed', 0.0)) if info else 0.0:.3f} m/s
Distance to target: {float(info.get('goal_distance', 0.0)) if info else 0.0:.3f} m
Waypoint: {wp_line}
Hold: {int(round(100 * float(info.get('hold_progress', 0.0)))) if info else 0}%  Success: {bool(info.get('success')) if info else False}
Holds completed: {self.sim_successes}

PERFORMANCE
{'─' * 34}
FPS: {self.fps_counter:.1f}
Speed: {self.sim_speed_multiplier:.1f}x real time
Trajectory Pts: {len(self.trajectory_points)}

TRAJECTORY
{'─' * 34}
Show: {'ON' if self.show_trajectory else 'OFF'}
Points: {len(self.trajectory_points)}
"""
        first = self.metrics_text.yview()[0]
        self.metrics_text.config(state="normal")
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, metrics_text)
        self.metrics_text.config(state="disabled")
        self.metrics_text.yview_moveto(first)

    # ============================================================== loop

    def _handle_keyboard(self, event):
        """Handle keyboard events (ignored while typing in a text box)."""
        focus = self.root.focus_get()
        if isinstance(focus, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text, tk.Spinbox)):
            return
        # Joint selection with Left/Right
        if event.keysym == "Left":
            self.selected_joint = max(0, self.selected_joint - 1)
            self._log(f"[OK] Selected joint J{self.selected_joint + 1}")
        elif event.keysym == "Right":
            self.selected_joint = min(self.config.dof - 1, self.selected_joint + 1)
            self._log(f"[OK] Selected joint J{self.selected_joint + 1}")
        # Joint control with Up/Down on selected joint
        elif event.keysym == "Up":
            self._increment_joint(self.selected_joint, 0.05)
        elif event.keysym == "Down":
            self._increment_joint(self.selected_joint, -0.05)
        elif event.char in ("t", "T"):
            # Toggle trajectory
            self.show_trajectory = not self.show_trajectory
            if hasattr(self, "_trail_var"):
                self._trail_var.set(self.show_trajectory)
            self._log(f"[OK] Trajectory {'ON' if self.show_trajectory else 'OFF'}")

    def _default_model(self):
        """The deployed model (R3), if it is on disk."""
        for label, model in self._package_choices.items():
            if label.startswith("R3"):
                return model
        return next(iter(self._package_choices.values()), None)

    def run(self):
        """Run the GUI event loop"""
        self.create_window()

        # Bind keyboard events
        self.root.bind("<Up>", self._handle_keyboard)
        self.root.bind("<Down>", self._handle_keyboard)
        self.root.bind("<Left>", self._handle_keyboard)
        self.root.bind("<Right>", self._handle_keyboard)
        self.root.bind("<t>", self._handle_keyboard)
        self.root.bind("<T>", self._handle_keyboard)

        # Model given on the command line, else preselect the deployed one.
        initial = self._initial_model or (None if self.sim_model_path else self._default_model())
        if initial is not None:
            self._load_model(str(initial))
        elif self.sim_model_path:
            self._load_model(self.sim_model_path + ".zip")
        if self.sim_waypoint_choice == "custom" and self.sim_custom_waypoints:
            self._wp_custom_var.set(self.sim_custom_waypoints)
            self._on_custom_committed()
        self._wp_choice_var.set(self.sim_waypoint_choice)
        self._loop_var.set(self.sim_loop)
        self._refresh_points_hint()
        if self.sim_model_path and self._pending_autostart:
            self.root.after(300, self._start_simulation)

        self._last_step_wall = None
        self._last_slow_update = 0.0

        def update_loop():
            """Main update loop"""
            if not self.running:
                return
            tick_start = time()

            # Advance the simulation by the real time that has passed, so
            # "1.0x real time" is true whatever the frame rate. (Stepping a
            # fixed count per frame tied the arm's speed to the frame rate -
            # 0.3x at 7 frames a second - and rounded 2.5 steps a frame to 2.)
            if self.simulation_active and not self.sim_paused:
                base = getattr(self.sim_env, "unwrapped", self.sim_env)
                dt = float(getattr(base, "dt", 0.02) or 0.02)
                if self._last_step_wall is not None:
                    elapsed = min(tick_start - self._last_step_wall, 0.25)
                    self.sim_step_accumulator += elapsed * self.sim_speed_multiplier / dt
                self._last_step_wall = tick_start
                n_steps = int(self.sim_step_accumulator)
                self.sim_step_accumulator -= n_steps
                for _ in range(min(n_steps, 60)):
                    self._simulation_step()
                    if not self.simulation_active:
                        break
            else:
                self._last_step_wall = None
            if self.playing_back and self.playback_frames and not self.simulation_active:
                if self.playback_index < len(self.playback_frames):
                    frame_state = self.playback_frames[self.playback_index]
                    self.controller.apply_state(frame_state)
                    self._refresh_joint_labels()
                    self.playback_index += 1
                else:
                    self.playing_back = False
                    self.playback_button.config(text="Playback")
                    self._log("[OK] Playback finished")

            # Record current frame if recording
            if self.recording:
                state = self.controller.get_state(timestamp=time())
                self.recorder.record_frame(state)

                # Update trajectory
                ee_pos = self.controller.get_end_effector_position()
                self.trajectory_points.append(ee_pos)

            # Update visualizations.
            #
            # Redrawing every matplotlib canvas every tick costs a whole core
            # and starves the Tk event loop, so the window stops answering
            # clicks and Windows marks it "not responding". The arm view only
            # repaints what moved (see _update_visualization), so it keeps up
            # with motion every frame; the three plots and the readouts are
            # readable at 4 Hz, and when nothing is moving a 1 Hz refresh keeps
            # them current (including the final trace after a run ends).
            moving = bool(self.simulation_active or self.playing_back or self.recording)
            self._update_visualization()
            self._update_progress_strip()
            if tick_start - self._last_slow_update >= (0.25 if moving else 1.0):
                self._last_slow_update = tick_start
                self._update_readouts()
                self._update_metrics_display()
                self._update_plots()

            # Update FPS counter
            current_time = time()
            if current_time - self.last_frame_time >= 1.0:
                self.fps_counter = self.frame_count
                self.last_frame_time = current_time
                self.frame_count = 0
            else:
                self.frame_count += 1

            # Next frame: aim for 25 frames a second.
            spent_ms = 1000.0 * (time() - tick_start)
            self.root.after(max(5, int(40 - spent_ms)), update_loop)

        # Start update loop
        update_loop()

        # Run Tkinter event loop
        self.root.mainloop()

    def on_closing(self):
        """Handle window closing"""
        self.running = False
        if self.simulation_active:
            self._stop_simulation()
        self.root.destroy()
        # Force-exit to prevent PyTorch/OpenMP GIL crash during interpreter cleanup
        os._exit(0)


def main(argv=None):
    """Entry point for GUI application.

    python -m rl_armMotion.two_d.gui.app [--package DIR|MODEL.zip] [--which best] [--waypoints 2|3|spread] [--autostart]
    """
    import argparse
    parser = argparse.ArgumentParser(description="Interactive arm GUI")
    parser.add_argument("--package", default=None, help="model package directory or a model .zip inside one")
    parser.add_argument("--which", default="best", help="best | last | final")
    parser.add_argument("--waypoints", default=None, help="1 | 2 | 3 | spread | 'x,y;x,y' (default: the package's set)")
    parser.add_argument("--autostart", action="store_true", help="start the policy simulation immediately")
    parser.add_argument("--no-loop", action="store_true")
    args = parser.parse_args(argv)

    config = ArmConfiguration.get_preset(DEFAULT_PRESET)
    gui = ArmControllerGUI(config)
    if args.package:
        pkg = Path(args.package)
        model_zip = pkg if pkg.is_file() else pkg / f"sac_model_{args.which}.zip"
        if not model_zip.exists():
            model_zip = pkg / "sac_model.zip"
        gui._initial_model = model_zip
        if args.waypoints:
            w = args.waypoints.strip().lower()
            gui.sim_waypoint_choice = {"1": "1 waypoint", "2": "2 waypoints", "3": "3 waypoints", "spread": "spread"}.get(w, "custom")
            if gui.sim_waypoint_choice == "custom":
                gui.sim_custom_waypoints = args.waypoints
        gui.sim_loop = not args.no_loop
        gui._pending_autostart = bool(args.autostart)
    gui.run()


if __name__ == "__main__":
    main()
