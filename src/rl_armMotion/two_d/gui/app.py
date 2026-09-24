"""Interactive arm controller GUI using Tkinter and Matplotlib (fully open-source)"""

import os
import pickle
import sys
from collections import deque
from pathlib import Path

# Ensure src directory is in path for imports.
# File lives in `.../src/rl_armMotion/two_d/gui/app.py`, so parents[3] is `src/`.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from time import time

from rl_armMotion.two_d.config import ArmConfiguration
from rl_armMotion.two_d.environments.task_env import ArmTaskEnv, ActionSmoother, DEFAULT_PRESET, OBS_VERSION
from rl_armMotion.two_d.gui.waypoint_editing import index_near, nearest_placeable
from rl_armMotion.two_d.evaluation import (
    CANONICAL_WAYPOINTS, EVAL_STANDARD, WAYPOINTS_SPREAD, detect_actuation_mode, find_manifest,
    load_package, make_env, read_manifest, resolve_waypoints,
)
from rl_armMotion.two_d.models.trainers import RLTrainer
from rl_armMotion.two_d.utils import ArmKinematics, ArmController, MotionRecorder, ArmVisualizer, dynamics


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
        self.sim_env = None
        self.sim_obs = None
        self.sim_model = None
        self.sim_algorithm = None
        self.sim_model_path = ""
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

        # Torque history for live plotting
        self.torque_time_hist = deque(maxlen=2000)
        self.shoulder_torque_hist = deque(maxlen=2000)
        self.elbow_torque_hist = deque(maxlen=2000)

        # End-effector velocity/acceleration history for live plotting
        self.ee_kin_time_hist = deque(maxlen=2000)
        self.ee_vx_hist = deque(maxlen=2000)
        self.ee_vy_hist = deque(maxlen=2000)
        self.ee_ax_hist = deque(maxlen=2000)
        self.ee_ay_hist = deque(maxlen=2000)
        self._ee_prev_pos_xy = None
        self._ee_prev_vel_xy = np.zeros(2, dtype=float)
        self._ee_prev_time = None
        self._ee_plot_time = 0.0

        # Create main window
        self.root = tk.Tk()
        self.root.title("RL Arm Motion - Interactive Controller")
        self.root.geometry("1200x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

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

    def _set_model_details_text(self, text: str):
        """Update model-details text box in visualization panel."""
        if self.sim_model_details_text is None:
            return
        self.sim_model_details_text.config(state="normal")
        self.sim_model_details_text.delete(1.0, tk.END)
        self.sim_model_details_text.insert(1.0, text)
        self.sim_model_details_text.config(state="disabled")

    def _set_controller_state_from_arrays(self, angles, velocities):
        """Apply angle/velocity arrays to controller and refresh labels."""
        self.controller.angles = np.asarray(angles, dtype=float).copy()
        self.controller.velocities = np.asarray(velocities, dtype=float).copy()
        self._compute_positions()
        for i in range(self.config.dof):
            self.angle_labels[i].config(text=f"{self.controller.angles[i]:.2f}rad")

    def _estimate_joint_torques(self, angles, velocities, accelerations):
        """Joint torques from the arm's rigid-body inverse dynamics
        (M(q) q'' + C(q, q') q' + G(q)), i.e. what the actuators must supply."""
        q = np.asarray(angles, dtype=float)
        qd = np.asarray(velocities, dtype=float)
        qdd = np.asarray(accelerations, dtype=float)
        try:
            tau = dynamics.inverse_dynamics(q, qd, qdd, self.config)
        except Exception:
            tau = np.zeros(max(q.shape[0], 2))
        return float(tau[0]), float(tau[1]) if tau.shape[0] > 1 else 0.0

    def _append_torque_sample(self, shoulder_torque, elbow_torque, current_time):
        """Append a new torque sample to history buffers."""
        self.torque_time_hist.append(float(current_time))
        self.shoulder_torque_hist.append(float(shoulder_torque))
        self.elbow_torque_hist.append(float(elbow_torque))
        self.latest_shoulder_torque = float(shoulder_torque)
        self.latest_elbow_torque = float(elbow_torque)

    def _update_torque_plot(self):
        """Redraw torque plot with latest history."""
        if self.torque_ax is None or self.torque_canvas is None:
            return

        self.torque_ax.clear()
        if self.torque_time_hist:
            t = np.array(self.torque_time_hist, dtype=float)
            t = t - t[0]
            shoulder = np.array(self.shoulder_torque_hist, dtype=float)
            elbow = np.array(self.elbow_torque_hist, dtype=float)

            self.torque_ax.plot(t, shoulder, linewidth=1.4, label="Shoulder Torque")
            self.torque_ax.plot(t, elbow, linewidth=1.4, label="Elbow Torque")
            self.torque_ax.legend(loc="upper right", fontsize=8)

            # Keep recent window readable for long runs.
            if t[-1] > 20.0:
                self.torque_ax.set_xlim(t[-1] - 20.0, t[-1])

        self.torque_ax.set_title("Shoulder/Elbow Torque")
        self.torque_ax.set_xlabel("Time (s)")
        self.torque_ax.set_ylabel("Torque (Nm)")
        self.torque_ax.grid(True, alpha=0.3)
        self.torque_canvas.draw_idle()

    def _reset_ee_kinematics(self):
        """Reset end-effector velocity/acceleration histories."""
        self.ee_kin_time_hist.clear()
        self.ee_vx_hist.clear()
        self.ee_vy_hist.clear()
        self.ee_ax_hist.clear()
        self.ee_ay_hist.clear()
        self._ee_prev_pos_xy = None
        self._ee_prev_vel_xy = np.zeros(2, dtype=float)
        self._ee_prev_time = None
        self._ee_plot_time = 0.0

    def _append_ee_kinematics_sample(self, now_s: float):
        """Append one velocity/acceleration sample from current end-effector pose."""
        ee_pos = np.asarray(self.controller.get_end_effector_position(), dtype=float)
        pos_xy = ee_pos[:2]

        if self._ee_prev_time is None or self._ee_prev_pos_xy is None:
            vx, vy, ax, ay = 0.0, 0.0, 0.0, 0.0
            dt = 0.0
        else:
            dt = max(now_s - float(self._ee_prev_time), 1e-8)
            vel_xy = (pos_xy - self._ee_prev_pos_xy) / dt
            acc_xy = (vel_xy - self._ee_prev_vel_xy) / dt
            vx, vy = float(vel_xy[0]), float(vel_xy[1])
            ax, ay = float(acc_xy[0]), float(acc_xy[1])
            self._ee_prev_vel_xy = vel_xy

        self._ee_plot_time += dt
        self.ee_kin_time_hist.append(float(self._ee_plot_time))
        self.ee_vx_hist.append(vx)
        self.ee_vy_hist.append(vy)
        self.ee_ax_hist.append(ax)
        self.ee_ay_hist.append(ay)
        self._ee_prev_pos_xy = pos_xy.copy()
        self._ee_prev_time = float(now_s)

    def _update_vel_acc_plot(self):
        """Redraw end-effector velocity/acceleration subplot."""
        if (
            self.velacc_ax_vel is None
            or self.velacc_ax_acc is None
            or self.velacc_canvas is None
        ):
            return

        self.velacc_ax_vel.clear()
        self.velacc_ax_acc.clear()

        if not self.simulation_active:
            self.velacc_ax_vel.set_title("End-Effector Velocity")
            self.velacc_ax_vel.set_ylabel("m/s")
            self.velacc_ax_vel.grid(True, alpha=0.3)
            self.velacc_ax_vel.text(
                0.5, 0.5, "Simulation not running",
                ha="center", va="center", transform=self.velacc_ax_vel.transAxes, fontsize=8
            )

            self.velacc_ax_acc.set_title("End-Effector Acceleration")
            self.velacc_ax_acc.set_xlabel("Time (s)")
            self.velacc_ax_acc.set_ylabel("m/s^2")
            self.velacc_ax_acc.grid(True, alpha=0.3)
            self.velacc_canvas.draw_idle()
            return

        if self.ee_kin_time_hist:
            t = np.asarray(self.ee_kin_time_hist, dtype=float)
            vx = np.asarray(self.ee_vx_hist, dtype=float)
            vy = np.asarray(self.ee_vy_hist, dtype=float)
            ax = np.asarray(self.ee_ax_hist, dtype=float)
            ay = np.asarray(self.ee_ay_hist, dtype=float)

            self.velacc_ax_vel.plot(t, vx, linewidth=1.2, label="Vx")
            self.velacc_ax_vel.plot(t, vy, linewidth=1.2, label="Vy")
            self.velacc_ax_vel.legend(loc="upper right", fontsize=7)

            self.velacc_ax_acc.plot(t, ax, linewidth=1.2, label="Ax")
            self.velacc_ax_acc.plot(t, ay, linewidth=1.2, label="Ay")
            self.velacc_ax_acc.legend(loc="upper right", fontsize=7)

            if t[-1] > 20.0:
                self.velacc_ax_vel.set_xlim(t[-1] - 20.0, t[-1])
                self.velacc_ax_acc.set_xlim(t[-1] - 20.0, t[-1])

        self.velacc_ax_vel.set_title("End-Effector Velocity")
        self.velacc_ax_vel.set_ylabel("m/s")
        self.velacc_ax_vel.grid(True, alpha=0.3)

        self.velacc_ax_acc.set_title("End-Effector Acceleration")
        self.velacc_ax_acc.set_xlabel("Time (s)")
        self.velacc_ax_acc.set_ylabel("m/s^2")
        self.velacc_ax_acc.grid(True, alpha=0.3)

        self.velacc_canvas.draw_idle()

    def _create_properties_frame(self, parent):
        """Create left frame with property sliders"""
        frame = ttk.LabelFrame(parent, text="ARM PROPERTIES", padding=10)

        # Link length sliders
        ttk.Label(frame, text="Link Lengths (m):", font=("Arial", 10, "bold")).pack(fill="x", pady=(0, 5))
        for i in range(self.config.dof):
            # Create a row for each link
            row_frame = ttk.Frame(frame)
            row_frame.pack(fill="x", pady=2)
            row_frame.columnconfigure(1, weight=5)
            row_frame.columnconfigure(3, weight=2)

            ttk.Label(row_frame, text=f"Link {i+1}:", width=10).grid(row=0, column=0, sticky="w")
            var = tk.DoubleVar(value=self.config.link_lengths[i])
            self.slider_vars[f"link_len_{i}"] = var
            slider = ttk.Scale(
                row_frame,
                from_=0.1,
                to=2.0,
                variable=var,
                orient="horizontal",
                command=lambda val, idx=i: self._on_link_length_change(idx, val)
            )
            slider.grid(row=0, column=1, sticky="ew", padx=5)

            val_label = ttk.Label(row_frame, text=f"{var.get():.2f}", width=8)
            val_label.grid(row=0, column=2, sticky="e")
            self.slider_vars[f"link_len_{i}_label"] = val_label

            entry_var = tk.StringVar(value=f"{var.get():.2f}")
            self.slider_vars[f"link_len_{i}_entry"] = entry_var
            entry = ttk.Entry(row_frame, textvariable=entry_var, width=8)
            entry.grid(row=0, column=3, sticky="ew", padx=(4, 0))
            entry.bind("<Return>", lambda _e, idx=i: self._on_link_length_entry(idx))
            entry.bind("<FocusOut>", lambda _e, idx=i: self._on_link_length_entry(idx))

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=10)

        # Mass sliders (first 3 for brevity)
        ttk.Label(frame, text="Masses (kg) - First 3:", font=("Arial", 10, "bold")).pack(fill="x", pady=(0, 5))
        for i in range(min(3, self.config.dof)):
            row_frame = ttk.Frame(frame)
            row_frame.pack(fill="x", pady=2)
            row_frame.columnconfigure(1, weight=5)
            row_frame.columnconfigure(3, weight=2)

            ttk.Label(row_frame, text=f"Mass {i+1}:", width=10).grid(row=0, column=0, sticky="w")
            var = tk.DoubleVar(value=self.config.masses[i])
            self.slider_vars[f"mass_{i}"] = var
            slider = ttk.Scale(
                row_frame,
                from_=0.1,
                to=10.0,
                variable=var,
                orient="horizontal",
                command=lambda val, idx=i: self._on_mass_change(idx, val)
            )
            slider.grid(row=0, column=1, sticky="ew", padx=5)

            val_label = ttk.Label(row_frame, text=f"{var.get():.2f}", width=8)
            val_label.grid(row=0, column=2, sticky="e")
            self.slider_vars[f"mass_{i}_label"] = val_label

            entry_var = tk.StringVar(value=f"{var.get():.2f}")
            self.slider_vars[f"mass_{i}_entry"] = entry_var
            entry = ttk.Entry(row_frame, textvariable=entry_var, width=8)
            entry.grid(row=0, column=3, sticky="ew", padx=(4, 0))
            entry.bind("<Return>", lambda _e, idx=i: self._on_mass_entry(idx))
            entry.bind("<FocusOut>", lambda _e, idx=i: self._on_mass_entry(idx))

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=10)

        # Damping slider
        ttk.Label(frame, text="Damping:", font=("Arial", 10, "bold")).pack(fill="x", pady=(0, 5))
        damping_row = ttk.Frame(frame)
        damping_row.pack(fill="x", pady=2)

        damping_var = tk.DoubleVar(value=self.config.damping)
        self.slider_vars["damping"] = damping_var
        slider = ttk.Scale(
            damping_row,
            from_=0.0,
            to=1.0,
            variable=damping_var,
            orient="horizontal",
            command=self._on_damping_change
        )
        slider.pack(side="left", fill="x", expand=True, padx=5)

        damping_label = ttk.Label(damping_row, text=f"{damping_var.get():.2f}", width=8)
        damping_label.pack(side="left")
        self.slider_vars["damping_label"] = damping_label

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=10)

        # Action buttons in one row, equal width.
        button_row = ttk.Frame(frame)
        button_row.pack(fill="x", pady=5)
        for col in range(3):
            button_row.columnconfigure(col, weight=1, uniform="props_btns")

        ttk.Button(button_row, text="Reset", command=self._on_reset_defaults, width=10).grid(
            row=0, column=0, sticky="ew", padx=2
        )
        ttk.Button(button_row, text="Save", command=self._on_save_config, width=10).grid(
            row=0, column=1, sticky="ew", padx=2
        )
        ttk.Button(button_row, text="Load", command=self._on_load_config, width=10).grid(
            row=0, column=2, sticky="ew", padx=2
        )

        return frame

    def _create_control_frame(self, parent):
        """Create joint control frame"""
        frame = ttk.LabelFrame(parent, text="JOINT CONTROL", padding=10)

        info_text = ttk.Label(
            frame,
            text="Use arrow keys or buttons to move joints\nUp/Down: change joint value\nLeft/Right: select joint",
            justify="left",
            font=("Arial", 9)
        )
        info_text.pack(fill="x", pady=(0, 10))

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=5)

        # Joint controls
        for i in range(self.config.dof):
            joint_row = ttk.Frame(frame)
            joint_row.pack(fill="x", pady=3)

            ttk.Label(joint_row, text=f"J{i+1}:", font=("Arial", 10, "bold"), width=5).pack(side="left", padx=2)
            ttk.Button(joint_row, text="-", width=3, command=lambda idx=i: self._increment_joint(idx, -0.05)).pack(side="left", padx=2)

            angle_label = ttk.Label(joint_row, text=f"{self.controller.angles[i]:.2f}rad", width=12, font=("Courier", 10))
            angle_label.pack(side="left", padx=5)
            self.angle_labels[i] = angle_label

            ttk.Button(joint_row, text="+", width=3, command=lambda idx=i: self._increment_joint(idx, +0.05)).pack(side="left", padx=2)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=10)

        # All action buttons in one row, equal width.
        actions_row = ttk.Frame(frame)
        actions_row.pack(fill="x", pady=5)
        for col in range(6):
            actions_row.columnconfigure(col, weight=1, uniform="joint_btns")

        self.record_button = ttk.Button(actions_row, text="Record", command=self._on_record, width=9)
        self.record_button.grid(row=0, column=0, sticky="ew", padx=1)
        ttk.Button(actions_row, text="Clear", command=self._on_clear_recording, width=9).grid(
            row=0, column=1, sticky="ew", padx=1
        )
        self.playback_button = ttk.Button(actions_row, text="Playback", command=self._on_playback, width=9)
        self.playback_button.grid(row=0, column=2, sticky="ew", padx=1)
        ttk.Button(actions_row, text="Save", command=self._on_save_motion, width=9).grid(
            row=0, column=3, sticky="ew", padx=1
        )
        ttk.Button(actions_row, text="Load", command=self._on_load_motion, width=9).grid(
            row=0, column=4, sticky="ew", padx=1
        )
        ttk.Button(actions_row, text="Reset", command=self._on_reset_arm, width=9).grid(
            row=0, column=5, sticky="ew", padx=1
        )

        return frame

    def _create_metrics_frame(self, parent):
        """Create metrics display frame"""
        frame = ttk.LabelFrame(parent, text="METRICS & STATUS", padding=10)

        self.metrics_text = tk.Text(
            frame,
            height=20,
            width=35,
            font=("Courier", 9),
            bg="black",
            fg="white",
            state="disabled"
        )
        self.metrics_text.pack(fill="both", expand=True)

        return frame

    def _scrollable_body(self, parent):
        """A scrollable area inside ``parent``.

        The simulation controls live in a pane of a Panedwindow, so when that
        pane is short everything below the fold is simply cut off - which is how
        the Plot/Remove buttons became unreachable, with the Loop checkbox
        visibly clipped to "[x] L" on the row above them. Putting the rows in a
        scrolled canvas means no control can be lost to a small window.
        """
        canvas = tk.Canvas(parent, highlightthickness=0, borderwidth=0)
        bar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")

        def _resize_region(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _match_width(event):
            canvas.itemconfigure(window, width=event.width)

        def _wheel(event):
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        body.bind("<Configure>", _resize_region)
        canvas.bind("<Configure>", _match_width)
        for widget in (canvas, body):
            widget.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _wheel))
            widget.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        return body
    def _create_model_selection_frame(self, parent):
        """Create arm-model selection frame."""
        frame = ttk.LabelFrame(parent, text="ARM MODEL SELECTION", padding=5)
        body = self._scrollable_body(frame)

        control_row = ttk.Frame(body)
        control_row.pack(fill="x", pady=(0, 6))
        ttk.Button(control_row, text="Load Model", command=self._on_select_model).pack(
            side="left", padx=(0, 4), fill="x"
        )
        self.sim_toggle_button = ttk.Button(
            control_row,
            text="Run Simulation",
            command=self._on_toggle_simulation,
            state="disabled",
        )
        self.sim_toggle_button.pack(side="left", padx=(0, 4), fill="x")

        self.sim_model_label = ttk.Label(
            control_row,
            text="Model: not selected",
            font=("Courier", 8),
            anchor="w",
        )
        self.sim_model_label.pack(side="left", fill="x", expand=True, padx=(6, 0))

        # Simulation speed control
        # Waypoint demo controls
        wp_row = ttk.Frame(body)
        wp_row.pack(fill="x", pady=(2, 4))
        ttk.Label(wp_row, text="Waypoints:", font=("Courier", 8)).pack(side="left", padx=(0, 4))
        self._wp_choice_var = tk.StringVar(value=self.sim_waypoint_choice)
        wp_box = ttk.Combobox(
            wp_row, textvariable=self._wp_choice_var, state="readonly", width=13,
            values=["from package", "1 waypoint", "2 waypoints", "3 waypoints", "spread", "custom"],
        )
        wp_box.pack(side="left")
        wp_box.bind("<<ComboboxSelected>>", self._on_waypoint_choice)
        self._wp_custom_var = tk.StringVar(value="")
        custom_entry = ttk.Entry(wp_row, textvariable=self._wp_custom_var, width=22, font=("Courier", 8))
        custom_entry.pack(side="left", padx=(4, 0))
        custom_entry.bind("<KeyRelease>", lambda e: setattr(self, "sim_custom_waypoints", self._wp_custom_var.get()))
        self._loop_var = tk.BooleanVar(value=self.sim_loop)
        ttk.Checkbutton(wp_row, text="Loop", variable=self._loop_var,
                        command=lambda: setattr(self, "sim_loop", bool(self._loop_var.get()))).pack(side="left", padx=(6, 0))

        # Adding and removing are two named buttons, not one checkbox plus a
        # right-click nobody can discover. They latch, so the panel is either in
        # "the next click adds a point" or "the next click deletes one" and the
        # button that is pressed in says which. Picking "custom" above turns
        # adding on, because that is the only reason to pick it.
        place_row = ttk.Frame(body)
        place_row.pack(fill="x", pady=(6, 0))
        self._click_mode_var = tk.StringVar(value="off")
        self._add_btn = ttk.Radiobutton(place_row, text="+ Add point", value="add", width=12,
                                        variable=self._click_mode_var, style="Toolbutton",
                                        command=self._on_click_mode)
        self._add_btn.pack(side="left")
        self._del_btn = ttk.Radiobutton(place_row, text="- Remove point", value="remove", width=15,
                                        variable=self._click_mode_var, style="Toolbutton",
                                        command=self._on_click_mode)
        self._del_btn.pack(side="left", padx=(4, 0))
        ttk.Radiobutton(place_row, text="Done", value="off", width=6,
                        variable=self._click_mode_var, style="Toolbutton",
                        command=self._on_click_mode).pack(side="left", padx=(4, 0))
        # Kept so anything that toggled placement by the old name still works.
        self._place_var = tk.BooleanVar(value=False)

        place_row2 = ttk.Frame(body)
        place_row2.pack(fill="x", pady=(0, 4))
        self._place_hint = ttk.Label(place_row2, text="press '+ Plot points', then click on the arm panel",
                                     font=("Courier", 7), foreground="#666666")
        self._place_hint.pack(side="left", padx=(18, 6))
        ttk.Button(place_row2, text="Go", width=5,
                   command=self._on_restart_with_points).pack(side="left")
        ttk.Button(place_row2, text="Undo", width=6,
                   command=self._undo_last_point).pack(side="left", padx=(4, 0))
        ttk.Button(place_row2, text="Clear", width=6,
                   command=self._on_clear_points).pack(side="left", padx=(4, 0))

        speed_row = ttk.Frame(body)
        speed_row.pack(fill="x", pady=(2, 4))
        ttk.Label(speed_row, text="Sim Speed:", font=("Courier", 8)).pack(side="left", padx=(0, 4))
        self._sim_speed_var = tk.IntVar(value=5)
        speed_slider = ttk.Scale(
            speed_row, from_=1, to=20, orient="horizontal",
            variable=self._sim_speed_var, length=160,
        )
        speed_slider.pack(side="left")
        self._sim_speed_label = ttk.Label(speed_row, text="1.0x", font=("Courier", 8), width=6)
        self._sim_speed_label.pack(side="left", padx=(4, 0))

        def _on_speed_change(event=None):
            steps = int(self._sim_speed_var.get())
            self.sim_speed_multiplier = steps / 5.0  # slider 5 = real time
            self._sim_speed_label.config(text=f"{self.sim_speed_multiplier:.1f}x")

        speed_slider.bind("<Motion>", _on_speed_change)
        speed_slider.bind("<ButtonRelease-1>", _on_speed_change)

        # Action smoothing control
        smooth_row = ttk.Frame(body)
        smooth_row.pack(fill="x", pady=(2, 4))
        self._smooth_var = tk.BooleanVar(value=self.action_smoothing)
        smooth_check = ttk.Checkbutton(
            smooth_row, text="Smooth motion",
            variable=self._smooth_var,
            command=self._on_smooth_toggle,
        )
        smooth_check.pack(side="left", padx=(0, 8))
        ttk.Label(smooth_row, text="α:", font=("Courier", 8)).pack(side="left")
        self._smooth_alpha_var = tk.DoubleVar(value=self.action_smooth_alpha)
        alpha_slider = ttk.Scale(
            smooth_row, from_=0.1, to=0.9, orient="horizontal",
            variable=self._smooth_alpha_var, length=100,
        )
        alpha_slider.pack(side="left")
        self._smooth_alpha_label = ttk.Label(smooth_row, text="0.3", font=("Courier", 8), width=4)
        self._smooth_alpha_label.pack(side="left", padx=(4, 0))

        def _on_alpha_change(event=None):
            val = round(self._smooth_alpha_var.get(), 2)
            self.action_smooth_alpha = val
            self._smooth_alpha_label.config(text=f"{val:.2f}")

        alpha_slider.bind("<Motion>", _on_alpha_change)
        alpha_slider.bind("<ButtonRelease-1>", _on_alpha_change)

        hold_row = ttk.Frame(body)
        hold_row.pack(fill="x", pady=(2, 4))
        ttk.Label(hold_row, text="Hold:", font=("Courier", 8)).pack(side="left", padx=(0, 4))
        self.sim_hold_bar = ttk.Progressbar(hold_row, orient="horizontal", length=160, mode="determinate", maximum=100)
        self.sim_hold_bar.pack(side="left")
        self.sim_progress_label = ttk.Label(hold_row, text="reached 0/0  |  episodes 0/0", font=("Courier", 8))
        self.sim_progress_label.pack(side="left", padx=(8, 0))

        details_frame = ttk.LabelFrame(body, text="Model Details", padding=4)
        details_frame.pack(fill="x", pady=(0, 6))
        self.sim_model_details_text = tk.Text(
            details_frame,
            height=3,
            font=("Courier", 8),
            wrap="word",
            state="disabled",
        )
        self.sim_model_details_text.pack(fill="x", expand=False)
        self._set_model_details_text(
            "Name: -\nAlgorithm: -\nPolicy: -\nPath: -\nStatus: Select a trained model to view details"
        )

        return frame

    def _create_torque_frame(self, parent):
        """Create left-side torque plotting frame."""
        frame = ttk.LabelFrame(parent, text="TORQUE vs TIME", padding=5)

        self.torque_fig = Figure(figsize=(5.6, 2.8), dpi=100)
        self.torque_ax = self.torque_fig.add_subplot(111)
        # Reserve margin so axis labels remain readable in compact layouts.
        self.torque_fig.subplots_adjust(left=0.16, right=0.97, bottom=0.22, top=0.88)
        self.torque_ax.set_title("Shoulder/Elbow Torque")
        self.torque_ax.set_xlabel("Time (s)")
        self.torque_ax.set_ylabel("Torque (Nm)")
        self.torque_ax.grid(True, alpha=0.3)

        self.torque_canvas = FigureCanvasTkAgg(self.torque_fig, master=frame)
        self.torque_canvas.draw()
        self.torque_canvas.get_tk_widget().pack(fill="both", expand=True)

        return frame

    def _create_velocity_acceleration_frame(self, parent):
        """Create right-side velocity/acceleration plotting frame."""
        frame = ttk.LabelFrame(parent, text="VELOCITY & ACCELERATION PLOT", padding=5)

        self.velacc_fig = Figure(figsize=(4.2, 3.0), dpi=100)
        self.velacc_ax_vel = self.velacc_fig.add_subplot(211)
        self.velacc_ax_acc = self.velacc_fig.add_subplot(212, sharex=self.velacc_ax_vel)
        self.velacc_fig.subplots_adjust(left=0.16, right=0.97, bottom=0.14, top=0.92, hspace=0.38)

        self.velacc_ax_vel.set_title("End-Effector Velocity")
        self.velacc_ax_vel.set_ylabel("m/s")
        self.velacc_ax_vel.grid(True, alpha=0.3)

        self.velacc_ax_acc.set_title("End-Effector Acceleration")
        self.velacc_ax_acc.set_xlabel("Time (s)")
        self.velacc_ax_acc.set_ylabel("m/s^2")
        self.velacc_ax_acc.grid(True, alpha=0.3)

        self.velacc_canvas = FigureCanvasTkAgg(self.velacc_fig, master=frame)
        self.velacc_canvas.draw()
        self.velacc_canvas.get_tk_widget().pack(fill="both", expand=True)

        return frame

    def _create_visualization_frame(self, parent):
        """Create matplotlib visualization frame"""
        frame = ttk.LabelFrame(parent, text="ARM VISUALIZATION", padding=5)

        # Create matplotlib figure
        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal", adjustable="box")

        # Embed in Tkinter
        self.canvas_agg = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas_agg.mpl_connect("button_press_event", self._on_canvas_click)
        self.canvas_agg.draw()
        self.canvas_agg.get_tk_widget().pack(fill="both", expand=True)

        return frame

    def _sync_ui_to_config(self):
        """Synchronize UI sliders to current config after config change"""
        # Update all link length sliders
        for i, val in enumerate(self.config.link_lengths):
            self.slider_vars[f"link_len_{i}"].set(val)
            self.slider_vars[f"link_len_{i}_label"].config(text=f"{val:.2f}")
            self.slider_vars[f"link_len_{i}_entry"].set(f"{val:.2f}")

        # Update all mass sliders (only first 3 are shown in GUI)
        for i in range(min(3, self.config.dof)):
            self.slider_vars[f"mass_{i}"].set(self.config.masses[i])
            self.slider_vars[f"mass_{i}_label"].config(text=f"{self.config.masses[i]:.2f}")
            self.slider_vars[f"mass_{i}_entry"].set(f"{self.config.masses[i]:.2f}")

        # Update damping slider
        self.slider_vars["damping"].set(self.config.damping)
        self.slider_vars["damping_label"].config(text=f"{self.config.damping:.2f}")

    def _compute_positions(self):
        """Recompute arm positions after config change"""
        # When config changes (link lengths, etc.), recompute positions with current angles
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
        # Create main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Title
        title_label = ttk.Label(main_frame, text="RL Arm Motion - Interactive Controller", font=("Arial", 16, "bold"))
        title_label.pack(fill="x", pady=(0, 10))

        ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=5)

        # Content panes (left, center, right) - user resizable
        content_pane = ttk.Panedwindow(main_frame, orient="horizontal")
        content_pane.pack(fill="both", expand=True)

        # Left pane: arm model properties + torque
        left_frame = ttk.Frame(content_pane)
        # Center pane: joint control + model selection + arm visualization
        center_frame = ttk.Frame(content_pane)
        # Right pane: metrics + velocity/acceleration plots
        right_frame = ttk.Frame(content_pane)
        left_frame.configure(width=400)
        center_frame.configure(width=400)
        right_frame.configure(width=400)
        content_pane.add(left_frame, weight=1)
        content_pane.add(center_frame, weight=1)
        content_pane.add(right_frame, weight=1)

        # Left pane internal layout: equal-height properties + torque
        left_pane = ttk.Panedwindow(left_frame, orient="vertical")
        left_pane.pack(fill="both", expand=True)
        properties_host = ttk.Frame(left_pane)
        torque_host = ttk.Frame(left_pane)
        left_pane.add(properties_host, weight=1)
        left_pane.add(torque_host, weight=1)

        # Middle pane internal layout:
        # top: joint control (0.3), middle: model selection (0.2), bottom: visualization (0.5)
        middle_pane = ttk.Panedwindow(center_frame, orient="vertical")
        middle_pane.pack(fill="both", expand=True)
        controls_host = ttk.Frame(middle_pane)
        model_select_host = ttk.Frame(middle_pane)
        viz_host = ttk.Frame(middle_pane)
        middle_pane.add(controls_host, weight=30)
        middle_pane.add(model_select_host, weight=20)
        middle_pane.add(viz_host, weight=50)

        # Left-top: fixed arm model properties (non-scrollable)
        properties_frame = self._create_properties_frame(properties_host)
        properties_frame.pack(fill="both", expand=True)

        # Left-side torque plot
        torque_frame = self._create_torque_frame(torque_host)
        torque_frame.pack(fill="both", expand=True)

        # Middle-top: joint control
        control_frame = self._create_control_frame(controls_host)
        control_frame.pack(fill="both", expand=True)

        # Middle-middle: arm model selection
        model_frame = self._create_model_selection_frame(model_select_host)
        model_frame.pack(fill="both", expand=True)

        # Middle-bottom: arm visualization
        viz_frame = self._create_visualization_frame(viz_host)
        viz_frame.pack(fill="both", expand=True)

        # Right pane internal layout: metrics (top) / vel+acc plots (bottom), 0.5/0.5.
        right_pane = ttk.Panedwindow(right_frame, orient="vertical")
        right_pane.pack(fill="both", expand=True)
        metrics_host = ttk.Frame(right_pane)
        velacc_host = ttk.Frame(right_pane)
        right_pane.add(metrics_host, weight=1)
        right_pane.add(velacc_host, weight=1)

        metrics_frame = self._create_metrics_frame(metrics_host)
        metrics_frame.pack(fill="both", expand=True)

        velacc_frame = self._create_velocity_acceleration_frame(velacc_host)
        velacc_frame.pack(fill="both", expand=True)

        # Bottom: Exit button
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill="x", pady=(10, 0))
        ttk.Separator(bottom_frame, orient="horizontal").pack(fill="x", pady=(0, 10))
        ttk.Button(bottom_frame, text="Exit", command=self.on_closing, width=15).pack()

        # Set useful initial divider positions while keeping panes user-resizable.
        # Retry until geometry is fully realized so center/right panes never collapse.
        def _set_initial_pane_widths(retries=10):
            try:
                width = int(content_pane.winfo_width())
                if width < 900:
                    if retries > 0:
                        self.root.after(80, lambda: _set_initial_pane_widths(retries - 1))
                    return
                # Main panes: equal width (1/3 each).
                content_pane.sashpos(0, int(width * (1.0 / 3.0)))
                content_pane.sashpos(1, int(width * (2.0 / 3.0)))

                left_height = int(left_pane.winfo_height())
                middle_height = int(middle_pane.winfo_height())
                right_height = int(right_pane.winfo_height())
                if left_height > 300 and middle_height > 300 and right_height > 300:
                    # Left pane split: equal height (0.5 / 0.5).
                    left_pane.sashpos(0, int(left_height * 0.50))
                    # Middle pane split: top/middle/bottom = 0.3 / 0.2 / 0.5.
                    middle_pane.sashpos(0, int(middle_height * 0.30))
                    middle_pane.sashpos(1, int(middle_height * 0.50))
                    # Right pane split: metrics/vel-acc = 0.5 / 0.5.
                    right_pane.sashpos(0, int(right_height * 0.50))
                elif retries > 0:
                    self.root.after(80, lambda: _set_initial_pane_widths(retries - 1))
            except Exception:
                if retries > 0:
                    self.root.after(80, lambda: _set_initial_pane_widths(retries - 1))

        self.root.after(120, _set_initial_pane_widths)

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
        self.angle_labels[joint_id].config(text=f"{self.controller.angles[joint_id]:.2f}rad")

    def _on_record(self):
        """Handle record button"""
        if self.simulation_active:
            self._stop_simulation()
        self.recording = not self.recording
        if self.recording:
            self.recorder.start_recording()
            self.record_button.config(text="Stop Recording (REC)")
            print("[OK] Recording started")
        else:
            self.recorder.stop_recording()
            self.record_button.config(text="Record")
            print(f"[OK] Recording stopped ({self.recorder.get_num_frames()} frames)")

    def _on_clear_recording(self):
        """Clear recording"""
        if self.simulation_active:
            self._stop_simulation()
        self.recorder.clear_frames()
        print("[OK] Recording cleared")

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
                print(f"[OK] Motion saved to {filepath}")
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
                print(f"[OK] Loaded {self.recorder.get_num_frames()} frames")
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
                self.playback_button.config(text="Stop Playback")
                print("[OK] Playback started")
            else:
                self.playback_button.config(text="Playback")
                print("[OK] Playback stopped")
        else:
            messagebox.showwarning("Warning", "No frames recorded to playback")

    def _on_select_model(self):
        """Load a trained model for policy simulation."""
        filepath = filedialog.askopenfilename(
            title="Select trained model (.zip)",
            filetypes=[("Model files", "*.zip"), ("All files", "*.*")],
        )
        if not filepath:
            return

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

        self.sim_model_label.config(text=f"Model: {Path(model_base).name} ({algorithm})")
        self.sim_toggle_button.config(state="normal", text="Run Simulation")
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
            env_desc = self.sim_manifest.get("env", {})
            details_text += (
                f"\nPackage: {self.sim_package_dir}"
                f"\nPreset: {env_desc.get('preset')}  obs v{env_desc.get('obs_version')}  "
                f"{env_desc.get('actuation_mode')}  waypoints {env_desc.get('waypoints')}"
                f"\nSeed {self.sim_manifest.get('seed')}  best score {self.sim_manifest.get('best_score')}"
            )
        self._set_model_details_text(details_text)
        print(f"[OK] Loaded model: {model_base} ({algorithm})")

    def _on_smooth_toggle(self):
        """Toggle action smoothing on/off."""
        self.action_smoothing = self._smooth_var.get()
        state = "ON" if self.action_smoothing else "OFF"
        print(f"[OK] Action smoothing {state} (alpha={self.action_smooth_alpha:.2f})")
        if self.simulation_active:
            print("  (restart simulation to apply)")

    def _on_toggle_simulation(self):
        """Start/stop policy simulation."""
        if self.simulation_active:
            self._stop_simulation()
            return
        self._start_simulation()

    def _on_place_toggle(self):
        """Legacy entry point: the checkbox became the mode buttons below."""
        self._click_mode_var.set("add" if self._place_var.get() else "off")
        self._on_click_mode()

    def _on_click_mode(self):
        """Adding, removing, or neither - whichever button is pressed in."""
        mode = self._click_mode_var.get()
        self._place_mode = mode in ("add", "remove")
        self._place_var.set(self._place_mode)
        # A running episode owns the panel and hides the points being placed,
        # so entering either mode stops it: you are about to define a new run.
        if self._place_mode and self.simulation_active:
            self._stop_simulation()
        self._refresh_place_hint()
        print(f"[OK] {'clicks ADD points' if mode == 'add' else 'clicks REMOVE points' if mode == 'remove' else 'click placement off'}")

    def _on_waypoint_choice(self, _event=None):
        """Picking 'custom' is only ever a prelude to plotting your own points."""
        self.sim_waypoint_choice = self._wp_choice_var.get()
        if self.sim_waypoint_choice == "custom":
            self._click_mode_var.set("add")
        else:
            self._click_mode_var.set("off")
        self._on_click_mode()

    def _refresh_place_hint(self):
        if getattr(self, "_place_hint", None) is None:
            return
        n = len(self.clicked_waypoints)
        mode = self._click_mode_var.get() if hasattr(self, "_click_mode_var") else "off"
        if mode == "add":
            text = f"click the panel to add ({n} placed)" + (" - press Go" if n else "")
        elif mode == "remove":
            text = f"click a point to delete it ({n} placed)" if n else "nothing to remove"
        else:
            text = f"{n} point(s) ready - press Go" if n else "press '+ Plot points', then click on the arm panel"
        self._place_hint.config(text=text)

    def _on_clear_points(self):
        self.clicked_waypoints = []
        self.clicked_requests = []
        self._refresh_place_hint()
        self.sim_custom_waypoints = ""
        if hasattr(self, "_wp_custom_var"):
            self._wp_custom_var.set("")
        print("[OK] Placed waypoints cleared")

    def _on_restart_with_points(self):
        if not self.clicked_waypoints:
            messagebox.showwarning("Waypoints", "Tick 'Use MY OWN points' and click on the arm panel first.")
            return
        if self.simulation_active:
            self._stop_simulation()
        self._start_simulation()

    def _feasibility_env(self):
        base = getattr(self.sim_env, "unwrapped", None) if self.sim_env is not None else None
        if base is None:
            base = ArmTaskEnv(actuation_mode="torque", config=self.config)
        return base

    def _nearest_placeable(self, env, wp, existing):
        """Nearest holdable point clear of the others (shared with the demo window)."""
        return nearest_placeable(env, wp, existing)

    def _remove_point_near(self, wp):
        """Drop the placed point nearest the click, if one is close enough."""
        if not self.clicked_waypoints:
            return False
        i = index_near(self.clicked_waypoints, wp)
        if i is None:
            return False
        gone = self.clicked_waypoints.pop(i)
        if i < len(self.clicked_requests):
            self.clicked_requests.pop(i)
        self._sync_clicked_points(f"removed the point at ({gone[0]:.2f}, {gone[1]:.2f})")
        return True

    def _undo_last_point(self):
        if not self.clicked_waypoints:
            print("[WARN] no points to undo")
            return
        gone = self.clicked_waypoints.pop()
        if self.clicked_requests:
            self.clicked_requests.pop()
        self._sync_clicked_points(f"undid the point at ({gone[0]:.2f}, {gone[1]:.2f})")

    def _sync_clicked_points(self, message=None):
        """Keep the text box, the hint and the console in step with the points."""
        self.sim_custom_waypoints = ";".join(f"{x:.3f},{y:.3f}" for x, y in self.clicked_waypoints)
        self.sim_waypoint_choice = "custom" if self.clicked_waypoints else self.sim_waypoint_choice
        if hasattr(self, "_wp_custom_var"):
            self._wp_custom_var.set(self.sim_custom_waypoints)
        if hasattr(self, "_wp_choice_var") and self.clicked_waypoints:
            self._wp_choice_var.set("custom")
        self._refresh_place_hint()
        if message:
            print(f"[OK] {message}; {len(self.clicked_waypoints)} point(s) left")

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
        """
        if not self._place_mode or event.inaxes is not self.ax or event.xdata is None:
            return
        env = self._feasibility_env()
        shoulder = np.asarray(env.shoulder_base_position, dtype=float)
        wp = [float(event.xdata) + shoulder[0], float(event.ydata) + shoulder[1]]
        # In remove mode every click deletes; right-click deletes in any mode.
        if getattr(self, "_click_mode_var", None) is not None and self._click_mode_var.get() == "remove":
            if not self._remove_point_near(wp):
                print("[WARN] click closer to one of your points to remove it")
            return
        if getattr(event, "button", 1) in (2, 3):
            if not self._remove_point_near(wp):
                print("[WARN] right-click closer to a placed point to remove it")
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
                print(f"[WARN] no point near ({wp[0]:.2f}, {wp[1]:.2f}) can be held and kept "
                      f"{2 * env.max_curriculum_tolerance:.2f} m clear of the others; "
                      f"try somewhere else in the shaded region")
                return
            print(f"[OK] ({wp[0]:.2f}, {wp[1]:.2f}) moved {moved * 100:.0f} cm to "
                  f"({snapped[0]:.2f}, {snapped[1]:.2f}), the nearest point the arm can reach and hold")
            self.clicked_requests.append((list(wp), list(snapped)))
            wp = snapped
        else:
            self.clicked_requests.append((list(wp), list(wp)))
        self.clicked_waypoints = self.clicked_waypoints + [wp]
        self._sync_clicked_points()
        print(f"[OK] point {len(self.clicked_waypoints)} placed at ({wp[0]:.3f}, {wp[1]:.3f}); "
              f"use '- Remove points' to delete one, or press Go to run")

    def _feasible_cloud_points(self):
        """Shoulder-local scatter of feasible waypoint positions (cached per config)."""
        key = (self.config.name, tuple(self.config.link_lengths))
        if self._feasible_cloud_key != key:
            env = self._feasibility_env()
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
                print(f"[OK] Action smoothing ON (alpha={self.action_smooth_alpha})")
            else:
                if self.action_smoothing and raw_env.use_activation_dynamics:
                    print("[OK] Action smoothing skipped: this model has muscle activation dynamics "
                          "(an extra smoother puts it out of distribution)")
                self.sim_env = raw_env
            self._feasible_cloud_key = None
            self.sim_model = model
            print(f"[OK] {raw_env.actuation_mode}-mode model in {raw_env.preset_name}: "
                  f"{raw_env.num_waypoints} waypoints, tolerance {raw_env.goal_tolerance} m, "
                  f"hold {raw_env.hold_steps_required} steps")
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
            if free_order:
                print(f"[OK] Free-order model: it sees all {raw_env.free_order_slots} points every step "
                      f"and chooses the order itself; the panel draws no route.")
            if wps is not None and not trained_random:
                trained_wps = (self.sim_manifest or {}).get("env", {}).get("waypoints")
                print(f"[WARN] This is a FIXED-waypoint model (trained only on {trained_wps}); "
                      f"it is not expected to reach other points. Use a random-waypoint (Stage R3) package "
                      f"for arbitrary targets.")
                if self.sim_model_label is not None:
                    self.sim_model_label.config(
                        text=f"Model: {Path(self.sim_model_path).name}  [FIXED-waypoint model: custom points unsupported]")

            self.sim_episode_seed = 0
            self.sim_obs, info = self.sim_env.reset(seed=self.sim_episode_seed)
            self.sim_last_info = info
            self._set_controller_state_from_arrays(info["joint_angles"], info["joint_velocities"])

            self.simulation_active = True
            self.sim_step_count = 0
            self.sim_episode_count = 1
            self.sim_successes = 0
            self.sim_episode_reward = 0.0
            self.sim_time = 0.0
            self.sim_step_accumulator = 0.0
            self.sim_success_banner_until = 0.0
            self.prev_sim_velocities = np.array(info["joint_velocities"], dtype=float)
            self.torque_time_hist.clear()
            self.shoulder_torque_hist.clear()
            self.elbow_torque_hist.clear()
            self.latest_shoulder_torque = 0.0
            self.latest_elbow_torque = 0.0
            self._reset_ee_kinematics()
            self.sim_toggle_button.config(text="Stop Simulation")
            print("[OK] Policy simulation started")
        except Exception as e:
            self.simulation_active = False
            self.sim_model = None
            messagebox.showerror("Simulation Error", f"Failed to start simulation:\n{e}")

    def _stop_simulation(self):
        """Stop policy simulation and cleanup."""
        self.simulation_active = False
        self.sim_obs = None
        self.sim_model = None
        if self.sim_env is not None:
            try:
                self.sim_env.close()
            except Exception:
                pass
            self.sim_env = None
        if self.sim_toggle_button is not None:
            self.sim_toggle_button.config(text="Run Simulation")
        self.sim_time = 0.0
        self.prev_sim_velocities = np.zeros(self.config.dof, dtype=float)
        self._reset_ee_kinematics()
        self._update_vel_acc_plot()
        print("[OK] Policy simulation stopped")

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
        accelerations = (velocities_arr - self.prev_sim_velocities) / max(dt, 1e-8)
        shoulder_tau, elbow_tau = self._estimate_joint_torques(angles, velocities_arr, accelerations)
        self.sim_time += dt
        self._append_torque_sample(shoulder_tau, elbow_tau, self.sim_time)
        self.prev_sim_velocities = velocities_arr.copy()
        self._set_controller_state_from_arrays(angles, velocities_arr)

        if info.get("success") and info.get("success_step") == info.get("step"):
            self.sim_successes += 1
            self.sim_success_banner_until = time() + 1.0
            print(f"[OK] HOLD COMPLETE at step {info['step']} "
                  f"({info['waypoints_reached']}/{info['num_waypoints']} waypoints)")

        if terminated or truncated:
            if not self.sim_loop:
                self._stop_simulation()
                return
            self.sim_episode_count += 1
            self.sim_episode_seed += 1
            self.sim_episode_reward = 0.0
            self.sim_obs, info = self.sim_env.reset(seed=self.sim_episode_seed)
            self.sim_last_info = info
            self.prev_sim_velocities = np.array(info["joint_velocities"], dtype=float)
            self._set_controller_state_from_arrays(info["joint_angles"], info["joint_velocities"])

    def _on_reset_arm(self):
        """Reset arm to home position"""
        if self.simulation_active:
            self._stop_simulation()
        self.controller.set_home_position()
        self.trajectory_points = []
        self._reset_ee_kinematics()
        for i in range(self.config.dof):
            self.angle_labels[i].config(text=f"{self.controller.angles[i]:.2f}rad")
        print("[OK] Arm reset to home position")

    def _on_reset_defaults(self):
        """Reset to default configuration"""
        if self.simulation_active:
            self._stop_simulation()
        self.config = ArmConfiguration.get_preset(DEFAULT_PRESET)
        self.controller = ArmController(self.config)
        self.visualizer = ArmVisualizer(dof=self.config.dof, link_lengths=self.config.link_lengths)
        self._sync_ui_to_config()  # Update all UI sliders
        self._compute_positions()  # Recompute positions
        self._reset_ee_kinematics()
        print("[OK] Reset to default configuration")
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
                print(f"[OK] Configuration saved to {filepath}")
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
                self._reset_ee_kinematics()
                messagebox.showinfo("Success", f"Loaded configuration: {self.config.name}")
                print(f"[OK] Loaded configuration: {self.config.name}")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading config: {e}")

    def _update_visualization(self):
        """Update the arm visualization"""
        self.ax.clear()

        # Get current positions
        positions = self.controller.positions

        # Plot links
        self.ax.plot(positions[:, 0], positions[:, 1], "b-o", linewidth=2, markersize=6)

        # Mark base
        self.ax.plot(positions[0, 0], positions[0, 1], "go", markersize=12, label="Base")

        # Mark end-effector
        self.ax.plot(positions[-1, 0], positions[-1, 1], "r*", markersize=20, label="End-effector")

        # Plot trajectory if enabled
        if self.show_trajectory and self.trajectory_points:
            traj_array = np.array(self.trajectory_points)
            self.ax.plot(traj_array[:, 0], traj_array[:, 1], "g--", alpha=0.5, linewidth=1, label="Trajectory")

        # Torso / head silhouette (side view, +x in front of the body).
        reach = float(sum(self.config.link_lengths))
        torso_w, torso_h = 0.22 * reach, 0.9 * reach
        self.ax.add_patch(plt.Rectangle((-torso_w, -torso_h), torso_w, torso_h, color="#00000012", zorder=0))
        self.ax.add_patch(plt.Circle((-0.5 * torso_w, 0.32 * reach), 0.17 * reach, color="#00000012", zorder=0))
        self.ax.add_patch(plt.Circle((0.0, 0.0), reach, color="#00000022", fill=False, linestyle="--", linewidth=0.8))

        title = f"Arm Configuration - FPS: {self.fps_counter:.0f}"
        info = self.sim_last_info if self.simulation_active else None
        if self._place_mode or (self.clicked_waypoints and not self.simulation_active):
            try:
                cloud = self._feasible_cloud_points()
                self.ax.scatter(cloud[:, 0], cloud[:, 1], s=6, c="#2ecc71", alpha=0.25, zorder=1, linewidths=0)
            except Exception:
                pass
        if self.clicked_waypoints and not self.simulation_active:
            shoulder = np.asarray(self._feasibility_env().shoulder_base_position, dtype=float)
            for i, wp in enumerate(self.clicked_waypoints):
                p = np.asarray(wp, dtype=float) - shoulder
                # If the click was moved to somewhere holdable, show where it
                # was asked for and the move, so nothing is relocated silently.
                if i < len(self.clicked_requests):
                    asked, used = (np.asarray(v, dtype=float) for v in self.clicked_requests[i])
                    if float(np.linalg.norm(asked - used)) > 1e-6:
                        a = asked - shoulder
                        self.ax.plot([a[0], p[0]], [a[1], p[1]], color="#888888", linestyle=":",
                                     linewidth=1.0, zorder=3)
                        self.ax.scatter(a[0], a[1], s=60, facecolors="none", edgecolors="#888888",
                                        linewidths=1.2, zorder=3)
                self.ax.add_patch(plt.Circle((p[0], p[1]), 0.05, color="red", fill=False, linewidth=1.2, zorder=4))
                self.ax.scatter(p[0], p[1], c="red", s=160, marker="X", edgecolors="black", linewidths=1.5, zorder=10)
                self.ax.annotate(f"{i + 1}", (p[0], p[1]), textcoords="offset points", xytext=(8, 8),
                                 fontsize=10, fontweight="bold")
            title = f"{len(self.clicked_waypoints)} of your own point(s) placed - press Go"
        if info is not None:
            # Waypoints are in the workspace frame; the plot is shoulder-local.
            shoulder = np.asarray(info.get("shoulder_position", [0.0, 0.0]), dtype=float)
            wps = info.get("waypoints") or [info.get("goal_position")]
            k = int(info.get("current_waypoint_index", 0))
            tol = float(info.get("goal_tolerance", 0.05))
            n_wp = len(wps)
            # A free-order policy picks its own route, so the panel must not
            # draw one: no line through the points in list order and no numbers
            # implying a sequence. Points are grey until the arm has visited
            # them, green afterwards.
            free_order = bool(info.get("free_order"))
            visited = list(info.get("visited") or [])
            # No line is drawn between the waypoints. It was only ever
            # decoration joining them in list order, but it reads as a track the
            # arm is being steered along, which is the opposite of what happens:
            # the policy is given a target position and works out the movement
            # itself, and the path it actually takes is the trail it leaves.
            for i, wp in enumerate(wps):
                p = np.asarray(wp, dtype=float) - shoulder
                if free_order:
                    done = bool(visited[i]) if i < len(visited) else False
                    colour = "green" if done else "lightgray"
                else:
                    colour = "green" if i < k or info.get("success") else ("red" if i == k else "lightgray")
                self.ax.add_patch(plt.Circle((p[0], p[1]), tol, color=colour, alpha=0.15, zorder=4))
                self.ax.add_patch(plt.Circle((p[0], p[1]), tol, color=colour, fill=False, linewidth=1.2, zorder=4))
                self.ax.scatter(p[0], p[1], c=colour, s=160, marker="X", edgecolors="black", linewidths=1.5, zorder=10)
                if not free_order:
                    self.ax.annotate(f"{i + 1}", (p[0], p[1]), textcoords="offset points", xytext=(8, 8),
                                     fontsize=10, fontweight="bold")
            hold_pct = int(round(100.0 * float(info.get("hold_progress", 0.0))))
            reached = int(info.get("waypoints_reached", 0))
            state = "HOLD COMPLETE" if info.get("success") else ("settling" if info.get("in_goal_region") else "moving")
            if free_order:
                title = f"{reached}/{n_wp} visited (arm's own order) | hold {hold_pct}% | {state}"
            else:
                title = f"WP {min(k + 1, n_wp)}/{n_wp} | hold {hold_pct}% | {state} | {reached}/{n_wp} reached"
            if self.sim_hold_bar is not None:
                self.sim_hold_bar["value"] = hold_pct
            if self.sim_progress_label is not None:
                self.sim_progress_label.config(
                    text=f"reached {reached}/{n_wp}  |  episodes {self.sim_successes}/{max(self.sim_episode_count, 1)}")

        self.ax.set_xlabel("X (m)  [+x = in front of the body]")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title(title, fontsize=9 if info is not None else 10)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper right")

        # Use dynamic axis limits based on arm reach
        xlim_min, xlim_max = self._calculate_axis_limits()
        self.ax.set_xlim(xlim_min, xlim_max)
        self.ax.set_ylim(xlim_min, xlim_max)

        self.canvas_agg.draw_idle()

    def _update_metrics_display(self):
        """Update metrics display"""
        pos = self.controller.get_end_effector_position()

        metrics_text = f"""
CURRENT STATE:
{'═' * 32}
Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})

Joint Angles (rad):
{', '.join(f'{a:.3f}' for a in self.controller.angles)}

Joint Velocities (rad/s):
{', '.join(f'{v:.3f}' for v in self.controller.velocities)}

RECORDING:
{'═' * 32}
Status: {'REC' if self.recording else 'STOPPED'}
Frames: {self.recorder.get_num_frames()}
Config: {self.config.name}

PLAYBACK:
{'═' * 32}
Status: {'PLAYING' if self.playing_back else 'IDLE'}
Progress: {self.playback_index}/{len(self.playback_frames)}

SIMULATION:
{'═' * 32}
Status: {'RUNNING' if self.simulation_active else 'IDLE'}
Model: {Path(self.sim_model_path).name if self.sim_model_path else 'None'}
Algo: {self.sim_algorithm if self.sim_algorithm else '-'}
Episode: {self.sim_episode_count}
Steps: {self.sim_step_count}
Ep Reward: {self.sim_episode_reward:.3f}
Shoulder τ: {self.latest_shoulder_torque:.3f} Nm
Elbow τ: {self.latest_elbow_torque:.3f} Nm
Waypoint: {(int(self.sim_last_info.get('current_waypoint_index', 0)) + 1) if self.sim_last_info else '-'}/{int(self.sim_last_info.get('num_waypoints', 0)) if self.sim_last_info else '-'}
Hold: {int(round(100 * float(self.sim_last_info.get('hold_progress', 0.0)))) if self.sim_last_info else 0}%  Success: {bool(self.sim_last_info.get('success')) if self.sim_last_info else False}
Holds completed: {self.sim_successes}

PERFORMANCE:
{'═' * 32}
FPS: {self.fps_counter:.1f}
Trajectory Pts: {len(self.trajectory_points)}

TRAJECTORY:
{'═' * 32}
Show: {'ON' if self.show_trajectory else 'OFF'}
Points: {len(self.trajectory_points)}
        """

        # Update text widget
        self.metrics_text.config(state="normal")
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, metrics_text)
        self.metrics_text.config(state="disabled")

    def _handle_keyboard(self, event):
        """Handle keyboard events"""
        # Joint selection with Left/Right
        if event.keysym == "Left":
            self.selected_joint = max(0, self.selected_joint - 1)
            print(f"[OK] Selected joint {self.selected_joint}")
        elif event.keysym == "Right":
            self.selected_joint = min(self.config.dof - 1, self.selected_joint + 1)
            print(f"[OK] Selected joint {self.selected_joint}")
        # Joint control with Up/Down on selected joint
        elif event.keysym == "Up":
            self._increment_joint(self.selected_joint, 0.05)
        elif event.keysym == "Down":
            self._increment_joint(self.selected_joint, -0.05)
        elif event.char == 't':
            # Toggle trajectory
            self.show_trajectory = not self.show_trajectory
            print(f"[OK] Trajectory {'ON' if self.show_trajectory else 'OFF'}")

    def run(self):
        """Run the GUI event loop"""
        self.create_window()

        # Bind keyboard events
        self.root.bind("<Up>", self._handle_keyboard)
        self.root.bind("<Down>", self._handle_keyboard)
        self.root.bind("<Left>", self._handle_keyboard)
        self.root.bind("<Right>", self._handle_keyboard)
        self.root.bind("<t>", self._handle_keyboard)

        if getattr(self, "sim_model_path", "") and getattr(self, "_pending_autostart", False):
            if self.sim_model_label is not None:
                self.sim_model_label.config(text=f"Model: {Path(self.sim_model_path).name} (SAC)")
            if self.sim_toggle_button is not None:
                self.sim_toggle_button.config(state="normal")
            if hasattr(self, "_wp_choice_var"):
                self._wp_choice_var.set(self.sim_waypoint_choice)
            if hasattr(self, "_loop_var"):
                self._loop_var.set(self.sim_loop)
            self.root.after(300, self._start_simulation)

        def update_loop():
            """Main update loop"""
            if not self.running:
                return

            # Handle playback
            if self.simulation_active:
                self.sim_step_accumulator += self.sim_steps_per_frame * self.sim_speed_multiplier
                n_steps = int(self.sim_step_accumulator)
                self.sim_step_accumulator -= n_steps
                for _ in range(n_steps):
                    self._simulation_step()
                    if not self.simulation_active:
                        break
            elif self.playing_back and self.playback_frames:
                if self.playback_index < len(self.playback_frames):
                    frame_state = self.playback_frames[self.playback_index]
                    self.controller.apply_state(frame_state)
                    self.playback_index += 1
                else:
                    self.playing_back = False
                    self.playback_button.config(text="Playback")
                    print("[OK] Playback finished")

            # Record current frame if recording
            if self.recording:
                state = self.controller.get_state(timestamp=time())
                self.recorder.record_frame(state)

                # Update trajectory
                ee_pos = self.controller.get_end_effector_position()
                self.trajectory_points.append(ee_pos)

            # Update visualizations.
            #
            # Redrawing all four matplotlib canvases every tick costs a whole
            # core and starves the Tk event loop, so the window stops answering
            # clicks and Windows marks it "not responding" - even when the arm
            # is standing still and there is nothing new to draw. Only the arm
            # panel needs to keep up with motion; the supporting plots are
            # readable at 5 Hz, and when nothing is moving nothing needs
            # redrawing at all beyond an occasional refresh.
            self._frame_tick = getattr(self, "_frame_tick", 0) + 1
            moving = bool(self.simulation_active or self.playing_back or self.recording)
            slow_tick = self._frame_tick % 4 == 0
            if moving or slow_tick:
                self._update_visualization()
            if moving and slow_tick:
                self._update_metrics_display()
                self._update_torque_plot()
                self._update_vel_acc_plot()
            elif not moving and self._frame_tick % 20 == 0:
                # Idle: a 1 Hz refresh keeps the panels current (including the
                # final trace after a run ends) for almost no cost.
                self._update_metrics_display()
                self._update_torque_plot()
                self._update_vel_acc_plot()
            if self.simulation_active:
                self._append_ee_kinematics_sample(time())

            # Update FPS counter
            current_time = time()
            if current_time - self.last_frame_time >= 1.0:
                self.fps_counter = self.frame_count
                self.last_frame_time = current_time
                self.frame_count = 0
            else:
                self.frame_count += 1

            # Schedule next update (50ms = ~20 FPS base, matplotlib will render at higher FPS)
            self.root.after(50, update_loop)

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
        gui.sim_model_path = gui._normalize_model_base(str(model_zip))
        gui.sim_algorithm = "SAC"
        mp = find_manifest(str(model_zip))
        gui.sim_package_dir = str(mp.parent) if mp else None
        gui.sim_manifest = read_manifest(mp) if mp else None
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
