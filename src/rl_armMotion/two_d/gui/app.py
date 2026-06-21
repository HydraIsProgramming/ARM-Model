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
from rl_armMotion.two_d.environments.task_env import ArmTaskEnv, ActionSmoother
from rl_armMotion.two_d.models.trainers import RLTrainer
from rl_armMotion.two_d.utils import ArmKinematics, ArmController, MotionRecorder, ArmVisualizer


class ArmControllerGUI:
    """Main GUI application for interactive arm control using Tkinter"""

    def __init__(self, config: ArmConfiguration = None):
        """
        Initialize GUI application.

        Args:
            config: ArmConfiguration object (uses default if None)
        """
        self.config = config or ArmConfiguration.get_preset("2dof_simple")
        self.controller = ArmController(self.config)
        self.recorder = MotionRecorder()
        self.visualizer = ArmVisualizer(dof=self.config.dof)

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
        self.sim_steps_per_frame = 5  # default: real-time (dt=0.01 × 5 steps = 0.05s/frame)
        self.sim_toggle_button = None
        self.sim_model_label = None
        self.sim_model_details_text = None
        self.sim_time = 0.0
        self.action_smoothing = True
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
        """
        Estimate shoulder/elbow torque from simplified planar dynamics.

        Returns:
            (tau_shoulder, tau_elbow) in Nm
        """
        angles = np.asarray(angles, dtype=float)
        velocities = np.asarray(velocities, dtype=float)
        accelerations = np.asarray(accelerations, dtype=float)

        if angles.shape[0] < 2:
            tau0 = (
                float(self.config.inertias[0]) * float(accelerations[0])
                + float(self.config.damping) * float(velocities[0])
            )
            return tau0, 0.0

        g = 9.81
        l1, l2 = float(self.config.link_lengths[0]), float(self.config.link_lengths[1])
        m1, m2 = float(self.config.masses[0]), float(self.config.masses[1])
        i1, i2 = float(self.config.inertias[0]), float(self.config.inertias[1])
        damping = float(self.config.damping)

        t1, t2 = float(angles[0]), float(angles[1])
        w1, w2 = float(velocities[0]), float(velocities[1])
        a1, a2 = float(accelerations[0]), float(accelerations[1])

        # Approximate inertial component.
        tau1_inertia = (i1 + i2) * a1 + i2 * a2
        tau2_inertia = i2 * (a1 + a2)

        # Gravity load component.
        tau1_gravity = g * (
            m1 * (l1 * 0.5) * np.cos(t1)
            + m2 * (l1 * np.cos(t1) + (l2 * 0.5) * np.cos(t1 + t2))
        )
        tau2_gravity = g * (m2 * (l2 * 0.5) * np.cos(t1 + t2))

        # Viscous damping component.
        tau1_damping = damping * w1
        tau2_damping = damping * w2

        tau1 = tau1_inertia + tau1_gravity + tau1_damping
        tau2 = tau2_inertia + tau2_gravity + tau2_damping
        return float(tau1), float(tau2)

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

    def _create_model_selection_frame(self, parent):
        """Create arm-model selection frame."""
        frame = ttk.LabelFrame(parent, text="ARM MODEL SELECTION", padding=5)

        control_row = ttk.Frame(frame)
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
        speed_row = ttk.Frame(frame)
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
            self.sim_steps_per_frame = steps
            multiplier = steps / 5.0  # 5 steps/frame = 1x real-time
            self._sim_speed_label.config(text=f"{multiplier:.1f}x")

        speed_slider.bind("<Motion>", _on_speed_change)
        speed_slider.bind("<ButtonRelease-1>", _on_speed_change)

        # Action smoothing control
        smooth_row = ttk.Frame(frame)
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

        details_frame = ttk.LabelFrame(frame, text="Model Details", padding=4)
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
            print("✓ Recording started")
        else:
            self.recorder.stop_recording()
            self.record_button.config(text="Record")
            print(f"✓ Recording stopped ({self.recorder.get_num_frames()} frames)")

    def _on_clear_recording(self):
        """Clear recording"""
        if self.simulation_active:
            self._stop_simulation()
        self.recorder.clear_frames()
        print("✓ Recording cleared")

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
                print(f"✓ Motion saved to {filepath}")
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
                print(f"✓ Loaded {self.recorder.get_num_frames()} frames")
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
                print("✓ Playback started")
            else:
                self.playback_button.config(text="Playback")
                print("✓ Playback stopped")
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
        self._set_model_details_text(details_text)
        print(f"✓ Loaded model: {model_base} ({algorithm})")

    def _on_smooth_toggle(self):
        """Toggle action smoothing on/off."""
        self.action_smoothing = self._smooth_var.get()
        state = "ON" if self.action_smoothing else "OFF"
        print(f"✓ Action smoothing {state} (alpha={self.action_smooth_alpha:.2f})")
        if self.simulation_active:
            print("  (restart simulation to apply)")

    def _on_toggle_simulation(self):
        """Start/stop policy simulation."""
        if self.simulation_active:
            self._stop_simulation()
            return
        self._start_simulation()

    def _start_simulation(self):
        """Initialize simulation environment and model."""
        if not self.sim_model_path or not self.sim_algorithm:
            messagebox.showwarning("Warning", "Please select a trained model first")
            return

        try:
            if self.playing_back:
                self.playing_back = False
                self.playback_button.config(text="Playback")

            if self.sim_env is not None:
                self.sim_env.close()

            # Detect actuation mode from the trained model's action space so
            # the simulation env matches what the policy was trained against.
            # A muscle-mode SAC model has action_space Box(0, 1, (4,)) — four
            # extensor + flexor activations per joint. A velocity-mode model
            # has action_space Box(-1, 1, (2,)) — two normalised joint
            # velocity commands. Without this auto-detect, loading a muscle-
            # mode model into a default velocity-mode env throws
            # "Action spaces do not match" from Stable-Baselines3.
            from stable_baselines3 import SAC, PPO, A2C
            alg_class_map = {"SAC": SAC, "PPO": PPO, "A2C": A2C}
            alg_class = alg_class_map.get(self.sim_algorithm, SAC)
            inspect_model = alg_class.load(self.sim_model_path)
            ac_shape = inspect_model.action_space.shape
            ac_low_min = float(inspect_model.action_space.low.min())
            if ac_shape == (4,) and ac_low_min == 0.0:
                detected_actuation = "muscle"
            elif ac_shape == (2,) and ac_low_min == -1.0:
                detected_actuation = "velocity"
            else:
                raise ValueError(
                    f"Cannot determine actuation mode from model action space "
                    f"{inspect_model.action_space}. Expected Box(0,1,(4,)) for "
                    f"muscle mode or Box(-1,1,(2,)) for velocity mode."
                )

            raw_env = ArmTaskEnv(actuation_mode=detected_actuation)
            if self.action_smoothing and detected_actuation == "muscle":
                self.sim_env = ActionSmoother(raw_env, alpha=self.action_smooth_alpha)
                print(f"✓ Action smoothing ON (alpha={self.action_smooth_alpha})")
            else:
                self.sim_env = raw_env
            inspect_model.set_env(self.sim_env)
            self.sim_model = inspect_model
            print(f"✓ Detected {detected_actuation}-mode model, env constructed to match")

            self.sim_obs, _ = self.sim_env.reset()
            base_env = getattr(self.sim_env, "unwrapped", self.sim_env)
            state_info = base_env.get_state_info()
            self._set_controller_state_from_arrays(
                state_info["joint_angles"],
                state_info["joint_velocities"],
            )

            self.simulation_active = True
            self.sim_step_count = 0
            self.sim_episode_count = 1
            self.sim_episode_reward = 0.0
            self.sim_time = 0.0
            self.prev_sim_velocities = np.array(state_info["joint_velocities"], dtype=float)
            self.torque_time_hist.clear()
            self.shoulder_torque_hist.clear()
            self.elbow_torque_hist.clear()
            self.latest_shoulder_torque = 0.0
            self.latest_elbow_torque = 0.0
            self._reset_ee_kinematics()
            self.sim_toggle_button.config(text="Stop Simulation")
            print("✓ Policy simulation started")
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
        print("✓ Policy simulation stopped")

    def _simulation_step(self):
        """Run a single simulation step if active."""
        if not self.simulation_active or self.sim_model is None or self.sim_env is None or self.sim_obs is None:
            return

        action, _ = self.sim_model.predict(self.sim_obs, deterministic=True)
        self.sim_obs, reward, terminated, truncated, info = self.sim_env.step(action)
        self.sim_step_count += 1
        self.sim_episode_reward += float(reward)

        angles = info.get("joint_angles", self.controller.angles)
        velocities = info.get("joint_velocities", self.controller.velocities)
        velocities_arr = np.asarray(velocities, dtype=float)
        dt = float(getattr(self.sim_env, "dt", self.config.dt))
        accelerations = (velocities_arr - self.prev_sim_velocities) / max(dt, 1e-8)
        shoulder_tau, elbow_tau = self._estimate_joint_torques(angles, velocities_arr, accelerations)
        self.sim_time += dt
        self._append_torque_sample(shoulder_tau, elbow_tau, self.sim_time)
        self.prev_sim_velocities = velocities_arr.copy()

        self._set_controller_state_from_arrays(angles, velocities)

        if terminated or truncated:
            self.sim_episode_count += 1
            self.sim_episode_reward = 0.0
            self.sim_obs, _ = self.sim_env.reset()
            base_env = getattr(self.sim_env, "unwrapped", self.sim_env)
            state_info = base_env.get_state_info()
            self.prev_sim_velocities = np.array(state_info["joint_velocities"], dtype=float)
            self._set_controller_state_from_arrays(
                state_info["joint_angles"],
                state_info["joint_velocities"],
            )

    def _on_reset_arm(self):
        """Reset arm to home position"""
        if self.simulation_active:
            self._stop_simulation()
        self.controller.set_home_position()
        self.trajectory_points = []
        self._reset_ee_kinematics()
        for i in range(self.config.dof):
            self.angle_labels[i].config(text=f"{self.controller.angles[i]:.2f}rad")
        print("✓ Arm reset to home position")

    def _on_reset_defaults(self):
        """Reset to default configuration"""
        if self.simulation_active:
            self._stop_simulation()
        self.config = ArmConfiguration.get_preset("2dof_simple")
        self.controller = ArmController(self.config)
        self.visualizer = ArmVisualizer(dof=self.config.dof, link_lengths=self.config.link_lengths)
        self._sync_ui_to_config()  # Update all UI sliders
        self._compute_positions()  # Recompute positions
        self._reset_ee_kinematics()
        print("✓ Reset to default configuration")
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
                print(f"✓ Configuration saved to {filepath}")
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
                print(f"✓ Loaded configuration: {self.config.name}")
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

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title(f"Arm Configuration - FPS: {self.fps_counter:.0f}")
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
            print(f"✓ Selected joint {self.selected_joint}")
        elif event.keysym == "Right":
            self.selected_joint = min(self.config.dof - 1, self.selected_joint + 1)
            print(f"✓ Selected joint {self.selected_joint}")
        # Joint control with Up/Down on selected joint
        elif event.keysym == "Up":
            self._increment_joint(self.selected_joint, 0.05)
        elif event.keysym == "Down":
            self._increment_joint(self.selected_joint, -0.05)
        elif event.char == 't':
            # Toggle trajectory
            self.show_trajectory = not self.show_trajectory
            print(f"✓ Trajectory {'ON' if self.show_trajectory else 'OFF'}")

    def run(self):
        """Run the GUI event loop"""
        self.create_window()

        # Bind keyboard events
        self.root.bind("<Up>", self._handle_keyboard)
        self.root.bind("<Down>", self._handle_keyboard)
        self.root.bind("<Left>", self._handle_keyboard)
        self.root.bind("<Right>", self._handle_keyboard)
        self.root.bind("<t>", self._handle_keyboard)

        def update_loop():
            """Main update loop"""
            if not self.running:
                return

            # Handle playback
            if self.simulation_active:
                for _ in range(max(1, self.sim_steps_per_frame)):
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
                    print("✓ Playback finished")

            # Record current frame if recording
            if self.recording:
                state = self.controller.get_state(timestamp=time())
                self.recorder.record_frame(state)

                # Update trajectory
                ee_pos = self.controller.get_end_effector_position()
                self.trajectory_points.append(ee_pos)

            # Update visualizations
            self._update_visualization()
            self._update_metrics_display()
            self._update_torque_plot()
            if self.simulation_active:
                self._append_ee_kinematics_sample(time())
            self._update_vel_acc_plot()

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


def main():
    """Entry point for GUI application"""
    config = ArmConfiguration.get_preset("2dof_simple")
    gui = ArmControllerGUI(config)
    gui.run()


if __name__ == "__main__":
    main()
