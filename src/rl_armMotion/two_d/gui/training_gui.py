"""Training GUI for RL algorithms with real-time visualization and metrics."""

import argparse
import queue
import threading
import tkinter as tk
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from rl_armMotion.two_d.config import ArmConfiguration
from rl_armMotion.two_d.environments.task_env import ArmTaskEnv
from rl_armMotion.two_d.training.ppo_trainer_wrapper import RLTrainerWithMetrics
from rl_armMotion.two_d.utils.visualization import ArmVisualizer


class TrainingGUI:
    """GUI for training RL policies with live metrics."""

    ALGORITHMS = ["PPO", "SAC", "A2C"]
    GOAL_DIRECTIONS = ["EAST", "WEST", "NORTH"]

    def __init__(
        self,
        total_timesteps: int = 100000,
        save_dir: str = "./trained_models",
        algorithm: str = "SAC",
    ):
        # Default algorithm is SAC, following Fischer et al. (2021)
        # "Reinforcement learning control of a biomechanical model of the upper
        # extremity", Scientific Reports 11:14445. SAC is preferred over PPO for
        # this task because of its sample efficiency on continuous-control problems
        # and its native support for entropy-regularised exploration, both of
        # which Fischer et al. found essential for learning the goal-reaching task.
        self.total_timesteps = int(total_timesteps)
        self.save_dir = save_dir
        self.selected_algorithm = algorithm.upper()

        if self.selected_algorithm not in self.ALGORITHMS:
            self.selected_algorithm = "SAC"

        # Create a probe env once to initialize arm visualization defaults.
        env = ArmTaskEnv(goal_direction="EAST")
        self.arm_visualizer = ArmVisualizer(
            link_lengths=np.asarray(env.config.link_lengths, dtype=float),
            dof=env.config.dof,
        )
        self.default_shoulder = np.asarray(env.shoulder_base_position, dtype=float)
        self.default_goal_height = float(env.goal_height)
        self.default_goal_position = np.asarray(env.goal_position, dtype=float)
        self.default_goal_direction = str(env.goal_direction)
        env.close()

        # Create root window
        self.root = tk.Tk()
        self.root.title("RL Training Dashboard - Arm Motion")
        self.root.geometry("1500x860")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Threading components
        self.metrics_queue = queue.Queue(maxsize=200)
        self.training_thread: Optional[threading.Thread] = None
        self.trainer: Optional[RLTrainerWithMetrics] = None
        self.training_active = False

        # Data buffers for plots
        self.episode_numbers = deque(maxlen=1000)
        self.episode_rewards = deque(maxlen=1000)
        self.moving_average = deque(maxlen=1000)
        self.policy_losses = deque(maxlen=10000)
        self.value_losses = deque(maxlen=10000)
        self.entropies = deque(maxlen=10000)

        # Matplotlib figures
        self.fig_rewards: Optional[Figure] = None
        self.fig_losses: Optional[Figure] = None
        self.fig_entropy: Optional[Figure] = None
        self.fig_arm: Optional[Figure] = None

        # Axes
        self.ax_rewards = None
        self.ax_losses_policy = None
        self.ax_losses_value = None
        self.ax_entropy = None
        self.ax_arm = None

        # Canvas references
        self.canvas_rewards = None
        self.canvas_losses = None
        self.canvas_entropy = None
        self.canvas_arm = None

        # UI state
        self.status_text = None
        self.metrics_text = None
        self.training_button = None
        self.stop_button = None
        self.save_button = None
        self.algorithm_combo = None
        self.timesteps_entry = None
        self.goal_direction_combo = None
        self.open_config_button = None
        self.config_name_entry = None
        self.episode_counter = 0
        self.start_time: Optional[datetime] = None
        self.algorithm_var = tk.StringVar(value=self.selected_algorithm)
        self.timesteps_var = tk.StringVar(value=f"{self.total_timesteps}")
        self.goal_direction_var = tk.StringVar(value="EAST")
        self.selected_config_name_var = tk.StringVar(value=str(env.config.name))
        self.selected_config_path: Optional[str] = None
        self.selected_arm_config = env.config
        self.selected_goal_direction = "EAST"

        # Goal-mode state. The training GUI now supports three ways of choosing
        # the agent's training target:
        #   "Direction"    — pre-defined EAST/WEST/NORTH (legacy behaviour)
        #   "Single Point" — user clicks once on the arm-visualisation canvas
        #                    to place a single goal at an arbitrary (x, y)
        #   "Waypoints"    — user clicks multiple times to place a sequence
        #                    A -> B -> C -> ... that the agent must visit in
        #                    order (intermediate waypoints require touch-and-
        #                    go; the final waypoint requires the full hold)
        self.GOAL_MODES = ["Direction", "Single Point", "Waypoints"]
        self.goal_mode_var = tk.StringVar(value="Direction")
        self.goal_mode_combo = None
        self.clear_points_button = None
        self.clicked_target: Optional[np.ndarray] = None
        self.clicked_waypoints: List[np.ndarray] = []

        # Actuation mode (Phase 6). Default "velocity" matches the historical
        # action space (direct velocity commands in [-1, 1]). "muscle"
        # switches to the Hill-type antagonist-pair action space ([0, 1]
        # extensor/flexor activations per joint) and routes the env's step
        # through Hill-type muscle dynamics following Fischer et al. (2021).
        self.ACTUATION_MODES = ["velocity", "muscle"]
        self.actuation_mode_var = tk.StringVar(value="velocity")
        self.actuation_mode_combo = None
        # matplotlib connection id for the canvas click handler; None when
        # the handler is not currently attached (Direction mode or training
        # is in progress).
        self.click_cid: Optional[int] = None

        self.create_window()

    def create_window(self) -> None:
        """Create GUI layout with plots and controls."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel: Training curves
        left_frame = ttk.LabelFrame(main_frame, text="Training Curves", padding=5)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.fig_rewards = Figure(figsize=(6, 2.8), dpi=100)
        self.ax_rewards = self.fig_rewards.add_subplot(111)
        self.ax_rewards.set_xlabel("Episode")
        self.ax_rewards.set_ylabel("Reward")
        self.ax_rewards.set_title("Episode Rewards & Moving Average")
        self.ax_rewards.grid(True, alpha=0.3)
        self.canvas_rewards = FigureCanvasTkAgg(self.fig_rewards, master=left_frame)
        self.canvas_rewards.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_losses = Figure(figsize=(6, 2.8), dpi=100)
        self.ax_losses_policy = self.fig_losses.add_subplot(121)
        self.ax_losses_policy.set_xlabel("Step")
        self.ax_losses_policy.set_ylabel("Loss")
        self.ax_losses_policy.set_title("Policy/Actor Loss")
        self.ax_losses_policy.grid(True, alpha=0.3)

        self.ax_losses_value = self.fig_losses.add_subplot(122)
        self.ax_losses_value.set_xlabel("Step")
        self.ax_losses_value.set_ylabel("Loss")
        self.ax_losses_value.set_title("Value/Critic Loss")
        self.ax_losses_value.grid(True, alpha=0.3)

        self.canvas_losses = FigureCanvasTkAgg(self.fig_losses, master=left_frame)
        self.canvas_losses.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_entropy = Figure(figsize=(6, 2.8), dpi=100)
        self.ax_entropy = self.fig_entropy.add_subplot(111)
        self.ax_entropy.set_xlabel("Step")
        self.ax_entropy.set_ylabel("Entropy")
        self.ax_entropy.set_title("Policy Entropy")
        self.ax_entropy.grid(True, alpha=0.3)

        self.canvas_entropy = FigureCanvasTkAgg(self.fig_entropy, master=left_frame)
        self.canvas_entropy.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Center panel: Arm visualization
        center_frame = ttk.LabelFrame(main_frame, text="Arm Visualization", padding=5)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        model_frame = ttk.Frame(center_frame)
        model_frame.pack(fill=tk.X, padx=2, pady=2)

        self.open_config_button = ttk.Button(
            model_frame,
            text="Open Saved Arm Config",
            command=self._on_open_saved_arm_config,
        )
        self.open_config_button.pack(side=tk.LEFT, padx=(0, 6))

        self.config_name_entry = ttk.Entry(
            model_frame,
            textvariable=self.selected_config_name_var,
            state="readonly",
        )
        self.config_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.fig_arm = Figure(figsize=(5.5, 8), dpi=100)
        self.ax_arm = self.fig_arm.add_subplot(111)
        self.canvas_arm = FigureCanvasTkAgg(self.fig_arm, master=center_frame)
        self.canvas_arm.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_arm_pose()

        # Right panel: setup, metrics and controls
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)

        setup_frame = ttk.LabelFrame(right_frame, text="Training Setup", padding=5)
        setup_frame.pack(fill=tk.X, pady=5)

        ttk.Label(setup_frame, text="Algorithm:").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self.algorithm_combo = ttk.Combobox(
            setup_frame,
            textvariable=self.algorithm_var,
            values=self.ALGORITHMS,
            state="readonly",
            width=12,
        )
        self.algorithm_combo.grid(row=0, column=1, sticky="ew", padx=2, pady=2)

        ttk.Label(setup_frame, text="Timesteps:").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        self.timesteps_entry = ttk.Entry(setup_frame, textvariable=self.timesteps_var, width=14)
        self.timesteps_entry.grid(row=1, column=1, sticky="ew", padx=2, pady=2)

        ttk.Label(setup_frame, text="Goal Mode:").grid(row=2, column=0, sticky="w", padx=2, pady=2)
        self.goal_mode_combo = ttk.Combobox(
            setup_frame,
            textvariable=self.goal_mode_var,
            values=self.GOAL_MODES,
            state="readonly",
            width=12,
        )
        self.goal_mode_combo.grid(row=2, column=1, sticky="ew", padx=2, pady=2)
        self.goal_mode_combo.bind("<<ComboboxSelected>>", self._on_goal_mode_changed)

        ttk.Label(setup_frame, text="Goal Direction:").grid(row=3, column=0, sticky="w", padx=2, pady=2)
        self.goal_direction_combo = ttk.Combobox(
            setup_frame,
            textvariable=self.goal_direction_var,
            values=self.GOAL_DIRECTIONS,
            state="readonly",
            width=12,
        )
        self.goal_direction_combo.grid(row=3, column=1, sticky="ew", padx=2, pady=2)
        self.goal_direction_combo.bind("<<ComboboxSelected>>", self._on_goal_direction_changed)

        ttk.Label(setup_frame, text="Actuation:").grid(row=4, column=0, sticky="w", padx=2, pady=2)
        self.actuation_mode_combo = ttk.Combobox(
            setup_frame,
            textvariable=self.actuation_mode_var,
            values=self.ACTUATION_MODES,
            state="readonly",
            width=12,
        )
        self.actuation_mode_combo.grid(row=4, column=1, sticky="ew", padx=2, pady=2)

        # "Clear Points" only does anything in Single Point or Waypoints modes;
        # it is grayed out in Direction mode but always present so the layout
        # does not shift when the mode changes.
        self.clear_points_button = ttk.Button(
            setup_frame,
            text="Clear Click Targets",
            command=self._on_clear_points,
            state=tk.DISABLED,
        )
        self.clear_points_button.grid(row=5, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        setup_frame.columnconfigure(1, weight=1)

        metrics_frame = ttk.LabelFrame(right_frame, text="Training Metrics", padding=5)
        metrics_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.metrics_text = tk.Text(
            metrics_frame,
            width=36,
            height=20,
            font=("Courier", 9),
            state=tk.DISABLED,
        )
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

        status_frame = ttk.LabelFrame(right_frame, text="Status", padding=5)
        status_frame.pack(fill=tk.X, pady=5)

        self.status_text = tk.Label(
            status_frame,
            text="Ready to train",
            font=("Courier", 10),
            fg="green",
        )
        self.status_text.pack(fill=tk.X)

        control_frame = ttk.LabelFrame(right_frame, text="Controls", padding=5)
        control_frame.pack(fill=tk.X, pady=5)

        self.training_button = ttk.Button(
            control_frame,
            text="Start Training",
            command=self._on_start_training,
        )
        self.training_button.pack(fill=tk.X, pady=2)

        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Training",
            command=self._on_stop_training,
            state=tk.DISABLED,
        )
        self.stop_button.pack(fill=tk.X, pady=2)

        self.save_button = ttk.Button(
            control_frame,
            text="Save Model & Results",
            command=self._on_save_model,
            state=tk.DISABLED,
        )
        self.save_button.pack(fill=tk.X, pady=2)

        # Run-validators button: runs Fitts' Law and 2/3 Power Law harnesses
        # against the most recent trained model and writes JSON + PNG output
        # to a user-chosen directory (default = the model's save dir).
        # Disabled until a training run has completed.
        self.run_validators_button = ttk.Button(
            control_frame,
            text="Run Fischer Validators",
            command=self._on_run_validators,
            state=tk.DISABLED,
        )
        self.run_validators_button.pack(fill=tk.X, pady=2)
        self.validators_thread: Optional[threading.Thread] = None
        self.last_save_dir: Optional[str] = None

        self._on_goal_direction_changed()
        self._schedule_metrics_check()

    def _set_setup_controls_enabled(self, enabled: bool) -> None:
        """Enable/disable algorithm and timestep controls while training."""
        state = "readonly" if enabled else "disabled"
        entry_state = "normal" if enabled else "disabled"
        self.algorithm_combo.config(state=state)
        self.timesteps_entry.config(state=entry_state)
        self.goal_mode_combo.config(state=state)
        if self.actuation_mode_combo is not None:
            self.actuation_mode_combo.config(state=state)
        # Goal direction follows the mode: only enabled when mode == Direction
        # and the user is not currently training.
        mode = self.goal_mode_var.get()
        if enabled and mode == "Direction":
            self.goal_direction_combo.config(state="readonly")
        else:
            self.goal_direction_combo.config(state="disabled")
        # Clear-points button only meaningful in click modes; always off while training.
        if enabled and mode in {"Single Point", "Waypoints"}:
            self.clear_points_button.config(state=tk.NORMAL)
        else:
            self.clear_points_button.config(state=tk.DISABLED)
        # Canvas clicks are accepted only while NOT training and only in click modes.
        if enabled and mode in {"Single Point", "Waypoints"}:
            self._enable_canvas_clicks()
        else:
            self._disable_canvas_clicks()

    def _on_goal_mode_changed(self, _event: Optional[tk.Event] = None) -> None:
        """Switch between Direction, Single Point, and Waypoints goal modes.

        Side effects: enables/disables the relevant widgets (the Goal
        Direction dropdown is only meaningful in Direction mode; the Clear
        Points button is only meaningful in click modes), attaches or
        detaches the canvas click handler, and refreshes the arm-pose
        preview so the user sees the visual state of their chosen mode
        immediately.
        """
        mode = self.goal_mode_var.get()
        if mode == "Direction":
            self.goal_direction_combo.config(state="readonly")
            self.clear_points_button.config(state=tk.DISABLED)
            self._disable_canvas_clicks()
        else:
            self.goal_direction_combo.config(state="disabled")
            self.clear_points_button.config(state=tk.NORMAL)
            self._enable_canvas_clicks()

        # Refresh the preview so the user can see what the chosen mode looks
        # like (no goal yet for Single Point / Waypoints if they haven't
        # clicked, or the existing direction goal otherwise).
        self._on_goal_direction_changed()

    def _enable_canvas_clicks(self) -> None:
        """Connect the matplotlib click handler if not already attached."""
        if self.click_cid is None and self.canvas_arm is not None:
            self.click_cid = self.canvas_arm.mpl_connect(
                "button_press_event", self._on_canvas_click
            )

    def _disable_canvas_clicks(self) -> None:
        """Disconnect the matplotlib click handler if currently attached."""
        if self.click_cid is not None and self.canvas_arm is not None:
            try:
                self.canvas_arm.mpl_disconnect(self.click_cid)
            except Exception:
                pass
            self.click_cid = None

    def _on_canvas_click(self, event) -> None:
        """Convert a click on the arm-visualisation canvas into a goal position.

        Only acts on clicks inside the arm-pose axes. Rejects clicks outside
        the arm's reachable workspace (the distance from the shoulder must
        not exceed the sum of link lengths). In Single Point mode a click
        replaces any existing target. In Waypoints mode a click appends to
        the ordered waypoint list.
        """
        if self.training_active:
            return
        if event.inaxes is not self.ax_arm:
            return
        if event.xdata is None or event.ydata is None:
            return

        point = np.array([float(event.xdata), float(event.ydata)], dtype=np.float32)

        shoulder = np.asarray(self.default_shoulder, dtype=float)
        reach = float(np.sum(self.arm_visualizer.link_lengths))
        distance_from_shoulder = float(np.linalg.norm(point - shoulder))
        if distance_from_shoulder > reach:
            self.status_text.config(
                text=(
                    f"Click outside reachable workspace (distance "
                    f"{distance_from_shoulder:.2f} m exceeds reach "
                    f"{reach:.2f} m)"
                ),
                fg="red",
            )
            return

        mode = self.goal_mode_var.get()
        if mode == "Single Point":
            self.clicked_target = point
            self.status_text.config(
                text=(
                    f"Single goal set at ({point[0]:.2f}, {point[1]:.2f}) m"
                ),
                fg="blue",
            )
        elif mode == "Waypoints":
            self.clicked_waypoints.append(point)
            self.status_text.config(
                text=(
                    f"Waypoint {len(self.clicked_waypoints)} added at "
                    f"({point[0]:.2f}, {point[1]:.2f}) m"
                ),
                fg="blue",
            )
        else:
            return

        # Re-render with the new target(s) visible.
        self._on_goal_direction_changed()

    def _on_clear_points(self) -> None:
        """Discard all click-set targets in the current mode."""
        if self.training_active:
            return
        self.clicked_target = None
        self.clicked_waypoints = []
        self.status_text.config(
            text="Click targets cleared. Click the arm canvas to set a new goal.",
            fg="black",
        )
        self._on_goal_direction_changed()

    def _update_default_goal_target(self, direction: str) -> None:
        """Update default directional target using current arm reach."""
        direction = str(direction).strip().upper()
        max_reach = float(np.sum(self.arm_visualizer.link_lengths))
        shoulder = np.asarray(self.default_shoulder, dtype=float)

        if direction == "EAST":
            goal = np.array([shoulder[0] + max_reach, shoulder[1]], dtype=float)
        elif direction == "WEST":
            goal = np.array([shoulder[0] - max_reach, shoulder[1]], dtype=float)
        elif direction == "NORTH":
            goal = np.array([shoulder[0], shoulder[1] + max_reach], dtype=float)
        else:
            goal = np.array([shoulder[0], shoulder[1]], dtype=float)
            direction = "HEIGHT"

        self.default_goal_position = goal
        self.default_goal_height = float(goal[1])
        self.default_goal_direction = direction

    def _on_goal_direction_changed(self, _event: Optional[tk.Event] = None) -> None:
        """Refresh preview when goal direction changes."""
        direction = self.goal_direction_var.get().strip().upper()
        self._update_default_goal_target(direction)
        initial_angles = np.asarray(
            self.selected_arm_config.initial_angles[:2],
            dtype=float,
        )
        preview_payload = {
            "joint_angles": initial_angles,
            "shoulder_position": self.default_shoulder,
            "goal_height": self.default_goal_height,
            "goal_position": self.default_goal_position,
            "goal_direction": direction,
        }
        self._draw_arm_pose(preview_payload)

    def _on_open_saved_arm_config(self) -> None:
        """Open saved arm configuration for visualization panel."""
        default_dir = Path("arm_configuration")
        initial_dir = str(default_dir if default_dir.exists() else Path.cwd())
        selected = filedialog.askopenfilename(
            title="Open saved arm configuration",
            initialdir=initial_dir,
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return

        try:
            config = ArmConfiguration.from_json(selected)
            if int(config.dof) != 2:
                messagebox.showerror("Invalid Configuration", "Training GUI supports only 2-DOF arm configurations")
                return

            self.selected_arm_config = config
            self.selected_config_path = selected
            self.selected_config_name_var.set(str(config.name))
            self.arm_visualizer = ArmVisualizer(
                link_lengths=np.asarray(config.link_lengths, dtype=float),
                dof=int(config.dof),
            )
            self._on_goal_direction_changed()
            self.status_text.config(text=f"Loaded arm config: {config.name}", fg="green")
        except Exception as exc:
            messagebox.showerror("Load Error", f"Failed to load arm configuration:\n{exc}")

    def _on_start_training(self) -> None:
        """Start training in background thread."""
        if self.training_active:
            messagebox.showwarning("Warning", "Training already in progress")
            return

        try:
            timesteps = int(str(self.timesteps_var.get()).replace(",", "").strip())
            if timesteps <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Timesteps must be a positive integer")
            return

        algorithm = self.algorithm_var.get().strip().upper()
        if algorithm not in self.ALGORITHMS:
            messagebox.showerror("Invalid Input", f"Algorithm must be one of: {', '.join(self.ALGORITHMS)}")
            return

        goal_direction = self.goal_direction_var.get().strip().upper()
        if goal_direction not in self.GOAL_DIRECTIONS:
            messagebox.showerror("Invalid Input", f"Goal direction must be one of: {', '.join(self.GOAL_DIRECTIONS)}")
            return

        self.total_timesteps = timesteps
        self.selected_algorithm = algorithm
        self.selected_goal_direction = goal_direction
        self.training_active = True
        self.episode_counter = 0
        self.start_time = datetime.now()

        self.training_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)
        self.run_validators_button.config(state=tk.DISABLED)
        self._set_setup_controls_enabled(False)
        self.status_text.config(text=f"Training {algorithm} ({goal_direction}) in progress...", fg="blue")

        self.episode_numbers.clear()
        self.episode_rewards.clear()
        self.moving_average.clear()
        self.policy_losses.clear()
        self.value_losses.clear()
        self.entropies.clear()

        self.training_thread = threading.Thread(
            target=self._training_loop,
            daemon=True,
        )
        self.training_thread.start()

    def _on_stop_training(self) -> None:
        """Request training stop."""
        if not self.training_active:
            return

        self.training_active = False
        if self.trainer is not None:
            self.trainer.request_stop()

        self.status_text.config(text="Stopping...", fg="orange")
        self.stop_button.config(state=tk.DISABLED)

    def _on_save_model(self) -> None:
        """Save trained model and results."""
        if self.trainer is None:
            messagebox.showerror("Error", "No trained model available")
            return

        save_dir = filedialog.askdirectory(
            title="Select directory to save model",
            initialdir=self.last_save_dir or self.save_dir,
        )

        if not save_dir:
            return

        try:
            self.status_text.config(text="Saving model...", fg="blue")
            self.root.update()

            save_paths = self.trainer.save_model_and_results(save_dir)
            self._save_plots(save_dir)
            self.last_save_dir = save_dir

            self.status_text.config(
                text=f"Model saved to {save_dir}",
                fg="green",
            )

            messagebox.showinfo(
                "Success",
                f"Model saved to {save_dir}\n\n"
                f"Files:\n"
                f"- Model: {save_paths['model']}\n"
                f"- History: {save_paths['history']}\n"
                f"- Stats: {save_paths['stats']}",
            )

        except Exception as exc:
            self.status_text.config(text="Save failed", fg="red")
            messagebox.showerror("Error", f"Failed to save model:\n{str(exc)}")

    def _on_run_validators(self) -> None:
        """Run the Fitts' Law and 2/3 Power Law validation harnesses on the
        most recent trained model and save JSON results plus PNG plots to a
        user-chosen directory.

        The harnesses execute in a background thread so the GUI stays
        responsive. Progress is reported via the status bar; the final
        result (success or error) is shown in a message box.
        """
        if self.trainer is None or getattr(self.trainer, "trainer", None) is None:
            messagebox.showerror("Error", "No trained model available")
            return
        if getattr(self.trainer.trainer, "model", None) is None:
            messagebox.showerror("Error", "Trainer has no model attribute")
            return
        if self.validators_thread is not None and self.validators_thread.is_alive():
            messagebox.showinfo(
                "Validators already running",
                "A validator run is already in progress. Please wait.",
            )
            return

        output_dir = filedialog.askdirectory(
            title="Choose directory to save validator results",
            initialdir=self.last_save_dir or self.save_dir,
        )
        if not output_dir:
            return

        # Disable the button and start the worker thread.
        self.run_validators_button.config(state=tk.DISABLED)
        self.status_text.config(
            text="Running Fischer validators (this may take ~1 minute)...",
            fg="blue",
        )
        self.last_save_dir = output_dir

        self.validators_thread = threading.Thread(
            target=self._validators_worker,
            args=(output_dir,),
            daemon=True,
        )
        self.validators_thread.start()

    def _validators_worker(self, output_dir: str) -> None:
        """Background-thread body that actually runs the two validators."""
        from rl_armMotion.two_d.validation import (
            FittsLawValidator,
            PowerLawValidator,
        )

        try:
            model = self.trainer.trainer.model
            # Use a fresh env so the validators do not interfere with the
            # trainer's in-memory env state.
            env = ArmTaskEnv()
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # --- Fitts' Law sweep -----------------------------------------
            fl = FittsLawValidator(model=model, env=env)
            fl_result = fl.run(
                n_trials_per_condition=10,
                max_steps_per_trial=200,
                seed=42,
            )
            fl_json = output_path / f"fitts_law_{timestamp}.json"
            fl_png = output_path / f"fitts_law_{timestamp}.png"
            fl_result.save_json(str(fl_json))
            fl_result.plot(
                save_path=str(fl_png),
                show=False,
                title=f"Fitts' Law - {self.selected_algorithm}",
            )
            fl_summary = fl_result.regression_summary()

            # --- 2/3 Power Law sweep --------------------------------------
            pl = PowerLawValidator(model=model, env=env)
            pl_result = pl.run(
                n_trials=20,
                max_steps_per_trial=200,
                seed=42,
            )
            pl_json = output_path / f"power_law_{timestamp}.json"
            pl_png = output_path / f"power_law_{timestamp}.png"
            pl_result.save_json(str(pl_json))
            pl_result.plot(
                save_path=str(pl_png),
                show=False,
                title=f"2/3 Power Law - {self.selected_algorithm}",
            )
            pl_summary = pl_result.regression_summary()

            # Schedule GUI updates on the main thread.
            self.root.after(
                0,
                self._on_validators_complete,
                fl_summary, pl_summary,
                str(fl_json), str(fl_png), str(pl_json), str(pl_png),
            )
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            self.root.after(0, self._on_validators_error, err_msg)

    def _on_validators_complete(
        self,
        fl_summary: str,
        pl_summary: str,
        fl_json: str,
        fl_png: str,
        pl_json: str,
        pl_png: str,
    ) -> None:
        """Main-thread callback when the validators finish successfully."""
        self.run_validators_button.config(state=tk.NORMAL)
        self.status_text.config(text="Fischer validators completed", fg="green")
        messagebox.showinfo(
            "Validators Complete",
            "Fitts' Law:\n"
            f"  {fl_summary}\n\n"
            "Two-thirds Power Law:\n"
            f"  {pl_summary}\n\n"
            "Saved files:\n"
            f"- {fl_json}\n"
            f"- {fl_png}\n"
            f"- {pl_json}\n"
            f"- {pl_png}",
        )

    def _on_validators_error(self, error: str) -> None:
        """Main-thread callback when the validators raise an exception."""
        self.run_validators_button.config(state=tk.NORMAL)
        self.status_text.config(text="Validators failed", fg="red")
        messagebox.showerror(
            "Validators Error",
            f"Failed to run validators:\n{error}",
        )

    def _training_loop(self) -> None:
        """Background training thread execution."""
        try:
            mode = self.goal_mode_var.get()
            actuation = self.actuation_mode_var.get().strip().lower()
            if actuation not in self.ACTUATION_MODES:
                actuation = "velocity"

            if mode == "Single Point":
                if self.clicked_target is None:
                    self.metrics_queue.put({
                        "type": "training_error",
                        "error": (
                            "Single Point mode requires a clicked target. "
                            "Click on the arm visualisation panel to place one."
                        ),
                    })
                    return
                training_env = ArmTaskEnv(actuation_mode=actuation)
                training_env.set_goal_position(self.clicked_target)
            elif mode == "Waypoints":
                if not self.clicked_waypoints:
                    self.metrics_queue.put({
                        "type": "training_error",
                        "error": (
                            "Waypoints mode requires at least one clicked "
                            "waypoint. Click the arm canvas to add points."
                        ),
                    })
                    return
                training_env = ArmTaskEnv(actuation_mode=actuation)
                training_env.set_waypoints(self.clicked_waypoints)
            else:
                training_env = ArmTaskEnv(
                    goal_direction=self.selected_goal_direction,
                    actuation_mode=actuation,
                )

            self.trainer = RLTrainerWithMetrics(
                env=training_env,
                total_timesteps=self.total_timesteps,
                algorithm=self.selected_algorithm,
                metrics_queue=self.metrics_queue,
                should_stop=lambda: not self.training_active,
                check_freq=100,
            )

            result = self.trainer.train()

            stopped_early = bool(result.get("stopped_early", False)) or not self.training_active
            if stopped_early:
                self.metrics_queue.put({"type": "training_stopped", "result": result})
            else:
                self.metrics_queue.put({"type": "training_complete", "result": result})

        except Exception as exc:
            self.metrics_queue.put({
                "type": "training_error",
                "error": str(exc),
            })
        finally:
            self.training_active = False

    def _schedule_metrics_check(self) -> None:
        """Non-blocking check for queued metrics."""
        try:
            while True:
                metrics = self.metrics_queue.get_nowait()
                self._process_metrics(metrics)
        except queue.Empty:
            pass

        self.root.after(100, self._schedule_metrics_check)

    def _process_metrics(self, metrics: Dict[str, Any]) -> None:
        """Process metrics from queue."""
        msg_type = metrics.get("type", "metrics")

        if msg_type in {"episode_completed", "metrics_update"}:
            self._update_from_episode_data(metrics)
        elif msg_type == "training_complete":
            self._on_training_complete(metrics.get("result", {}))
        elif msg_type == "training_stopped":
            self._on_training_stopped(metrics.get("result", {}))
        elif msg_type == "training_error":
            self._on_training_error(metrics.get("error", "Unknown error"))

    def _update_from_episode_data(self, metrics: Dict[str, Any]) -> None:
        """Update GUI from training metrics payload."""
        episode_rewards = metrics.get("episode_rewards", []) or []
        policy_losses = metrics.get("policy_losses", []) or []
        value_losses = metrics.get("value_losses", []) or []
        entropies = metrics.get("entropies", []) or []

        # Replace series from payload to avoid duplicate accumulation.
        self.episode_rewards.clear()
        self.episode_rewards.extend(list(episode_rewards)[-1000:])

        self.episode_numbers.clear()
        self.episode_numbers.extend(range(1, len(self.episode_rewards) + 1))

        self.moving_average.clear()
        reward_list = list(self.episode_rewards)
        for idx in range(len(reward_list)):
            start = max(0, idx - 99)
            self.moving_average.append(float(np.mean(reward_list[start: idx + 1])))

        self.policy_losses.clear()
        self.policy_losses.extend(list(policy_losses)[-10000:])

        self.value_losses.clear()
        self.value_losses.extend(list(value_losses)[-10000:])

        self.entropies.clear()
        self.entropies.extend(list(entropies)[-10000:])

        self.episode_counter = int(metrics.get("episodes", len(self.episode_rewards)))

        self._update_plots()
        self._draw_arm_pose(metrics)
        self._update_metrics_display(metrics)

    def _update_plots(self) -> None:
        """Update training curve plots."""
        try:
            self.ax_rewards.clear()
            if self.episode_rewards:
                ep_list = list(self.episode_numbers)
                reward_list = list(self.episode_rewards)
                ma_list = list(self.moving_average)

                self.ax_rewards.plot(ep_list, reward_list, alpha=0.35, label="Episode Reward")
                self.ax_rewards.plot(ep_list, ma_list, linewidth=2, label="100-ep Moving Avg")
                self.ax_rewards.legend(loc="best")

            self.ax_rewards.set_xlabel("Episode")
            self.ax_rewards.set_ylabel("Reward")
            self.ax_rewards.set_title("Episode Rewards & Moving Average")
            self.ax_rewards.grid(True, alpha=0.3)
            self.canvas_rewards.draw_idle()
        except Exception:
            pass

        try:
            self.ax_losses_policy.clear()
            if self.policy_losses:
                steps = list(range(len(self.policy_losses)))
                self.ax_losses_policy.plot(steps, list(self.policy_losses), linewidth=1)

            self.ax_losses_policy.set_xlabel("Step")
            self.ax_losses_policy.set_ylabel("Loss")
            self.ax_losses_policy.set_title("Policy/Actor Loss")
            self.ax_losses_policy.grid(True, alpha=0.3)

            self.ax_losses_value.clear()
            if self.value_losses:
                steps = list(range(len(self.value_losses)))
                self.ax_losses_value.plot(steps, list(self.value_losses), linewidth=1)

            self.ax_losses_value.set_xlabel("Step")
            self.ax_losses_value.set_ylabel("Loss")
            self.ax_losses_value.set_title("Value/Critic Loss")
            self.ax_losses_value.grid(True, alpha=0.3)

            self.canvas_losses.draw_idle()
        except Exception:
            pass

        try:
            self.ax_entropy.clear()
            if self.entropies:
                steps = list(range(len(self.entropies)))
                self.ax_entropy.plot(steps, list(self.entropies), linewidth=1, color="green")

            self.ax_entropy.set_xlabel("Step")
            self.ax_entropy.set_ylabel("Entropy")
            self.ax_entropy.set_title("Policy Entropy")
            self.ax_entropy.grid(True, alpha=0.3)
            self.canvas_entropy.draw_idle()
        except Exception:
            pass

    def _draw_arm_pose(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Draw the latest arm pose in workspace coordinates."""
        try:
            payload = metrics or {}
            joint_angles = np.asarray(
                payload.get("joint_angles", np.array([-np.pi / 2, 0.0])),
                dtype=float,
            )
            shoulder = np.asarray(
                payload.get("shoulder_position", self.default_shoulder),
                dtype=float,
            )
            goal_height = float(payload.get("goal_height", self.default_goal_height))
            goal_direction = str(payload.get("goal_direction", self.goal_direction_var.get().strip().upper()))
            goal_position = np.asarray(
                payload.get("goal_position", self.default_goal_position),
                dtype=float,
            )

            positions = self.arm_visualizer.forward_kinematics(joint_angles)
            positions[:, 0] += shoulder[0]
            positions[:, 1] += shoulder[1]

            ee_from_info = payload.get("end_effector_position")
            if ee_from_info is not None:
                ee = np.asarray(ee_from_info, dtype=float)
            else:
                ee = positions[-1, :2]

            self.ax_arm.clear()
            self.ax_arm.plot(positions[:, 0], positions[:, 1], "b-o", linewidth=2)
            self.ax_arm.scatter(shoulder[0], shoulder[1], c="green", s=100, label="Shoulder")
            self.ax_arm.scatter(ee[0], ee[1], c="red", s=120, marker="*", label="End-effector")

            # In click modes (Single Point / Waypoints) the user-clicked
            # targets are the source of truth and replace the direction-based
            # goal visualisation; otherwise fall back to the legacy direction-
            # based rendering.
            mode = self.goal_mode_var.get() if self.goal_mode_var is not None else "Direction"

            if mode == "Single Point":
                # Show the clicked target (red X) if one has been placed.
                target = self.clicked_target
                if target is not None:
                    self.ax_arm.scatter(
                        target[0], target[1], c="red", s=180, marker="X",
                        edgecolors="black", linewidths=1.5, zorder=10,
                        label="Goal (clicked)",
                    )
                    self.ax_arm.plot(
                        [shoulder[0], target[0]], [shoulder[1], target[1]],
                        color="red", linestyle="--", linewidth=1.0, alpha=0.6,
                    )
                else:
                    # Hint the user that they need to click.
                    self.ax_arm.text(
                        shoulder[0], shoulder[1] - 0.3,
                        "Click on the canvas to set a single goal",
                        ha="center", va="center", color="gray", fontsize=9, style="italic",
                    )

            elif mode == "Waypoints":
                # Show the full waypoint sequence with the active one highlighted.
                # During training the env reports current_waypoint_index in the
                # payload; before training we treat waypoint 0 as the active one.
                current_idx = int(payload.get("current_waypoint_index", 0))
                waypoints = self.clicked_waypoints
                if waypoints:
                    wp_arr = np.asarray(waypoints, dtype=float)
                    # Connecting line through the sequence
                    if len(waypoints) >= 2:
                        self.ax_arm.plot(
                            wp_arr[:, 0], wp_arr[:, 1],
                            color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
                            zorder=5,
                        )
                    # Markers
                    for i, wp in enumerate(waypoints):
                        if i < current_idx:
                            colour = "green"        # already visited
                        elif i == current_idx:
                            colour = "red"          # active waypoint
                        else:
                            colour = "lightgray"    # pending
                        label = (
                            "Active waypoint" if (i == current_idx and i == 0)
                            else None
                        )
                        self.ax_arm.scatter(
                            wp[0], wp[1], c=colour, s=160, marker="X",
                            edgecolors="black", linewidths=1.5, zorder=10,
                            label=label,
                        )
                        self.ax_arm.annotate(
                            f"{i+1}",
                            (wp[0], wp[1]),
                            textcoords="offset points",
                            xytext=(8, 8),
                            fontsize=10, fontweight="bold",
                        )
                else:
                    self.ax_arm.text(
                        shoulder[0], shoulder[1] - 0.3,
                        "Click on the canvas to add waypoints (A -> B -> C ...)",
                        ha="center", va="center", color="gray", fontsize=9, style="italic",
                    )

            else:
                # Legacy Direction mode rendering.
                if goal_direction == "HEIGHT":
                    self.ax_arm.axhline(
                        goal_height,
                        color="orange",
                        linestyle="--",
                        linewidth=1.5,
                        label="Goal Height",
                    )
                else:
                    self.ax_arm.scatter(
                        goal_position[0],
                        goal_position[1],
                        c="orange",
                        s=100,
                        marker="X",
                        label=f"Goal ({goal_direction})",
                    )
                    self.ax_arm.plot(
                        [shoulder[0], goal_position[0]],
                        [shoulder[1], goal_position[1]],
                        color="orange",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.7,
                    )

            reach = float(np.sum(self.arm_visualizer.link_lengths)) + 0.4
            self.ax_arm.set_xlim(shoulder[0] - reach, shoulder[0] + reach)
            self.ax_arm.set_ylim(shoulder[1] - reach, shoulder[1] + reach)
            self.ax_arm.set_aspect("equal")
            title_suffix = (
                f"Waypoints ({int(payload.get('current_waypoint_index', 0))+1}/"
                f"{len(self.clicked_waypoints)})"
                if mode == "Waypoints" and self.clicked_waypoints
                else (mode if mode != "Direction" else goal_direction)
            )
            self.ax_arm.set_title(f"Policy Execution ({title_suffix})")
            self.ax_arm.set_xlabel("X (m)")
            self.ax_arm.set_ylabel("Y (m)")
            self.ax_arm.grid(True, alpha=0.3)
            self.ax_arm.legend(loc="upper right")
            self.canvas_arm.draw_idle()
        except Exception:
            pass

    def _update_metrics_display(self, metrics: Dict[str, Any]) -> None:
        """Update metrics text display."""
        try:
            elapsed = 0.0
            if self.start_time:
                elapsed = (datetime.now() - self.start_time).total_seconds()

            goal_pos = np.asarray(
                metrics.get("goal_position", self.default_goal_position),
                dtype=float,
            )
            goal_pos_text = f"[{goal_pos[0]:.3f}, {goal_pos[1]:.3f}]"

            # Curriculum block (Fischer et al. 2021 adaptive curriculum).
            # Only rendered if the trainer reports curriculum is enabled.
            curriculum = metrics.get("curriculum", {}) or {}
            if curriculum.get("enabled", False):
                cur_tol = curriculum.get("current_tolerance")
                init_tol = curriculum.get("initial_tolerance")
                min_tol = curriculum.get("min_tolerance")
                sr = curriculum.get("recent_success_rate")
                sr_text = f"{sr * 100:.1f}%" if isinstance(sr, (int, float)) else "N/A"
                stage = curriculum.get("curriculum_stage", 0)
                wfilled = curriculum.get("window_filled", 0)
                wsize = curriculum.get("window_size", 0)
                ep_since = curriculum.get("episodes_since_last_decay", 0)
                curriculum_text = (
                    f"\n"
                    f"Curriculum (Fischer 2021):\n"
                    f"  Stage:           {stage}\n"
                    f"  Goal Tolerance:  {cur_tol:.3f}m"
                    f" (init {init_tol:.2f}m -> min {min_tol:.2f}m)\n"
                    f"  Window Filled:   {wfilled}/{wsize}\n"
                    f"  Success Rate:    {sr_text}\n"
                    f"  Since Decay:     {ep_since} ep\n"
                )
            else:
                curriculum_text = ""

            # Waypoint progress block, only rendered if the env reports
            # waypoint state in this update.
            num_wp = metrics.get("num_waypoints", 0)
            if num_wp and num_wp > 0:
                cur_wp = metrics.get("current_waypoint_index", 0)
                waypoint_text = (
                    f"\n"
                    f"Waypoint Progress:\n"
                    f"  Active:          {cur_wp + 1}/{num_wp}\n"
                )
            else:
                waypoint_text = ""

            text = (
                f"Algorithm:         {metrics.get('algorithm', self.selected_algorithm)}\n"
                f"Episode:           {self.episode_counter}\n"
                f"Timesteps:         {int(metrics.get('timesteps', 0)):,}\n"
                f"Training Time:     {elapsed:.1f}s\n"
                f"\n"
                f"Rewards:\n"
                f"  Mean (100ep):    {metrics.get('mean_reward', 0):.2f}\n"
                f"  Best:            {metrics.get('best_reward', 0):.2f}\n"
                f"\n"
                f"Task Metrics:\n"
                f"  Success Rate:    {metrics.get('success_rate', 0):.1f}%\n"
                f"  Avg Distance:    {metrics.get('avg_goal_distance', 0):.3f}m\n"
                f"  Best Distance:   {metrics.get('best_distance', 0):.3f}m\n"
                f"  Goal Direction:  {metrics.get('goal_direction', self.goal_direction_var.get())}\n"
                f"  Goal Position:   {goal_pos_text}\n"
                f"  Height Error:    {metrics.get('height_error', 0):.3f}m\n"
                f"  Orient Error:    {metrics.get('orientation_error', 0):.3f}rad\n"
                f"  Hold Progress:   {metrics.get('hold_counter', 0)}/{metrics.get('hold_steps_required', 0)}"
                f" ({metrics.get('hold_progress', 0) * 100:.1f}%)\n"
                f"  In Goal Region:  {'Yes' if metrics.get('in_goal_region', False) else 'No'}\n"
                f"  Gradient Norm:   {metrics.get('gradient_norm', 0):.3f}\n"
                f"{curriculum_text}"
                f"{waypoint_text}"
                f"\n"
                f"Optimizer Metrics:\n"
                f"  Policy/Actor:    {metrics.get('policy_loss', 0):.4f}\n"
                f"  Value/Critic:    {metrics.get('value_loss', 0):.4f}\n"
                f"  Entropy:         {metrics.get('entropy', 0):.4f}\n"
            )

            self.metrics_text.config(state=tk.NORMAL)
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(1.0, text)
            self.metrics_text.config(state=tk.DISABLED)
        except Exception:
            pass

    def _on_training_complete(self, result: Dict[str, Any]) -> None:
        """Handle training completion."""
        self.training_active = False
        self.training_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL)
        self.run_validators_button.config(state=tk.NORMAL)
        self._set_setup_controls_enabled(True)
        self.status_text.config(text="Training completed successfully", fg="green")

        messagebox.showinfo(
            "Training Complete",
            f"Algorithm: {self.selected_algorithm}\n"
            f"Episodes: {self.episode_counter}\n"
            f"Best reward: {result.get('best_reward', 0):.2f}",
        )

    def _on_training_stopped(self, _result: Optional[Dict[str, Any]] = None) -> None:
        """Handle user-requested training stop."""
        self.training_active = False
        self.training_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL if self.trainer is not None else tk.DISABLED)
        self.run_validators_button.config(
            state=tk.NORMAL if self.trainer is not None else tk.DISABLED
        )
        self._set_setup_controls_enabled(True)
        self.status_text.config(text="Training stopped", fg="orange")

    def _on_training_error(self, error: str) -> None:
        """Handle training error."""
        self.training_active = False
        self.training_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self._set_setup_controls_enabled(True)
        self.status_text.config(text="Training failed", fg="red")

        messagebox.showerror(
            "Training Error",
            f"Training failed with error:\n{error}",
        )

    def _save_plots(self, save_dir: str) -> None:
        """Save plot figures as PNG."""
        try:
            rewards_path = Path(save_dir) / "rewards.png"
            self.fig_rewards.savefig(rewards_path, dpi=150, bbox_inches="tight")

            losses_path = Path(save_dir) / "losses.png"
            self.fig_losses.savefig(losses_path, dpi=150, bbox_inches="tight")

            entropy_path = Path(save_dir) / "entropy.png"
            self.fig_entropy.savefig(entropy_path, dpi=150, bbox_inches="tight")

            arm_path = Path(save_dir) / "arm_pose.png"
            self.fig_arm.savefig(arm_path, dpi=150, bbox_inches="tight")
        except Exception:
            pass

    def _on_close(self) -> None:
        """Handle window close while training is active."""
        if self.training_active:
            if not messagebox.askyesno("Quit", "Training is still running. Stop and quit?"):
                return
            self._on_stop_training()

        self.root.destroy()

    def run(self) -> None:
        """Start GUI mainloop."""
        self.root.mainloop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL Arm Motion Training GUI")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--algorithm", type=str, default="SAC", help="Algorithm: SAC (default, per Fischer 2021) / PPO / A2C")
    parser.add_argument("--save-dir", type=str, default="./trained_models", help="Default save directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    gui = TrainingGUI(
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        algorithm=args.algorithm,
    )
    gui.run()
