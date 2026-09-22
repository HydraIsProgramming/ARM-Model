"""Training wrapper: metrics streaming for the GUI, curriculum/checkpoint
callback, run manifests and model packages.

Every run (CLI or GUI) writes a self-contained package directory (see
evaluation.py for the layout) so a saved model can be reloaded in the exact
env it was trained in, evaluated, demonstrated, or trained further with
``continue_from``.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from rl_armMotion.two_d.environments.task_env import ArmTaskEnv
from rl_armMotion.two_d.evaluation import EVAL_STANDARD, PACKAGE_FILES, EvalStandard, read_manifest
from rl_armMotion.two_d.models.callbacks import GUICallback
from rl_armMotion.two_d.models.trainers import RLTrainer
from rl_armMotion.two_d.training.curriculum_callback import (
    DEFAULT_HOLD_LADDER,
    DEFAULT_TOLERANCE_LADDER,
    EvalCheckpointCurriculumCallback,
)
from rl_armMotion.two_d.training.reproducibility import set_global_seed, write_run_manifest


class RLTrainerWithMetrics:
    """Wrapper around RLTrainer that streams metrics for GUI visualisation,
    attaches the evaluation-driven curriculum for SAC and writes packages."""

    SUPPORTED_ALGORITHMS = ("PPO", "SAC", "A2C")

    def __init__(
        self,
        env: Optional[ArmTaskEnv] = None,
        total_timesteps: int = 100000,
        algorithm: str = "PPO",
        hyperparams: Optional[Dict[str, Any]] = None,
        preload_model_path: Optional[str] = None,
        metrics_queue: Optional[queue.Queue] = None,
        metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        check_freq: int = 100,
        use_curriculum: Optional[bool] = None,
        curriculum_kwargs: Optional[Dict[str, Any]] = None,
        eval_env_fn: Optional[Callable[[], ArmTaskEnv]] = None,
        eval_save_dir: Optional[str] = None,
        run_dir: Optional[str] = None,
        seed: Optional[int] = None,
        n_envs: int = 1,
        standard: EvalStandard = EVAL_STANDARD,
        tolerance_ladder: Optional[Sequence[float]] = None,
        hold_ladder: Optional[Sequence[int]] = None,
        eval_freq: int = 20_000,
        checkpoint_every: int = 100_000,
        continue_from: Optional[str] = None,
        continue_drop_buffer: bool = False,
        advance_consecutive: int = 2,
        cooldown_rounds: int = 1,
        cli_args: Optional[Dict[str, Any]] = None,
        manifest_extra: Optional[Dict[str, Any]] = None,
        **legacy_kwargs: Any,
    ):
        for key in ("use_hold_curriculum", "orientation_ladder"):
            legacy_kwargs.pop(key, None)
        if legacy_kwargs:
            raise TypeError(f"unexpected keyword arguments: {sorted(legacy_kwargs)}")
        if env is None:
            env = ArmTaskEnv()

        self.algorithm = algorithm.upper()
        if self.algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Algorithm {self.algorithm} not supported for GUI trainer. "
                f"Choose from {list(self.SUPPORTED_ALGORITHMS)}"
            )

        self.env = env                       # template env (metrics, describe())
        self.env_description = env.describe()
        self.total_timesteps = total_timesteps
        self.preload_model_path = preload_model_path
        self.metrics_queue = metrics_queue
        self.metrics_callback = metrics_callback
        self.check_freq = check_freq
        self._external_should_stop = should_stop
        self.seed = seed
        self.n_envs = max(1, int(n_envs))
        self.standard = standard
        self.eval_freq = int(eval_freq)
        self.checkpoint_every = int(checkpoint_every)
        self.cli_args = dict(cli_args) if cli_args else {}
        self.manifest_extra = dict(manifest_extra) if manifest_extra else {}
        self.run_dir: Optional[Path] = Path(run_dir or eval_save_dir).resolve() if (run_dir or eval_save_dir) else None
        self.continue_from: Optional[Path] = Path(continue_from).resolve() if continue_from else None
        self.continue_drop_buffer = bool(continue_drop_buffer)
        self.advance_consecutive = int(advance_consecutive)
        self.cooldown_rounds = int(cooldown_rounds)
        self.lineage: List[Dict[str, Any]] = []
        self.resume_state: Optional[Dict[str, Any]] = None
        self.manifest: Optional[Dict[str, Any]] = None

        # --- seeding: everything downstream of here derives from `seed` ----
        set_global_seed(seed)
        hyperparams = dict(hyperparams or {})
        if seed is not None:
            hyperparams.setdefault("seed", int(seed))
            env.reset(seed=int(seed))
            env.action_space.seed(int(seed))

        # --- vectorised envs (DummyVecEnv; env stepping is ~1% of wall time)
        if self.n_envs > 1:
            from stable_baselines3.common.env_util import make_vec_env
            from stable_baselines3.common.vec_env import DummyVecEnv
            desc = self.env_description

            def _factory():
                return ArmTaskEnv.from_description(desc)

            training_env = make_vec_env(_factory, n_envs=self.n_envs, seed=seed, vec_env_cls=DummyVecEnv)
            if self.algorithm == "SAC":
                # Keep the update-to-data ratio at 1 gradient step per env step.
                hyperparams.setdefault("train_freq", 1)
                hyperparams.setdefault("gradient_steps", self.n_envs)
        else:
            training_env = env
        self.training_env = training_env

        self.trainer = RLTrainer(env=training_env, algorithm=self.algorithm, hyperparams=hyperparams)

        # --- continue an earlier package: model + replay buffer + curriculum
        if self.continue_from is not None:
            self._load_continuation(self.continue_from)

        # Metrics tracking
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.episode_goals_reached = deque(maxlen=1000)
        self.episode_goal_distances = deque(maxlen=1000)
        self.policy_losses = deque(maxlen=10000)
        self.value_losses = deque(maxlen=10000)
        self.entropies = deque(maxlen=10000)

        # Pose snapshots for arm panel
        self.latest_joint_angles = np.array(self.env.config.initial_angles, dtype=float)
        self.latest_end_effector_position = np.zeros(2, dtype=float)
        self.latest_shoulder_position = np.array(self.env.shoulder_base_position, dtype=float)
        self.goal_height = float(self.env.goal_height)
        self.goal_position = np.asarray(self.env.goal_position, dtype=float)
        self.goal_direction = str(self.env.goal_direction)
        self.latest_height_error = 0.0
        self.latest_orientation_error = 0.0
        self.latest_hold_counter = 0
        self.latest_hold_steps_required = int(self.env.hold_steps_required)
        self.latest_hold_progress = 0.0
        self.latest_in_goal_region = False
        self.latest_gradient_norm = 0.0
        self.latest_waypoint_index = 0
        self.latest_waypoints = self.env_description.get("waypoints")

        self.best_reward = -float("inf")
        self.best_distance = float("inf")
        self.timesteps_trained = 0
        self.episodes_completed = 0
        self.start_time: Optional[datetime] = None
        self.training_active = False
        self.stop_requested = False

        if use_curriculum is None:
            self.use_curriculum = self.algorithm == "SAC"
        else:
            self.use_curriculum = bool(use_curriculum)
        self.curriculum_kwargs = dict(curriculum_kwargs) if curriculum_kwargs else {}
        self.tolerance_ladder = list(tolerance_ladder) if tolerance_ladder else list(
            self.curriculum_kwargs.get("tolerance_ladder", DEFAULT_TOLERANCE_LADDER))
        self.hold_ladder = list(hold_ladder) if hold_ladder else list(
            self.curriculum_kwargs.get("hold_ladder", DEFAULT_HOLD_LADDER))
        self.eval_env_fn = eval_env_fn
        self.eval_checkpoint_callback: Optional[EvalCheckpointCurriculumCallback] = None
        self.curriculum_callback = None  # legacy attribute read by older GUI code

        self.metrics_lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Continuation
    # ------------------------------------------------------------------ #
    def _load_continuation(self, package: Path) -> None:
        manifest = read_manifest(package)
        env_desc = manifest.get("env", {})
        if int(env_desc.get("obs_version", -1)) != int(self.env_description.get("obs_version", -2)):
            raise ValueError("cannot continue: observation version differs from the current env")
        if tuple(env_desc.get("config", {}).get("link_lengths", ())) != tuple(self.env_description["config"]["link_lengths"]):
            raise ValueError("cannot continue: arm preset differs from the package")
        model_path = package / PACKAGE_FILES["last"]
        if not model_path.exists():
            model_path = package / PACKAGE_FILES["final"]
        if not model_path.exists():
            raise FileNotFoundError(f"no {PACKAGE_FILES['last']} or {PACKAGE_FILES['final']} in {package}")
        AlgorithmClass = self.trainer.ALGORITHM_MAP[self.algorithm]
        device = self.trainer.hyperparams.get("device", "cpu")
        requested_starts = self.trainer.hyperparams.get("learning_starts")
        self.trainer.model = AlgorithmClass.load(str(model_path), env=self.training_env, device=device)
        buf = package / PACKAGE_FILES["replay_buffer"]
        same_task = (env_desc.get("waypoints") == self.env_description.get("waypoints")
                     and env_desc.get("random_waypoints") == self.env_description.get("random_waypoints"))
        if self.continue_drop_buffer or not same_task:
            # Warm start on a different task (e.g. 2 -> 3 waypoints): keep the
            # weights, refill the buffer with a short on-policy warm-up, and
            # restart the curriculum for the new task.
            buf = None
            self.resume_state = None
            if hasattr(self.trainer.model, "learning_starts"):
                self.trainer.model.learning_starts = int(requested_starts or 2_000)
        else:
            # Same task: no second random warm-up, the buffer already holds experience.
            if hasattr(self.trainer.model, "learning_starts"):
                self.trainer.model.learning_starts = 0
            cur = package / PACKAGE_FILES["curriculum"]
            if cur.exists():
                with open(cur, "r", encoding="utf-8") as f:
                    self.resume_state = json.load(f)
        if buf is not None and buf.exists() and hasattr(self.trainer.model, "load_replay_buffer"):
            self.trainer.model.load_replay_buffer(str(buf))
        self.lineage = list(manifest.get("lineage", [])) + [{
            "parent_run_dir": str(package),
            "parent_model": model_path.name,
            "parent_timesteps": int(self.trainer.model.num_timesteps),
            "parent_seed": manifest.get("seed"),
        }]
        print(f"[Trainer] continuing from {package} at {self.trainer.model.num_timesteps:,} steps "
              f"(buffer {'loaded' if (buf is not None and buf.exists()) else 'dropped'}, curriculum "
              f"{'restored' if self.resume_state else 'fresh'}, learning_starts {self.trainer.model.learning_starts})")

    # ------------------------------------------------------------------ #
    def request_stop(self) -> None:
        self.stop_requested = True

    def should_stop(self) -> bool:
        if self.stop_requested:
            return True
        if self._external_should_stop is None:
            return False
        try:
            return bool(self._external_should_stop())
        except Exception:
            return False

    def _default_eval_env_fn(self) -> Callable[[], ArmTaskEnv]:
        desc = self.env_description

        def _fn():
            return ArmTaskEnv.from_description(desc, terminate_on_success=True)

        return _fn

    def write_manifest(self, extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if self.run_dir is None:
            return None
        payload = dict(self.manifest_extra)
        if extra:
            payload.update(extra)
        self.manifest = write_run_manifest(
            self.run_dir / PACKAGE_FILES["manifest"],
            seed=self.seed,
            algorithm=self.algorithm,
            env_description=self.env_description,
            resolved_hyperparams=self.trainer.resolved_hyperparams,
            cli_args=self.cli_args,
            eval_standard=self.standard.to_dict(),
            curriculum={
                "enabled": self.use_curriculum and self.algorithm == "SAC",
                "tolerance_ladder": self.tolerance_ladder,
                "hold_ladder": self.hold_ladder,
                "eval_freq": self.eval_freq,
                "checkpoint_every": self.checkpoint_every,
                "advance_consecutive": self.advance_consecutive,
                "cooldown_rounds": self.cooldown_rounds,
            },
            lineage=self.lineage,
            extra={"total_timesteps": int(self.total_timesteps), "n_envs": self.n_envs, **payload},
            project_root=Path(__file__).resolve().parents[4],
        )
        return self.manifest

    def train(self) -> Dict[str, Any]:
        self.start_time = datetime.now()
        self.training_active = True
        self.stop_requested = False

        if self.preload_model_path and self.continue_from is None:
            self.trainer.load(self.preload_model_path)

        gui_callback = TrainingGUICallback(trainer=self, check_freq=self.check_freq)
        callbacks: List[Any] = [gui_callback]
        if self.use_curriculum and self.algorithm == "SAC" and self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.eval_checkpoint_callback = EvalCheckpointCurriculumCallback(
                run_dir=str(self.run_dir),
                eval_env_fn=self.eval_env_fn or self._default_eval_env_fn(),
                standard=self.standard,
                eval_freq=self.eval_freq,
                tolerance_ladder=self.tolerance_ladder,
                hold_ladder=self.hold_ladder,
                advance_consecutive=self.advance_consecutive,
                cooldown_rounds=self.cooldown_rounds,
                checkpoint_every=self.checkpoint_every,
                resume_state=self.resume_state,
                model_stem=f"{self.algorithm.lower()}_model",
                verbose=1,
            )
            callbacks.append(self.eval_checkpoint_callback)

        if self.run_dir is not None:
            self.write_manifest({"status": "training", "started": self.start_time.isoformat(timespec="seconds")})

        try:
            self.trainer.model.learn(
                total_timesteps=self.total_timesteps,
                callback=callbacks,
                progress_bar=False,
                reset_num_timesteps=self.continue_from is None,
            )
            result = {
                "history": self.trainer.training_history,
                "best_reward": float(self.trainer.best_reward),
                "total_timesteps": self.trainer.model.num_timesteps,
                "training_duration": (datetime.now() - self.start_time).total_seconds(),
            }
            final_metrics = self._get_training_summary()
            return {**result, **final_metrics, "stopped_early": bool(self.stop_requested)}
        except Exception as exc:
            self._safe_queue_put({"error": str(exc), "type": "training_error"})
            raise
        finally:
            self.training_active = False

    def get_current_metrics(self) -> Dict[str, Any]:
        with self.metrics_lock:
            episode_rewards_list = list(self.episode_rewards)
            episode_goals = list(self.episode_goals_reached)
            episode_distances = list(self.episode_goal_distances)

            mean_reward = float(np.mean(episode_rewards_list[-100:])) if episode_rewards_list else 0.0
            mean_distance = float(np.mean(episode_distances[-100:])) if episode_distances else 0.0
            success_rate = float(np.mean(episode_goals[-100:]) * 100.0) if episode_goals else 0.0

            policy_loss = float(self.policy_losses[-1]) if self.policy_losses else 0.0
            value_loss = float(self.value_losses[-1]) if self.value_losses else 0.0
            entropy = float(self.entropies[-1]) if self.entropies else 0.0

            elapsed_time = 0.0
            if self.start_time is not None:
                elapsed_time = (datetime.now() - self.start_time).total_seconds()

            if self.eval_checkpoint_callback is not None:
                curriculum_info = self.eval_checkpoint_callback.get_progress()
            else:
                curriculum_info = {"enabled": False}

            return {
                "algorithm": self.algorithm,
                "timesteps": self.timesteps_trained,
                "episodes": self.episodes_completed,
                "episode_rewards": episode_rewards_list,
                "episode_lengths": list(self.episode_lengths),
                "mean_reward": mean_reward,
                "best_reward": float(self.best_reward),
                "success_rate": success_rate,
                "avg_goal_distance": mean_distance,
                "best_distance": float(self.best_distance),
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "policy_losses": list(self.policy_losses),
                "value_losses": list(self.value_losses),
                "entropies": list(self.entropies),
                "training_time_sec": elapsed_time,
                "joint_angles": self.latest_joint_angles.tolist(),
                "end_effector_position": self.latest_end_effector_position.tolist(),
                "shoulder_position": self.latest_shoulder_position.tolist(),
                "goal_height": self.goal_height,
                "goal_position": self.goal_position.tolist(),
                "goal_direction": self.goal_direction,
                "height_error": float(self.latest_height_error),
                "orientation_error": float(self.latest_orientation_error),
                "hold_counter": int(self.latest_hold_counter),
                "hold_steps_required": int(self.latest_hold_steps_required),
                "hold_progress": float(self.latest_hold_progress),
                "in_goal_region": bool(self.latest_in_goal_region),
                "gradient_norm": float(self.latest_gradient_norm),
                "current_waypoint_index": int(self.latest_waypoint_index),
                "waypoints": self.latest_waypoints,
                "curriculum": curriculum_info,
            }

    def _push_periodic_metrics(self) -> None:
        metrics = self.get_current_metrics()
        metrics["type"] = "metrics_update"
        self._emit_metrics(metrics)

    def _add_episode_result(self, reward: float, length: int, goal_reached: bool, goal_distance: float) -> None:
        with self.metrics_lock:
            self.episode_rewards.append(float(reward))
            self.episode_lengths.append(int(length))
            self.episode_goals_reached.append(float(goal_reached))
            self.episode_goal_distances.append(float(goal_distance))
            self.episodes_completed += 1
            if reward > self.best_reward:
                self.best_reward = float(reward)
            if goal_distance < self.best_distance:
                self.best_distance = float(goal_distance)
        metrics = self.get_current_metrics()
        metrics["type"] = "episode_completed"
        self._emit_metrics(metrics)

    def _add_algorithm_metrics(self, policy_loss: float, value_loss: float, entropy: float) -> None:
        with self.metrics_lock:
            self.policy_losses.append(float(policy_loss))
            self.value_losses.append(float(value_loss))
            self.entropies.append(float(entropy))

    def _set_timesteps(self, timesteps: int) -> None:
        with self.metrics_lock:
            self.timesteps_trained = int(timesteps)

    def _set_latest_pose(self, info: Dict[str, Any]) -> None:
        with self.metrics_lock:
            ja = info.get("joint_angles")
            if ja is not None:
                self.latest_joint_angles = np.asarray(ja, dtype=float)
            ee = info.get("end_effector_position")
            if ee is not None:
                self.latest_end_effector_position = np.asarray(ee, dtype=float)
            sp = info.get("shoulder_position")
            if sp is not None:
                self.latest_shoulder_position = np.asarray(sp, dtype=float)
            if info.get("goal_height") is not None:
                self.goal_height = float(info["goal_height"])
            if info.get("goal_position") is not None:
                self.goal_position = np.asarray(info["goal_position"], dtype=float)
            if info.get("goal_direction") is not None:
                self.goal_direction = str(info["goal_direction"])
            if info.get("height_error") is not None:
                self.latest_height_error = float(info["height_error"])
            if info.get("orientation_error") is not None:
                self.latest_orientation_error = float(info["orientation_error"])
            if info.get("hold_counter") is not None:
                self.latest_hold_counter = int(info["hold_counter"])
            if info.get("hold_steps_required") is not None:
                self.latest_hold_steps_required = int(info["hold_steps_required"])
            if info.get("hold_progress") is not None:
                self.latest_hold_progress = float(info["hold_progress"])
            if info.get("in_goal_region") is not None:
                self.latest_in_goal_region = bool(info["in_goal_region"])
            if info.get("current_waypoint_index") is not None:
                self.latest_waypoint_index = int(info["current_waypoint_index"])
            if info.get("waypoints") is not None:
                self.latest_waypoints = info["waypoints"]

    def _get_training_summary(self) -> Dict[str, Any]:
        return {
            "final_metrics": self.get_current_metrics(),
            "total_episodes": self.episodes_completed,
            "algorithm": self.algorithm,
        }

    def evaluate_policy(self, num_episodes: int = 5, deterministic: bool = True) -> Dict[str, Any]:
        return self.trainer.evaluate(env=self.env, num_episodes=num_episodes, deterministic=deterministic)

    def save_model_and_results(self, save_dir: str) -> Dict[str, str]:
        """Write the final model, history, stats and (if not already) the manifest."""
        import csv

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        if self.run_dir is None:
            self.run_dir = save_path.resolve()

        model_path = os.path.join(save_dir, f"{self.algorithm.lower()}_model")
        self.trainer.save(model_path)
        if hasattr(self.trainer.model, "save_replay_buffer"):
            try:
                self.trainer.model.save_replay_buffer(str(save_path / PACKAGE_FILES["replay_buffer"]))
            except Exception:
                pass
        if self.eval_checkpoint_callback is not None:
            self.eval_checkpoint_callback._write_state()

        history_path = os.path.join(save_dir, "training_history.csv")
        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "length", "goal_reached", "goal_distance"])
            for i, row in enumerate(zip(self.episode_rewards, self.episode_lengths,
                                        self.episode_goals_reached, self.episode_goal_distances)):
                writer.writerow([i, *row])

        stats_path = os.path.join(save_dir, "training_stats.json")
        final_metrics = self.get_current_metrics()
        stats = {
            "algorithm": self.algorithm,
            "total_timesteps": final_metrics["timesteps"],
            "total_episodes": final_metrics["episodes"],
            "mean_reward": final_metrics["mean_reward"],
            "best_reward": final_metrics["best_reward"],
            "success_rate": final_metrics["success_rate"],
            "hold_progress": final_metrics.get("hold_progress", 0.0),
            "training_time_sec": final_metrics["training_time_sec"],
            "stopped_early": bool(self.stop_requested),
            "curriculum": final_metrics.get("curriculum", {}),
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)

        cp = self.eval_checkpoint_callback.get_progress() if self.eval_checkpoint_callback else {}
        self.write_manifest({
            "status": "finished" if not self.stop_requested else "stopped",
            "finished": datetime.now().isoformat(timespec="seconds"),
            "final_timesteps": int(self.trainer.model.num_timesteps),
            "best_score": cp.get("best_score"),
            "best_saved_at": cp.get("best_saved_at"),
        })
        return {"model": model_path, "history": history_path, "stats": stats_path,
                "manifest": str(self.run_dir / PACKAGE_FILES["manifest"])}

    def _safe_queue_put(self, message: Dict[str, Any]) -> None:
        if self.metrics_queue is None:
            return
        try:
            self.metrics_queue.put_nowait(message)
        except queue.Full:
            pass

    def _emit_metrics(self, metrics: Dict[str, Any]) -> None:
        self._safe_queue_put(metrics)
        if self.metrics_callback is not None:
            try:
                self.metrics_callback(metrics)
            except Exception:
                pass


class TrainingGUICallback(GUICallback):
    """Callback that extracts episode and optimizer metrics for GUI updates."""

    def __init__(self, trainer: RLTrainerWithMetrics, check_freq: int = 100, verbose: int = 0):
        super().__init__(check_freq=check_freq, verbose=verbose)
        self.trainer = trainer
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []

    @staticmethod
    def _ensure_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return [value]

    def _on_step(self) -> bool:
        self.trainer._set_timesteps(self.num_timesteps)
        if self.trainer.should_stop():
            return False

        infos = self._ensure_list(self.locals.get("infos"))
        rewards = self._ensure_list(self.locals.get("rewards"))
        dones = [bool(v) for v in self._ensure_list(self.locals.get("dones"))]

        env_count = max(len(infos), len(rewards), len(dones), 1)
        if len(self._episode_rewards) != env_count:
            self._episode_rewards = [0.0] * env_count
            self._episode_lengths = [0] * env_count

        for idx in range(env_count):
            reward = float(rewards[idx]) if idx < len(rewards) else 0.0
            self._episode_rewards[idx] += reward
            self._episode_lengths[idx] += 1

            info = infos[idx] if idx < len(infos) and isinstance(infos[idx], dict) else {}
            if info and idx == 0:
                self.trainer._set_latest_pose(info)

            done = dones[idx] if idx < len(dones) else False
            if not done:
                continue

            episode_reward = self._episode_rewards[idx]
            episode_length = self._episode_lengths[idx]
            episode_info = info.get("episode")
            if isinstance(episode_info, dict):
                episode_reward = float(episode_info.get("r", episode_reward))
                episode_length = int(episode_info.get("l", episode_length))

            self.trainer._add_episode_result(
                reward=episode_reward,
                length=episode_length,
                goal_reached=bool(info.get("success", info.get("goal_reached", False))),
                goal_distance=float(info.get("goal_distance", 0.0)),
            )
            self._episode_rewards[idx] = 0.0
            self._episode_lengths[idx] = 0

        if self.num_timesteps % self.check_freq == 0:
            self._extract_algorithm_metrics()
            self.trainer._push_periodic_metrics()
        return True

    def _extract_algorithm_metrics(self) -> None:
        model = self.model
        logger = getattr(model, "logger", None)
        if logger is None:
            return
        logger_data = getattr(logger, "name_to_value", {})
        if not isinstance(logger_data, dict):
            return
        policy_loss = self._get_first_present(logger_data, ["train/policy_loss", "train/actor_loss", "train/loss"])
        value_loss = self._get_first_present(logger_data, ["train/value_loss", "train/critic_loss", "train/entropy_loss"])
        entropy = self._get_first_present(logger_data, ["train/entropy_loss", "train/ent_coef_loss"])
        if entropy == 0.0 and hasattr(model, "policy"):
            try:
                if hasattr(model.policy, "log_std"):
                    entropy = float(model.policy.log_std.mean().item())
            except Exception:
                entropy = 0.0
        self.trainer._add_algorithm_metrics(policy_loss=policy_loss, value_loss=value_loss, entropy=entropy)

    @staticmethod
    def _get_first_present(data: Dict[str, Any], keys: List[str]) -> float:
        for key in keys:
            if key in data:
                try:
                    return float(data[key])
                except Exception:
                    continue
        return 0.0


class PPOTrainerWithMetrics(RLTrainerWithMetrics):
    """Backward-compatible PPO-only wrapper."""

    def __init__(self, env=None, total_timesteps=100000, metrics_queue=None, metrics_callback=None,
                 should_stop=None, hyperparams=None, check_freq=100):
        super().__init__(env=env, total_timesteps=total_timesteps, algorithm="PPO", hyperparams=hyperparams,
                         metrics_queue=metrics_queue, metrics_callback=metrics_callback,
                         should_stop=should_stop, check_freq=check_freq)


class PPOGUICallback(TrainingGUICallback):
    """Backward-compatible callback alias for PPO code paths."""


__all__ = ["RLTrainerWithMetrics", "TrainingGUICallback", "PPOTrainerWithMetrics", "PPOGUICallback"]
