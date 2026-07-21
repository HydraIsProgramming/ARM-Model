"""Core training infrastructure for RL agents

Provides base and concrete trainer classes for managing RL model training,
evaluation, and checkpoint management with support for Stable-Baselines3
algorithms (PPO, SAC, A3C, etc.).
"""

import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
from stable_baselines3 import A2C, DQN, PPO, SAC

try:
    from stable_baselines3 import A3C  # type: ignore[attr-defined]
except Exception:
    A3C = None

from rl_armMotion.two_d.environments.task_env import ArmTaskEnv


@dataclass
class LinearDecaySchedule:
    """Pickle-safe linear learning-rate schedule."""

    initial_value: float
    final_value: float

    def __call__(self, progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 to 0.0 over training.
        return self.final_value + (self.initial_value - self.final_value) * float(progress_remaining)


def linear_decay(initial_value: float, final_value: float):
    """Linear schedule for SB3 learning rate."""
    return LinearDecaySchedule(
        initial_value=float(initial_value),
        final_value=float(final_value),
    )


def _make_pickle_safe(value: Any) -> Any:
    """
    Convert objects into metadata-safe values for pickle.

    This prevents failures when hyperparams include local callables/closures.
    """
    if isinstance(value, dict):
        return {k: _make_pickle_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_pickle_safe(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_make_pickle_safe(v) for v in value)
    if isinstance(value, set):
        return {_make_pickle_safe(v) for v in value}
    if callable(value):
        name = getattr(value, "__qualname__", getattr(value, "__name__", value.__class__.__name__))
        module = getattr(value, "__module__", "unknown")
        return f"<callable {module}.{name}>"

    try:
        pickle.dumps(value)
        return value
    except Exception:
        return repr(value)


class BaseTrainer(ABC):
    """Abstract base trainer class defining the training interface."""

    def __init__(self, env: Any, algorithm: str = "PPO", hyperparams: Optional[Dict] = None):
        """
        Initialize base trainer.

        Args:
            env: Gymnasium environment instance
            algorithm: Algorithm name (PPO, SAC, A3C, etc.)
            hyperparams: Hyperparameter dictionary for algorithm
        """
        self.env = env
        self.algorithm = algorithm
        self.hyperparams = hyperparams or {}
        self.model = None
        self.best_reward = -float("inf")
        self.best_model = None
        self.training_history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "timesteps": [],
            "mean_rewards": [],
        }

    @abstractmethod
    def train(self, total_timesteps: int, callback=None) -> Dict[str, Any]:
        """Train the model."""
        pass

    @abstractmethod
    def evaluate(self, env: Optional[Any] = None, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the trained model."""
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the trained model."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load a trained model."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            "algorithm": self.algorithm,
            "best_reward": float(self.best_reward),
            "episodes": len(self.training_history["episode_rewards"]),
            "total_timesteps": sum(self.training_history["episode_lengths"]),
            "mean_reward": float(np.mean(self.training_history["episode_rewards"])) if self.training_history["episode_rewards"] else 0.0,
        }


class RLTrainer(BaseTrainer):
    """Concrete trainer for Stable-Baselines3 algorithms."""

    ALGORITHM_MAP = {
        "PPO": PPO,
        "SAC": SAC,
        "A2C": A2C,
        "DQN": DQN,
    }
    if A3C is not None:
        ALGORITHM_MAP["A3C"] = A3C

    def __init__(
        self,
        env: Optional[ArmTaskEnv] = None,
        algorithm: str = "PPO",
        hyperparams: Optional[Dict] = None,
    ):
        """
        Initialize RL trainer with Stable-Baselines3.

        Args:
            env: ArmTaskEnv instance (created if None)
            algorithm: Algorithm name from ALGORITHM_MAP
            hyperparams: Algorithm-specific hyperparameters
        """
        if env is None:
            env = ArmTaskEnv()

        if algorithm not in self.ALGORITHM_MAP:
            raise ValueError(f"Algorithm {algorithm} not supported. Choose from {list(self.ALGORITHM_MAP.keys())}")

        # Set defaults for algorithm if not provided
        if hyperparams is None:
            hyperparams = self._get_default_hyperparams(algorithm)

        super().__init__(env, algorithm, hyperparams)

        # Create model
        AlgorithmClass = self.ALGORITHM_MAP[algorithm]
        self.model = AlgorithmClass(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            **hyperparams,
        )

        self.start_time = None
        self.train_start_time = None

    @staticmethod
    def _get_default_hyperparams(algorithm: str) -> Dict[str, Any]:
        """Get default hyperparameters for algorithm."""
        defaults = {
            "PPO": {
                "learning_rate": linear_decay(3e-4, 3e-5),
                "gamma": 0.99,       # was 0.995; better for ~800-step episodes
                "gae_lambda": 0.95,  # was 0.98; reduces variance, faster convergence
                "ent_coef": 0.01,    # was 0.005; more exploration early in training
                "n_steps": 1024,     # was 4096; 4x more frequent gradient updates
                "batch_size": 128,   # was 256; scaled down to match shorter rollouts
                "n_epochs": 8,       # was 10; balanced with smaller batch size
                "clip_range": 0.2,
            },
            "SAC": {
                "learning_rate": 3e-4,
                "buffer_size": 200_000,   # was 1_000_000; sufficient for this task, less memory
                "batch_size": 256,
                "learning_starts": 1000,  # begin learning after warm-up steps
            },
            "A3C": {
                "learning_rate": 7e-4,
                "n_steps": 5,
            },
            "A2C": {
                "learning_rate": 7e-4,
                "n_steps": 5,
            },
            "DQN": {
                "learning_rate": 1e-4,
                "buffer_size": 200_000,  # was 1_000_000; sufficient for this task
                "batch_size": 32,
            },
        }
        return defaults.get(algorithm, {})

    def train(
        self,
        total_timesteps: int,
        callback=None,
        patience: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            total_timesteps: Total timesteps to train
            callback: Training callback
            patience: Early stopping patience (episodes without improvement)

        Returns:
            Training history dictionary
        """
        self.train_start_time = datetime.now()
        self.start_time = self.train_start_time

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=False,
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")

        return {
            "history": self.training_history,
            "best_reward": float(self.best_reward),
            "total_timesteps": self.model.num_timesteps,
            "training_duration": (datetime.now() - self.train_start_time).total_seconds(),
        }

    def evaluate(
        self,
        env: Optional[ArmTaskEnv] = None,
        num_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate trained model on environment.

        Args:
            env: Environment for evaluation (uses self.env if None)
            num_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic policy

        Returns:
            Evaluation metrics dictionary
        """
        eval_env = env or self.env
        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0

            while True:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                episode_length += 1

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }

    def save(self, filepath: str) -> None:
        """
        Save trained model and metadata.

        Args:
            filepath: Path to save model (without extension)
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        # Save Stable-Baselines3 model
        self.model.save(filepath)

        # Save metadata
        metadata = {
            "algorithm": self.algorithm,
            "hyperparams": _make_pickle_safe(self.hyperparams),
            "training_history": self.training_history,
            "best_reward": float(self.best_reward),
            "timestamp": datetime.now().isoformat(),
        }

        with open(f"{filepath}_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def load(self, filepath: str) -> None:
        """
        Load trained model and metadata.

        Args:
            filepath: Path to model file (without .zip extension)
        """
        # Load Stable-Baselines3 model
        AlgorithmClass = self.ALGORITHM_MAP[self.algorithm]
        self.model = AlgorithmClass.load(filepath, env=self.env)

        # Load metadata if available
        try:
            with open(f"{filepath}_metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
                self.training_history = metadata.get("training_history", {})
                self.best_reward = metadata.get("best_reward", -float("inf"))
        except FileNotFoundError:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        stats = super().get_stats()
        stats["model_timesteps"] = self.model.num_timesteps if self.model else 0
        return stats

    def update_best_reward(self, reward: float) -> bool:
        """
        Update best reward and model.

        Args:
            reward: Current reward

        Returns:
            True if new best reward
        """
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_model = self.model
            return True
        return False


__all__ = ["BaseTrainer", "RLTrainer"]
