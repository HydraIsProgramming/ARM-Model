"""Adaptive goal-tolerance curriculum following Fischer et al. (2021).

This module implements the adaptive curriculum scheme described in Fischer,
Hoinville, Eickhoff, and Lilienthal (2021), "Reinforcement learning control
of a biomechanical model of the upper extremity", Scientific Reports 11:14445,
https://doi.org/10.1038/s41598-021-93760-1.

Fischer et al. trained their seven-degree-of-freedom MuJoCo arm with SAC and
found that learning a precise goal-reaching policy from the outset was
intractable: the reward landscape is too sparse when the success region is
small. Their solution was an adaptive curriculum on the position tolerance.
The agent begins training with a wide goal radius (~60 cm), so success is
common and the policy receives a strong learning signal. Once the rolling
success rate over the recent training window exceeds 80 %, the tolerance is
shrunk multiplicatively. This continues until a precision target (~2 cm) is
reached. The curriculum is "adaptive" because it advances on the agent's
performance, not on a fixed timestep schedule.

This callback implements that scheme as a Stable-Baselines3 BaseCallback. It
queries the underlying environment via the `set_goal_tolerance` hook added to
ArmTaskEnv, and detects episode terminations via `dones` and `infos` produced
by the SB3 rollout loop. It supports both a bare environment and a vectorised
environment with multiple parallel workers.
"""

from collections import deque
from typing import Any, Callable, List, Optional

from stable_baselines3.common.callbacks import BaseCallback


__all__ = ["AdaptiveCurriculumCallback", "EvalBasedCurriculumCallback", "HoldCurriculumCallback"]


class AdaptiveCurriculumCallback(BaseCallback):
    """Shrink the environment's goal tolerance as the success rate improves.

    Parameters
    ----------
    initial_tolerance : float, default 0.60
        Starting position tolerance in metres. The Fischer 2021 paper used
        approximately 60 cm for the initial radius on their 7-DOF arm; the
        value is appropriate here too because the 2-DOF arm in this project
        has a comparable maximum reach.
    min_tolerance : float, default 0.02
        Minimum (final) tolerance in metres below which no further decay is
        applied. Fischer used 2 cm as the precision target.
    success_rate_threshold : float, default 0.80
        Recent-window success rate (in [0, 1]) above which the tolerance is
        shrunk. Fischer used 80 %.
    decay_factor : float, default 0.80
        Multiplicative shrink factor applied each time the threshold is
        exceeded. Must lie in the open interval (0, 1). A value of 0.80
        corresponds to a 20 % reduction per curriculum stage.
    window_size : int, default 50
        Number of most-recent episodes counted toward the rolling success
        rate. Smaller windows react faster but are noisier.
    min_episodes_before_decay : int, default 20
        Minimum number of episodes that must elapse since the previous decay
        before another decay is permitted. Prevents the curriculum from
        oscillating during the first few episodes after a stage change.
    success_key : str, default "goal_reached"
        Key in the per-step ``info`` dict that ArmTaskEnv uses to flag a
        terminated-by-goal episode.
    verbose : int, default 0
        Stable-Baselines3 verbosity. ``1`` prints a one-line announcement
        each time a curriculum stage advances.

    Attributes
    ----------
    current_tolerance : float
        The position tolerance currently applied to the environment.
    curriculum_stage : int
        Number of times the tolerance has been shrunk since training started.
        Stage 0 is the initial wide tolerance; each successful decay
        increments the counter.
    """

    def __init__(
        self,
        initial_tolerance: float = 0.60,
        min_tolerance: float = 0.02,
        success_rate_threshold: float = 0.80,
        decay_factor: float = 0.80,
        window_size: int = 50,
        min_episodes_before_decay: int = 20,
        success_key: str = "goal_reached",
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)

        if not 0.0 < success_rate_threshold <= 1.0:
            raise ValueError(
                f"success_rate_threshold must lie in (0, 1], got {success_rate_threshold}"
            )
        if not 0.0 < decay_factor < 1.0:
            raise ValueError(
                f"decay_factor must lie in (0, 1), got {decay_factor}"
            )
        if min_tolerance <= 0.0:
            raise ValueError(f"min_tolerance must be positive, got {min_tolerance}")
        if initial_tolerance < min_tolerance:
            raise ValueError(
                f"initial_tolerance ({initial_tolerance}) must not be smaller "
                f"than min_tolerance ({min_tolerance})"
            )
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if min_episodes_before_decay < 0:
            raise ValueError(
                f"min_episodes_before_decay must be non-negative, got {min_episodes_before_decay}"
            )

        self.initial_tolerance = float(initial_tolerance)
        self.min_tolerance = float(min_tolerance)
        self.success_rate_threshold = float(success_rate_threshold)
        self.decay_factor = float(decay_factor)
        self.window_size = int(window_size)
        self.min_episodes_before_decay = int(min_episodes_before_decay)
        self.success_key = str(success_key)

        self.current_tolerance: float = float(initial_tolerance)
        self.curriculum_stage: int = 0
        self._episode_results: deque = deque(maxlen=self.window_size)
        self._episodes_since_last_decay: int = 0

    # ------------------------------------------------------------------ #
    # SB3 hooks
    # ------------------------------------------------------------------ #
    def _on_training_start(self) -> None:
        """Apply the initial wide tolerance to the training environment."""
        self._apply_tolerance(self.current_tolerance)
        if self.verbose:
            print(
                f"[Curriculum] Initial goal tolerance set to "
                f"{self.current_tolerance:.3f} m (Fischer 2021 protocol; "
                f"target {self.min_tolerance:.3f} m, decay x{self.decay_factor}, "
                f"threshold {self.success_rate_threshold:.0%}, window "
                f"{self.window_size})"
            )

    def _on_step(self) -> bool:
        """Update the rolling success rate and shrink the tolerance if appropriate.

        SB3 places the per-environment ``infos`` and ``dones`` arrays into
        ``self.locals`` on each step. We only act on completed episodes;
        successful episodes are those for which ``info[success_key]`` is True.
        """
        infos: List[Any] = list(self.locals.get("infos", []))
        dones: List[Any] = list(self.locals.get("dones", []))

        # Account for either a single-env or VecEnv rollout
        for info, done in zip(infos, dones):
            if not bool(done):
                continue
            success = bool(info.get(self.success_key, False)) if isinstance(info, dict) else False
            self._episode_results.append(1 if success else 0)
            self._episodes_since_last_decay += 1

        self._maybe_advance_curriculum()
        return True

    # ------------------------------------------------------------------ #
    # Internal logic
    # ------------------------------------------------------------------ #
    def _maybe_advance_curriculum(self) -> None:
        """Shrink tolerance if the recent success rate exceeds the threshold.

        This method implements the central decision rule of the Fischer 2021
        adaptive curriculum. It is called once per environment step and
        decides whether to leave the current goal tolerance in place or
        decay it. Four preconditions must all hold before a decay fires:

          1. The rolling window has filled up (we need ``window_size``
             completed episodes before the success-rate estimate is
             statistically meaningful).
          2. The cooldown since the last decay has elapsed (prevents two
             back-to-back decays on the same successful window — would let
             the curriculum oscillate).
          3. The current tolerance is still above the precision floor (no
             point decaying once we've reached the target).
          4. The rolling success rate equals or exceeds the threshold
             (Fischer's 80% — the empirical signal that the agent has
             mastered the current stage).

        When all four hold, the tolerance is multiplied by the decay factor
        (clamped to the minimum), the env's set_goal_tolerance is called so
        the next episode's goal-region check uses the new value, the stage
        counter increments, and the cooldown resets.
        """
        # Precondition 1: enough data to estimate the success rate.
        if len(self._episode_results) < self.window_size:
            return
        # Precondition 2: cooldown between consecutive decays.
        if self._episodes_since_last_decay < self.min_episodes_before_decay:
            return
        # Precondition 3: already at the precision floor.
        if self.current_tolerance <= self.min_tolerance:
            return

        # Precondition 4: the agent is succeeding often enough.
        success_rate = sum(self._episode_results) / float(len(self._episode_results))
        if success_rate < self.success_rate_threshold:
            return

        # All four preconditions hold — apply one decay step.
        new_tolerance = max(
            self.current_tolerance * self.decay_factor,
            self.min_tolerance,
        )
        if new_tolerance >= self.current_tolerance:
            # Guard against zero-effect decays (e.g., if decay_factor == 1.0).
            return

        self.current_tolerance = float(new_tolerance)
        self.curriculum_stage += 1
        self._episodes_since_last_decay = 0
        self._apply_tolerance(self.current_tolerance)

        if self.verbose:
            print(
                f"[Curriculum] Stage {self.curriculum_stage}: "
                f"recent success rate {success_rate:.1%} >= "
                f"{self.success_rate_threshold:.0%}; tolerance shrunk to "
                f"{self.current_tolerance:.3f} m"
            )

    def _apply_tolerance(self, tolerance: float) -> None:
        """Push the new tolerance to every underlying ArmTaskEnv instance.

        Handles three rollout topologies:
          1. A bare ArmTaskEnv exposing set_goal_tolerance directly.
          2. A VecEnv (DummyVecEnv / SubprocVecEnv) supporting env_method.
          3. A VecEnv wrapper exposing an .envs attribute (DummyVecEnv).
        """
        env = self.training_env

        if hasattr(env, "env_method"):
            try:
                env.env_method("set_goal_tolerance", tolerance)
                return
            except Exception:
                # fall through to direct application
                pass

        if hasattr(env, "set_goal_tolerance"):
            env.set_goal_tolerance(tolerance)
            return

        if hasattr(env, "envs"):
            for sub in env.envs:
                # Walk through TimeLimit / Monitor wrappers to reach the base env
                target = sub
                for _ in range(8):
                    if hasattr(target, "set_goal_tolerance"):
                        target.set_goal_tolerance(tolerance)
                        break
                    if hasattr(target, "env"):
                        target = target.env
                    else:
                        break

    # ------------------------------------------------------------------ #
    # Public reporting helpers (for the GUI / metrics pipeline)
    # ------------------------------------------------------------------ #
    def get_progress(self) -> dict:
        """Return a snapshot of curriculum state suitable for GUI display."""
        sr: Optional[float]
        if self._episode_results:
            sr = sum(self._episode_results) / float(len(self._episode_results))
        else:
            sr = None
        return {
            "current_tolerance": float(self.current_tolerance),
            "min_tolerance": float(self.min_tolerance),
            "initial_tolerance": float(self.initial_tolerance),
            "curriculum_stage": int(self.curriculum_stage),
            "recent_success_rate": sr,
            "window_filled": len(self._episode_results),
            "window_size": int(self.window_size),
            "episodes_since_last_decay": int(self._episodes_since_last_decay),
        }


class EvalBasedCurriculumCallback(BaseCallback):
    """Curriculum that advances based on periodic deterministic evaluation.

    Unlike AdaptiveCurriculumCallback which measures success rate from the
    stochastic training policy (systematically under-counting because SAC's
    exploration noise disrupts holds), this callback periodically runs a
    batch of deterministic rollouts on a separate eval environment and uses
    *that* success rate for curriculum decisions.

    Parameters
    ----------
    eval_env_fn : Callable
        Zero-argument factory that returns a fresh ArmTaskEnv instance for
        evaluation. A factory is used (instead of a pre-built env) so the
        callback can set the current tolerance on each eval round.
    initial_tolerance : float
        Starting position tolerance in metres.
    min_tolerance : float
        Floor tolerance below which no further decay occurs.
    success_rate_threshold : float
        Eval success rate (in [0, 1]) above which tolerance is shrunk.
    decay_factor : float
        Multiplicative shrink factor in (0, 1).
    eval_freq : int
        Run an evaluation round every this many training timesteps.
    n_eval_episodes : int
        Number of deterministic episodes per evaluation round.
    success_key : str
        Key in the info dict that flags a successful termination.
    verbose : int
        SB3 verbosity level.
    """

    def __init__(
        self,
        eval_env_fn: Callable,
        initial_tolerance: float = 0.60,
        min_tolerance: float = 0.02,
        success_rate_threshold: float = 0.80,
        decay_factor: float = 0.80,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 20,
        success_key: str = "goal_reached",
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.eval_env_fn = eval_env_fn
        self.initial_tolerance = float(initial_tolerance)
        self.min_tolerance = float(min_tolerance)
        self.success_rate_threshold = float(success_rate_threshold)
        self.decay_factor = float(decay_factor)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.success_key = str(success_key)

        self.current_tolerance: float = float(initial_tolerance)
        self.curriculum_stage: int = 0
        self._last_eval_step: int = 0
        self._recent_success_rate: Optional[float] = None

    def _on_training_start(self) -> None:
        self._apply_tolerance(self.current_tolerance)
        if self.verbose:
            print(
                f"[EvalCurriculum] Initial tolerance {self.current_tolerance:.3f} m  "
                f"(eval every {self.eval_freq} steps, {self.n_eval_episodes} episodes, "
                f"threshold {self.success_rate_threshold:.0%}, decay x{self.decay_factor})"
            )

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self._run_eval_and_maybe_advance()
            self._last_eval_step = self.num_timesteps
        return True

    def _run_eval_and_maybe_advance(self) -> None:
        eval_env = self.eval_env_fn()
        if hasattr(eval_env, "set_goal_tolerance"):
            eval_env.set_goal_tolerance(self.current_tolerance)

        successes = 0
        for _ in range(self.n_eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
            if info.get(self.success_key, False):
                successes += 1

        eval_env.close()
        self._recent_success_rate = successes / self.n_eval_episodes

        if self.verbose:
            print(
                f"[EvalCurriculum] step {self.num_timesteps}: "
                f"eval success {self._recent_success_rate:.0%} "
                f"({successes}/{self.n_eval_episodes}) at tol "
                f"{self.current_tolerance:.3f} m"
            )

        if self.current_tolerance <= self.min_tolerance:
            return
        if self._recent_success_rate < self.success_rate_threshold:
            return

        new_tol = max(
            self.current_tolerance * self.decay_factor,
            self.min_tolerance,
        )
        if new_tol >= self.current_tolerance:
            return

        self.current_tolerance = float(new_tol)
        self.curriculum_stage += 1
        self._apply_tolerance(self.current_tolerance)

        if self.verbose:
            print(
                f"[EvalCurriculum] Stage {self.curriculum_stage}: "
                f"tolerance shrunk to {self.current_tolerance:.3f} m"
            )

    def _apply_tolerance(self, tolerance: float) -> None:
        env = self.training_env
        if hasattr(env, "env_method"):
            try:
                env.env_method("set_goal_tolerance", tolerance)
                return
            except Exception:
                pass
        if hasattr(env, "set_goal_tolerance"):
            env.set_goal_tolerance(tolerance)
            return
        if hasattr(env, "envs"):
            for sub in env.envs:
                target = sub
                for _ in range(8):
                    if hasattr(target, "set_goal_tolerance"):
                        target.set_goal_tolerance(tolerance)
                        break
                    if hasattr(target, "env"):
                        target = target.env
                    else:
                        break

    def get_progress(self) -> dict:
        return {
            "current_tolerance": float(self.current_tolerance),
            "min_tolerance": float(self.min_tolerance),
            "initial_tolerance": float(self.initial_tolerance),
            "curriculum_stage": int(self.curriculum_stage),
            "recent_success_rate": self._recent_success_rate,
            "eval_freq": int(self.eval_freq),
            "n_eval_episodes": int(self.n_eval_episodes),
            "last_eval_step": int(self._last_eval_step),
        }


class HoldCurriculumCallback(BaseCallback):
    """Gradually increase the hold-steps requirement as the agent learns to hold.

    Starts at initial_hold_steps (e.g. 10) and advances through a fixed
    ladder of steps (e.g. [10, 15, 20]) each time the rolling success rate
    exceeds success_rate_threshold. This prevents the agent from getting
    stuck trying to hold for the full 20 steps before it has learned basic
    hold behaviour.

    Parameters
    ----------
    initial_hold_steps : int
        Starting hold requirement (default 10 — half the production value).
    hold_ladder : list of int
        Ordered stages to advance through (default [10, 15, 20]).
    success_rate_threshold : float
        Rolling success rate above which the hold requirement advances.
    window_size : int
        Number of recent episodes for the rolling success rate.
    min_episodes_before_advance : int
        Cooldown between advances.
    success_key : str
        Info dict key flagging a successful episode.
    verbose : int
        SB3 verbosity level.
    """

    def __init__(
        self,
        initial_hold_steps: int = 10,
        hold_ladder: Optional[List[int]] = None,
        success_rate_threshold: float = 0.60,
        window_size: int = 50,
        min_episodes_before_advance: int = 20,
        success_key: str = "goal_reached",
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.hold_ladder: List[int] = hold_ladder if hold_ladder is not None else [10, 15, 20]
        if initial_hold_steps not in self.hold_ladder:
            self.hold_ladder = sorted(set([initial_hold_steps] + self.hold_ladder))
        self.success_rate_threshold = float(success_rate_threshold)
        self.window_size = int(window_size)
        self.min_episodes_before_advance = int(min_episodes_before_advance)
        self.success_key = str(success_key)

        self._ladder_idx: int = self.hold_ladder.index(initial_hold_steps)
        self.current_hold_steps: int = initial_hold_steps
        self._episode_results: deque = deque(maxlen=self.window_size)
        self._episodes_since_last_advance: int = 0

    def _on_training_start(self) -> None:
        self._apply_hold_steps(self.current_hold_steps)
        if self.verbose:
            print(
                f"[HoldCurriculum] Initial hold requirement: "
                f"{self.current_hold_steps} steps "
                f"(ladder: {self.hold_ladder}, threshold: {self.success_rate_threshold:.0%})"
            )

    def _on_step(self) -> bool:
        infos: List[Any] = list(self.locals.get("infos", []))
        dones: List[Any] = list(self.locals.get("dones", []))
        for info, done in zip(infos, dones):
            if not bool(done):
                continue
            success = bool(info.get(self.success_key, False)) if isinstance(info, dict) else False
            self._episode_results.append(1 if success else 0)
            self._episodes_since_last_advance += 1
        self._maybe_advance()
        return True

    def _maybe_advance(self) -> None:
        if self._ladder_idx >= len(self.hold_ladder) - 1:
            return
        if len(self._episode_results) < self.window_size:
            return
        if self._episodes_since_last_advance < self.min_episodes_before_advance:
            return
        success_rate = sum(self._episode_results) / float(len(self._episode_results))
        if success_rate < self.success_rate_threshold:
            return

        self._ladder_idx += 1
        self.current_hold_steps = self.hold_ladder[self._ladder_idx]
        self._episodes_since_last_advance = 0
        self._apply_hold_steps(self.current_hold_steps)

        if self.verbose:
            print(
                f"[HoldCurriculum] Advanced to hold={self.current_hold_steps} steps "
                f"(success rate {success_rate:.1%}, stage {self._ladder_idx}/{len(self.hold_ladder)-1})"
            )

    def _apply_hold_steps(self, steps: int) -> None:
        env = self.training_env
        if hasattr(env, "env_method"):
            try:
                env.env_method("set_hold_steps_required", steps)
                return
            except Exception:
                pass
        if hasattr(env, "set_hold_steps_required"):
            env.set_hold_steps_required(steps)
            return
        if hasattr(env, "envs"):
            for sub in env.envs:
                target = sub
                for _ in range(8):
                    if hasattr(target, "set_hold_steps_required"):
                        target.set_hold_steps_required(steps)
                        break
                    if hasattr(target, "env"):
                        target = target.env
                    else:
                        break

    def get_progress(self) -> dict:
        sr = sum(self._episode_results) / float(len(self._episode_results)) if self._episode_results else None
        return {
            "current_hold_steps": int(self.current_hold_steps),
            "ladder_stage": int(self._ladder_idx),
            "hold_ladder": self.hold_ladder,
            "recent_success_rate": sr,
            "window_filled": len(self._episode_results),
        }
