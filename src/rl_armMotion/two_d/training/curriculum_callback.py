"""Evaluation-driven curriculum and checkpointing (one callback).

Every ``eval_freq`` steps the callback runs two evaluations with
``evaluation.evaluate_policy`` (seeded, randomised resets, so N episodes are N
samples):

1. **Training-difficulty eval** at the CURRENT tolerance / hold requirement.
   When its success rate reaches ``advance_threshold`` one ladder advances
   (tolerance first, then hold), with a one-round cooldown.
2. **Production eval** at ``EVAL_STANDARD`` (terminate_on_success=True). Its
   continuous score (success rate, waypoints + hold progress, closest final
   distance, speed) selects ``sac_model_best.zip``.

The callback also writes ``sac_model_last.zip`` every round,
``curriculum_state.json`` (ladder indices, best score, history), periodic
``checkpoints/step_XXXXXX.zip`` and the replay buffer, which is what makes a
killed or finished run continuable (``--continue-from``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rl_armMotion.two_d.evaluation import EVAL_STANDARD, PACKAGE_FILES, EvalStandard, evaluate_policy

__all__ = ["EvalCheckpointCurriculumCallback", "_apply_env_setting"]

DEFAULT_TOLERANCE_LADDER = (0.15, 0.10, 0.07, 0.05)
DEFAULT_HOLD_LADDER = (10, 15, 25)


def _apply_env_setting(env, method_name: str, value) -> None:
    """Call ``method_name(value)`` on every underlying ArmTaskEnv (VecEnv or bare)."""
    if hasattr(env, "env_method"):
        try:
            env.env_method(method_name, value)
            return
        except Exception:
            pass
    if hasattr(env, method_name):
        getattr(env, method_name)(value)
        return
    if hasattr(env, "envs"):
        for sub in env.envs:
            target = sub
            for _ in range(8):
                if hasattr(target, method_name):
                    getattr(target, method_name)(value)
                    break
                if hasattr(target, "env"):
                    target = target.env
                else:
                    break


class EvalCheckpointCurriculumCallback(BaseCallback):
    def __init__(
        self,
        run_dir: str,
        eval_env_fn: Callable[[], Any],
        standard: EvalStandard = EVAL_STANDARD,
        eval_freq: int = 20_000,
        n_train_eval_episodes: int = 5,
        tolerance_ladder: Sequence[float] = DEFAULT_TOLERANCE_LADDER,
        hold_ladder: Sequence[int] = DEFAULT_HOLD_LADDER,
        advance_threshold: float = 0.8,
        checkpoint_every: int = 100_000,
        save_replay_buffer: bool = True,
        resume_state: Optional[Dict[str, Any]] = None,
        model_stem: str = "sac_model",
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.run_dir = Path(run_dir)
        self.eval_env_fn = eval_env_fn
        self.standard = standard
        self.eval_freq = int(eval_freq)
        self.n_train_eval_episodes = int(n_train_eval_episodes)
        self.tolerance_ladder = [float(t) for t in tolerance_ladder]
        self.hold_ladder = [int(h) for h in hold_ladder]
        if any(b > a for a, b in zip(self.tolerance_ladder, self.tolerance_ladder[1:])):
            raise ValueError("tolerance_ladder must be non-increasing")
        if any(b < a for a, b in zip(self.hold_ladder, self.hold_ladder[1:])):
            raise ValueError("hold_ladder must be non-decreasing")
        self.advance_threshold = float(advance_threshold)
        self.checkpoint_every = int(checkpoint_every)
        self.save_replay_buffer = bool(save_replay_buffer)
        self.model_stem = model_stem

        self.tolerance_idx = 0
        self.hold_idx = 0
        self.best_score: Optional[List[float]] = None
        self.best_saved_at: Optional[int] = None
        self.history: List[Dict[str, Any]] = []
        self.rounds = 0
        self._cooldown = 0
        self._last_eval_step = 0
        self._last_checkpoint_step = 0
        self._eval_env = None
        self._last_train_success: Optional[float] = None
        self._last_prod_success: Optional[float] = None
        if resume_state:
            self._load_state(resume_state)

    # ------------------------------------------------------------------ #
    @property
    def current_tolerance(self) -> float:
        return self.tolerance_ladder[self.tolerance_idx]

    @property
    def current_hold_steps(self) -> int:
        return self.hold_ladder[self.hold_idx]

    def _apply_training_difficulty(self) -> None:
        _apply_env_setting(self.training_env, "set_goal_tolerance", self.current_tolerance)
        _apply_env_setting(self.training_env, "set_hold_steps_required", self.current_hold_steps)

    def _on_training_start(self) -> None:
        self._apply_training_difficulty()
        self._last_eval_step = self.num_timesteps
        self._last_checkpoint_step = self.num_timesteps
        if self.verbose:
            print(f"[Curriculum] eval every {self.eval_freq:,} steps; tolerance ladder "
                  f"{self.tolerance_ladder} (now {self.current_tolerance}), hold ladder "
                  f"{self.hold_ladder} (now {self.current_hold_steps}); production standard "
                  f"tol {self.standard.tolerance} m, hold {self.standard.hold_steps}; run dir {self.run_dir}")

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self._last_eval_step = self.num_timesteps
            self._run_round()
        if self.checkpoint_every > 0 and self.num_timesteps - self._last_checkpoint_step >= self.checkpoint_every:
            self._last_checkpoint_step = self.num_timesteps
            self._save_checkpoint()
        return True

    def _on_training_end(self) -> None:
        self._save_last(with_buffer=True)
        self._write_state()

    # ------------------------------------------------------------------ #
    def _get_eval_env(self):
        if self._eval_env is None:
            self._eval_env = self.eval_env_fn()
        return self._eval_env

    def _run_round(self) -> None:
        env = self._get_eval_env()
        base = getattr(env, "unwrapped", env)
        self.rounds += 1
        round_seed = int(self.standard.seed) + 1000 * self.rounds

        # 1. training-difficulty eval
        base.set_goal_tolerance(self.current_tolerance)
        base.set_hold_steps_required(self.current_hold_steps)
        base.set_terminate_on_success(True)
        train_res = evaluate_policy(self.model, env, n_episodes=self.n_train_eval_episodes,
                                    seed=round_seed, human_likeness=False)
        self._last_train_success = train_res.success_rate

        # 2. production eval
        base.set_goal_tolerance(self.standard.tolerance)
        base.set_hold_steps_required(self.standard.hold_steps)
        base.set_hold_decrement(self.standard.hold_decrement)
        base.set_terminate_on_success(True)
        prod_res = evaluate_policy(self.model, env, n_episodes=self.standard.n_episodes,
                                   seed=self.standard.seed, max_steps=self.standard.max_steps)
        self._last_prod_success = prod_res.success_rate
        score = list(prod_res.score())

        improved = self.best_score is None or score > list(self.best_score)
        if improved:
            self.best_score = score
            self.best_saved_at = int(self.num_timesteps)
            self.model.save(str(self.run_dir / self.model_stem) + "_best")
            prod_res.save(self.run_dir / PACKAGE_FILES["best_eval"])
        self._save_last(with_buffer=False)

        # 3. curriculum advance (one ladder per round, cooldown one round)
        advanced = None
        if self._cooldown > 0:
            self._cooldown -= 1
        elif train_res.success_rate >= self.advance_threshold:
            if self.tolerance_idx < len(self.tolerance_ladder) - 1:
                self.tolerance_idx += 1
                advanced = f"tolerance -> {self.current_tolerance:.3f} m"
            elif self.hold_idx < len(self.hold_ladder) - 1:
                self.hold_idx += 1
                advanced = f"hold -> {self.current_hold_steps} steps"
            if advanced:
                self._apply_training_difficulty()
                self._cooldown = 1

        entry = {
            "step": int(self.num_timesteps),
            "train_tolerance": self.current_tolerance,
            "train_hold_steps": self.current_hold_steps,
            "train_success_rate": train_res.success_rate,
            "train_mean_waypoints": train_res.mean_waypoints,
            "prod_success_rate": prod_res.success_rate,
            "prod_mean_waypoints": prod_res.mean_waypoints,
            "prod_mean_max_hold": prod_res.mean_max_hold,
            "prod_mean_final_distance": prod_res.mean_final_distance,
            "prod_score": score,
            "improved": bool(improved),
            "advanced": advanced,
        }
        self.history.append(entry)
        self._write_state()
        if self.verbose:
            print(f"[Curriculum] step {self.num_timesteps:,}: train(tol {entry['train_tolerance']:.3f}, "
                  f"hold {entry['train_hold_steps']}) {train_res.summary_line()} | "
                  f"production {prod_res.summary_line()}"
                  + (" | NEW BEST" if improved else "")
                  + (f" | {advanced}" if advanced else ""), flush=True)

    # ------------------------------------------------------------------ #
    def _save_last(self, with_buffer: bool) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.run_dir / self.model_stem) + "_last")
        if with_buffer and self.save_replay_buffer and hasattr(self.model, "save_replay_buffer"):
            try:
                self.model.save_replay_buffer(str(self.run_dir / PACKAGE_FILES["replay_buffer"]))
            except Exception as exc:  # pragma: no cover
                if self.verbose:
                    print(f"[Curriculum] replay buffer not saved: {exc}")

    def _save_checkpoint(self) -> None:
        ck = self.run_dir / PACKAGE_FILES["checkpoints"]
        ck.mkdir(parents=True, exist_ok=True)
        self.model.save(str(ck / f"step_{int(self.num_timesteps):07d}"))
        self._save_last(with_buffer=True)
        self._write_state()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "tolerance_ladder": self.tolerance_ladder,
            "hold_ladder": self.hold_ladder,
            "tolerance_idx": self.tolerance_idx,
            "hold_idx": self.hold_idx,
            "best_score": self.best_score,
            "best_saved_at": self.best_saved_at,
            "num_timesteps": int(self.num_timesteps) if self.model is not None else 0,
            "rounds": self.rounds,
            "cooldown": self._cooldown,
            "history": self.history,
        }

    def _write_state(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / PACKAGE_FILES["curriculum"], "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, indent=2)

    def _load_state(self, st: Dict[str, Any]) -> None:
        self.tolerance_idx = int(st.get("tolerance_idx", 0))
        self.hold_idx = int(st.get("hold_idx", 0))
        self.best_score = st.get("best_score")
        self.best_saved_at = st.get("best_saved_at")
        self.rounds = int(st.get("rounds", 0))
        self._cooldown = int(st.get("cooldown", 0))
        self.history = list(st.get("history", []))
        self.tolerance_idx = min(self.tolerance_idx, len(self.tolerance_ladder) - 1)
        self.hold_idx = min(self.hold_idx, len(self.hold_ladder) - 1)

    def get_progress(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "current_tolerance": float(self.current_tolerance),
            "initial_tolerance": float(self.tolerance_ladder[0]),
            "min_tolerance": float(self.tolerance_ladder[-1]),
            "current_hold_steps": int(self.current_hold_steps),
            "hold_ladder": self.hold_ladder,
            "tolerance_ladder": self.tolerance_ladder,
            "recent_success_rate": self._last_train_success,
            "production_success_rate": self._last_prod_success,
            "best_score": self.best_score,
            "best_saved_at": self.best_saved_at,
            "eval_freq": int(self.eval_freq),
            "curriculum_stage": (int(self.tolerance_idx), int(self.hold_idx)),
            "rounds": self.rounds,
            "window_filled": self.n_train_eval_episodes,
            "window_size": self.n_train_eval_episodes,
            "episodes_since_last_decay": 0,
        }
