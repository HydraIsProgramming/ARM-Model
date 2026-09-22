"""One evaluation standard, one evaluator, one model-package loader.

Every script, callback and GUI that measures a policy goes through
``evaluate_policy`` with ``EVAL_STANDARD``; every one that loads a saved model
goes through ``load_package`` so it runs in exactly the env the model was
trained in (rebuilt from ``run_manifest.json``).

Package layout written by training (see training/ppo_trainer_wrapper.py)::

    <run_dir>/
      run_manifest.json        seed, threads, versions, env.describe(), hyperparameters, lineage
      sac_model_best.zip       best production-eval checkpoint      <- demos load this
      sac_model_last.zip       most recent checkpoint
      replay_buffer.pkl        SB3 replay buffer at the last checkpoint
      curriculum_state.json    ladder indices, best score, history
      best_eval.json           per-episode rows of the best checkpoint's evaluation
      checkpoints/step_XXXXXX.zip
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from rl_armMotion.two_d.environments.task_env import ArmTaskEnv, DEFAULT_PRESET, OBS_VERSION
from rl_armMotion.two_d.kinematics_metrics import episode_metrics

__all__ = [
    "EvalStandard", "EVAL_STANDARD", "WAYPOINT_A", "WAYPOINT_B", "WAYPOINT_C",
    "CANONICAL_WAYPOINTS", "CANONICAL_WAYPOINTS_V1", "WAYPOINTS_SPREAD", "resolve_waypoints", "make_env",
    "detect_actuation_mode", "EpisodeResult", "EvalResult", "evaluate_policy",
    "find_manifest", "read_manifest", "env_from_manifest", "load_package", "standard_overrides",
    "PACKAGE_FILES", "DEFAULT_PRESET", "OBS_VERSION",
]


@dataclass(frozen=True)
class EvalStandard:
    max_steps: int = ArmTaskEnv.DEFAULT_MAX_EPISODE_STEPS
    tolerance: float = ArmTaskEnv.DEFAULT_GOAL_TOLERANCE
    hold_steps: int = ArmTaskEnv.DEFAULT_HOLD_STEPS
    hold_decrement: int = ArmTaskEnv.DEFAULT_HOLD_DECREMENT
    ee_speed_tolerance: float = ArmTaskEnv.DEFAULT_EE_SPEED_TOLERANCE
    joint_speed_tolerance: float = ArmTaskEnv.DEFAULT_JOINT_SPEED_TOLERANCE
    intermediate_hold_steps: int = ArmTaskEnv.DEFAULT_INTERMEDIATE_HOLD_STEPS
    n_episodes: int = 10
    seed: int = 20260921

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


EVAL_STANDARD = EvalStandard()

# Canonical waypoints, workspace frame, shoulder at [1.0, 0.0], right arm.
# IK-checked inside the human range of motion; every point sits >= 0.19 m
# inside the 0.65 m reach circle, i.e. farther than the coarsest curriculum
# tolerance (0.15 m). Stage B on the v1 set showed why that matters: with the
# final waypoint 9 cm from the reach circle, a straight arm resting on the
# elbow's end-range spring counted as "in goal" at 15 cm, and policies that
# learned that rest could not unlearn it when the tolerance tightened.
WAYPOINT_A = [1.40, 0.15]
WAYPOINT_B = [1.20, 0.40]
WAYPOINT_C = [1.40, -0.20]
CANONICAL_WAYPOINTS: Dict[int, List[List[float]]] = {
    1: [WAYPOINT_A],
    2: [WAYPOINT_A, WAYPOINT_C],
    3: [WAYPOINT_A, WAYPOINT_B, WAYPOINT_C],
}
# v1 set used by Stage A / B / B2 packages (kept for reference; manifests carry their own waypoints).
CANONICAL_WAYPOINTS_V1: Dict[int, List[List[float]]] = {
    1: [[1.45, 0.15]],
    2: [[1.45, 0.15], [1.50, -0.25]],
    3: [[1.45, 0.15], [1.25, 0.45], [1.50, -0.25]],
}
REACH_MARGIN_EXTRA = 0.03   # m beyond the coarsest curriculum tolerance
# The earlier "wide triangle" set scaled from the 1.8 m arm to the 0.65 m arm.
# Its first point sits 6 degrees from the shoulder limit, so it is the
# alternative demo set, not the default.
WAYPOINTS_SPREAD: List[List[float]] = [[0.711, 0.433], [1.433, 0.181], [1.000, -0.578]]

PACKAGE_FILES = {
    "manifest": "run_manifest.json",
    "best": "sac_model_best.zip",
    "last": "sac_model_last.zip",
    "final": "sac_model.zip",
    "replay_buffer": "replay_buffer.pkl",
    "curriculum": "curriculum_state.json",
    "best_eval": "best_eval.json",
    "checkpoints": "checkpoints",
}


def resolve_waypoints(spec: Union[None, int, str, Sequence[Sequence[float]]]) -> Optional[List[List[float]]]:
    """Accept 2 / 3 / "spread" / "x,y;x,y" / a list; None passes through."""
    if spec is None:
        return None
    if isinstance(spec, (int, np.integer)):
        return [list(w) for w in CANONICAL_WAYPOINTS[int(spec)]]
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in ("", "none"):
            return None
        if s == "spread":
            return [list(w) for w in WAYPOINTS_SPREAD]
        if s.endswith("v1") and s[:-2].isdigit():
            return [list(w) for w in CANONICAL_WAYPOINTS_V1[int(s[:-2])]]
        if s.isdigit():
            return [list(w) for w in CANONICAL_WAYPOINTS[int(s)]]
        return [[float(x) for x in pair.split(",")] for pair in s.split(";") if pair.strip()]
    return [list(map(float, w)) for w in spec]


def make_env(
    actuation_mode: str = ArmTaskEnv.ACTUATION_MUSCLE,
    waypoints: Union[None, int, str, Sequence[Sequence[float]]] = None,
    preset: str = DEFAULT_PRESET,
    standard: EvalStandard = EVAL_STANDARD,
    tolerance: Optional[float] = None,
    hold_steps: Optional[int] = None,
    terminate_on_success: bool = False,
    goal_direction: str = "EAST",
    **env_kwargs: Any,
) -> ArmTaskEnv:
    """Build an env at the evaluation standard (overridable per field)."""
    env = ArmTaskEnv(
        preset=preset,
        actuation_mode=actuation_mode,
        goal_direction=goal_direction,
        goal_tolerance=standard.tolerance if tolerance is None else tolerance,
        ee_speed_tolerance=standard.ee_speed_tolerance,
        joint_speed_tolerance=standard.joint_speed_tolerance,
        hold_steps_required=standard.hold_steps if hold_steps is None else hold_steps,
        hold_decrement=standard.hold_decrement,
        intermediate_hold_steps=standard.intermediate_hold_steps,
        max_episode_steps=standard.max_steps,
        terminate_on_success=terminate_on_success,
        **env_kwargs,
    )
    wps = resolve_waypoints(waypoints)
    if wps:
        env.set_waypoints(wps)
    return env


def detect_actuation_mode(model: Any) -> str:
    """Infer the actuation mode from a loaded model's action space."""
    space = getattr(model, "action_space", None)
    if space is None:
        raise ValueError("model has no action_space")
    low_min = float(np.min(space.low))
    if low_min >= 0.0:
        return ArmTaskEnv.ACTUATION_MUSCLE
    return ArmTaskEnv.ACTUATION_TORQUE


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeResult:
    seed: int
    success: bool
    success_step: Optional[int]
    waypoints_reached: int
    num_waypoints: int
    max_hold_progress: float
    final_distance: float
    min_final_distance: float
    steps: int
    episode_return: float
    hold_steps_after_success: int = 0
    waypoints: Optional[List[List[float]]] = None
    human_likeness: Optional[Dict[str, Any]] = None
    trajectory: Optional[Dict[str, np.ndarray]] = None

    def to_row(self) -> Dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k != "trajectory"}
        if d.get("human_likeness"):
            d["human_likeness"] = {k: v for k, v in d["human_likeness"].items() if k != "segments"}
        return d


@dataclass
class EvalResult:
    episodes: List[EpisodeResult]
    standard: Dict[str, Any] = field(default_factory=dict)
    env_description: Dict[str, Any] = field(default_factory=dict)

    # --- aggregates --------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.episodes)

    @property
    def success_rate(self) -> float:
        return float(np.mean([e.success for e in self.episodes])) if self.episodes else 0.0

    @property
    def successes(self) -> int:
        return int(sum(e.success for e in self.episodes))

    @property
    def mean_waypoints(self) -> float:
        return float(np.mean([e.waypoints_reached for e in self.episodes])) if self.episodes else 0.0

    @property
    def mean_max_hold(self) -> float:
        return float(np.mean([e.max_hold_progress for e in self.episodes])) if self.episodes else 0.0

    @property
    def mean_final_distance(self) -> float:
        return float(np.mean([e.min_final_distance for e in self.episodes])) if self.episodes else float("inf")

    @property
    def mean_steps_to_success(self) -> float:
        s = [e.success_step for e in self.episodes if e.success and e.success_step is not None]
        return float(np.mean(s)) if s else float("inf")

    @property
    def mean_return(self) -> float:
        return float(np.mean([e.episode_return for e in self.episodes])) if self.episodes else 0.0

    def sd(self, attr: str) -> float:
        vals = [getattr(e, attr) for e in self.episodes]
        vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
        return float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0

    def score(self) -> Tuple[float, float, float, float]:
        """Continuous, lexicographic: more successes, then more progress
        (waypoints + hold), then closer, then faster."""
        steps = self.mean_steps_to_success
        return (
            round(self.success_rate, 6),
            round(self.mean_waypoints + self.mean_max_hold, 6),
            round(-self.mean_final_distance, 6),
            round(-(steps if np.isfinite(steps) else 1e9), 6),
        )

    def human_likeness_summary(self) -> Dict[str, float]:
        keys = ["straightness", "symmetry", "submovements", "dimensionless_jerk", "movement_time", "peak_speed"]
        out = {}
        for k in keys:
            vals = [e.human_likeness[k] for e in self.episodes
                    if e.human_likeness and np.isfinite(e.human_likeness.get(k, float("nan")))]
            out[k] = float(np.mean(vals)) if vals else float("nan")
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "n_episodes": self.n,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "mean_waypoints_reached": self.mean_waypoints,
            "sd_waypoints_reached": self.sd("waypoints_reached"),
            "mean_max_hold_progress": self.mean_max_hold,
            "mean_min_final_distance": self.mean_final_distance,
            "mean_steps_to_success": self.mean_steps_to_success if np.isfinite(self.mean_steps_to_success) else None,
            "mean_return": self.mean_return,
            "sd_return": self.sd("episode_return"),
            "score": list(self.score()),
            "human_likeness": self.human_likeness_summary(),
        }

    def summary_line(self) -> str:
        s = self.summary()
        steps = s["mean_steps_to_success"]
        return (f"{s['successes']}/{s['n_episodes']} success, wp {s['mean_waypoints_reached']:.2f}, "
                f"hold {s['mean_max_hold_progress']:.2f}, dist {s['mean_min_final_distance']:.3f} m, "
                f"steps {steps:.0f}" if steps is not None else
                f"{s['successes']}/{s['n_episodes']} success, wp {s['mean_waypoints_reached']:.2f}, "
                f"hold {s['mean_max_hold_progress']:.2f}, dist {s['mean_min_final_distance']:.3f} m")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "standard": self.standard,
            "env": self.env_description,
            "episodes": [e.to_row() for e in self.episodes],
        }

    def save(self, path: Union[str, Path], trajectories: bool = False) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=_json_default)
        if trajectories:
            arrays = {}
            for i, e in enumerate(self.episodes):
                if e.trajectory:
                    for k, v in e.trajectory.items():
                        arrays[f"ep{i}_{k}"] = np.asarray(v)
            if arrays:
                np.savez_compressed(path.with_suffix(".trajectories.npz"), **arrays)
        return path


def _json_default(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return str(o)


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
def evaluate_policy(
    model: Any,
    env: ArmTaskEnv,
    n_episodes: Optional[int] = None,
    seed: Optional[int] = None,
    max_steps: Optional[int] = None,
    deterministic: bool = True,
    log_trajectories: bool = False,
    human_likeness: bool = True,
    progress: Optional[Callable[[int, EpisodeResult], None]] = None,
) -> EvalResult:
    """Roll out ``model`` for N seeded episodes; every episode is a distinct sample."""
    n_episodes = int(EVAL_STANDARD.n_episodes if n_episodes is None else n_episodes)
    seed = int(EVAL_STANDARD.seed if seed is None else seed)
    base = getattr(env, "unwrapped", env)
    max_steps = int(base.max_episode_steps if max_steps is None else max_steps)
    dt = float(base.dt)
    episodes: List[EpisodeResult] = []
    for i in range(n_episodes):
        ep_seed = seed + i
        obs, info = env.reset(seed=ep_seed)
        ret = 0.0
        max_hold = 0.0
        min_final = float("inf")
        final_idx = base.num_waypoints - 1
        final_goal = np.asarray(base.waypoints[-1] if base.waypoints is not None else base.goal_position, dtype=float)
        hold_after = 0
        pos_log, idx_log, q_log, qd_log, a_log, tau_log, v_log = [], [], [], [], [], [], []
        pos_log.append(np.asarray(info["end_effector_position"], dtype=float))
        idx_log.append(int(info["current_waypoint_index"]))
        steps = 0
        for t in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(action)
            steps = t + 1
            ret += float(r)
            if info["current_waypoint_index"] == final_idx:
                max_hold = max(max_hold, float(info["hold_progress"]))
            min_final = min(min_final, float(np.linalg.norm(final_goal - np.asarray(info["end_effector_position"], dtype=float))))
            if info["success"] and info["in_goal_region"]:
                hold_after += 1
            pos_log.append(np.asarray(info["end_effector_position"], dtype=float))
            idx_log.append(int(info["current_waypoint_index"]))
            if log_trajectories:
                q_log.append(np.asarray(info["joint_angles"], dtype=float))
                qd_log.append(np.asarray(info["joint_velocities"], dtype=float))
                a_log.append(np.asarray(action, dtype=float))
                tau_log.append(np.asarray(info["actuator_torque"], dtype=float))
                v_log.append(np.asarray(info["ee_velocity"], dtype=float))
            if terminated or truncated:
                break
        positions = np.asarray(pos_log)
        hl = episode_metrics(positions, np.asarray(idx_log), dt, info.get("success_step")) if human_likeness else None
        traj = None
        if log_trajectories:
            traj = {
                "hand": positions, "waypoint_index": np.asarray(idx_log), "q": np.asarray(q_log),
                "qd": np.asarray(qd_log), "action": np.asarray(a_log), "torque": np.asarray(tau_log),
                "hand_velocity": np.asarray(v_log),
            }
        ep = EpisodeResult(
            seed=ep_seed,
            success=bool(info["success"]),
            success_step=info.get("success_step"),
            waypoints=info.get("waypoints"),
            waypoints_reached=int(info["waypoints_reached"]),
            num_waypoints=int(info["num_waypoints"]),
            max_hold_progress=float(max_hold),
            final_distance=float(np.linalg.norm(final_goal - np.asarray(info["end_effector_position"], dtype=float))),
            min_final_distance=float(min_final),
            steps=int(steps),
            episode_return=float(ret),
            hold_steps_after_success=int(hold_after),
            human_likeness=hl,
            trajectory=traj,
        )
        episodes.append(ep)
        if progress is not None:
            progress(i, ep)
    std = {"n_episodes": n_episodes, "seed": seed, "max_steps": max_steps, "deterministic": deterministic,
           "tolerance": float(base.goal_tolerance), "hold_steps": int(base.hold_steps_required),
           "hold_decrement": int(base.hold_decrement), "ee_speed_tolerance": float(base.ee_speed_tolerance)}
    return EvalResult(episodes=episodes, standard=std, env_description=base.describe())


# --------------------------------------------------------------------------- #
# Packages
# --------------------------------------------------------------------------- #
def find_manifest(path: Union[str, Path]) -> Optional[Path]:
    """Accept a package dir, a model zip inside one, or a manifest path."""
    p = Path(path)
    if p.is_file():
        if p.name == PACKAGE_FILES["manifest"]:
            return p
        cand = p.parent / PACKAGE_FILES["manifest"]
        if cand.exists():
            return cand
        cand = p.parent.parent / PACKAGE_FILES["manifest"]  # checkpoints/step_x.zip
        return cand if cand.exists() else None
    cand = p / PACKAGE_FILES["manifest"]
    return cand if cand.exists() else None


def read_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    mp = find_manifest(path)
    if mp is None:
        raise FileNotFoundError(f"no {PACKAGE_FILES['manifest']} found for {path}")
    with open(mp, "r", encoding="utf-8") as f:
        return json.load(f)


def env_from_manifest(manifest: Union[str, Path, Dict[str, Any]], **overrides: Any) -> ArmTaskEnv:
    """Rebuild the training env from a manifest (path or dict)."""
    data = manifest if isinstance(manifest, dict) else read_manifest(manifest)
    desc = data.get("env", data)
    return ArmTaskEnv.from_description(desc, **overrides)


def standard_overrides(standard: Optional[EvalStandard] = EVAL_STANDARD) -> Dict[str, Any]:
    """Env settings that put a rebuilt env at the evaluation standard.

    A run manifest records the env as it was at the START of training
    (curriculum tolerance 0.15 m, hold 10), so a model reloaded from it must
    be measured at the production standard, not at its warm-up difficulty.
    """
    if standard is None:
        return {}
    return {
        "goal_tolerance": standard.tolerance,
        "hold_steps_required": standard.hold_steps,
        "hold_decrement": standard.hold_decrement,
        "ee_speed_tolerance": standard.ee_speed_tolerance,
        "joint_speed_tolerance": standard.joint_speed_tolerance,
        "intermediate_hold_steps": standard.intermediate_hold_steps,
        "max_episode_steps": standard.max_steps,
    }


def load_package(
    run_dir: Union[str, Path],
    which: str = "best",
    device: str = "cpu",
    standard: Optional[EvalStandard] = EVAL_STANDARD,
    **env_overrides: Any,
) -> Tuple[Any, ArmTaskEnv, Dict[str, Any]]:
    """(model, env, manifest) for a saved run. ``which`` = best | last | final |
    step_XXXXXX | a path to a model zip. The env is rebuilt from the manifest
    and set to ``standard`` (pass ``standard=None`` to keep the manifest's
    training-start settings); ``env_overrides`` win over both. Refuses
    obs-version mismatches."""
    from stable_baselines3 import SAC, PPO, A2C

    overrides = standard_overrides(standard)
    overrides.update(env_overrides)
    env_overrides = overrides

    run = Path(run_dir)
    if run.is_file():
        model_path = run
        run = run.parent if run.parent.name != PACKAGE_FILES["checkpoints"] else run.parent.parent
    else:
        if which in PACKAGE_FILES:
            model_path = run / PACKAGE_FILES[which]
        elif which.startswith("step_"):
            model_path = run / PACKAGE_FILES["checkpoints"] / f"{which}.zip"
        else:
            model_path = Path(which)
        if not model_path.exists() and which == "best":
            model_path = run / PACKAGE_FILES["last"]
        if not model_path.exists() and which in ("best", "last"):
            model_path = run / PACKAGE_FILES["final"]
    if not model_path.exists():
        raise FileNotFoundError(f"model {model_path} not found")
    manifest = read_manifest(run)
    env = env_from_manifest(manifest, **env_overrides)
    versions = manifest.get("versions", {})
    _warn_version_drift(versions)
    algo = str(manifest.get("algorithm", "SAC")).upper()
    cls = {"SAC": SAC, "PPO": PPO, "A2C": A2C}.get(algo, SAC)
    model = cls.load(str(model_path), env=env, device=device)
    if tuple(model.observation_space.shape) != tuple(env.observation_space.shape):
        raise ValueError(
            f"model observation shape {model.observation_space.shape} != env "
            f"{env.observation_space.shape} (OBS_VERSION {OBS_VERSION}); the model was "
            f"trained on a different observation layout")
    return model, env, manifest


def _warn_version_drift(versions: Dict[str, str]) -> None:
    try:
        import warnings
        from rl_armMotion.two_d.training.reproducibility import collect_versions
        now = collect_versions()
        drift = {k: (versions.get(k), now.get(k)) for k in ("torch", "stable_baselines3", "gymnasium", "numpy")
                 if versions.get(k) and versions.get(k) != now.get(k)}
        if drift:
            warnings.warn(f"library versions differ from the run manifest: {drift}")
    except Exception:
        pass
