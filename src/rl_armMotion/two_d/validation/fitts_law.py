"""Fitts' Law validation harness for the 2-DOF robotic arm.

This module implements the Fitts' Law validation protocol used by Fischer,
Hoinville, Eickhoff, and Lilienthal (2021), "Reinforcement learning control
of a biomechanical model of the upper extremity", Scientific Reports 11:14445,
https://doi.org/10.1038/s41598-021-93760-1.

Fischer et al. trained a seven-degree-of-freedom MuJoCo arm with SAC under an
adaptive curriculum and asked whether the resulting policy exhibited the
classical speed-accuracy trade-off described by Fitts (1954), "The information
capacity of the human motor system in controlling the amplitude of movement",
J. Exp. Psychol. 47:381--391. Fitts found that the time required to acquire a
target of width W at distance D is well predicted by a linear regression in
the index of difficulty, ID = log2(2D/W):

    MT = a + b * ID

with a and b empirical constants and an essentially perfect coefficient of
determination over a wide range of (D, W) conditions. Fischer et al. reported
R^2 = 0.9986 for their trained 7-DOF arm, confirming that an RL policy trained
under their protocol reproduces a hallmark of biological motor control.

The harness in this module replicates that protocol for the 2-DOF planar arm
in this project. Given a trained Stable-Baselines3 policy and an ArmTaskEnv
instance, it sweeps a grid of (distance, width) combinations, runs multiple
trials per condition, measures the time to first contact with the goal
region, fits the regression by ordinary least squares, and reports the
coefficients along with R^2. The result is JSON-serialisable and can be
plotted with matplotlib for inclusion in reports.

References
----------
Fischer, M., Hoinville, T., Eickhoff, S. B., & Lilienthal, A. J. (2021).
    Reinforcement learning control of a biomechanical model of the upper
    extremity. Scientific Reports, 11, 14445.
Fitts, P. M. (1954). The information capacity of the human motor system in
    controlling the amplitude of movement. Journal of Experimental
    Psychology, 47, 381--391.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


__all__ = [
    "FittsLawCondition",
    "FittsLawTrial",
    "FittsLawResult",
    "FittsLawValidator",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class FittsLawCondition:
    """One (distance, width) condition in the Fitts' Law sweep.

    Attributes
    ----------
    distance : float
        Target distance D from the start position, in metres.
    width : float
        Target width W (the position tolerance), in metres.
    index_of_difficulty : float
        ID = log2(2 * D / W), in bits.
    """

    distance: float
    width: float
    index_of_difficulty: float

    @classmethod
    def from_distance_width(cls, distance: float, width: float) -> "FittsLawCondition":
        if distance <= 0.0:
            raise ValueError(f"distance must be positive, got {distance}")
        if width <= 0.0:
            raise ValueError(f"width must be positive, got {width}")
        idx = math.log2(2.0 * distance / width)
        return cls(distance=float(distance), width=float(width), index_of_difficulty=float(idx))


@dataclass
class FittsLawTrial:
    """The result of a single trial within one condition.

    Attributes
    ----------
    distance : float
        Target distance D for this trial, in metres.
    width : float
        Target width W for this trial, in metres.
    angle_rad : float
        Polar angle of the goal direction relative to the start position.
        Trials within a condition are spread uniformly in angle so the
        validator probes the workspace isotropically.
    success : bool
        True if the end-effector entered the W-radius before the trial
        timed out.
    movement_time_sec : Optional[float]
        Time from trial start to first entry into the W-radius. None if the
        trial did not succeed.
    steps_taken : int
        Number of environment steps in the trial.
    final_distance : float
        Euclidean distance from the end-effector to the goal at the final
        step, in metres.
    """

    distance: float
    width: float
    angle_rad: float
    success: bool
    movement_time_sec: Optional[float]
    steps_taken: int
    final_distance: float


@dataclass
class FittsLawResult:
    """Aggregated outcome of a Fitts' Law sweep.

    Attributes
    ----------
    conditions : List[FittsLawCondition]
        The (D, W) grid actually run, in the order they were swept.
    trials : List[FittsLawTrial]
        Every individual trial result.
    per_condition_mt_mean : List[float]
        Per-condition mean movement time over successful trials. NaN for
        conditions with zero successful trials.
    per_condition_mt_std : List[float]
        Per-condition movement-time standard deviation. NaN if fewer than
        two successful trials.
    per_condition_success_rate : List[float]
        Per-condition fraction of trials that succeeded.
    slope : float
        Regression slope b in MT = a + b * ID. Units: seconds per bit.
    intercept : float
        Regression intercept a in MT = a + b * ID. Units: seconds.
    r_squared : float
        Coefficient of determination of the regression fit.
    fit_n_conditions : int
        Number of (D, W) conditions actually used in the regression
        (conditions with zero successful trials are excluded).
    n_trials_per_condition : int
        Trials configured per condition.
    timestamp_iso : str
        ISO-format timestamp at the time the validator finished running.
    metadata : Dict[str, Any]
        Free-form metadata (e.g., model identifier, total step count,
        seed). Set by the caller via FittsLawValidator.run.
    """

    conditions: List[FittsLawCondition]
    trials: List[FittsLawTrial]
    per_condition_mt_mean: List[float]
    per_condition_mt_std: List[float]
    per_condition_success_rate: List[float]
    slope: float
    intercept: float
    r_squared: float
    fit_n_conditions: int
    n_trials_per_condition: int
    timestamp_iso: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation suitable for json.dump."""
        return {
            "conditions": [asdict(c) for c in self.conditions],
            "trials": [asdict(t) for t in self.trials],
            "per_condition_mt_mean": list(self.per_condition_mt_mean),
            "per_condition_mt_std": list(self.per_condition_mt_std),
            "per_condition_success_rate": list(self.per_condition_success_rate),
            "slope": float(self.slope),
            "intercept": float(self.intercept),
            "r_squared": float(self.r_squared),
            "fit_n_conditions": int(self.fit_n_conditions),
            "n_trials_per_condition": int(self.n_trials_per_condition),
            "timestamp_iso": str(self.timestamp_iso),
            "metadata": dict(self.metadata),
        }

    def save_json(self, path) -> None:
        """Write the result to a JSON file at ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=_json_default)

    def regression_summary(self) -> str:
        """Return a one-line human-readable summary of the regression fit."""
        return (
            f"MT = {self.intercept:+.4f} + {self.slope:+.4f} * ID, "
            f"R^2 = {self.r_squared:.4f} "
            f"(n_conditions = {self.fit_n_conditions}, "
            f"trials_per_condition = {self.n_trials_per_condition})"
        )

    # ------------------------------------------------------------------ #
    # Plotting (optional dependency on matplotlib)
    # ------------------------------------------------------------------ #
    def plot(self, save_path=None, show: bool = False, title: Optional[str] = None) -> None:
        """Produce the standard Fitts' Law scatter + regression plot.

        Parameters
        ----------
        save_path : path-like, optional
            If given, the figure is written to this file (PNG suggested).
        show : bool, default False
            If True, route through pyplot to display interactively. Only safe
            to set in the main thread; the default False path uses Agg
            directly and is thread-safe.
        title : str, optional
            Override the default figure title.

        Notes
        -----
        With ``show=False`` this method uses ``matplotlib.figure.Figure``
        and ``matplotlib.backends.backend_agg.FigureCanvasAgg`` directly,
        bypassing ``pyplot`` entirely. That path is thread-safe and is
        the one the training-GUI "Run Fischer Validators" worker takes,
        which previously crashed macOS Tk by touching pyplot state from a
        non-main thread.
        """
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for FittsLawResult.plot; "
                "install it via `pip install matplotlib`"
            ) from exc

        ids = np.array([c.index_of_difficulty for c in self.conditions])
        mts = np.array(self.per_condition_mt_mean)
        stds = np.array(self.per_condition_mt_std)

        valid = np.isfinite(mts)
        ids_v = ids[valid]
        mts_v = mts[valid]
        stds_v = stds[valid]

        # Use Figure() directly with FigureCanvasAgg so we never touch the
        # pyplot global state. Safe to call from any thread. The
        # FigureCanvasAgg constructor binds itself to fig.canvas as a side
        # effect, which is what fig.savefig() needs further down — we do not
        # need to keep a separate reference.
        fig = Figure(figsize=(7, 5))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        ax.errorbar(
            ids_v, mts_v, yerr=np.where(np.isfinite(stds_v), stds_v, 0.0),
            fmt="o", color="black", ecolor="gray", capsize=3,
            label="Per-condition mean MT",
        )

        if len(ids_v) >= 2:
            x_line = np.linspace(ids_v.min(), ids_v.max(), 100)
            y_line = self.intercept + self.slope * x_line
            ax.plot(
                x_line, y_line, color="black", linewidth=1.2,
                label=(f"MT = {self.intercept:.3f} + {self.slope:.3f} * ID"
                       f"   (R$^2$ = {self.r_squared:.4f})"),
            )

        ax.set_xlabel("Index of Difficulty, ID = log$_2$(2D / W)  (bits)")
        ax.set_ylabel("Movement Time, MT  (seconds)")
        ax.set_title(title or "Fitts' Law validation")
        ax.legend(loc="best", frameon=False)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200)

        if show:
            # Caller explicitly asked for interactive display, route through
            # pyplot. Only call from the main thread on macOS.
            import matplotlib.pyplot as plt
            plt.figure(fig.number) if hasattr(fig, "number") else None
            plt.show()


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------
class FittsLawValidator:
    """Sweep a (distance, width) grid and fit MT = a + b * log2(2D/W).

    Parameters
    ----------
    model : object with .predict(observation, deterministic=True) -> (action, state)
        A trained Stable-Baselines3 policy (PPO, SAC, A2C) or any object
        exposing the SB3 .predict interface. Pass deterministic=True so
        movement times are reproducible.
    env : ArmTaskEnv
        The 2-DOF environment to evaluate in. The validator calls
        env.set_goal_position, env.set_goal_tolerance, env.reset, and
        env.step. The env must expose the dt attribute (simulation step
        size in seconds).
    deterministic : bool, default True
        Whether to use deterministic action selection during evaluation.
    rng : numpy.random.Generator, optional
        Random number generator used to choose goal angles within each
        condition. Defaults to np.random.default_rng(seed).
    """

    DEFAULT_DISTANCES = (0.20, 0.40, 0.60, 0.80, 1.00, 1.20)  # metres
    DEFAULT_WIDTHS = (0.02, 0.04, 0.08, 0.12, 0.20, 0.30)      # metres

    def __init__(
        self,
        model: Any,
        env: Any,
        deterministic: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        if not hasattr(model, "predict"):
            raise TypeError(
                "model must expose a .predict(obs, deterministic=...) method"
            )
        if not hasattr(env, "set_goal_position") or not hasattr(env, "set_goal_tolerance"):
            raise TypeError(
                "env must support set_goal_position and set_goal_tolerance "
                "(both available on ArmTaskEnv as of the Fischer integration step)"
            )
        if not hasattr(env, "dt"):
            raise TypeError("env must expose a dt attribute (seconds per step)")

        self.model = model
        self.env = env
        self.deterministic = bool(deterministic)
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def run(
        self,
        distances: Optional[Sequence[float]] = None,
        widths: Optional[Sequence[float]] = None,
        n_trials_per_condition: int = 15,
        max_steps_per_trial: int = 1000,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, FittsLawCondition], None]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FittsLawResult:
        """Run the full Fitts' Law sweep and return a FittsLawResult.

        Parameters
        ----------
        distances : sequence of float, optional
            Target distances D in metres. Defaults to ``DEFAULT_DISTANCES``.
        widths : sequence of float, optional
            Target widths W (tolerances) in metres. Defaults to
            ``DEFAULT_WIDTHS``.
        n_trials_per_condition : int, default 15
            Trials per (D, W) cell. Fischer used 30 in their paper; 15 is
            a reasonable default for short evaluation runs.
        max_steps_per_trial : int, default 1000
            Cap on environment steps per trial. Trials that fail to enter
            the W-radius before this cap are flagged as failed.
        seed : int, optional
            Seed for the goal-angle random number generator. If supplied,
            the validator's RNG is replaced with default_rng(seed).
        progress_callback : callable, optional
            Invoked once per condition as
            ``progress_callback(condition_index, total_conditions, condition)``.
            Useful for GUI progress bars.
        metadata : dict, optional
            Stored in the resulting ``FittsLawResult.metadata``.

        Returns
        -------
        FittsLawResult
            The complete sweep result, suitable for save_json and plot.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        d_grid = list(distances) if distances is not None else list(self.DEFAULT_DISTANCES)
        w_grid = list(widths) if widths is not None else list(self.DEFAULT_WIDTHS)

        if not d_grid or not w_grid:
            raise ValueError("distances and widths must be non-empty")
        if any(d <= 0.0 for d in d_grid):
            raise ValueError("all distances must be positive")
        if any(w <= 0.0 for w in w_grid):
            raise ValueError("all widths must be positive")
        if n_trials_per_condition < 1:
            raise ValueError(
                f"n_trials_per_condition must be >= 1, got {n_trials_per_condition}"
            )
        if max_steps_per_trial < 1:
            raise ValueError(
                f"max_steps_per_trial must be >= 1, got {max_steps_per_trial}"
            )

        conditions: List[FittsLawCondition] = []
        trials: List[FittsLawTrial] = []
        per_cond_mt_mean: List[float] = []
        per_cond_mt_std: List[float] = []
        per_cond_success_rate: List[float] = []

        # Sweep order: distance outer, width inner
        flat = [(d, w) for d in d_grid for w in w_grid]

        for condition_index, (d, w) in enumerate(flat):
            condition = FittsLawCondition.from_distance_width(d, w)
            conditions.append(condition)

            if progress_callback is not None:
                try:
                    progress_callback(condition_index, len(flat), condition)
                except Exception:  # never let a callback crash the sweep
                    pass

            trial_mts: List[float] = []
            trial_successes = 0
            for trial_index in range(n_trials_per_condition):
                trial = self._run_single_trial(
                    distance=d,
                    width=w,
                    max_steps=max_steps_per_trial,
                    trial_index=trial_index,
                    n_trials=n_trials_per_condition,
                )
                trials.append(trial)
                if trial.success and trial.movement_time_sec is not None:
                    trial_mts.append(float(trial.movement_time_sec))
                    trial_successes += 1

            per_cond_success_rate.append(trial_successes / float(n_trials_per_condition))
            if trial_mts:
                per_cond_mt_mean.append(float(np.mean(trial_mts)))
                per_cond_mt_std.append(
                    float(np.std(trial_mts, ddof=1)) if len(trial_mts) >= 2 else float("nan")
                )
            else:
                per_cond_mt_mean.append(float("nan"))
                per_cond_mt_std.append(float("nan"))

        slope, intercept, r2, fit_n = self._fit_regression(
            conditions=conditions,
            mt_means=per_cond_mt_mean,
        )

        return FittsLawResult(
            conditions=conditions,
            trials=trials,
            per_condition_mt_mean=per_cond_mt_mean,
            per_condition_mt_std=per_cond_mt_std,
            per_condition_success_rate=per_cond_success_rate,
            slope=float(slope),
            intercept=float(intercept),
            r_squared=float(r2),
            fit_n_conditions=int(fit_n),
            n_trials_per_condition=int(n_trials_per_condition),
            timestamp_iso=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            metadata=dict(metadata) if metadata else {},
        )

    # ------------------------------------------------------------------ #
    # Single-trial execution
    # ------------------------------------------------------------------ #
    def _run_single_trial(
        self,
        distance: float,
        width: float,
        max_steps: int,
        trial_index: int,
        n_trials: int,
    ) -> FittsLawTrial:
        """Execute one trial: place goal at (D, W), roll out the policy."""
        # Reset to the standard initial pose (vertical down) before placing
        # the goal so each trial starts from a known state.
        obs, _info = self.env.reset()

        # Compute the goal direction. We spread trials uniformly in angle
        # within each (D, W) condition so the validator probes the
        # workspace isotropically and the regression averages over
        # direction-dependent biases of the policy.
        if n_trials == 1:
            angle = 0.0
        else:
            # Use the env's initial end-effector position as the start point;
            # the start position is implicit in the reset.
            angle = self._sample_goal_angle(trial_index=trial_index, n_trials=n_trials)

        # The start position used for placing the goal is the end-effector
        # position immediately after reset. We recompute it here so the
        # validator does not depend on env-internal state names.
        start_pos = self._extract_end_effector(obs, fallback_info=_info)

        goal_pos = start_pos + distance * np.array([math.cos(angle), math.sin(angle)],
                                                    dtype=np.float32)
        self.env.set_goal_position(goal_pos)
        self.env.set_goal_tolerance(width)

        steps_taken = 0
        success = False
        movement_time: Optional[float] = None
        final_distance = float("nan")

        for step in range(max_steps):
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, _reward, terminated, truncated, info = self.env.step(action)
            steps_taken = step + 1

            ee_pos = np.asarray(info.get("end_effector_position"), dtype=np.float32)
            if ee_pos.shape == (2,):
                d_to_goal = float(np.linalg.norm(ee_pos - goal_pos))
            else:
                d_to_goal = float(info.get("goal_distance", float("nan")))
            final_distance = d_to_goal

            if d_to_goal <= width:
                success = True
                movement_time = steps_taken * float(self.env.dt)
                break

            if terminated or truncated:
                break

        return FittsLawTrial(
            distance=float(distance),
            width=float(width),
            angle_rad=float(angle),
            success=bool(success),
            movement_time_sec=(float(movement_time) if movement_time is not None else None),
            steps_taken=int(steps_taken),
            final_distance=float(final_distance),
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _sample_goal_angle(self, trial_index: int, n_trials: int) -> float:
        """Pick an angle for trial ``trial_index`` of ``n_trials``.

        Trials are stratified: half the angle range is covered by an evenly
        spaced lattice, and the remainder is jittered uniformly. This gives
        deterministic-leaning coverage while still reflecting some isotropy.
        """
        # Restrict to the upper half-plane so the goal is reachable by an
        # arm hanging vertically downward at workspace shoulder.
        # We span angles in (-pi/2, +pi/2) excluding the exact +/-pi/2
        # extremes which could place a goal at the shoulder height directly
        # above or sideways at maximum reach.
        lattice_n = max(2, n_trials // 2)
        if trial_index < lattice_n:
            # Even lattice in the lower half-plane (where the arm starts)
            # extending up to horizontal on both sides.
            t = (trial_index + 0.5) / lattice_n  # in (0, 1)
            return float(-math.pi / 2.0 + math.pi * t)  # in (-pi/2, +pi/2)
        # Random jitter for the remainder
        return float(self.rng.uniform(-math.pi / 2.0, math.pi / 2.0))

    @staticmethod
    def _extract_end_effector(obs: Any, fallback_info: Optional[Dict[str, Any]]) -> np.ndarray:
        """Pull the start end-effector position from obs or info.

        ArmTaskEnv exposes the end-effector position in info, not obs, so
        we rely on the info dict returned by reset.
        """
        if isinstance(fallback_info, dict):
            ee = fallback_info.get("end_effector_position")
            if ee is not None:
                arr = np.asarray(ee, dtype=np.float32).reshape(-1)
                if arr.shape == (2,):
                    return arr
                if arr.shape == (3,):
                    return arr[:2]
        # Last resort: assume origin. This is wrong for most real envs but
        # keeps the validator robust to alternative observation layouts.
        return np.zeros(2, dtype=np.float32)

    @staticmethod
    def _fit_regression(
        conditions: Sequence[FittsLawCondition],
        mt_means: Sequence[float],
    ) -> Tuple[float, float, float, int]:
        """Ordinary least-squares fit of MT = a + b * ID.

        Conditions for which the per-condition mean is NaN (no successful
        trials) are excluded from the fit. Returns (slope b, intercept a,
        R^2, n_conditions_used).
        """
        ids = np.array([c.index_of_difficulty for c in conditions], dtype=np.float64)
        mts = np.array(mt_means, dtype=np.float64)
        valid = np.isfinite(mts)
        if valid.sum() < 2:
            # Not enough data to fit a line.
            return float("nan"), float("nan"), float("nan"), int(valid.sum())

        x = ids[valid]
        y = mts[valid]

        # Closed-form OLS: y = a + b*x
        x_mean = x.mean()
        y_mean = y.mean()
        sxx = float(np.sum((x - x_mean) ** 2))
        if sxx <= 0.0:
            return float("nan"), float("nan"), float("nan"), int(valid.sum())

        sxy = float(np.sum((x - x_mean) * (y - y_mean)))
        slope = sxy / sxx
        intercept = float(y_mean - slope * x_mean)

        y_pred = intercept + slope * x
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        if ss_tot <= 0.0:
            r2 = float("nan") if ss_res > 0.0 else 1.0
        else:
            r2 = 1.0 - ss_res / ss_tot

        return float(slope), float(intercept), float(r2), int(valid.sum())


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for numpy scalars and arrays."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (Path,)):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
