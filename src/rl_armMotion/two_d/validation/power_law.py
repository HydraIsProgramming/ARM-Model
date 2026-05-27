"""Two-thirds Power Law validation harness for the 2-DOF robotic arm.

This module implements the second of the two emergent-behaviour benchmarks
used by Fischer, Hoinville, Eickhoff, and Lilienthal (2021), "Reinforcement
learning control of a biomechanical model of the upper extremity",
Scientific Reports 11:14445, https://doi.org/10.1038/s41598-021-93760-1.

The 2/3 Power Law was originally reported by Lacquaniti, Terzuolo, and
Viviani (1983), "The law relating the kinematic and figural aspects of
drawing movements", Acta Psychologica 54:115--130. They observed that when
humans trace planar curves, the tangential speed V of the hand and the
curvature C of the path satisfy

    V = K * C ** (-1 / 3),

equivalently log V = log K - (1/3) * log C,

with K ("the velocity gain factor") approximately constant within a
movement segment. The slope of -1/3 in log-log space is the empirical
fingerprint of natural human motion. Fischer et al. (2021) reported a
correlation coefficient R = 0.84 between log V and log C for their
trained SAC policy on a 7-DOF MuJoCo arm, demonstrating that an RL
agent under the right reward and curriculum reproduces this property of
biological motor control.

The harness in this module replicates that protocol for the 2-DOF planar
arm in this project. Given a trained policy and an ArmTaskEnv, it runs a
configurable number of goal-reaching trials, logs per-step end-effector
positions, computes V and C by central differences, filters out
near-stationary or near-straight samples where the formula is unstable,
fits log V against log C by ordinary least squares, and reports slope,
intercept, R, R^2, and a log-log scatter plot.

References
----------
Fischer, M., Hoinville, T., Eickhoff, S. B., & Lilienthal, A. J. (2021).
    Reinforcement learning control of a biomechanical model of the upper
    extremity. Scientific Reports, 11, 14445.
Lacquaniti, F., Terzuolo, C., & Viviani, P. (1983). The law relating the
    kinematic and figural aspects of drawing movements. Acta Psychologica,
    54, 115--130.
Viviani, P., & Stucchi, N. (1992). Biological movements look uniform:
    Evidence of motor-perceptual interactions. Journal of Experimental
    Psychology: Human Perception and Performance, 18, 603--623.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


__all__ = [
    "PowerLawTrial",
    "PowerLawResult",
    "PowerLawValidator",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class PowerLawTrial:
    """The trajectory captured during a single goal-reaching trial.

    Attributes
    ----------
    distance : float
        Goal distance D used for this trial, in metres.
    angle_rad : float
        Goal angle relative to the start position, in radians.
    success : bool
        Whether the end-effector reached the goal tolerance.
    n_steps : int
        Number of environment steps recorded.
    positions : List[List[float]]
        Per-step end-effector position [x, y]. Length n_steps.
    """

    distance: float
    angle_rad: float
    success: bool
    n_steps: int
    positions: List[List[float]]


@dataclass
class PowerLawResult:
    """Aggregated outcome of a 2/3 Power Law validation sweep.

    Attributes
    ----------
    trials : List[PowerLawTrial]
        Every trial result, including failures.
    log_curvature : List[float]
        log10(C) values used in the regression after filtering.
    log_velocity : List[float]
        log10(V) values used in the regression after filtering.
    slope : float
        OLS slope of log V vs log C. The 2/3 Power Law predicts -1/3.
    intercept : float
        OLS intercept (= log K).
    r_pearson : float
        Pearson correlation coefficient between log V and log C. Fischer
        et al. (2021) reported R = 0.84 for their trained policy.
    r_squared : float
        Coefficient of determination of the linear fit.
    n_samples_used : int
        Number of (V, C) samples retained after threshold filtering.
    n_samples_dropped : int
        Number of samples discarded by the velocity or curvature filter.
    velocity_threshold : float
        Minimum |V| (m/s) required for a sample to be retained.
    curvature_threshold : float
        Minimum |C| (1/m) required for a sample to be retained.
    n_trials : int
        Total trials run.
    n_successful_trials : int
        Trials that reached the goal tolerance.
    timestamp_iso : str
        ISO-format timestamp at which the validator finished.
    metadata : Dict[str, Any]
        Free-form metadata (model identifier, total step count, seed,
        etc.). Set by the caller via PowerLawValidator.run.
    """

    trials: List[PowerLawTrial]
    log_curvature: List[float]
    log_velocity: List[float]
    slope: float
    intercept: float
    r_pearson: float
    r_squared: float
    n_samples_used: int
    n_samples_dropped: int
    velocity_threshold: float
    curvature_threshold: float
    n_trials: int
    n_successful_trials: int
    timestamp_iso: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation suitable for json.dump."""
        return {
            "trials": [asdict(t) for t in self.trials],
            "log_curvature": list(self.log_curvature),
            "log_velocity": list(self.log_velocity),
            "slope": float(self.slope),
            "intercept": float(self.intercept),
            "r_pearson": float(self.r_pearson),
            "r_squared": float(self.r_squared),
            "n_samples_used": int(self.n_samples_used),
            "n_samples_dropped": int(self.n_samples_dropped),
            "velocity_threshold": float(self.velocity_threshold),
            "curvature_threshold": float(self.curvature_threshold),
            "n_trials": int(self.n_trials),
            "n_successful_trials": int(self.n_successful_trials),
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
        """One-line human-readable summary of the regression."""
        return (
            f"log V = {self.intercept:+.4f} + {self.slope:+.4f} * log C, "
            f"R = {self.r_pearson:.4f}, R^2 = {self.r_squared:.4f} "
            f"(n_samples = {self.n_samples_used}, "
            f"n_trials = {self.n_successful_trials}/{self.n_trials})"
        )

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #
    def plot(
        self,
        save_path=None,
        show: bool = False,
        title: Optional[str] = None,
    ) -> None:
        """Produce the standard log V vs log C scatter + regression plot.

        Thread-safe path. With ``show=False`` (the default and the value
        used by the training-GUI worker thread), the routine uses
        ``matplotlib.figure.Figure`` plus ``FigureCanvasAgg`` directly and
        never touches ``pyplot``; the previous implementation crashed
        macOS Tk by calling ``plt.subplots()`` from a non-main thread.
        """
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for PowerLawResult.plot; "
                "install it via `pip install matplotlib`"
            ) from exc

        log_c = np.asarray(self.log_curvature, dtype=float)
        log_v = np.asarray(self.log_velocity, dtype=float)

        fig = Figure(figsize=(7, 5))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        ax.scatter(
            log_c, log_v, s=8, c="black", alpha=0.4,
            label=f"Per-step samples (n = {len(log_c)})",
        )

        if len(log_c) >= 2:
            x_line = np.linspace(log_c.min(), log_c.max(), 100)
            y_line = self.intercept + self.slope * x_line
            ax.plot(
                x_line, y_line, color="black", linewidth=1.4,
                label=(
                    f"Fit: log V = {self.intercept:.3f} {self.slope:+.3f} * log C   "
                    f"(R = {self.r_pearson:.3f}, R$^2$ = {self.r_squared:.3f})"
                ),
            )
            # Reference -1/3 slope through the data centroid
            cx = float(log_c.mean())
            cy = float(log_v.mean())
            y_ref = cy + (-1.0 / 3.0) * (x_line - cx)
            ax.plot(
                x_line, y_ref, color="gray", linewidth=1.0, linestyle="--",
                label="Reference: slope = -1/3 (2/3 Power Law)",
            )

        ax.set_xlabel("log$_{10}$ C   (curvature, 1/m)")
        ax.set_ylabel("log$_{10}$ V   (tangential velocity, m/s)")
        ax.set_title(title or "Two-thirds Power Law validation")
        ax.legend(loc="best", frameon=False, fontsize=9)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200)

        if show:
            import matplotlib.pyplot as plt
            plt.show()


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------
class PowerLawValidator:
    """Drive trials, log trajectories, fit log V against log C.

    Parameters
    ----------
    model : object with .predict(observation, deterministic=True) -> (action, state)
        Trained SB3 policy or any object exposing the same interface.
    env : ArmTaskEnv
        The 2-DOF environment. Must expose set_goal_position,
        set_goal_tolerance, dt, and the standard reset/step API.
    deterministic : bool, default True
        Action-selection mode for evaluation.
    rng : numpy.random.Generator, optional
        RNG for goal placement.
    """

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
                "env must support set_goal_position and set_goal_tolerance"
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
        n_trials: int = 30,
        max_steps_per_trial: int = 1000,
        distance_range: Tuple[float, float] = (0.30, 1.20),
        goal_tolerance: float = 0.05,
        velocity_threshold: float = 0.01,
        curvature_threshold: float = 0.10,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, "PowerLawTrial"], None]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PowerLawResult:
        """Run the sweep and return a PowerLawResult.

        Parameters
        ----------
        n_trials : int, default 30
            Number of independent goal-reaching trials.
        max_steps_per_trial : int, default 1000
            Per-trial step cap.
        distance_range : (low, high), default (0.30, 1.20)
            Goal distances are sampled uniformly from this metre range.
        goal_tolerance : float, default 0.05
            Position tolerance W (metres). Used only as the trial-success
            criterion; not part of the V/C analysis.
        velocity_threshold : float, default 0.01
            Per-step samples with |V| below this value are excluded from
            the regression (V near zero gives meaningless log V).
        curvature_threshold : float, default 0.10
            Per-step samples with |C| below this value are excluded from
            the regression (C near zero gives meaningless log C and
            corresponds to nearly-straight motion where the law does not
            apply).
        seed : int, optional
            Seed for the goal-placement RNG.
        progress_callback : callable, optional
            Invoked once per trial as
            ``progress_callback(trial_index, total_trials, trial)``.
        metadata : dict, optional
            Stored in PowerLawResult.metadata.
        """
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}")
        if max_steps_per_trial < 4:
            # Need at least 4 steps for one valid central-difference sample
            raise ValueError(
                f"max_steps_per_trial must be >= 4, got {max_steps_per_trial}"
            )
        d_lo, d_hi = float(distance_range[0]), float(distance_range[1])
        if not (0.0 < d_lo <= d_hi):
            raise ValueError(
                f"distance_range must satisfy 0 < low <= high, got {distance_range}"
            )
        if goal_tolerance <= 0.0:
            raise ValueError(f"goal_tolerance must be positive, got {goal_tolerance}")
        if velocity_threshold < 0.0:
            raise ValueError(
                f"velocity_threshold must be non-negative, got {velocity_threshold}"
            )
        if curvature_threshold < 0.0:
            raise ValueError(
                f"curvature_threshold must be non-negative, got {curvature_threshold}"
            )

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        trials: List[PowerLawTrial] = []
        n_success = 0

        for trial_index in range(n_trials):
            distance = float(self.rng.uniform(d_lo, d_hi))
            angle = float(self.rng.uniform(-math.pi / 2.0, math.pi / 2.0))
            trial = self._run_single_trial(
                distance=distance,
                angle=angle,
                tolerance=goal_tolerance,
                max_steps=max_steps_per_trial,
            )
            trials.append(trial)
            if trial.success:
                n_success += 1
            if progress_callback is not None:
                try:
                    progress_callback(trial_index, n_trials, trial)
                except Exception:
                    pass

        # Concatenate (V, C) samples across all trials, with trial-internal
        # central differences computed in isolation so a step boundary does
        # not pollute another trial's derivative estimates.
        log_c_samples: List[float] = []
        log_v_samples: List[float] = []
        n_dropped = 0

        for trial in trials:
            if trial.n_steps < 4:
                continue
            log_c, log_v, n_drop = self._extract_log_v_log_c(
                positions=np.asarray(trial.positions, dtype=float),
                dt=float(self.env.dt),
                v_thresh=velocity_threshold,
                c_thresh=curvature_threshold,
            )
            log_c_samples.extend(log_c.tolist())
            log_v_samples.extend(log_v.tolist())
            n_dropped += int(n_drop)

        slope, intercept, r_p, r2, n_used = self._fit_regression(
            log_c=np.asarray(log_c_samples, dtype=float),
            log_v=np.asarray(log_v_samples, dtype=float),
        )

        return PowerLawResult(
            trials=trials,
            log_curvature=log_c_samples,
            log_velocity=log_v_samples,
            slope=float(slope),
            intercept=float(intercept),
            r_pearson=float(r_p),
            r_squared=float(r2),
            n_samples_used=int(n_used),
            n_samples_dropped=int(n_dropped),
            velocity_threshold=float(velocity_threshold),
            curvature_threshold=float(curvature_threshold),
            n_trials=int(n_trials),
            n_successful_trials=int(n_success),
            timestamp_iso=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            metadata=dict(metadata) if metadata else {},
        )

    # ------------------------------------------------------------------ #
    # Single-trial execution
    # ------------------------------------------------------------------ #
    def _run_single_trial(
        self,
        distance: float,
        angle: float,
        tolerance: float,
        max_steps: int,
    ) -> PowerLawTrial:
        """Roll out the policy for one trial, returning the trajectory."""
        obs, info = self.env.reset()
        start_pos = self._extract_end_effector(info)
        goal_pos = start_pos + distance * np.array(
            [math.cos(angle), math.sin(angle)], dtype=np.float32
        )
        self.env.set_goal_position(goal_pos)
        self.env.set_goal_tolerance(tolerance)

        positions: List[List[float]] = [list(map(float, start_pos))]
        success = False

        for _ in range(max_steps):
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, _r, terminated, truncated, info = self.env.step(action)
            ee = info.get("end_effector_position")
            if ee is None:
                ee = self._extract_end_effector(info)
            ee_arr = np.asarray(ee, dtype=float).reshape(-1)
            if ee_arr.shape == (2,):
                positions.append([float(ee_arr[0]), float(ee_arr[1])])
            elif ee_arr.shape == (3,):
                positions.append([float(ee_arr[0]), float(ee_arr[1])])
            else:
                positions.append([float("nan"), float("nan")])

            d_to_goal = float(np.linalg.norm(np.asarray(positions[-1]) - goal_pos))
            if d_to_goal <= tolerance:
                success = True
                break
            if terminated or truncated:
                break

        return PowerLawTrial(
            distance=float(distance),
            angle_rad=float(angle),
            success=bool(success),
            n_steps=len(positions),
            positions=positions,
        )

    # ------------------------------------------------------------------ #
    # Numerical V and C
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_log_v_log_c(
        positions: np.ndarray,
        dt: float,
        v_thresh: float,
        c_thresh: float,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Compute log10 V and log10 C at every interior frame.

        Uses central differences for velocity and acceleration so each
        frame index t in [1, n-2] yields one (V, C) sample. Curvature
        is computed as |x' y'' - y' x''| / (x'^2 + y'^2)^(3/2).

        Samples with V < v_thresh, C < c_thresh, or any non-finite
        derivative are dropped.

        Returns
        -------
        log_c : np.ndarray
            log10(C) values for retained samples.
        log_v : np.ndarray
            log10(V) values for retained samples.
        n_dropped : int
            Number of interior frames discarded by filtering.
        """
        if positions.shape[0] < 3:
            return np.empty(0), np.empty(0), 0

        # Central differences over interior frames.
        x = positions[:, 0]
        y = positions[:, 1]
        # First derivative
        vx = (x[2:] - x[:-2]) / (2.0 * dt)
        vy = (y[2:] - y[:-2]) / (2.0 * dt)
        # Second derivative
        ax = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / (dt * dt)
        ay = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / (dt * dt)

        v = np.sqrt(vx * vx + vy * vy)
        # Curvature magnitude
        cross = vx * ay - vy * ax
        denom = np.power(v, 3.0)
        # Avoid divide-by-zero; the velocity filter below catches these.
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(denom > 0.0, np.abs(cross) / denom, np.inf)

        finite = np.isfinite(v) & np.isfinite(c)
        keep = finite & (v >= v_thresh) & (c >= c_thresh)
        n_total = int(v.size)
        n_kept = int(keep.sum())
        n_dropped = n_total - n_kept

        if n_kept == 0:
            return np.empty(0), np.empty(0), n_dropped

        log_c = np.log10(c[keep])
        log_v = np.log10(v[keep])
        return log_c, log_v, n_dropped

    # ------------------------------------------------------------------ #
    # Regression
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fit_regression(
        log_c: np.ndarray,
        log_v: np.ndarray,
    ) -> Tuple[float, float, float, float, int]:
        """Closed-form OLS fit of log V = intercept + slope * log C.

        Returns (slope, intercept, r_pearson, r_squared, n_samples).
        Returns NaNs if fewer than two samples are supplied or if log C
        is constant.
        """
        n = int(min(len(log_c), len(log_v)))
        if n < 2:
            return float("nan"), float("nan"), float("nan"), float("nan"), n

        x = log_c[:n].astype(np.float64)
        y = log_v[:n].astype(np.float64)
        x_mean = x.mean()
        y_mean = y.mean()
        sxx = float(np.sum((x - x_mean) ** 2))
        if sxx <= 0.0:
            return float("nan"), float("nan"), float("nan"), float("nan"), n

        sxy = float(np.sum((x - x_mean) * (y - y_mean)))
        syy = float(np.sum((y - y_mean) ** 2))
        slope = sxy / sxx
        intercept = float(y_mean - slope * x_mean)

        if syy <= 0.0:
            r_pearson = float("nan")
            r2 = float("nan")
        else:
            r_pearson = sxy / math.sqrt(sxx * syy)
            y_pred = intercept + slope * x
            ss_res = float(np.sum((y - y_pred) ** 2))
            r2 = 1.0 - ss_res / syy

        return float(slope), float(intercept), float(r_pearson), float(r2), n

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_end_effector(info: Optional[Dict[str, Any]]) -> np.ndarray:
        """Pull a 2D end-effector position from an info dict."""
        if isinstance(info, dict):
            ee = info.get("end_effector_position")
            if ee is not None:
                arr = np.asarray(ee, dtype=np.float32).reshape(-1)
                if arr.shape == (2,):
                    return arr
                if arr.shape == (3,):
                    return arr[:2]
        return np.zeros(2, dtype=np.float32)


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
