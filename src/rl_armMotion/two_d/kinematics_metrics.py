"""Human-likeness metrics for hand trajectories.

Reference ranges for point-to-point reaching in healthy adults (Morasso 1981;
Flash & Hogan 1985; Abend, Bizzi & Morasso 1982):

- path straightness = path length / straight-line distance     ~ 1.05 - 1.15
- velocity-profile symmetry = time-to-peak / movement time     ~ 0.40 - 0.50
- submovements = number of prominent speed peaks               1 (2 at most)
- dimensionless jerk = sqrt(0.5 * int j^2 dt * T^5 / L^2)      lower is smoother;
  minimum-jerk straight reaches score ~ 20-30

``segment_metrics`` scores one movement (from one waypoint activation to the
next); ``episode_metrics`` splits an evaluation trajectory at the waypoint
switches and averages the segment scores.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["segment_metrics", "episode_metrics", "HUMAN_REFERENCE"]

HUMAN_REFERENCE = {
    "straightness": (1.0, 1.15),
    "symmetry": (0.35, 0.55),
    "submovements": (1, 2),
    "dimensionless_jerk": (0.0, 60.0),
}


def _speed(positions: np.ndarray, dt: float) -> np.ndarray:
    v = np.gradient(positions, dt, axis=0)
    return np.linalg.norm(v, axis=1)


def segment_metrics(positions: np.ndarray, dt: float, speed_floor: float = 0.05) -> Dict[str, float]:
    """Metrics for one movement given hand positions (T, 2) sampled at dt."""
    pos = np.asarray(positions, dtype=float)
    if pos.shape[0] < 4:
        return {"straightness": float("nan"), "symmetry": float("nan"),
                "submovements": float("nan"), "dimensionless_jerk": float("nan"),
                "movement_time": float(pos.shape[0] * dt), "peak_speed": float("nan")}
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    path_len = float(np.sum(seg))
    chord = float(np.linalg.norm(pos[-1] - pos[0]))
    straightness = path_len / chord if chord > 1e-6 else float("nan")

    speed = _speed(pos, dt)
    peak = float(np.max(speed))
    # Movement time: from first to last sample above a fraction of the peak.
    thr = max(speed_floor, 0.1 * peak)
    moving = np.where(speed > thr)[0]
    if moving.size >= 2:
        t_start, t_end = int(moving[0]), int(moving[-1])
    else:
        t_start, t_end = 0, pos.shape[0] - 1
    mt = max((t_end - t_start) * dt, dt)
    t_peak = int(np.argmax(speed))
    symmetry = float(np.clip((t_peak - t_start) * dt / mt, 0.0, 1.0))

    # Count prominent peaks in the speed profile inside the movement window.
    s = speed[t_start:t_end + 1]
    peaks = 0
    for i in range(1, s.size - 1):
        if s[i] > s[i - 1] and s[i] >= s[i + 1] and s[i] > 0.2 * peak:
            peaks += 1
    peaks = max(peaks, 1)

    # Dimensionless jerk (Hogan & Sternad 2009).
    seg_pos = pos[t_start:t_end + 1]
    if seg_pos.shape[0] >= 5:
        jerk = np.gradient(np.gradient(np.gradient(seg_pos, dt, axis=0), dt, axis=0), dt, axis=0)
        integ = float(np.sum(np.sum(jerk ** 2, axis=1)) * dt)
        L = float(np.linalg.norm(seg_pos[-1] - seg_pos[0]))
        dj = float(np.sqrt(0.5 * integ * mt ** 5 / L ** 2)) if L > 1e-6 else float("nan")
    else:
        dj = float("nan")
    return {
        "straightness": float(straightness),
        "symmetry": symmetry,
        "submovements": float(peaks),
        "dimensionless_jerk": dj,
        "movement_time": float(mt),
        "peak_speed": peak,
    }


def episode_metrics(positions: np.ndarray, waypoint_index: np.ndarray, dt: float,
                    success_step: Optional[int] = None) -> Dict[str, Any]:
    """Average segment metrics over the movements of one episode.

    ``waypoint_index[t]`` is the active waypoint after step t; a segment runs
    from the step a waypoint becomes active until it is completed (or until
    ``success_step`` for the last one).
    """
    pos = np.asarray(positions, dtype=float)
    idx = np.asarray(waypoint_index, dtype=int)
    n = pos.shape[0]
    bounds: List[tuple] = []
    start = 0
    for t in range(1, n):
        if idx[t] != idx[t - 1]:
            bounds.append((start, t))
            start = t
    end = n - 1 if success_step is None else min(int(success_step), n - 1)
    if end > start:
        bounds.append((start, end))
    segs = [segment_metrics(pos[a:b + 1], dt) for a, b in bounds if b - a >= 4]
    keys = ["straightness", "symmetry", "submovements", "dimensionless_jerk", "movement_time", "peak_speed"]
    out: Dict[str, Any] = {"n_segments": len(segs), "segments": segs}
    for k in keys:
        vals = [s[k] for s in segs if np.isfinite(s[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["within_human_range"] = {
        k: (bool(lo <= out[k] <= hi) if np.isfinite(out[k]) else None)
        for k, (lo, hi) in HUMAN_REFERENCE.items()
    }
    return out
