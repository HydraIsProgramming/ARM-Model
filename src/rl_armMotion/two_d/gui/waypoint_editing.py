"""Placing and removing waypoints by clicking, shared by both GUIs.

The rules live here rather than in a window class so the demo interface and the
full controller behave identically, and so they can be tested without a display.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["nearest_placeable", "index_near", "REMOVE_RADIUS"]

# How close a click has to be to a placed point to count as "that one".
# Generous enough to hit with a mouse, tight enough that a click in open space
# is not read as a delete.
REMOVE_RADIUS = 0.12   # m


def nearest_placeable(env, wp: Sequence[float],
                      existing: Sequence[Sequence[float]]) -> Tuple[Optional[List[float]], float]:
    """The nearest point the arm can actually hold that also clears the spacing rule.

    Returns ``(point, distance_moved)``, or ``(None, 0.0)`` when nothing within
    reach of the click qualifies. A click that is already fine comes back
    unchanged with a distance of zero.
    """
    pts = env.feasible_points()
    target = np.asarray(wp, dtype=float)
    for idx in np.argsort(np.linalg.norm(pts - target, axis=1))[:400]:
        cand = [float(v) for v in pts[idx]]
        try:
            env._validate_waypoint_spacing(
                [np.asarray(w, dtype=float) for w in list(existing) + [cand]])
        except ValueError:
            continue
        return cand, float(np.linalg.norm(np.asarray(cand) - target))
    return None, 0.0


def index_near(points: Sequence[Sequence[float]], wp: Sequence[float],
               radius: float = REMOVE_RADIUS) -> Optional[int]:
    """Index of the placed point nearest ``wp``, if one is within ``radius``."""
    if not len(points):
        return None
    target = np.asarray(wp, dtype=float)
    d = [float(np.linalg.norm(np.asarray(p, dtype=float) - target)) for p in points]
    i = int(np.argmin(d))
    return i if d[i] <= radius else None
