"""Generate the system figure for the conference paper.

Produces docs/paper/figures/system.pdf -- a two-panel vector diagram:

  (a) the 2-DOF arm model: link structure, joint angles, and the antagonist
      muscle pairs that drive each joint;
  (b) the three-waypoint task layout with tolerance radii and traversal order.

The figure deliberately carries little numeric annotation: exact values live in
Tables I and II, and duplicating them here only invites drift. Geometry is read
from the live ArmConfiguration preset so the drawing cannot diverge from the
code.

Run from the project root:
    python docs/paper/make_system_figure.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rl_armMotion.two_d.config import ArmConfiguration

CFG = ArmConfiguration.get_preset("2dof_simple")
L0, L1 = float(CFG.link_lengths[0]), float(CFG.link_lengths[1])
SHOULDER = np.array([1.0, 0.0])
REACH = L0 + L1

WAYPOINTS = [(0.2, 1.2), (2.2, 0.5), (1.0, -1.6)]
TOL = 0.6

INK = "#1a1a1a"
GREY = "#9a9a9a"
MUSCLE_E = "#c0392b"
MUSCLE_F = "#2471a3"
ACCENT = "#117a4a"


def ik(target):
    d = np.asarray(target, float) - SHOULDER
    r = min(float(np.hypot(*d)), REACH - 1e-6)
    c1 = (r**2 - L0**2 - L1**2) / (2 * L0 * L1)
    th1 = float(np.arccos(np.clip(c1, -1.0, 1.0)))
    th0 = float(np.arctan2(d[1], d[0]) -
                np.arctan2(L1 * np.sin(th1), L0 + L1 * np.cos(th1)))
    return th0, th1


def fk(th0, th1):
    elbow = SHOULDER + L0 * np.array([np.cos(th0), np.sin(th0)])
    ee = elbow + L1 * np.array([np.cos(th0 + th1), np.sin(th0 + th1)])
    return elbow, ee


def short_arc(ax, centre, diam, a_deg, b_deg, color):
    lo, hi = sorted((a_deg, b_deg))
    ax.add_patch(Arc(centre, diam, diam, angle=0, theta1=lo, theta2=hi,
                     color=color, lw=1.4, zorder=7))


def draw_arm(ax, th0, th1, alpha=1.0, lw=3.0):
    elbow, ee = fk(th0, th1)
    ax.plot([SHOULDER[0], elbow[0]], [SHOULDER[1], elbow[1]], "-",
            color=INK, lw=lw, alpha=alpha, solid_capstyle="round", zorder=4)
    ax.plot([elbow[0], ee[0]], [elbow[1], ee[1]], "-",
            color=INK, lw=lw, alpha=alpha, solid_capstyle="round", zorder=4)
    ax.plot(*elbow, "o", color="white", mec=INK, mew=1.5, ms=8,
            alpha=alpha, zorder=5)
    ax.plot(*ee, "o", color=ACCENT, mec=INK, mew=1.0, ms=8, alpha=alpha,
            zorder=6)
    return elbow, ee


def muscle_pair(ax, prox_dir, joint, dist_pt, span, off):
    dist_dir = (dist_pt - joint) / (np.linalg.norm(dist_pt - joint) + 1e-9)
    bis = prox_dir + dist_dir
    n = bis / (np.linalg.norm(bis) + 1e-9)
    for sign, col in ((+1, MUSCLE_E), (-1, MUSCLE_F)):
        p = joint + prox_dir * span + n * sign * off
        q = joint + dist_dir * span + n * sign * off
        ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=2.2,
                solid_capstyle="round", alpha=0.95, zorder=3)


def style(ax, ylim):
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 3.3)
    ax.set_ylim(*ylim)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(labelsize=6.5, color=GREY, length=2.5)
    ax.set_xlabel("x (m)", fontsize=7.5, labelpad=1)
    ax.set_ylabel("y (m)", fontsize=7.5, labelpad=1)


fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.1, 3.45))
YLIM = (-2.55, 2.30)

# ---------------------------------------------------------------- panel (a)
axA.add_patch(Circle(SHOULDER, REACH, fill=False, ls=(0, (5, 4)),
                     lw=0.9, ec=GREY, zorder=1))
axA.text(SHOULDER[0], REACH + 0.10, "reachable workspace",
         ha="center", fontsize=6.6, color="#777777")

th0, th1 = ik((2.2, 0.5))
elbow, ee = draw_arm(axA, th0, th1)

# reference rays and joint-angle arcs
axA.plot([SHOULDER[0], SHOULDER[0] + 1.15], [SHOULDER[1], SHOULDER[1]],
         ls=(0, (2, 2)), lw=0.8, color=GREY, zorder=2)
short_arc(axA, SHOULDER, 1.30, 0.0, np.degrees(th0), ACCENT)
axA.text(SHOULDER[0] + 0.72, -0.42, r"$\theta_0$", fontsize=9.5, color=ACCENT)

ext = elbow + (elbow - SHOULDER) / np.linalg.norm(elbow - SHOULDER) * 0.70
axA.plot([elbow[0], ext[0]], [elbow[1], ext[1]],
         ls=(0, (2, 2)), lw=0.8, color=GREY, zorder=2)
short_arc(axA, elbow, 0.90, np.degrees(th0), np.degrees(th0 + th1), ACCENT)
axA.text(elbow[0] + 0.52, elbow[1] + 0.06, r"$\theta_1$",
         fontsize=9.5, color=ACCENT)

# antagonist muscle pairs
muscle_pair(axA, np.array([-1.0, 0.0]), SHOULDER, elbow, 0.36, 0.145)
u_prox = (SHOULDER - elbow) / np.linalg.norm(SHOULDER - elbow)
muscle_pair(axA, u_prox, elbow, ee, 0.32, 0.125)

axA.plot(*SHOULDER, "s", color=INK, ms=8, zorder=8)
axA.text(SHOULDER[0] - 0.22, 0.42, "shoulder\n(fixed)", fontsize=6.6,
         ha="center", va="center")
axA.text(elbow[0] + 0.02, elbow[1] - 0.42, "elbow", fontsize=6.6,
         ha="center", va="center")
axA.text(ee[0] + 0.16, ee[1] + 0.30, "end effector", fontsize=6.6,
         color=ACCENT, ha="center", va="center")

axA.plot([], [], color=MUSCLE_E, lw=2.2, label="extensor")
axA.plot([], [], color=MUSCLE_F, lw=2.2, label="flexor")
axA.legend(loc="lower left", fontsize=6.4, frameon=False,
           handlelength=1.5, borderaxespad=0.4)
style(axA, YLIM)
axA.set_title("(a)  2-DOF arm, antagonist muscle actuation",
              fontsize=8.2, pad=5)

# ---------------------------------------------------------------- panel (b)
axB.add_patch(Circle(SHOULDER, REACH, fill=False, ls=(0, (5, 4)),
                     lw=0.9, ec=GREY, zorder=1))
axB.plot(*SHOULDER, "s", color=INK, ms=8, zorder=8)
axB.text(SHOULDER[0] + 0.15, -0.05, "shoulder", fontsize=6.6,
         ha="left", va="center")

# hand-placed label anchors, chosen to clear every tolerance disc
LABELS = {
    1: (-0.62, 1.86, "center"),
    2: (2.62, -0.42, "center"),
    3: (1.00, -2.40, "center"),
}
for i, (wx, wy) in enumerate(WAYPOINTS, start=1):
    axB.add_patch(Circle((wx, wy), TOL, fc=ACCENT, alpha=0.11,
                         ec=ACCENT, lw=0.9, ls=(0, (3, 3)), zorder=2))
    axB.plot(wx, wy, "*", color=ACCENT, ms=17, mec=INK, mew=0.6, zorder=6)
    lx, ly, ha = LABELS[i]
    axB.text(lx, ly, f"$W_{i}$ ({wx}, {wy})", fontsize=6.8, ha=ha,
             va="center", zorder=7)

for a, b in ((0, 1), (1, 2)):
    p, q = np.array(WAYPOINTS[a]), np.array(WAYPOINTS[b])
    d = (q - p) / np.linalg.norm(q - p)
    axB.add_patch(FancyArrowPatch(tuple(p + d * TOL * 0.95),
                                  tuple(q - d * TOL * 0.95),
                                  arrowstyle="-|>", mutation_scale=10,
                                  lw=1.2, color=INK, alpha=0.65,
                                  connectionstyle="arc3,rad=0.15", zorder=5))

axB.annotate("tolerance radius", xy=(WAYPOINTS[0][0] - TOL * 0.70,
                                     WAYPOINTS[0][1] - TOL * 0.70),
             xytext=(-0.72, 0.44), textcoords="data",
             fontsize=6.4, color=ACCENT, ha="center",
             arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT))
style(axB, YLIM)
axB.set_title("(b)  Three-waypoint task", fontsize=8.2, pad=5)

fig.tight_layout(pad=0.5, w_pad=1.1)
out = Path(__file__).resolve().parent / "figures" / "system.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
print(f"Wrote {out}")
print(f"  theta0 = {np.degrees(th0):+.1f} deg, theta1 = {np.degrees(th1):+.1f} deg")
for i, (wx, wy) in enumerate(WAYPOINTS, start=1):
    print(f"  W{i} ({wx}, {wy})  r = "
          f"{np.hypot(wx-SHOULDER[0], wy-SHOULDER[1]):.3f} m")
