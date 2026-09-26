"""
Explanatory diagrams for the technical report (not data charts - those are in
plot_training.py). Each diagram shows the real mechanism, drawn from the code's
own numbers where it has them: the arm schematic is computed from the preset
and the feasible-target grid, not sketched.

    python scripts/report_diagrams.py            # -> project_assets/report/build/diagrams/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Wedge  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "project_assets" / "report" / "build" / "diagrams"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
LINE = "#8d8c87"
BLUE = "#2a78d6"          # the agent / policy
ORANGE = "#eb6834"        # the environment / physics
GREEN = "#1baf7a"         # reward, feasible region
PALE_BLUE = "#e3eefb"
PALE_ORANGE = "#fde9e0"
PALE_GREEN = "#dff3ea"
PALE_GREY = "#f0efec"


def _style():
    plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "font.family": "sans-serif", "font.size": 10, "text.color": INK})


def box(ax, x, y, w, h, title, lines=(), face=PALE_GREY, edge=LINE, title_size=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                facecolor=face, edgecolor=edge, linewidth=1.2))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=INK)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.085 - i * 0.043, t, ha="center", va="top", fontsize=8.6,
                color=INK_2)


def arrow(ax, p, q, label=None, label_xy=None, color=INK_2, rad=0.0, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, linewidth=1.4,
                                 color=color, connectionstyle=f"arc3,rad={rad}"))
    if label:
        lx, ly = label_xy if label_xy else ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        ax.text(lx, ly, label, ha=ha, va="center", fontsize=8.6, color=INK,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none"))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


# --------------------------------------------------------------------- 1. loop

def training_loop():
    """One control step, 20 ms: what the policy sees, what it outputs, what
    happens between its output and the next observation."""
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1)
    ax.axis("off")

    box(ax, 0.03, 0.55, 0.27, 0.38, "SAC policy  (the agent)",
        ["neural network",
         "18 -> 256 -> 256 -> 4",
         "stochastic while training,",
         "deterministic when deployed"], face=PALE_BLUE, edge=BLUE)

    box(ax, 0.40, 0.66, 0.24, 0.27, "Muscles", ["4 Hill-type muscles, 2 per joint",
                                                "activation dynamics 40/60 ms",
                                                "force-length, force-velocity"],
        face=PALE_ORANGE, edge=ORANGE)
    box(ax, 0.72, 0.66, 0.25, 0.27, "Arm physics", ["M(q)q'' + C q' + G(q) = tau",
                                                    "gravity, coupled inertia",
                                                    "RK2, 4 sub-steps of 5 ms"],
        face=PALE_ORANGE, edge=ORANGE)
    box(ax, 0.72, 0.20, 0.25, 0.30, "Task", ["targets, 5 cm tolerance",
                                             "0.5 s hold at the last one",
                                             "range of motion, torque limits",
                                             "10 s episodes"],
        face=PALE_ORANGE, edge=ORANGE)
    box(ax, 0.36, 0.06, 0.28, 0.25, "Reward", ["10 terms: progress, distance,",
                                               "effort, smoothness, joint limits,",
                                               "speed, acceleration, bonuses"],
        face=PALE_GREEN, edge=GREEN)
    box(ax, 0.03, 0.06, 0.27, 0.34, "Replay buffer", ["300,000 transitions",
                                                      "(s, a, r, s') sampled in",
                                                      "batches of 256 to update",
                                                      "the actor and twin critics"],
        face=PALE_GREY, edge=LINE)

    ax.add_patch(FancyBboxPatch((0.345, 0.03), 0.64, 0.925, boxstyle="round,pad=0.01,rounding_size=0.02",
                                facecolor="none", edgecolor=ORANGE, linewidth=1.0, linestyle=(0, (4, 3))))
    ax.text(0.985, 0.012, "environment (simulated arm + task)", ha="right", va="top", fontsize=9,
            color=ORANGE, fontweight="bold")

    arrow(ax, (0.30, 0.80), (0.40, 0.80), "action a", (0.35, 0.86))
    arrow(ax, (0.64, 0.80), (0.72, 0.80), "torque", (0.68, 0.845))
    arrow(ax, (0.845, 0.66), (0.845, 0.50), "new state", (0.905, 0.585))
    arrow(ax, (0.72, 0.30), (0.64, 0.22), "outcome", (0.70, 0.23), ha="left")
    arrow(ax, (0.72, 0.45), (0.30, 0.62), "observation s'  (18 numbers)", (0.52, 0.52))
    arrow(ax, (0.36, 0.17), (0.30, 0.17), "r", (0.33, 0.20))
    arrow(ax, (0.165, 0.40), (0.165, 0.55), "learn", (0.21, 0.475))

    ax.text(0.5, 1.02, "One control step (20 ms): the policy sees 18 numbers and outputs 4 muscle activations",
            ha="center", fontsize=12, fontweight="bold", color=INK)
    save(fig, "d1_training_loop")


# ------------------------------------------------------------------ 2. SAC

def sac_networks():
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def column(x, y0, n, label, colour, h=0.46):
        ax.add_patch(FancyBboxPatch((x, y0), 0.075, h, boxstyle="round,pad=0.005,rounding_size=0.012",
                                    facecolor=colour, edgecolor=LINE, linewidth=1.0))
        ax.text(x + 0.0375, y0 + h / 2, str(n), ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x + 0.0375, y0 - 0.045, label, ha="center", va="top", fontsize=8.2, color=INK_2)

    # actor
    ax.text(0.22, 0.95, "Actor  pi(a | s)", ha="center", fontsize=11.5, fontweight="bold", color=BLUE)
    y = 0.30
    column(0.02, y, 18, "state s", PALE_GREY)
    column(0.13, y, 256, "ReLU", PALE_BLUE)
    column(0.24, y, 256, "ReLU", PALE_BLUE)
    ax.add_patch(FancyBboxPatch((0.35, y + 0.25), 0.075, 0.21, boxstyle="round,pad=0.005,rounding_size=0.012",
                                facecolor=PALE_BLUE, edgecolor=LINE))
    ax.text(0.3875, y + 0.355, "4\nmean", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.35, y), 0.075, 0.21, boxstyle="round,pad=0.005,rounding_size=0.012",
                                facecolor=PALE_BLUE, edgecolor=LINE))
    ax.text(0.3875, y + 0.105, "4\nlog std", ha="center", va="center", fontsize=9, fontweight="bold")
    for x0, x1 in ((0.095, 0.13), (0.205, 0.24), (0.315, 0.35)):
        arrow(ax, (x0, y + 0.23), (x1, y + 0.23))
    ax.text(0.22, 0.13, "sample a = tanh(mean + std * noise), rescaled to [0, 1]",
            ha="center", fontsize=8.6, color=INK_2)
    ax.text(0.22, 0.06, "4 muscle activations; the noise is the exploration",
            ha="center", fontsize=8.6, color=INK_2)

    # critics
    ax.text(0.74, 0.95, "Twin critics  Q1(s, a), Q2(s, a)", ha="center", fontsize=11.5,
            fontweight="bold", color=ORANGE)
    for k, yy in enumerate((0.60, 0.20)):
        column(0.54, yy, 22, "s and a", PALE_GREY, h=0.25)
        column(0.65, yy, 256, "ReLU", PALE_ORANGE, h=0.25)
        column(0.76, yy, 256, "ReLU", PALE_ORANGE, h=0.25)
        column(0.87, yy, 1, f"Q{k + 1}", PALE_ORANGE, h=0.25)
        for x0, x1 in ((0.615, 0.65), (0.725, 0.76), (0.835, 0.87)):
            arrow(ax, (x0, yy + 0.125), (x1, yy + 0.125))
    ax.text(0.74, 0.06, "the smaller of Q1, Q2 is used, which curbs over-estimation; each critic has a",
            ha="center", fontsize=8.6, color=INK_2)
    ax.text(0.74, 0.01, "slowly updated target copy (tau = 0.005)", ha="center", fontsize=8.6, color=INK_2)
    ax.plot([0.485, 0.485], [0.02, 0.93], color="#e4e3df", linewidth=1)
    save(fig, "d2_sac_networks")


# ------------------------------------------------------------------ 3. arm

def arm_schematic():
    from rl_armMotion.two_d.config import ArmConfiguration
    from rl_armMotion.two_d.evaluation import make_env

    cfg = ArmConfiguration.get_preset("2dof_human_right")
    env = make_env(actuation_mode="muscle", waypoints=None)
    S = np.asarray(env.shoulder_base_position, float)
    L1, L2 = cfg.link_lengths
    reach = L1 + L2
    pts = env.feasible_points() - S

    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    ax.set_aspect("equal")
    ax.set_xlim(-0.55, 0.78)
    ax.set_ylim(-0.86, 0.74)
    ax.axis("off")

    # body silhouette
    ax.add_patch(plt.Rectangle((-0.16, -0.62), 0.16, 0.62, color="#00000010"))
    ax.add_patch(plt.Circle((-0.08, 0.22), 0.11, color="#00000010"))
    ax.text(-0.08, -0.42, "torso", ha="center", fontsize=8.5, color=INK_2)

    ax.add_patch(plt.Circle((0, 0), reach, fill=False, linestyle=":", color=LINE, linewidth=1.1))
    ax.text(reach * np.cos(np.radians(55)) + 0.02, reach * np.sin(np.radians(55)) + 0.03,
            f"full reach {reach:.2f} m", fontsize=8.5, color=INK_2, ha="left")
    ax.scatter(pts[:, 0], pts[:, 1], s=5, color=GREEN, alpha=0.30, linewidths=0, zorder=1)
    ax.text(0.60, -0.66, f"green = the {len(pts)} targets\nthe arm can reach AND hold",
            fontsize=8.5, color=INK, ha="center")
    ax.text(-0.54, -0.84, "The model has no body collision: targets behind and above the head\n"
                          "count as reachable because nothing in the physics prevents it.",
            fontsize=7.8, color=INK_2, ha="left", va="bottom")

    # shoulder range of motion: theta0 in [-150, 90] deg
    lo, hi = np.degrees(cfg.joint_limits_min[0]), np.degrees(cfg.joint_limits_max[0])
    ax.add_patch(Wedge((0, 0), 0.09, lo, hi, width=0.018, color=BLUE, alpha=0.55))
    ax.text(-0.10, 0.03, f"shoulder\n{lo:.0f} to +{hi:.0f} deg", fontsize=8.3, color=BLUE,
            ha="right", va="bottom")

    # a representative pose
    q = np.radians([-35.0, 95.0])
    elbow = np.array([L1 * np.cos(q[0]), L1 * np.sin(q[0])])
    hand = elbow + np.array([L2 * np.cos(q[0] + q[1]), L2 * np.sin(q[0] + q[1])])
    ax.plot([0, elbow[0], hand[0]], [0, elbow[1], hand[1]], "-", color=INK, linewidth=5,
            solid_capstyle="round", zorder=4)
    for p in (np.zeros(2), elbow):
        ax.plot(*p, "o", color="white", markersize=10, markeredgecolor=INK, markeredgewidth=2, zorder=5)
    ax.plot(*hand, "o", color=ORANGE, markersize=10, zorder=5)

    # elbow range 0..145 deg, measured from straight
    base = np.degrees(q[0])
    ax.add_patch(Wedge(tuple(elbow), 0.07, base, base + np.degrees(cfg.joint_limits_max[1]),
                       width=0.016, color=ORANGE, alpha=0.6, zorder=3))
    ax.text(elbow[0] + 0.07, elbow[1] - 0.08, "elbow\n0 to 145 deg", fontsize=8.3, color=ORANGE)

    # labels sit off each link, on the side away from the other link
    ax.text(elbow[0] / 2 - 0.10, elbow[1] / 2 - 0.10,
            f"upper arm\n{L1:.2f} m, {cfg.masses[0]:.2f} kg", fontsize=8.3, color=INK, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.9))
    mid = (elbow + hand) / 2
    ax.text(mid[0] + 0.07, mid[1] - 0.02, f"forearm + hand\n{L2:.2f} m, {cfg.masses[1]:.2f} kg",
            fontsize=8.3, color=INK, ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.9))
    ax.text(hand[0] + 0.03, hand[1] + 0.02, "hand", fontsize=8.3, color=ORANGE)
    ax.annotate("", (-0.40, -0.40), (-0.40, -0.24), arrowprops=dict(arrowstyle="-|>", color=INK_2))
    ax.text(-0.39, -0.20, r"gravity 9.81 m/s$^2$", fontsize=8.2, color=INK_2, ha="center")
    ax.text(0.78, -0.30, "in front of\nthe body  ->", fontsize=8.5, color=INK_2, ha="right", va="center")
    ax.set_title("The arm: a human right arm seen from the side (Winter 2009, 75 kg / 1.75 m adult)",
                 fontsize=11.5, fontweight="bold", loc="left")
    save(fig, "d3_arm_and_workspace")


# ----------------------------------------------------------- 4. curriculum

def curriculum_ladder():
    fig, ax = plt.subplots(figsize=(10.2, 3.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rungs = [("15 cm", "hold 10"), ("10 cm", "hold 10"), ("7 cm", "hold 10"),
             ("5 cm", "hold 10"), ("5 cm", "hold 15"), ("5 cm", "hold 25")]
    w, gap = 0.125, 0.035
    for i, (tol, hold) in enumerate(rungs):
        x = 0.02 + i * (w + gap)
        final = i == len(rungs) - 1
        face = PALE_GREEN if final else (PALE_BLUE if i < 4 else PALE_ORANGE)
        edge = GREEN if final else (BLUE if i < 4 else ORANGE)
        ax.add_patch(FancyBboxPatch((x, 0.38), w, 0.34, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    facecolor=face, edgecolor=edge, linewidth=1.3))
        ax.text(x + w / 2, 0.61, tol, ha="center", fontsize=12, fontweight="bold")
        ax.text(x + w / 2, 0.47, hold, ha="center", fontsize=9.5, color=INK_2)
        if i < len(rungs) - 1:
            arrow(ax, (x + w + 0.004, 0.55), (x + w + gap - 0.004, 0.55))
    ax.text(0.02 + 1.5 * (w + gap) + w / 2, 0.83, "tighten the position tolerance first",
            ha="center", fontsize=9.5, color=BLUE)
    ax.text(0.02 + 4.5 * (w + gap) + w / 2, 0.83, "then lengthen the hold",
            ha="center", fontsize=9.5, color=ORANGE)
    ax.text(0.02 + 5 * (w + gap) + w / 2, 0.33, "= the production standard", ha="center",
            fontsize=8.8, color=GREEN, va="top")
    ax.text(0.02, 0.17, "Every 20,000 steps the policy is tested at its current rung (5 episodes). Two rounds in a row "
                        "at 80% or better move it up one rung,", fontsize=8.6, color=INK_2)
    ax.text(0.02, 0.08, "then it waits one round. Hold = number of 20 ms steps the hand must stay inside the "
                        "tolerance (25 steps = 0.5 s).", fontsize=8.6, color=INK_2)
    ax.text(0.02, 0.97, "The curriculum: training starts easy and tightens only when the policy has earned it",
            fontsize=11.5, fontweight="bold", va="top")
    save(fig, "d4_curriculum")


# ------------------------------------------------------------ 5. history

def research_path():
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("A", "2 fixed points", "64%", PALE_GREY, LINE),
        ("B", "3 fixed points", "40%", PALE_GREY, LINE),
        ("B3", "better target\ngeometry", "100%", PALE_GREY, LINE),
        ("R3", "3 RANDOM points\n(deployed)", "92%", PALE_BLUE, BLUE),
    ]
    w, gap = 0.19, 0.055
    for i, (name, what, res, face, edge) in enumerate(steps):
        x = 0.02 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.50), w, 0.34, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    facecolor=face, edgecolor=edge, linewidth=1.4))
        ax.text(x + w / 2, 0.78, name, ha="center", fontsize=13, fontweight="bold")
        ax.text(x + w / 2, 0.665, what, ha="center", va="center", fontsize=8.8, color=INK_2)
        ax.text(x + w / 2, 0.545, f"success {res}", ha="center", fontsize=9, fontweight="bold")
        if i < len(steps) - 1:
            arrow(ax, (x + w + 0.005, 0.67), (x + w + gap - 0.005, 0.67))
    # branches from R3
    x3 = 0.02 + 3 * (w + gap)
    branches = [("R4  end-range rule", "97%", 0.35), ("R5  speed cost", "59%", 0.225),
                ("R6  binding speed cost", "78%", 0.10)]
    for label, res, yy in branches:
        ax.add_patch(FancyBboxPatch((0.44, yy), 0.27, 0.10, boxstyle="round,pad=0.008,rounding_size=0.015",
                                    facecolor=PALE_ORANGE, edgecolor=ORANGE, linewidth=1.1))
        ax.text(0.455, yy + 0.05, label, va="center", fontsize=8.8)
        ax.text(0.70, yy + 0.05, res, va="center", ha="right", fontsize=8.8, fontweight="bold")
        arrow(ax, (x3 + w / 2, 0.50), (0.71, yy + 0.05), rad=-0.15)
    ax.add_patch(FancyBboxPatch((0.02, 0.10), 0.37, 0.22, boxstyle="round,pad=0.008,rounding_size=0.015",
                                facecolor=PALE_GREY, edgecolor=LINE, linewidth=1.0))
    ax.text(0.035, 0.28, "Free order: the policy picks the route", fontsize=8.8,
            fontweight="bold", va="top")
    ax.text(0.035, 0.215, "F3  3 points: stalled twice at ~400K steps", fontsize=8.6, color=INK_2, va="top")
    ax.text(0.035, 0.165, "F2  2 points: 49%, chooses both visiting orders", fontsize=8.6, color=INK_2,
            va="top")
    ax.text(0.02, 0.97, "How the model was developed: each stage fixed the failure the previous one exposed",
            fontsize=11.5, fontweight="bold", va="top")
    ax.set_ylim(-0.07, 1)
    ax.text(0.02, -0.05, "success = share of 20 fresh episodes completed at the production standard, mean over "
                        "5 seeds. R4 onward are scored on a corrected target set, so R4's 97% is not directly "
                        "comparable to R3's 92%.", fontsize=7.8, color=INK_2, va="bottom")
    save(fig, "d5_research_path")


def main():
    _style()
    print(f"writing diagrams to {OUT}")
    training_loop()
    sac_networks()
    arm_schematic()
    curriculum_ladder()
    research_path()


if __name__ == "__main__":
    main()
