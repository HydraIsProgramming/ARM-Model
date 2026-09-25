"""
Report figures from the exported training data.

    python scripts/export_training_data.py      # first: writes the CSVs
    python scripts/plot_training.py             # then: draws the figures

Reads project_assets/report/data/*.csv and writes PNG (for slides and documents)
and PDF (vector, for LaTeX) to project_assets/report/figures/. Every figure is
drawn from the CSVs alone, so anyone can regenerate or restyle them without the
models or the run directories.

Conventions, applied to every figure so they read as one set:
  * one y-axis per chart; measures on different scales get their own panels
  * a stage keeps the same colour in every figure (R3 blue, R4 orange, R5 aqua,
    R6 yellow - a set validated for colour-blind separation); figures that
    compare many stages encode the stage by POSITION instead and use one hue
  * individual seeds are thin grey lines, the mean over seeds is the coloured
    line, so the spread between seeds stays visible rather than averaged away
  * labels, values and legends are in ink colours, never in the series colour
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SEED = "#c9c8c3"
BAND = "#1baf7a"
STAGE_COLOUR = {"R3": "#2a78d6", "R4": "#eb6834", "R5": "#1baf7a", "R6": "#eda100"}
ONE_HUE = "#2a78d6"
NEG = "#e34948"

STAGE_LABEL = {
    "A": "A  - 2 fixed points", "B": "B  - 3 fixed points (v1)", "B3": "B3 - 3 fixed points (v2)",
    "R3": "R3 - 3 random points (deployed)", "R4": "R4 - + end-range rule",
    "R5": "R5 - + speed cost (quadratic)", "R6": "R6 - + speed cost (linear+quad)",
    "F2": "F2 - 2 points, free order", "F3": "F3 - 3 points, free order",
    "F3_attempt1": "F3 attempt 1 (reward cliff)", "A_attempt1": "A attempt 1 (straight-arm stall)",
    "B_warm": "B warm-started", "B2": "B2 (stopped)",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "project_assets" / "report" / "data"))
    p.add_argument("--out", default=str(ROOT / "project_assets" / "report" / "figures"))
    p.add_argument("--champion", default=str(ROOT / "project_assets" / "outputs" / "CHAMPION_HUMAN_V1_R3"))
    return p.parse_args()


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    })


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def steps_axis(ax):
    ax.set_xlabel("environment steps")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e6 else (f"{v / 1e3:.0f}K" if v else "0")))


def pct_axis(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(-0.03, 1.03)


def mean_by_step(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.groupby("step", as_index=False)[col].mean()


def headline(ax, title, subtitle=None):
    """Bold headline with an optional smaller line beneath it, never on top of it."""
    ax.set_title(title, pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")


def end_labels(ax, ends, gap):
    """Label line ends, pushed apart so two lines ending together stay readable."""
    ends = sorted(ends, key=lambda e: e[1])
    last = -1e18
    for x, y, text in ends:
        y = max(y, last + gap)
        direct_label(ax, x, y, text)
        last = y


def direct_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", ha="left", fontsize=9, color=INK_2)


# ------------------------------------------------------------------ figures

def fig_learning_curves_R(rounds, out):
    """Production success over training for the random-point stages."""
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ends = []
    for stage in ("R3", "R4", "R5", "R6"):
        d = rounds[rounds.stage == stage]
        if d.empty:
            continue
        m = d.groupby("step")["prod_success_rate"].mean().rolling(3, min_periods=1).mean()
        ax.plot(m.index, m.values, color=STAGE_COLOUR[stage], label=STAGE_LABEL[stage])
        ends.append((m.index[-1], m.values[-1], stage))
    end_labels(ax, ends, 0.06)
    steps_axis(ax)
    pct_axis(ax)
    ax.set_ylabel("success at the production standard")
    headline(ax, "R4 learns fastest, R3 and R4 finish highest; the speed cost slows R5 and R6",
             "mean of 5 seeds, 3-round rolling mean; each round = 10 fresh episodes, 5 cm / 0.5 s hold")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1.08e6)
    save(fig, out, "fig01_learning_curves_random_points")


def fig_learning_small_multiples(rounds, out):
    """Every main stage, same scale, seeds visible."""
    stages = [s for s in ("A", "B", "B3", "R3", "R4", "R5", "R6", "F2", "F3") if s in set(rounds.stage)]
    cols = 3
    rows = int(np.ceil(len(stages) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.1 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    xmax = rounds.step.max()
    for ax, stage in zip(axes, stages):
        d = rounds[rounds.stage == stage]
        for _, g in d.groupby("seed"):
            ax.plot(g.step, g.prod_success_rate, color=SEED, linewidth=1.0)
        m = d.groupby("step")["prod_success_rate"].mean()
        ax.plot(m.index, m.values, color=ONE_HUE)
        ax.set_title(STAGE_LABEL.get(stage, stage), fontsize=10)
        pct_axis(ax)
        ax.set_xlim(0, xmax * 1.02)
        steps_axis(ax)
        ax.set_xlabel("")
    for ax in axes[len(stages):]:
        ax.set_visible(False)
    fig.suptitle("Production success during training, every stage on the same scale",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.text(0.01, -0.01, "blue = mean of seeds, grey = individual seeds; x-axis = environment steps",
             fontsize=9, color=INK_2)
    fig.tight_layout()
    save(fig, out, "fig02_learning_curves_all_stages")


def fig_curriculum(rounds, out):
    """How the training difficulty ratcheted - tolerance rung by seed."""
    stages = [s for s in ("R3", "R4", "R5", "R6") if s in set(rounds.stage)]
    fig, axes = plt.subplots(1, len(stages), figsize=(3.2 * len(stages), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, stage in zip(axes, stages):
        d = rounds[rounds.stage == stage]
        for i, (_, g) in enumerate(d.groupby("seed")):
            ax.step(g.step, g.train_tolerance * 100, where="post",
                    color=STAGE_COLOUR[stage], alpha=0.35 + 0.13 * i, linewidth=1.4)
        ax.set_title(stage, fontsize=11)
        ax.set_yticks([15, 10, 7, 5])
        ax.invert_yaxis()
        steps_axis(ax)
    axes[0].set_ylabel("training tolerance (cm) - tighter is harder")
    fig.suptitle("Curriculum: each seed tightens 15 -> 10 -> 7 -> 5 cm once it passes two rounds running",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out, "fig03_curriculum_tolerance")


def fig_episode_reward(sac, out):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ends = []
    for stage in ("R3", "R4", "R5", "R6"):
        d = sac[sac.stage == stage]
        if d.empty:
            continue
        d = d.assign(bucket=(d.total_timesteps // 10000) * 10000)
        m = d.groupby("bucket")["ep_rew_mean"].mean()
        ax.plot(m.index, m.values, color=STAGE_COLOUR[stage], label=STAGE_LABEL[stage])
        ends.append((m.index[-1], m.values[-1], stage))
    end_labels(ax, ends, 38)
    steps_axis(ax)
    ax.set_ylabel("mean episode return (training)")
    headline(ax, "Training return rises throughout every random-point stage",
             "mean of 5 seeds, 10K-step bins. R5 and R6 sit lower partly by construction: their "
             "speed cost is itself a charge, so returns are not comparable across stages")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.1e6)
    save(fig, out, "fig04_episode_return")


def fig_sac_dynamics(sac, out, stage="R3"):
    """SAC's internals - separate panels, never a shared second axis."""
    d = sac[sac.stage == stage].copy()
    if d.empty:
        return
    panels = [("ent_coef", "entropy temperature  alpha", True),
              ("critic_loss", "critic (Q) loss", True),
              ("actor_loss", "actor (policy) loss", False)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.6, 7.4), sharex=True)
    for ax, (col, label, log) in zip(axes, panels):
        for _, g in d.groupby("seed"):
            ax.plot(g.total_timesteps, g[col], color=SEED, linewidth=0.9)
        m = d.assign(bucket=(d.total_timesteps // 10000) * 10000).groupby("bucket")[col].mean()
        ax.plot(m.index, m.values, color=STAGE_COLOUR.get(stage, ONE_HUE))
        ax.set_ylabel(label)
        if log:
            ax.set_yscale("log")
    steps_axis(axes[-1])
    headline(axes[0], f"SAC internals, Stage {stage}: the temperature overshoots, recovers, then holds",
             "alpha is auto-tuned toward a target entropy: it crashes in the first ~40K steps, climbs "
             "back and settles near 0.04; the actor loss falls as the learned Q-values rise")
    fig.tight_layout()
    save(fig, out, f"fig05_sac_internals_{stage}")


def fig_training_time(timing, out):
    stages = [s for s in ("A_attempt1", "A", "B", "B_warm", "B2", "B3", "R3", "R4", "R5", "R6",
                          "F3_attempt1", "F3", "F2") if s in set(timing.stage)]
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(stages) + 1.2))
    for i, stage in enumerate(stages):
        d = timing[timing.stage == stage]
        ax.barh(i, d.training_time_h.mean(), color=ONE_HUE, height=0.55, alpha=0.9)
        ax.plot(d.training_time_h, [i] * len(d), "o", color=INK, markersize=3.5, alpha=0.6)
        steps = d.total_timesteps.mean()
        steps_txt = f"{steps / 1e6:.1f}M" if steps >= 1e6 else f"{steps / 1e3:,.0f}K"
        ax.text(d.training_time_h.max() + 0.08, i,
                f"{d.training_time_h.mean():.1f} h   ({steps_txt} steps, "
                f"{d.steps_per_s.mean():.0f} steps/s)", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels([STAGE_LABEL.get(s, s) for s in stages], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("wall-clock hours per seed (5 seeds ran in parallel)")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, timing.training_time_h.max() * 1.55)
    ax.set_axisbelow(True)
    ax.set_title("Training time per stage")
    save(fig, out, "fig06_training_time")


def fig_final_success(finals, out):
    stages = [s for s in ("A", "B", "B3", "R3", "R4", "R5", "R6", "F2") if s in set(finals.stage)]
    fig, ax = plt.subplots(figsize=(8.6, 0.5 * len(stages) + 1.2))
    for i, stage in enumerate(stages):
        d = finals[finals.stage == stage]
        # seeds with the same score would hide behind one another: spread each
        # group of equal values evenly so every seed is its own visible dot
        offsets = np.zeros(len(d))
        for _, idx in d.groupby(d.success_rate.round(3)).groups.items():
            pos = [d.index.get_loc(j) for j in idx]
            if len(pos) > 1:
                offsets[pos] = np.linspace(-0.3, 0.3, len(pos))
        ax.plot(d.success_rate, i + offsets, "o", color=ONE_HUE, markersize=8, alpha=0.85,
                markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none")
        mean = d.success_rate.mean()
        ax.plot([mean, mean], [i - 0.3, i + 0.3], color=INK, linewidth=2)
        ax.text(1.04, i, f"{mean:.0%}  ({(d.success_rate >= 0.8).sum()}/{len(d)} seeds >= 80%)",
                va="center", fontsize=8.5, color=INK_2)
    ax.axvline(0.8, color=INK_2, linestyle=":", linewidth=1)
    ax.text(0.8, len(stages) - 0.35, "pass line (80%)", ha="center", va="top",
            fontsize=8, color=INK_2)
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels([STAGE_LABEL.get(s, s) for s in stages], fontsize=9)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(-0.03, 1.4)
    ax.spines["bottom"].set_bounds(0, 1.0)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlabel("final success, 20 fresh episodes per seed (dots = seeds, bar = mean)")
    headline(ax, "Final evaluation by stage",
             "R4 onward are scored on the corrected target set (end-range rule), so R3 and R4 are "
             "not directly comparable here")
    save(fig, out, "fig07_final_success")


def fig_human_likeness(finals, out):
    metrics = [("hl_peak_speed", "peak hand speed (m/s)", (1.0, 1.5)),
               ("hl_straightness", "path straightness", (1.0, 1.15)),
               ("hl_symmetry", "velocity symmetry", (0.35, 0.55)),
               ("hl_submovements", "submovements per reach", (1.0, 2.0)),
               ("hl_dimensionless_jerk", "dimensionless jerk", (0.0, 60.0)),
               ("hl_movement_time", "reach duration (s)", (0.4, 1.2))]
    stages = [s for s in ("R3", "R4", "R5", "R6") if s in set(finals.stage)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, (col, label, (lo, hi)) in zip(axes.ravel(), metrics):
        if col not in finals:
            ax.set_visible(False)
            continue
        ax.axhspan(lo, hi, color=BAND, alpha=0.13, linewidth=0)
        for i, stage in enumerate(stages):
            d = finals[finals.stage == stage][col].dropna()
            ax.plot([i] * len(d), d, "o", color=STAGE_COLOUR[stage], markersize=7, alpha=0.75,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
            # median, not mean: one outlier episode can move a seed's jerk by 40x
            ax.plot([i - 0.25, i + 0.25], [d.median()] * 2, color=INK, linewidth=2)
        if col == "hl_dimensionless_jerk":
            ax.set_yscale("log")
            ax.axhspan(1, hi, color=BAND, alpha=0.13, linewidth=0)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages)
        ax.set_xlim(-0.6, len(stages) - 0.4)
        ax.set_title(label + ("  (log scale)" if col == "hl_dimensionless_jerk" else ""), fontsize=10)
    fig.suptitle("Human-likeness per stage against the human range (green band)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.text(0.01, -0.01, "dots = seeds (20 episodes each), bar = median; bands from the motor-control "
                          "literature: Morasso 1981, Flash & Hogan 1985, Hogan & Sternad 2009",
             fontsize=9, color=INK_2)
    fig.tight_layout()
    save(fig, out, "fig08_human_likeness")


def fig_speed_vs_accuracy(finals, out):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.axvspan(1.0, 1.5, color=BAND, alpha=0.13, linewidth=0)
    ax.text(1.25, 1.0, "human range", ha="center", va="top", fontsize=8.5, color=INK_2)
    for stage in ("R3", "R4", "R5", "R6"):
        d = finals[finals.stage == stage]
        if d.empty or "hl_peak_speed" not in d:
            continue
        ax.plot(d.hl_peak_speed, d.success_rate, "o", color=STAGE_COLOUR[stage], markersize=8,
                alpha=0.7, markeredgecolor=SURFACE, markeredgewidth=1.2)
        mx, my = d.hl_peak_speed.mean(), d.success_rate.mean()
        ax.plot(mx, my, "D", color=STAGE_COLOUR[stage], markersize=11, markeredgecolor=INK,
                markeredgewidth=1.2, label=STAGE_LABEL[stage])
        direct_label(ax, mx, my, stage)
    ax.set_xlabel("peak hand speed (m/s)")
    ax.set_ylabel("final success")
    pct_axis(ax)
    headline(ax, "Speed against accuracy: the speed cost bought human speed with reliability",
             "small dots = seeds, diamonds = stage mean")
    ax.legend(loc="center right", fontsize=8.5)
    save(fig, out, "fig09_speed_vs_accuracy")


def reward_term_rollout(champion: Path, data: Path):
    """One episode of the deployed model with every reward term recorded."""
    sys.path.insert(0, str(ROOT / "src"))
    from rl_armMotion.two_d.evaluation import load_package
    model, env, _ = load_package(champion, which="best", terminate_on_success=True)
    base = getattr(env, "unwrapped", env)
    base.random_waypoints = None
    base.set_waypoints([[1.27, 0.27], [1.39, -0.01], [1.21, -0.29]])
    obs, info = base.reset(seed=0)
    rows = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = base.step(action)
        row = {"step": int(info["step"]), "time_s": info["step"] * base.dt, "reward": reward,
               "waypoints_reached": info["waypoints_reached"],
               "completed": info.get("waypoint_completed"), "ee_speed": info["ee_speed"]}
        row.update({f"term_{k}": float(v) for k, v in base._last_reward_terms.items()})
        rows.append(row)
        if terminated or truncated:
            break
    df = pd.DataFrame(rows)
    df.to_csv(data / "reward_terms_one_episode.csv", index=False)
    return df


def fig_reward_terms(df, out):
    terms = [c for c in df.columns if c.startswith("term_")]
    totals = df[terms].sum().rename(lambda c: c[5:]).sort_values()
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    colours = [NEG if v < 0 else ONE_HUE for v in totals.values]
    ax.barh(range(len(totals)), totals.values, color=colours, height=0.6)
    for i, v in enumerate(totals.values):
        v_txt = "0.0" if abs(v) < 0.05 else f"{v:+.1f}"
        ax.text(v + (0.6 if v >= 0 else -0.6), i, v_txt, va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(totals)))
    ax.set_yticklabels(totals.index)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    span = max(abs(totals.min()), abs(totals.max()))
    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.set_xlabel("total contribution to the episode's return")
    headline(ax, f"Where one episode's return comes from ({df.reward.sum():+.1f} over {len(df)} steps)",
             "deployed R3 model on the demo points, scored under the reward it was trained with; "
             "blue = rewards, red = costs")
    ax.set_axisbelow(True)
    save(fig, out, "fig10_reward_terms_breakdown")

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.plot(df.time_s, df.reward, color=ONE_HUE, linewidth=1.4)
    for _, r in df[df.completed.notna()].iterrows():
        ax.axvline(r.time_s, color=INK_2, linestyle=":", linewidth=1)
        ax.text(r.time_s, ax.get_ylim()[1], f" point {int(r.completed) + 1}", fontsize=8,
                color=INK_2, va="top")
    last = df.iloc[-1]
    if last.reward > 5:
        ax.annotate("hold complete (+50)", (last.time_s, last.reward), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8.5, color=INK_2)
    ax.set_xlabel("time in the episode (s)")
    ax.set_ylabel("reward per step")
    ax.set_title("Reward per step through one episode: spikes are the completion bonuses")
    save(fig, out, "fig11_reward_timeline")


def main():
    args = parse_args()
    data, out = Path(args.data), Path(args.out)
    style()
    rounds = pd.read_csv(data / "curriculum_rounds.csv")
    sac = pd.read_csv(data / "sac_training_log.csv")
    timing = pd.read_csv(data / "run_timing.csv")
    finals = pd.read_csv(data / "final_evaluation.csv")
    print(f"writing figures to {out}")
    fig_learning_curves_R(rounds, out)
    fig_learning_small_multiples(rounds, out)
    fig_curriculum(rounds, out)
    fig_episode_reward(sac, out)
    fig_sac_dynamics(sac, out, "R3")
    fig_training_time(timing, out)
    fig_final_success(finals, out)
    fig_human_likeness(finals, out)
    fig_speed_vs_accuracy(finals, out)
    champion = Path(args.champion)
    if champion.exists():
        fig_reward_terms(reward_term_rollout(champion, data), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
