"""
Build the model study: what each model does, how the models compare, what is wrong with each,
muscles against actuators, how the training budget was cut, and how reward weights change learning.

    python scripts/export_training_data.py     # training runs -> CSV (curriculum rounds, final evals, timing)
    python scripts/study_measurements.py       # the same three tests on every model -> report/study/*.json
    python scripts/model_profiles.py           # every model on the same 3 points -> report/profiles/
    python scripts/build_study_pdf.py          # -> report/ARM_Model_Study.pdf and docs/MODEL_STUDY.md

Every number is read from those files at build time, so the study cannot drift from the
measurements. A model whose data is not on disk yet (a variant still training) is shown as
"pending" rather than left out, so the gaps stay visible. One document is assembled once and
rendered twice: as the PDF and as Markdown.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report_pdf as R                                        # noqa: E402  fonts, styles, flowables
import plot_training as PT                                          # noqa: E402  chart style

import matplotlib                                                   # noqa: E402
import matplotlib.pyplot as plt                                     # noqa: E402
from reportlab.lib.styles import ParagraphStyle                     # noqa: E402
from reportlab.lib.units import inch                                # noqa: E402
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer  # noqa: E402

ROOT = R.ROOT
REPORT = R.REPORT
DATA = R.DATA
PROFILES = REPORT / "profiles"
STUDY = REPORT / "study"
FIGS = REPORT / "build" / "study"
OUT_PDF = REPORT / "ARM_Model_Study.pdf"
OUT_MD = ROOT / "docs" / "MODEL_STUDY.md"
RUNS = Path(os.environ.get("ARM_MODEL_OUTPUT_DIR", "C:/arm-model-runs"))

MUSCLE_C, ACTUATOR_C = "#2a78d6", "#eb6834"          # categorical slots 1 and 2: muscles, actuators
FOCUS = [("R3", "#2a78d6"), ("R7", "#eb6834"), ("R8", "#1baf7a"), ("R8F", "#eda100")]
PENDING = "pending"

# model, family, actuation, what it was trained to do, the one change from the model before it
MODELS = [
    ("A", "fixed points", "muscle", "reach 2 fixed points (v1 set), hold at the second",
     "first working model (a first attempt stalled with a straight arm)"),
    ("B", "fixed points", "muscle", "reach 3 fixed points (v1 set), hold at the third", "a third point"),
    ("B3", "fixed points", "muscle", "reach 3 fixed points (v2 set, all well inside reach)",
     "points moved inside the reach circle; muscle activation lag (40/60 ms) and a hand-acceleration penalty"),
    ("R3", "random points", "muscle", "reach 3 RANDOM points per episode in the reach-margin region",
     "random points every episode - the first model that generalises"),
    ("R4", "random points", "muscle", "as R3", "targets that need an end-range posture are no longer drawn"),
    ("R5", "random points", "muscle", "as R4", "hand-speed cost at human pace: cap 1.2 m/s, weight 10"),
    ("R6", "random points", "muscle", "as R4", "speed cost with a linear slope at the cap (linear 3 + quadratic 5)"),
    ("F2", "free order", "muscle", "reach 2 random points in an order it chooses itself",
     "free-order pilot: all targets visible at once, no ordering signal"),
    ("R7", "full workspace", "muscle", "reach 3 random points anywhere a human arm can reach",
     "targets limited only by the joint ranges (5° clear of each stop), not the 18 cm reach margin"),
    ("R8", "actuators", "torque", "as R7, with joint-torque actuators", "muscles replaced by shoulder and elbow torque motors"),
    ("R8E", "step saver", "torque", "as R8", "250-step training episodes (evaluation stays 500)"),
    ("R8N", "step saver", "torque", "as R8", "3-step returns"),
    ("R8U", "step saver", "torque", "as R8", "2 gradient updates per step"),
    ("R8T", "step saver", "torque", "as R8", "TQC instead of SAC"),
    ("R8F", "actuators", "torque", "as R8", "the step savers that worked, combined"),
    ("R8FA", "reward weight", "torque", "as R8F", "acceleration penalty saturates at 10 m/s² (ACCEL_REF 10 → 5)"),
    ("R8FP", "reward weight", "torque", "as R8F", "torque-rate penalty W_ACTION_RATE = 0.02"),
]
MODEL_INFO = {m[0]: m for m in MODELS}
ACTUATION = {m[0]: m[2] for m in MODELS}

# Hand-written verdicts. Numbers in them are inserted from the data ({...} fields); anything a
# measurement can say is said by the tables, not here.
NOTES = {
    "A": dict(
        good="Learns a reach-and-hold quickly ({t80}); the smallest task, a sanity check of the whole pipeline.",
        wrong="Memorises two points: on the study's three shared points it reaches {prof_reached} of the 2 "
              "it is given. Bimodal across seeds ({final}): a seed either solves it or stalls. No muscle lag "
              "and no acceleration penalty, so its hand acceleration peaks at {peak_accel} - several times "
              "anything human."),
    "B": dict(
        good="Shows the policy can chain three targets with a hold at the end.",
        wrong="Only {final} across seeds: the v1 points include one at the edge of reach, which most seeds "
              "never learn to hold. Warm-starting it from A did not help (kept as a failure record). Still "
              "memorises fixed points."),
    "B3": dict(
        good="Every seed solves its three points ({final}); smooth, slow reaches; the first 'clean' "
              "configuration (muscle activation lag and a hand-acceleration penalty), which every later "
              "model keeps.",
        wrong="Only knows its own three points - which are the study's shared points, so its profile is "
              "its best case, not a test. Cannot take a point clicked anywhere else. Even there it is slow "
              "to settle: {hold_extra} beyond the half-second hold before the last point counts."),
    "R3": dict(
        good="The first model that generalises: {final} on fresh random targets, {clicks} on point sets "
              "clicked freely. Fast on the shared points ({prof_time}). Deployed until R7.",
        wrong="Only covers the reach-margin region: on the full human workspace it completes {regions} of "
              "episodes. Moves faster than people (peak {peak_speed}); its speed cost does not bind - "
              "it pays it and speeds anyway."),
    "R4": dict(
        good="Highest success of the random-point models on its own region ({final}); no longer asked to "
              "hold poses that lean on a joint's end-range spring.",
        wrong="Same region limit as R3, same too-fast hand (peak {peak_speed}), and the highest acceleration "
              "on the shared points ({peak_accel})."),
    "R5": dict(
        good="The speed cost binds: peak hand speed {peak_speed}, inside the human range.",
        wrong="Success collapses to {final}: weight 10 on speed above 1.2 m/s makes moving expensive, and "
              "several seeds learn to move too slowly to finish. Proof that a penalty can be right in "
              "direction and wrong in size."),
    "R6": dict(
        good="Keeps human hand speed (peak {peak_speed}) and recovers much of R5's lost success ({final}).",
        wrong="Still below R4, with a large spread across seeds - the speed/accuracy trade-off is not free."),
    "F2": dict(
        good="Shows a policy can choose its own route between two targets.",
        wrong="Only {final}; the three-point free-order runs (F3) never chained more than one point. "
              "Needs a different observation or reward before it is useful; not pursued."),
    "R7": dict(
        good="Covers the whole human workspace: {regions} of episodes on the full-workspace test, and its "
              "champion completes {final_champion} of the standard 20-episode evaluation. Smoothest muscle "
              "model on the shared points (peak acceleration {peak_accel}). Deployed in the GUI.",
        wrong="Settling into the final hold can drag on: on the shared points it reaches the last target at "
              "{t_in_final}, as fast as R3, then needs {hold_extra} beyond the half-second hold while the hand "
              "makes small corrections in and out of the 5 cm target (Figure 3). Over the 30 audit episodes the "
              "median extra is {hold_med}, but one episode in ten needs {hold_p90} or more. Needs the most "
              "training ({t80}; trained to 2M steps). Still exceeds human hand speed (peak {peak_speed}): it "
              "keeps R3's non-binding speed cost."),
}


# ===================================================================== the document

def md(text: str) -> str:
    """Reportlab mini-markup -> Markdown."""
    t = re.sub(r"<br\s*/?>", " ", str(text))
    t = re.sub(r"</?b>", "**", t)
    t = re.sub(r"</?i>", "*", t)
    t = re.sub(r"<super>(.*?)</super>", r"^\1", t)
    t = re.sub(r"<sub>(.*?)</sub>", r"_\1", t)
    t = re.sub(r"<[^>]+>", "", t)
    return t.replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ").replace("&amp;", "&")


BULLET = ParagraphStyle("bullet", parent=R.S["body"], leftIndent=14, bulletIndent=3, spaceAfter=3)


class Doc:
    """One document, two renderings: the PDF story and docs/MODEL_STUDY.md."""

    def __init__(self):
        self.blocks = []

    def h1(self, t): self.blocks.append(("h1", t))
    def h2(self, t): self.blocks.append(("h2", t))
    def p(self, t): self.blocks.append(("p", t))
    def styled(self, t, style): self.blocks.append(("styled", t, style))
    def space(self, h): self.blocks.append(("space", h))
    def small(self, t): self.blocks.append(("small", t))
    def bullets(self, items): self.blocks.append(("bullets", list(items)))
    def table(self, rows, widths, **kw): self.blocks.append(("table", rows, widths, kw))
    def fig(self, path, caption, max_h=None): self.blocks.append(("fig", Path(path), caption, max_h))
    def key(self, items): self.blocks.append(("key", items))
    def page(self): self.blocks.append(("page",))

    def story(self):
        out = []
        for b in self.blocks:
            kind = b[0]
            if kind in ("h1", "h2", "small"):
                out.append(R.P(b[1], kind))
            elif kind == "p":
                out.append(R.P(b[1]))
            elif kind == "styled":
                out.append(R.P(b[1], b[2]))
            elif kind == "space":
                out.append(Spacer(1, b[1]))
            elif kind == "bullets":
                out += [Paragraph(t, BULLET, bulletText="•") for t in b[1]]
                out.append(Spacer(1, 4))
            elif kind == "table":
                rows, widths, kw = b[1], b[2], b[3]
                out += [R.table(rows, [w * R.TEXT_W for w in widths], **kw), Spacer(1, 8)]
            elif kind == "fig":
                if b[1].exists():
                    out.append(R.figure(b[1], b[2], max_h=b[3]))
                else:
                    out.append(R.P(f"[figure pending: {b[1].name}]", "small"))
            elif kind == "key":
                out += [R.keybox(b[1]), Spacer(1, 10)]
            elif kind == "page":
                out.append(PageBreak())
        return out

    def markdown(self, md_path: Path) -> str:
        lines = []
        for b in self.blocks:
            kind = b[0]
            if kind == "h1":
                lines += ["", f"## {md(b[1])}", ""]
            elif kind == "h2":
                lines += ["", f"### {md(b[1])}", ""]
            elif kind in ("p", "small"):
                lines += [md(b[1]), ""]
            elif kind == "styled":
                lines += [{"title": "# ", "subtitle": "*"}.get(b[2], "") + md(b[1])
                          + ("*" if b[2] == "subtitle" else ""), ""]
            elif kind == "bullets":
                lines += [f"- {md(t)}" for t in b[1]] + [""]
            elif kind == "table":
                rows = [[md(c).replace("|", "/").replace("\n", " ") for c in r] for r in b[1]]
                lines.append("| " + " | ".join(rows[0]) + " |")
                lines.append("|" + "---|" * len(rows[0]))
                lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
                lines.append("")
            elif kind == "fig":
                rel = os.path.relpath(b[1], md_path.parent).replace("\\", "/")
                lines += [f"![{md(b[2])[:80]}]({rel})", "", f"*{md(b[2])}*", ""]
            elif kind == "key":
                lines += ["| " + " | ".join(md(l) for _, l in b[1]) + " |", "|" + "---|" * len(b[1]),
                          "| " + " | ".join(f"**{md(v)}**" for v, _ in b[1]) + " |", ""]
        return "\n".join(lines).strip() + "\n"


def on_page(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setFont("DejaVu", 7.8)
    canvas.setFillColor(R.INK_2)
    canvas.drawString(R.MARGIN, R.PAGE_H - 0.5 * inch, "Human-arm RL models — model study")
    canvas.drawRightString(R.PAGE_W - R.MARGIN, 0.5 * inch, f"{doc.page}")
    canvas.setStrokeColor(R.RULE)
    canvas.line(R.MARGIN, R.PAGE_H - 0.56 * inch, R.PAGE_W - R.MARGIN, R.PAGE_H - 0.56 * inch)
    canvas.restoreState()


# ===================================================================== the data

def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


class Data:
    def __init__(self):
        self.finals = _csv(DATA / "final_evaluation.csv")
        self.rounds = _csv(DATA / "curriculum_rounds.csv")
        self.timing = _csv(DATA / "run_timing.csv")
        self.config = _csv(DATA / "stage_config.csv")
        self.weights = _csv(DATA / "reward_weights.csv")
        self.profiles = _csv(PROFILES / "profiles_summary.csv")
        self.study = {}
        for f in sorted(STUDY.glob("*_*.json")):
            name, test = f.stem.rsplit("_", 1)
            try:
                self.study.setdefault(name, {})[test] = json.loads(f.read_text(encoding="utf-8"))
            except ValueError:
                pass
        self._profiles_all = None
        self.champions = {d.name.rsplit("_", 1)[-1]: d for d in (ROOT / "project_assets" / "outputs").glob(
            "CHAMPION_HUMAN_V1_*") if (d / "run_manifest.json").exists()}

    # ---- per-stage numbers
    def has_stage(self, stage):
        return not self.rounds.empty and (self.rounds.stage == stage).any()

    def final_success(self, stage):
        if self.finals.empty:
            return []
        return self.finals[self.finals.stage == stage].success_rate.dropna().tolist()

    def champion_final(self, model):
        d = self.champions.get(model)
        if d is None or not (d / "final_eval.json").exists():
            return None
        ev = json.loads((d / "final_eval.json").read_text(encoding="utf-8"))
        s = ev.get("summary", ev)
        return s.get("successes"), s.get("n_episodes"), s.get("success_rate")

    def _first_sustained(self, stage, col, window=3, thr=0.8):
        """Per seed: the first evaluation round at which the mean of the last `window` rounds of `col` is
        >= thr (None if never), and the last step the seed was trained to."""
        out = {}
        if self.rounds.empty or col not in self.rounds:
            return out
        for seed, g in self.rounds[self.rounds.stage == stage].groupby("seed"):
            g = g.sort_values("step")
            vals = pd.to_numeric(g[col], errors="coerce")
            if vals.isna().all():
                continue
            roll = vals.rolling(window).mean()
            hit = g.step[roll >= thr - 1e-9]
            out[int(seed)] = (int(hit.iloc[0]) if len(hit) else None, int(g.step.max()))
        return out

    def t80_fixed(self, stage):
        """Steps to a sustained 80 % on the fixed 10-episode production evaluation (every model has it)."""
        return self._first_sustained(stage, "prod_success_rate")

    def t80_probe(self, stage):
        """Steps to a sustained 80 % on fresh probe episodes (R8 family onwards)."""
        return self._first_sustained(stage, "probe_success_rate")

    def training(self, stage):
        """True while any seed of the stage is still training (its final evaluation is not written yet)."""
        if self.timing.empty or "finished" not in self.timing:
            return False
        t = self.timing[self.timing.stage == stage]
        return bool(len(t)) and not t.finished.astype(str).str.lower().eq("true").all()

    def hold_settle(self, model):
        """On the shared points: when the hand first enters the last target, and how much longer than the
        required 0.5 s hold it then takes to complete it (time spent jittering in and out of the target)."""
        if self._profiles_all is None:
            p = PROFILES / "profiles_all.csv"
            self._profiles_all = pd.read_csv(p) if p.exists() else pd.DataFrame()
        df = self._profiles_all
        if df.empty or not (df.model == model).any():
            return None
        d = df[df.model == model]
        fin = d[(d.target_index == d.target_index.max()) & d.in_goal.astype(bool)]
        done = d[d.success.astype(bool)]
        if not len(fin) or not len(done):
            return None
        return float(fin.t.iloc[0]), max(0.0, float(done.t.iloc[0] - fin.t.iloc[0] - 0.5))

    def hours(self, stage):
        if self.timing.empty:
            return None
        t = self.timing[(self.timing.stage == stage) & (self.timing.training_time_h > 0)]
        return (float(t.training_time_h.mean()), float(t.steps_per_s.mean()), int(t.total_timesteps.mean())) \
            if len(t) else None

    def weight(self, stage, key):
        if self.weights.empty:
            return None
        w = self.weights[(self.weights.stage == stage) & (self.weights.weight == key)]
        return float(w.value.iloc[0]) if len(w) else None

    # ---- per-model measurements
    def regions(self, model):
        r = self.study.get(model, {}).get("regions")
        return next(iter(r.values())) if r else None

    def limits(self, model):
        return self.study.get(model, {}).get("limits")

    def clicks(self, model):
        return self.study.get(model, {}).get("clicks")

    def profile(self, model):
        if self.profiles.empty:
            return None
        p = self.profiles[self.profiles.model == model]
        return p.iloc[0].to_dict() if len(p) else None


# ---- formatting

def pct(x, digits=0):
    return PENDING if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{digits}f}%"


def kfmt(step):
    return "—" if step is None else (f"{step / 1e6:.2f}M" if step >= 1e6 else f"{step / 1e3:.0f}K")


def final_txt(vals):
    if not vals:
        return PENDING
    m = float(np.mean(vals))
    return f"{100 * m:.0f}% ± {100 * float(np.std(vals, ddof=1)) if len(vals) > 1 else 0:.0f} ({len(vals)} seeds)"


def t80_txt(t80, training=False):
    if not t80:
        return PENDING
    if training:
        hit = [s for s, _ in t80.values() if s is not None]
        return (f"training ({len(hit)} of {len(t80)} seeds there so far, at {kfmt(max(l for _, l in t80.values()))})")
    hit = sorted(s for s, _ in t80.values() if s is not None)
    miss = [last for s, last in t80.values() if s is None]
    if not hit:
        return f"not reached ({len(miss)} seeds, ≤{kfmt(max(miss))})"
    txt = f"{kfmt(float(np.median(hit)))} median, {kfmt(hit[0])}–{kfmt(hit[-1])}"
    return txt + (f"; {len(miss)} of {len(t80)} seeds never" if miss else "")


def wilson(k, n, z=1.96):
    if not n:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def facts(D: Data, m: str) -> dict:
    """The numbers the hand-written notes refer to."""
    prof = D.profile(m) or {}
    reg = D.regions(m)
    clk = D.clicks(m)
    lim = D.limits(m) or {}
    champ = D.champion_final(m)
    peak_speed = lim.get("peak_hand_speed", prof.get("peak_speed_mps"))
    return {
        "final": final_txt(D.final_success(m)),
        "final_champion": f"{champ[0]}/{champ[1]}" if champ else PENDING,
        "regions": pct(reg["episode_success"]) if reg else PENDING,
        "clicks": pct(clk["success_rate"]) if clk else PENDING,
        "t80": t80_txt(D.t80_fixed(m), D.training(m)),
        "hold_extra": f"{D.hold_settle(m)[1]:.1f} s" if D.hold_settle(m) else PENDING,
        "hold_med": f"{lim['final_hold_extra_s']:.2f} s" if lim.get("final_hold_extra_s") is not None else PENDING,
        "hold_p90": f"{lim['final_hold_extra_p90_s']:.2f} s" if lim.get("final_hold_extra_p90_s") is not None
                    else PENDING,
        "t_in_final": f"{D.hold_settle(m)[0]:.2f} s" if D.hold_settle(m) else PENDING,
        "peak_speed": f"{peak_speed:.2f} m/s" if peak_speed is not None else PENDING,
        "peak_accel": f"{prof['peak_accel_mps2']:.0f} m/s²" if prof else PENDING,
        "prof_time": f"{prof['time_to_success_s']:.2f} s" if prof and prof.get("success") else
                     ("did not finish" if prof else PENDING),
        "prof_reached": int(prof["reached"]) if prof else PENDING,
    }


# ===================================================================== figures

def _save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_success(D: Data, models):
    """Final 20-episode success, one dot per seed, mean as a bar."""
    rows = [(m, D.final_success(m)) for m in models if D.final_success(m)]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(rows) + 0.9))
    for i, (m, vals) in enumerate(rows):
        c = ACTUATOR_C if ACTUATION.get(m) == "torque" else MUSCLE_C
        ax.barh(i, np.mean(vals), height=0.56, color=c, alpha=0.9)
        ax.scatter(vals, [i] * len(vals), s=16, color="#0b0b0b", zorder=3, linewidths=0)
        ax.text(1.05, i, f"{100 * np.mean(vals):.0f}%", va="center", fontsize=8.5, color=PT.INK_2)
    ax.set_yticks(range(len(rows)), [m for m, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.13)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_xlabel("episodes completed (20 fresh episodes per seed)")
    ax.legend(handles=[matplotlib.patches.Patch(color=MUSCLE_C, label="muscles"),
                       matplotlib.patches.Patch(color=ACTUATOR_C, label="torque actuators"),
                       matplotlib.lines.Line2D([], [], marker="o", ls="", color="#0b0b0b", ms=4, label="one seed")],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8)
    return _save(fig, "success_by_model")


def fig_regions(D: Data, models):
    rows = [(m, D.regions(m)) for m in models if D.regions(m)]
    if not rows:
        return None
    names = ["old area", "outer ring", "overhead", "hanging low", "close in", "behind body"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.34 * len(rows) + 1.2), gridspec_kw={"width_ratios": [1, 1.7]})
    ax = axes[0]
    for i, (m, r) in enumerate(rows):
        c = ACTUATOR_C if ACTUATION.get(m) == "torque" else MUSCLE_C
        ax.barh(i, r["episode_success"], height=0.56, color=c)
        ax.text(r["episode_success"] + 0.02, i, f"{100 * r['episode_success']:.0f}%", va="center", fontsize=8,
                color=PT.INK_2)
    ax.set_yticks(range(len(rows)), [m for m, _ in rows]); ax.invert_yaxis()
    ax.set_xlim(0, 1.15); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_title("Whole episodes (3 targets + hold)", fontsize=10)
    ax = axes[1]
    grid = np.array([[(r["regions"].get(n, {}).get("completed", 0) / max(r["regions"].get(n, {}).get("attempted", 0), 1))
                      for n in names] for _, r in rows])
    im = ax.imshow(grid, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"]), vmin=0, vmax=1, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{100 * grid[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if grid[i, j] > 0.6 else PT.INK)
    ax.set_xticks(range(len(names)), names, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), [m for m, _ in rows]); ax.grid(False)
    ax.set_title("Targets completed by region (%)", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return _save(fig, "regions_by_model")


def fig_t80(D: Data, models, probe=False):
    rows = []
    for m in models:
        t = D.t80_probe(m) if probe else D.t80_fixed(m)
        if t:
            rows.append((m, t))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 0.38 * len(rows) + 0.9))
    xmax = max(last for _, t in rows for _, last in t.values())
    for i, (m, t) in enumerate(rows):
        c = ACTUATOR_C if ACTUATION.get(m) == "torque" else MUSCLE_C
        hits = [s for s, _ in t.values() if s is not None]
        miss = [last for s, last in t.values() if s is None]
        ax.scatter(hits, [i] * len(hits), s=26, color=c, zorder=3, linewidths=0)
        if miss:
            ax.scatter([xmax * 1.04] * len(miss), [i] * len(miss), s=26, facecolors="none", edgecolors=c, zorder=3)
        if hits:
            ax.plot([np.median(hits)] * 2, [i - 0.3, i + 0.3], color=PT.INK, lw=1.6)
    ax.set_yticks(range(len(rows)), [m for m, _ in rows]); ax.invert_yaxis()
    PT.steps_axis(ax)
    ax.set_xlim(0, xmax * 1.08)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_xlabel("environment steps until success stays at 80 % or more "
                  + ("(fresh probe episodes)" if probe else "(fixed evaluation set)"))
    ax.legend(handles=[matplotlib.lines.Line2D([], [], marker="o", ls="", color=PT.INK_2, ms=5, label="one seed"),
                       matplotlib.lines.Line2D([], [], marker="o", ls="", mfc="none", color=PT.INK_2, ms=5,
                                               label="never got there"),
                       matplotlib.lines.Line2D([], [], color=PT.INK, lw=1.6, label="median")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    return _save(fig, "t80_probe" if probe else "t80_fixed")


def _mean_curve(D: Data, stage, col, seeds=None, window=3):
    r = D.rounds[(D.rounds.stage == stage)]
    if seeds is not None:
        r = r[r.seed.isin(seeds)]
    if r.empty or col not in r or r[col].isna().all():
        return None
    piv = r.pivot_table(index="step", columns="seed", values=col)
    piv = piv.rolling(window, min_periods=1).mean()
    return piv.index.values, piv.mean(axis=1).values, piv.min(axis=1).values, piv.max(axis=1).values


def fig_curves(D: Data, stages, name, col="prod_success_rate", seeds=None, title=""):
    curves = [(s, c, _mean_curve(D, s, col, seeds)) for s, c in stages]
    curves = [x for x in curves if x[2] is not None]
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    for s, c, (x, m, lo, hi) in curves:
        ax.fill_between(x, lo, hi, color=c, alpha=0.12, linewidth=0)
        ax.plot(x, m, color=c, lw=2, label=s)
        ax.text(x[-1] * 1.01, m[-1], s, color=PT.INK, fontsize=8.5, va="center")
    ax.axhline(0.8, color=PT.INK_2, lw=0.8, ls=":")
    PT.steps_axis(ax); PT.pct_axis(ax)
    ax.set_ylabel("success (rolling 3 evaluations)")
    if title:
        ax.set_title(title, fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, name)


def fig_step_savers(D: Data, variants):
    have = [v for v in variants if D.has_stage(v)]
    if not have or not D.has_stage("R8"):
        return None
    fig, axes = plt.subplots(1, len(have), figsize=(3.2 * len(have), 3.0), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], have):
        seeds = sorted(D.rounds[D.rounds.stage == v].seed.unique())
        for s, c in (("R8", "#a6a49e"), (v, ACTUATOR_C)):
            cur = _mean_curve(D, s, "probe_success_rate", seeds)
            if cur is None:
                continue
            x, m, lo, hi = cur
            ax.fill_between(x, lo, hi, color=c, alpha=0.15, linewidth=0)
            ax.plot(x, m, color=c, lw=1.8, label=s if s == v else "R8 (same seeds)")
        ax.axhline(0.8, color=PT.INK_2, lw=0.8, ls=":")
        PT.steps_axis(ax); PT.pct_axis(ax)
        ax.set_title(v, fontsize=10)
        ax.legend(loc="lower right", fontsize=7.5)
    axes[0][0].set_ylabel("fresh-episode success")
    fig.tight_layout()
    return _save(fig, "step_savers")


def fig_penalties(D: Data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    a = np.linspace(0, 30, 400)
    ax = axes[0]
    for ref, c, lab in ((10.0, MUSCLE_C, "ACCEL_REF 10: saturates at 20 m/s² (R3–R8)"),
                        (5.0, ACTUATOR_C, "ACCEL_REF 5: saturates at 10 m/s² (R8FA)")):
        ax.plot(a, 0.02 * np.minimum((a / ref) ** 2, 4.0), color=c, lw=2, label=lab)
    ax.set_xlabel("hand acceleration (m/s²)"); ax.set_ylabel("penalty per step")
    ax.set_title("Acceleration penalty, W_ACCEL = 0.02", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax = axes[1]
    v = np.linspace(0, 3, 400)
    for stage, c in (("R3", MUSCLE_C), ("R5", ACTUATOR_C), ("R6", "#1baf7a")):
        cap = D.weight(stage, "EE_SPEED_CAP") or 1.2
        wq = D.weight(stage, "W_SPEED_EXCESS") or 0.0
        wl = D.weight(stage, "W_SPEED_LINEAR") or 0.0
        pc = D.weight(stage, "SPEED_PENALTY_CAP")
        pc = np.inf if pc is None or np.isnan(pc) else pc
        over = np.maximum(0, v - cap)
        ax.plot(v, wl * over + wq * np.minimum(over ** 2, pc), color=c, lw=2,
                label=f"{stage}{' / R4 / R7 / R8' if stage == 'R3' else ''}: cap {cap:g} m/s, "
                      f"quadratic {wq:g}" + (f", linear {wl:g}" if wl else ""))
    ax.axvspan(0, 1.5, color="#1baf7a", alpha=0.07, lw=0)
    ax.text(0.05, ax.get_ylim()[1] * 0.92, "human peak hand speed ≤ 1.5 m/s", fontsize=7.5, color=PT.INK_2)
    ax.set_xlabel("hand speed (m/s)"); ax.set_ylabel("penalty per step")
    ax.set_title("Hand-speed penalty by stage", fontsize=10)
    ax.legend(fontsize=7.5, loc="center left")
    fig.tight_layout()
    return _save(fig, "penalty_shapes")


def fig_profiles_focus(D: Data):
    p = PROFILES / "profiles_all.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    focus = [(m, c) for m, c in FOCUS if (df.model == m).any()]
    if not focus:
        return None
    panels = [("speed", "Hand speed", "m/s"), ("accel", "Hand acceleration", "m/s²"),
              ("shoulder_deg", "Shoulder angle", "degrees"), ("elbow_deg", "Elbow angle", "degrees"),
              ("tau_shoulder", "Shoulder torque", "N·m"), ("tau_elbow", "Elbow torque", "N·m")]
    fig, axes = plt.subplots(3, 2, figsize=(10, 8.2), sharex=True)
    for ax, (col, title, unit) in zip(axes.flat, panels):
        for m, c in focus:
            d = df[df.model == m]
            ax.plot(d.t, d[col], color=c, lw=1.4,
                    label=f"{m} ({'muscles' if ACTUATION.get(m) == 'muscle' else 'actuators'})")
        ax.set_title(title, fontsize=10); ax.set_ylabel(unit)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    axes[0][0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return _save(fig, "profiles_focus")


def fig_profiles_grid(D: Data, col, unit, name):
    p = PROFILES / "profiles_all.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    order = [m for m, *_ in MODELS if (df.model == m).any()]
    if not order:
        return None
    n = len(order)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 1.9 * rows + 0.4), sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax, m in zip(axes.flat, order):
        d = df[df.model == m]
        ax.plot(d.t, d[col], color=ACTUATOR_C if ACTUATION.get(m) == "torque" else MUSCLE_C, lw=1.3)
        ax.set_title(m, fontsize=9.5)
        done = d[d.success.astype(bool)] if "success" in d else d.iloc[0:0]
        if len(done):
            ax.axvline(done.t.iloc[0], color=PT.INK_2, lw=0.7, ls=":")
    for ax in axes[:, 0]:
        ax.set_ylabel(unit)
    fig.tight_layout()
    return _save(fig, name)


# ===================================================================== sections

def title_page(doc: Doc, D: Data):
    doc.space(1.1 * inch)
    doc.styled("Human-arm models: a study", "title")
    doc.space(10)
    doc.styled("What each arm model does, how the models compare, what is wrong with each, muscles against "
               "torque actuators, cutting the training budget, and how reward weights change learning", "subtitle")
    doc.space(20)
    doc.p(f"CP493 Directed Research — Wilfrid Laurier University · Ranjot Sandhu · {date.today():%B %Y}")
    n_models = sum(1 for m, *_ in MODELS if D.final_success(m) or m in D.champions)
    r7 = D.regions("R7")
    r8 = D.regions("R8")
    doc.key([(str(n_models), "models compared<br/>on the same tests"),
             (pct(r7["episode_success"]) if r7 else PENDING, "R7 (muscles) on the<br/>full human workspace"),
             (pct(r8["episode_success"]) if r8 else PENDING, "R8 (actuators) on the<br/>full human workspace"),
             ("3", "shared points for every<br/>model's profiles")])
    doc.p("Every model in this study was trained from scratch by reinforcement learning (Soft Actor-Critic) to "
          "move a simulated two-joint human arm to target points and hold the last one still. They differ in "
          "one setting at a time: the task, the workspace, the reward, or — new in R8 — what drives the joints. "
          "Every number below was measured on the same tests, from files written by the measurement scripts.")
    doc.page()


def s_overview(doc: Doc, D: Data):
    doc.h1("1. The models at a glance")
    rows = [["Model", "Drive", "Trained to", "What changed from the one before", "Steps", "Seeds"]]
    for m, fam, act, task, change in MODELS:
        h = D.hours(m)
        n = len(D.final_success(m)) or (len(D.t80_fixed(m)) if D.t80_fixed(m) else 0)
        if not n and m not in D.champions:
            steps, seeds = PENDING, "—"
        else:
            steps = ("training" if D.training(m) else kfmt(h[2])) if h else "—"
            seeds = str(n) if n else "1"
        rows.append([f"<b>{m}</b>", "actuators" if act == "torque" else "muscles", task, change, steps, seeds])
    doc.table(rows, [0.07, 0.1, 0.28, 0.39, 0.08, 0.08], font_size=7.6)
    doc.p("<b>Families.</b> A, B and B3 memorise fixed points. R3–R6 reach random points in a region kept "
          "18 cm inside the arm's reach. F2 chooses its own order. R7 covers the whole human workspace. R8 "
          "keeps R7's task and reward but replaces the four muscles with two torque motors; its variants "
          "(R8E, R8N, R8U, R8T, R8F) each change one training setting to cut the steps needed, and R8FA / "
          "R8FP change one reward weight.")


def s_methods(doc: Doc):
    doc.h1("2. How every model was measured")
    doc.bullets([
        "<b>Final evaluation.</b> Each seed's best checkpoint, 20 fresh episodes at the production standard "
        "(5 cm tolerance, half-second hold), deterministic policy. Reported as mean ± standard deviation over seeds.",
        "<b>Full-workspace region test.</b> 200 three-target episodes drawn from the whole human workspace — the "
        "same episodes for every model — scored as whole episodes and per region (overhead, behind the body, "
        "hanging low, close in, the outer ring, and the old reach-margin area).",
        "<b>Clicked points.</b> 80 trials of three points chosen uniformly anywhere in reach, as a user would "
        "click them in the GUI; with a 95 % Wilson interval.",
        "<b>Human-likeness audit.</b> 30 episodes: hard limits (range of motion, speed, torque), hand speed "
        "against the human range, how often the speed cost is exceeded, peak torque, end-range use, hold drift, "
        "and co-contraction (muscles) or torque smoothness (actuators).",
        "<b>Profiles on three shared points.</b> Every model, the same start pose and the same three points "
        "(1.40, 0.15), (1.20, 0.40), (1.40, −0.20) m, for 5 s: hand velocity and acceleration, joint angles "
        "and joint torques.",
        "<b>Steps to 80 % (T80).</b> The first evaluation round (every 20,000 steps) at which success averaged "
        "over the last three rounds reaches 80 %. Measured on the fixed 10-episode evaluation set for every "
        "model; from R8 on, also on 10 fresh episodes per round (the probe), which cannot be overfitted.",
    ])


def s_compare(doc: Doc, D: Data, figs):
    doc.h1("3. Head-to-head")
    models = [m for m, *_ in MODELS]
    rows = [["Model", "Final success (20 ep / seed)", "Full workspace", "Clicked points", "Shared 3 points"]]
    for m in models:
        f = facts(D, m)
        clk = D.clicks(m)
        ci = ""
        if clk and clk.get("wilson95"):
            lo, hi = clk["wilson95"]
            ci = f" ({100 * lo:.0f}–{100 * hi:.0f})"
        prof = D.profile(m)
        shared = (f"all 3 in {prof['time_to_success_s']:.2f} s" if prof and prof["success"] else
                  (f"{int(prof['reached'])} of {int(prof['n_points'])}" if prof else PENDING))
        if all(v == PENDING for v in (f["final"], f["regions"], f["clicks"])) and not prof:
            continue
        rows.append([f"<b>{m}</b>", f["final"], f["regions"], f["clicks"] + ci, shared])
    doc.table(rows, [0.09, 0.3, 0.17, 0.22, 0.22], font_size=7.8)
    if figs.get("success"):
        doc.fig(figs["success"], "Figure 1. Final success of every seed (dots) and the mean (bar). Fixed-point "
                "models are scored on their own points; the random-point models on fresh random targets in "
                "the region they were trained on.", max_h=3.4 * inch)
    if figs.get("regions"):
        doc.fig(figs["regions"], "Figure 2. The full-workspace region test: the same 200 episodes for every "
                "model. Models trained on the reach-margin region (R3–R6) only do well in the 'old area'.",
                max_h=3.6 * inch)

    doc.h2("Movement quality (human-likeness audit, 30 episodes)")
    rows = [["Model", "Peak hand speed", "Steps over speed cap", "Peak torque (shoulder / elbow)",
             "End-range steps", "Final-hold extra (median / p90)",
             "Co-contraction hold / move, or RMS torque rate"]]
    for m in models:
        lim = D.limits(m)
        if not lim:
            continue
        pt = lim.get("peak_torque_pct") or [np.nan, np.nan]
        if "rms_torque_rate" in lim:
            last = f"{lim['rms_torque_rate'][0]:.0f} / {lim['rms_torque_rate'][1]:.0f} N·m/s"
        elif lim.get("cocontraction_hold") is not None:
            last = f"{lim['cocontraction_hold']:.2f} / {lim['cocontraction_move']:.2f}"
        else:
            last = "—"
        hx, hp = lim.get("final_hold_extra_s"), lim.get("final_hold_extra_p90_s")
        rows.append([f"<b>{m}</b>", f"{lim['peak_hand_speed']:.2f} m/s", pct(lim["steps_above_speed_cap"]),
                     f"{pt[0]:.0f}% / {pt[1]:.0f}%", pct(lim["steps_in_end_range"], 1),
                     "—" if hx is None else f"{hx:.2f} / {hp:.2f} s", last])
    if len(rows) > 1:
        doc.table(rows, [0.08, 0.13, 0.13, 0.18, 0.12, 0.1, 0.26], font_size=7.6)
        doc.small("Final-hold extra: once the hand first reaches the last target, the time beyond the required "
                  "0.5 s hold before the hold completes (successful episodes). "
                  "Human peak hand speed in point-to-point reaching is roughly 0.5–1.5 m/s. 'Steps over speed cap' "
                  "counts steps above the cap of the model's own reward (1.2 or 1.5 m/s). Co-contraction is the mean "
                  "of the smaller activation in each agonist/antagonist pair: people stiffen by co-contracting "
                  "while they hold still.")
    else:
        doc.p("<i>Audit results pending.</i>")


def s_profiles(doc: Doc, D: Data, figs):
    doc.h1("4. Profiles: every model on the same three points")
    doc.p("Each model starts from the same pose and is given the same three points. Velocity and acceleration "
          "are of the hand (end effector); the torques are the net joint torques the drive applies — summed "
          "muscle torque for muscle models, the motor torque for actuator models.")
    if not D.profiles.empty:
        rows = [["Model", "Result", "Extra hold time", "Peak speed", "Peak accel.", "RMS accel.",
                 "Steps > 10 / > 20 m/s²", "Peak torque sh / el", "RMS torque rate sh / el"]]
        order = [m for m, *_ in MODELS]
        for m in order:
            p = D.profile(m)
            if not p:
                continue
            res = (f"{p['time_to_success_s']:.2f} s" if p["success"] else
                   f"{int(p['reached'])}/{int(p['n_points'])} reached")
            hs = D.hold_settle(m)
            rows.append([f"<b>{m}</b>", res, f"{hs[1]:.1f} s" if hs else "—",
                         f"{p['peak_speed_mps']:.2f}", f"{p['peak_accel_mps2']:.0f}",
                         f"{p['rms_accel_mps2']:.1f}",
                         f"{100 * p['accel_steps_over_10']:.0f}% / {100 * p['accel_steps_over_20']:.0f}%",
                         f"{p['peak_tau_shoulder']:.1f} / {p['peak_tau_elbow']:.1f}",
                         f"{p['rms_torque_rate_shoulder']:.0f} / {p['rms_torque_rate_elbow']:.0f}"])
        doc.table(rows, [0.07, 0.12, 0.1, 0.09, 0.09, 0.09, 0.14, 0.15, 0.15], font_size=7.4)
        doc.small("<b>Extra hold time</b>: after the hand first enters the last target, the time beyond the "
                  "required 0.5 s hold that the model needs to complete it - time spent drifting or jittering in "
                  "and out of the target. Speeds in m/s, accelerations in m/s², torques in N·m, torque rates in "
                  "N·m/s. A and F2 were "
                  "given points 1 and 3 only (they take two targets). A, B and F2 were not trained on these "
                  "points and are expected to fail; they are shown for completeness.")
    if figs.get("focus"):
        doc.fig(figs["focus"], "Figure 3. The deployed models and the actuator models on the shared points. "
                "On these points R7 reaches the last target at about 1.7 s but keeps making small corrections - "
                "hand acceleration and elbow torque oscillate - and completes the hold only at 4.7 s.",
                max_h=6.6 * inch)
    for key, cap in (("grid_speed", "Figure 4. Hand speed of every model on the shared points (dotted line: "
                                    "task complete)."),
                     ("grid_accel", "Figure 5. Hand acceleration of every model on the shared points.")):
        if figs.get(key):
            doc.fig(figs[key], cap, max_h=4.8 * inch)


def s_models(doc: Doc, D: Data):
    doc.h1("5. Model by model")
    for m, fam, act, task, change in MODELS:
        f = facts(D, m)
        has_any = D.final_success(m) or m in D.champions or D.profile(m) or D.has_stage(m)
        doc.h2(f"{m} — {task}")
        doc.p(f"<b>Changed:</b> {change}. <b>Drive:</b> {'torque actuators' if act == 'torque' else 'four muscles'}.")
        if not has_any:
            doc.p("<i>Not trained yet — results pending.</i>")
            continue
        doc.p(f"<b>Measured:</b> final success {f['final']}; full workspace {f['regions']}; clicked points "
              f"{f['clicks']}; steps to 80 % {f['t80']}; shared points {f['prof_time']}; peak hand speed "
              f"{f['peak_speed']}; peak hand acceleration {f['peak_accel']}.")
        note = NOTES.get(m)
        if note:
            doc.p("<b>Good:</b> " + note["good"].format(**f))
            doc.p("<b>Wrong:</b> " + note["wrong"].format(**f))
        else:
            doc.p("<i>Verdict written once its results are in.</i>")


def s_actuators(doc: Doc, D: Data, figs):
    doc.h1("6. Muscles against torque actuators (R7 vs R8)")
    doc.p("R8 keeps everything about R7 — the full human workspace, the reward, the curriculum, the network, the "
          "training budget — and changes only the drive. R7's policy sets four muscle activations (shoulder and "
          "elbow, each an agonist/antagonist pair) that act through Hill-type force-length-velocity curves with "
          "a 40/60 ms activation lag. R8's policy sets two joint torques directly, up to the same limits "
          "(25 N·m shoulder, 12 N·m elbow). Because a motor has no activation state to observe, R8's observation "
          "carries the previous torque command instead (16 inputs instead of 18).")
    rows = [["", "R7 (muscles)", "R8 (actuators)"]]
    def both(fn):
        return [fn("R7"), fn("R8")]
    rows.append(["Final success"] + both(lambda m: final_txt(D.final_success(m))))
    rows.append(["Full workspace"] + both(lambda m: pct(D.regions(m)["episode_success"]) if D.regions(m) else PENDING))
    rows.append(["Clicked points"] + both(lambda m: pct(D.clicks(m)["success_rate"]) if D.clicks(m) else PENDING))
    rows.append(["Steps to 80 % (fixed set)"] + both(lambda m: t80_txt(D.t80_fixed(m), D.training(m))))
    rows.append(["Training time per run"] + both(lambda m: f"{D.hours(m)[0]:.1f} h ({D.hours(m)[1]:.0f} steps/s)"
                                                 if D.hours(m) else PENDING))
    rows.append(["Peak hand speed / accel. (shared pts)"] + both(
        lambda m: f"{D.profile(m)['peak_speed_mps']:.2f} m/s / {D.profile(m)['peak_accel_mps2']:.0f} m/s²"
        if D.profile(m) else PENDING))
    rows.append(["RMS torque rate sh / el (shared pts)"] + both(
        lambda m: f"{D.profile(m)['rms_torque_rate_shoulder']:.0f} / {D.profile(m)['rms_torque_rate_elbow']:.0f} N·m/s"
        if D.profile(m) else PENDING))
    doc.table(rows, [0.36, 0.32, 0.32], font_size=8)
    if figs.get("curves_drive"):
        doc.fig(figs["curves_drive"], "Figure 6. Learning curves on the fixed evaluation set: mean over seeds "
                "(line) and the range across seeds (band). Dotted line: 80 %.", max_h=3.2 * inch)
    doc.h2("What to expect, and why")
    doc.bullets([
        "<b>Easier control problem.</b> Two outputs instead of four, no redundancy (a muscle pair can produce the "
        "same torque with any amount of co-contraction), no force-length-velocity curves, no activation lag: the "
        "torque the policy asks for is the torque the joint gets. This usually means fewer steps to learn.",
        "<b>Less built-in smoothness.</b> The muscles' lag low-pass filters the command; a motor follows every "
        "change instantly. Unless the reward charges for it, an actuator policy can switch torque from one "
        "limit to the other in 20 ms — measured here as the RMS torque rate, and the reason for the R8FP variant.",
        "<b>No stiffness by co-contraction.</b> A muscle arm can stiffen itself to hold still; an actuator arm "
        "can only hold by active feedback. Hold drift in the audit shows whether that matters.",
        "<b>Less biological.</b> Torque control is the standard robotics abstraction; muscles are what a human arm "
        "has. For a model of human motion, R7 stays the more faithful model; R8 is the better controller if it "
        "measures as good with less training.",
    ])


def s_steps(doc: Doc, D: Data, figs):
    doc.h1("7. Cutting the training steps")
    doc.p("The muscle models need 0.6–1.2 million steps to hold 80 % success (table below), several hours of "
          "CPU per run. Each R8 variant changes one thing and is matched against R8's seeds 1–3; screening runs "
          "stop 200,000 steps after they first hold 80 % on fresh episodes, which is long enough to see "
          "whether the success lasts.")
    rows = [["Stage", "Change", "Steps to 80 % (fixed set)", "Steps to 80 % (fresh probes)", "Hours / run"]]
    for m in ["R3", "R4", "R7", "R8", "R8E", "R8N", "R8U", "R8T", "R8F", "R8FA", "R8FP"]:
        if not D.has_stage(m):
            if m.startswith("R8"):
                rows.append([f"<b>{m}</b>", MODEL_INFO[m][4], PENDING, PENDING, PENDING])
            continue
        h = D.hours(m)
        tr = D.training(m)
        rows.append([f"<b>{m}</b>", MODEL_INFO[m][4], t80_txt(D.t80_fixed(m), tr),
                     t80_txt(D.t80_probe(m), tr) if m.startswith("R8") else "—",
                     "training" if tr else (f"{h[0]:.1f}" if h else "—")])
    doc.table(rows, [0.08, 0.34, 0.21, 0.21, 0.16], font_size=7.8)
    if figs.get("t80"):
        doc.fig(figs["t80"], "Figure 7. Steps to a sustained 80 % on the fixed evaluation set, per seed.",
                max_h=3.4 * inch)
    if figs.get("savers"):
        doc.fig(figs["savers"], "Figure 8. Each step saver against R8 on the same seeds (fresh-episode probes).",
                max_h=3.0 * inch)
    doc.h2("Why each change could help")
    doc.bullets([
        "<b>Shorter training episodes (R8E).</b> A successful 10-second episode is mostly holding still once the "
        "last point is reached. Cutting training episodes to 5 s spends the same steps on twice as many fresh "
        "targets. Evaluation still uses the full 10 s, so it cannot flatter the result.",
        "<b>n-step returns (R8N).</b> The critic learns from 3 real rewards before bootstrapping, so the "
        "+10 / +50 bonuses reach the actions that earned them in fewer updates. The cost is a little bias from "
        "off-policy data.",
        "<b>More updates per step (R8U).</b> Two gradient updates per environment step learn more from each "
        "sample. It halves steps only if it does not destabilise the critic, and it doubles compute per step, "
        "so wall-clock time can get worse even when steps improve.",
        "<b>TQC (R8T).</b> Truncated Quantile Critics replace SAC's two critics with distributional critics and "
        "drop the most optimistic quantiles, curbing the over-estimation that slows SAC. Same network sizes, "
        "same hyper-parameters otherwise.",
        "<b>The combination (R8F).</b> The changes that cut steps without hurting final success, combined and run "
        "on five seeds, are the recommended recipe.",
    ])


def s_weights(doc: Doc, D: Data, figs):
    doc.h1("8. How reward weights change learning")
    doc.p("The reward paid every 20 ms is a sum of terms. Two are the task: the shaping and distance terms pull "
          "the hand toward the active point, and the bonuses (+10 per intermediate point, +50 for the final "
          "hold) pay for finishing. The others are penalties: effort, joint speed, joint limits, hand speed, "
          "hand acceleration and, new here, torque rate. Changing a weight changes what the policy learns in "
          "three ways:")
    doc.bullets([
        "<b>It moves the optimum.</b> Except for the shaping term (which is potential-based and provably leaves "
        "the best policy unchanged), every penalty trades task reward for something else. Weight 10 on hand speed "
        "above 1.2 m/s (R5) made human-paced movement optimal — and made slow, unfinished movement nearly as "
        "good: success fell from R4's {r4} to {r5}.".format(
            r4=final_txt(D.final_success("R4")), r5=final_txt(D.final_success("R5"))),
        "<b>It changes what is easy to find.</b> Penalties are felt from the first step; the bonuses only after "
        "the policy has stumbled on success. A penalty that is large early makes 'do nothing' look good before "
        "the agent ever finishes a task — the straight-arm stall of stage A's first attempt is the extreme case. "
        "Large penalties therefore cost steps even when they leave the final optimum acceptable.",
        "<b>It changes the gradient, not just the level.</b> A capped penalty stops pushing once it saturates. "
        "R3's speed term is quadratic from a 1.5 m/s cap at weight 0.5: too shallow near the cap to matter, so "
        "R3 and R7 pay it and move faster than people (see the audit table). R6 added a linear part so the "
        "penalty has a slope right at the cap: {r6}, with human hand speed.".format(r6=final_txt(D.final_success("R6"))),
    ])
    if figs.get("penalties"):
        doc.fig(figs["penalties"], "Figure 9. Left: the hand-acceleration penalty before and after the change to "
                "a 10 m/s² ceiling. Right: the hand-speed penalties of R3, R5 and R6.", max_h=3.2 * inch)
    doc.h2("The acceleration penalty: maxing out at 10 m/s² instead of 20")
    doc.p("The term is −W_ACCEL·min((|a|/ACCEL_REF)², 4). Keeping W_ACCEL = 0.02 and the cap of 4, setting "
          "ACCEL_REF to 5 instead of 10 makes the penalty saturate at 10 m/s² instead of 20 m/s², with the same "
          "maximum of 0.08 per step. Two things change at once:")
    doc.bullets([
        "<b>Below 10 m/s², every acceleration costs four times as much</b> (the curve is four times steeper). "
        "Gentle reaches become relatively more expensive, so the policy should move with lower, smoother "
        "accelerations — or, if the extra cost outweighs the task reward early on, explore less and learn more "
        "slowly.",
        "<b>Above 10 m/s², extra acceleration is free.</b> A 30 m/s² jolt costs no more than a 10 m/s² push. "
        "The old setting kept discouraging spikes up to 20 m/s². For R7 on the shared points, {r7_10} of steps "
        "are above 10 m/s² and {r7_20} above 20 m/s², so the new ceiling sits inside R7's working range.".format(
            r7_10=pct(D.profile("R7")["accel_steps_over_10"]) if D.profile("R7") else PENDING,
            r7_20=pct(D.profile("R7")["accel_steps_over_20"]) if D.profile("R7") else PENDING),
        "<b>Measured in R8FA</b> against R8F on the same seeds: steps to 80 %, final success, and the profile's "
        "peak and RMS acceleration — table below.",
    ])
    doc.h2("The torque-rate penalty (R8FP)")
    doc.p("−W_ACTION_RATE·‖a<sub>t</sub> − a<sub>t−1</sub>‖² charges for changing the torque command between "
          "steps. Muscles get this smoothing for free from their activation lag; motors do not. At 0.02, a full "
          "swing of both motors from one limit to the other costs 0.16 — twice the acceleration penalty's "
          "maximum — while a steady command costs nothing. Expected effect: a lower torque rate and smoother hand "
          "acceleration, at some cost in how fast the arm can correct itself while holding.")
    rows = [["Stage", "Change", "Final success", "Steps to 80 % (probes)", "Peak / RMS accel. (shared pts)",
             "RMS torque rate"]]
    for m in ["R8F", "R8FA", "R8FP"]:
        p = D.profile(m)
        rows.append([f"<b>{m}</b>", MODEL_INFO[m][4], final_txt(D.final_success(m)),
                     t80_txt(D.t80_probe(m), D.training(m)),
                     f"{p['peak_accel_mps2']:.0f} / {p['rms_accel_mps2']:.1f} m/s²" if p else PENDING,
                     f"{p['rms_torque_rate_shoulder']:.0f} / {p['rms_torque_rate_elbow']:.0f} N·m/s" if p else PENDING])
    doc.table(rows, [0.08, 0.3, 0.14, 0.16, 0.17, 0.15], font_size=7.6)
    doc.h2("Weights by stage")
    keys = ["W_EFFORT", "W_SMOOTH", "W_LIMIT", "EE_SPEED_CAP", "W_SPEED_EXCESS", "W_SPEED_LINEAR", "W_ACCEL",
            "ACCEL_REF", "W_ACTION_RATE"]
    rows = [["Stage"] + [k.replace("_", " ") for k in keys]]
    for m in ["A", "B", "B3", "R3", "R4", "R5", "R6", "R7", "R8", "R8F", "R8FA", "R8FP"]:
        if D.weights.empty or not (D.weights.stage == m).any():
            continue
        rows.append([f"<b>{m}</b>"] + [R.fmt_w(D.weight(m, k)) if D.weight(m, k) is not None else "off"
                                       for k in keys])
    if len(rows) > 1:
        doc.table(rows, [0.1] + [0.1] * len(keys), font_size=7.2)
        doc.small("Read from each stage's own run manifest. 'off' means the term did not exist yet for that stage.")


def s_recommend(doc: Doc, D: Data):
    doc.h1("9. Recommendations")
    doc.bullets([
        "<b>For a model of human movement, keep R7</b> (muscles, full workspace) as the deployed model until an "
        "actuator model matches it on the full-workspace and clicked-point tests.",
        "<b>For the fewest training steps,</b> use the R8F recipe if it holds its final success — see section 7.",
        "<b>Make the speed cost bind gently</b> (R6's linear slope at a smaller weight) rather than strongly (R5); "
        "R7 and R8 still move faster than people.",
        "<b>Keep every old model.</b> All champions stay in project_assets/outputs (superseded packages move to "
        "outputs/_superseded, never deleted); every seed, checkpoint and replay buffer stays in C:/arm-model-runs.",
    ])


def appendix(doc: Doc, D: Data):
    imgs = [(m, PROFILES / f"{m}.png") for m, *_ in MODELS if (PROFILES / f"{m}.png").exists()]
    if not imgs:
        return
    doc.page()
    doc.h1("Appendix — every model's profile")
    for i, (m, p) in enumerate(imgs):
        doc.fig(p, f"Figure A{i + 1}. {m}: hand velocity, hand acceleration, joint angles and joint torques on "
                f"the shared points.", max_h=4.2 * inch)


# ===================================================================== main

def main():
    PT.style()
    D = Data()
    models = [m for m, *_ in MODELS]
    figs = {
        "success": fig_success(D, models),
        "regions": fig_regions(D, models),
        "t80": fig_t80(D, ["A", "B3", "R3", "R4", "R5", "R6", "F2", "R7", "R8", "R8E", "R8N", "R8U", "R8T", "R8F",
                           "R8FA", "R8FP"]),
        "curves_drive": fig_curves(D, [("R3", "#2a78d6"), ("R7", "#eb6834"), ("R8", "#1baf7a")], "curves_drive"),
        "savers": fig_step_savers(D, ["R8E", "R8N", "R8U", "R8T"]),
        "penalties": fig_penalties(D),
        "focus": fig_profiles_focus(D),
        "grid_speed": fig_profiles_grid(D, "speed", "m/s", "profiles_grid_speed"),
        "grid_accel": fig_profiles_grid(D, "accel", "m/s²", "profiles_grid_accel"),
    }
    doc = Doc()
    title_page(doc, D)
    s_overview(doc, D)
    s_methods(doc)
    doc.page()
    s_compare(doc, D, figs)
    doc.page()
    s_profiles(doc, D, figs)
    doc.page()
    s_models(doc, D)
    doc.page()
    s_actuators(doc, D, figs)
    doc.page()
    s_steps(doc, D, figs)
    doc.page()
    s_weights(doc, D, figs)
    s_recommend(doc, D)
    appendix(doc, D)

    story = doc.story()
    pdf = SimpleDocTemplate(str(OUT_PDF), pagesize=R.letter, leftMargin=R.MARGIN, rightMargin=R.MARGIN,
                            topMargin=R.MARGIN, bottomMargin=R.MARGIN, title="Human-arm RL models — model study",
                            author="Ranjot Sandhu")
    pdf.build(story, onFirstPage=on_page, onLaterPages=on_page)
    OUT_MD.write_text(doc.markdown(OUT_MD), encoding="utf-8")
    print(f"wrote {OUT_PDF}\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
