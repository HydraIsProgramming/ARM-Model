"""
Build the technical report PDF: the model, its constraints, the reward, and how
it was trained - with every diagram, graph and equation.

    python scripts/export_training_data.py   # training data -> CSV
    python scripts/plot_training.py          # CSV -> figures
    python scripts/report_diagrams.py        # explanatory diagrams
    python scripts/build_report_pdf.py       # everything -> the PDF

Tables of results, timing and reward weights are read from the exported CSVs at
build time, so the document cannot drift from the data. Descriptive tables
(constraints, observation layout) mirror docs/MODEL_SPECIFICATION.md.

Equations are typeset with matplotlib's mathtext (no LaTeX install is needed)
and text uses DejaVu Sans, which carries the Greek letters and maths symbols the
standard PDF fonts lack.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from PIL import Image as PILImage                                   # noqa: E402

from reportlab.lib import colors                                    # noqa: E402
from reportlab.lib.enums import TA_CENTER                           # noqa: E402
from reportlab.lib.pagesizes import letter                          # noqa: E402
from reportlab.lib.styles import ParagraphStyle                     # noqa: E402
from reportlab.lib.units import inch                                # noqa: E402
from reportlab.pdfbase import pdfmetrics                            # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont                        # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "project_assets" / "report"
DATA = REPORT / "data"
FIGS = REPORT / "figures"
DIAG = REPORT / "build" / "diagrams"
EQS = REPORT / "build" / "equations"
OUT = REPORT / "ARM_RL_Model_Technical_Report.pdf"

PAGE_W, PAGE_H = letter
MARGIN = 0.75 * inch
TEXT_W = PAGE_W - 2 * MARGIN

INK = colors.HexColor("#0b0b0b")
INK_2 = colors.HexColor("#52514e")
RULE = colors.HexColor("#d9d8d3")
HEAD_BG = colors.HexColor("#eef3fa")
BLUE = colors.HexColor("#2a78d6")
ROW_ALT = colors.HexColor("#f7f7f5")

# --------------------------------------------------------------------- fonts

_fdir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(_fdir / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(_fdir / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(_fdir / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                              italic="DejaVu-Oblique", boldItalic="DejaVu-Bold")

S = {
    "title": ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=26, leading=31, textColor=INK),
    "subtitle": ParagraphStyle("subtitle", fontName="DejaVu", fontSize=13.5, leading=19, textColor=INK_2),
    "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=16.5, leading=21, textColor=INK,
                         spaceBefore=4, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=12, leading=16, textColor=INK,
                         spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="DejaVu", fontSize=9.6, leading=14, textColor=INK, spaceAfter=6),
    "small": ParagraphStyle("small", fontName="DejaVu", fontSize=8.4, leading=11.5, textColor=INK_2),
    "caption": ParagraphStyle("caption", fontName="DejaVu", fontSize=8.4, leading=11.5, textColor=INK_2,
                              spaceBefore=2, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=8.2, leading=10.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DejaVu-Bold", fontSize=8.2, leading=10.6, textColor=INK),
    "key": ParagraphStyle("key", fontName="DejaVu-Bold", fontSize=20, leading=24, textColor=BLUE,
                          alignment=TA_CENTER),
    "keylabel": ParagraphStyle("keylabel", fontName="DejaVu", fontSize=8.4, leading=11, textColor=INK_2,
                               alignment=TA_CENTER),
}


# ----------------------------------------------------------------- helpers

def P(text, style="body"):
    return Paragraph(text, S[style])


def eq(name, tex, size=12.5, width=None):
    """Render one line of maths with mathtext and return a flowable at print scale."""
    EQS.mkdir(parents=True, exist_ok=True)
    path = EQS / f"{name}.png"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=size, color="#0b0b0b")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    w, h = PILImage.open(path).size
    w_pt, h_pt = w * 72 / 300, h * 72 / 300
    if width and w_pt > width:
        h_pt, w_pt = h_pt * width / w_pt, width
    img = Image(str(path), width=w_pt, height=h_pt)
    img.hAlign = "LEFT"
    return img


def eqs(lines, prefix):
    """A block of equation lines, indented slightly."""
    out = []
    for i, tex in enumerate(lines):
        e = eq(f"{prefix}_{i}", tex, width=TEXT_W - 14)
        out.append(Table([[e]], colWidths=[TEXT_W], style=[("LEFTPADDING", (0, 0), (-1, -1), 14),
                                                          ("TOPPADDING", (0, 0), (-1, -1), 1),
                                                          ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    out.append(Spacer(1, 6))
    return out


def figure(path, caption, width=TEXT_W, max_h=None):
    w, h = PILImage.open(path).size
    ht = width * h / w
    if max_h and ht > max_h:
        width, ht = width * max_h / ht, max_h
    img = Image(str(path), width=width, height=ht)
    return KeepTogether([img, P(caption, "caption")])


def table(rows, widths, header=True, zebra=True, bold_first_col=False, font_size=None):
    styles = {k: S[k] for k in ("cell", "cellb")}
    if font_size:
        styles = {k: ParagraphStyle(f"{k}{font_size}", parent=v, fontSize=font_size, leading=font_size * 1.3)
                  for k, v in styles.items()}
    data = []
    for r, row in enumerate(rows):
        cells = []
        for c, v in enumerate(row):
            style = "cellb" if (header and r == 0) or (bold_first_col and c == 0) else "cell"
            cells.append(v if not isinstance(v, str) else Paragraph(v, styles[style]))
        data.append(cells)
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK_2),
          ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
          ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), HEAD_BG))
    if zebra:
        for r in range(1, len(rows)):
            if r % 2 == 0:
                st.append(("BACKGROUND", (0, r), (-1, r), ROW_ALT))
    t.setStyle(TableStyle(st))
    return t


def keybox(items):
    """A row of headline numbers."""
    cells = [[Paragraph(v, S["key"]) for v, _ in items], [Paragraph(l, S["keylabel"]) for _, l in items]]
    w = TEXT_W / len(items)
    t = Table(cells, colWidths=[w] * len(items))
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, RULE), ("INNERGRID", (0, 0), (-1, -1), 0.4, RULE),
                           ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafaf8")),
                           ("TOPPADDING", (0, 0), (-1, 0), 9), ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def on_page(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setFont("DejaVu", 7.8)
    canvas.setFillColor(INK_2)
    canvas.drawString(MARGIN, PAGE_H - 0.5 * inch, "Reinforcement learning model of a human arm — technical report")
    canvas.drawRightString(PAGE_W - MARGIN, 0.5 * inch, f"{doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.line(MARGIN, PAGE_H - 0.56 * inch, PAGE_W - MARGIN, PAGE_H - 0.56 * inch)
    canvas.restoreState()


# --------------------------------------------------------------- the data

def load():
    timing = pd.read_csv(DATA / "run_timing.csv")
    finals = pd.read_csv(DATA / "final_evaluation.csv")
    weights = pd.read_csv(DATA / "reward_weights.csv").pivot(index="stage", columns="weight", values="value")
    terms = pd.read_csv(DATA / "reward_terms_one_episode.csv")
    return timing, finals, weights, terms


def fmt_w(v):
    if pd.isna(v):
        return "off"
    if np.isinf(v):
        return "∞"
    return f"{v:g}"


# --------------------------------------------------------------- sections

def title_page(story, timing, finals):
    r3 = timing[timing.stage == "R3"]
    story += [Spacer(1, 1.3 * inch),
              P("Teaching a simulated human arm to reach", "title"), Spacer(1, 10),
              P("A reinforcement-learning model: its design, the constraints on the arm, the reward "
                "function, and how it was trained", "subtitle"),
              Spacer(1, 26),
              P("<b>CP493 Directed Research</b> — Wilfrid Laurier University", "body"),
              P("Ranjot Sandhu", "body"),
              P(date.today().strftime("%B %Y"), "body"),
              Spacer(1, 40),
              keybox([("SAC", "algorithm<br/>Soft Actor-Critic"),
                      (f"{r3.training_time_h.mean():.1f} h", "to train one model<br/>(1 M steps, CPU)"),
                      ("92%", "success on unseen<br/>random targets"),
                      ("95%", "on freely clicked<br/>point sets")]),
              Spacer(1, 30),
              P("The model is a two-joint, muscle-driven model of a human right arm, built to human "
                "proportions and limits. A neural network controls its four muscles; it was trained, "
                "by trial and reward alone, to reach any three points in its workspace and hold the last "
                "one still. No demonstrations or motion data were used.", "body"),
              PageBreak()]


def s1_summary(story, timing):
    tot_h = timing.training_time_h.sum()
    tot_steps = timing.total_timesteps.sum() / 1e6
    story += [P("1. Summary", "h1"),
              P("<b>What was built.</b> A physics simulation of a human right arm seen from the side: upper arm "
                "and forearm-plus-hand, driven by four Hill-type muscles (an agonist and an antagonist at the "
                "shoulder and the elbow), with gravity, realistic segment masses, human range of motion and "
                "human torque limits. A neural-network policy reads 18 numbers describing the arm and its "
                "target, and outputs the four muscle activations 50 times a second."),
              P("<b>How it learns.</b> With <b>Soft Actor-Critic</b> (SAC), an off-policy deep "
                "reinforcement-learning algorithm. The policy is never shown how to move; it is only scored, "
                "every 20 ms, by a reward function (section 6) that pays for approaching and holding the target "
                "and charges for effort, jerky or overly fast motion, and straining the joints."),
              P("<b>What it can do.</b> The deployed model (stage <b>R3</b>) reaches three randomly placed "
                "targets and holds the last one for half a second in <b>92 %</b> of fresh episodes (mean of 5 "
                "independently trained seeds), and in <b>95 %</b> of sets of three points clicked freely by a "
                "user. It never exceeds a hard physical limit, and it stiffens its arm the way people do, by "
                "co-contracting opposing muscles while it holds still."),
              P(f"<b>How long it takes.</b> One model — 1,000,000 simulated steps — trains in about "
                f"<b>{timing[timing.stage == 'R3'].training_time_h.mean():.1f} hours</b> on a CPU (section 2). "
                f"The whole project ran {len(timing)} training runs: {tot_steps:.1f} million steps and "
                f"{tot_h:.0f} seed-hours of compute."),
              figure(DIAG / "d5_research_path.png",
                     "Figure 1. The development path. Each stage fixed the failure the previous one exposed; "
                     "the random-point stage R3 is the one that generalises to targets it has never seen."),
              PageBreak()]


def s2_time(story, timing):
    g = timing.groupby("stage").agg(seeds=("seed", "count"), steps=("total_timesteps", "mean"),
                                    hours=("training_time_h", "mean"), sps=("steps_per_s", "mean"),
                                    total=("training_time_h", "sum"))
    r3 = g.loc["R3"]
    rows = [["stage", "what it trained", "steps per model", "hours per model", "steps / s",
             "seed-hours, whole stage"]]
    what = {"A": "2 fixed targets", "B": "3 fixed targets (v1)", "B3": "3 fixed targets (v2)",
            "R3": "3 random targets (deployed)", "R4": "+ end-range rule", "R5": "+ speed cost",
            "R6": "+ binding speed cost", "F2": "2 targets, free order"}
    for s in ("A", "B", "B3", "R3", "R4", "R5", "R6", "F2"):
        if s in g.index:
            r = g.loc[s]
            steps = f"{r.steps / 1e6:.1f} M" if r.steps >= 1e6 else f"{r.steps / 1e3:,.0f} K"
            name = f"<b>{s}</b>" if s == "R3" else s
            rows.append([name, what[s], steps, f"{r.hours:.1f}", f"{r.sps:.0f}", f"{r.total:.0f}"])
    story += [P("2. How long it takes to train one model", "h1"),
              keybox([(f"{r3.hours:.1f} h", "wall-clock time,<br/>one model"),
                      ("1,000,000", "simulated steps<br/>(20 ms each)"),
                      ("2,000", "practice episodes<br/>of 10 seconds"),
                      (f"{r3.sps:.0f}", "steps per second<br/>on a CPU")]),
              Spacer(1, 12),
              P(f"<b>One model takes about {r3.hours:.1f} hours to train.</b> That is the measured wall-clock "
                f"time of the deployed R3 model: 1,000,000 environment steps at {r3.sps:.0f} steps per second. "
                "A step is 20 ms of simulated time, so the arm practises for 20,000 seconds — about "
                "<b>5.6 hours of simulated movement</b>, split into 2,000 attempts of 10 seconds each. "
                "Training therefore ran at roughly real-time speed."),
              P("<b>Hardware and set-up.</b> Training ran on the CPU; for networks this small the GPU was "
                "measured 3.5–5× <i>slower</i>, because moving tiny batches to and from the GPU costs more than "
                "the arithmetic it saves. Five models (five random seeds) were trained side by side, four CPU "
                "threads each, so a whole five-seed stage also finishes in about 5–6 hours of wall-clock time — "
                f"R3 used {r3.total:.0f} seed-hours in total. Each environment step triggers one gradient update "
                "of three 256×256 networks on a batch of 256 samples, and those updates, not the physics, "
                "dominate the time."),
              table(rows, [0.55 * inch, 1.9 * inch, 0.95 * inch, 0.95 * inch, 0.75 * inch, 1.9 * inch]),
              Spacer(1, 4),
              P("Hours per model are wall-clock, measured with five seeds sharing the machine. Seed-hours sum "
                "all five.", "small"),
              Spacer(1, 8),
              P("<b>One caveat.</b> Every timing here was measured with five seeds sharing the machine. A single "
                "model trained alone, with the whole CPU to itself, was never timed; it would probably finish "
                "somewhat sooner than 5.9 hours, but by how much was not measured."),
              PageBreak()]

    names = {"A_attempt1": "A, attempt 1", "A": "A", "B": "B", "B_warm": "B, warm-started from A",
             "B2": "B2", "B3": "B3", "R3": "<b>R3</b>", "R4": "R4", "R5": "R5", "R6": "R6",
             "F3_attempt1": "F3, attempt 1", "F3": "F3", "F2": "F2"}
    notes = {"A_attempt1": "stalled: straight arm on its elbow stop", "A": "2 fixed targets",
             "B": "3 fixed targets (v1)", "B_warm": "worse than training from scratch",
             "B2": "+ activation dynamics, acceleration cost; stopped", "B3": "3 fixed targets (v2)", "R3": "3 random targets (deployed)",
             "R4": "+ end-range rule", "R5": "+ speed cost", "R6": "+ binding speed cost",
             "F3_attempt1": "stopped: reward cliff", "F3": "stopped: no progress", "F2": "2 targets, free order"}
    allrows = [["stage", "what it was", "seeds", "steps per seed", "hours per seed", "seed-hours"]]
    for s in ("A_attempt1", "A", "B", "B_warm", "B2", "B3", "R3", "R4", "R5", "R6", "F3_attempt1", "F3", "F2"):
        if s in g.index:
            r = g.loc[s]
            steps = f"{r.steps / 1e6:.1f} M" if r.steps >= 1e6 else f"{r.steps / 1e3:,.0f} K"
            allrows.append([names[s], notes[s], f"{int(r.seeds)}", steps, f"{r.hours:.1f}", f"{r.total:.1f}"])
    allrows.append(["<b>total</b>", f"<b>{len(timing)} training runs</b>", "",
                    f"<b>{timing.total_timesteps.sum() / 1e6:.1f} M</b>", "",
                    f"<b>{timing.training_time_h.sum():.0f}</b>"])
    wall = timing.groupby("stage").training_time_h.max().sum()
    story += [figure(FIGS / "fig06_training_time.png",
                     "Figure 2. Wall-clock hours per model for every stage, including the failed attempts "
                     "(dots are individual seeds).", max_h=3.9 * inch),
              P("2.1 The whole project", "h2"),
              P(f"Counting every attempt, including the ones that failed, the project trained {len(timing)} "
                f"models for {timing.total_timesteps.sum() / 1e6:.1f} million steps in "
                f"{timing.training_time_h.sum():.0f} seed-hours. Because each stage's seeds ran at the same "
                f"time, that took about <b>{wall:.0f} hours</b> of machine time."),
              table(allrows, [1.45 * inch, 2.3 * inch, 0.5 * inch, 0.95 * inch, 0.9 * inch, 0.9 * inch]),
              PageBreak()]


def s3_system(story):
    story += [P("3. The system", "h1"),
              P("Every 20 ms the policy receives an <b>observation</b> of 18 numbers and chooses an "
                "<b>action</b>: four muscle activations between 0 and 1. The muscles turn activation into "
                "force according to their length and speed; the forces become joint torques; the rigid-body "
                "equations move the arm under gravity; the task checks whether the hand has reached or is "
                "holding its target; and the reward function scores what happened. The transition — what "
                "the policy saw, what it did, what it earned and what it saw next — is stored in a replay "
                "buffer, from which SAC samples batches to improve the policy."),
              figure(DIAG / "d1_training_loop.png",
                     "Figure 3. One control step. Everything inside the dashed boundary is the environment: "
                     "the simulated arm, its muscles, the task and the reward. The agent only sees the "
                     "observation and only acts through the four muscles.", max_h=3.05 * inch),
              P("3.1 What the policy sees and does", "h2"),
              P("<b>Observation</b> — 18 numbers, all scaled to small ranges so the network trains well. Because "
                "the target is part of the observation, the same policy can reach any target; including "
                "the muscles' own activation state lets it account for their 40–60 ms lag."),
              table([["#", "observation component", "count", "scale / range"],
                     ["1–4", "sine and cosine of the two joint angles", "4", "−1 … 1"],
                     ["5–6", "joint angular velocities", "2", "÷ 8 rad/s"],
                     ["7–8", "vector from the hand to the active target", "2", "÷ reach 0.65 m"],
                     ["9", "distance from the hand to the active target", "1", "÷ reach"],
                     ["10–11", "hand velocity", "2", "÷ 2 m/s"],
                     ["12", "which target is active (progress through the list)", "1", "0 … 1"],
                     ["13", "hand currently inside the goal (the reached test)", "1", "0 or 1"],
                     ["14", "hold progress at the current target", "1", "0 … 1"],
                     ["15–18", "current activation of each of the four muscles", "4", "0 … 1"]],
                    [0.6 * inch, 3.85 * inch, 0.65 * inch, 1.9 * inch]),
              Spacer(1, 6),
              P("<b>Action</b> — 4 numbers in [0, 1]: the commanded activation of the shoulder's two muscles "
                "and the elbow's two muscles. The policy never sets a torque or a position directly."),
              PageBreak()]


def s4_constraints(story):
    story += [P("4. The arm and every constraint on it", "h1"),
              figure(DIAG / "d3_arm_and_workspace.png",
                     "Figure 4. The arm at a representative pose, its joint ranges, and — in green — every "
                     "target it is allowed to be asked to reach: points it can both reach and hold still.",
                     max_h=5.3 * inch),
              P("4.1 Body dimensions", "h2"),
              P("Segment proportions follow Winter (2009) for a 75 kg, 1.75 m adult."),
              table([["segment", "length", "mass", "centre of mass from joint", "moment of inertia (about COM)"],
                     ["upper arm", "0.30 m", "2.10 kg", "0.131 m", "0.0196 kg·m²"],
                     ["forearm + hand", "0.35 m", "1.65 kg", "0.175 m", "0.0237 kg·m²"]],
                    [1.3 * inch, 0.8 * inch, 0.8 * inch, 1.8 * inch, 2.3 * inch]),
              PageBreak()]

    hard = [["constraint", "value", "kind", "how it is enforced"],
            ["shoulder range of motion", "−150° … +90° (flexion −60° … 180°)", "<b>hard</b>",
             "the integrator clamps the angle and removes velocity into the stop"],
            ["elbow range of motion", "0° … 145°", "<b>hard</b>", "as above"],
            ["joint torque", "shoulder 25 N·m, elbow 12 N·m", "<b>hard</b>",
             "net torque clipped; about 40 % of an adult's maximum"],
            ["joint speed", "8 rad/s", "<b>hard</b>", "safety clamp; muscles weaken with speed well before"],
            ["muscle activation", "0 … 1 per muscle", "<b>hard</b>", "the bounds of the action space"],
            ["activation rise / fall", "40 ms / 60 ms", "<b>hard</b>",
             "first-order filter: muscle force cannot change instantly"],
            ["gravity", "9.81 m/s²", "body", "gravity vector in the equations of motion"],
            ["end-range stiffness", "20 N·m/rad inside 10° of each stop", "body",
             "a spring that pushes the joint back out, like soft tissue"],
            ["joint damping", "0.30 N·m·s/rad", "body", "viscous resistance"],
            ["muscle force–length, force–velocity", "Hill-type (Thelen 2003; Hill 1938)", "body",
             "muscle force depends on fibre length and speed (4.3)"],
            ["hand speed", "cap 1.5 m/s (R3)", "soft", "a reward cost above the cap"],
            ["hand acceleration", "reference 10 m/s²", "soft", "a reward cost (minimum-jerk proxy)"],
            ["end-range posture", "within 10° of a stop", "soft", "a reward cost for entering the zone"],
            ["reached", "within 5 cm, hand < 0.10 m/s, joints < 1.0 rad/s", "task", "all three at once"],
            ["held", "25 steps in a row (0.5 s)", "task", "counter +1 inside, −5 per step outside"],
            ["episode length", "500 steps × 20 ms = 10 s", "task", "time limit, not a failure"]]
    story += [P("4.2 Constraints at a glance", "h2"),
              P("<b>Hard</b> limits are properties of the simulated body: nothing the policy does can exceed "
                "them, and none was exceeded over every step of 30 audited episodes. <b>Soft</b> limits are costs "
                "in the reward: the policy can exceed them, and pays for it. <b>Task</b> rules define success."),
              table(hard, [1.55 * inch, 2.05 * inch, 0.55 * inch, 2.85 * inch]),
              PageBreak()]

    story += [P("4.3 Physics", "h2"),
              P("The arm obeys the rigid-body equations of motion for a two-link chain, with the full coupling "
                "between the joints and gravity acting on both segments:")]
    story += eqs([r"$M(q)\,\ddot{q} \;+\; C(q,\dot{q})\,\dot{q} \;+\; G(q) \;=\; \tau_{\mathrm{muscle}} \;+\; \tau_{\mathrm{passive}}$",
                  r"$G(q) = g\,\left[(m_1 l_{c1} + m_2 L_1)\cos\theta_0 + m_2 l_{c2}\cos(\theta_0+\theta_1),\;\; m_2 l_{c2}\cos(\theta_0+\theta_1)\right]^{T}$",
                  r"$\tau_{\mathrm{passive}} = -B\,\dot{q} \;-\; K\,\max(0,\ q - (q_{\max}-\delta)) \;+\; K\,\max(0,\ (q_{\min}+\delta) - q)$"],
                 "phys")
    story += [P("Here q = (θ<sub>0</sub>, θ<sub>1</sub>) are the shoulder and elbow angles, M the 2×2 mass "
                "matrix, C the Coriolis and centrifugal terms, B = 0.30 N·m·s/rad the damping, K = 20 N·m/rad the "
                "end-range spring stiffness and δ = 10° its margin. The equations are integrated with a "
                "second-order (midpoint) method in four 5 ms sub-steps per 20 ms control step; total energy of "
                "the undamped arm drifts by only 0.3 % over 2 s, a check that the simulation is sound."),
              P("4.4 Muscles", "h2"),
              P("Each joint has two lumped muscles pulling in opposite directions (shoulder: anterior vs "
                "posterior deltoid; elbow: biceps and brachialis vs triceps). A muscle's force is its "
                "activation times its maximum force, scaled by how its length and speed affect it:")]
    story += eqs([r"$F \;=\; a \cdot F_{\max} \cdot f_L(\ell) \cdot f_V(\nu), \qquad \tau_{\mathrm{muscle}} = r\,(F_{+} - F_{-})$",
                  r"$f_L(\ell) = \exp\!\left(-\left(\frac{\ell/\ell_{\mathrm{opt}} - 1}{0.45}\right)^{2}\right) \qquad \mathrm{(force\!-\!length,\ Thelen\ 2003)}$",
                  r"$f_V(\nu) = \frac{1-\nu}{1+\nu/0.25}\ \ \mathrm{for}\ 0 \leq \nu < 1 \qquad \mathrm{(force\!-\!velocity,\ Hill\ 1938;\ eccentric\ branch\ up\ to\ 1.5)}$",
                  r"$\ell = \ell_{\mathrm{opt}} - s\,r\,(\theta - \theta_{\mathrm{ref}}), \qquad \nu = \frac{s\,r\,\dot{\theta}}{\ell_{\mathrm{opt}}\,V_{\max}}, \qquad V_{\max} = 6\ \ell_{\mathrm{opt}}/\mathrm{s}$",
                  r"$\dot{a} = \frac{u - a}{\tau_{\mathrm{act}}}, \qquad \tau_{\mathrm{act}} = 40\ \mathrm{ms\ rising},\ \ 60\ \mathrm{ms\ falling} \qquad \mathrm{(activation\ dynamics,\ Zajac\ 1989)}$"],
                 "musc")
    story += [table([["joint", "moment arm r", "max force F<sub>max</sub>", "optimal length ℓ<sub>opt</sub>",
                      "force reaches zero at"],
                     ["shoulder", "0.04 m", "625 N", "0.20 m", "30 rad/s"],
                     ["elbow", "0.03 m", "400 N", "0.10 m", "20 rad/s"]],
                    [1.0 * inch, 1.2 * inch, 1.4 * inch, 1.5 * inch, 1.9 * inch]),
              Spacer(1, 6),
              P("Consequences worth noting: a muscle is weakest at the ends of the range (f<sub>L</sub> never "
                "falls below 0.45 inside it) and weakens as it shortens faster; equal activation of both "
                "muscles gives no net torque but stiffens the joint — which the trained policy uses to hold "
                "still. There is deliberately no penalty on co-contraction."),
              PageBreak()]

    rules = [["rule", "a target is allowed only if …", "why"],
             ["1  reachable", "an elbow-down pose reaches it inside the range of motion", "otherwise impossible"],
             ["2  holdable", "holding it needs ≤ 80 % of each joint's torque limit",
              "otherwise the arm sags under gravity"],
             ["3  not degenerate", "it is at least 0.18 m inside the full-reach circle",
              "near full reach a straight arm resting on its elbow stop counted as 'in goal'"],
             ["4  distinct", "targets are > 0.30 m apart, and away from the start pose",
              "so a motionless arm can never satisfy two at once"],
             ["5  comfortable", "its holding pose clears every joint's end-range zone",
              "otherwise the reward charges for the only pose that holds it"]]
    story += [P("4.5 Which targets the arm can be asked to reach", "h2"),
              P("Random targets are drawn from a 2 cm grid, keeping only points that pass five rules. The "
                "result is the green region of Figure 4 — 744 points. When a user clicks a point that fails "
                "a rule, the interface moves it to the nearest point that passes."),
              table(rules, [1.3 * inch, 2.9 * inch, 2.8 * inch]),
              Spacer(1, 6),
              P("Rules 3 and 5 were added after failures: rule 3 when Stage B's arm learned to rest a straight "
                "arm on its elbow stop and be scored as holding, and rule 5 when targets needing an end-range "
                "pose proved measurably harder (75 % against 88 % success) for a reason the reward itself "
                "created."),
              PageBreak()]


def s5_rl(story):
    story += [P("5. The reinforcement-learning model", "h1"),
              P("5.1 The algorithm: Soft Actor-Critic", "h2"),
              P("SAC (Haarnoja et al. 2018) is an off-policy actor-critic method for continuous actions. "
                "Instead of maximising reward alone, it maximises reward <i>plus the entropy of the policy</i>, "
                "which keeps the policy exploring until the reward justifies committing to one way of moving:")]
    story += eqs([r"$J(\pi) \;=\; \sum_{t}\ \mathbb{E}\left[\ r(s_t, a_t) \;+\; \alpha\,\mathcal{H}(\pi(\cdot\,|\,s_t))\ \right]$"],
                 "sac_obj")
    story += [P("It suits this problem for three reasons: the action space is continuous (four activations); "
                "exploration matters, because the arm must discover strategies such as co-contracting to "
                "hold still; and being off-policy it reuses every past transition from a replay buffer, "
                "which makes it sample-efficient. SAC is also what Fischer et al. (2021) used to control a full "
                "biomechanical model of the upper extremity. The implementation here is Stable-Baselines3 2.7.1 "
                "(Raffin et al. 2021)."),
              P("5.2 The networks", "h2"),
              figure(DIAG / "d2_sac_networks.png",
                     "Figure 5. The actor maps the 18-number state to the mean and spread of a Gaussian over "
                     "the 4 activations. Two critics each estimate the value Q of a state-action pair; SAC "
                     "uses the smaller of the two to avoid over-estimating."),
              P("5.3 What is optimised", "h2"),
              P("Each gradient step samples a batch of 256 transitions (s, a, r, s') from the replay buffer and "
                "updates four things:")]
    story += eqs([r"$y \;=\; r + \gamma\,(1-d)\left(\ \min_{i=1,2} Q_{\bar\theta_i}(s', \tilde{a}') \;-\; \alpha\,\log \pi_\phi(\tilde{a}'\,|\,s')\ \right), \qquad \tilde{a}' \sim \pi_\phi(\cdot\,|\,s')$",
                  r"$\mathcal{L}_{Q} \;=\; \frac{1}{2}\sum_{i=1,2} \mathbb{E}\left[(Q_{\theta_i}(s,a) - y)^2\right] \qquad \mathrm{(critics)}$",
                  r"$\mathcal{L}_{\pi} \;=\; \mathbb{E}_s\left[\ \alpha\,\log\pi_\phi(\tilde{a}\,|\,s) \;-\; \min_{i} Q_{\theta_i}(s,\tilde{a})\ \right], \qquad \tilde{a} = \tanh(\mu_\phi(s) + \sigma_\phi(s)\,\epsilon) \qquad \mathrm{(actor)}$",
                  r"$\mathcal{L}_{\alpha} \;=\; -\,\mathbb{E}\left[\ \log\alpha\ (\log\pi_\phi(\tilde{a}\,|\,s) + \bar{\mathcal{H}})\ \right], \qquad \bar{\mathcal{H}} = -4 \qquad \mathrm{(temperature)}$",
                  r"$\bar\theta_i \;\leftarrow\; \tau\,\theta_i + (1-\tau)\,\bar\theta_i, \qquad \tau = 0.005 \qquad \mathrm{(target\ networks)}$"],
                 "sac_loss")
    story += [P("d marks a true termination. Training episodes never terminate on success — they are cut off "
                "at the 10 s time limit — so the critic keeps valuing the future, and the policy learns to go "
                "on holding the target rather than to stop once it arrives. The temperature α is tuned "
                "automatically toward a target entropy of −4 (minus the number of actions).")]

    hp = [["hyperparameter", "value", "hyperparameter", "value"],
          ["networks", "MLP 256 × 256, ReLU", "discount γ", "0.99 (≈ 2 s horizon)"],
          ["critics", "twin Q + target copies", "learning rate", "3 × 10<super>−4</super> (Adam)"],
          ["replay buffer", "300,000 transitions", "batch size", "256"],
          ["random warm-up", "5,000 steps", "target update τ", "0.005"],
          ["entropy α", "automatic, target −4", "updates per step", "1"],
          ["training length", "1,000,000 steps", "device", "CPU"],
          ["seeds", "5 per stage (1–5)", "episode", "500 steps = 10 s"]]
    story += [KeepTogether([P("5.4 Hyperparameters", "h2"),
                            table(hp, [1.3 * inch, 2.05 * inch, 1.35 * inch, 2.3 * inch])]),
              PageBreak(),
              P("5.5 The curriculum", "h2"),
              P("Asking for 5 cm precision and a half-second hold from the first step would leave the policy "
                "with almost no successes to learn from. Training therefore starts with a 15 cm tolerance and "
                "a short hold, and tightens only when the policy has earned it:"),
              figure(DIAG / "d4_curriculum.png", "Figure 6. The six rungs of the curriculum."),
              figure(FIGS / "fig03_curriculum_tolerance.png",
                     "Figure 7. When each seed tightened its tolerance. Every seed of every random-point "
                     "stage reached the final 5 cm rung.", max_h=2.6 * inch),
              P("<b>Choosing the model to keep.</b> Separately from the curriculum, every 20,000 steps the "
                "policy is also scored at the full production standard (5 cm, 0.5 s hold) on 10 fresh "
                "episodes. The best-scoring checkpoint is saved as the model; ties are broken by waypoints "
                "reached, then closeness, then speed."),
              PageBreak()]


def s6_reward(story, weights, terms):
    story += [P("6. The reward function", "h1"),
              P("The reward is computed after every 20 ms step. It is the sum of ten terms. Distances are "
                "measured as fractions of the arm's reach R = 0.65 m. Let d be the distance from the hand to "
                "the active target, and define the <b>potential</b>"),
              ]
    story += eqs([r"$\Phi(s) \;=\; -\,\frac{d}{R}$"], "phi")
    story += [P("so that Φ is higher (closer to zero) the nearer the hand is. The full reward is:")]
    lines = [
        ("shaping", r"$r_t \;=\;\ \ W_{\mathrm{shape}}\ (\gamma\,\Phi(s_{t+1}) - \Phi(s_t))$"),
        ("distance", r"$\qquad -\ W_{\mathrm{dist}}\ d_{t+1}/R$"),
        ("effort", r"$\qquad -\ W_{\mathrm{effort}}\ \|a_t\|^2$"),
        ("smoothness", r"$\qquad -\ W_{\mathrm{smooth}}\ \|\dot{q}_{t+1}\|\,/\,\omega_{\max}$"),
        ("joint limits", r"$\qquad -\ W_{\mathrm{limit}}\ (n_{\mathrm{stop}} + \sum_j e_j)$"),
        ("hand speed", r"$\qquad -\ (W_{\mathrm{lin}}\,\delta + W_{\mathrm{speed}}\,\min(\delta^2,\ C_{\mathrm{speed}})),"
                       r"\quad \delta = \max(0,\ \|v\| - v_{\mathrm{cap}})$"),
        ("hand acceleration", r"$\qquad -\ W_{\mathrm{accel}}\ \min((\|\dot{v}\|/A_{\mathrm{ref}})^2,\ C_{\mathrm{accel}})$"),
        ("hold stream", r"$\qquad +\ R_{\mathrm{goal}}\ \mathbb{1}[\mathrm{in\ goal\ at\ the\ final\ target}]$"),
        ("intermediate bonus", r"$\qquad +\ R_{\mathrm{inter}}\ \mathbb{1}[\mathrm{an\ intermediate\ target\ completed}]$"),
        ("success bonus", r"$\qquad +\ R_{\mathrm{succ}}\ \mathbb{1}[\mathrm{the\ final\ hold\ completed,\ first\ time}]$"),
    ]
    label_w = 1.35 * inch
    rows = [[eq(f"reward_{i}", tex, size=12, width=TEXT_W - label_w - 10), P(name, "small")]
            for i, (name, tex) in enumerate(lines)]
    rt = Table(rows, colWidths=[TEXT_W - label_w, label_w])
    rt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("LEFTPADDING", (0, 0), (0, -1), 14),
                            ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                            ("LINEBEFORE", (1, 0), (1, -1), 0.5, RULE)]))
    story += [rt, Spacer(1, 8)]
    story += [P("where a is the action, q̇ the joint velocities, ω<sub>max</sub> = 8 rad/s, n<sub>stop</sub> "
                "the number of joints on a hard stop, e<sub>j</sub> ∈ [0, 1] how far joint j has entered its "
                "10° end-range zone, v the hand velocity and v̇ its acceleration. γ = 0.99 is SAC's discount."),
              PageBreak()]

    r3 = weights.loc["R3"]
    why = [["term", "weight in R3", "what it does", "why it is there"],
           ["shaping", fmt_w(r3.W_SHAPING), "rewards moving toward the target, charges moving away",
            "a learning signal from the first update; potential-based (Ng et al. 1999), so it cannot change "
            "which behaviour is optimal"],
           ["distance", fmt_w(r3.W_DIST), "a steady cost for being far from the target",
            "keeps a pull toward the goal even when the hand is still"],
           ["effort", fmt_w(r3.W_EFFORT), "prefers smaller activations", "a metabolic-cost proxy, kept small"],
           ["smoothness", fmt_w(r3.W_SMOOTH), "discourages needless joint speed", "mild smoothing"],
           ["joint limits", fmt_w(r3.W_LIMIT), "charges for resting on a stop or in end range",
            "the first attempt rested a straight arm on its elbow stop"],
           ["speed", f"cap {fmt_w(r3.EE_SPEED_CAP)} m/s, quadratic {fmt_w(r3.W_SPEED_EXCESS)}",
            "charges for hand speed above the cap", "without it the hand whipped at 4–5 m/s"],
           ["acceleration", fmt_w(r3.W_ACCEL), "charges for sharp changes in hand velocity",
            "produces the bell-shaped speed profile of human reaching"],
           ["hold stream", f"+{fmt_w(r3.R_IN_GOAL)} per step", "pays for every step spent holding the final target",
            "pays for holding, not touching; nothing is paid at intermediate targets"],
           ["intermediate", f"+{fmt_w(r3.R_INTERMEDIATE)}", "once per intermediate target completed",
            "makes finishing a target worth leaving it"],
           ["success", f"+{fmt_w(r3.R_SUCCESS)}", "once, when the final hold completes", "marks the task done"]]
    story += [P("6.1 What each term is for", "h2"),
              table(why, [0.95 * inch, 1.35 * inch, 1.95 * inch, 2.75 * inch]),
              Spacer(1, 8),
              P("6.2 The reward the deployed model learned from", "h2"),
              P("Substituting the R3 weights (A<sub>ref</sub> = 10 m/s², C<sub>accel</sub> = 4; the linear speed "
                "term did not yet exist):")]
    story += eqs([r"$r_t = 2\,(0.99\,\Phi_{t+1} - \Phi_t) \;-\; 0.5\,\frac{d}{0.65} \;-\; 0.02\,\|a\|^2 \;-\; 0.05\,\frac{\|\dot{q}\|}{8} \;-\; 0.1\,(n_{\mathrm{stop}} + \sum_j e_j)$",
                  r"$\qquad -\; 0.5\,\max(0,\ \|v\| - 1.5)^2 \;-\; 0.02\,\min\!((\|\dot{v}\|/10)^2,\ 4) \;+\; 1\cdot\mathbb{1}[\ldots] \;+\; 10\cdot\mathbb{1}[\ldots] \;+\; 50\cdot\mathbb{1}[\ldots]$"],
                 "r3reward")
    story += [P("There is deliberately <b>no penalty on co-contraction</b>, because squeezing both muscles is "
                "how a muscle-driven arm stiffens to hold still — and the trained model does exactly that "
                "(co-contraction 0.48 while holding, against 0.23 while moving)."),
              PageBreak()]

    cols = [("W_SHAPING", "shape"), ("W_DIST", "dist"), ("W_EFFORT", "effort"), ("W_SMOOTH", "smooth"),
            ("W_LIMIT", "limit"), ("EE_SPEED_CAP", "v cap"), ("W_SPEED_EXCESS", "speed quad."),
            ("W_SPEED_LINEAR", "speed lin."), ("SPEED_PENALTY_CAP", "quad. cap"), ("W_ACCEL", "accel"),
            ("R_IN_GOAL", "hold"), ("R_INTERMEDIATE", "inter."), ("R_SUCCESS", "success")]
    rows = [["stage"] + [c[1] for c in cols]]
    for s in ("A", "B", "B3", "R3", "R4", "R5", "R6", "F2"):
        if s not in weights.index:
            continue
        r = weights.loc[s]
        vals = []
        for k, _ in cols:
            v = r.get(k, np.nan)
            if k in ("SPEED_PENALTY_CAP",) and pd.isna(v) and not pd.isna(r.get("W_SPEED_EXCESS")):
                vals.append("∞")
            else:
                vals.append(fmt_w(v))
        rows.append([f"<b>{s}</b>" if s == "R3" else s] + vals)
    story += [P("6.3 How the weights changed from stage to stage", "h2"),
              P("The reward was refined as each stage exposed a problem. The weights below are read from each "
                "trained model's own saved record, not from the current code. 'off' means the term did not "
                "exist when that model was trained."),
              table(rows, [w * inch for w in (0.42, 0.46, 0.42, 0.5, 0.56, 0.45, 0.5, 0.52, 0.5, 0.5, 0.5,
                                              0.42, 0.5, 0.6)], font_size=7.0),
              Spacer(1, 6),
              P("Stage A had no speed or acceleration cost; B added the quadratic speed cost; the acceleration "
                "cost arrived with B2 (a short attempt that was stopped) and stayed from B3 on; R5 lowered the "
                "speed cap to 1.2 m/s, raised the speed weight twenty-fold and capped the quadratic at 2.25; R6 "
                "added a linear speed term, because a purely quadratic cost has zero slope at the cap and so "
                "can never hold the hand's peak speed exactly at it."),
              P("6.4 Free-order variant", "h2"),
              P("In the free-order experiments (F2, F3) the policy sees all targets at once and picks the "
                "order itself. The potential then becomes the length of the shortest route from the hand "
                "through every target not yet visited, U, which rewards choosing an efficient order:")]
    story += eqs([r"$\Phi(s) \;=\; -\,\frac{1}{R}\ \min_{\sigma\ \in\ \mathrm{perm}(U)} \left(\ \|p - g_{\sigma_1}\| \;+\; \sum_i \|g_{\sigma_i} - g_{\sigma_{i+1}}\|\ \right)$"],
                 "tour")

    total = terms.reward.sum()
    fig8 = PILImage.open(FIGS / "fig10_reward_terms_breakdown.png").size
    h8 = min(3.4 * inch, TEXT_W * fig8[1] / fig8[0])
    story += [KeepTogether([
              P("6.5 One episode, term by term", "h2"),
              P(f"The deployed model on the three demonstration targets finishes in {len(terms)} steps "
                f"({len(terms) * 0.02:.1f} s) and earns {total:+.1f}. The bonuses for completing targets and "
                "holding the last one dominate; distance is the largest cost. A policy that does nothing for "
                "the whole 10 s scores about −355."),
              Image(str(FIGS / "fig10_reward_terms_breakdown.png"), width=h8 * fig8[0] / fig8[1], height=h8),
              P("Figure 8. Each reward term's total over one episode of the deployed model.", "caption")]),
              figure(FIGS / "fig11_reward_timeline.png",
                     "Figure 9. Reward per step through the same episode: +10 at each intermediate target, +50 "
                     "when the final hold completes, and +1 per step while holding.", max_h=2.9 * inch),
              PageBreak()]


def s7_results(story, finals):
    g = finals.groupby("stage").agg(seeds=("seed", "count"), mean=("success_rate", "mean"),
                                    sd=("success_rate", "std"),
                                    passed=("success_rate", lambda x: int((x >= 0.8).sum())))
    rows = [["stage", "task", "seeds ≥ 80 %", "success (mean ± sd over seeds)"]]
    task = {"A": "2 fixed targets", "B": "3 fixed targets (v1)", "B3": "3 fixed targets (v2)",
            "R3": "3 random targets (deployed)", "R4": "+ end-range rule*", "R5": "+ speed cost (quadratic)",
            "R6": "+ binding speed cost", "F2": "2 random targets, free order"}
    for s in ("A", "B", "B3", "R3", "R4", "R5", "R6", "F2"):
        if s in g.index:
            r = g.loc[s]
            rows.append([f"<b>{s}</b>" if s == "R3" else s, task[s], f"{int(r.passed)} / {int(r.seeds)}",
                         f"{r['mean']:.0%} ± {r.sd:.0%}"])
    story += [P("7. Training results", "h1"),
              figure(FIGS / "fig01_learning_curves_random_points.png",
                     "Figure 10. How success grew during training for the four random-target stages.",
                     max_h=3.6 * inch),
              table(rows, [0.6 * inch, 2.4 * inch, 1.1 * inch, 2.9 * inch]),
              Spacer(1, 4),
              P("Final evaluation: each seed's best model on 20 fresh episodes at 5 cm and a 0.5 s hold. "
                "*R4 onward are scored on a corrected target set; on identical episodes R3 and R4 are "
                "statistically indistinguishable (94.4 % vs 93.0 %, p = 0.67).", "small"),
              PageBreak(),
              figure(FIGS / "fig02_learning_curves_all_stages.png",
                     "Figure 11. Learning curves for every stage on the same scale. Blue is the mean of the "
                     "five seeds; grey lines are individual seeds.", max_h=5.1 * inch),
              figure(FIGS / "fig07_final_success.png",
                     "Figure 12. Final success of every seed of every stage.", max_h=2.9 * inch),
              PageBreak()]

    story += [P("7.1 A model that generalises, not one that memorises", "h2"),
              P("A model trained on fixed targets learns one trajectory, not how to reach: B3, perfect on its "
                "own three targets, fails on any others. Trained on targets re-drawn every episode, R3 has to "
                "use the target in its observation, and generalises. The best seed of each, on the same 20 "
                "random target sets:"),
              table([["model", "success", "targets reached", "final error"],
                     ["B3 (trained on 3 fixed targets)", "0 %", "0.25 of 3", "24.7 cm"],
                     ["<b>R3 (trained on random targets)</b>", "<b>100 %</b>", "<b>3.00 of 3</b>", "<b>2.1 cm</b>"]],
                    [2.9 * inch, 1.1 * inch, 1.5 * inch, 1.5 * inch]),
              P("7.2 Does it move like a person?", "h2"),
              P("Six measures from the motor-control literature, compared with their human ranges. R3 is "
                "accurate but reaches fast (about 2 m/s peak against a human 1.0–1.5 m/s) with slightly curved "
                "paths. Adding a speed cost (R5, R6) brought the speed into the human range, but at a cost in "
                "reliability — a genuine trade-off, shown in Figure 14."),
              figure(FIGS / "fig08_human_likeness.png",
                     "Figure 13. Human-likeness per stage. Dots are seeds, the bar is the median, the green band "
                     "the human range.", max_h=4.0 * inch),
              PageBreak(),
              figure(FIGS / "fig09_speed_vs_accuracy.png",
                     "Figure 14. Every attempt to slow the hand to human speed also lowered success.",
                     max_h=3.6 * inch),
              table([["best model", "peak hand speed", "human measures in range", "reaches all 3 clicked points"],
                     ["<b>R3 (deployed)</b>", "2.14 m/s", "4 of 6", "<b>38 / 40 = 95 %</b>"],
                     ["R5", "1.35 m/s", "6 of 6", "28 / 40 = 70 %"],
                     ["R6", "1.30 m/s", "5 of 6", "17 / 40 = 42 %"]],
                    [1.6 * inch, 1.4 * inch, 1.9 * inch, 2.1 * inch]),
              Spacer(1, 6),
              P("These are each stage's best seed. Across all five seeds, by median, R6 is inside 5 of the 6 human "
                "ranges, R4 and R5 inside 4 and R3 inside 3. R3 is deployed because it is the most reliable.",
                "small"),
              PageBreak(),
              P("7.3 Inside the learning algorithm", "h2"),
              figure(FIGS / "fig04_episode_return.png",
                     "Figure 15. Return per training episode rises throughout. R5 and R6 sit lower partly "
                     "because their speed cost is itself a charge.", max_h=3.3 * inch),
              figure(FIGS / "fig05_sac_internals_R3.png",
                     "Figure 16. SAC's internal quantities during R3's training. The entropy temperature α "
                     "crashes in the first 40,000 steps, recovers and settles near 0.04; the actor loss keeps "
                     "falling as the learned values rise.", max_h=4.2 * inch),
              PageBreak()]


def s8_lessons(story):
    items = [
        ("A straight arm resting on its elbow stop", "Stage A's first attempt stalled on every seed. A flexed "
         "resting pose, an end-range spring and a joint-limit cost fixed it."),
        ("A target too close to full reach", "Stage B's final target let a straight arm resting on its stop "
         "count as holding, a shortcut that could not be unlearned. Feasibility rule 3 now forbids such targets."),
        ("Advancing the curriculum too soon", "Seeds tightened a rung after one good round and collapsed. It now "
         "takes two consecutive good rounds."),
        ("Warm-starting from an easier task", "Starting three-target training from a two-target model was worse "
         "than training from scratch."),
        ("Memorising instead of reaching", "Every fixed-target model scored 0 % on new targets — which is why R3 "
         "trains on random ones."),
        ("A reward that punished progress", "In the first free-order run, finishing a target made the per-step "
         "reward worse, and the policy learned to reach one target and stop."),
        ("Retraining that bought nothing", "Retraining on the corrected target set (R4) was statistically no "
         "better than R3 (p = 0.67); the gain came from the rule itself."),
    ]
    story += [P("8. What did not work", "h1"),
              P("Negative results are part of the record: each shaped the final design.")]
    story += [table([["problem", "what happened and what fixed it"]] + [[a, b] for a, b in items],
                    [2.0 * inch, 5.0 * inch])]
    story += [Spacer(1, 10),
              P("9. Reproducing this work", "h1"),
              P("Every training run records its seed, library versions, configuration and reward weights; on "
                "the same machine with the same thread count, the same record reproduces bit-identical weights. "
                "The main commands:"),
              table([["command", "what it does"],
                     ["python run_campaign.py --stage R3", "train five seeds of a stage and pick the best"],
                     ["python scripts/verify_human_limits.py &lt;package&gt;",
                      "audit hard limits and human-likeness"],
                     ["python scripts/test_click_points.py &lt;package&gt; --anywhere",
                      "test on freely clicked points"],
                     ["python -m rl_armMotion.two_d.gui.app --package &lt;package&gt;", "the interactive demo"],
                     ["python scripts/build_report_pdf.py", "rebuild this document from the data"]],
                    [3.6 * inch, 3.4 * inch]),
              PageBreak(),
              P("References", "h1")]
    refs = [
        "Fischer, F., Bachinski, M., Klar, M., Fleig, A., &amp; Müller, J. (2021). Reinforcement learning control "
        "of a biomechanical model of the upper extremity. <i>Scientific Reports</i>, 11, 14445.",
        "Flash, T., &amp; Hogan, N. (1985). The coordination of arm movements: an experimentally confirmed "
        "mathematical model. <i>Journal of Neuroscience</i>, 5(7), 1688–1703.",
        "Haarnoja, T., Zhou, A., Abbeel, P., &amp; Levine, S. (2018). Soft Actor-Critic: off-policy maximum "
        "entropy deep reinforcement learning with a stochastic actor. <i>ICML</i>.",
        "Haarnoja, T., et al. (2018). Soft Actor-Critic algorithms and applications. arXiv:1812.05905.",
        "Hill, A. V. (1938). The heat of shortening and the dynamic constants of muscle. <i>Proceedings of the "
        "Royal Society B</i>, 126, 136–195.",
        "Hogan, N., &amp; Sternad, D. (2009). Sensitivity of smoothness measures to movement duration, amplitude, "
        "and arrests. <i>Journal of Motor Behavior</i>, 41(6), 529–534.",
        "Morasso, P. (1981). Spatial control of arm movements. <i>Experimental Brain Research</i>, 42, 223–227.",
        "Ng, A. Y., Harada, D., &amp; Russell, S. (1999). Policy invariance under reward transformations: theory "
        "and application to reward shaping. <i>ICML</i>.",
        "Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., &amp; Dormann, N. (2021). "
        "Stable-Baselines3: reliable reinforcement learning implementations. <i>JMLR</i>, 22(268), 1–8.",
        "Thelen, D. G. (2003). Adjustment of muscle mechanics model parameters to simulate dynamic contractions "
        "in older adults. <i>Journal of Biomechanical Engineering</i>, 125(1), 70–77.",
        "Winter, D. A. (2009). <i>Biomechanics and Motor Control of Human Movement</i> (4th ed.). Wiley.",
        "Zajac, F. E. (1989). Muscle and tendon: properties, models, scaling, and application to biomechanics "
        "and motor control. <i>Critical Reviews in Biomedical Engineering</i>, 17(4), 359–411.",
    ]
    ref_style = ParagraphStyle("ref", parent=S["small"], fontSize=8.8, leading=12.5, spaceAfter=6,
                               leftIndent=14, firstLineIndent=-14)
    story += [Paragraph(r, ref_style) for r in refs]


def main():
    timing, finals, weights, terms = load()
    story = []
    title_page(story, timing, finals)
    s1_summary(story, timing)
    s2_time(story, timing)
    s3_system(story)
    s4_constraints(story)
    s5_rl(story)
    s6_reward(story, weights, terms)
    s7_results(story, finals)
    s8_lessons(story)
    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=0.8 * inch, bottomMargin=0.75 * inch,
                            title="Teaching a simulated human arm to reach",
                            author="Ranjot Sandhu", subject="Reinforcement learning model: design, constraints, "
                                                            "reward and training")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
