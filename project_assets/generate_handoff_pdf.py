"""Generate the project handoff PDF for Windows PC transfer."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors

OUTPUT_PATH = "project_assets/Project_Handoff_Windows.pdf"

def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"], fontSize=22, spaceAfter=6,
        textColor=HexColor("#1a1a2e"),
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"], fontSize=16, spaceBefore=18,
        spaceAfter=8, textColor=HexColor("#16213e"),
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=13, spaceBefore=14,
        spaceAfter=6, textColor=HexColor("#0f3460"),
    )
    h3 = ParagraphStyle(
        "H3", parent=styles["Heading3"], fontSize=11, spaceBefore=10,
        spaceAfter=4, textColor=HexColor("#1a1a2e"),
    )
    body = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10, leading=14,
        spaceAfter=6,
    )
    code_style = ParagraphStyle(
        "Code", parent=styles["Code"], fontSize=8.5, leading=11,
        spaceAfter=6, backColor=HexColor("#f5f5f5"),
        borderColor=HexColor("#cccccc"), borderWidth=0.5,
        borderPadding=6, fontName="Courier",
    )
    bullet = ParagraphStyle(
        "Bullet", parent=body, leftIndent=20, bulletIndent=10,
        spaceBefore=2, spaceAfter=2,
    )
    small = ParagraphStyle(
        "Small", parent=body, fontSize=9, textColor=HexColor("#555555"),
    )

    story = []

    # ── TITLE PAGE ──
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("RL Arm Motion Project", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Windows PC Handoff Document", ParagraphStyle(
        "Subtitle", parent=styles["Heading2"], fontSize=16,
        textColor=HexColor("#555555"), alignment=TA_CENTER,
    )))
    story.append(Spacer(1, 24))
    story.append(HRFlowable(width="80%", color=HexColor("#0f3460")))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Student: Ranjot Sandhu", ParagraphStyle(
        "Meta", parent=body, alignment=TA_CENTER, fontSize=11,
    )))
    story.append(Paragraph("Date: June 21, 2026", ParagraphStyle(
        "Meta", parent=body, alignment=TA_CENTER, fontSize=11,
    )))
    story.append(Paragraph("Purpose: Multi-seed parallel training on Windows", ParagraphStyle(
        "Meta", parent=body, alignment=TA_CENTER, fontSize=11,
    )))
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph(
        "<b>This document contains everything needed to continue the project on a "
        "Windows PC, including: project state, best model details, what to run, "
        "and what has been tried.</b>", body
    ))
    story.append(PageBreak())

    # ── TABLE OF CONTENTS ──
    story.append(Paragraph("Table of Contents", h1))
    toc_items = [
        "1. Project Summary and Current State",
        "2. The Goal: Replicate fischer_waypoints_v2_500k",
        "3. Best Model Specification",
        "4. Environment Configuration (Critical)",
        "5. What Has Been Tried (Don't Repeat These)",
        "6. Multi-Seed Training Strategy",
        "7. Windows Setup Instructions",
        "8. Training Command and Scripts",
        "9. Evaluation Script",
        "10. File Manifest",
        "11. Key Lessons and Warnings",
    ]
    for item in toc_items:
        story.append(Paragraph(item, bullet))
    story.append(PageBreak())

    # ── SECTION 1: PROJECT SUMMARY ──
    story.append(Paragraph("1. Project Summary and Current State", h1))
    story.append(Paragraph(
        "This is a reinforcement learning project that trains a 2-DOF planar robotic arm "
        "to navigate through a sequence of 3 waypoints using Hill-type muscle actuation "
        "(Fischer et al., 2021 protocol). The arm has a fixed shoulder at workspace "
        "coordinate [1.0, 0.0] m with link lengths 1.0 m + 0.8 m.", body
    ))
    story.append(Paragraph("<b>Current Status:</b>", body))
    story.append(Paragraph(
        "The best model (<font face='Courier'>fischer_waypoints_v2_500k</font>) achieves "
        "3/3 waypoints in 181 steps at 0.6m tolerance with 100% deterministic consistency. "
        "However, this model was produced by a single lucky random seed. All attempts to "
        "retrain or improve it have failed. The next step is to run many parallel seeds on "
        "Windows to replicate (or beat) this result.", body
    ))

    story.append(Paragraph("<b>What needs to happen on Windows:</b>", body))
    for item in [
        "Run 20-50 parallel SAC training seeds (500K steps each)",
        "Auto-evaluate each seed at 0.6m, 0.4m, 0.2m tolerance",
        "Pick the best seed that achieves 3/3 waypoints",
        "The env changes (12 deg orientation, hold grace -5) are already in the code",
    ]:
        story.append(Paragraph(f"• {item}", bullet))

    # ── SECTION 2: THE GOAL ──
    story.append(Paragraph("2. The Goal: Replicate fischer_waypoints_v2_500k", h1))
    story.append(Paragraph(
        "The benchmark to beat or match:", body
    ))
    data = [
        ["Metric", "Target Value"],
        ["Waypoints completed", "3/3"],
        ["Waypoint sequence", "[2.3,-1.0], [1.0,1.5], [2.5,0.0]"],
        ["Steps to complete", "181 (at 0.6m tolerance)"],
        ["Steps at 0.4m", "206"],
        ["Consistency", "5/5 identical runs"],
        ["Algorithm", "SAC (Soft Actor-Critic)"],
        ["Actuation", "Hill-type muscle (4D action space)"],
        ["Training steps", "500,000"],
    ]
    t = Table(data, colWidths=[2.2 * inch, 4.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f5f5f5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Success probability per seed:</b> Based on 8 training attempts, roughly 1 in 8 "
        "seeds achieves 3/3. Running 20+ seeds in parallel should yield at least 2-3 winners.", body
    ))
    story.append(PageBreak())

    # ── SECTION 3: BEST MODEL SPEC ──
    story.append(Paragraph("3. Best Model Specification", h1))

    story.append(Paragraph("3.1 Architecture", h2))
    arch_data = [
        ["Parameter", "Value"],
        ["Policy class", "SACPolicy (MlpPolicy)"],
        ["Network", "[256, 256] (2 hidden layers)"],
        ["Observation dim", "11"],
        ["Action dim", "4 (muscle activations [0,1])"],
        ["Learning rate", "3e-4"],
        ["Replay buffer", "200,000"],
        ["Batch size", "256"],
        ["Tau (soft update)", "0.005"],
        ["Gamma (discount)", "0.99"],
        ["Entropy coef", "auto (learned)"],
    ]
    t = Table(arch_data, colWidths=[2.2 * inch, 4.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f5f5f5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)

    story.append(Paragraph("3.2 Observation Space (11D)", h2))
    obs_data = [
        ["Index", "Component", "Range"],
        ["0", "sin(shoulder_angle)", "[-1, 1]"],
        ["1", "cos(shoulder_angle)", "[-1, 1]"],
        ["2", "sin(elbow_angle)", "[-1, 1]"],
        ["3", "cos(elbow_angle)", "[-1, 1]"],
        ["4", "shoulder_velocity_norm", "[-1, 1]"],
        ["5", "elbow_velocity_norm", "[-1, 1]"],
        ["6", "signed_height_error", "[-1.8, 1.8]"],
        ["7", "signed_orientation_error", "[-pi, pi]"],
        ["8", "gradient_norm", "[0, 1]"],
        ["9", "in_goal_region (binary)", "{0, 1}"],
        ["10", "hold_progress", "[0, 1]"],
    ]
    t = Table(obs_data, colWidths=[0.6 * inch, 2.8 * inch, 3.3 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f5f5f5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)

    story.append(Paragraph("3.3 Action Space (4D, Hill-type Muscles)", h2))
    story.append(Paragraph(
        "Each joint has an extensor and flexor muscle with activation in [0,1]. "
        "Net torque = max_torque * (extensor - flexor). Co-contraction "
        "(simultaneous activation) is penalized.", body
    ))
    act_data = [
        ["Index", "Component", "Range"],
        ["0", "shoulder_extensor", "[0, 1]"],
        ["1", "shoulder_flexor", "[0, 1]"],
        ["2", "elbow_extensor", "[0, 1]"],
        ["3", "elbow_flexor", "[0, 1]"],
    ]
    t = Table(act_data, colWidths=[0.6 * inch, 2.8 * inch, 3.3 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f5f5f5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)

    story.append(Paragraph("3.4 Reward Function", h2))
    reward_data = [
        ["Label", "Weight", "Component", "Purpose"],
        ["P1", "-2.0", "goal_distance", "Primary distance signal"],
        ["P2", "-1.0", "orientation_error", "Alignment penalty"],
        ["P3", "-0.15", "velocity_norm", "Smoothness"],
        ["P4", "-0.20", "gradient_norm", "Anti-overshoot"],
        ["P5", "-0.01", "action_norm", "Effort cost"],
        ["P7", "-0.08", "co-contraction", "Reduce muscle tremor"],
        ["B1", "+1.5", "progress", "Potential-based shaping"],
        ["B2", "up to +4.0", "proximity", "Convex near goal"],
        ["B3", "+10.0", "in_goal_region", "Stay-in-goal constant"],
        ["B4", "+2.0/step", "hold_counter", "Hold-growth bonus"],
        ["--", "+25.0", "waypoint touch", "Transition bonus"],
        ["--", "+150.0", "hold complete", "Terminal success"],
    ]
    t = Table(reward_data, colWidths=[0.5 * inch, 1.0 * inch, 1.8 * inch, 3.4 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f5f5f5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ── SECTION 4: ENVIRONMENT CONFIG ──
    story.append(Paragraph("4. Environment Configuration (Critical)", h1))
    story.append(Paragraph(
        "<b>These settings are already in the code.</b> Do NOT change them. They are in "
        "<font face='Courier'>src/rl_armMotion/two_d/environments/task_env.py</font>.", body
    ))

    story.append(Paragraph("4.1 Arm Properties", h2))
    arm_data = [
        ["Parameter", "Value"],
        ["DOF", "2 (shoulder + elbow)"],
        ["Link lengths", "[1.0, 0.8] meters"],
        ["Link masses", "[2.0, 1.5] kg"],
        ["Shoulder position", "[1.0, 0.0] m from workspace origin"],
        ["Shoulder limits", "[-180, 180] degrees"],
        ["Elbow limits", "[0, 120] degrees"],
        ["dt", "0.02 seconds"],
        ["Damping", "0.5"],
    ]
    t = Table(arm_data, colWidths=[2.2 * inch, 4.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f5f5f5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)

    story.append(Paragraph("4.2 Environment Improvements (Already Applied)", h2))
    story.append(Paragraph(
        "These three changes are already in task_env.py. They improve hold stability "
        "at evaluation time:", body
    ))
    for item in [
        "<b>Orientation tolerance: 12 degrees</b> (was 10 deg). Line ~151: "
        "<font face='Courier'>DEFAULT_ORIENTATION_TOLERANCE_DEG: float = 12.0</font>",
        "<b>Hold grace: decrement by 5</b> (was hard reset to 0). Line ~799: "
        "<font face='Courier'>self.hold_counter = max(0, self.hold_counter - 5)</font>",
        "<b>ActionSmoother class</b> (line ~86): EMA wrapper for smooth muscle commands. "
        "Used at evaluation/visualization time, NOT during training.",
    ]:
        story.append(Paragraph(f"• {item}", bullet))

    story.append(Paragraph("4.3 Hold Criterion", h2))
    story.append(Paragraph(
        "The arm must hold position for 20 consecutive steps within ALL of:", body
    ))
    for item in [
        "Position: within tolerance radius (0.6m default)",
        "Orientation: within 12 degrees of target",
        "Velocity: below 0.3 rad/s",
    ]:
        story.append(Paragraph(f"• {item}", bullet))
    story.append(Paragraph(
        "Intermediate waypoints use touch-and-go (position only). The final waypoint "
        "requires the full hold criterion.", body
    ))
    story.append(PageBreak())

    # ── SECTION 5: WHAT HAS BEEN TRIED ──
    story.append(Paragraph("5. What Has Been Tried (Don't Repeat These)", h1))
    story.append(Paragraph(
        "<b>All of these produced worse results than the original model. "
        "Do not attempt them again.</b>", body
    ))

    tried_data = [
        ["Approach", "Result", "Why It Failed"],
        ["Fine-tune from v2 (lr=3e-4)", "2/3 waypoints", "Catastrophic forgetting"],
        ["Fine-tune from v2 (lr=1e-5)", "Score 60 vs orig 181", "Still degraded policy"],
        ["V3 training (EMA vel, eval curriculum)", "2/3 waypoints", "Eval curriculum didn't help"],
        ["Retrain with env changes", "1/3 waypoints", "Different seed, worse outcome"],
        ["5 multi-seed search", "Best: 2/3", "None matched 3/3 (variance)"],
        ["1M extended training", "No improvement", "More steps != better"],
        ["Jerk penalty experiments", "Slower convergence", "Extra penalty hurt learning"],
    ]
    t = Table(tried_data, colWidths=[2.0 * inch, 1.5 * inch, 3.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#8B0000")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#fff0f0")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Key lessons:</b>", body))
    for item in [
        "Never fine-tune a working model - SAC exploration noise destroys delicate hold behavior",
        "Curriculum advancement is not needed - the shaped reward (P1-B4) is sufficient",
        "More training steps do not help - 500K is the sweet spot",
        "The ONLY way to get 3/3 is to find the right random seed",
    ]:
        story.append(Paragraph(f"• {item}", bullet))

    # ── SECTION 6: MULTI-SEED STRATEGY ──
    story.append(Paragraph("6. Multi-Seed Training Strategy", h1))
    story.append(Paragraph(
        "The strategy is simple: run the exact same training command with different random "
        "seeds in parallel, then evaluate all of them and pick the winner.", body
    ))

    story.append(Paragraph("6.1 Recommended Approach", h2))
    for i, item in enumerate([
        "Run 20-50 seeds in parallel (each is independent, no GPU needed)",
        "Each seed trains for 500K steps (~35 min on single core, faster with parallelism)",
        "After all training completes, run the evaluation script on every seed",
        "Pick any seed that achieves 3/3 waypoints at 0.6m tolerance",
        "Bonus: if multiple seeds hit 3/3, pick the one with fewest steps",
    ], 1):
        story.append(Paragraph(f"{i}. {item}", bullet))

    story.append(Paragraph("6.2 Expected Outcomes", h2))
    story.append(Paragraph(
        "Based on prior experiments (~1 in 8 seeds succeeds):", body
    ))
    seed_data = [
        ["Seeds Run", "Expected 3/3 Winners", "Confidence"],
        ["10", "1-2", "Moderate"],
        ["20", "2-3", "Good"],
        ["30", "3-4", "High"],
        ["50", "5-7", "Very high"],
    ]
    t = Table(seed_data, colWidths=[1.5 * inch, 2.5 * inch, 2.7 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f5f5f5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ── SECTION 7: WINDOWS SETUP ──
    story.append(Paragraph("7. Windows Setup Instructions", h1))

    story.append(Paragraph("7.1 Prerequisites", h2))
    for item in [
        "Python 3.10+ installed",
        "Git installed",
        "Project repository cloned or copied to Windows PC",
    ]:
        story.append(Paragraph(f"• {item}", bullet))

    story.append(Paragraph("7.2 Environment Setup", h2))
    story.append(Paragraph(
        "Run these commands in PowerShell or Command Prompt:", body
    ))
    setup_code = (
        "cd C:\\path\\to\\Project\n"
        "python -m venv venv\n"
        "venv\\Scripts\\activate\n"
        "pip install -r requirements.txt\n"
        "pip install -e .\n"
        "pip install stable-baselines3[extra] torch"
    )
    story.append(Paragraph(setup_code.replace("\n", "<br/>"), code_style))

    story.append(Paragraph("7.3 Verify Installation", h2))
    verify_code = (
        "python -c \"import stable_baselines3; print('SB3 OK')\"\n"
        "python -c \"from rl_armMotion.two_d.environments.task_env import ArmTaskEnv; print('Env OK')\"\n"
        "python -c \"from rl_armMotion.two_d.training.ppo_trainer_wrapper import RLTrainerWithMetrics; print('Trainer OK')\""
    )
    story.append(Paragraph(verify_code.replace("\n", "<br/>"), code_style))

    # ── SECTION 8: TRAINING COMMANDS ──
    story.append(Paragraph("8. Training Command and Scripts", h1))

    story.append(Paragraph("8.1 Single Seed Training Command", h2))
    story.append(Paragraph(
        "This is the exact command that produced the best model. Run it as-is:", body
    ))
    train_cmd = (
        "python scripts/train_fischer_session.py ^\n"
        "    --timesteps 500000 ^\n"
        "    --actuation-mode muscle ^\n"
        "    --goal-direction EAST ^\n"
        "    --waypoints \"2.3,-1.0;1.0,1.5;2.5,0.0\" ^\n"
        "    --save-dir ./project_assets/outputs/seed_run_1"
    )
    story.append(Paragraph(train_cmd.replace("\n", "<br/>"), code_style))

    story.append(Paragraph("8.2 Parallel Multi-Seed Script (Windows Batch)", h2))
    story.append(Paragraph(
        "Save this as <font face='Courier'>run_seeds.bat</font> in the project root:", body
    ))
    bat_script = (
        '@echo off\n'
        'setlocal\n'
        'set VENV=venv\\Scripts\\python\n'
        'set SCRIPT=scripts\\train_fischer_session.py\n'
        'set STEPS=500000\n'
        'set WP=2.3,-1.0;1.0,1.5;2.5,0.0\n'
        '\n'
        'for /L %%i in (1,1,20) do (\n'
        '    echo Starting seed %%i...\n'
        '    start /B %VENV% %SCRIPT% ^\n'
        '        --timesteps %STEPS% ^\n'
        '        --actuation-mode muscle ^\n'
        '        --goal-direction EAST ^\n'
        '        --waypoints "%WP%" ^\n'
        '        --save-dir .\\project_assets\\outputs\\seed_%%i\n'
        ')\n'
        'echo All 20 seeds launched.\n'
        'echo Monitor progress: dir project_assets\\outputs\\seed_*\\training_log.txt'
    )
    story.append(Paragraph(bat_script.replace("\n", "<br/>").replace("  ", "&nbsp;&nbsp;"), code_style))
    story.append(Paragraph(
        "<b>Note:</b> Adjust the range (1,1,20) to run more or fewer seeds. "
        "Each seed runs on a single CPU core. On a machine with N cores, "
        "run at most N-2 seeds in parallel to keep the system responsive.", small
    ))

    story.append(Paragraph("8.3 Alternative: Python Parallel Launcher", h2))
    story.append(Paragraph(
        "Save as <font face='Courier'>run_parallel_seeds.py</font>:", body
    ))
    py_launcher = (
        'import subprocess, sys, os\n'
        'from pathlib import Path\n'
        'from concurrent.futures import ProcessPoolExecutor\n'
        '\n'
        'NUM_SEEDS = 20\n'
        'TIMESTEPS = 500_000\n'
        'WP = "2.3,-1.0;1.0,1.5;2.5,0.0"\n'
        'PYTHON = sys.executable\n'
        'SCRIPT = str(Path("scripts/train_fischer_session.py"))\n'
        '\n'
        'def run_seed(seed_id):\n'
        '    save_dir = f"./project_assets/outputs/seed_{seed_id}"\n'
        '    cmd = [\n'
        '        PYTHON, SCRIPT,\n'
        '        "--timesteps", str(TIMESTEPS),\n'
        '        "--actuation-mode", "muscle",\n'
        '        "--goal-direction", "EAST",\n'
        '        "--waypoints", WP,\n'
        '        "--save-dir", save_dir,\n'
        '    ]\n'
        '    print(f"Launching seed {seed_id}...")\n'
        '    result = subprocess.run(cmd, capture_output=True, text=True)\n'
        '    status = "OK" if result.returncode == 0 else "FAIL"\n'
        '    print(f"Seed {seed_id}: {status}")\n'
        '    return seed_id, result.returncode\n'
        '\n'
        'if __name__ == "__main__":\n'
        '    max_workers = min(NUM_SEEDS, os.cpu_count() - 2)\n'
        '    print(f"Running {NUM_SEEDS} seeds with {max_workers} workers")\n'
        '    with ProcessPoolExecutor(max_workers=max_workers) as ex:\n'
        '        results = list(ex.map(run_seed, range(1, NUM_SEEDS + 1)))\n'
        '    ok = sum(1 for _, rc in results if rc == 0)\n'
        '    print(f"\\nDone: {ok}/{NUM_SEEDS} completed successfully")\n'
        '    print("Run: python scripts/pick_best_seed.py")'
    )
    story.append(Paragraph(py_launcher.replace("\n", "<br/>").replace("  ", "&nbsp;&nbsp;"), code_style))
    story.append(PageBreak())

    # ── SECTION 9: EVALUATION SCRIPT ──
    story.append(Paragraph("9. Evaluation Script", h1))
    story.append(Paragraph(
        "After all seeds complete, run the existing evaluation script "
        "(<font face='Courier'>scripts/pick_best_seed.py</font>). "
        "It evaluates each seed at 0.6m, 0.4m, and 0.2m tolerance and ranks them.", body
    ))
    story.append(Paragraph(
        "<b>Important:</b> Update the glob pattern in pick_best_seed.py to match your "
        "output directory naming. The current script looks for "
        "<font face='Courier'>multiseed_run_*</font>; change it to "
        "<font face='Courier'>seed_*</font> if using the naming above.", body
    ))
    eval_cmd = "python scripts/pick_best_seed.py"
    story.append(Paragraph(eval_cmd, code_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "The script outputs a ranked table showing waypoint completion and a composite "
        "score. Copy the best seed's <font face='Courier'>sac_model.zip</font> as the "
        "final model.", body
    ))

    story.append(Paragraph("9.1 Quick Manual Evaluation", h2))
    story.append(Paragraph("To evaluate a single model manually:", body))
    manual_eval = (
        'python -c "\n'
        'import sys; sys.path.insert(0, \'src\')\n'
        'from stable_baselines3 import SAC\n'
        'from rl_armMotion.two_d.environments.task_env import ArmTaskEnv\n'
        'model = SAC.load(\'project_assets/outputs/seed_1/sac_model\')\n'
        'env = ArmTaskEnv(actuation_mode=\'muscle\', goal_direction=\'EAST\')\n'
        'env.set_waypoints([[2.3,-1.0],[1.0,1.5],[2.5,0.0]], tolerance=0.6)\n'
        'obs, _ = env.reset()\n'
        'for step in range(800):\n'
        '    action, _ = model.predict(obs, deterministic=True)\n'
        '    obs, r, term, trunc, info = env.step(action)\n'
        '    if term: print(f\'3/3 in {step} steps\'); break\n'
        'else: print(f\'{info[\"current_waypoint_index\"]}/3 stuck\')\n'
        '"'
    )
    story.append(Paragraph(manual_eval.replace("\n", "<br/>").replace("  ", "&nbsp;&nbsp;"), code_style))
    story.append(PageBreak())

    # ── SECTION 10: FILE MANIFEST ──
    story.append(Paragraph("10. File Manifest", h1))
    story.append(Paragraph("Critical files to transfer to Windows:", body))

    files_data = [
        ["Path", "Purpose"],
        ["src/rl_armMotion/", "Main package (all source code)"],
        ["scripts/train_fischer_session.py", "Training script (THE one to use)"],
        ["scripts/pick_best_seed.py", "Multi-seed evaluation + ranking"],
        ["scripts/evaluate_trained_model.py", "Detailed model evaluation"],
        ["requirements.txt", "Python dependencies"],
        ["pyproject.toml", "Package configuration"],
        ["project_assets/outputs/fischer_waypoints_v2_500k/", "BEST MODEL (reference)"],
        ["project_assets/Final_Model_Report.md", "Model documentation"],
        ["project_assets/Work_Rundown_Since_May_1.md", "Full work history"],
        ["tests/", "Unit tests (52 tests)"],
        ["CLAUDE.md", "Project instructions and history"],
    ]
    t = Table(files_data, colWidths=[3.5 * inch, 3.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f5f5f5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Do NOT transfer:</b> __pycache__ directories, .claude/ worktrees, "
        "venv/ (recreate on Windows), .git/ (optional, can re-clone).", small
    ))

    # ── SECTION 11: WARNINGS ──
    story.append(Paragraph("11. Key Lessons and Warnings", h1))

    story.append(Paragraph("11.1 Do NOT:", h2))
    for item in [
        "<b>Fine-tune the original model</b> - any continued training destroys it",
        "<b>Change the reward weights</b> - they are carefully tuned",
        "<b>Change the observation space</b> - the policy depends on the 11D structure",
        "<b>Add the ActionSmoother during training</b> - only use it at evaluation/visualization time",
        "<b>Increase training beyond 500K steps</b> - more steps don't help, 500K is optimal",
        "<b>Use PPO or A2C</b> - SAC is the only algorithm that worked for this task",
        "<b>Change the waypoint order or coordinates</b> - the benchmark uses exactly [2.3,-1.0], [1.0,1.5], [2.5,0.0]",
    ]:
        story.append(Paragraph(f"• {item}", bullet))

    story.append(Paragraph("11.2 DO:", h2))
    for item in [
        "Run many seeds (20+) in parallel",
        "Use the exact same training command as the original",
        "Evaluate at multiple tolerances (0.6m, 0.4m, 0.2m)",
        "Keep the original model as reference - never overwrite it",
        "Check consistency (run each winner 5 times to confirm determinism)",
    ]:
        story.append(Paragraph(f"• {item}", bullet))

    story.append(Paragraph("11.3 Security Constraints", h2))
    for item in [
        "Do NOT merge claude/fischer-integration branch to main",
        "Do NOT update the CP493 progress report PDF",
        "Save point save-point-2026-05-05 must remain recoverable",
        'Do NOT change the file "arm_tl_flowchart.html"',
    ]:
        story.append(Paragraph(f"• {item}", bullet))

    story.append(Spacer(1, 24))
    story.append(HRFlowable(width="100%", color=HexColor("#0f3460")))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<i>Document generated June 21, 2026. Transfer the entire Project/ directory "
        "to your Windows PC, set up the venv, run the multi-seed script, and evaluate. "
        "Good luck finding a 3/3 seed!</i>", small
    ))

    doc.build(story)
    print(f"PDF written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
