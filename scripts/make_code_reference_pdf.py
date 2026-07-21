"""Generate a Code Reference PDF documenting every source file in the project."""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

OUT = Path(__file__).parent.parent / "docs" / "CODE_REFERENCE.pdf"

# ── Colours ───────────────────────────────────────────────────────────────────
BRAND_BLUE  = colors.HexColor("#1a3a5c")
BRAND_GOLD  = colors.HexColor("#c8a951")
SECTION_BG  = colors.HexColor("#e8eef4")
LIGHT_GRAY  = colors.HexColor("#f7f7f7")
MID_GRAY    = colors.HexColor("#cccccc")
DARK_GRAY   = colors.HexColor("#555555")
CODE_BG     = colors.HexColor("#f0f0f0")
GREEN       = colors.HexColor("#2d6a2d")
WHITE       = colors.white
TAG_BLUE    = colors.HexColor("#dce8f5")

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

title_s = ParagraphStyle("DocTitle", fontSize=24, leading=30,
    textColor=BRAND_BLUE, fontName="Helvetica-Bold",
    spaceAfter=4, alignment=TA_CENTER)
subtitle_s = ParagraphStyle("DocSub", fontSize=11, leading=15,
    textColor=DARK_GRAY, fontName="Helvetica",
    spaceAfter=2, alignment=TA_CENTER)
section_s = ParagraphStyle("Section", fontSize=13, leading=18,
    textColor=WHITE, fontName="Helvetica-Bold",
    spaceBefore=14, spaceAfter=6, backColor=BRAND_BLUE, borderPad=6)
file_header_s = ParagraphStyle("FileHeader", fontSize=11, leading=15,
    textColor=BRAND_BLUE, fontName="Helvetica-Bold",
    spaceBefore=10, spaceAfter=2)
filepath_s = ParagraphStyle("FilePath", fontSize=8.5, leading=12,
    textColor=DARK_GRAY, fontName="Courier",
    spaceAfter=4, backColor=CODE_BG, borderPad=3)
body_s = ParagraphStyle("Body", fontSize=9.5, leading=14,
    textColor=colors.HexColor("#222222"), fontName="Helvetica",
    spaceAfter=4)
bullet_s = ParagraphStyle("Bullet", parent=body_s,
    leftIndent=14, firstLineIndent=-10, spaceAfter=2)
label_s = ParagraphStyle("Label", fontSize=8.5, leading=12,
    textColor=DARK_GRAY, fontName="Helvetica-Bold", spaceAfter=1)
code_s = ParagraphStyle("Code", fontSize=8, leading=11,
    textColor=colors.HexColor("#333"), fontName="Courier",
    backColor=CODE_BG, leftIndent=10, rightIndent=10,
    spaceBefore=2, spaceAfter=4, borderPad=3)
toc_s = ParagraphStyle("TOC", fontSize=10, leading=14,
    textColor=BRAND_BLUE, fontName="Helvetica", spaceAfter=3,
    leftIndent=12)
toc_section_s = ParagraphStyle("TOCSec", fontSize=11, leading=16,
    textColor=BRAND_BLUE, fontName="Helvetica-Bold", spaceAfter=2,
    spaceBefore=6)


def hr(color=BRAND_GOLD, thickness=1.5):
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceBefore=4, spaceAfter=4)


def mini_table(rows, col_widths=None):
    if col_widths is None:
        col_widths = [1.5*inch, 5.5*inch]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("FONTNAME",  (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",  (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE",  (0,0), (-1,-1), 8.5),
        ("VALIGN",    (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",(0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("LEFTPADDING",(0,0),(-1,-1), 4),
        ("RIGHTPADDING",(0,0),(-1,-1), 4),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[LIGHT_GRAY, WHITE]),
        ("GRID",(0,0),(-1,-1),0.3,MID_GRAY),
    ]))
    return t


def classes_table(entries):
    """entries = list of (class_name, description)"""
    data = [["Class / Function", "What it does"]]
    for name, desc in entries:
        data.append([name, desc])
    t = Table(data, colWidths=[2.2*inch, 4.8*inch], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0), BRAND_BLUE),
        ("TEXTCOLOR",    (0,0),(-1,0), WHITE),
        ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0),(-1,0), 8.5),
        ("FONTNAME",     (0,1),(-1,-1),"Helvetica"),
        ("FONTSIZE",     (0,1),(-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, LIGHT_GRAY]),
        ("GRID",         (0,0),(-1,-1), 0.3, MID_GRAY),
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ("TOPPADDING",   (0,0),(-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("LEFTPADDING",  (0,0),(-1,-1), 5),
        ("RIGHTPADDING", (0,0),(-1,-1), 5),
    ]))
    return t


def file_block(title, path, lines, purpose, what_it_does, classes, inputs_outputs=None, notes=None):
    """Build a KeepTogether block for one file."""
    elems = []
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(title, file_header_s))
    elems.append(Paragraph(path, filepath_s))
    elems.append(Paragraph(f"<b>Lines of code:</b> {lines}", label_s))
    elems.append(Spacer(1, 2))
    elems.append(Paragraph("<b>Purpose</b>", label_s))
    elems.append(Paragraph(purpose, body_s))
    elems.append(Paragraph("<b>What it does</b>", label_s))
    for point in what_it_does:
        elems.append(Paragraph(f"• {point}", bullet_s))
    if inputs_outputs:
        elems.append(Spacer(1, 2))
        elems.append(Paragraph("<b>Inputs / Outputs</b>", label_s))
        elems.append(mini_table(inputs_outputs))
    if classes:
        elems.append(Spacer(1, 2))
        elems.append(Paragraph("<b>Key Classes &amp; Functions</b>", label_s))
        elems.append(classes_table(classes))
    if notes:
        elems.append(Spacer(1, 2))
        elems.append(Paragraph(f"<i>Note: {notes}</i>",
            ParagraphStyle("Note", parent=body_s, textColor=DARK_GRAY, fontSize=8.5)))
    elems.append(hr(MID_GRAY, 0.4))
    return elems


# ══════════════════════════════════════════════════════════════════════════════
# FILE CATALOGUE
# ══════════════════════════════════════════════════════════════════════════════

CATALOGUE = [

    # ── SECTION 1: Configuration ─────────────────────────────────────────────
    {"_section": "1. Configuration"},

    {"title": "ArmConfiguration",
     "path": "src/rl_armMotion/two_d/config/arm_config.py",
     "lines": 246,
     "purpose": "Stores every physical parameter of the arm in one place so nothing is hardcoded. Think of it as the 'blueprint' file — link lengths, masses, joint limits, damping, initial angles.",
     "what": [
         "Defines the <b>ArmConfiguration</b> dataclass with all arm physical properties.",
         "Provides <b>save/load</b> methods so configurations can be written to JSON and re-loaded.",
         "Sets the defaults: 2 DOF, link lengths [1.0, 0.8] m, shoulder at [-90°, 0°] home pose.",
         "Used by every other module — the environment, GUIs, and kinematics all read from this.",
     ],
     "classes": [
         ("ArmConfiguration", "Dataclass holding all arm parameters (link lengths, masses, joint limits, damping, initial angles). Default values match the research arm."),
     ],
     "io": None,
     "notes": "Changing values here affects the entire system. Never change during a training run."},

    # ── SECTION 2: Environments ───────────────────────────────────────────────
    {"_section": "2. Environments"},

    {"title": "ArmTaskEnv  (core environment)",
     "path": "src/rl_armMotion/two_d/environments/task_env.py",
     "lines": 926,
     "purpose": "This is the most important file in the project. It is the Gymnasium-compatible environment that the SAC agent interacts with during training. It defines the physics, the reward, the observation, and the success condition.",
     "what": [
         "<b>Physics step:</b> updates joint angles and velocities each timestep — either via velocity commands (2D) or Hill-type muscle forces (4D).",
         "<b>Observation (11D):</b> [sin θ₀, cos θ₀, sin θ₁, cos θ₁, ω₀, ω₁, height_error, orientation_error, gradient_norm, in_goal_region, hold_progress].",
         "<b>10-term shaped reward:</b> approach reward, orientation reward, velocity penalty, smoothness penalty, gradient bonus, in-zone bonus, hold-growth bonus, waypoint advance bonus, waypoint touch bonus, terminal success bonus.",
         "<b>Waypoint navigation:</b> agent visits waypoints in order — intermediate ones advance on touch, the final one requires a 20-step hold.",
         "<b>Hold criterion:</b> position within tolerance AND orientation within 12° AND velocity &lt; 0.3 rad/s for hold_steps_required consecutive steps.",
         "<b>Adaptive curriculum hooks:</b> set_goal_tolerance() and set_hold_steps_required() let the curriculum callbacks adjust difficulty during training.",
         "<b>ActionSmoother wrapper:</b> applies EMA smoothing to actions at eval/viz time (NOT during training).",
     ],
     "io": [
         ["Action in", "4D float [0,1] muscle activations OR 2D float [-1,1] joint velocities"],
         ["Observation out", "11D float32 numpy array"],
         ["Reward out", "Scalar float (shaped, up to ~50,000 for a perfect episode)"],
         ["Info dict", "goal_reached, current_waypoint_index, hold_counter, hold_steps_required, joint_angles, end_effector_position, ..."],
     ],
     "classes": [
         ("ArmTaskEnv", "Main Gymnasium environment. All training, evaluation, and GUI simulation runs through this class."),
         ("ActionSmoother", "gym.ActionWrapper that applies EMA smoothing to actions — used only in the interactive GUI simulation, not during SAC training."),
     ],
     "notes": "Never change the observation space shape or reward weights without re-training from scratch."},

    {"title": "SimpleArmEnv  (template / reference)",
     "path": "src/rl_armMotion/two_d/environments/simple_arm.py",
     "lines": 119,
     "purpose": "A minimal 7-DOF arm template environment that shows the standard Gymnasium structure. Not used in training — kept as a reference scaffold for future 3D work.",
     "what": [
         "Shows the standard reset/step/render pattern for a Gymnasium environment.",
         "Useful as a starting point for extending the project to 3D or more DOF.",
     ],
     "classes": [
         ("SimpleArmEnv", "Template 7-DOF arm environment. Not used in the current 2-DOF training pipeline."),
     ],
     "io": None,
     "notes": "This file is not used in any training or GUI workflow. It is a scaffold for future work."},

    # ── SECTION 3: GUIs ───────────────────────────────────────────────────────
    {"_section": "3. Graphical User Interfaces"},

    {"title": "Unified Launcher  (entry point)",
     "path": "src/rl_armMotion/two_d/gui/__main__.py",
     "lines": 293,
     "purpose": "The front door of the project. Running 'python -m rl_armMotion.two_d.gui' opens a small launcher window that lets you start the Training GUI and the Interactive Arm GUI without using the command line.",
     "what": [
         "Spawns the Training GUI and Interactive Arm GUI as <b>independent subprocesses</b> so each gets its own Tkinter window.",
         "Form fields for algorithm (SAC/PPO/A2C), timestep budget, and save directory are pre-filled with Fischer 2021 defaults.",
         "Closing the launcher does NOT kill running training sessions — they keep going in the background.",
         "Tracks live PIDs in a status bar so you can see what's running.",
     ],
     "classes": [
         ("LauncherApp", "Tkinter window with buttons to spawn both GUIs. Manages subprocess PIDs."),
         ("main()", "Entry point — called by 'python -m rl_armMotion.two_d.gui'."),
     ],
     "io": [["Launch command", "python -m rl_armMotion.two_d.gui"]],
     "notes": None},

    {"title": "Training Dashboard GUI",
     "path": "src/rl_armMotion/two_d/gui/training_gui.py",
     "lines": 1154,
     "purpose": "The training control panel. Lets you start a SAC training run and watch it live — reward curves, loss curves, arm pose, and curriculum progress all update in real time.",
     "what": [
         "<b>Live reward / loss / entropy plots</b> update every 100 training steps.",
         "<b>Live arm pose panel</b> shows where the end-effector is during training.",
         "<b>Curriculum metrics panel:</b> current goal tolerance, curriculum stage, rolling success rate.",
         "<b>Goal mode selector:</b> Direction preset (EAST/WEST/NORTH), Single-Point click, or Waypoint sequence click.",
         "<b>Save Model &amp; Results button:</b> saves model .zip, training_history.csv, training_stats.json, Fitts PNG, Power Law PNG.",
         "<b>Run Validators button:</b> runs Fitts' Law and 2/3 Power Law sweeps on the loaded model.",
         "Uses a background thread for training so the GUI stays responsive.",
     ],
     "classes": [
         ("TrainingGUI", "Main Tkinter window. Manages the training thread, live plots, and all UI controls."),
     ],
     "io": [
         ["Start training", "Click 'Start Training' — spawns a background thread calling RLTrainerWithMetrics.train()"],
         ["Save output", "project_assets/outputs/<chosen_dir>/sac_model.zip + CSV + JSON + PNGs"],
     ],
     "notes": None},

    {"title": "Interactive Arm GUI",
     "path": "src/rl_armMotion/two_d/gui/app.py",
     "lines": 1210,
     "purpose": "The simulation and visualisation window. Load a trained model and watch the arm navigate to waypoints. Also supports manual joint control, parameter tuning, and motion recording.",
     "what": [
         "<b>Load Model button:</b> loads a .zip model file and sets the actuation mode automatically.",
         "<b>Run Simulation button:</b> runs the trained policy in the environment and animates the arm.",
         "<b>Manual control:</b> sliders for direct joint angle control.",
         "<b>Motion recording/playback:</b> record a manual sequence and play it back.",
         "<b>Torque plots:</b> shoulder and elbow torque vs time chart.",
         "<b>Live metrics panel:</b> hold counter, waypoint index, end-effector position.",
     ],
     "classes": [
         ("ArmControllerGUI", "Main Tkinter application. Handles model loading, simulation loop, rendering, and manual controls."),
     ],
     "io": [
         ["Load", "Select .zip model file via file dialog"],
         ["Simulate", "Runs env.step() in a loop, renders arm pose at ~30 fps"],
     ],
     "notes": "Uses ActionSmoother EMA wrapper during simulation for smoother visual motion."},

    # ── SECTION 4: Models ─────────────────────────────────────────────────────
    {"_section": "4. Models & Agents"},

    {"title": "RLTrainer  (core trainer)",
     "path": "src/rl_armMotion/two_d/models/trainers.py",
     "lines": 303,
     "purpose": "The thin wrapper around Stable-Baselines3 that creates the SAC (or PPO/A2C) model, runs .learn(), saves, and loads. Every training run goes through this.",
     "what": [
         "Creates a Stable-Baselines3 model (SAC by default) with the correct hyperparameters.",
         "Provides <b>train()</b>, <b>evaluate()</b>, <b>save()</b>, and <b>load()</b> methods.",
         "Supports a linear learning-rate schedule (pickle-safe for SB3 compatibility).",
         "All algorithm choices (SAC/PPO/A2C) are routed through this single interface.",
     ],
     "classes": [
         ("RLTrainer", "Creates and manages the SB3 model. Used by RLTrainerWithMetrics and train_fischer_session.py."),
         ("BaseTrainer", "Abstract base class defining the trainer interface."),
         ("LinearDecaySchedule", "Pickle-safe linear LR schedule dataclass."),
     ],
     "io": [
         ["Input", "ArmTaskEnv instance + algorithm name + hyperparams dict"],
         ["Output", "Trained SB3 model saved as .zip"],
     ],
     "notes": None},

    {"title": "SACAgent",
     "path": "src/rl_armMotion/two_d/models/agents/sac_agent.py",
     "lines": 75,
     "purpose": "Holds the SAC hyperparameters that match Fischer et al. (2021). This is the only agent configuration that successfully solves the 3-waypoint task.",
     "what": [
         "Stores the Fischer 2021 SAC hyperparameters: lr=3e-4, batch=256, gamma=0.99, buffer=1M, ent_coef=auto, tau=0.005.",
         "Documents why each hyperparameter was chosen (with reference to the paper).",
         "Motor babbling note: SB3's SAC starts with random exploration automatically — no separate babbling phase needed.",
     ],
     "classes": [
         ("SACAgent", "Hyperparameter store for SAC. get_hyperparams() returns the Fischer 2021 config dict."),
     ],
     "io": None,
     "notes": "PPO and A2C were tested and do not reliably solve the hold criterion. SAC is the only working algorithm."},

    {"title": "PPOAgent",
     "path": "src/rl_armMotion/two_d/models/agents/ppo_agent.py",
     "lines": 48,
     "purpose": "PPO hyperparameter wrapper. Available in the GUI but does not achieve reliable 3/3 waypoint completion.",
     "what": [
         "Standard PPO config: lr=3e-4, n_steps=2048, batch=64, epochs=10, gamma=0.99.",
         "Available for experimentation but SAC should always be preferred for this task.",
     ],
     "classes": [("PPOAgent", "PPO hyperparameter store.")],
     "io": None,
     "notes": None},

    {"title": "A3CAgent",
     "path": "src/rl_armMotion/two_d/models/agents/a3c_agent.py",
     "lines": 51,
     "purpose": "A2C/A3C hyperparameter wrapper. Available in the GUI for experimentation.",
     "what": [
         "Uses A2C from Stable-Baselines3 (true async A3C is not supported by SB3).",
         "Standard config: lr=7e-4, n_steps=5, gamma=0.99.",
     ],
     "classes": [("A3CAgent", "A2C/A3C hyperparameter store.")],
     "io": None,
     "notes": None},

    {"title": "GUICallback  (SB3 training callbacks)",
     "path": "src/rl_armMotion/two_d/models/callbacks.py",
     "lines": 227,
     "purpose": "Stable-Baselines3 callbacks that extract metrics from the training loop and pass them to the GUI. Without these, the Training GUI would have no live data.",
     "what": [
         "<b>GUICallback:</b> fires every N steps, extracts episode rewards, losses, and entropy, pushes them into a queue for the GUI thread to read.",
         "<b>MetricsTracker:</b> accumulates a rolling history of training metrics.",
         "<b>PPOMetricsCallback:</b> PPO-specific metric extraction (policy loss, value loss, explained variance).",
     ],
     "classes": [
         ("GUICallback", "Base SB3 callback — collects episode data and pushes to a thread-safe queue every check_freq steps."),
         ("MetricsTracker", "Rolling window accumulator for reward, loss, distance statistics."),
         ("PPOMetricsCallback", "PPO-specific extension of GUICallback."),
     ],
     "io": None,
     "notes": None},

    # ── SECTION 5: Training Infrastructure ────────────────────────────────────
    {"_section": "5. Training Infrastructure"},

    {"title": "RLTrainerWithMetrics  (main training pipeline)",
     "path": "src/rl_armMotion/two_d/training/ppo_trainer_wrapper.py",
     "lines": 556,
     "purpose": "The bridge between the raw SB3 trainer and the rest of the project. This is what train_fischer_session.py and the Training GUI both use. It automatically attaches all curriculum callbacks and streams live metrics.",
     "what": [
         "<b>Auto-attaches AdaptiveCurriculumCallback</b> whenever SAC is used — no manual setup needed.",
         "<b>Auto-attaches HoldCurriculumCallback</b> — starts hold requirement at 10 steps, graduates to 15 then 20.",
         "<b>Streams metrics</b> (reward, loss, entropy, hold progress, curriculum stage) through a thread-safe queue to the GUI.",
         "<b>save_model_and_results()</b> saves model .zip, training_history.csv, and training_stats.json.",
         "The <b>TrainingGUICallback</b> (inner class) is what actually fires every step and extracts data.",
     ],
     "classes": [
         ("RLTrainerWithMetrics", "Main training wrapper. Attach curriculum callbacks, stream metrics, save results."),
         ("TrainingGUICallback", "SB3 callback that extracts per-step data and feeds the metrics pipeline."),
         ("PPOTrainerWithMetrics", "Backward-compatible PPO-only alias."),
     ],
     "io": [
         ["Input", "ArmTaskEnv + total_timesteps + algorithm + optional hyperparams"],
         ["Output", "Trained model in memory + optional save to disk"],
     ],
     "notes": None},

    {"title": "Curriculum Callbacks",
     "path": "src/rl_armMotion/two_d/training/curriculum_callback.py",
     "lines": 505,
     "purpose": "Three SB3 callbacks that make the training task gradually harder as the agent improves. This is the Fischer et al. 2021 curriculum methodology — without it the agent cannot learn to hold precisely.",
     "what": [
         "<b>AdaptiveCurriculumCallback:</b> starts goal tolerance at 0.60 m (very easy to enter), shrinks it by 20% each time the rolling 50-episode success rate exceeds 80%, down to 0.02 m (very precise). Matches Fischer 2021 exactly.",
         "<b>HoldCurriculumCallback (new):</b> starts hold_steps_required at 10 (easy hold), advances to 15 then 20 as the agent achieves &gt;60% success. Solves the 2/3 → 3/3 bottleneck.",
         "<b>EvalBasedCurriculumCallback:</b> alternative curriculum that advances based on periodic deterministic evaluation rather than stochastic training success rate.",
         "All three push updates to the environment via set_goal_tolerance() / set_hold_steps_required() hooks.",
     ],
     "classes": [
         ("AdaptiveCurriculumCallback", "Shrinks position tolerance from 0.60 m → 0.02 m as success rate rises. Fischer 2021 protocol."),
         ("HoldCurriculumCallback", "Increases hold requirement from 10 → 15 → 20 steps as agent learns. New in this project."),
         ("EvalBasedCurriculumCallback", "Curriculum based on periodic deterministic evaluation rather than training rollouts."),
     ],
     "io": None,
     "notes": "Both AdaptiveCurriculumCallback and HoldCurriculumCallback run simultaneously — they operate independently on different aspects of the task."},

    # ── SECTION 6: Utilities ──────────────────────────────────────────────────
    {"_section": "6. Utilities"},

    {"title": "Hill-Type Muscle Model",
     "path": "src/rl_armMotion/two_d/utils/muscle_model.py",
     "lines": 274,
     "purpose": "Implements the biomechanically realistic muscle physics used in muscle actuation mode. Instead of directly commanding joint velocities, the policy commands muscle activation levels (0–1), and this module converts them to forces and then torques.",
     "what": [
         "<b>Force equation:</b> F = activation × F_max × f_L(length) × f_V(velocity).",
         "<b>f_L (force-length curve):</b> peaks at optimal fibre length, drops off for shorter/longer fibres. Gaussian shape.",
         "<b>f_V (force-velocity curve):</b> Hill's hyperbolic equation — force decreases as muscle shortens faster (concentric), increases as it lengthens (eccentric).",
         "<b>Antagonist pairs:</b> each joint has an extensor and a flexor muscle — net torque = (F_extensor - F_flexor) × moment_arm.",
         "This naturally produces smooth, biologically plausible motion — the policy doesn't need a smoothness reward term.",
     ],
     "classes": [
         ("MuscleParameters", "Dataclass: F_max, optimal length, moment arm, max velocity, pennation angle per muscle."),
         ("HillTypeMuscle", "Computes muscle force given activation, current length and velocity. apply_to_joint() returns net torque."),
     ],
     "io": [
         ["Input", "activation [0,1], current fibre length, current fibre velocity"],
         ["Output", "Force in Newtons → converted to torque by ArmTaskEnv"],
     ],
     "notes": "This is the same muscle model used in Fischer et al. (2021) and in OpenSim / MuJoCo musculoskeletal simulations."},

    {"title": "Arm Kinematics",
     "path": "src/rl_armMotion/two_d/utils/arm_kinematics.py",
     "lines": 306,
     "purpose": "Forward kinematics, joint control, and motion recording utilities. Computes where the end-effector is given joint angles.",
     "what": [
         "<b>ArmKinematics.forward_kinematics():</b> given joint angles, returns (x, y) positions of each joint using standard 2D rotation matrices.",
         "<b>ArmController:</b> wraps kinematics with joint limit enforcement and velocity integration.",
         "<b>MotionRecorder:</b> records a sequence of arm states to a list; can save/load as JSON for playback in the Interactive GUI.",
     ],
     "classes": [
         ("ArmState", "Dataclass: joint angles, velocities, link endpoint positions, timestamp."),
         ("ArmKinematics", "forward_kinematics() → 2D joint positions. inverse_kinematics() stub for future use."),
         ("ArmController", "Integrates velocity commands, enforces joint limits, delegates to ArmKinematics."),
         ("MotionRecorder", "Records/replays sequences of ArmState for the GUI playback feature."),
     ],
     "io": None,
     "notes": None},

    {"title": "Visualization Utilities",
     "path": "src/rl_armMotion/two_d/utils/visualization.py",
     "lines": 402,
     "purpose": "Matplotlib and Plotly helpers for rendering the arm, plotting trajectories, and creating training dashboards.",
     "what": [
         "<b>ArmVisualizer.render_arm():</b> draws the 2-link arm on a Matplotlib axes given joint angles.",
         "<b>ArmVisualizer.plot_trajectory():</b> plots end-effector path over an episode.",
         "<b>SimulationVisualizer:</b> real-time animation loop used by the Interactive GUI.",
         "Also contains Plotly-based comparison dashboards (used in legacy notebooks).",
     ],
     "classes": [
         ("ArmVisualizer", "Matplotlib arm renderer and trajectory plotter."),
         ("SimulationVisualizer", "Real-time animation loop for the GUI."),
     ],
     "io": None,
     "notes": "The Plotly dependency (plotly) is optional — comment it out in requirements.txt if not needed."},

    {"title": "Parallel Environment Utilities",
     "path": "src/rl_armMotion/two_d/utils/parallel_env.py",
     "lines": 191,
     "purpose": "Utilities for running multiple environment instances in parallel using multiprocessing. Used for data collection and future distributed training experiments.",
     "what": [
         "run_single_simulation(): runs one episode of any Gymnasium environment in a subprocess.",
         "<b>ParallelEnvironmentRunner:</b> manages a pool of worker processes, each running an independent simulation.",
         "<b>VectorEnvironment:</b> wraps multiple environments for synchronous stepping (similar to SB3 DummyVecEnv but custom).",
     ],
     "classes": [
         ("SimulationResult", "Dataclass: env_id, episode_reward, episode_length, final observation, info dict."),
         ("ParallelEnvironmentRunner", "Manages a multiprocessing Pool of simulation workers."),
         ("VectorEnvironment", "Synchronous multi-env wrapper for step/reset."),
     ],
     "io": None,
     "notes": "Not used in the main training pipeline — the main SAC training uses SB3's built-in single-env loop. This is for custom data collection experiments."},

    # ── SECTION 7: Validation ─────────────────────────────────────────────────
    {"_section": "7. Validation Harnesses"},

    {"title": "Fitts' Law Validator",
     "path": "src/rl_armMotion/two_d/validation/fitts_law.py",
     "lines": 569,
     "purpose": "Tests whether the trained arm policy reproduces the classical human speed-accuracy trade-off (Fitts' Law). This is one of the two scientific benchmarks from Fischer et al. (2021) that prove the policy behaves like a biological arm.",
     "what": [
         "<b>Fitts' Law:</b> Movement Time = a + b × ID, where ID = log₂(2×Distance/Width). Humans follow this law across all target sizes and distances.",
         "Sweeps 12 target distances × 3 widths = 36 conditions, runs 10 trials each by default.",
         "Measures the time from movement onset to first entering the goal zone.",
         "Fits OLS regression on MT vs ID, reports R² and the a, b coefficients.",
         "Saves results to <b>fitts_law.json</b> and a scatter/regression <b>fitts_law.png</b>.",
         "Fischer et al. achieved R² = 0.9986 on their 7-DOF arm — a result close to this is expected.",
     ],
     "classes": [
         ("FittsLawValidator", "Main harness — sweeps conditions, runs rollouts, fits regression."),
         ("FittsLawResult", "Stores all trial data and regression coefficients. save_json() and plot() methods."),
         ("FittsLawCondition", "One (distance, width) test condition."),
         ("FittsLawTrial", "One deterministic rollout result for one condition."),
     ],
     "io": [
         ["Input", "Trained SB3 model + ArmTaskEnv instance"],
         ["Output", "fitts_law.json (all trial data) + fitts_law.png (regression plot)"],
     ],
     "notes": None},

    {"title": "2/3 Power Law Validator",
     "path": "src/rl_armMotion/two_d/validation/power_law.py",
     "lines": 555,
     "purpose": "Tests whether the trained policy reproduces the 2/3 Power Law — the relationship between hand speed and path curvature observed in all human arm movements. The second Fischer 2021 benchmark.",
     "what": [
         "<b>2/3 Power Law:</b> V = K × C^(-1/3), i.e. the arm naturally slows down on tight curves. Humans follow this without being explicitly trained to.",
         "Runs 20 circular arc trajectories through the workspace.",
         "Extracts per-step end-effector speed V and curvature C by central differences.",
         "Fits log-log regression: log V = log K + slope × log C. Slope should be close to -1/3.",
         "Saves results to <b>power_law.json</b> and a log-log scatter <b>power_law.png</b>.",
         "Fischer et al. reported R = 0.84 on their arm.",
     ],
     "classes": [
         ("PowerLawValidator", "Main harness — runs arc trials, computes V and C, fits regression."),
         ("PowerLawResult", "Stores all trial data and regression. save_json() and plot() methods."),
         ("PowerLawTrial", "One trajectory trial result."),
     ],
     "io": [
         ["Input", "Trained SB3 model + ArmTaskEnv instance"],
         ["Output", "power_law.json + power_law.png"],
     ],
     "notes": None},

    # ── SECTION 8: Scripts ────────────────────────────────────────────────────
    {"_section": "8. Scripts"},

    {"title": "train_fischer_session.py  (headless training CLI)",
     "path": "scripts/train_fischer_session.py",
     "lines": 268,
     "purpose": "Command-line script for running a full Fischer 2021 training session without opening any GUI. Used by run_auto_search.py to train each seed.",
     "what": [
         "Parses CLI flags: --timesteps, --actuation-mode, --goal-direction, --waypoints, --save-dir, --num-threads.",
         "Builds ArmTaskEnv, wraps it in RLTrainerWithMetrics (which auto-attaches both curriculum callbacks).",
         "Trains for the requested number of steps, then saves model + CSV + JSON.",
         "Runs Fitts' Law and 2/3 Power Law validators on the trained model and saves PNGs.",
         "All output logged to training_log.txt in the save directory.",
     ],
     "io": [
         ["CLI example", "python scripts/train_fischer_session.py --timesteps 500000 --actuation-mode muscle --waypoints '2.3,-1.0;1.0,1.5;2.5,0.0' --save-dir ./outputs/run1"],
         ["Output files", "sac_model.zip, training_history.csv, training_stats.json, fitts_law.png, power_law.png, training_log.txt"],
     ],
     "classes": [
         ("parse_args()", "Defines all CLI flags with defaults and help text."),
         ("main()", "Orchestrates env construction → training → saving → validation."),
     ],
     "notes": None},

    {"title": "run_auto_search.py  (automated champion search)",
     "path": "run_auto_search.py",
     "lines": 303,
     "purpose": "The automated batch search that runs 5 seeds in parallel and stops as soon as one achieves 3/3 waypoints. This is the main script running during the research phase.",
     "what": [
         "Trains 5 seeds in parallel, each calling train_fischer_session.py as a subprocess.",
         "<b>Early killing:</b> at 250K steps, any seed with reward &lt; 0 is killed — saves ~45 min per bad seed.",
         "<b>Thread pinning:</b> each seed gets 3 PyTorch threads (total = 15 threads on 24 cores), preventing CPU fights.",
         "<b>Auto-evaluation:</b> after each batch, evaluates all completed models with deterministic rollouts.",
         "<b>Winner analysis:</b> when 3/3 found — runs 5 consistency checks, evaluates at 3 tolerances (0.6/0.4/0.2 m), copies model to CHAMPION/, writes REPLICATION_REPORT.md.",
         "<b>Resume-safe:</b> on restart, skips seeds that already have a saved model.",
     ],
     "io": [
         ["Run", "python run_auto_search.py"],
         ["Output", "project_assets/outputs/Parallel_Seeds/seed_NNN/ per seed"],
         ["Champion", "project_assets/outputs/CHAMPION/sac_model.zip + REPLICATION_REPORT.md"],
     ],
     "classes": [
         ("get_latest_reward()", "Parses a seed's stdout.log to extract the latest ep_rew_mean."),
         ("evaluate_model()", "Loads a model and runs one deterministic rollout, returns waypoints reached."),
         ("analyze_winner()", "Full consistency analysis + CHAMPION copy + REPLICATION_REPORT generation."),
         ("run_seed()", "Launches one training subprocess, monitors it, kills if below threshold."),
         ("main()", "Outer batch loop — runs batches until champion found or max_batches reached."),
     ],
     "notes": "Reduce SEEDS_PER_BATCH from 5 to 3 if the computer runs slowly during training."},
]


def build_pdf():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=letter,
        leftMargin=0.8*inch, rightMargin=0.8*inch,
        topMargin=0.9*inch, bottomMargin=0.8*inch,
        title="RL Arm Motion — Code Reference",
        author="Ranjot Sandhu",
    )
    story = []

    # ── Cover ──────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("RL Arm Motion", title_s))
    story.append(Paragraph("Code Reference Guide", title_s))
    story.append(hr())
    story.append(Paragraph("CP493 Directed Research · Wilfrid Laurier University", subtitle_s))
    story.append(Paragraph("Ranjot Sandhu · 2026", subtitle_s))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph(
        "This document describes every source file in the project — what it does, "
        "what classes it contains, and how it fits into the overall system. "
        "Use it as a map when reading the code.",
        ParagraphStyle("Intro", parent=body_s, alignment=TA_CENTER, textColor=DARK_GRAY)
    ))
    story.append(Spacer(1, 0.1*inch))
    story.append(hr(MID_GRAY, 0.5))

    # ── Architecture overview table ────────────────────────────────────────
    story.append(Paragraph("System Architecture at a Glance", section_s))
    arch_data = [
        ["Layer", "Files", "Role"],
        ["Configuration", "config/arm_config.py", "All arm physical parameters in one place"],
        ["Environment", "environments/task_env.py", "Physics, reward, observation, success logic — the core"],
        ["Training", "training/ppo_trainer_wrapper.py\ntraining/curriculum_callback.py", "SAC training loop + adaptive curriculum"],
        ["Models", "models/trainers.py\nmodels/agents/sac_agent.py", "SB3 model creation and hyperparameters"],
        ["GUIs", "gui/__main__.py\ngui/training_gui.py\ngui/app.py", "Launcher, Training Dashboard, Interactive Arm"],
        ["Utilities", "utils/muscle_model.py\nutils/arm_kinematics.py", "Hill muscle physics, forward kinematics"],
        ["Validation", "validation/fitts_law.py\nvalidation/power_law.py", "Fischer 2021 scientific benchmarks"],
        ["Scripts", "scripts/train_fischer_session.py\nrun_auto_search.py", "Headless training CLI + automated seed search"],
    ]
    arch_t = Table(arch_data, colWidths=[1.4*inch, 2.8*inch, 2.8*inch], repeatRows=1)
    arch_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,0), BRAND_BLUE),
        ("TEXTCOLOR",  (0,0),(-1,0), WHITE),
        ("FONTNAME",   (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0),(-1,0), 8.5),
        ("FONTNAME",   (0,1),(-1,-1),"Helvetica"),
        ("FONTSIZE",   (0,1),(-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, LIGHT_GRAY]),
        ("GRID",       (0,0),(-1,-1), 0.3, MID_GRAY),
        ("VALIGN",     (0,0),(-1,-1), "TOP"),
        ("TOPPADDING", (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("LEFTPADDING",(0,0),(-1,-1), 5),
    ]))
    story.append(arch_t)
    story.append(PageBreak())

    # ── File sections ──────────────────────────────────────────────────────
    for entry in CATALOGUE:
        if "_section" in entry:
            story.append(Paragraph(entry["_section"], section_s))
            continue

        elems = file_block(
            title=entry["title"],
            path=entry["path"],
            lines=entry["lines"],
            purpose=entry["purpose"],
            what_it_does=entry["what"],
            classes=entry["classes"],
            inputs_outputs=entry.get("io"),
            notes=entry.get("notes"),
        )
        story += elems

    doc.build(story)
    print(f"PDF written to: {OUT}")


if __name__ == "__main__":
    build_pdf()
