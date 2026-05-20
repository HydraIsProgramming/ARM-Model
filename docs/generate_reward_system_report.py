"""
Reward System Report - Ranjot Sandhu
Concise standalone description of the reward function used in the
2-DOF RL Arm Motion training environment.
Target length: 6-8 pages.
"""

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

BLACK = colors.black

def S(name, **kw): return ParagraphStyle(name, **kw)

cover_inst   = S("c_inst",  fontName="Times-Roman",       fontSize=14, textColor=BLACK, alignment=TA_CENTER, spaceAfter=4,  leading=18)
cover_dept   = S("c_dept",  fontName="Times-Italic",      fontSize=12, textColor=BLACK, alignment=TA_CENTER, spaceAfter=4,  leading=16)
cover_title  = S("c_title", fontName="Times-Bold",        fontSize=22, textColor=BLACK, alignment=TA_CENTER, spaceAfter=12, leading=28)
cover_sub    = S("c_sub",   fontName="Times-Italic",      fontSize=14, textColor=BLACK, alignment=TA_CENTER, spaceAfter=8,  leading=20)
cover_label  = S("c_lbl",   fontName="Times-Bold",        fontSize=11, textColor=BLACK, alignment=TA_CENTER, spaceAfter=2,  leading=14)
cover_text   = S("c_text",  fontName="Times-Roman",       fontSize=12, textColor=BLACK, alignment=TA_CENTER, spaceAfter=4,  leading=16)

h1     = S("h1",     fontName="Times-Bold",        fontSize=15, textColor=BLACK, spaceBefore=14, spaceAfter=8,  leading=18, keepWithNext=1)
h2     = S("h2",     fontName="Times-Bold",        fontSize=12, textColor=BLACK, spaceBefore=10, spaceAfter=4,  leading=15, keepWithNext=1)

body   = S("body",   fontName="Times-Roman",       fontSize=11, textColor=BLACK, leading=14, spaceAfter=6, alignment=TA_JUSTIFY)
bullet = S("bul",    fontName="Times-Roman",       fontSize=11, textColor=BLACK, leading=14, leftIndent=22, bulletIndent=8, spaceAfter=3, alignment=TA_JUSTIFY)
code   = S("code",   fontName="Courier",           fontSize=9,  textColor=BLACK, leading=11, spaceAfter=6, leftIndent=22, rightIndent=22)
caption= S("cap",    fontName="Times-Italic",      fontSize=10, textColor=BLACK, leading=12, spaceAfter=10, alignment=TA_CENTER)
ref_style  = S("ref",  fontName="Times-Roman", fontSize=10, textColor=BLACK, leading=12, leftIndent=24, firstLineIndent=-24, spaceAfter=5, alignment=TA_JUSTIFY)
abst_style = S("abst", fontName="Times-Roman", fontSize=11, textColor=BLACK, leading=14, leftIndent=20, rightIndent=20, spaceAfter=8, alignment=TA_JUSTIFY)

def SP(h_=0.1): return Spacer(1, h_ * inch)
def HR(thickness=0.5, color=BLACK, sa=8, sb=4):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=sa, spaceBefore=sb)
def section(num, title):
    label = f"{num}.&nbsp;&nbsp;{title}" if num else title
    return Paragraph(label, h1)
def subsection(title): return Paragraph(title, h2)
def para(t): return Paragraph(t, body)
def bul(t): return Paragraph(f"•&nbsp;&nbsp;{t}", bullet)
def cap(n, t): return Paragraph(f"<b>Table {n}.</b> {t}", caption)
def code_block(text):
    safe = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return Paragraph(safe.replace("\n", "<br/>"), code)

def make_table(data, col_widths, header=True, font_size=9):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME",      (0,0), (-1,-1), "Times-Roman"),
        ("FONTSIZE",      (0,0), (-1,-1), font_size),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LINEABOVE",     (0,0), (-1,0),  1.2, BLACK),
        ("LINEBELOW",     (0,0), (-1,0),  0.5, BLACK),
        ("LINEBELOW",     (0,-1),(-1,-1), 1.2, BLACK),
    ]
    if header:
        style.append(("FONTNAME", (0,0), (-1,0), "Times-Bold"))
    t.setStyle(TableStyle(style))
    return t

def on_page(canvas, doc):
    canvas.saveState()
    if doc.page == 1:
        canvas.restoreState(); return
    canvas.setFont("Times-Italic", 9)
    canvas.setFillColor(BLACK)
    canvas.drawString(0.85*inch, LETTER[1] - 0.55*inch,
                      "Reward System Specification")
    canvas.drawRightString(LETTER[0] - 0.85*inch, LETTER[1] - 0.55*inch,
                           "R. Sandhu  |  Wilfrid Laurier University")
    canvas.setStrokeColor(BLACK); canvas.setLineWidth(0.4)
    canvas.line(0.85*inch, LETTER[1] - 0.65*inch, LETTER[0] - 0.85*inch, LETTER[1] - 0.65*inch)
    canvas.setFont("Times-Roman", 10)
    canvas.drawCentredString(LETTER[0]/2, 0.55*inch, f"{doc.page}")
    canvas.restoreState()

OUTPUT = "/Users/ranjotsandhu/Documents/Project/docs/Reward_System_Report.pdf"

doc = SimpleDocTemplate(
    OUTPUT, pagesize=LETTER,
    topMargin=0.95*inch, bottomMargin=0.85*inch,
    leftMargin=0.95*inch, rightMargin=0.95*inch,
    title="Reward System Specification - CP493 RL Arm Motion",
    author="Ranjot Sandhu",
    subject="Specification of the reward function used in the 2-DOF RL arm training environment",
)

story = []

# ─── COVER ────────────────────────────────────────────────────────────────────
story.append(SP(0.4))
story.append(Paragraph("WILFRID LAURIER UNIVERSITY", cover_inst))
story.append(Paragraph("Department of Physics and Computer Science", cover_dept))
story.append(SP(0.3))
story.append(HRFlowable(width="65%", thickness=0.8, color=BLACK, hAlign="CENTER", spaceAfter=14, spaceBefore=6))
story.append(SP(0.6))
story.append(Paragraph("Specification of the Reward Function for the Two-Degree-of-Freedom Robotic Arm Reinforcement Learning Environment", cover_title))
story.append(SP(0.15))
story.append(Paragraph("CP493 Directed Research Project", cover_sub))
story.append(SP(0.8))
story.append(HRFlowable(width="50%", thickness=0.5, color=BLACK, hAlign="CENTER", spaceAfter=14, spaceBefore=4))
story.append(SP(0.3))
story.append(Paragraph("Prepared by", cover_label))
story.append(Paragraph("Ranjot Sandhu", S("c_au", fontName="Times-Bold", fontSize=14, textColor=BLACK, alignment=TA_CENTER, spaceAfter=4)))
story.append(SP(0.3))
story.append(Paragraph("Submitted to", cover_label))
story.append(Paragraph("Professor Sukhjit Sehra", cover_text))
story.append(SP(0.6))
story.append(HRFlowable(width="40%", thickness=0.5, color=BLACK, hAlign="CENTER", spaceAfter=10, spaceBefore=4))
story.append(Paragraph("May 13, 2026", cover_text))
story.append(Paragraph("Waterloo, Ontario, Canada", S("c_loc", fontName="Times-Italic", fontSize=11, textColor=BLACK, alignment=TA_CENTER)))
story.append(PageBreak())

# ─── 1. OVERVIEW ─────────────────────────────────────────────────────────────
story.append(section("1", "Overview"))
story.append(HR(0.6, BLACK, 10, 4))
story.append(para(
    "The reward function for the two-degree-of-freedom robotic arm training environment "
    "(<i>ArmTaskEnv</i>) is a dense, shaped reward that combines five continuous penalty terms, "
    "four positive shaping bonuses, and one terminal bonus. The function is implemented in the "
    "<i>step()</i> method of the environment at "
    "<i>src/rl_armMotion/two_d/environments/task_env.py</i>, lines 457 through 480, and is "
    "evaluated at every environment step (one hundred times per simulated second at the default "
    "time step of dt = 0.01 s)."))

story.append(para(
    "The reward is designed for the goal-reaching task in which the arm must (i) bring its "
    "end-effector to a target position in the workspace, (ii) align the end-effector with a "
    "target orientation, and (iii) hold that pose stably for a configurable number of "
    "consecutive steps before the episode terminates with a success bonus. The shaped form "
    "of the reward is necessary because purely terminal rewards produce a sparse signal that is "
    "very difficult to learn from on continuous-control tasks."))

story.append(SP(0.05))
story.append(subsection("Functional form"))
story.append(para(
    "The total reward at each step is the sum of three components: continuous penalties on "
    "task error and motion quality, shaping bonuses that increase as the arm approaches and "
    "occupies the goal, and a one-time terminal bonus when the hold criterion is met. Formally,"))
story.append(code_block(
    "r_t = -PENALTY_TERMS + SHAPING_BONUSES + TERMINAL_BONUS"))
story.append(para(
    "Each of the three groups is itemised below in Sections 2, 3 and 4. Section 5 defines the "
    "goal-region criteria. Section 6 explains how the goal tolerance interacts with the adaptive "
    "curriculum scheduler used during training."))
story.append(PageBreak())

# ─── 2. CONTINUOUS PENALTIES ────────────────────────────────────────────────
story.append(section("2", "Continuous Penalty Terms"))
story.append(HR(0.6, BLACK, 10, 4))
story.append(para(
    "Five penalty terms are subtracted from the reward on every step. These provide the primary "
    "gradient that pulls the policy toward goal-reaching and smooth motion."))

story.append(SP(0.05))
pen_data = [
    ["#", "Term", "Coefficient", "Description"],
    ["P1", "-c1 * goal_distance",
     "c1 = 2.0",
     "Euclidean distance from the end-effector to the goal position, in metres. The dominant signal in the reward."],
    ["P2", "-c2 * orientation_error",
     "c2 = 1.0",
     "Absolute orientation error of the end-effector relative to the target orientation, in radians."],
    ["P3", "-c3 * velocity_norm",
     "c3 = 0.15",
     "L2 norm of the joint angular velocity vector. Penalises rapid or jerky motion."],
    ["P4", "-c4 * gradient_norm",
     "c4 = 0.20",
     "Normalised rate of change of the combined task error from one step to the next. Discourages large step-to-step error fluctuations such as overshoot."],
    ["P5", "-c5 * ||action||",
     "c5 = 0.01",
     "L2 norm of the action vector. A small energy cost: a mild preference for low-magnitude commands all else equal."],
]
story.append(make_table(pen_data, [0.4*inch, 1.7*inch, 0.9*inch, 3.0*inch], font_size=9))
story.append(SP(0.05))
story.append(cap(1, "The five continuous penalty terms applied at every step."))

story.append(SP(0.1))
story.append(subsection("Rationale"))
story.append(para(
    "Terms P1 and P2 are the task-error terms. They are the only terms whose minimisation is "
    "necessary for success: a policy that drives them both to zero is, by definition, in the "
    "goal region. Term P3 is a smooth-motion regulariser inspired by the minimum-jerk principle "
    "of biological motor control (Flash and Hogan, 1985). Term P4 penalises high-frequency "
    "oscillation in the error signal, which is the symptom of an over-aggressive controller. "
    "Term P5 is a small action-magnitude cost; it has only marginal effect on the policy but "
    "provides a tie-breaker between otherwise equivalent actions."))
story.append(PageBreak())

# ─── 3. SHAPING BONUSES ─────────────────────────────────────────────────────
story.append(section("3", "Shaping Bonuses"))
story.append(HR(0.6, BLACK, 10, 4))
story.append(para(
    "Four positive bonuses are added to the reward when the corresponding condition is "
    "satisfied. These accelerate learning by providing a strong positive gradient as the arm "
    "approaches and occupies the goal."))

story.append(SP(0.05))
bon_data = [
    ["#", "Bonus", "Value", "When applied"],
    ["B1", "Progress bonus",
     "+1.5 * progress",
     "When the combined task error strictly decreased from the previous step. progress = previous_total_error - total_error."],
    ["B2", "Proximity bonus",
     "+4.0 * (1 - d/dthr)",
     "When the end-effector lies within 3x the goal tolerance. dthr = 3 * height_tolerance. The bonus ramps linearly from 0 at the boundary to 4.0 at the goal centre."],
    ["B3", "In-goal constant",
     "+10.0",
     "Every step the arm satisfies all three goal-region criteria (position, orientation, velocity). A strong constant pull to remain in the goal."],
    ["B4", "Hold growth bonus",
     "+2.0 * hold_counter",
     "Every step the arm is in the goal, growing linearly with the number of consecutive in-goal steps. Strongly rewards holding the target pose."],
]
story.append(make_table(bon_data, [0.4*inch, 1.5*inch, 1.5*inch, 2.6*inch], font_size=9))
story.append(SP(0.05))
story.append(cap(2, "The four positive shaping bonuses."))

story.append(SP(0.1))
story.append(subsection("Rationale"))
story.append(para(
    "Bonus B1 reflects classical potential-based reward shaping (Ng, Harada, and Russell, 1999): "
    "rewarding improvement leaves the optimal policy unchanged but accelerates convergence. "
    "Bonus B2 makes the reward landscape locally convex near the goal so the agent has a clear "
    "incentive to drive the last few centimetres of approach. Bonuses B3 and B4 together "
    "address the stability part of the task: once the arm enters the goal region, the constant "
    "B3 alone would let it drift in and out; the linearly-growing B4 makes it strictly better "
    "to stay than to leave and re-enter, so the agent learns to settle and hold."))
story.append(PageBreak())

# ─── 4. TERMINAL BONUS ──────────────────────────────────────────────────────
story.append(section("4", "Terminal Bonus and Episode Termination"))
story.append(HR(0.6, BLACK, 10, 4))
story.append(para(
    "The episode terminates as a success when the arm has held a valid goal pose for "
    "<i>hold_steps_required</i> = 20 consecutive steps, which corresponds to approximately "
    "0.2 seconds of stable contact at the default dt = 0.01 s. On the step that triggers "
    "termination, a single large terminal bonus is added to the reward."))

story.append(SP(0.05))
term_data = [
    ["Trigger", "Reward effect", "Episode outcome"],
    ["hold_counter >= 20", "+150.0 added to that step", "terminated = True (success)"],
    ["step_count >= 1000", "no additional reward",       "truncated = True (timeout)"],
]
story.append(make_table(term_data, [2.0*inch, 2.0*inch, 2.0*inch], font_size=10))
story.append(SP(0.05))
story.append(cap(3, "Episode-termination conditions."))

story.append(SP(0.15))
story.append(subsection("Rationale"))
story.append(para(
    "The 150-unit terminal bonus is large enough relative to the per-step penalties (which are "
    "rarely more negative than about -5 at peak goal distance) to make goal-reaching the "
    "dominant trajectory-level signal. The 20-step hold requirement is the practical operational "
    "criterion for a successful task: a brief brush against the target without stability does "
    "not count. The 1000-step truncation limit caps episode length at 10 seconds of simulation "
    "and is in practice rarely reached by a converged policy."))
story.append(PageBreak())

# ─── 5. GOAL CRITERIA ──────────────────────────────────────────────────────
story.append(section("5", "Goal-Region Criteria"))
story.append(HR(0.6, BLACK, 10, 4))
story.append(para(
    "An arm pose is considered to be in the goal region if all three of the following are "
    "satisfied simultaneously. The values below are the defaults; all three are configurable "
    "at construction time and the position tolerance is dynamically adjusted by the adaptive "
    "curriculum scheduler described in Section 6."))

story.append(SP(0.05))
crit_data = [
    ["Criterion", "Default value", "Configurable as"],
    ["Position error < tolerance",     "0.10 m",        "goal_tolerance / height_tolerance"],
    ["Orientation error < tolerance",  "10 degrees",    "orientation_tolerance_deg"],
    ["Joint velocity norm < tolerance","0.30 rad/s",    "hold_velocity_tolerance"],
]
story.append(make_table(crit_data, [2.4*inch, 1.5*inch, 2.3*inch], font_size=10))
story.append(SP(0.05))
story.append(cap(4, "The three simultaneous criteria for goal-region occupancy."))

story.append(SP(0.1))
story.append(para(
    "The simultaneous requirement is important: the position and orientation criteria together "
    "encode the spatial accuracy of the reach, and the velocity criterion encodes that the "
    "arm has come to rest at the target. Without the velocity criterion the agent could "
    "achieve a transient pass through the target without holding, which would not meet the "
    "physical definition of a successful goal-reaching task."))
story.append(PageBreak())

# ─── 6. CURRICULUM ──────────────────────────────────────────────────────────
story.append(section("6", "Interaction with the Adaptive Curriculum"))
story.append(HR(0.6, BLACK, 10, 4))
story.append(para(
    "The reward function itself is fixed; the practical difficulty of the task is modulated "
    "during training by the adaptive curriculum scheduler implemented in "
    "<i>src/rl_armMotion/two_d/training/curriculum_callback.py</i>. This scheduler "
    "follows the protocol of Fischer, Hoinville, Eickhoff, and Lilienthal (2021), "
    "<i>Scientific Reports</i> 11:14445, in which the position-tolerance radius is shrunk "
    "multiplicatively each time the agent's recent success rate exceeds a threshold."))

story.append(SP(0.05))
curr_data = [
    ["Parameter",              "Value", "Effect"],
    ["initial_tolerance",      "0.60 m",  "Position tolerance at the start of training. Wide so early random exploration produces successes."],
    ["min_tolerance",          "0.02 m",  "Final precision target. Tolerance is never shrunk below this value."],
    ["success_rate_threshold", "80%",     "If the rolling success rate over the recent window exceeds this, the tolerance is decayed."],
    ["decay_factor",           "0.80",    "Multiplicative shrink applied at each decay event."],
    ["window_size",            "50 ep.",  "Number of recent episodes counted in the rolling success rate."],
    ["min_episodes_before_decay","20 ep.","Cooldown between consecutive decays, to prevent oscillation."],
]
story.append(make_table(curr_data, [1.7*inch, 0.8*inch, 3.7*inch], font_size=9))
story.append(SP(0.05))
story.append(cap(5, "Curriculum parameters and their effects on the reward function's goal criterion."))

story.append(SP(0.1))
story.append(para(
    "The interaction is straightforward: the curriculum calls "
    "<i>env.set_goal_tolerance(new_value)</i> at each decay event, which updates the "
    "<i>height_tolerance</i> attribute used by the goal-region check in Section 5 and by the "
    "proximity-bonus threshold in Section 3. The reward formula is unchanged. As a result, the "
    "same reward function produces a much easier task early in training (large goal radius, "
    "frequent successes, strong learning signal) and a much stricter task late in training "
    "(small goal radius, precise final convergence)."))
story.append(PageBreak())

# ─── 7. HISTORICAL NOTE ────────────────────────────────────────────────────
story.append(section("7", "Historical Note on Physics Constraints"))
story.append(HR(0.6, BLACK, 10, 4))
story.append(para(
    "An earlier development phase of this project (commits <i>bf6c899</i> and <i>d89cd2b</i>, "
    "21 March 2026) introduced four explicit physics-grounded reward constraints, labelled A "
    "for gravitational potential energy, C for joint acceleration, J for mechanical energy, and "
    "K for jerk, each grounded in classical mechanics and the computational-motor-control "
    "literature. These were intended to give the reward function direct biomechanical "
    "interpretability."))
story.append(para(
    "Across subsequent iterations the four explicit constraints were merged into the simpler, "
    "more learnable shaping form documented in Sections 2 to 4 of this report. Specifically, "
    "the jerk and joint-acceleration constraints (K and C) were absorbed into the velocity-norm "
    "penalty P3 and the gradient-norm penalty P4, the energy constraints (A and J) were "
    "absorbed into the action-norm penalty P5, and the goal-distance term (P1) was retained "
    "unchanged. The current shaped form was found to converge faster on the 2-DOF arm than the "
    "explicit-constraint form while retaining the same qualitative effect on policy behaviour. "
    "The full four-constraint form is preserved in the git history and can be restored if a "
    "future experiment requires explicit biomechanical interpretability of the reward terms."))
story.append(PageBreak())

# ─── 8. RELATIONSHIP TO FISCHER 2021'S REWARD ─────────────────────────────
story.append(section("8", "Relationship to Fischer 2021's Reward Function"))
story.append(HR(0.6, BLACK, 10, 4))
story.append(para(
    "The reward function documented in this report was not changed as part of the integration "
    "of the Fischer et al. (2021) methodology into this project. This section explains why that "
    "was the correct choice given the current state of the environment, and identifies the "
    "reward decision as future work coupled to the muscle wiring planned in Phase 6."))

story.append(subsection("8.1  Fischer's Reward Function"))
story.append(para(
    "Fischer et al. (2021) used a deliberately simple three-term reward in their MuJoCo "
    "seven-degree-of-freedom musculoskeletal arm:"))
story.append(code_block(
    "r_t = -|| p_end_effector - p_target ||      (distance penalty)\n"
    "      -  alpha * || muscle_activation ||^2  (effort cost, alpha = 0.05)\n"
    "      +  R_success   if distance < tolerance (sparse terminal bonus, R_success = 1.0)"))
story.append(para(
    "There are no proximity bonuses, no orientation term, no gradient-norm penalty, no progress "
    "bonus, and no hold-growth bonus. The position tolerance is what their adaptive curriculum "
    "shrinks from 0.60 m to 0.02 m over training. The sparseness is intentional. Fischer's "
    "scientific claim is that human-like motion (Fitts' Law, the 2/3 Power Law) emerges from "
    "the Hill-type muscle dynamics and the adaptive curriculum, not from clever reward "
    "engineering. Adding shaping terms to the reward would weaken the emergence claim because "
    "the agent would simply be doing what the reward told it to do."))

story.append(subsection("8.2  Why Our Reward Is Different"))
story.append(para(
    "The current training environment <i>ArmTaskEnv</i> uses direct velocity-command actuation: "
    "the action vector is a normalised joint-velocity command in the closed interval minus one "
    "to plus one, applied directly to the joints with only the kinematic limits clipping the "
    "result. There is no muscle in the actuation path. Without muscle dynamics in the env, the "
    "agent has no physical regularisation that pulls toward smooth motion; the reward must "
    "supply that regularisation explicitly. Terms P3 (velocity-norm penalty) and P4 "
    "(gradient-norm penalty) in our reward exist exactly to do the smoothing work that Fischer's "
    "muscle force-velocity curve does physically. Term B2 (proximity ramp) makes the reward "
    "landscape convex near the goal so the agent can find the last few centimetres of approach "
    "without depending on a stochastic visit to the sparse success region."))
story.append(para(
    "If the reward were stripped to Fischer's three-term form while keeping direct-velocity "
    "actuation, two problems would arise. First, the agent would have no incentive to move "
    "smoothly: high-velocity, jerky trajectories that nevertheless reach the target would "
    "score equally well to smooth ones. The Fitts' Law and Power Law harnesses would "
    "consequently report poor numbers — but for the wrong reason. Second, the sparse success "
    "bonus alone is too weak a signal for SAC to bootstrap from in early training without the "
    "physical smoothing that muscles provide. Training would either diverge or take many more "
    "samples than is practical."))

story.append(subsection("8.3  When the Reward Decision Becomes Active"))
story.append(para(
    "Phase 6 of the Fischer integration plan, documented in <i>progress.md</i> Section 7.3, "
    "is to wire the Hill-type muscle model added in Step 5 into the training environment's "
    "step function. When that happens, the action space will become muscle activations in the "
    "closed interval zero to one rather than velocity commands. The muscle force-velocity "
    "curve will then physically regularise the motion, and the case for keeping the velocity- "
    "and gradient-norm penalties in the reward weakens. Three options will be on the table:"))
story.append(bul("<b>Option A.</b> Keep the present shaped reward unchanged. Lowest risk on "
                 "convergence; weakens the emergence claim because the reward is still doing "
                 "smoothing work that the muscles also do."))
story.append(bul("<b>Option B.</b> Switch to Fischer's three-term reward exactly. Strongest "
                 "scientific claim; higher risk on convergence because the 2-DOF arm has fewer "
                 "muscles and lower inertia than Fischer's 7-DOF arm."))
story.append(bul("<b>Option C (recommended).</b> Hybrid: keep distance, effort, and sparse "
                 "success bonus, drop the velocity-norm, gradient-norm, and proximity-ramp "
                 "shaping. Sparse enough to preserve the emergence claim; dense enough to be "
                 "tractable on 2-DOF."))
story.append(para(
    "The choice should be made when Phase 6 lands, with actual training runs available for "
    "side-by-side comparison. The Fitts' Law and Power Law validation harnesses must in any "
    "case be re-run after the muscle wiring: their numbers will change regardless of which "
    "reward option is chosen."))

story.append(subsection("8.4  Summary of the Position Taken in This Report"))
story.append(para(
    "The present shaped reward documented in Sections 2 to 4 of this report is the principled "
    "choice <i>given the current direct-velocity actuation</i>. Fischer's three-term reward is "
    "not currently appropriate because the env lacks the physical mechanism (muscle dynamics) "
    "that supplies smoothness in Fischer's setup. The reward and the actuation layer are "
    "coupled, and the next coherent step is to change both together in Phase 6 — not to change "
    "the reward in isolation."))

story.append(PageBreak())

# ─── REFERENCES ──────────────────────────────────────────────────────────
story.append(section("", "References"))
story.append(HR(0.6, BLACK, 10, 4))
refs = [
    "Fischer, M., Hoinville, T., Eickhoff, S. B., &amp; Lilienthal, A. J. (2021). Reinforcement learning control of a biomechanical model of the upper extremity. <i>Scientific Reports</i>, 11, 14445.",
    "Flash, T., &amp; Hogan, N. (1985). The coordination of arm movements: an experimentally confirmed mathematical model. <i>Journal of Neuroscience</i>, 5(7), 1688–1703.",
    "Haarnoja, T., Zhou, A., Abbeel, P., &amp; Levine, S. (2018). Soft Actor-Critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. <i>Proceedings of the 35th International Conference on Machine Learning</i>, PMLR 80:1861–1870.",
    "Ng, A. Y., Harada, D., &amp; Russell, S. (1999). Policy invariance under reward transformations: theory and application to reward shaping. <i>Proceedings of the 16th International Conference on Machine Learning</i>, 278–287.",
    "Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., &amp; Dormann, N. (2021). Stable-Baselines3: Reliable reinforcement learning implementations. <i>Journal of Machine Learning Research</i>, 22(268), 1–8.",
    "Sutton, R. S., &amp; Barto, A. G. (2018). <i>Reinforcement Learning: An Introduction</i> (2nd ed.). MIT Press.",
    "Towers, M. et al. (2024). Gymnasium: A standard interface for reinforcement learning environments. <i>arXiv preprint</i> arXiv:2407.17032.",
]
for r in refs:
    story.append(Paragraph(r, ref_style))

doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"Report saved to: {OUTPUT}")
