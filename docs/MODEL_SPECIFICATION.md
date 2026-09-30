# Model Specification — Human-Arm v2 (reward, constraints, physics, training)

Physics, constraints and algorithm settings are read from the code on branch `human-arm-v2`
(`src/rl_armMotion/two_d/`: `config/arm_config.py`, `utils/dynamics.py`, `utils/muscle_model.py`,
`environments/task_env.py`, `training/curriculum_callback.py`, `models/trainers.py`, `evaluation.py`).
The **reward weights differ between training stages**, so §3.2 gives each stage's weights as recorded in
its own `run_manifest.json` rather than today's defaults. The data and figures behind every number in §7
are in `project_assets/report/` (see its README).

Notation: $q = (\theta_0, \theta_1)$ joint angles (rad), $\dot q$ joint velocities (rad/s), $p$ hand
position (m, workspace frame), $v = \dot p$ hand velocity, $g^{(k)}$ the active waypoint, $R = 0.65$ m the
arm's reach, $\Delta t = 0.02$ s the control step, $a \in [0,1]^4$ the action (muscle activations).

---

## 0. Constraints at a glance

Two kinds of limit act on the arm. **Hard** limits are properties of the simulated body and cannot be exceeded
whatever the policy does. **Soft** limits are costs in the reward; the policy can exceed them, and pays for it.
The distinction matters: the trained arm never breaks a hard limit (verified over every step of 30 episodes by
`scripts/verify_human_limits.py`), but it does trade against soft ones — the deployed R3 model peaks at 2.14 m/s
against its 1.5 m/s speed cap.

| constraint | value | kind | how it is enforced |
|---|---|---|---|
| segment lengths | upper arm 0.30 m, forearm + hand 0.35 m | body | Winter (2009), 75 kg / 1.75 m adult |
| segment masses | 2.10 kg, 1.65 kg (COM 0.131 m, 0.175 m) | body | enter the mass matrix and gravity |
| gravity | 9.81 m/s² | body | $G(q)$ in the equations of motion (§1.5) |
| shoulder range of motion | −150° … +90° (flexion −60° … 180°) | **hard** | integrator clamps the angle and removes velocity into the stop |
| elbow range of motion | 0° … 145° | **hard** | as above |
| joint torque | shoulder 25 N·m, elbow 12 N·m | **hard** | net actuator torque clipped (~40 % of adult maximum) |
| joint speed | 8 rad/s | **hard** | safety clamp; muscles lose force with speed well before (§1.6) |
| muscle activation | 0 … 1 per muscle | **hard** | the action space |
| activation rise / fall | 40 ms / 60 ms | **hard** | first-order filter, force cannot change instantly (B3 onward) |
| passive end-range stiffness | 20 N·m/rad inside 10° of each stop | body | a spring pushing the joint back out (§1.3) |
| joint damping | 0.30 N·m·s/rad | body | viscous term in $	au_{	ext{passive}}$ |
| muscle force–length, force–velocity | Hill-type (Thelen 2003; Hill 1938) | body | $F = a\,F_{\max}\,f_L\,f_V$ (§1.6) |
| hand speed | cap 1.5 m/s (R3, R4) / 1.2 m/s (R5, R6) | soft | reward cost above the cap (§3.1) |
| hand acceleration | reference 10 m/s² | soft | reward cost, minimum-jerk proxy (§3.1) |
| end-range posture | 10° from each stop | soft | reward cost for entering the zone (§3.1) |
| target: reachable | IK solution inside the ROM | task | feasibility rule 1 (§2.1) |
| target: holdable | static gravity torque ≤ 80 % of the limit | task | rule 2 |
| target: not degenerate | ≥ 0.18 m inside the reach circle | task | rule 3 |
| target: distinct | > 0.30 m apart, and from the start pose | task | rule 4 |
| target: comfortable | holding pose clears the end-range zone | task | rule 5 (R4 onward) |
| "reached" | within 5 cm, hand < 0.10 m/s, joints < 1.0 rad/s | task | in-goal test (§2.2) |
| "held" | 25 consecutive in-goal steps (0.5 s), −5 per step outside | task | hold counter (§2.3) |
| episode | 500 steps × 20 ms = 10 s | task | truncation, not termination, in training |

---

## 1. The arm (constraints from human anatomy)

### 1.1 Geometry and inertia — Winter (2009), 75 kg / 1.75 m adult

| segment | length $L$ | mass $m$ | COM offset $l_c$ (from proximal joint) | inertia about COM $I$ |
|---|---|---|---|---|
| upper arm (humerus) | 0.30 m | 2.10 kg (0.028 M) | 0.131 m (0.436 L) | 0.0196 kg·m² ($m(0.322L)^2$) |
| forearm + hand (to hand centre) | 0.35 m | 1.65 kg (0.022 M) | 0.175 m (0.682 × 0.256 m forearm) | 0.0237 kg·m² ($m(0.468 \times 0.256)^2$) |

Reach $R = L_1 + L_2 = 0.65$ m. Shoulder fixed at workspace $(1.0, 0.0)$; $+x$ is in front of the body, $+y$ up.
Angle convention: $\theta_0$ counter-clockwise from $+x$ (hanging arm $= -90^\circ$); $\theta_1$ = elbow
flexion, $0$ = straight.

### 1.2 Range of motion (hard stops)

| joint | limits | human meaning |
|---|---|---|
| shoulder $\theta_0$ | $[-150^\circ, +90^\circ]$ | flexion $-60^\circ \ldots 180^\circ$ |
| elbow $\theta_1$ | $[0^\circ, 145^\circ]$ | full extension … full flexion |

At a stop the angle is clamped and the velocity component pointing into the stop is set to zero, so an arm
resting on a limit reports zero velocity. A joint whose net generalised force pushes into a stop is *locked*
for that sub-step ($\ddot q_j = 0$) and the reduced system is solved for the free joints, so torque into a
stop is absorbed there and not transmitted up the chain.

### 1.3 Passive end-range stiffness (soft tissue)

Inside a margin $\delta = 10^\circ$ of every limit a linear spring engages, $K = 20$ N·m/rad:

$$\tau_{\text{passive},j} = -B_j \dot q_j \;-\; K \max\!\big(0,\; q_j - (q_{j,\max} - \delta)\big) \;+\; K \max\!\big(0,\; (q_{j,\min} + \delta) - q_j\big)$$

with viscous joint damping $B_j = 0.30$ N·m·s/rad (both joints). The relaxed arm therefore hangs with the
elbow a few degrees flexed, as a human elbow does.

### 1.4 Torque limits

$$|\tau_0| \le 25 \text{ N·m (shoulder)}, \qquad |\tau_1| \le 12 \text{ N·m (elbow)}$$

Peak gravity torque with the arm horizontal is $G_0 = g\,(m_1 l_{c1} + m_2 L_1 + m_2 l_{c2}) = 10.4$ N·m and
$G_1 = g\, m_2 l_{c2} = 2.8$ N·m, so the limits are 2.4× and 4.3× gravity — the arm can hold anywhere it can
reach, but cannot whip (adult maxima are ~60 N·m). Hard safety clamp on joint speed $|\dot q_j| \le 8$ rad/s
(rarely reached; the muscle force–velocity curve is the real limiter). Numerical guard $|\ddot q_j| \le 2000$ rad/s².

### 1.5 Rigid-body dynamics (gravity, coupled inertia, Coriolis)

$$M(q)\,\ddot q + C(q,\dot q)\,\dot q + G(q) = \tau_{\text{act}} + \tau_{\text{passive}}$$

Implemented generically for $N$ links (composite-rigid-body mass matrix, recursive Newton–Euler bias). For the
two-link arm the closed forms (the test oracle, matched to $10^{-15}$) are

$$M = \begin{bmatrix} I_1 + m_1 l_{c1}^2 + I_2 + m_2\,(L_1^2 + l_{c2}^2 + 2 L_1 l_{c2}\cos\theta_1) & I_2 + m_2\,(l_{c2}^2 + L_1 l_{c2}\cos\theta_1) \\ I_2 + m_2\,(l_{c2}^2 + L_1 l_{c2}\cos\theta_1) & I_2 + m_2 l_{c2}^2 \end{bmatrix}$$

$$h = m_2 L_1 l_{c2}\sin\theta_1, \qquad C\dot q = \begin{bmatrix} -h\,(2\dot\theta_0\dot\theta_1 + \dot\theta_1^2) \\ h\,\dot\theta_0^2 \end{bmatrix}$$

$$G(q) = g \begin{bmatrix} (m_1 l_{c1} + m_2 L_1)\cos\theta_0 + m_2 l_{c2}\cos(\theta_0+\theta_1) \\ m_2 l_{c2}\cos(\theta_0+\theta_1) \end{bmatrix}, \qquad g = 9.81 \text{ m/s}^2$$

Integration: second-order midpoint (RK2), 4 sub-steps of 5 ms per 20 ms control step (50 Hz control).
Energy drift of the undamped free pendulum is 0.3 % over 2 s.

### 1.6 Muscles (Hill-type antagonist pairs)

Each joint is driven by two lumped muscles, one per direction (shoulder: anterior vs posterior deltoid;
elbow: biceps/brachialis vs triceps). Contractile force

$$F = a \cdot F_{\max} \cdot f_L(\ell) \cdot f_V(\nu)$$

- **Force–length** (Thelen 2003 Gaussian): $f_L(\ell) = \exp\!\big(-((\ell/\ell_{\text{opt}} - 1)/\sigma_L)^2\big)$, $\sigma_L = 0.45$.
- **Force–velocity** (Hill 1938 hyperbola, $\nu = V/V_{\max}$ normalised shortening velocity, $a_H = 0.25$):

$$f_V(\nu) = \begin{cases} \dfrac{1-\nu}{1+\nu/a_H} & 0 \le \nu < 1 \quad(\text{concentric; } f_V(V_{\max}) = 0) \\[6pt] 0 & \nu \ge 1 \\[4pt] f_{\text{ecc}} - (f_{\text{ecc}}-1)\,e^{k\nu} & \nu < 0 \quad(\text{eccentric; } f_{\text{ecc}} = 1.5,\; k = \tfrac{1+1/a_H}{f_{\text{ecc}}-1}) \end{cases}$$

  The slope is continuous at $\nu = 0$ ($-(1+1/a_H) = -5$); at half $V_{\max}$ force is down to 0.167.
- **Fibre geometry** (constant moment arm $r$, fibre length linear in joint angle, $s=+1$ for the muscle that
  raises $\theta$, $s=-1$ for its antagonist, $\theta_{\text{ref}}$ = mid-range angle):

$$\ell = \ell_{\text{opt}} - s\, r\,(\theta - \theta_{\text{ref}}), \qquad V = s\, r\,\dot\theta / \ell_{\text{opt}}, \qquad \tau_{\text{muscle}} = r\,(F_{+} - F_{-})$$

| joint | moment arm $r$ | $F_{\max}$ (= $\tau_{\max}/r$) | $\ell_{\text{opt}}$ | $V_{\max}$ |
|---|---|---|---|---|
| shoulder | 0.04 m | 625 N | 0.20 m | 6 $\ell_{\text{opt}}$/s |
| elbow | 0.03 m | 400 N | 0.10 m | 6 $\ell_{\text{opt}}$/s |

  Consequences: $f_L \ge 0.45$ everywhere in the range of motion (weaker at the ends, never dead). Force falls to
  zero at a joint speed of $V_{\max}\,\ell_{\text{opt}}/r$ = **30 rad/s** at the shoulder and **20 rad/s** at the
  elbow, well above the 8 rad/s velocity clamp, so within reach the muscles weaken with speed but never quit
  (about half force at 5 rad/s at the shoulder). Equal co-contraction gives zero net torque but damps motion —
  a stiffening strategy the policy is free to use; there is no co-contraction penalty.
- **Activation dynamics** (Zajac 1989, first order, used in the "clean" Stage B3 models): with commanded
  excitation $u$ and activation state $a$,

$$\dot a = \frac{u - a}{\tau}, \qquad \tau = \begin{cases} 40 \text{ ms} & u > a \\ 60 \text{ ms} & u \le a \end{cases}$$

  so force cannot change instantly (no bang-bang commands). The four activations are appended to the observation.
  Off in Stages A/B (instantaneous activation).

- Net actuator torque is clipped to the joint torque limits.

### 1.7 Initial state randomisation

Every episode starts from the hanging pose $(\,-90^\circ,\ 10^\circ\,)$ plus a seeded uniform jitter of
$\pm 5^\circ$ (shoulder) and $-5^\circ \ldots +10^\circ$ (elbow), zero velocity; a start pose inside the first
waypoint's tolerance is resampled. The 10° elbow rest keeps the hand off the reach circle, which is a kinematic
singularity (elbow flexion has no first-order effect on hand distance there).

---

## 2. The task

### 2.1 Waypoints (v2 set, workspace frame, shoulder at (1.0, 0.0))

| | position | IK (θ₀, θ₁) | margin inside reach circle |
|---|---|---|---|
| A | (1.40, 0.15) | (−34°, 98°) | 0.22 m |
| B | (1.20, 0.40) | (12°, 93°) | 0.20 m |
| C | (1.40, −0.20) | (−78°, 93°) | 0.20 m |

2-waypoint task = [A, C]; 3-waypoint task = [A, B, C]. Pairwise spacings 0.32–0.63 m.

**Feasibility rules** (checked before any training run; a waypoint must satisfy all):
1. reachable with the elbow-down IK branch inside the range of motion;
2. static hold torque $|G(q^*)| \le 0.8\,\tau_{\max}$ per joint;
3. distance to the reach circle $\ge \max(\text{tolerance},\ \text{coarsest curriculum tolerance}) + 0.03$ m
   $= 0.18$ m. (Stage B on the older v1 set, whose final point sat 9 cm inside the circle, showed why: a straight
   arm resting on the elbow spring counted as "in goal" at the 15 cm rung and that shortcut could not be
   unlearned.)
4. Pairwise spacing $> 2 \times$ coarsest curriculum tolerance $= 0.30$ m, and start-pose distance to the first
   waypoint $>$ coarsest tolerance + jitter margin, so a motionless arm can never satisfy two goals at once.
5. The holding pose must clear every joint's passive end-range zone: `q*_j - q_min,j >= m_min,j` and
   `q_max,j - q*_j >= m_max,j`, with `m = 10` deg for this preset. Inside that zone the spring pushes the joint
   back out *and* the end-range term of the joint-limit penalty charges for staying, so such a target asks the
   policy to hold a posture the same MDP penalises. This is the joint-space twin of rule 3. No extra buffer is
   applied: the arm needs only one pose inside the tolerance ball that clears the zone, and the target's own
   pose is that pose - requiring the whole ball to clear it (a Jacobian-norm buffer
   `|dq_j| <= ||row_j(J^-1)|| * dx`) removes 41 % of the grid and the entire near band. Effect on the sampler:
   930 -> 744 points, near band 425 -> 277, minimum radius 0.255 m, elbow capped at 135 deg (= 145 - 10).
   A joint with no passive margin is exempt, so gravity-free legacy presets keep their feasible set.
   Added in Stage R4.

**Full human workspace (Stage R7 on).** Rules 3 and 5 together excluded 62 % of what the arm can reach within
its joint ranges (0.30 of 0.79 m² on the 2 cm grid; rule 3 alone removed 954 of 1984 grid points, rule 5 186,
rule 2 none). Rule 3 in particular kept every target where the elbow is bent at least ~88°, so the arm was never
asked to reach far out, overhead, low or behind the hip. With `joint_margin_deg` set (CLI `--joint-margin-deg`),
rules 3 and 5 are replaced by one human-limits rule:

3'. the target is reachable with every joint at least `joint_margin_deg` (5°) inside its hard stop, i.e. within
    the human range of motion without jamming a joint against its stop; `reach_margin` (0 m) is the only distance
    rule, and the inner limit is just the IK.

Rules 1, 2 and 4 are unchanged. Effect: 744 → **1850** grid points, 0.228–0.649 m from the shoulder and up to
0.43 m behind it. What remains excluded is what this arm genuinely cannot do: closer than 0.23 m (elbow ≤ 140°),
full extension itself (elbow on its stop), and the pocket in front of the chest that would need more than 55° of
shoulder extension — a person reaches there by moving the elbow out to the side, which a sagittal-plane arm
cannot. Both settings are recorded in the manifest; a manifest without them (every model up to R6) rebuilds with
rules 3 and 5, so older packages keep their original region.

Before R7, the deployed R3 on 200 three-target episodes from the full workspace
(`scripts/eval_workspace_regions.py`): 10 % episode success — 100 % of targets in its own area and close in, but
17 % in the outer ring, 12 % overhead, 15 % hanging low and 57 % behind the body.

### 2.1b Random waypoints (the generalising task, Stages R3 and R4)

`ArmTaskEnv(random_waypoints=N)` draws a fresh ordered set of N targets at every `reset()`
instead of using a fixed set. The sampler is built once per env:

1. a 2 cm grid (`waypoint_grid=0.02`) is laid over the bounding box of the reach circle,
   optionally restricted to `waypoint_region=(x_min, x_max, y_min, y_max)`;
2. every grid point is passed through the three feasibility rules of §2.1 — reachable with
   the required margin inside the reach circle, IK inside the ROM, static hold torque
   $|G(q^*)| \le 0.8\,	au_{\max}$ — leaving **744 feasible points** for the right-arm preset (930 before rule 5);
3. per reset, N points are drawn without replacement and rejected until the pairwise
   spacing rule of §2.1 and the start-pose distance rule both hold.

The observation (§2.4) already carries the active goal and the waypoint index, so nothing
about the policy interface changes — which is exactly why a fixed-waypoint policy *can*
ignore the goal channel and memorise a trajectory, and a random-waypoint policy cannot.
The sampled set is recorded per episode by `evaluate_policy`, and the manifest records
`random_waypoints`, `waypoint_region` and `waypoint_grid` so the task is reproducible.

### 2.2 In-goal criterion

$$\text{in\_goal} \iff \|p - g^{(k)}\| < \varepsilon \;\wedge\; \|v\| < 0.10 \text{ m/s} \;\wedge\; \|\dot q\| < 1.0 \text{ rad/s}$$

with $\varepsilon = 0.05$ m at the evaluation standard (curriculum-controlled during training, §4). The joint-speed
bound closes the loophole where joints move in the Jacobian null space while the hand is momentarily still.
Hand orientation is **not** a criterion: for a two-link arm it is a function of hand position once the elbow
branch is fixed by the range of motion (the earlier orientation criterion was the binding failure of the old
model). A `require_orientation` flag exists for redundant (3-link) arms.

### 2.3 Hold counter and waypoint progression

$$c_{t+1} = \begin{cases} \min(c_t + 1,\ 500) & \text{in\_goal and the goal ball was entered from outside} \\ \max(0,\ c_t - 5) & \text{otherwise} \end{cases}$$

- **Intermediate waypoint** completes when $c \ge 5$ (0.1 s settled inside the ball); then $k \leftarrow k+1$, $c \leftarrow 0$.
- **Final waypoint**: success is declared the first time $c \ge 25$ (0.5 s). The episode does **not** end on
  success during training — it runs to the 500-step (10 s) limit and the agent keeps earning the in-goal reward
  for staying still, which is what makes the learned hold stable. Evaluation and demos use `terminate_on_success`.
- Time limit is reported as Gymnasium `truncated` (value bootstrapped), never `terminated`.

### 2.4 Observation (`OBS_VERSION = 2`, all components bounded)

| index | component | range |
|---|---|---|
| 0–3 | $\sin\theta_0, \cos\theta_0, \sin\theta_1, \cos\theta_1$ | [−1, 1] |
| 4–5 | $\dot\theta_j / 8$ | [−1, 1] |
| 6–7 | $(g^{(k)} - p)/R$ (x, y) | [−2, 2] |
| 8 | $\|g^{(k)} - p\|/R$ | [0, 2] |
| 9–10 | $v / 2.0$ m/s (clipped) | [−1, 1] |
| 11 | $k/(N-1)$ waypoint progress (0 if $N=1$) | [0, 1] |
| 12 | in_goal flag | {0, 1} |
| 13 | hold progress $c / c_{\text{req}}$ | [0, 1] |
| 14–17 | muscle activations $a$ (only with activation dynamics) | [0, 1] |

The goal and the waypoint index are in the observation, so the process is Markov across waypoint switches and
the same observation supports random waypoints later.

### 2.5 Action

$a \in [0,1]^4$: (shoulder +, shoulder −, elbow +, elbow −) muscle excitations. (`torque` mode: $a \in [-1,1]^2$,
$\tau = a \cdot \tau_{\max}$, used as the non-muscle baseline.)

---

## 3. Reward

The reward is computed once per 20 ms control step, after the physics and the waypoint bookkeeping. All
distances are normalised by the reach $R = L_1 + L_2 = 0.65$ m.

### 3.1 The reward function (ordered task — every stage except F2/F3)

Let $p_t$ be the hand position, $g^{(k)}$ the active waypoint, $d_t = \lVert g^{(k)} - p_t \rVert$, and define the
potential on the augmented state (hand position, active waypoint index)

$$\Phi(s) = -\frac{d}{R}.$$

The reward for the transition $s_t \to s_{t+1}$ under action $a_t$ is the sum of ten terms:

$$
\begin{aligned}
r_t \;=\;& \underbrace{W_{\text{shape}}\big(\gamma\,\Phi(s_{t+1}) - \Phi(s_t)\big)}_{\text{shaping}}
\;\underbrace{-\;W_{\text{dist}}\,\frac{d_{t+1}}{R}}_{\text{distance}}
\;\underbrace{-\;W_{\text{effort}}\,\lVert a_t\rVert^2}_{\text{effort}}
\;\underbrace{-\;W_{\text{smooth}}\,\frac{\lVert\dot q_{t+1}\rVert}{\omega_{\max}}}_{\text{smoothness}} \\
&\underbrace{-\;W_{\text{limit}}\Big(n^{\text{stop}}_{t+1} + \textstyle\sum_j e_{j,t+1}\Big)}_{\text{joint limit}}
\;\underbrace{-\;\Big(W_{\text{lin}}\,\delta_{t+1} + W_{\text{speed}}\,\min\!\big(\delta_{t+1}^2,\ C_{\text{speed}}\big)\Big)}_{\text{speed excess}}
\;\underbrace{-\;W_{\text{accel}}\,\min\!\Big(\big(\tfrac{\lVert\dot v_{t+1}\rVert}{A_{\text{ref}}}\big)^2,\ C_{\text{accel}}\Big)}_{\text{hand acceleration}} \\
&\underbrace{+\;R_{\text{goal}}\,\mathbb{1}[\text{in goal} \wedge k = N-1]}_{\text{in-goal stream}}
\;\underbrace{+\;R_{\text{inter}}\,\mathbb{1}[\text{intermediate waypoint completed}]}_{\text{intermediate bonus}}
\;\underbrace{+\;R_{\text{succ}}\,\mathbb{1}[\text{hold completed, first time}]}_{\text{success bonus}}
\end{aligned}
$$

with

| symbol | meaning |
|---|---|
| $\gamma = 0.99$ | SAC discount; the shaping uses the same $\gamma$ so it is exactly potential-based |
| $\omega_{\max} = 8$ rad/s | joint velocity limit, used only to normalise |
| $n^{\text{stop}}$ | number of joints resting on a hard stop this step |
| $e_j \in [0,1]$ | how far joint $j$ has entered its 10° passive end-range zone (0 outside, 1 at the stop) |
| $\delta = \max(0,\ \lVert v\rVert - v_{\text{cap}})$ | hand speed above the cap, m/s |
| $\lVert\dot v\rVert = \lVert v_{t+1} - v_t\rVert / \Delta t$ | hand acceleration, a minimum-jerk proxy (Flash & Hogan 1985) |
| $N$ | number of waypoints; $k = N-1$ is the final one |

**In goal** means all three at once: position error $< \varepsilon$ (0.05 m at the production standard), hand
speed $< 0.10$ m/s, and joint speed $\lVert\dot q\rVert < 1.0$ rad/s. An **intermediate** waypoint is completed
after 5 consecutive in-goal steps (0.1 s); the **final** one after the hold counter reaches 25 (0.5 s), where
the counter rises by 1 per in-goal step and falls by 5 per step outside.

### 3.2 The weights each stage was actually trained with

These are read from each run's own manifest, not from the current code — the weights changed between stages,
and quoting today's defaults for an older model would misreport what it learned from. "off" means the term did
not exist when that stage was trained.

| stage | $W_{\text{shape}}$ | $W_{\text{dist}}$ | $W_{\text{effort}}$ | $W_{\text{smooth}}$ | $W_{\text{limit}}$ | $v_{\text{cap}}$ | $W_{\text{speed}}$ | $W_{\text{lin}}$ | $C_{\text{speed}}$ | $W_{\text{accel}}$ | $R_{\text{goal}}$ | $R_{\text{inter}}$ | $R_{\text{succ}}$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 2.0 | 0.5 | 0.02 | 0.05 | 0.1 | off | off | off | off | off | 1 | 10 | 50 |
| B | 2.0 | 0.5 | 0.02 | 0.05 | 0.1 | 1.5 | 0.5 | off | ∞ | off | 1 | 10 | 50 |
| B3 | 2.0 | 0.5 | 0.02 | 0.05 | 0.1 | 1.5 | 0.5 | off | ∞ | 0.02 | 1 | 10 | 50 |
| **R3 (deployed)** | **2.0** | **0.5** | **0.02** | **0.05** | **0.1** | **1.5** | **0.5** | **off** | **∞** | **0.02** | **1** | **10** | **50** |
| R4 | 2.0 | 0.5 | 0.02 | 0.05 | 0.1 | 1.5 | 0.5 | off | ∞ | 0.02 | 1 | 10 | 50 |
| R5 | 2.0 | 0.5 | 0.02 | 0.05 | 0.1 | 1.2 | 10.0 | off | 2.25 | 0.02 | 1 | 10 | 50 |
| R6 | 2.0 | 0.5 | 0.02 | 0.05 | 0.1 | 1.2 | 5.0 | 3.0 | 2.25 | 0.02 | 1 | 10 | 50 |
| F2 | 2.0 | 0.5 | 0.02 | 0.05 | 0.1 | 1.5 | 0.5 | off | ∞ | 0.02 | 1 | 10 | 50 |

$A_{\text{ref}} = 10$ m/s², $C_{\text{accel}} = 4$ throughout. So the reward the deployed R3 model learned from is,
explicitly,

$$
r_t = 2\big(0.99\,\Phi_{t+1} - \Phi_t\big) - 0.5\,\tfrac{d_{t+1}}{0.65} - 0.02\lVert a_t\rVert^2 - 0.05\,\tfrac{\lVert\dot q_{t+1}\rVert}{8}
- 0.1\big(n^{\text{stop}} + \textstyle\sum_j e_j\big) - 0.5\max(0, \lVert v\rVert - 1.5)^2 - 0.02\min\!\big((\tfrac{\lVert\dot v\rVert}{10})^2, 4\big)
+ \mathbb{1}[\ldots] + 10\,\mathbb{1}[\ldots] + 50\,\mathbb{1}[\ldots].
$$

Rebuilding a model from its manifest reproduces exactly its own row: a key absent from a manifest is treated as
"off" (weight 0, cap unbounded) rather than filled with today's default (`ArmTaskEnv.LEGACY_REWARD_OFF`). Until
that rule existed, rebuilding R3 silently switched on the R6-era linear speed term; one R3 episode was scored
−42 for speed where its own reward charges −3.8. Success, reach and hold were never affected — they come from
the hold logic, not the reward — but a return computed after the fact for an older model was.

### 3.3 Why each term is there

| term | job | where it came from |
|---|---|---|
| shaping | a gradient toward the goal from the first update | potential-based (Ng, Harada & Russell 1999), so it cannot change which policy is optimal; two-sided so leaving the goal costs what approaching it paid |
| distance | a dense pull that does not vanish when the hand is still | without it the only signal at rest is the shaping, which is zero when nothing moves |
| effort | prefer lower activations for the same result | a metabolic-cost proxy; kept small so it never outweighs reaching |
| smoothness | discourage needless joint speed | mild; the acceleration term does most of the smoothing |
| joint limit | stay off the hard stops and out of end range | Stage A's first attempt: every seed rested a straight arm on its elbow stop at the kinematic singularity |
| speed excess | human reaching speed, not whip-cracking | Stage A peaked at 4–5 m/s; quadratic above 1.5 m/s from Stage B; linear term added in R6 because a quadratic has zero slope at the cap and cannot hold the peak there |
| hand acceleration | bell-shaped velocity profiles | minimum-jerk proxy; brought symmetry from ~0.2 toward the human 0.35–0.55 |
| in-goal stream | pay for **holding**, not touching | only at the final waypoint, so parking at an intermediate one earns nothing |
| intermediate bonus | make completing a waypoint worth leaving it | +10 once per waypoint |
| success bonus | mark the task done | +50 once; episodes do **not** end on success in training, so the policy learns to keep holding |

There is **no co-contraction penalty**: squeezing both muscles of a pair is how a muscle-driven arm stiffens to
hold still, and the trained policy does exactly that (0.48 co-contraction holding vs 0.23 moving, R3).

### 3.4 Free-order variant (Stages F2, F3)

When the policy sees every waypoint at once and chooses the order (`free_order=True`), the potential becomes the
shortest remaining tour from the hand through the unvisited waypoints $U$,

$$\Phi(s) = -\frac{1}{R}\,\min_{\sigma \in \text{perm}(U)} \Big(\lVert p - g_{\sigma_1}\rVert + \sum_{i} \lVert g_{\sigma_i} - g_{\sigma_{i+1}}\rVert\Big),$$

(exact for up to 6 points, greedy beyond), and the dense term becomes $W_{\text{dist}}\,\Phi(s_{t+1})$ instead of
the distance to one waypoint. Both depend only on the state, so the shaping stays policy-invariant. The first F3
run keyed the dense term to the *nearest* outstanding waypoint instead; that term jumped from −0.007 to −0.25 the
moment a waypoint was completed, so finishing one was charged, and the policy learned to reach one point and stop.

### 3.5 Scale

Dense terms lie roughly in $[-1.5, +1]$ per step. The deployed model on the demonstration points (99 steps)
collects +50 success, +25 in-goal, +20 intermediate and +3.6 shaping, against −19.8 distance, −3.8 speed, −3.5
effort, −2.2 acceleration and −1.4 smoothness: **+67.9 in total**. A policy that does nothing for 500 steps scores
−355 (measured, all muscles off). (Figure 10 of the report pack.)

---

## 4. Curriculum (training difficulty) and checkpoint selection

Evaluation every 20 000 steps with seeded, randomised-start episodes (`evaluate_policy`):

1. **Training-difficulty evaluation** (5 episodes at the current rung). A rung tightens only after **two
   consecutive** rounds with success ≥ 80 %, then waits one cooldown round (the Stage A run tightened on a single
   round and several seeds collapsed):
   - position tolerance ladder $\varepsilon$: 0.15 → 0.10 → 0.07 → 0.05 m;
   - then hold ladder $c_{\text{req}}$: 10 → 15 → 25 steps.
2. **Production evaluation** (10 episodes, always at the standard: 0.05 m, 25 steps, decrement 5, stop on
   success). Its continuous score
   $$\text{score} = \big(\text{success rate},\; \overline{\text{waypoints reached}} + \overline{\text{max hold progress}},\; -\overline{\text{closest final distance}},\; -\overline{\text{steps to success}}\big)$$
   (lexicographic) selects `sac_model_best.zip`; `sac_model_last.zip`, the replay buffer, checkpoints and the
   curriculum state are also written, which is what makes a run reloadable and continuable.

---

## 5. Learning algorithm (Soft Actor-Critic, Stable-Baselines3 2.7.1)

| hyperparameter | value |
|---|---|
| policy / critics | MLP 256 × 256, `MlpPolicy`, twin Q |
| learning rate | 3 × 10⁻⁴ (Adam) |
| discount $\gamma$ | 0.99 (2 s horizon at 50 Hz; also used in the shaping term) |
| replay buffer | 300 000 transitions (600 episodes) |
| batch size | 256 |
| warm-up (random actions) | 5 000 steps (10 episodes) |
| target smoothing $\tau$ | 0.005 |
| entropy coefficient | automatic (target entropy −4 = −|A|) |
| update-to-data ratio | 1 gradient step per environment step |
| device | CPU (measured 3.5–5× faster than the GPU for this network) |
| budget | 300 000 (Stage A), 600 000 (B, B3), 1 000 000 (R3–R6), 500 000 (F2) steps per seed |
| seeds | 5 per stage (1–5), identical across stages for matched comparison |

**What SAC optimises.** Soft Actor-Critic (Haarnoja et al. 2018) maximises return plus policy entropy,

$$J(\pi) = \sum_t \mathbb{E}\big[\, r(s_t, a_t) + \alpha\,\mathcal{H}\big(\pi(\cdot \mid s_t)\big) \big],$$

which keeps the policy exploring until the reward justifies committing. Stable-Baselines3 implements it with two
critics $Q_{\theta_1}, Q_{\theta_2}$, slowly-updated target copies $\bar\theta_i$, and a tanh-squashed Gaussian
actor $\pi_\phi$ (its output is rescaled to the muscle-activation range [0, 1]). Per gradient step, on a batch
from the replay buffer:

$$y = r + \gamma\,(1 - \text{done})\Big(\min_{i=1,2} Q_{\bar\theta_i}(s', \tilde a') - \alpha \log \pi_\phi(\tilde a' \mid s')\Big), \quad \tilde a' \sim \pi_\phi(\cdot \mid s')$$

$$\mathcal{L}_Q = \tfrac{1}{2}\sum_{i=1,2} \mathbb{E}\big[(Q_{\theta_i}(s,a) - y)^2\big] \qquad \text{(critic)}$$

$$\mathcal{L}_\pi = \mathbb{E}_s\Big[\alpha \log \pi_\phi(\tilde a \mid s) - \min_{i} Q_{\theta_i}(s, \tilde a)\Big], \quad \tilde a = \tanh(\mu_\phi(s) + \sigma_\phi(s)\,\epsilon) \qquad \text{(actor)}$$

$$\mathcal{L}_\alpha = -\,\mathbb{E}\Big[\log\alpha\,\big(\log \pi_\phi(\tilde a \mid s) + \bar{\mathcal{H}}\big)\Big], \quad \bar{\mathcal{H}} = -\dim(A) = -4 \qquad \text{(temperature)}$$

$$\bar\theta_i \leftarrow \tau\,\theta_i + (1 - \tau)\,\bar\theta_i, \quad \tau = 0.005 \qquad \text{(target update)}$$

"done" is 1 only for a true termination. Training episodes end by **truncation** at 500 steps, never by success,
so the critic bootstraps through the time limit and the policy is valued for holding, not for stopping. In
practice the temperature $\alpha$ starts at 1, crashes to ~0.007 within 40K steps, climbs back and settles near
0.04 (report figure 5), and the actor loss falls steadily as the learned Q-values rise.

Seeding contract: the seed reaches SAC (its RNG and the action space), the environment (`reset(seed)`), torch
(`use_deterministic_algorithms`) and NumPy; thread count and library versions are recorded. Same manifest ⇒
bit-identical weights (`tests/test_reproducibility.py`).

---

## 6. Evaluation standard (`EVAL_STANDARD`)

Tolerance 0.05 m; hold 25 consecutive steps (0.5 s); hold decrement 5; hand speed < 0.10 m/s; joint speed
< 1.0 rad/s; intermediate settle 5 steps; 500-step episodes; 10 seeded episodes during training, 20 in the
campaign report; randomised start poses. A seed "passes" at ≥ 80 % success; a stage passes with ≥ 3 of 5 seeds.

Human-likeness metrics computed per movement segment (`kinematics_metrics.py`): path straightness
(path length / chord; human 1.00–1.15), velocity-profile symmetry (time-to-peak / movement time; human
0.35–0.55), number of speed peaks (1–2), dimensionless jerk $\sqrt{\tfrac12 \int \|\dddot p\|^2 dt \cdot T^5 / L^2}$
(< 60 for smooth reaches).

---

## 7. Results summary (for the report)

| stage | task | actuation | seeds ≥ 80 % | success over seeds | champion |
|---|---|---|---|---|---|
| A | 2 waypoints (v1) | muscle, instantaneous | 3 / 5 | 0.64 ± 0.50 | 20/20, 2.8 cm, 1.1 s |
| B | 3 waypoints (v1) | muscle, instantaneous | 2 / 5 | 0.40 ± 0.55 | 20/20, 1.2 cm, 2.1 s |
| **B3** | **3 waypoints (v2)** | **muscle + activation dynamics + accel. penalty** | **5 / 5** | **1.00 ± 0.00** | **20/20, 0.1 cm, 5.0 s** (fastest seed 1.8 s) |
| **R3** | **3 RANDOM waypoints per episode** | **muscle + activation dynamics + accel. penalty** | **5 / 5** | **0.92 ± 0.06** | **20/20 on unseen triples, 2.1 cm, 2.6 s** |
| **R4** | **3 RANDOM waypoints, end-range rule 5** | **muscle + activation dynamics + accel. penalty** | **5 / 5** | **0.97 ± 0.03** | **20/20 on unseen triples, 2.3 cm, 2.4 s** |
| R5 | R4 + human speed cost (quadratic, cap 1.2, weight 10) | muscle + activation dynamics | 3 / 5 | 0.59 ± 0.41 | 18/20, 2.9 cm; peak 1.35 m/s |
| R6 | R4 + binding speed cost (linear 3 + quadratic 5) | muscle + activation dynamics | 3 / 5 | 0.78 ± 0.24 | 19/20, 4.7 cm; peak 1.30 m/s |
| F2 | 2 random points, FREE ORDER (policy chooses the route) | muscle + activation dynamics | 0 / 5 | 0.49 ± 0.12 | 13/20; chooses both visiting orders |
| F3 | 3 random points, free order | muscle + activation dynamics | — | stopped at ~400K, 0–10 % | two attempts; reaches one point and stops |

Stages A, B and B3 train on one fixed waypoint set and memorise it: on 20 random triples the B3 champion
reaches 0.25 of 3 waypoints (0 % success, 24.7 cm final error) while the R3 champion reaches 3.00 of 3
(100 %, 2.1 cm). Only R3 answers "go to any three points I choose".

R3 and R4 cannot be compared through their own campaign numbers, because rule 5 changed which targets
exist. On the identical corrected task, matched seeds, identical episodes (100 per seed) they are
indistinguishable: R3 94.4 % +/- 3.2, R4 93.0 % +/- 5.8, paired difference -1.4 % +/- 6.9 (t = -0.46,
p = 0.67). The gain came from the rule, not from retraining on it - see RESULTS.md.

**The speed/accuracy trade-off.** Every version of the human speed cost bought human-like motion at the price
of reliability. Measured the way a user actually uses the arm — three freely clicked points, 40 trials each:

| champion (best seed) | peak hand speed | human-likeness metrics in band | reaches all 3 clicked points |
|---|---|---|---|
| R3 | 2.14 m/s | 4 of 6 | **38/40 = 95 %** |
| R5 | 1.35 m/s | **6 of 6** | 28/40 = 70 % |
| R6 | 1.30 m/s | 5 of 6 (submovements 2.5) | 17/40 = 42 % |

These are the champions — each stage's best seed, audited over 30 episodes. Across all five seeds of a stage
the picture is less flattering, and the statistic matters: by **median** over seeds R6 is inside 5 of 6 bands,
R4 and R5 4, R3 3; by **mean**, R4 is inside 5 and R5 only 3, because jerk and submovements are skewed by
single outlier episodes (R5's per-seed jerk runs from 62 to 4,771). So "R5 moves like a human" is a claim
about its best seed, not about the stage. Report figure 8 shows every seed with its median.

R3 is therefore the deployed model: the most reliable, and the one that shows human co-contraction (0.48 while
holding against 0.23 while moving). The R5 champion is the one to cite for human-like kinematics, with the
caveat above.

**Compute.** 63 training runs across 13 stages and attempts: 36.7 M environment steps and 193 seed-hours on a
CPU. Five seeds ran in parallel, so a 1 M-step stage took about 5–6 hours of wall-clock time
(R3 5.9 h, R4 5.5 h, R5 4.7 h, R6 5.2 h per seed).

Negative results kept on record: warm-starting a 3-waypoint policy from a 2-waypoint model is worse than
training from scratch; a final waypoint within the coarse curriculum tolerance of the reach circle induces a
straight-arm "rest on the elbow spring" solution that cannot be unlearned; retraining on the corrected
target set (R4) bought nothing measurable over R3 (p = 0.67); a free-order dense term keyed to the nearest
waypoint charged the arm for completing one, and it learned to stop after the first point.
