# Model Specification — Human-Arm v2 (reward, constraints, physics, training)

Everything below is read from the code as of commit `607c766` on branch `human-arm-v2`
(`src/rl_armMotion/two_d/`: `config/arm_config.py`, `utils/dynamics.py`, `utils/muscle_model.py`,
`environments/task_env.py`, `training/curriculum_callback.py`, `models/trainers.py`, `evaluation.py`).
Every number is recorded in each trained model's `run_manifest.json`, so a package reproduces exactly the
reward and constraints it was trained with.

Notation: $q = (\theta_0, \theta_1)$ joint angles (rad), $\dot q$ joint velocities (rad/s), $p$ hand
position (m, workspace frame), $v = \dot p$ hand velocity, $g^{(k)}$ the active waypoint, $R = 0.65$ m the
arm's reach, $\Delta t = 0.02$ s the control step, $a \in [0,1]^4$ the action (muscle activations).

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

  Consequences: $f_L \ge 0.45$ everywhere in the range of motion (weaker at the ends, never dead); unloaded joint
  speed $\approx 12$ rad/s; equal co-contraction gives zero net torque but damps motion (a stiffening strategy the
  policy is free to use — there is no co-contraction penalty).
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

All distances are normalised by the reach $R$. Let $d_t = \|p_t - g^{(k_t)}\|$ and the potential
$\Phi(s) = -d/R$ on the augmented state (hand position, active waypoint index). Per step:

$$
\begin{aligned}
r_t \;=\;& W_{\text{shaping}}\,\big(\gamma\,\Phi(s_{t+1}) - \Phi(s_t)\big) && \text{potential-based shaping, } W = 2.0,\ \gamma = 0.99 \\
&- W_{\text{dist}}\, d_{t+1}/R && W = 0.5 \\
&- W_{\text{effort}}\, \|a_t\|^2 && W = 0.02 \\
&- W_{\text{smooth}}\, \|\dot q_{t+1}\| / 8 && W = 0.05 \\
&- W_{\text{limit}}\,\Big(\#\{\text{joints on a hard stop}\} + \sum_j e_j\Big) && W = 0.1 \\
&- W_{\text{speed}}\, \max(0,\ \|v_{t+1}\| - 1.5)^2 && W = 0.5 \text{ (human speed cap, m/s)} \\
&- W_{\text{accel}}\, \min\!\big((\|\dot v_{t+1}\|/10)^2,\ 4\big) && W = 0.02 \text{ (Stage B3; 0 in A/B)} \\
&+ R_{\text{goal}}\,[\text{in\_goal} \wedge k = N-1] && 1.0 \text{ per step, final waypoint only} \\
&+ R_{\text{inter}}\,[\text{intermediate waypoint completed this step}] && 10.0 \\
&+ R_{\text{succ}}\,[\text{hold completed this step, first time}] && 50.0
\end{aligned}
$$

where $e_j \in [0,1]$ is the fraction of the $10^\circ$ end-range spring margin joint $j$ has entered
(end-range postures are avoided by humans and were a free "prop" for a straight arm), and
$\|\dot v\| = \|v_{t+1} - v_t\|/\Delta t$ is the hand acceleration (a Flash & Hogan minimum-jerk proxy).

Design notes:
- **Shaping is two-sided and potential-based** (Ng, Harada & Russell 1999) on the augmented state, so it is
  policy-invariant; on the step a waypoint completes, $\Phi(s_t)$ uses the old goal and $\Phi(s_{t+1})$ the new
  one, and the returned observation already describes the new goal.
- **Why hovering loses.** With non-terminating episodes the in-goal stream is only available at the final
  waypoint; sitting at an intermediate waypoint earns ≈ 0/step while completing it earns +10 and unlocks the
  stream. Cycling in and out of the final ball to avoid completing the hold earns strictly less than holding
  (tested with scripted trajectories).
- **Scale.** Dense terms lie in about [−1.5, +1] per step; a three-waypoint episode that reaches the final hold at
  step 150 and holds earns ≈ +350 + 70; zero action for a full episode scores ≈ −340.
- No co-contraction penalty: co-contraction is how a muscle-driven arm stiffens to hold precisely.

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
| budget | 300 000 steps (2 waypoints), 600 000 steps (3 waypoints) |
| seeds | 5 per stage (1–5), identical across stages for matched comparison |

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

Stages A, B and B3 train on one fixed waypoint set and memorise it: on 20 random triples the B3 champion
reaches 0.25 of 3 waypoints (0 % success, 24.7 cm final error) while the R3 champion reaches 3.00 of 3
(100 %, 2.1 cm). Only R3 answers "go to any three points I choose".

R3 and R4 cannot be compared through their own campaign numbers, because rule 5 changed which targets
exist. On the identical corrected task, matched seeds, identical episodes (100 per seed) they are
indistinguishable: R3 94.4 % +/- 3.2, R4 93.0 % +/- 5.8, paired difference -1.4 % +/- 6.9 (t = -0.46,
p = 0.67). The gain came from the rule, not from retraining on it - see RESULTS.md.

Negative results kept on record: warm-starting a 3-waypoint policy from a 2-waypoint model is worse than
training from scratch; a final waypoint within the coarse curriculum tolerance of the reach circle induces a
straight-arm "rest on the elbow spring" solution that cannot be unlearned.
