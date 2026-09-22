# Empirical Results — Fischer 2021 Methodology on the 2-DOF Arm

> **Retraction notice (2026-09-21).** Everything below was measured on the earlier
> 1.8 m, gravity-free arm with the 11-D observation, before the human-arm v2
> rebuild. Those runs are historical and must not be cited as results of the
> current model: (1) the "92×" headline comparison between muscle and velocity
> actuation was confounded (100K vs 500K budgets, unmatched seeds) and is
> **reversed** by the matched-seed comparison in `docs/paper/main.tex`; (2) the
> environment had no gravity, uncoupled inertia, an inverted muscle velocity
> sign and a non-Markov observation, and its evaluations repeated one
> deterministic rollout. Results for the human-arm model are the
> Stage A / B3 / R3 sections below, produced by `run_campaign.py` with
> `campaign_report.json` in each run directory.

## Human-arm v2 - Stage R4 (end-range feasibility rule), 2026-09-22 - **PASS 5/5, and a negative result**

Stage R3's investigation turned up a real inconsistency in the task rather than in the
policy: `waypoint_feasibility` accepted targets whose holding pose lands inside a joint's
passive end-range zone, where the spring pushes the joint back out *and* the joint-limit
penalty (W_LIMIT) charges for staying. 20 % of the sampled grid was like that, 28.7 % of
targets within 0.35 m of the shoulder, some needing the elbow at its 145 deg hard stop.
Rule 5 of the spec now rejects them, the joint-space twin of the reach-circle rule.

**The rule is justified on its own.** Same R3 champion, same episode shape, 120 targets
from each group:

| targets | success | 95 % interval | median closest approach |
|---|---|---|---|
| the rule keeps | 105/120 = 88 % | 80-92 % | 0.026 m |
| the rule drops | 90/120 = 75 % | 67-82 % | 0.035 m |

The removed targets really were the hard ones, and they were hard for a reason the MDP
itself created. Removing them lifted the *unchanged* R3 champion from 37/40 = 92 % on the
old task to 59/60 = 98 % on the corrected one.

**Stage R4** retrained the same configuration from scratch on the corrected task: five
seeds x 1,000,000 steps, manifests pinned to commit `f8bb816`. Best checkpoint per seed on
20 fresh random triples:

| seed | success | final error | time to hold | straightness | symmetry |
|---|---|---|---|---|---|
| **1** | **20/20** | 2.3 cm | 2.4 s | 1.12 | 0.56 |
| 2 | 20/20 | 3.0 cm | 2.4 s | 1.15 | 0.53 |
| 3 | 19/20 | 2.4 cm | 2.5 s | 1.14 | 0.56 |
| 4 | 19/20 | 2.7 cm | 2.8 s | 1.16 | 0.51 |
| 5 | 19/20 | 2.9 cm | 2.2 s | 1.17 | 0.57 |

Mean 0.97 +/- 0.03, 5/5 seeds >= 80 % -> PASS. Champion (seed 1) in
`project_assets/outputs/CHAMPION_HUMAN_V1_R4/`. Path straightness improved to 1.12-1.17
against R3's 1.19-1.22, which puts three of the five seeds inside the human 1.0-1.15 band.

### The negative result: retraining on the corrected task bought nothing

R4's 0.97 cannot be compared with R3's 0.92, because rule 5 changed which targets exist.
Run on the **identical corrected task, matched seeds, identical episodes** (100 episodes
per seed, same evaluation seed):

| seed | R3 | R4 | R4 - R3 |
|---|---|---|---|
| 1 | 99 % | 94 % | -5 |
| 2 | 94 % | 83 % | -11 |
| 3 | 92 % | 98 % | +6 |
| 4 | 91 % | 95 % | +4 |
| 5 | 96 % | 95 % | -1 |
| **mean** | **94.4 % +/- 3.2** | **93.0 % +/- 5.8** | **-1.4 +/- 6.9** |

Paired t = -0.46, p = 0.67. **Training on the cleaner distribution made no measurable
difference**; the entire gain came from the feasibility rule, which a policy trained
before it still benefits from. A policy that has learned goal-conditioned reaching
transfers to a task with the same structure and a subset of the targets, so there was
nothing left for the retrain to teach.

Two practical consequences. First, the best single model for arbitrary points is still
**R3 seed 1 at 99 % (95-100 %)**, with R4 seed 3 next at 98 % - so
`CHAMPION_HUMAN_V1_R3` remains the demo model and `CHAMPION_HUMAN_V1_R4` is kept as the
matched record. Second, champion selection on 20 episodes is noisy: R4's campaign picked
seed 1 (94 % at n=100) over seed 3 (98 %), a difference the 20-episode eval could not see.

## Human-arm v2 — Stage R3 (ANY three points), 2026-09-22 — **PASS 5/5**

The Stage A/B/B3 models are trained on one fixed waypoint set and *memorise* it: on the
matched test below, the B3 champion reaches 0.25 of 3 waypoints when the points move.
Stage R3 trains the generalising policy — **three waypoints re-sampled from the feasible
region at every reset**, so the arm has to use the goal in its observation rather than
replay a trajectory.

Task: `ArmTaskEnv(random_waypoints=3)` samples from a 2 cm grid of 930 feasible points
(inside the reach circle with margin, inside the ROM, static hold torque <= 0.8 tau_max),
with a minimum spacing check. Five seeds x 1,000,000 steps, clean config (activation
dynamics 40/60 ms, W_ACCEL 0.02). Each seed's best checkpoint on 20 fresh random triples
at the evaluation standard (5 cm, 25-step / 0.5 s hold, hand speed < 0.10 m/s).

| seed | success | final error | time to hold | straightness | symmetry | submovements | dim. jerk |
|---|---|---|---|---|---|---|---|
| **1** | **20/20** | 2.1 cm | 2.6 s | 1.19 | 0.56 | 1.5 | 33 |
| 2 | 19/20 | 2.3 cm | 2.5 s | 1.21 | 0.58 | 1.3 | 30 |
| 3 | 18/20 | 2.4 cm | 2.4 s | 1.21 | 0.56 | 1.3 | 28 |
| 4 | 17/20 | 5.6 cm | 2.7 s | 1.22 | 0.47 | 1.3 | 1339 |
| 5 | 18/20 | 5.0 cm | 2.6 s | 1.21 | 0.52 | 1.6 | 35 |

Mean success **0.92 +/- 0.06** over seeds; **5/5 seeds >= 80 % -> PASS**. Champion: seed 1,
promoted to `project_assets/outputs/CHAMPION_HUMAN_V1_R3/` (full packages with replay
buffers in `C:/arm-model-runs/stageR3_cont/`). Every seed finished on the last tolerance
rung (5 cm) and the last hold rung (25 steps).

### Memoriser vs generaliser — same 20 random triples, same env

| policy | success | waypoints reached | final error |
|---|---|---|---|
| B3 champion (trained on 3 *fixed* points) | 0 % | 0.25 / 3 | 24.7 cm |
| **R3 champion (trained on *random* points)** | **100 %** | **3.00 / 3** | **2.1 cm** |

This is the whole point of the stage: a policy that scores 20/20 on its training waypoints
is worth nothing when the user places different ones.

### Champion failure analysis (40 fresh random triples)

`python scripts/diagnose_random_reach.py project_assets/outputs/CHAMPION_HUMAN_V1_R3 --episodes 40`

```
episode success   37/40 = 92%
waypoint 1        39/40 completed, median closest approach 0.024 m
waypoint 2        38/39 completed, 0.035 m
waypoint 3        37/38 completed (the one needing the 0.5 s hold), 0.020 m
median duration   2.6 s per three-point episode, of a 10 s budget
```

All three failures came within 1.5x the tolerance (closest approach ~0.052 m against a
0.050 m target), so what fails is the last centimetre, not the reach.

**Correction (2026-09-22).** An earlier version of this section reported a near-shoulder
weakness — `near/level` 50 %, `near/low` 80 %, `near/high` 86 % against 100 % mid-range —
and called it the next thing to fix. That was wrong. Those buckets held 2, 5 and 7 episodes:
the "50 %" was one failure out of two. Sampling each band deliberately with
`scripts/probe_region.py` shows no such gap:

| band | n | success | 95 % interval |
|---|---|---|---|
| near (r < 0.35 m) | 200 | 92 % | 87–95 % |
| mid (0.35–0.50 m) | 200 | 87 % | 82–91 % |

At n = 60 the same test put near *below* mid (87 % vs 93 %); at n = 200 it is above. The
direction flips with the sample, which is what no effect looks like. The champion is
uniformly 87–92 % across the reachable workspace, and there is no region to fix.
`diagnose_random_reach.py` now prints a Wilson interval for every region and pools any
bucket under 10 episodes, so that mistake cannot be read off the table again.

One real asymmetry did come out of the investigation, and it is a task-definition issue
rather than a policy one: `waypoint_feasibility` accepts targets whose inverse-kinematics
pose sits inside the elbow's 10° passive end-range zone — 28.7 % of near-band points and
20 % of the whole grid, some needing the elbow at its 145° hard stop. The environment
penalises end-range postures (W_LIMIT) and springs the joint back out of them, so the
sampler can ask for a pose the reward charges for. The measured cost is small and not
separable from noise (85 %, 78–90 %, for targets inside the zone against 89 %, 84–93 %,
for targets clearing it, `--split elbow`), which is why nothing was changed on the strength
of it; the honest fix, if it is ever wanted, is a ROM-margin rule in `waypoint_feasibility`
mirroring the reach-circle rule of §2.1, followed by a retrain from scratch.

### What the mid-run diagnosis showed

The run was paused at ~480K steps with seeds at 20-50 % and resumed on 2026-09-22. At the
pause the same analysis gave 40 % episode success, per-waypoint completion 85/82/57 %, and
a median closest approach of 0.058 m — *outside* the 5 cm tolerance. 11 of 12 failures were
already within 1.5x the tolerance, i.e. the arm was arriving at the right place and failing
to finish. That is why the remaining 500K steps on the 5 cm rung were spent on precision
rather than on a changed reward: closest approach tightened to 0.034 m by 740K and to
0.020-0.035 m at 1M, and success went 40 % -> 92 %.

Human-likeness: straightness 1.19-1.22 (human 1.0-1.15) is slightly above the band — a
random-point policy curves more than a policy that has memorised one path; symmetry
0.47-0.58 straddles the human 0.35-0.55; 1.3-1.6 submovements per segment matches human
point-to-point reaching. Seed 4's dimensionless jerk (1339) is a single outlier episode,
the other seeds sit at 28-35.

## Human-arm v2 — Stage A (2 waypoints), 2026-09-21

Five seeds × 300K steps, `run_campaign.py --stage A`, muscle actuation, preset `2dof_human_right`.
Each seed's best checkpoint re-evaluated on 20 fresh randomised episodes at the evaluation standard
(5 cm tolerance, 25-step / 0.5 s hold, hand speed < 0.10 m/s).

| seed | success | final error | time to hold | straightness | symmetry | dim. jerk |
|---|---|---|---|---|---|---|
| 1 | 20/20 | 4.1 cm | 1.2 s | 1.17 | 0.32 | 16 |
| 2 | 0/20 | 6.9 cm | nan s | 1.18 | 0.39 | 12 |
| 3 | 20/20 | 3.9 cm | 3.2 s | 1.34 | 0.20 | 562 |
| 4 | 4/20 | 4.8 cm | 4.1 s | 1.39 | 0.13 | 3014 |
| 5 | 20/20 | 2.8 cm | 1.1 s | 1.18 | 0.41 | 13 |

Mean success 0.64 ± 0.50 over seeds; **3/5 seeds ≥ 80 % → PASS**. Champion: seed 5
(`project_assets/outputs/CHAMPION_HUMAN_V1_A/`, full package with replay buffer in
`C:/arm-model-runs/stageA/seed_5`). Wall time 113 min for the stage (5 runs in parallel).

Observations that fed back into the code: (1) all seeds initially stalled with a straight arm on the reach
circle — fixed by a flexed elbow rest pose, an elbow end-range spring and a joint-limit penalty; (2) several
seeds passed one evaluation round, tightened a curriculum rung and collapsed — the ladder now requires two
consecutive passing rounds; (3) hand speed peaked at 4–5 m/s (symmetry 0.2–0.4 vs human ≈ 0.45) — a
quadratic penalty above 1.5 m/s was added for Stage B. Human-likeness reference ranges: straightness
1.0–1.15, symmetry 0.35–0.55, dimensionless jerk < 60.

## Human-arm v2 — Stage B3 (3 waypoints, v2 set, clean config), 2026-09-22 — **PASS 5/5**

Five seeds × 600K steps, `run_campaign.py --stage B3`: v2 waypoints A (1.40, 0.15), B (1.20, 0.40),
C (1.40, −0.20); muscle activation dynamics (τ 40/60 ms); hand-acceleration penalty (W_ACCEL 0.02);
two-round curriculum hysteresis; 1.5 m/s hand-speed cap. Best checkpoint per seed on 20 fresh randomised
episodes at the evaluation standard (5 cm, 25-step / 0.5 s hold).

| seed | success | final error | time to hold | straightness | symmetry | dim. jerk | peak speed |
|---|---|---|---|---|---|---|---|
| 1 | 20/20 | 1.7 cm | 1.8 s | 1.08 | 0.66 | 15 | 2.06 m/s |
| 2 | 20/20 | 2.2 cm | 1.9 s | 1.08 | 0.65 | 15 | 2.27 m/s |
| 3 | 20/20 | 2.9 cm | 3.0 s | 1.07 | 0.48 | 15 | 1.96 m/s |
| 4 | 20/20 | 0.1 cm | 5.0 s | 1.18 | 0.58 | 17 | 2.00 m/s |
| 5 | 20/20 | 3.5 cm | 3.3 s | 1.09 | 0.66 | 17 | 2.07 m/s |

**Every seed 20/20; mean success 1.00 ± 0.00.** Champion by score: seed 4 (1 mm final error;
seeds 1–2 are the fastest at ≈ 1.9 s). Promoted to `project_assets/outputs/CHAMPION_HUMAN_V1_B3/`; full
packages with replay buffers in `C:/arm-model-runs/stageB3/`. Seeds 1–4 were perfect in every one of their
last six evaluation rounds; seed 5 reached the standard at 560K.

Same seeds, same budget, same arm — only the waypoint geometry rule and the clean actuation differ:

| | Stage B (v1 points, plain) | Stage B3 (v2 points, clean) |
|---|---|---|
| seeds passing (≥ 80 %) | 2 / 5 | **5 / 5** |
| success rate over seeds | 0.40 ± 0.55 | **1.00 ± 0.00** |
| velocity-profile symmetry (human 0.35–0.55) | 0.37 | 0.61 |
| path straightness (human 1.0–1.15) | 1.20 | 1.10 |
| peak hand speed | 1.83 m/s | 2.07 m/s |

Straightness is inside the human band for both. Symmetry moved from front-loaded (0.28–0.43, hard burst then
creep) to 0.48–0.66 — through the human range and, for three seeds, slightly beyond it (peak late in the
move), which is the acceleration penalty trading speed for smoothness; W_ACCEL 0.01 would be the next
adjustment if a more front-loaded profile is wanted.

## Human-arm v2 — Stage B (3 waypoints, v1 set), 2026-09-21

Five seeds × 600K steps, `run_campaign.py --stage B`, muscle actuation, curriculum with two-round
hysteresis and the 1.5 m/s hand-speed cap. Best checkpoint per seed on 20 fresh randomised episodes at the
evaluation standard.

| seed | success | waypoints | final error | time to hold | straightness | symmetry | dim. jerk |
|---|---|---|---|---|---|---|---|
| 1 | 20/20 | 3.0 | 1.2 cm | 2.1 s | 1.08 | 0.28 | 32 |
| 2 | 0/20 | 2.0 | 0.6 cm | nan s | 1.65 | 0.36 | 17689 |
| 3 | 0/20 | 2.0 | 1.6 cm | nan s | 1.06 | 0.43 | 24 |
| 4 | 0/20 | 2.0 | 1.3 cm | nan s | 1.17 | 0.36 | 19 |
| 5 | 20/20 | 3.0 | 4.2 cm | 2.3 s | 1.06 | 0.43 | 18 |

Mean success 0.40 ± 0.55; **2/5 seeds ≥ 80 % → FAIL** by the campaign rule, although the two passing seeds
are perfect (seed 1: 20/20 in 2.1 s with a 1.2 cm final error, fourteen 10/10 evaluation rounds out of its
last sixteen). Champion promoted to `project_assets/outputs/CHAMPION_HUMAN_V1_B/`.

Diagnosis (the rule says fix the MDP, not add seeds): the three failing seeds pass *through* the final
waypoint to within 0.6–1.6 cm but never stop there. Two of them park motionless 10–12 cm past it with the
elbow at 2–3° — a straight arm resting on the elbow's passive end-range spring with zero elbow muscle
torque. That rest was inside the 15 cm coarse-curriculum tolerance (C sits only 9 cm inside the reach
circle), so it was rewarded early and could not be unlearned when the tolerance tightened. Fixes:
waypoint set v2 (every point ≥ 0.19 m inside the reach circle), a feasibility rule that rejects goals
closer to the reach circle than the coarsest curriculum tolerance, and an end-range spring engagement
penalty. Stage B3 re-runs the 3-waypoint task with those fixes plus the clean muscle dynamics.

## Human-arm v2 — Stage B warm-start experiment (negative result), 2026-09-21

Alongside the five from-scratch Stage B seeds, five runs were warm-started from the matching Stage A
packages (`--continue-from … --drop-buffer`: weights kept, replay buffer refilled with a 5K-step warm-up,
curriculum restarted). After ~80K Stage B steps every warm-started run was at 0/5 on the 3-waypoint task
at the 15 cm training tolerance, reaching 0–1 waypoints, while every from-scratch seed was at 5/5. The
warm runs were stopped at ~380K total steps (packages kept in `C:/arm-model-runs/stageB/seed_*_warm`).
A policy that has learned "A then C" has to unlearn the direct route once B is inserted, and refitting
the critic on a small random buffer erodes it further. Conclusion: for a new waypoint sequence, train
from scratch; use `--continue-from` (with the buffer) only to extend the same task.

---

**Run date**: 2026-06-18
**Run script**: `scripts/train_fischer_session.py`

| Run | Steps | Model | Raw data |
|-----|-------|-------|----------|
| **Primary (500K)** | 500,000 | [`fischer_muscle_500k/sac_model.zip`](project_assets/outputs/fischer_muscle_500k/sac_model.zip) | [`fischer_muscle_500k/`](project_assets/outputs/fischer_muscle_500k/) |
| Earlier (200K) | 200,000 | [`fischer_muscle_200k/sac_model.zip`](project_assets/outputs/fischer_muscle_200k/sac_model.zip) | [`fischer_muscle_200k/`](project_assets/outputs/fischer_muscle_200k/) |
| Velocity baseline | 100,000 | — | see §6 |

---

## 1. Headline Result

A SAC policy trained for **500,000** environment steps on the 2-DOF arm in **muscle mode** (Hill-type biomechanical actuation) produced a 2/3 Power Law slope of **−0.330**, against a canonical biological value of −1/3 = −0.3333. That is an absolute deviation of **0.0031 — under 1%**.

| Benchmark | Fischer 2021 (reported) | **Muscle, 500K** | Muscle, 200K | Velocity, 100K (baseline) |
|-----------|-------------------------|------------------|--------------|---------------------------|
| **Power Law slope** | −0.333 | **−0.330** | −0.389 | −0.620 |
| **Deviation from −1/3** | — | **0.0031** | 0.056 | 0.287 |
| **Power Law \|R\|** | 0.84 | **0.626** | 0.440 | 0.176 |
| **Power Law R²** | — | **0.392** | 0.194 | — |
| **Fitts' Law R²** | 0.9986 | 0.531 | 0.582 | 0.537 |

**The central finding**: holding the algorithm, reward function, and curriculum fixed and changing *only* the actuation model moves the Power Law slope from −0.620 (velocity) to −0.330 (muscle) — a 92× reduction in deviation from the biological value (0.287 → 0.0031). Because the reward function is byte-identical between the two conditions, the human-like speed–curvature relationship is attributable to the **muscle dynamics themselves**, not to reward shaping or algorithm choice. This is the central claim of Fischer's paper, reproduced here on a 2-DOF arm at 1/500th the training cost.

**Two honest caveats on this table:**

1. **The 200K and 500K runs are separate runs, not a controlled sweep.** They differ in random initialisation as well as budget, so the −0.389 → −0.330 progression is two samples rather than a measured trend. The velocity-vs-muscle contrast is robust (both muscle runs beat the velocity baseline by a wide margin); the specific rate of improvement with training budget is not established.
2. **Fitts' R² went slightly *down* from 200K to 500K** (0.582 → 0.531). This is not noise — it is explained by curriculum progress, see §3.

---

## 2. Training Run Configuration

Both muscle-mode runs used identical settings apart from the timestep budget.

| Parameter | 500K run | 200K run |
|-----------|----------|----------|
| Algorithm | SAC | SAC |
| Timesteps | **500,000** | 200,000 |
| Actuation mode | `muscle` | `muscle` |
| Goal direction | EAST (goal at `[2.80, 0.00]`) | EAST |
| Adaptive curriculum | enabled | enabled |
| Curriculum initial / min tolerance | 0.60 m / 0.02 m | 0.60 m / 0.02 m |
| Curriculum decay factor | ×0.80 | ×0.80 |
| Curriculum success threshold | 80% over 50-episode window | same |
| Motor babbling | `learning_starts = 5000` | same |
| Wall-clock training time | **30:14** | 12:49 |
| Total episodes completed | **534** | 253 |

Settings sourced from `SACAgent.DEFAULT_HYPERPARAMS` (Fischer-aligned, Haarnoja et al. 2018) and the Fischer 2021 curriculum protocol.

---

## 3. Training Convergence

| Metric | **500K run** | 200K run |
|--------|--------------|----------|
| Best episode reward | **29,012.10** | 26,671.34 |
| Mean reward (last 100 episodes) | **28,809.96** | 25,342.25 |
| Final hold progress | **0.85** | — |
| Final curriculum stage | **1** (advanced once) | 2 (advanced twice) |
| Final goal tolerance | **0.480 m** (0.60 → 0.480) | 0.384 m (0.60 → 0.480 → 0.384) |

### Why the 500K run scored lower on Fitts' Law

The 500K run advanced the curriculum **less** than the 200K run — one decay versus two — finishing at a 0.480 m tolerance rather than 0.384 m. This is the direct cause of the small Fitts' R² regression noted in §1: the Fitts validation grid includes target widths down to 0.02 m, and conditions far below the tolerance the policy actually trained at are effectively unreachable, adding scatter to the high-ID end of the fit.

Curriculum progress is therefore **not monotonic in training budget**. The curriculum advances on a *rolling success rate*, and a policy that converges to a different behaviour can plateau below the 80% threshold indefinitely regardless of how many further steps it is given. This is one of the motivations for the deterministic-evaluation curriculum introduced later in the project: the stochastic training policy's exploration noise disrupts the terminal hold and systematically under-reports competence, which suppresses curriculum advancement.

Note that this affects **Fitts' Law only**. The Power Law is measured from trajectory shape during free motion, not from terminal precision, so it is unaffected by where the tolerance curriculum stalled — which is why the 500K run improves markedly on the Power Law while regressing slightly on Fitts.

---

## 4. Fitts' Law Validation

### 4.1 Sweep parameters

| Parameter | Value |
|-----------|-------|
| Distance grid | 6 values from 0.20 m to 1.20 m |
| Width grid | 6 values from 0.02 m to 0.30 m |
| Total conditions | 36 |
| Trials per condition | 10 |
| Trial cap | 200 steps |
| Validator seed | 42 |

### 4.2 Result — 500K run (primary)

```
MT = 0.0227 + 0.0364 · ID    (R² = 0.5313)
```

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Slope b | 0.0364 s/bit | Information processing rate ≈ 27 bits/s — within human range |
| Intercept a | 0.0227 s | Small positive reaction-time offset |
| R² | 0.5313 | Moderate — see §3 for why this sits below the 200K run |
| Valid conditions in fit | 20 of 36 | Excluded conditions have W below the 0.480 m tolerance the curriculum reached |

Plot: [`fischer_muscle_500k/fitts_law.png`](project_assets/outputs/fischer_muscle_500k/fitts_law.png)

### 4.3 Result — 200K run (earlier)

```
MT = 0.0004 + 0.0448 · ID    (R² = 0.5824)
```

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Slope b | 0.0448 s/bit | ≈ 22 bits/s — within human range |
| Intercept a | 0.0004 s | Near-zero reaction time |
| R² | 0.5824 | Moderate — Fischer's 0.9986 requires much longer training |
| Valid conditions in fit | 21 of 36 | The remaining 15 had W ≤ 0.05 m, below the 0.384 m the curriculum reached |

Plot: [`fischer_muscle_200k/fitts_law.png`](project_assets/outputs/fischer_muscle_200k/fitts_law.png)

In both runs per-condition mean MT increases monotonically with the index of difficulty, confirming the qualitative Fitts relationship. Neither reaches Fischer's R², and the limiting factor is identifiable: the tolerance curriculum never reached the precision the high-ID conditions demand.

---

## 5. Two-Thirds Power Law Validation

### 5.1 Sweep parameters

| Parameter | Value |
|-----------|-------|
| Trials | 20 |
| Distance range | 0.30 m to 1.20 m |
| Trial cap | 200 steps |
| Velocity threshold (filter) | 0.01 m/s |
| Curvature threshold (filter) | 0.10 / m |
| Validator seed | 42 |

### 5.2 Result — 500K run (primary, headline number)

```
log V = −0.4521 + (−0.3302) · log C    (R = −0.6260, R² = 0.3919)
```

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Slope β | **−0.3302** | **0.94% below the canonical −0.3333 — essentially exact** |
| Pearson R | −0.626 magnitude | Below Fischer's 0.84, but a strong relationship |
| R² | 0.3919 | Substantial for biological-motion data |
| V/C samples used | 3,373 | 607 dropped by threshold filtering |

Plot: [`fischer_muscle_500k/power_law.png`](project_assets/outputs/fischer_muscle_500k/power_law.png)

This is the strongest result the project has produced. The fitted exponent is within one percent of the value observed in human arm movement, and both the exponent and the correlation improved over the 200K run.

### 5.3 Result — 200K run (earlier)

```
log V = −0.4622 + (−0.3885) · log C    (R = −0.4404, R² = 0.1939)
```

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Slope β | −0.389 | 17% above canonical −0.333 |
| Pearson R | −0.440 magnitude | Well above zero |
| R² | 0.1939 | Modest but non-trivial |
| Total V/C samples | 3,583 | After threshold filtering |

Plot: [`fischer_muscle_200k/power_law.png`](project_assets/outputs/fischer_muscle_200k/power_law.png)

In both runs the fitted regression tracks closely to the dashed −1/3 reference line through the data centroid, confirming that the policy's motion follows a power-law speed–curvature relationship near the canonical biological exponent — and the 500K run lands almost exactly on it.

---

## 6. Velocity-Mode Baseline for Comparison

An earlier SAC run trained for 100,000 timesteps in **velocity-mode actuation** (the legacy direct-velocity action space, no muscle dynamics) on the same EAST task. The validators were run on that model and produced:

| Benchmark | Velocity mode | Muscle 200K | **Muscle 500K** |
|-----------|---------------|-------------|-----------------|
| Power Law slope | −0.620 | −0.389 | **−0.330** |
| Deviation from −1/3 | 0.287 | 0.056 | **0.0031** |
| Power Law \|R\| | 0.176 | 0.440 | **0.626** |
| Fitts R² | 0.537 | 0.582 | 0.531 |

The action-space change from velocity commands to muscle activations is the only difference between these runs that affects the agent's controller — all used SAC, all used the same shaped reward, all used the adaptive curriculum from 0.60 m. The divergence in the Power Law numbers is direct evidence that **biomechanical actuation is the source of the human-like motion characteristics**, not reward shaping or algorithm choice.

Under velocity actuation the exponent of −0.620 means the policy decelerates through curves roughly twice as sharply as a human does. Under muscle actuation it lands at −0.330. The mechanism is the Hill force–velocity curve: a muscle weakens as its shortening velocity rises, imposing an intrinsic speed penalty that no reward term needs to encode. Velocity mode has no equivalent — a commanded velocity is achieved instantaneously regardless of configuration.

This matches Fischer's central methodological claim, and is the result the conference paper in [`docs/paper/`](docs/paper/) is built around.

---

## 7. Reproducing These Numbers

Reproduce the primary 500K run:

```bash
python scripts/train_fischer_session.py \
    --algorithm SAC \
    --timesteps 500000 \
    --actuation-mode muscle \
    --goal-direction EAST \
    --seed 42 \
    --save-dir ./project_assets/outputs/fischer_muscle_500k_replicate \
    --fitts-trials 10 \
    --power-trials 20 \
    --validator-seed 42
```

Training plus both validator harnesses takes about **30 minutes** for the 500K run (13 minutes for the 200K equivalent). The original runs predate the `--seed` flag, so they are not bit-reproducible; runs made from now on are. Different seeds produce moderate scatter, but the qualitative findings — Power Law slope near −1/3 in muscle mode, far from it in velocity mode, and Fitts' MT increasing monotonically with ID — are stable across seeds.

---

## 8. What Would Improve These Numbers

The Power Law result is close to its ceiling. The improvements below target Fitts' Law, which is the weaker of the two benchmarks.

1. **Get the curriculum to advance further.** This is the binding constraint, not the raw timestep budget — the 500K run stalled at 0.480 m tolerance having advanced only once, and the high-ID Fitts conditions (W = 0.02 m) are unreachable at that precision. Longer training alone does not fix this, because the curriculum gates on a rolling success rate that can plateau indefinitely. The **deterministic-evaluation curriculum** added later in the project directly addresses this by removing the exploration-noise penalty from the success measurement; re-running the EAST benchmark with it is the highest-value next experiment.

2. **Reward Option C** (hybrid): keep distance + effort + sparse success + proximity ramp, drop the velocity- and gradient-norm penalties. This would let the muscle dynamics produce their natural curvature–speed relationship without the reward also penalising fast motion. Documented in [`docs/Reward_System_Report.pdf`](docs/Reward_System_Report.pdf) §8.

3. **More trials per validator condition.** 10 trials per Fitts cell and 20 Power Law trials are conservative. Doubling these would tighten the R² estimates without changing the qualitative result.

4. **Run the actuation comparison as a controlled sweep.** The velocity/muscle contrast is currently one run per condition at differing budgets. Several seeds per condition at a matched budget would let the effect size be reported with error bars — worth doing before journal submission.

These are documented future work in [`progress.md`](progress.md) §7.3 and §7.4.

---

## 9. Bottom Line

| Question | Answer |
|----------|--------|
| Did the methodology implementation work end-to-end? | **Yes.** Training, validators, and output writes all completed cleanly. |
| Did the trained policy reproduce Fischer's empirical signatures? | **The Power Law, yes — almost exactly** (−0.330 vs canonical −0.3333, under 1% deviation). **Fitts' Law, only directionally** (R² = 0.53 vs Fischer's 0.99), limited by curriculum progress rather than by method. |
| What is the headline scientific finding? | **Changing only the actuation model — algorithm, reward, and curriculum held fixed — moves the Power Law slope from −0.620 (velocity) to −0.330 (muscle), reducing deviation from the biological −1/3 by a factor of 92.** Because the reward function is identical across conditions, biomechanical actuation rather than reward shaping is the source of the human-like motion. |
| How strong is that claim? | **Strong on direction, weak on precision.** The velocity/muscle gap is far too large to be seed noise, but with one run per condition the effect size has no error bars. See §8 item 4. |
| Is the project ready to scale? | **Yes.** All infrastructure is in place, including seeded reproducibility, deterministic-eval curriculum, and best-model checkpointing. |
