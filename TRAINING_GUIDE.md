# Training Guide (human-arm v2)

How to train, save, reload and demonstrate a model on the human-arm waypoint
task, and what changed from the earlier 1.8 m gravity-free arm.

---

## 1. The arm

`ArmConfiguration.get_preset("2dof_human_right")` — a right arm seen from the
right side (+x in front of the body, +y up, gravity down), Winter (2009)
anthropometry for a 75 kg / 1.75 m adult:

| | upper arm | forearm + hand (hand centre) |
|---|---|---|
| length | 0.30 m | 0.35 m |
| mass | 2.10 kg | 1.65 kg |
| COM from proximal joint | 0.131 m | 0.175 m |
| inertia about COM | 0.0196 kg m² | 0.0237 kg m² |
| range of motion | θ₀ ∈ [−150°, +90°] (flexion −60°…180°) | θ₁ ∈ [0°, 145°] |
| torque limit | 25 N·m | 12 N·m |
| muscle pair (moment arm, F_max, L_opt) | 0.04 m, 625 N, 0.20 m | 0.03 m, 400 N, 0.10 m |

Dynamics (`utils/dynamics.py`): `M(q)q̈ + C(q,q̇)q̇ + G(q) = τ + τ_passive`,
N-link generic (CRBA mass matrix, RNEA bias), RK2 integration in 4 sub-steps
of 5 ms per 20 ms control step, viscous damping 0.3 N·m·s/rad, a soft spring
(20 N·m/rad) within 10° of every limit and hard stops that lock a joint pushed
into them and zero its velocity. The relaxed arm therefore rests with the
elbow a few degrees flexed, as a human elbow does, and the start pose is
10° ± 5–10° of elbow flexion: a fully straight arm puts the hand on the reach
circle, a kinematic singularity where elbow flexion has no first-order effect
on hand distance (the first Stage-A attempt stalled there on all five seeds).
Peak gravity torque with the arm horizontal is 10.4 / 2.8 N·m (shoulder /
elbow), so the limits are 2.4× / 4.3× gravity. The 2-link closed-form
equations are the test oracle (`tests/test_dynamics.py`).

Muscles (`utils/muscle_model.py`): one lumped Hill-type muscle per direction
and joint, `F = a·F_max·f_L(L)·f_V(v)`, hyperbolic force-velocity
(a = 0.25, unloaded joint speed ≈ 12 rad/s), Gaussian force-length with the
fibre at optimal length mid-range and ≥ 40 % force at the range ends,
eccentric plateau 1.5·F_max, net torque clipped to the joint limit.
Activation is instantaneous by default (`activation_time_constants` adds a
first-order filter and appends the activations to the observation).

`3dof_human_right` adds a wrist on the same code (Stage D).

## 2. The task

Waypoints in the workspace frame (shoulder at [1.0, 0.0]):

| set | points |
|---|---|
| 1 | A = (1.45, 0.15) |
| 2 | A, C = (1.50, −0.25) |
| 3 | A, B = (1.25, 0.45), C |
| spread | the old wide triangle scaled to 0.65 m reach (first point 6° from the shoulder limit) |

- Intermediate waypoints: hand inside the tolerance and slow for 5 steps (0.1 s), then the next becomes active.
- Final waypoint: `in_goal` for 25 consecutive steps (0.5 s) = **success**. `in_goal` = distance < tolerance
  AND hand speed < 0.10 m/s AND joint speed < 1.0 rad/s. Hand orientation is not a criterion for a
  2-link arm (it is determined by position).
- Training episodes do **not** end on success: the agent is paid +1 per step for staying still at the
  final waypoint until the 500-step (10 s) limit. Evaluation and demos may stop at success.
- Evaluation standard (`evaluation.EVAL_STANDARD`): tolerance 0.05 m, hold 25 steps, decrement 5,
  20 seeded episodes with randomised start poses (±5° shoulder, 0–10° elbow).

Observation (14-D, `OBS_VERSION = 2`): sin/cos of both joints, joint velocities / 8, (goal − hand)/reach
(x, y), |goal − hand|/reach, hand velocity / 2 m/s, waypoint progress k/(N−1), in-goal flag, hold progress.

Reward: potential-based shaping `2·(γΦ' − Φ)` with `Φ = −|goal − hand|/reach`, dense `−0.5·|goal − hand|/reach`,
effort `−0.02·|a|²`, smoothness `−0.05·|q̇|/8`, end-range avoidance `−0.1` per joint resting on a hard
stop, human speed cap `−0.5·max(0, |hand velocity| − 1.5 m/s)²` (Stage A policies whipped the hand at
4–5 m/s; humans peak at 1–1.5 m/s), +1 per in-goal step at the final waypoint, +10 per intermediate
waypoint, +50 once when the hold completes. Zero action for an episode scores about −340.

## 3. Train

```bash
python scripts/train_fischer_session.py --timesteps 300000 --waypoints 2 --seed 1 --save-dir C:/arm-model-runs/stageA/seed_1
```

Defaults: SAC, net 256×256, batch 256, buffer 300K, learning_starts 5000, γ 0.99, CPU (the GPU measured
3.5–5× slower for this SAC), 4 threads. Curriculum: tolerance ladder 0.15 → 0.10 → 0.07 → 0.05 m then hold
ladder 10 → 15 → 25 steps; a rung tightens after **two consecutive** 5-episode evaluations at the
*current* difficulty succeed ≥ 80 % (`--advance-consecutive`, `--cooldown-rounds`; Stage A attempt 2
advanced on a single round and several seeds collapsed right after tightening). Every 20K steps a
10-episode evaluation at the production standard selects `sac_model_best.zip`.

Campaign (5 seeds in parallel, report, promote the champion):

```bash
python run_campaign.py --stage A
```

Stage A = 2 waypoints × 300K steps; B = 3 waypoints × 600K; C = 3 waypoints + hand load + sensory
noise/delay. Pass = ≥ 3 of 5 seeds at ≥ 80 % success. If a stage fails, fix the env/reward/curriculum;
do not add seeds. Keep training output outside OneDrive (`ARM_MODEL_OUTPUT_DIR`, default `C:/arm-model-runs`).

## 4. The package a run writes

```
<run_dir>/
  run_manifest.json      seed, threads, versions, git commit, env.describe(), resolved hyperparameters, lineage
  sac_model_best.zip     best production-eval checkpoint   <- demos load this
  sac_model_last.zip     most recent checkpoint (every eval round)
  replay_buffer.pkl      SB3 replay buffer (every 50K steps and at the end)
  curriculum_state.json  ladder indices, best score, per-round history
  best_eval.json         per-episode rows for the best checkpoint
  final_eval.json        20-episode evaluation of the best checkpoint at the end of the run
  checkpoints/step_XXXXXXX.zip
```

Reload anywhere:

```python
from rl_armMotion.two_d.evaluation import load_package, evaluate_policy
model, env, manifest = load_package("C:/arm-model-runs/stageA/seed_1", which="best", terminate_on_success=True)
print(evaluate_policy(model, env).summary_line())
```

`load_package` rebuilds the exact training env from the manifest and refuses a model whose observation
version differs from the code. Continue a run (killed or finished) into a **new** package:

```bash
python scripts/train_fischer_session.py --continue-from C:/arm-model-runs/stageA/seed_1 --timesteps 100000 --save-dir C:/arm-model-runs/stageA/seed_1_cont
```

The source package is never modified; the new manifest records its parent (`lineage`). To warm-start a
*different* task from a trained policy (e.g. 3 waypoints from a 2-waypoint model) add `--drop-buffer`:
the weights are kept, the replay buffer is refilled with a short on-policy warm-up and the curriculum
restarts for the new task (`run_campaign.py --stage B --warm-start-from C:/arm-model-runs/stageA --suffix _warm`).

## 5. Reproducibility contract

Same `run_manifest.json` (seed, `--num-threads`, library versions in `requirements-lock.txt`,
env description, resolved hyperparameters) ⇒ identical weights. The seed reaches SAC (its own RNG
and the action space), the env (`reset(seed)`), torch (`use_deterministic_algorithms`) and numpy.
`tests/test_reproducibility.py::test_same_seed_same_weights_different_seed_differs` proves it. Different
seeds with the same config are how you *measure* variance; seed search is not a method.

## 6. Demo

```bash
python scripts/show_waypoints.py C:/arm-model-runs/stageA/seed_1 --episodes 3 --gif demo_2wp.gif
python -m rl_armMotion.two_d.gui.app        # Load Model -> pick sac_model_best.zip in a package -> Run Simulation
```

The GUI draws the torso silhouette, the waypoints with their tolerance circles (green visited, red
active, grey pending), a hold-progress bar and the reached count; the waypoint selector can override
the package's set (1 / 2 / 3 / spread / custom) and "Loop" replays episodes with new seeds.

## 7. Tests

```bash
python -m pytest -q                       # everything (the reproducibility tests train small SACs, ~1 min)
python -m pytest -q -m "not slow"
```

`test_dynamics` (closed-form oracle, energy, gravity, stops), `test_muscle_model`, `test_task_env`
(bounds, seeding, reward terms, transition consistency, hovering < holding), `test_feasibility`,
`test_evaluation`, `test_curriculum`, `test_reproducibility`, `test_validators_and_demo`.

## 8. Practical notes

- Process name is `python3.11`; check package logs (`training_log.txt`, `stdout.log`) rather than `Get-Process python`.
- Long detached runs have been killed around 75 min in the past; every package is continuable to within
  20K steps (`sac_model_last.zip`) and 50K steps with its buffer.
- Models trained before human-arm v2 (`CHAMPION`, `fischer_waypoints_v2_500k`, ...) are on a different
  arm and observation; `load_package` refuses them and they are kept only as history.
