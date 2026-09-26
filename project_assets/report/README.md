# Report pack — training data, figures and where the formulas are

Everything needed to write about how the arm was trained: every training run exported as plain CSV, eleven
figures drawn from those CSVs, and a pointer to the full specification of the model.

```
python scripts/export_training_data.py   # runs folder -> data/*.csv
python scripts/plot_training.py          # data/*.csv -> figures/*.png and *.pdf
```

Both scripts are idempotent. The figures read **only** the CSVs, so anyone can restyle or extend them without
the models or the 6 GB of run directories.

**[`ARM_RL_Model_Technical_Report.pdf`](ARM_RL_Model_Technical_Report.pdf)** puts it all in one document for
presenting: the model, every constraint, the full reward function, SAC's equations, how long training takes, and
the results, with all the figures plus five explanatory diagrams. Rebuild it with

```
python scripts/report_diagrams.py        # diagrams -> build/diagrams/
python scripts/build_report_pdf.py       # data + figures + diagrams -> the PDF
```

Its tables of timing, results and reward weights are read from `data/` at build time.

**The formulas** — reward function, per-stage weights, constraints, physics, muscle model, SAC's loss equations,
curriculum and evaluation — are in [`docs/MODEL_SPECIFICATION.md`](../../docs/MODEL_SPECIFICATION.md). Its §0 is a
one-table summary of every constraint; §3 is the reward.

---

## The training record in one paragraph

The arm is a two-joint, muscle-driven model of a human right arm, trained with **Soft Actor-Critic** (Stable-
Baselines3 2.7.1, 256×256 networks, CPU). **63 training runs** across 13 stages and attempts used **36.7 M
environment steps** and **193 seed-hours** of compute. Each stage ran 5 seeds in parallel; a 1 M-step stage took
about 5–6 hours of wall clock. The deployed model, **R3**, was trained for 1 M steps on three waypoints
re-sampled at random every episode, and reaches all three of a freely clicked set 95 % of the time.

| stage | what changed | seeds | steps / seed | hours / seed | seeds ≥ 80 % | mean success |
|---|---|---|---|---|---|---|
| A attempt 1 | 2 fixed points | 5 | 40 K | 0.26 | stalled | — |
| A | flexed-elbow rest, elbow spring, joint-limit cost | 5 | 300 K | 1.83 | 3 / 5 | 64 % |
| B | 3 fixed points (v1) + speed cost | 5 | 600 K | 3.79 | 2 / 5 | 40 % |
| B warm | B, warm-started from A | 5 | 364 K | 0.63 | — | worse than scratch |
| B2 | activation dynamics + acceleration cost | 3 | 160 K | 1.23 | stopped | — |
| B3 | 3 fixed points (v2 geometry) | 5 | 600 K | 3.01 | **5 / 5** | 100 % |
| **R3** | **3 random points per episode** | 5 | 1 M | 5.86 | **5 / 5** | **92 %** |
| R4 | + end-range feasibility rule | 5 | 1 M | 5.53 | 5 / 5 | 97 %* |
| R5 | + human speed cost (quadratic) | 5 | 1 M | 4.74 | 3 / 5 | 59 % |
| R6 | + binding speed cost (linear + quadratic) | 5 | 1 M | 5.20 | 3 / 5 | 78 % |
| F3 attempt 1 | 3 points, free order | 5 | 438 K | 2.35 | stopped | ~0 % |
| F3 | free order, corrected dense term | 5 | 396 K | 2.44 | stopped | ~0 % |
| F2 | 2 points, free order | 5 | 500 K | 2.27 | 0 / 5 | 49 % |

\*R4 is scored on the corrected target set; on identical episodes R3 and R4 are indistinguishable (p = 0.67).

---

## Data — `data/`

One row per observation, one column per variable. All files are UTF-8 CSV.

### `curriculum_rounds.csv` — 1,758 rows

Every evaluation round of every seed. A round runs every 20,000 environment steps.

| column | meaning |
|---|---|
| `stage`, `seed` | which run |
| `step` | environment steps at the round |
| `train_tolerance` | current curriculum rung: position tolerance, m (0.15 → 0.10 → 0.07 → 0.05) |
| `train_hold_steps` | current curriculum rung: steps to hold (10 → 15 → 25) |
| `train_success_rate` | success at the **current rung**, 5 episodes |
| `train_mean_waypoints`, `train_streak` | waypoints reached; consecutive passing rounds (2 advances the rung) |
| `prod_success_rate` | success at the **production standard** (5 cm, 25-step hold), 10 fresh episodes — the learning curve |
| `prod_mean_waypoints`, `prod_mean_max_hold` | waypoints reached; best hold progress, 0–1 |
| `prod_mean_final_distance` | closest approach to the final waypoint, m |
| `score_success` … `score_neg_steps` | the four parts of the lexicographic checkpoint score |
| `advanced`, `improved` | whether the rung tightened; whether a new best checkpoint was saved |

### `sac_training_log.csv` — 17,698 rows

Stable-Baselines3's own log, every ~2,000 steps: the training dynamics.

| column | meaning |
|---|---|
| `total_timesteps`, `episodes` | progress |
| `ep_rew_mean`, `ep_len_mean` | mean return and length of the last 100 training episodes |
| `actor_loss`, `critic_loss` | policy and Q-function losses (spec §5) |
| `ent_coef`, `ent_coef_loss` | the auto-tuned entropy temperature α and its loss |
| `fps` | environment steps per second |
| `time_elapsed` | seconds since this process started |
| `wall_clock_s` | **cumulative** seconds for the run — use this for time axes |
| `run_part` | 1, or 2 for R3's resumed half (see below) |

### `episodes.csv` — 37,960 rows

Every training episode: `episode`, `reward` (return), `length`, `goal_reached`, `goal_distance` (m). Written at the
end of a run, so stopped runs have none.

### `final_evaluation.csv` — 40 rows

Each seed's best checkpoint on 20 fresh episodes at the production standard: `success_rate`, `successes`,
`mean_waypoints_reached`, `mean_max_hold_progress`, `mean_final_distance_m`, `mean_steps_to_success`,
`best_checkpoint_step`, and the human-likeness measures `hl_straightness`, `hl_symmetry`, `hl_submovements`,
`hl_dimensionless_jerk`, `hl_movement_time`, `hl_peak_speed` (see figure 8 for their human ranges).

### `final_evaluation_episodes.csv` — 800 rows

The individual episodes behind those evaluations, including the `waypoints` each one was given (JSON).

### `run_timing.csv` — 63 rows

`total_timesteps`, `training_time_s`, `training_time_h`, `steps_per_s`, `episodes`, `parts`, `finished`.

### `stage_config.csv` — 13 rows / `reward_weights.csv` — 155 rows

Everything each stage was trained with, **read from its own manifest**: SAC hyperparameters (`sac_*`),
environment and task settings (`env_*`), arm anatomy (`arm_*`), and every reward weight (`w_*`).
`reward_weights.csv` is the same weights in long form (`stage`, `weight`, `value`) for plotting.

### `reward_terms_one_episode.csv` — 99 rows

One episode of the deployed R3 model on the demonstration points, with each reward term recorded per step
(`term_shaping`, `term_distance`, … `term_success`). Behind figures 10 and 11.

### Two things to know about the data

- **Stage R3 was paused at ~480 K steps and resumed**, into a second directory. Its logs are stitched: the SAC
  log's clock restarts at zero in the second process, so use `wall_clock_s`, not `time_elapsed`. The seam is at
  480,500 steps (`run_part` changes from 1 to 2) and is monotonic in both steps and time.
- **Evaluation returns are deliberately not exported.** Until a fix made during this export, rebuilding an older
  model switched on reward terms it had never trained with, so some evaluation files hold a return scored under
  the wrong reward, and which ones cannot be proven after the fact. Success, reach, hold and every human-likeness
  measure were never affected — they do not depend on the reward. For return, use `ep_rew_mean` from the SAC log,
  which was always computed under the reward in force at the time.

---

## Figures — `figures/`

Each figure exists as PNG (200 dpi) and PDF (vector, for LaTeX). A stage keeps the same colour in every figure —
R3 blue, R4 orange, R5 aqua, R6 yellow, a set checked for colour-blind separation. Grey lines are individual seeds;
the coloured line is their mean.

| file | shows | the finding |
|---|---|---|
| `fig01_learning_curves_random_points` | success over training, R3–R6 | R4 learns fastest; R3 and R4 finish highest; the speed cost slows R5 and R6 |
| `fig02_learning_curves_all_stages` | the same for every stage, one panel each, same scale | fixed-point stages are noisy and short; random-point stages climb steadily |
| `fig03_curriculum_tolerance` | each seed's tolerance rung over training | seeds tighten 15 → 10 → 7 → 5 cm at different times; the speed cost delays it |
| `fig04_episode_return` | training return, R3–R6 | return rises throughout; R5/R6 sit lower partly because their speed cost is itself a charge |
| `fig05_sac_internals_R3` | entropy temperature, critic loss, actor loss for R3 | α overshoots to ~0.007, recovers, settles near 0.04 |
| `fig06_training_time` | wall-clock hours per stage | 1 M-step stages take 5–6 h per seed |
| `fig07_final_success` | final success, every seed of every stage | B3, R3 and R4 pass on all 5 seeds; B and F2 do not |
| `fig08_human_likeness` | six kinematic measures against human ranges, per seed | by median over seeds R6 is inside 5 of 6 bands, R4 and R5 4, R3 3 — R3 is fast (2.0 m/s) and slightly curved |
| `fig09_speed_vs_accuracy` | peak hand speed against success, per seed | reaching human speed cost reliability in every attempt |
| `fig10_reward_terms_breakdown` | where one R3 episode's return comes from | +67.9 total; the success, in-goal and completion bonuses dominate |
| `fig11_reward_timeline` | reward per step through that episode | spikes at each waypoint and the +50 hold; +1/step while holding |
