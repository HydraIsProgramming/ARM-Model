# IEEE Conference Paper — Overleaf Package

LaTeX source for *Biomechanical Actuation and Curriculum Design for
Reinforcement Learning Control of a 2-DOF Robotic Arm*.

## Contents

```
docs/paper/
├── main.tex                          # the paper (IEEEtran, conference mode)
├── README.md                         # this file
└── figures/
    ├── power_law_muscle500k.png      # Fig. 1 — 2/3 power law, 500K muscle run
    ├── fitts_law_muscle500k.png      # Fig. 2 — Fitts' Law, 500K muscle run
    └── power_law_seed001.png         # spare (waypoint seed 001), not yet cited
```

## Uploading to Overleaf

1. Go to Overleaf → **New Project** → **Blank Project** (name it whatever you like).
2. Delete the default `main.tex` that Overleaf creates.
3. Click **Upload** and drag in `main.tex`.
4. Click **New Folder**, name it exactly `figures`, open it, and upload the
   three PNGs into it. The folder name must match — `\includegraphics` paths in
   `main.tex` are written as `figures/....png`.
5. Set the compiler: **Menu → Compiler → pdfLaTeX**.
6. Click **Recompile**.

`IEEEtran` ships with Overleaf, so there is nothing else to install.

Alternatively, zip the whole `docs/paper/` folder and use Overleaf's
**New Project → Upload Project**, which preserves the folder structure
automatically. This is the faster route.

## Where the numbers come from

Every figure in the results tables is read from the JSON artefacts written by
the training and validator harnesses. Nothing is hand-estimated.

| Paper value | Source file |
|---|---|
| Muscle 500K power law slope, R, Fitts R² | `project_assets/outputs/fischer_muscle_500k/{power_law,fitts_law}.json` |
| Muscle 200K power law slope, R, Fitts R² | `RESULTS.md` §1 (run `fischer_muscle_200k`) |
| Velocity 100K baseline | `RESULTS.md` §6 |
| Waypoint baseline (v2) row | `project_assets/outputs/fischer_waypoints_v2_500k/training_stats.json` |
| Seed 001 / 003 / 05 rows | `project_assets/outputs/Parallel_Seeds/seed_*/training_stats.json` |

If you re-run training, update Tables I and II from the new JSON files rather
than editing the prose — the causal claim in Section V-A depends only on the
velocity-vs-muscle contrast, not on the absolute values.

## Things to fill in before submitting

- **Supervisor / co-author**: the author block currently lists only you. Add
  your supervisor to the `\author{}` block if the submission requires it.
- **Acknowledgment**: currently a single generic line naming CP493. Expand if
  your supervisor or a funding source should be credited.
- **Figure 3**: `power_law_seed001.png` is included in `figures/` but not yet
  referenced. Add it to Section V-C if you want a visual for the waypoint
  results.
- **RQ3 status**: Section V-C states the three-waypoint task is open at time of
  writing. Update this once a 3/3 champion is found — that will also change the
  Conclusion's first future-work item.

## Honesty notes

Three results are deliberately reported as partial rather than positive,
because that is what the data shows:

- **Fitts' Law R² = 0.531** against Fischer's 0.999. The paper attributes this
  to *curriculum advancement*, not training budget. The evidence is direct: the
  500K run advanced the curriculum once (finishing at 0.480 m) while the 200K
  run advanced it twice (0.384 m), and correspondingly scored *higher* on
  Fitts' Law despite less than half the training. Training longer does not by
  itself buy precision, because the curriculum gates on a rolling success rate
  that can plateau indefinitely.
- **The three-waypoint task is unsolved.** No model has yet achieved 3/3 with a
  full terminal hold. The paper reports the failure mode and the mitigations
  rather than claiming success.
- **n = 1 per condition.** Every row in Table I is a single training run, so
  the effect has no error bars. Section VI states this explicitly and
  Section VII makes a properly powered replication the first item of future
  work.

The central claim — that muscle actuation, not reward shaping, produces the
power-law signature — is well supported, because the reward function was
byte-identical across the two conditions and the gap (a factor of 92 in
deviation from the biological value) is far too large to attribute to seed
variation. What is *not* established is the rate at which the exponent
converges with training budget; the 200K and 500K runs differ in random
initialisation as well as budget, so that progression is two samples rather
than a trend. The paper is careful to claim only the actuation contrast.
