# Training Guide

How to train a working model on this project, and what was wrong before.

---

## 1. The defect that broke everything (fixed)

**A 2-DOF arm cannot choose its end-effector orientation.** Two joints give two
degrees of freedom; fixing the hand's position `(x, y)` consumes both, so the
orientation `θ₀ + θ₁` is then *determined* by inverse kinematics.

The hold criterion nevertheless required position **and** orientation at once,
and the orientation it demanded was the shoulder-to-goal axis direction. That
value is correct only when the arm is fully extended. At the waypoints:

| Waypoint | Demanded | Achievable | Gap | Tolerance |
|---|---|---|---|---|
| W1 `[0.2, 1.2]` | +123.7° | +165.5° | **41.8°** | 12° |
| W2 `[2.2, 0.5]` | +22.6° | +72.9° | **50.3°** | 12° |
| W3 `[1.0, -1.6]` | −90.0° | −59.2° | **30.8°** | 12° |

Only **5–6%** of each goal region admitted a valid hold, and that share fell to
zero as the tolerance tightened — W1 and W2 become impossible below 0.4 m,
everything below 0.1 m.

This one defect explains:

- **0 of 30 seeds ever completed a hold** across ~70 hours of training.
- The old champion scoring **0/20 at 0.4 m tolerance** — not a precision limit,
  two waypoints are geometrically impossible there.
- The tolerance curriculum **always stalling** near 0.4 m.
- The capped Fitts' Law fit (its grid tests down to 0.02 m targets).

Single-target `EAST`/`WEST`/`NORTH` goals were unaffected, because they sit at
maximum reach where the two orientations coincide. That is why the motor-control
results were never in question.

### The fix has two parts, and you need both

1. **Target the achievable orientation.** `_set_explicit_goal_state` now takes
   the value inverse kinematics produces at the goal. Orientation error at each
   waypoint drops from 30–50° to **0.00°**, and feasibility now *improves* as
   tolerance tightens (100% at 0.1 m).

2. **Gate the orientation penalty by proximity.** Part 1 alone made training
   *worse*: agents that reached 2 of 3 waypoints reached **none**. The corrected
   target is a specific folded pose unrelated to travelling there, so charging
   it every step from far away sets orientation against distance and suppresses
   movement. `_orientation_gate()` ramps the penalty from 0 at three tolerance
   radii to 1 at the goal. **The hold criterion itself is unchanged and still
   strict** — only the shaping term is gated.

Check any waypoint set before spending compute:

```python
env.hold_feasibility()          # per-waypoint feasibility report
```

`run_auto_search.py` runs this automatically and refuses to start if a waypoint
is impossible.

---

## 2. Where training time actually goes

Profiled on this machine:

| | Throughput | Cost of a 500K run |
|---|---|---|
| Environment stepping | 12,172 steps/s | **0.7 min** |
| SAC gradient updates | ~120 updates/s | **~70 min** |

**99% of wall-clock is gradient updates.** SAC does one update per environment
step by default, so the simulator is irrelevant to run time. Optimising the
environment is wasted effort; the levers are network width, batch size, and the
update-to-data ratio.

### The GPU is slower — do not use it

| Config | CPU | GPU (RTX 4070 SUPER) |
|---|---|---|
| net [256,256], batch 256 | 98 upd/s | 28 upd/s |
| net [64,64], batch 64 | 134 upd/s | 26 upd/s |

**3.5–5× slower on GPU.** SAC fires many tiny kernels per update (actor, two
critics, two targets) and a small MLP at batch 64 never saturates the device —
launch latency dominates. `--device` defaults to `cpu`. **Do not set `auto`**:
Stable-Baselines3 resolves it to CUDA when a CUDA build is present, silently
selecting the slow path. Revisit only if the network and batch grow by an order
of magnitude.

### Speed knobs (measured)

| Change | Speedup | Risk |
|---|---|---|
| `--net-arch 64,64` (from 256,256) | ~1.4× | untested for final quality |
| `--batch-size 64` (from 256) | ~1.5× | untested for final quality |
| `--utd 0.5` (from 1.0) | 2× | halves optimiser work |

These were **not** used for the champion runs. Getting a working model matters
more than run time, and all three are unproven at this task. Reach for them only
when iterating on something else.

---

## 3. How to train

```bash
python run_final_search.py
```

Trains matched seeds, evaluates both the best checkpoint and the final model,
and promotes a winner to `project_assets/outputs/CHAMPION_V2/` with a
replication report. `--report` re-evaluates existing runs without retraining.

Configuration it uses, and why:

- **net [256,256], batch 256, utd 1.0** — the hyperparameters every successful
  run in this project has used.
- **hold decrement 5** — the ablation found the only 20/20 policy under this
  value, not the 3 that had been adopted on intuition.
- **device cpu** — see above.

### If the hold still will not train

Measurement on a model that reaches all three waypoints but never holds:

| Condition | Best reached | Tolerance | Fails |
|---|---|---|---|
| Position | 0.146 m | 0.60 m | 14.7% |
| **Orientation** | **0.2°** | **12°** | **95.8%** |
| Velocity | 0.003 | 0.50 | 23.7% |

Every condition is individually satisfiable — the arm passes through the correct
pose but does not settle in it. Orientation is the binding constraint, and it
was the only one of the three with no curriculum. Enable one:

```bash
--orientation-curriculum "40,25,18,12"
```

Training starts at 40° and tightens as deterministic evaluation succeeds.
Evaluation always scores at the strictest value, so reported success means the
same thing throughout.

---

## 4. Practical notes

- **Long runs get killed.** Background processes have died repeatedly around
  75 minutes. The best-checkpoint file (`sac_model_best.zip`) is written every
  25K steps, so an interrupted run is not a total loss — evaluate what it saved.
- **Process name is `python3.11`, not `python`.** `Get-Process python` returns
  nothing even while training runs. Check log file growth instead.
- **Models trained before the orientation fix are not comparable.** The fix
  changes observation component 8 (signed orientation error), so older policies
  are out of distribution and score worse than they did originally.
- **Never fine-tune a working model.** SAC exploration noise destroys the hold.
  Train from scratch.

---

## 5. Scripts

| Script | Purpose |
|---|---|
| `run_final_search.py` | Train seeds, evaluate, promote a champion |
| `run_auto_search.py` | Broader seed search, with a feasibility pre-flight |
| `run_config_ablation.py` | Isolate which hyperparameter causes a regression |
| `run_actuation_comparison.py` | Matched-budget velocity vs muscle comparison |
| `run_ablation.py` | Ablate the four terminal-hold mechanisms |
| `scripts/train_fischer_session.py` | Single training run; all CLI flags |
| `scripts/show_waypoints.py` | Animate a trained policy through the waypoints |
