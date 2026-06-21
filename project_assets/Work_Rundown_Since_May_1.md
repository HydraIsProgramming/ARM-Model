# RL Arm Motion Project — Work Rundown (May 1 to June 21, 2026)

**Student:** Ranjot Sandhu
**Course:** CP493
**Project:** Reinforcement Learning for Robotic Arm Motion Control

---

## Background (Prior to May 1)

Before May, the project had completed Phases 1-8:
- **Phase 1-3 (March 9-11):** Project setup, Gymnasium environment integration, visualization system
- **Phase 4-5 (March 11-16):** Interactive GUI with Tkinter + real-time arm control, dependency management
- **Phase 6 (March 16):** Physics-based constraints, jerk penalties, 2D arm environment improvements
- **Phase 7 (March 16-17):** Forward kinematics bug fix (serial chain vs rigid rod), 2-DOF conversion
- **Phase 8 (March 17-21):** Virtual task environment with workspace setup, Gymnasium API compliance, scientifically correct physics constraints with academic citations

At the start of May, the project had a working 2-DOF arm simulator with:
- 2-DOF planar arm with realistic joint constraints
- Gymnasium-compatible environment (`ArmTaskEnv`)
- Interactive GUI for visualization and control
- Basic PPO/A2C training pipeline
- 52 unit tests, all passing

---

## Week 1: May 5-9 — Fischer Protocol Integration (Steps 1-5)

### Objective
Integrate the Fischer et al. (2021) biomechanical protocol for biologically plausible arm reaching behavior.

### Work Done

**May 5 — Fischer Integration Steps 1-2:**
- Created save point of all documentation and training results before integration work
- **Step 1:** Switched default RL algorithm from PPO to SAC (Soft Actor-Critic), which is more sample-efficient for continuous control tasks. Added configurable goal tolerance parameter to `ArmTaskEnv`.
- **Step 2:** Implemented **adaptive curriculum** for training — the goal tolerance starts wide and shrinks as the agent improves. Added **motor babbling** (random exploration noise) to early training episodes. Created `AdaptiveCurriculumCallback` class for Stable-Baselines3.

**May 6 — Fischer Integration Steps 3-4:**
- **Step 3:** Built **Fitts' Law validation harness** — evaluates whether the trained agent follows Fitts' Law (movement time increases logarithmically with target distance/width). This is a key biomechanical benchmark.
- **Step 4:** Built **2/3 Power Law validation harness** — evaluates whether the agent's trajectory curvature and velocity follow the 2/3 Power Law (a fundamental motor control principle: speed decreases at curves). Implemented regression analysis and PNG plot generation.

**May 9 — Fischer Integration Step 5 + GUI Launcher:**
- **Step 5:** Implemented **Hill-type muscle dynamics** — replaced direct velocity commands with a biologically realistic antagonist-pair muscle model. Each joint has an extensor and flexor muscle with activation in [0,1]. Net torque = max_torque * (extensor - flexor). Added 7-DOF industrial arm preset.
- Created **unified GUI launcher** (`python -m rl_armMotion.two_d.gui`) for streamlined access
- Updated README and documentation

### Deliverables
- 5 Fischer integration commits
- Fitts' Law and Power Law validators with JSON/PNG output
- Hill-type muscle actuation mode
- Unified GUI launcher

---

## Week 2: May 19 — Click-to-Pick Training Mode + Documentation

### Objective
Enable interactive goal selection for training and catch up on documentation.

### Work Done

**May 19:**
- Implemented **click-to-pick goal training mode** — users can click on the visualization to set single-point goals or define waypoint sequences directly in the GUI. This makes it easy to define custom training scenarios without editing code.
- Committed the **Fischer Implementation Report** (10-page PDF) documenting the full integration
- Updated `progress.md` handoff document
- Updated the Reward System PDF for Phase 6 planning

### Deliverables
- Click-to-pick goal selection (single point + waypoint sequences)
- Fischer Implementation Report PDF (10 pages)
- Updated progress documentation

---

## Week 3: May 27 — Phase 6 Muscle Actuation in Environment + Bug Fixes

### Objective
Complete muscle actuation integration in the task environment and fix production bugs.

### Work Done

**May 27 (8 commits):**
- **Phase 6: Muscle actuation in `ArmTaskEnv`** — The Hill-type muscle model was fully integrated as an opt-in actuation mode in the Gymnasium environment. Agents can now train with `actuation_mode="muscle"` for biologically realistic control.
- **Training GUI muscle support** — Surfaced muscle actuation as a selectable option in the training GUI
- **Tier 1 polish** — Added curriculum metrics panel showing real-time tolerance, stage, and success rate. Added "Run Validators" button for one-click Fitts'/Power Law testing. README updates.
- **Bug fixes:**
  - Fixed Fischer validator plot crash on macOS — Matplotlib `pyplot` was being called from a worker thread; Tk requires main-thread rendering. Solution: deferred plot rendering to main thread.
  - Fixed spawned subprocess hang — training subprocesses inherited the launcher's process group and were killed on launcher exit. Solution: detach with `os.setpgrp()`.
  - Hardened launcher — added thread caps and log-file redirection for crash forensics
- Updated Fischer Implementation Report PDF for Phase 6 completion
- Updated progress.md handoff document

### Deliverables
- Muscle actuation mode in `ArmTaskEnv`
- Training GUI integration
- 3 bug fixes (macOS Tk crash, subprocess hang, launcher hardening)
- Updated reports and documentation

---

## Gap: May 28 - June 16

No commits during this period.

---

## Week 4: June 17-19 — Code Review, Training Infrastructure, Results

### Objective
Clean up codebase, add training CLI tools, and produce first major training results.

### Work Done

**June 17 — Code Review (2 commits):**
- **Phase 1:** Cleaned unused imports, clarified side-effect canvas binding in GUI
- **Phase 2:** Deduplicated goal-state mutation logic in `ArmTaskEnv` — the goal position was being set in multiple places with inconsistent logic; consolidated to a single `_apply_waypoint()` method

**June 18 — Training Infrastructure + First Results (10 commits):**
- Added `--actuation-mode` flag to `train_fischer_session.py` — allows choosing between velocity and muscle actuation from the command line
- Added `--waypoints` flag to `train_fischer_session.py` — enables training with arbitrary waypoint sequences (format: `"x1,y1;x2,y2;x3,y3"`)
- Built **evaluation harness** (`scripts/evaluate_trained_model.py`) — systematic model evaluation at multiple tolerances
- **200K muscle-mode training results** — First full training run with muscle actuation. Documented in `RESULTS.md`.
- **500K muscle-mode training results** — Extended training run. This became the baseline for all subsequent experiments.
- Fixed GUI model auto-detection — the GUI now detects whether a loaded model was trained with muscle or velocity actuation and configures the simulation accordingly
- Added `INSTALL.md` and reorganized requirements files for portability
- Expanded README installation section with GitHub-friendly setup guide
- Expanded `.gitignore` for Claude Code worktrees
- Added **Training Results Report PDF** for professor review

**June 19 — Documentation (2 commits):**
- Added class-level docstring to `RLTrainerWithMetrics` explaining its dual role (metrics streaming + callback management)
- Expanded inline documentation for all Fischer integration files

### Deliverables
- Training CLI with `--actuation-mode` and `--waypoints` flags
- Evaluation harness script
- 200K and 500K training results with RESULTS.md
- Training Results Report PDF
- Code review cleanup
- INSTALL.md and README improvements

---

## Week 5: June 19-21 — Waypoint Training, Optimization, and Final Model

### Objective
Train the best possible 3-waypoint model and optimize motion quality.

### Work Done

**June 19 — Best Model Trained:**
- Ran `fischer_waypoints_v2_500k` training: SAC, 500K steps, Hill-type muscle actuation, 3 waypoints [[2.3,-1.0], [1.0,1.5], [2.5,0.0]]
- **Result: 3/3 waypoints completed in 181 steps at 0.6m tolerance** — this became the project's best model
- Training took 33 minutes, produced Fitts' Law (R^2=0.3173) and Power Law (R^2=0.2070) validation results
- Curriculum stayed at stage 0 (0.6m tolerance) — the shaped reward was sufficient for learning without curriculum advancement

**June 19-20 — Improvement Attempts (all unsuccessful):**
1. **V3 training with improvements** (`improved_waypoints_500k_v3`):
   - Added EMA velocity smoothing, eval-based curriculum, co-contraction penalty
   - Result: 2/3 waypoints — worse than baseline
   - Learning: eval-based curriculum didn't help; the shaped reward is the primary learning signal

2. **Fine-tuning from v2** (`fischer_v2_finetuned_v2`):
   - Loaded best model, continued training with lr=3e-4
   - Result: **catastrophic forgetting** — model degraded from 3/3 to 2/3
   - The hold behavior is too delicate; SAC's exploration noise disrupts it

3. **Careful fine-tuning** (`fischer_v2_careful_finetune`):
   - Same approach but lr=1e-5 (30x smaller) with checkpoints every 25K
   - Result: still degraded (score 60.0 vs original 181.3)
   - Confirmed: any continued SAC training destroys the original policy

4. **Multi-seed search** (`multiseed_run_1` through `multiseed_run_5`):
   - 5 parallel training runs with different random seeds, same hyperparameters
   - Results: 0/3, 0/3, 0/3, ~2/3, ~2/3 — none matched original 3/3
   - Confirmed: RL training has **high variance**; the original was a fortunate seed

5. **Holdgrace retraining** (`fischer_v2_holdgrace_500k`):
   - Same training config with env improvements (12° orientation, hold grace -5)
   - Result: 1/3 — worse than original despite easier hold criterion
   - Higher mean reward (29,231 vs 22,187) but worse waypoint completion

**Key Insight Established:** The original model cannot be improved through any form of continued training. The improvement path is **environment-level changes applied at evaluation time**.

**June 20-21 — Environment Improvements:**
1. **Orientation tolerance: 10° to 12°** — Root cause analysis showed the hold counter was oscillating because orientation error drifted slightly past 10° every ~20 steps while position was well within tolerance. Relaxing to 12° eliminates the oscillation.

2. **Hold grace: decrement by 5 instead of hard reset** — When the arm briefly exits the goal region, the hold counter now decrements by 5 instead of resetting to 0. This prevents catastrophic loss of hold progress from single-step excursions.

3. **Action smoothing (ActionSmoother class)** — Created a `gym.ActionWrapper` that applies exponential moving average (EMA) filtering to muscle commands (alpha=0.3). Reduces jerkiness in arm trajectories with only 8 extra steps (+4.4% overhead).

4. **GUI integration** — Added "Smooth motion" checkbox and alpha slider to the ARM MODEL SELECTION panel. When enabled, the simulation wraps the environment with `ActionSmoother`.

5. **PyTorch/Tk crash fix** — Fixed a `PyEval_RestoreThread` GIL crash that occurred when closing the GUI window. PyTorch's libtorch tried to acquire the GIL during `Tcl_Exit`. Solution: `os._exit(0)` after `root.destroy()`.

### Deliverables
- **Final model:** `fischer_waypoints_v2_500k/sac_model.zip` — 3/3 waypoints, 181 steps, 100% deterministic
- 6 training experiments documented
- 3 environment improvements (orientation, hold grace, smoothing)
- ActionSmoother wrapper class
- GUI smoothing controls
- GIL crash fix

---

## Summary of All Completed Phases

| Phase | Description | Date | Status |
|---|---|---|---|
| 1 | Project setup and structure | March 9 | Complete |
| 2 | Gymnasium environment integration | March 9-11 | Complete |
| 3 | Visualization system (Matplotlib/Plotly) | March 11 | Complete |
| 4 | Interactive GUI with real-time control | March 11-16 | Complete |
| 5 | Dependency management | March 16 | Complete |
| 6 | GUI bug fixes and enhancements | March 16 | Complete |
| 7 | Forward kinematics bug fix + 2-DOF conversion | March 16-17 | Complete |
| 8 | Virtual task environment with workspace | March 17-21 | Complete |
| 9 | Fischer protocol integration (Steps 1-5) | May 5-9 | Complete |
| 10 | Click-to-pick training mode | May 19 | Complete |
| 11 | Muscle actuation in environment + bugs | May 27 | Complete |
| 12 | Code review + training infrastructure | June 17-18 | Complete |
| 13 | Model training + optimization | June 18-21 | Complete |

---

## Final System Architecture

```
rl_armMotion/
  two_d/
    config/
      arm_config.py           # ArmConfiguration dataclass, presets, JSON I/O
    environments/
      task_env.py             # ArmTaskEnv (Gymnasium), ActionSmoother, muscle actuation
      simple_arm.py           # Base arm environment
    gui/
      app.py                  # Interactive GUI (Tkinter + Matplotlib)
    models/
      callbacks.py            # Training callbacks
      trainers.py             # LinearDecaySchedule, RLTrainer
    training/
      ppo_trainer_wrapper.py  # RLTrainerWithMetrics (SAC/PPO/A2C)
      curriculum_callback.py  # AdaptiveCurriculumCallback, EvalBasedCurriculumCallback
    utils/
      arm_kinematics.py       # Forward kinematics, ArmController, MotionRecorder
      visualization.py        # ArmVisualizer, SimulationVisualizer
      parallel_env.py         # Parallel simulation utilities
    validation/
      fitts_law.py            # Fitts' Law validator
      power_law.py            # 2/3 Power Law validator

scripts/
  train_fischer_session.py    # Main training script (CLI)
  evaluate_trained_model.py   # Model evaluation harness
  finetune_from_v2.py         # Fine-tuning script
  finetune_careful.py         # Low-lr fine-tuning with checkpoints
  multi_seed_train.sh         # Multi-seed parallel training
  pick_best_seed.py           # Seed evaluation and ranking

project_assets/outputs/
  fischer_waypoints_v2_500k/  # FINAL BEST MODEL
    sac_model.zip             # Trained SAC policy
    training_history.csv      # Per-episode metrics
    training_stats.json       # Summary statistics
    fitts_law.json/png        # Fitts' Law validation
    power_law.json/png        # 2/3 Power Law validation
```

---

## Key Technical Achievements

1. **Hill-type muscle actuation** following Fischer et al. (2021) — biologically realistic 4D action space with antagonist-pair muscles
2. **3-waypoint reaching task** — agent navigates a sequence of goals with touch-and-go intermediate waypoints and full hold on the final target
3. **Shaped reward function** — 6 penalty terms + 4 bonus terms providing dense learning signal without requiring curriculum advancement
4. **Fischer validation harnesses** — Fitts' Law and 2/3 Power Law automated evaluation
5. **Action smoothing** — EMA-based wrapper for jerk reduction with minimal overhead
6. **Interactive GUI** — real-time visualization, model loading, parameter adjustment, smoothing controls
7. **Comprehensive training experiments** — systematic exploration of fine-tuning, multi-seed, and curriculum strategies with documented results

---

## Key Lessons Learned

1. **RL training has high variance** — 5 seeds with identical hyperparameters produced 0/3 to ~2/3 results; none matched the original 3/3
2. **Catastrophic forgetting in SAC** — even lr=1e-5 fine-tuning degrades a trained policy's delicate hold behavior
3. **Environment > retraining** — the best improvements came from environment-level changes (orientation tolerance, hold grace) rather than training changes
4. **Curriculum is not always needed** — the original model never advanced past curriculum stage 0; the shaped reward provided sufficient signal
5. **Deterministic evaluation is essential** — the final model produces identical results across repeated runs, confirming policy stability

---

## Commit Statistics (May 1 - June 21)

| Period | Commits | Key Focus |
|---|---|---|
| May 5-6 | 5 | Fischer integration (SAC, curriculum, validators) |
| May 9 | 4 | Hill-type muscles, GUI launcher |
| May 19 | 3 | Click-to-pick, documentation |
| May 27 | 8 | Muscle in env, GUI, bug fixes |
| June 17-19 | 14 | Code review, training infra, results, docs |
| **Total** | **34 commits** | |

### Training Experiments Conducted

| Experiment | Steps | Result |
|---|---|---|
| fischer_session (baseline) | 300K | Single goal EAST reaching |
| fischer_muscle_200k | 200K | First muscle-mode results |
| fischer_muscle_500k | 500K | Extended muscle training |
| fischer_waypoints_200k | 200K | First waypoint training |
| fischer_waypoints_v2_200k | 200K | Improved waypoint training |
| **fischer_waypoints_v2_500k** | **500K** | **BEST MODEL (3/3, 181 steps)** |
| fischer_2waypoints_500k | 500K | 2-waypoint variant |
| improved_waypoints_500k | 500K | Modified reward weights |
| improved_waypoints_500k_v3 | 500K | EMA velocity + eval curriculum |
| improved_waypoints_1M_v2 | 1M | Extended training |
| improved_waypoints_1M_v3 | 1M | V3 extended |
| fischer_waypoints_smooth_1M | 1M | Jerk penalty experiments |
| fischer_waypoints_smooth_v2_1M | 1M | Jerk penalty v2 |
| fischer_waypoints_final_1M | 1M | Final 1M attempt |
| fischer_waypoints_finetuned | varies | Fine-tune from v2 |
| fischer_v2_finetuned_v2 | 100K | Fine-tune lr=3e-4 |
| fischer_v2_careful_finetune | 100K | Fine-tune lr=1e-5 |
| fischer_v2_baseline_rerun | 500K | Reproduction attempt |
| fischer_v2_holdgrace_500k | 500K | With env improvements |
| multiseed_run_1 through _5 | 500K each | 5-seed search |
| **Total** | **~11.5M steps** | **23 experiments** |

---

*Report prepared June 21, 2026*
