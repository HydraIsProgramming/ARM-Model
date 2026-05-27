# RL Arm Motion

Reinforcement-learning project for controlling a 2-DOF robotic arm in a planar workspace with a GUI-first workflow.

## 1. Project Overview

### Goal
Train an RL policy that makes the arm:
1. Reach the target height.
2. Align the end-effector horizontally.
3. Hold that pose stably for consecutive steps.

### Environment Summary
- Environment class: `rl_armMotion.two_d.environments.task_env.ArmTaskEnv`
- Action space: 2D continuous `[-1, 1]` joint velocity commands
- Observation space: 11D state vector (angles as sin/cos, velocities, task errors, gradient, hold progress)
- Reward shaping includes:
  - distance and orientation terms
  - velocity and smoothness penalties
  - hold-progress and terminal success bonus

### Available RL Algorithms in GUI
- PPO
- SAC
- A2C

## 2. Main Interfaces

### Namespace Note
- Primary 2D code now lives under `rl_armMotion.two_d.*`.
- Legacy import paths (`rl_armMotion.gui.*`, `rl_armMotion.environments.*`, etc.) are kept as compatibility wrappers.

### Unified Launcher (recommended)
- File: `src/rl_armMotion/two_d/gui/__main__.py`
- A small launcher window that dispatches to the two GUIs below as
  independent subprocesses, so each gets a clean Tk root.
- Form fields for algorithm, timestep budget, and save directory are
  filled with Fischer et al. (2021) defaults: SAC, 100k steps, save dir
  `./project_assets/outputs/fischer_session`.
- Closing the launcher does not terminate child processes (training runs
  keep going).

### Training Dashboard GUI
- File: `src/rl_armMotion/two_d/gui/training_gui.py`
- Features:
  - real-time reward/loss/entropy plots
  - live arm pose visualization
  - metrics panel (success rate, distance, hold progress, gradient)
  - model and training artifact saving
  - **Goal mode selector** with three modes:
    - **Direction** — legacy `EAST` / `WEST` / `NORTH` preset dropdown
    - **Single Point** — click anywhere on the arm canvas to place one goal
    - **Waypoints** — click multiple times to build an ordered sequence
      (`A → B → C → ...`); intermediate waypoints advance on touch, the
      final waypoint requires the standard hold criterion
  - Click-set targets persist across resets; out-of-reach clicks are
    rejected with a friendly status-bar message
  - Adaptive-curriculum progress (Fischer et al. 2021) surfaced in the
    metrics panel: current goal tolerance, curriculum stage, rolling
    success rate, episodes since last decay

### Interactive Arm GUI
- File: `src/rl_armMotion/two_d/gui/app.py`
- Features:
  - interactive arm parameter tuning
  - manual joint controls
  - motion recording/playback
  - trained-model simulation mode
  - shoulder/elbow torque vs time plot

## 3. Installation

```bash
# from project root
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 4. How To Run

### 4.1 Launch the Unified Launcher (recommended)

A single command opens a launcher window from which both GUIs can be
started. The launcher's form fields default to the Fischer et al. (2021)
training protocol implemented in this project (SAC + adaptive curriculum
+ motor babbling).

```bash
python -m rl_armMotion.two_d.gui
```

From the launcher you can:
- Click `Open Arm Control GUI` to spawn the interactive arm GUI.
- Choose algorithm (SAC / PPO / A2C), timesteps, and save directory,
  then click `Start Training GUI` to spawn the training dashboard.
- Spawn either GUI multiple times in parallel; the launcher tracks live
  PIDs in its status bar.
- Quit the launcher at any time without killing running sessions.

### 4.2 Launch Training GUI directly

```bash
python -m rl_armMotion.two_d.gui.training_gui --timesteps 100000 --algorithm SAC --save-dir ./project_assets/outputs/fischer_session
```

### 4.3 Launch Interactive Arm GUI directly

```bash
python -m rl_armMotion.two_d.gui.app
```

## 5. Training + Simulation Workflow

1. Start the training GUI.
2. Select algorithm (`PPO`/`SAC`/`A2C`) and timesteps.
3. Click `Start Training`.
4. After training, click `Save Model & Results` and choose a folder under `project_assets/outputs/...`.
5. Open the interactive arm GUI.
6. Click `Load Model` and choose the saved model (`*_model.zip`).
7. Click `Run Simulation` to run policy-driven motion.
8. Observe:
   - arm motion panel
   - live metrics panel
   - torque plots (shoulder/elbow vs time)

## 6. Repository Structure

```text
Python_projrct/
├── src/rl_armMotion/
│   ├── two_d/                        # 2D implementation (migrated from legacy root modules)
│   ├── three_d/                      # 3D scaffold package for upcoming implementation
│   └── ...                           # compatibility wrappers + shared package metadata
├── docs/                             # Guides
├── data/                             # Raw/processed/models placeholders
├── project_assets/                   # Grouped artifacts and auxiliary material
│   ├── tests/                        # Automated tests
│   ├── outputs/                      # Training outputs and saved runs
│   ├── test_images/                  # Visualization test images
│   ├── test_runs/                    # Test dashboards / standalone run scripts
│   └── examples/                     # Example scripts
├── rl_intro.html                     # RL intro page
├── arm_rl_flowchart.html             # Workflow page
├── arm_rl_architecture.html          # Architecture page
├── requirements.txt
├── pyproject.toml
└── pytest.ini
```

## 7. Running Tests

```bash
# all tests
pytest

# a specific test file
pytest project_assets/tests/test_task_env.py -v

# with coverage
pytest --cov=src/rl_armMotion
```

## 8. Notes

- Preferred model file for simulation selection: `.zip` model generated by save action.
- Training artifacts usually include model, metadata, CSV history, JSON stats, and plot PNGs.
- Documentation pages are linked:
  - `rl_intro.html`
  - `arm_rl_flowchart.html`
  - `arm_rl_architecture.html`
