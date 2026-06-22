# RL Arm Motion

**CP493 Directed Research — Wilfrid Laurier University**  
**Student:** Ranjot Sandhu | **Supervisor:** [Professor Name]

Reinforcement-learning implementation of Fischer et al. (2021) on a 2-DOF planar robotic arm. The agent learns to navigate to a sequence of 3 workspace waypoints using Hill-type muscle actuation and an adaptive curriculum — the same methodology used in Fischer's 7-DOF MuJoCo arm.

## 1. Project Overview

### Research Goal
Train a SAC policy that:
1. Visits 3 ordered waypoints in the 2D workspace (touch-and-go at intermediate waypoints).
2. Holds the final waypoint stably for 20 consecutive steps (position + orientation within tolerance + near-zero velocity).
3. Uses biomechanically realistic Hill-type muscle actuation (4 muscles: shoulder extensor/flexor + elbow extensor/flexor).

### Environment Summary
- **Environment class:** `rl_armMotion.two_d.environments.task_env.ArmTaskEnv`
- **Arm geometry:** Shoulder fixed at [1.0, 0.0] m; upper arm 1.0 m; forearm 0.8 m; max reach 1.8 m
- **Action space:** 4D continuous [0, 1] — muscle activation per antagonist pair (muscle mode) or 2D [-1,1] joint velocities (velocity mode)
- **Observation space:** 11D — [sin θ₀, cos θ₀, sin θ₁, cos θ₁, ω₀, ω₁, height_error, orientation_error, gradient_norm, in_goal_region, hold_progress]
- **Algorithm:** SAC (Soft Actor-Critic) — the only algorithm that successfully learns the hold behaviour; PPO was tested and failed
- **Curriculum:** Adaptive tolerance decay (Fischer 2021 protocol) + adaptive hold curriculum (new)

### Key Training Parameters (do not change without re-training)
| Parameter | Value | Why |
|-----------|-------|-----|
| Total timesteps | 500,000 | Empirically determined sweet spot |
| Network architecture | MLP [256, 256] | Standard SAC default |
| Hold steps required | 10 → 15 → 20 (curriculum) | Graduated hold makes learning tractable |
| Hold decrement | 3 (per step outside zone) | Reduced from 5 to avoid brittle hold |
| Orientation tolerance | 12° | Matches Fischer et al. |
| Velocity tolerance | 0.3 rad/s | Matches Fischer et al. |

### Available RL Algorithms in GUI
- **SAC** (recommended — only algorithm that solves the task)
- PPO (available but does not achieve reliable hold)
- A2C (available but does not achieve reliable hold)

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

The full install takes about 5–10 minutes on a fresh machine, mostly waiting for the PyTorch download. A standalone version of these instructions also lives in [`INSTALL.md`](INSTALL.md) for users who prefer a dedicated guide.

### 3.1 Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10 or newer** | Tested on Python 3.10. Versions 3.11 and 3.12 should also work. Pre-installed on macOS 12+ and most modern Linux distributions; on Windows, install from [python.org](https://www.python.org/downloads/). |
| **`pip` and `venv`** | Ship with any standard Python install. |
| **Tkinter** | Bundled with Python on Windows and macOS. **Linux users only** need to install it separately — see §3.2. |
| **About 3 GB of disk space** | Mostly PyTorch and its dependencies. |
| **Git** | To clone the repository. |

### 3.2 Tkinter on Linux

Tkinter powers the training GUI but is not pip-installable. On most Linux distributions it must be installed separately at the OS level:

```bash
# Ubuntu / Debian
sudo apt install python3-tk

# Fedora / RHEL
sudo dnf install python3-tkinter

# Arch
sudo pacman -S tk
```

Windows and macOS users skip this step — Tkinter is included with the standard Python installers.

### 3.3 Clone the repository

```bash
git clone <repo-url>      # replace <repo-url> with the actual GitHub URL
cd Project                # or whatever directory contains README.md
```

### 3.4 Create a virtual environment

A virtual environment keeps this project's dependencies isolated from your system Python.

```bash
# Create the environment
python3 -m venv venv

# Activate it (choose the line that matches your OS / shell)
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate.bat       # Windows cmd.exe
# venv\Scripts\Activate.ps1       # Windows PowerShell
```

You should see `(venv)` appear at the start of your shell prompt — that means the environment is active.

### 3.5 Install dependencies

```bash
# Upgrade pip to the latest version
pip install --upgrade pip

# Install third-party libraries (numpy, gymnasium, stable-baselines3, PyTorch,
# matplotlib, reportlab, etc.). This is the longest step (~5 minutes plus
# the PyTorch download).
pip install -r requirements.txt

# Register THIS project as a Python package so imports like
# `from rl_armMotion.two_d.environments.task_env import ArmTaskEnv` resolve.
pip install -e .
```

If `pip install -r requirements.txt` fails on PyTorch, follow the platform-specific install command at <https://pytorch.org/get-started/locally/> first, then re-run `pip install -r requirements.txt`.

### 3.6 Verify the install

```bash
# Confirm the package imports
python -c "from rl_armMotion.two_d.environments.task_env import ArmTaskEnv; e = ArmTaskEnv(); print('OK:', e.action_space)"

# Run the test suite
pip install pytest      # or: pip install -r requirements-dev.txt
pytest project_assets/tests -v
```

Expected output: **49 passed, 1 failed**. The single failing test (`test_joint_limits_enforced`) is a known pre-existing float32-vs-float64 unit-in-the-last-place precision issue in the test itself and does **not** indicate a problem with the install.

### 3.7 Quick launch (recommended first run)

```bash
python -m rl_armMotion.two_d.gui
```

A launcher window opens. Click **Start Training GUI** → set `Actuation:` to `muscle` for Fischer-style smooth motion → click **Start Training**. See §4 for full launch options.

### 3.8 Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `ModuleNotFoundError: No module named 'rl_armMotion'` | The `pip install -e .` step in §3.5 was skipped or the venv isn't active. Re-activate the venv and run it. |
| `ModuleNotFoundError: No module named 'tkinter'` (Linux) | Tkinter is not installed at the OS level. Run the relevant command from §3.2. |
| `pip install` errors out on `torch` | PyTorch's standard wheels do not match your platform. Install PyTorch first via the platform-specific command at <https://pytorch.org/get-started/locally/>, then re-run `pip install -r requirements.txt`. |
| Training crashes mid-run with no traceback | Check `project_assets/outputs/training_logs/` for a timestamped log file — the launcher automatically captures stdout and stderr from every spawned subprocess. |
| `OMP: Error #179: Function Can't open SHM2 failed` (macOS) | The launcher already sets `OMP_NUM_THREADS=1` and related variables to prevent this. If you still see it, set those variables manually in your shell before invoking Python. |
| Validators button crash on macOS | Pre-existing matplotlib threading bug fixed in commit `da8812e`. Make sure you are on the latest commit of the `claude/fischer-integration` branch. |

### 3.9 Documents and deliverables shipped with the project

| Document | Path |
|----------|------|
| Fischer Implementation Report (10-page PDF) | [`docs/Fischer_Implementation_Report.pdf`](docs/Fischer_Implementation_Report.pdf) |
| Fischer Implementation Presentation (20-slide PPTX) | [`docs/Fischer_Implementation_Presentation.pptx`](docs/Fischer_Implementation_Presentation.pptx) |
| Reward System Specification (11-page PDF) | [`docs/Reward_System_Report.pdf`](docs/Reward_System_Report.pdf) |
| CP493 Progress Report (PDF) | [`docs/CP493_Progress_Report_Ranjot_Sandhu.pdf`](docs/CP493_Progress_Report_Ranjot_Sandhu.pdf) |
| Annotated academic reference report (PDF) | [`docs/references/RL_ArmMotion_Physics_Reference_Report.pdf`](docs/references/RL_ArmMotion_Physics_Reference_Report.pdf) |
| Empirical results from the 2026-06-18 muscle-mode training run | [`RESULTS.md`](RESULTS.md) |
| Project handoff document | [`progress.md`](progress.md) |
| Standalone install guide | [`INSTALL.md`](INSTALL.md) |

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

## 8. Headless Batch Training (no GUI)

For long training runs (500K steps), use the CLI script directly — this is faster and logs everything to a file.

### Single seed
```bash
python scripts/train_fischer_session.py \
    --timesteps 500000 \
    --actuation-mode muscle \
    --goal-direction EAST \
    --waypoints "2.3,-1.0;1.0,1.5;2.5,0.0" \
    --save-dir ./project_assets/outputs/my_run
```

### Automated parallel seed search (recommended)
Runs 5 seeds in parallel, kills weak seeds early, auto-evaluates, stops when a 3/3 winner is found:
```bash
python run_auto_search.py
```
Outputs go to `project_assets/outputs/Parallel_Seeds/seed_NNN/`. Champion model is copied to `project_assets/outputs/CHAMPION/` with a full replication report.

### Monitor progress
```powershell
# Windows PowerShell — check all seeds in the current batch
foreach ($i in 1..5) {
    $log = "project_assets\outputs\Parallel_Seeds\seed_{0:D3}\stdout.log" -f $i
    $lines = Get-Content $log -ErrorAction SilentlyContinue
    $ts = ($lines | Select-String "total_timesteps" | Select-Object -Last 1) -replace ".*\|\s*total_timesteps\s*\|\s*(\d+)\s*\|.*",'$1'
    $rew = ($lines | Select-String "ep_rew_mean" | Select-Object -Last 1) -replace ".*\|\s*ep_rew_mean\s*\|\s*([^\s|]+)\s*\|.*",'$1'
    $pct = if ($ts -match '^\d+$') { [math]::Round([int]$ts/500000*100,1) } else { 0 }
    "seed_{0:D3}: {1}% | reward={2}" -f $i, $pct, $rew
}
```

## 9. Loading and Evaluating a Saved Model

```python
import sys; sys.path.insert(0, 'src')
from stable_baselines3 import SAC
from rl_armMotion.two_d.environments.task_env import ArmTaskEnv

model = SAC.load('project_assets/outputs/CHAMPION/sac_model')
env = ArmTaskEnv(actuation_mode='muscle', goal_direction='EAST')
env.set_waypoints([[2.3, -1.0], [1.0, 1.5], [2.5, 0.0]], tolerance=0.6)

obs, _ = env.reset()
for step in range(800):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: waypoint {info['current_waypoint_index']}/3 | hold {info['hold_counter']}/{info['hold_steps_required']}")
    if terminated:
        print(f"SUCCESS — 3/3 waypoints in {step} steps")
        break
```

## 10. Important Warnings

- **NEVER fine-tune a working model.** SAC's exploration noise destroys the hold behaviour. Each new training phase must start from scratch.
- **Do NOT change the reward weights** without re-training from scratch — the policy is sensitive to the 10-term reward structure.
- **Do NOT change the observation space** — the 11D structure is what the policy was trained on.
- The champion model lives in `project_assets/outputs/CHAMPION/` — never overwrite it.

## 11. Notes

- Preferred model file for simulation: `.zip` model generated by save action or `run_auto_search.py`
- Training artefacts per run: `sac_model.zip`, `training_history.csv`, `training_stats.json`, `fitts_law.png`, `power_law.png`
- Documentation pages:
  - `rl_intro.html` — RL concepts intro
  - `arm_rl_flowchart.html` — training workflow diagram
  - `arm_rl_architecture.html` — system architecture diagram
