# Installation Guide

**Project**: RL Arm Motion — Fischer et al. (2021) RL methodology on a 2-DOF robotic arm
**Course**: CP493 Directed Research, Wilfrid Laurier University
**Author**: Ranjot Sandhu
**Supervisor**: Professor Sukhjit Sehra

This document covers everything needed to install and run the project on a fresh machine. The full install takes about 5–10 minutes plus download time for PyTorch.

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10 or newer | Tested on Python 3.10. Versions 3.11 and 3.12 should also work. |
| pip and venv | Both ship with Python on the major installers. |
| Tkinter | Ships with Python on Windows and macOS. On Linux, install separately (see below). |
| About 3 GB of disk space | Mostly PyTorch and its dependencies. |

### Tkinter on Linux

Tkinter powers the training GUI. On most Linux distributions it must be installed separately:

```bash
# Ubuntu / Debian
sudo apt install python3-tk

# Fedora / RHEL
sudo dnf install python3-tkinter

# Arch
sudo pacman -S tk
```

On Windows and macOS the official Python installers and Homebrew Python already include Tkinter — no extra step needed.

---

## 2. Get the code

```bash
git clone <repo-url>     # or unzip the archive into a working directory
cd Project               # or whatever directory contains README.md
```

---

## 3. Create a virtual environment and install

```bash
# Create an isolated Python environment
python3 -m venv venv

# Activate it
source venv/bin/activate         # macOS / Linux
# venv\Scripts\activate.bat      # Windows cmd.exe
# venv\Scripts\Activate.ps1      # Windows PowerShell

# Upgrade pip
pip install --upgrade pip

# Install the project and its dependencies
pip install -r requirements.txt
pip install -e .
```

The `pip install -e .` step registers the project as an editable Python package so that imports like `from rl_armMotion.two_d.environments.task_env import ArmTaskEnv` resolve correctly.

If the install fails on `torch`, see the PyTorch installation page (`https://pytorch.org/get-started/locally/`) for a platform-specific install command and run that one first, then re-run `pip install -r requirements.txt`.

---

## 4. Verify the install

```bash
# Confirm the package imports
python -c "from rl_armMotion.two_d.environments.task_env import ArmTaskEnv; e = ArmTaskEnv(); print('OK:', e.action_space)"

# Confirm the test suite passes
pip install pytest                  # or: pip install -r requirements-dev.txt
pytest project_assets/tests -v
```

The expected output of the test suite is **49 passed, 1 failed**. The single failure (`test_joint_limits_enforced`) is a pre-existing float32-vs-float64 unit-in-the-last-place precision issue in the test itself; it does not indicate a problem with the install.

---

## 5. Launch the project

```bash
# All-in-one launcher (recommended)
python -m rl_armMotion.two_d.gui
```

A launcher window opens with two buttons:

- **Open Arm Control GUI** — opens the interactive arm visualisation and teleop window
- **Start Training GUI** — opens the training dashboard, defaulted to SAC, 100,000 timesteps, save directory `./project_assets/outputs/fischer_session`

For Fischer-style smooth motion, in the training window change `Actuation:` from `velocity` to `muscle` before clicking `Start Training`.

Alternative direct-launch commands (no launcher):

```bash
python -m rl_armMotion.two_d.gui.app                                  # arm control only
python -m rl_armMotion.two_d.gui.training_gui --algorithm SAC \
       --timesteps 100000 --save-dir ./project_assets/outputs/run1    # training only
```

---

## 6. Where the deliverables live

| Document | Path |
|----------|------|
| Fischer Implementation Report (PDF, 10 pages) | `docs/Fischer_Implementation_Report.pdf` |
| Fischer Implementation Presentation (PPTX, 20 slides) | `docs/Fischer_Implementation_Presentation.pptx` |
| Reward System Specification (PDF, 11 pages) | `docs/Reward_System_Report.pdf` |
| CP493 Progress Report (PDF) | `docs/CP493_Progress_Report_Ranjot_Sandhu.pdf` |
| Annotated reference report (PDF) | `docs/references/RL_ArmMotion_Physics_Reference_Report.pdf` |
| Project handoff document | `progress.md` |

---

## 7. Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `ModuleNotFoundError: No module named 'rl_armMotion'` | Forgot `pip install -e .` after activating the venv. Re-run it. |
| `ModuleNotFoundError: No module named 'tkinter'` (Linux) | Tkinter is not installed. Run the Linux command in §1. |
| Training crashes mid-run with no traceback | Check `project_assets/outputs/training_logs/` for the timestamped log file. The launcher automatically captures stdout and stderr from every spawned subprocess. |
| `OMP: Error #179: Function Can't open SHM2 failed` (macOS) | The launcher already sets `OMP_NUM_THREADS=1` and related variables to prevent this. If you see it anyway, set those variables manually before invoking Python. |
| Validators button produces a crash on macOS | Fixed in commit `da8812e` — make sure you are on the latest commit of the `claude/fischer-integration` branch. |

---

## 8. Branch state

The Fischer integration lives on the branch `claude/fischer-integration` and has not yet been merged to `main`. To check out and run that branch:

```bash
git checkout claude/fischer-integration
git log --oneline -10           # confirm you see recent Fischer commits
```

The pre-Fischer state is preserved as the immutable tag `save-point-2026-05-05` and as the branch `backup/save-point-2026-05-05` for rollback if needed.
