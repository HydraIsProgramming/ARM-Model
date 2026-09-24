"""Shared fixtures: src on sys.path, oracle policies, small envs.

The matplotlib backend is pinned to Agg here, before anything can import
matplotlib. Without it the suite picks TkAgg, and any test that draws tries to
open a Tk connection: that fails outright when a GUI is already running (this
Python's Tcl/Tk cannot serve two apps - "Can't find a usable tk.tcl"), so
whether the tests passed depended on whether a window happened to be open.
MPLBACKEND is set rather than calling matplotlib.use() so it also applies inside
subprocesses the tests spawn.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
HERE = Path(__file__).resolve().parent
for p in (str(SRC), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rl_armMotion.two_d.environments.task_env import ArmTaskEnv  # noqa: E402
from helpers import OraclePolicy  # noqa: E402,F401


@pytest.fixture
def torque_env():
    from rl_armMotion.two_d.evaluation import make_env
    return make_env(actuation_mode="torque", waypoints=3, terminate_on_success=True)


@pytest.fixture
def muscle_env():
    from rl_armMotion.two_d.evaluation import make_env
    return make_env(actuation_mode="muscle", waypoints=2)
