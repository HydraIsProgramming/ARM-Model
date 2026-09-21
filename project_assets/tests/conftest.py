"""Shared fixtures: src on sys.path, oracle policies, small envs."""

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
