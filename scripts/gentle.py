"""
Run any command the gentle way: on chosen logical processors, in Windows efficiency mode.

    python scripts/gentle.py --cpus 16-23 -- python scripts/study_measurements.py

The command and everything it starts inherit the processor set and the below-normal priority, so
measurement and evaluation jobs can share the E-cores with training instead of loading the P-cores
(see run_campaign.py --cpus / --efficiency and scripts/hw_watchdog.py).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rl_armMotion.two_d.training.process_control import ECO_ENV, efficiency_mode, parse_cpus, set_cpus  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cpus", default="16-23", help="logical processors to use (default: the E-cores)")
    p.add_argument("--no-efficiency", action="store_true", help="skip efficiency mode")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not cmd:
        p.error("give a command after --")
    set_cpus(parse_cpus(a.cpus))
    if not a.no_efficiency:
        efficiency_mode()
        os.environ[ECO_ENV] = "1"
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
