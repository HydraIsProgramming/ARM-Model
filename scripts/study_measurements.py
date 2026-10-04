"""
Measure every model the same way for the model study.

    python scripts/study_measurements.py                          # every CHAMPION_HUMAN_V1_* package
    python scripts/study_measurements.py --only R3,R7 --extra R8E=C:/arm-model-runs/stageR8E/seed_2

Per model, written to project_assets/report/study/<model>_*.json:
  regions   200 three-target episodes from the FULL human workspace, identical for every model
            (scripts/eval_workspace_regions.py): episode success and per-region completion
  limits    30 episodes of the human-limits audit (scripts/verify_human_limits.py --json)
  clicks    80 trials of three points clicked anywhere in reach (scripts/test_click_points.py --json)
Existing results are kept unless --redo is given, so the script can be rerun as models are added.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "project_assets" / "report" / "study"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default=None)
    p.add_argument("--extra", action="append", default=[], metavar="NAME=RUN_DIR")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--skip", default="", help="comma list of tests to skip: regions,limits,clicks")
    return p.parse_args()


def models(args):
    found = {}
    for d in sorted((ROOT / "project_assets" / "outputs").glob("CHAMPION_HUMAN_V1_*")):
        if (d / "run_manifest.json").exists():
            found[d.name.rsplit("_", 1)[-1]] = d
    for item in args.extra:
        name, _, path = item.partition("=")
        found[name.strip()] = Path(path.strip())
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        found = {k: v for k, v in found.items() if k in keep}
    return found


def run(cmd, log):
    with open(log, "w", encoding="utf-8") as f:
        return subprocess.call([sys.executable] + cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    for name, pkg in models(args).items():
        jobs = {
            "regions": ["scripts/eval_workspace_regions.py", str(pkg), "--episodes", "200",
                        "--json", str(OUT / f"{name}_regions.json")],
            "limits": ["scripts/verify_human_limits.py", str(pkg), "--episodes", "30",
                       "--json", str(OUT / f"{name}_limits.json")],
            "clicks": ["scripts/test_click_points.py", str(pkg), "--anywhere", "--trials", "80",
                       "--json", str(OUT / f"{name}_clicks.json")],
        }
        for test, cmd in jobs.items():
            target = OUT / f"{name}_{test}.json"
            if test in skip or (target.exists() and not args.redo):
                continue
            rc = run(cmd, OUT / f"{name}_{test}.log")
            print(f"{name:5s} {test:8s} {'ok' if rc == 0 and target.exists() else f'FAILED rc={rc}'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
