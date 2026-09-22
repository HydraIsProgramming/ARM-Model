"""
Resume the paused Stage R3 runs (random 3-waypoint task) from their packages.

    python scripts/resume_stage_r3.py                 # continue all five seeds to 1M total steps
    python scripts/resume_stage_r3.py --report        # just evaluate/report what exists

Each seed continues from C:/arm-model-runs/stageR3/seed_<k> (last model + replay
buffer + curriculum state) into C:/arm-model-runs/stageR3_cont/seed_<k>; the source
packages are never modified. The final report evaluates the continued packages.
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_STAGE = Path("C:/arm-model-runs/stageR3")
OUT_STAGE = Path("C:/arm-model-runs/stageR3_cont")
TOTAL = 1_000_000
SEEDS = [1, 2, 3, 4, 5]


def steps_done(seed):
    st = json.load(open(SRC_STAGE / f"seed_{seed}" / "curriculum_state.json"))
    return int(st.get("num_timesteps", 0))


def run(seed, threads):
    src = SRC_STAGE / f"seed_{seed}"
    dst = OUT_STAGE / f"seed_{seed}"
    remaining = max(0, TOTAL - steps_done(seed))
    if remaining == 0:
        return seed, 0, "already complete"
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts" / "train_fischer_session.py"),
           "--continue-from", str(src), "--save-dir", str(dst), "--timesteps", str(remaining),
           "--seed", str(seed), "--num-threads", str(threads), "--waypoints", "none", "--random-waypoints", "3",
           "--activation-dynamics", "0.04,0.06", "--reward-weight", "W_ACCEL=0.02",
           "--checkpoint-every", "50000", "--eval-freq", "20000", "--skip-validators", "--overwrite"]
    with open(dst / "stdout.log", "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    return seed, rc, f"{remaining:,} steps"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if not a.report:
        with ProcessPoolExecutor(max_workers=len(seeds)) as ex:
            futs = {ex.submit(run, s, a.threads): s for s in seeds}
            for f in as_completed(futs):
                seed, rc, note = f.result()
                print(f"seed {seed}: {'ok' if rc == 0 else f'FAILED rc={rc}'} ({note})", flush=True)
    subprocess.call([sys.executable, str(ROOT / "run_campaign.py"), "--stage", "R3", "--report",
                     "--out", str(OUT_STAGE if any((OUT_STAGE / f'seed_{s}' / 'run_manifest.json').exists() for s in seeds) else SRC_STAGE),
                     "--seeds", a.seeds], cwd=str(ROOT))


if __name__ == "__main__":
    main()
