"""
Resume a paused campaign stage from its saved packages.

    python scripts/resume_stage.py --stage R7            # continue all five seeds to the stage's full length
    python scripts/resume_stage.py --stage R7 --report   # just evaluate / report / promote what exists

A stage is paused by stopping run_campaign.py right after a checkpoint (every
100K steps), where each seed has saved its model, replay buffer and curriculum
state together. Each seed then continues from C:/arm-model-runs/stage<S>/seed_<k>
into C:/arm-model-runs/stage<S>_cont/seed_<k> with exactly the stage's settings
from run_campaign.STAGES (the env is rebuilt from these arguments, so they must
match); the source packages are never modified. The final report evaluates and
promotes the continued packages. Generalises resume_stage_r3.py.
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run_campaign import STAGES  # noqa: E402

RUNS = Path("C:/arm-model-runs")
SEEDS = [1, 2, 3, 4, 5]


def steps_done(src):
    return int(json.load(open(src / "curriculum_state.json")).get("num_timesteps", 0))


def run(stage, seed, threads):
    src = RUNS / f"stage{stage}" / f"seed_{seed}"
    dst = RUNS / f"stage{stage}_cont" / f"seed_{seed}"
    if not (src / "replay_buffer.pkl").exists():
        return seed, 1, f"no replay buffer in {src}: pause only after a 100K checkpoint"
    remaining = max(0, int(STAGES[stage]["timesteps"]) - steps_done(src))
    if remaining == 0:
        return seed, 0, "already complete"
    if (dst / "final_eval.json").exists():
        return seed, 0, "continuation already finished"
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts" / "train_fischer_session.py"),
           "--continue-from", str(src), "--save-dir", str(dst), "--timesteps", str(remaining),
           "--seed", str(seed), "--num-threads", str(threads), "--waypoints", STAGES[stage]["waypoints"],
           "--checkpoint-every", "100000", "--eval-freq", "20000", "--skip-validators", "--overwrite",
           ] + list(STAGES[stage].get("extra", []))
    with open(dst / "stdout.log", "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    return seed, rc, f"{remaining:,} steps from {steps_done(src):,}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--report", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if not a.report:
        print(f"Resuming stage {a.stage}: seeds {seeds} -> {RUNS / f'stage{a.stage}_cont'}", flush=True)
        with ProcessPoolExecutor(max_workers=len(seeds)) as ex:
            futs = {ex.submit(run, a.stage, s, a.threads): s for s in seeds}
            for f in as_completed(futs):
                seed, rc, note = f.result()
                print(f"seed {seed}: {'ok' if rc == 0 else f'FAILED rc={rc}'} ({note})", flush=True)
    subprocess.call([sys.executable, str(ROOT / "run_campaign.py"), "--stage", a.stage, "--report", "--promote",
                     "--out", str(RUNS / f"stage{a.stage}_cont"), "--seeds", a.seeds], cwd=str(ROOT))


if __name__ == "__main__":
    main()
