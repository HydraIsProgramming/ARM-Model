"""
Matched-budget, multi-seed actuation comparison: velocity vs muscle.

Motivation
----------
The original comparison in the paper pitted a velocity run at 100K steps
against muscle runs at 200K and 500K, so training budget was confounded with
actuation model. A single matched pair at 500K (seed 42) then *reversed* the
apparent effect, and the spread across existing muscle runs (sd ~0.07, range
~0.21) turned out to exceed the effect being claimed (~0.12). The question is
therefore unresolvable at one seed per condition.

This harness runs SEEDS matched seeds in each condition at an identical
budget and reports mean +/- sd with a Welch t-test on the deviation from the
canonical -1/3 exponent, so the effect can be stated with error bars or
declared absent.

Run from the project root:
    python run_actuation_comparison.py            # run + report
    python run_actuation_comparison.py --report   # re-report existing results
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

TIMESTEPS = 500_000
# Seed 42 already has a completed pair from the first matched run; it is
# included here so that pair is reused rather than retrained.
SEEDS = [42, 201, 202, 203, 204]
GOAL = "EAST"
CONDITIONS = ["velocity", "muscle"]
CANON = -1.0 / 3.0

MAX_PARALLEL = 4
THREADS = max(1, os.cpu_count() // MAX_PARALLEL - 1)

ROOT = Path(__file__).parent
SCRIPT = str(ROOT / "scripts" / "train_fischer_session.py")
OUT = ROOT / "project_assets" / "outputs" / "ActuationComparison"
OUT.mkdir(parents=True, exist_ok=True)


def run(mode: str, seed: int) -> tuple:
    save_dir = OUT / f"{mode}_500k_seed{seed}"
    if (save_dir / "power_law.json").exists():
        print(f"  [{mode}/{seed}] already complete, skipping", flush=True)
        return mode, seed, 0
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, SCRIPT,
        "--timesteps", str(TIMESTEPS),
        "--actuation-mode", mode,
        "--goal-direction", GOAL,
        "--seed", str(seed),
        "--save-dir", str(save_dir),
        "--num-threads", str(THREADS),
        "--fitts-trials", "10",
        "--power-trials", "20",
        "--validator-seed", "42",
    ]
    print(f"  [{mode}/{seed}] starting", flush=True)
    t0 = time.time()
    with open(save_dir / "stdout.log", "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=f)
    print(f"  [{mode}/{seed}] {'ok' if rc == 0 else 'FAIL'} "
          f"({(time.time()-t0)/60:.0f} min)", flush=True)
    return mode, seed, rc


def collect(mode: str) -> list:
    rows = []
    for seed in SEEDS:
        d = OUT / f"{mode}_500k_seed{seed}"
        pw, ft, st = d / "power_law.json", d / "fitts_law.json", d / "training_stats.json"
        if not pw.exists():
            continue
        p = json.load(open(pw))
        f = json.load(open(ft)) if ft.exists() else {}
        s = json.load(open(st)) if st.exists() else {}
        rows.append({
            "seed": seed,
            "slope": p["slope"],
            "dev": abs(p["slope"] - CANON),
            "r": abs(p.get("r_pearson", 0.0)),
            "fitts": f.get("r_squared"),
            "reward": s.get("mean_reward"),
        })
    return rows


def stats(vals):
    n = len(vals)
    if n == 0:
        return None, None, 0
    m = sum(vals) / n
    if n < 2:
        return m, 0.0, n
    sd = (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5
    return m, sd, n


def report() -> None:
    data = {m: collect(m) for m in CONDITIONS}

    print(f"\n{'='*74}")
    print("MATCHED-BUDGET MULTI-SEED ACTUATION COMPARISON")
    print(f"  {TIMESTEPS:,} steps, goal {GOAL}, seeds matched across conditions")
    print(f"{'='*74}")

    for mode in CONDITIONS:
        rows = data[mode]
        print(f"\n{mode.upper()}  (n={len(rows)})")
        print(f"  {'seed':>5s} {'slope':>9s} {'dev':>8s} {'|R|':>7s} "
              f"{'fittsR2':>8s} {'mean rew':>10s}")
        for r in rows:
            fr = f"{r['fitts']:.3f}" if r["fitts"] is not None else "-"
            rw = f"{r['reward']:,.0f}" if r["reward"] is not None else "-"
            print(f"  {r['seed']:>5d} {r['slope']:>9.4f} {r['dev']:>8.4f} "
                  f"{r['r']:>7.3f} {fr:>8s} {rw:>10s}")
        m, sd, n = stats([r["dev"] for r in rows])
        ms, sds, _ = stats([r["slope"] for r in rows])
        if n:
            print(f"  slope     mean {ms:+.4f}  sd {sds:.4f}")
            print(f"  deviation mean {m:.4f}  sd {sd:.4f}")

    v_dev = [r["dev"] for r in data["velocity"]]
    m_dev = [r["dev"] for r in data["muscle"]]

    print(f"\n{'-'*74}")
    if len(v_dev) >= 2 and len(m_dev) >= 2:
        mv, sv, nv = stats(v_dev)
        mm, sm, nm = stats(m_dev)
        print(f"Deviation from canonical -1/3 (lower = more biological):")
        print(f"  velocity : {mv:.4f} +/- {sv:.4f}  (n={nv})")
        print(f"  muscle   : {mm:.4f} +/- {sm:.4f}  (n={nm})")
        try:
            from scipy import stats as sps
            t, p = sps.ttest_ind(v_dev, m_dev, equal_var=False)
            u, pu = sps.mannwhitneyu(v_dev, m_dev, alternative="two-sided")
            print(f"\n  Welch t-test      : t = {t:+.3f},  p = {p:.4f}")
            print(f"  Mann-Whitney U    : U = {u:.1f},  p = {pu:.4f}")
            verdict = ("SIGNIFICANT at alpha=0.05" if p < 0.05
                       else "NOT significant at alpha=0.05")
            print(f"  -> {verdict}")
            if p >= 0.05:
                print("  -> The data do not support an actuation effect on the")
                print("     power-law exponent at this sample size.")
            else:
                better = "velocity" if mv < mm else "muscle"
                print(f"  -> {better} is significantly closer to the biological value.")
        except Exception as exc:
            print(f"  (scipy unavailable for significance test: {exc})")

        out = {
            "timesteps": TIMESTEPS, "goal": GOAL, "seeds": SEEDS,
            "velocity": data["velocity"], "muscle": data["muscle"],
            "summary": {
                "velocity_dev_mean": mv, "velocity_dev_sd": sv,
                "muscle_dev_mean": mm, "muscle_dev_sd": sm,
            },
        }
        (OUT / "comparison_multiseed.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nSaved: {OUT / 'comparison_multiseed.json'}")
    else:
        print("Not enough completed runs for statistics yet.")


def main() -> None:
    if "--report" in sys.argv:
        report()
        return

    jobs = [(m, s) for s in SEEDS for m in CONDITIONS]
    todo = [(m, s) for m, s in jobs
            if not (OUT / f"{m}_500k_seed{s}" / "power_law.json").exists()]

    print("=" * 74)
    print("MULTI-SEED ACTUATION COMPARISON")
    print(f"  Conditions      : {CONDITIONS}")
    print(f"  Seeds           : {SEEDS} (matched across conditions)")
    print(f"  Total runs      : {len(jobs)}  ({len(todo)} remaining)")
    print(f"  Timesteps each  : {TIMESTEPS:,}")
    print(f"  Parallel        : {MAX_PARALLEL} ({THREADS} threads each)")
    print("=" * 74, flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futs = {ex.submit(run, m, s): (m, s) for m, s in todo}
        for f in as_completed(futs):
            f.result()
    print(f"\nAll runs complete in {(time.time()-t0)/3600:.1f} h", flush=True)
    report()


if __name__ == "__main__":
    main()
