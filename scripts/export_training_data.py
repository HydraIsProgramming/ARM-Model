"""
Export every training run into plain CSV files you can plot anywhere.

    python scripts/export_training_data.py
    python scripts/export_training_data.py --runs C:/arm-model-runs --out project_assets/report/data

Walks every stage directory under the runs folder and writes one tidy table per
kind of measurement - one row per observation, one column per variable - so any
of them loads straight into Excel, pandas, R or MATLAB with no reshaping.

    curriculum_rounds.csv          every evaluation round (each 20,000 steps) of every seed
    sac_training_log.csv           Stable-Baselines3's own log: reward, losses, entropy, speed
    episodes.csv                   every training episode's return and outcome
    final_evaluation.csv           each seed's 20-episode evaluation at the production standard
    final_evaluation_episodes.csv  the individual episodes behind those evaluations
    run_timing.csv                 wall-clock time and throughput per seed
    stage_config.csv               the exact configuration every stage was trained with
    reward_weights.csv             the same reward weights in long form, one row per weight

Two stages need care, and the script handles both rather than leaving it to the
reader. Stage R3 was paused at ~480K steps and resumed into a second directory,
so its logs are stitched: the SAC log's wall clock restarts at zero in the
second process, and the overlap between the two is resolved in favour of the
run that continued. Stage B holds two experiments in one directory, the plain
runs (seed_k) and the warm-start runs (seed_k_warm), which are exported as
separate stages B and B_warm.

The configuration comes from each run's own manifest, never from the current
code. The reward weights changed between stages - the deployed R3 model was
trained with a 1.5 m/s speed cap at weight 0.5, not the 1.2 m/s linear-plus-
quadratic cost the code carries now - so reading them from source would
misreport what any given model was actually trained on.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent

# stage key -> (directories in run order, seed-dir suffix, what it was, kept as the record of a failure?)
STAGES = [
    ("A_attempt1", ["stageA_attempt1_straight_arm_stall"], "",
     "2 fixed waypoints; first attempt - every seed stalled with a straight arm", True),
    ("A", ["stageA"], "", "2 fixed waypoints (v1 set)", False),
    ("B", ["stageB"], "", "3 fixed waypoints (v1 set)", False),
    ("B_warm", ["stageB"], "_warm", "3 fixed waypoints, warm-started from Stage A", True),
    ("B2", ["stageB2"], "", "3 fixed waypoints (v1), activation dynamics + accel. penalty; stopped", True),
    ("B3", ["stageB3"], "", "3 fixed waypoints (v2 set), clean config", False),
    ("R3", ["stageR3", "stageR3_cont"], "", "3 RANDOM waypoints per episode (the deployed model)", False),
    ("R4", ["stageR4"], "", "3 random waypoints + end-range feasibility rule", False),
    ("R5", ["stageR5"], "", "R4 + human hand-speed cost (quadratic, cap 1.2, weight 10)", False),
    ("R6", ["stageR6"], "", "R4 + binding speed cost (linear 3 + quadratic 5, cap 1.2)", False),
    ("F3_attempt1", ["stageF3_attempt1_reward_cliff"], "",
     "3 random points, FREE ORDER; dense term keyed to the nearest point (reward cliff)", True),
    ("F3", ["stageF3"], "", "3 random points, FREE ORDER, dense term = remaining tour; crashed at ~400K", True),
    ("F2", ["stageF2"], "", "2 random points, FREE ORDER pilot", False),
]

SAC_FIELDS = ["ep_len_mean", "ep_rew_mean", "episodes", "fps", "time_elapsed", "total_timesteps",
              "actor_loss", "critic_loss", "ent_coef", "ent_coef_loss", "learning_rate", "n_updates"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="C:/arm-model-runs", help="folder holding the stage<X> directories")
    p.add_argument("--out", default=str(ROOT / "project_assets" / "report" / "data"))
    return p.parse_args()


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def seeds_of(stage_dir: Path, suffix: str) -> List[int]:
    pat = re.compile(rf"^seed_(\d+){re.escape(suffix)}$")
    out = []
    for d in stage_dir.glob("seed_*"):
        m = pat.match(d.name)
        if m and d.is_dir():
            out.append(int(m.group(1)))
    return sorted(out)


def parse_sac_log(path: Path) -> List[Dict[str, float]]:
    """Stable-Baselines3 prints a boxed table every few episodes; turn each
    table into one row. A table is complete once n_updates (or total_timesteps,
    before learning starts) has been read."""
    rows, cur = [], {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\|\s+(\w+)\s+\|\s+(-?[\d.eE+-]+)\s+\|", line)
        if m and m.group(1) in SAC_FIELDS:
            try:
                cur[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
        elif line.startswith("---") and cur.get("total_timesteps") is not None:
            rows.append(cur)
            cur = {}
    if cur.get("total_timesteps") is not None:
        rows.append(cur)
    # a table is printed with its fields in one block; keep one row per step count
    dedup: Dict[float, Dict[str, float]] = {}
    for r in rows:
        dedup[r["total_timesteps"]] = {**dedup.get(r["total_timesteps"], {}), **r}
    return [dedup[k] for k in sorted(dedup)]


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> int:
    if not rows:
        return 0
    cols = columns or list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    return len(rows)


def flat(prefix: str, d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (dict, list, tuple)):
            out[f"{prefix}{k}"] = json.dumps(v)
        else:
            out[f"{prefix}{k}"] = v
    return out


def main():
    args = parse_args()
    runs, out = Path(args.runs), Path(args.out)
    rounds, sac, episodes, finals, final_eps, timing, config, weights = ([] for _ in range(8))

    for stage, dirs, suffix, what, failure in STAGES:
        dirs_p = [runs / d for d in dirs if (runs / d).exists()]
        if not dirs_p:
            print(f"  {stage:<12} (not on disk, skipped)")
            continue
        seeds = sorted(set().union(*[seeds_of(d, suffix) for d in dirs_p]))
        n_rounds = n_sac = 0
        for seed in seeds:
            sdirs = [d / f"seed_{seed}{suffix}" for d in dirs_p if (d / f"seed_{seed}{suffix}").exists()]
            key = dict(stage=stage, seed=seed)

            # -- curriculum rounds: merged across directories, later wins on a shared step
            merged: Dict[int, Dict[str, Any]] = {}
            for sd in sdirs:
                st = read_json(sd / "curriculum_state.json") or {}
                for r in st.get("history", []):
                    step = int(r.get("step", 0))
                    row = {**key, "step": step}
                    for k in ("train_tolerance", "train_hold_steps", "train_success_rate",
                              "train_mean_waypoints", "train_streak", "prod_success_rate",
                              "prod_mean_waypoints", "prod_mean_max_hold", "prod_mean_final_distance",
                              "advanced", "improved"):
                        row[k] = r.get(k)
                    score = r.get("prod_score") or []
                    for i, name in enumerate(("score_success", "score_progress",
                                              "score_neg_distance", "score_neg_steps")):
                        row[name] = score[i] if i < len(score) else None
                    merged[step] = row
            rounds.extend(merged[s] for s in sorted(merged))
            n_rounds += len(merged)

            # -- SAC log: stitched in run order, wall clock made cumulative
            offset, prev_end, stitched = 0.0, None, []
            for i, sd in enumerate(sdirs):
                log = parse_sac_log(sd / "stdout.log")
                if not log:
                    continue
                if stitched and i > 0:
                    first = log[0]["total_timesteps"]
                    stitched = [r for r in stitched if r["total_timesteps"] < first]
                    offset = prev_end
                for r in log:
                    r = dict(r)
                    r["wall_clock_s"] = offset + r.get("time_elapsed", 0.0)
                    r["run_part"] = i + 1
                    stitched.append(r)
                prev_end = stitched[-1]["wall_clock_s"] if stitched else offset
            for r in stitched:
                sac.append({**key, **r})
            n_sac += len(stitched)

            # -- training episodes (written by the last directory that finished)
            for sd in reversed(sdirs):
                p = sd / "training_history.csv"
                if p.exists():
                    with p.open(encoding="utf-8") as f:
                        for r in csv.DictReader(f):
                            episodes.append({**key, **r})
                    break

            # -- final evaluation (the campaign's 20 fresh episodes)
            last = sdirs[-1]
            ev = read_json(last / "campaign_eval.json")
            if ev:
                s = ev.get("summary", {})
                row = {**key, "n_episodes": s.get("n_episodes"), "success_rate": s.get("success_rate"),
                       "successes": s.get("successes"),
                       "mean_waypoints_reached": s.get("mean_waypoints_reached"),
                       "mean_max_hold_progress": s.get("mean_max_hold_progress"),
                       "mean_final_distance_m": s.get("mean_min_final_distance"),
                       "mean_steps_to_success": s.get("mean_steps_to_success")}
                # Returns are deliberately NOT exported from evaluations. Until the
                # legacy-reward fix, rebuilding an older model switched on reward
                # terms it never trained with, so an evaluation file re-run after
                # those terms were added holds a return scored under the wrong
                # reward - and which files those are cannot be proven after the
                # fact. Training return comes from the SAC log instead, which was
                # always computed under the reward in force at the time.
                row.update({f"hl_{k}": v for k, v in (s.get("human_likeness") or {}).items()
                            if not isinstance(v, (dict, list))})
                st = read_json(last / "curriculum_state.json") or {}
                row["best_checkpoint_step"] = st.get("best_saved_at")
                finals.append(row)
                for e in ev.get("episodes", []):
                    er = {**key, **{k: v for k, v in e.items()
                                    if k not in ("human_likeness", "waypoints", "episode_return")}}
                    er["waypoints"] = json.dumps(e.get("waypoints"))
                    er.update({f"hl_{k}": v for k, v in (e.get("human_likeness") or {}).items()
                               if not isinstance(v, (dict, list))})
                    final_eps.append(er)

            # -- timing
            total_t, total_steps, total_eps = 0.0, 0, 0
            for sd in sdirs:
                ts = read_json(sd / "training_stats.json")
                if ts and ts.get("training_time_sec"):
                    total_t += float(ts["training_time_sec"])
                    total_eps = max(total_eps, int(ts.get("total_episodes") or 0))
                else:
                    log = parse_sac_log(sd / "stdout.log")
                    if log:
                        total_t += float(log[-1].get("time_elapsed", 0.0))
            st = read_json(sdirs[-1] / "curriculum_state.json") or {}
            total_steps = int(st.get("num_timesteps") or (stitched[-1]["total_timesteps"] if stitched else 0))
            timing.append({**key, "total_timesteps": total_steps, "training_time_s": round(total_t, 1),
                           "training_time_h": round(total_t / 3600.0, 3),
                           "steps_per_s": round(total_steps / total_t, 1) if total_t else None,
                           "episodes": total_eps or (stitched[-1].get("episodes") if stitched else None),
                           "parts": len(sdirs), "finished": (sdirs[-1] / "final_eval.json").exists()})

        # -- configuration: from the FIRST run's manifest (the continuation's
        #    manifest records learning_starts=0 because the buffer was restored)
        man = None
        for seed in seeds:
            man = read_json(dirs_p[0] / f"seed_{seed}{suffix}" / "run_manifest.json")
            if man:
                break
        if man:
            env = man.get("env", {})
            hp = man.get("hyperparameters", {})
            cfg = env.get("config", {})
            row = {"stage": stage, "description": what, "is_failure_record": failure,
                   "seeds": len(seeds), "git_commit": (man.get("git_commit") or "")[:10],
                   "timesteps_requested": (man.get("cli_args") or {}).get("timesteps")}
            row.update(flat("sac_", {k: hp.get(k) for k in (
                "gamma", "learning_rate", "buffer_size", "learning_starts", "batch_size", "tau",
                "train_freq", "gradient_steps", "ent_coef", "policy_kwargs", "device")}))
            row.update(flat("env_", {k: env.get(k) for k in (
                "obs_version", "obs_dim", "action_dim", "actuation_mode", "dt", "physics_substeps",
                "max_episode_steps", "goal_tolerance", "ee_speed_tolerance", "joint_speed_tolerance",
                "hold_steps_required", "hold_decrement", "intermediate_hold_steps",
                "max_curriculum_tolerance", "discount", "random_waypoints", "free_order",
                "free_order_slots", "waypoints", "activation_time_constants")}))
            row.update(flat("arm_", {k: cfg.get(k) for k in (
                "link_lengths", "masses", "com_offsets", "inertias", "joint_limits_min",
                "joint_limits_max", "torque_limits", "velocity_limits", "damping", "gravity",
                "passive_limit_stiffness", "passive_limit_margin")}))
            rw = env.get("reward_weights") or {}
            row.update({f"w_{k}": v for k, v in rw.items()})
            config.append(row)
            for k, v in rw.items():
                weights.append({"stage": stage, "weight": k, "value": v})
        print(f"  {stage:<12} {len(seeds)} seeds, {n_rounds:>4} curriculum rounds, {n_sac:>5} SAC log rows")

    out.mkdir(parents=True, exist_ok=True)
    files = {
        "curriculum_rounds.csv": rounds, "sac_training_log.csv": sac, "episodes.csv": episodes,
        "final_evaluation.csv": finals, "final_evaluation_episodes.csv": final_eps,
        "run_timing.csv": timing, "stage_config.csv": config, "reward_weights.csv": weights,
    }
    print(f"\nwriting to {out}")
    for name, rows in files.items():
        n = write_csv(out / name, rows)
        print(f"  {name:<32} {n:>7,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
