# CHAMPION — 3/3 Waypoint Model

Generated: 2026-08-14 22:30

## Status

**Phase 1 goal achieved.** This model completes all three waypoints in order and
holds the final waypoint for the full 20 consecutive steps, with a
**20/20 deterministic success rate**
at the 0.6 m evaluation standard.

## Identity

| Field | Value |
|-------|-------|
| Source run | `Ablation/decrement_5/seed_102` |
| Model file | `sac_model_best.zip` (the periodic-eval checkpoint) |
| Seed | 102 |
| Discovered by | `run_ablation.py`, `decrement_5` condition |

**The final model from the same run scores 0/20.** This champion exists only
because best-model checkpointing retained the peak policy. Without it the run
would have been recorded as another failure.

## Training configuration — reproduce exactly

```bash
python scripts/train_fischer_session.py \
    --timesteps 500000 \
    --actuation-mode muscle \
    --goal-direction EAST \
    --waypoints "0.2,1.2;2.2,0.5;1.0,-1.6" \
    --seed 102 \
    --hold-decrement 5 \
    --save-dir ./project_assets/outputs/CHAMPION_REPLICATE
```

Note `--hold-decrement 5`, **not** the current default of 3. This model was
produced under the original, harsher hold decay.

## Environment settings (do NOT change)

- Algorithm: SAC, MlpPolicy [256, 256]
- Observation dim 11, action dim 4 (muscle activations in [0,1])
- Waypoints: [[0.2, 1.2], [2.2, 0.5], [1.0, -1.6]]
- Arm: 2-DOF, shoulder [1.0, 0.0], links 1.0 / 0.8 m
- Orientation tolerance 12 deg, velocity tolerance 0.3 rad/s
- Hold: 20 consecutive steps, decrement 5 per out-of-region step
- Hold curriculum 10 -> 15 -> 20, deterministic-eval driven
- Tolerance curriculum 0.60 -> 0.20 m floor

## Performance

### Multi-tolerance (20 episodes each, full 20-step hold)

| Tolerance | Successes | Rate | Mean steps |
|-----------|-----------|------|------------|
| 0.6 m | 20/20 | 100% | 490.0 |
| 0.4 m | 0/20 | 0% | - |
| 0.3 m | 0/20 | 0% | - |
| 0.2 m | 0/20 | 0% | - |

### Stricter hold requirements (10 episodes each, 0.6 m)

| Hold requirement | Successes | Rate |
|------------------|-----------|------|
| 20 steps | 10/10 | 100% |
| 30 steps | 10/10 | 100% |
| 50 steps | 10/10 | 100% |

### Training statistics

- **algorithm**: SAC
- **total_timesteps**: 500000
- **total_episodes**: 502
- **mean_reward**: 9569.375007600001
- **best_reward**: 20096.180066
- **success_rate**: 1.0
- **hold_progress**: 0.5333333333333333
- **gradient_norm**: 5.230605535144761e-05
- **training_time_sec**: 7023.699537
- **stopped_early**: False

## How to load

```python
import sys; sys.path.insert(0, 'src')
from stable_baselines3 import SAC
from rl_armMotion.two_d.environments.task_env import ArmTaskEnv

model = SAC.load('project_assets/outputs/CHAMPION/sac_model_best')
env = ArmTaskEnv(actuation_mode='muscle', goal_direction='EAST')
env.set_waypoints([[0.2, 1.2], [2.2, 0.5], [1.0, -1.6]], tolerance=0.6)

obs, _ = env.reset()
for step in range(800):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        print(f'SUCCESS - 3/3 waypoints in {step} steps')
        break
```

To watch it run:

```bash
python scripts/show_waypoints.py \
    project_assets/outputs/CHAMPION/sac_model_best \
    --waypoints "0.2,1.2;2.2,0.5;1.0,-1.6"
```

## Warnings

- **NEVER fine-tune this model.** SAC exploration noise destroys hold behaviour.
- **NEVER overwrite this directory.** It is the only known 3/3 model.
- Do not change reward weights or the 11D observation space.
- `sac_model_best.zip` is the champion. `sac_model.zip` in this directory is
  the final model from the same run and scores 0/20 — keep it only as evidence
  of the checkpointing effect.
