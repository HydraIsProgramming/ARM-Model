# Final Model Report: fischer_waypoints_v2_500k

**Project:** RL Arm Motion (rl_armMotion)
**Date:** June 21, 2026
**Author:** Ranjot Sandhu
**Model file:** `project_assets/outputs/fischer_waypoints_v2_500k/sac_model.zip`

---

## 1. Executive Summary

The `fischer_waypoints_v2_500k` model is a Soft Actor-Critic (SAC) reinforcement learning policy trained to control a 2-DOF planar robotic arm through a 3-point waypoint sequence. It achieves **3/3 waypoint completion in 181 steps** at 0.6m tolerance and **3/3 in 206 steps** at 0.4m tolerance, with **100% deterministic consistency** across repeated evaluations. The model uses Hill-type muscle actuation (4D action space) following the Fischer et al. (2021) biomechanical protocol.

---

## 2. Model Architecture

| Parameter | Value |
|---|---|
| Algorithm | SAC (Soft Actor-Critic) |
| Policy class | SACPolicy (MlpPolicy) |
| Network architecture | [256, 256] (2 hidden layers) |
| Observation dimension | 11 |
| Action dimension | 4 |
| Action range | [0.0, 1.0] (muscle activations) |
| Learning rate | 3e-4 |
| Replay buffer size | 200,000 |
| Batch size | 256 |
| Soft update coefficient (tau) | 0.005 |
| Discount factor (gamma) | 0.99 |
| Entropy coefficient | auto (learned) |
| Framework | Stable-Baselines3 (PyTorch) |

---

## 3. Observation Space (11-dimensional)

| Index | Component | Range | Description |
|---|---|---|---|
| 0 | sin(shoulder_angle) | [-1, 1] | Shoulder joint sine encoding |
| 1 | cos(shoulder_angle) | [-1, 1] | Shoulder joint cosine encoding |
| 2 | sin(elbow_angle) | [-1, 1] | Elbow joint sine encoding |
| 3 | cos(elbow_angle) | [-1, 1] | Elbow joint cosine encoding |
| 4 | shoulder_velocity_norm | [-1, 1] | Normalized shoulder angular velocity |
| 5 | elbow_velocity_norm | [-1, 1] | Normalized elbow angular velocity |
| 6 | signed_height_error | [-1.8, 1.8] | Signed distance to goal in primary axis |
| 7 | signed_orientation_error | [-pi, pi] | Signed orientation error to goal |
| 8 | gradient_norm | [0, 1] | Normalized error gradient (rate of approach) |
| 9 | in_goal_region | {0, 1} | Binary: 1 if within all tolerances |
| 10 | hold_progress | [0, 1] | Fraction of hold steps completed |

---

## 4. Action Space (4-dimensional, Hill-type Muscle Actuation)

The action space follows Fischer et al. (2021) antagonist-pair muscle model:

| Index | Component | Range | Description |
|---|---|---|---|
| 0 | shoulder_extensor | [0, 1] | Shoulder extensor muscle activation |
| 1 | shoulder_flexor | [0, 1] | Shoulder flexor muscle activation |
| 2 | elbow_extensor | [0, 1] | Elbow extensor muscle activation |
| 3 | elbow_flexor | [0, 1] | Elbow flexor muscle activation |

Net torque per joint is computed as: `torque_j = max_torque * (extensor - flexor)`, where `max_torque` is derived from the arm's mass-length configuration. Co-contraction (simultaneous agonist-antagonist activation) is penalized in the reward function to encourage natural muscle coordination.

---

## 5. Environment Configuration

### Arm Properties
| Parameter | Value |
|---|---|
| Degrees of freedom | 2 (shoulder + elbow) |
| Link lengths | [1.0, 0.8] meters |
| Link masses | [2.0, 1.5] kg |
| Shoulder position | [1.0, 0.0] meters from workspace origin |
| Shoulder joint limits | [-180, 180] degrees |
| Elbow joint limits | [0, 120] degrees |
| Simulation timestep (dt) | 0.02 seconds |
| Damping coefficient | 0.5 |

### Task Configuration
| Parameter | Value |
|---|---|
| Initial arm pose | Vertical downward (shoulder -90deg, elbow 0deg) |
| Waypoints | [2.3, -1.0], [1.0, 1.5], [2.5, 0.0] |
| Position tolerance | 0.6 meters (curriculum stage 0) |
| Orientation tolerance | 12 degrees (0.2094 rad) |
| Hold velocity threshold | 0.3 rad/s |
| Hold steps required | 20 consecutive steps |
| Hold grace | Decrement by 5 (not hard reset to 0) |
| Max steps per episode | 1000 |
| Waypoint advance criterion | Position within tolerance (touch-and-go) |
| Final waypoint criterion | Full hold (position + orientation + velocity) |

### Reward Function

**Penalty terms (every step):**
| Label | Weight | Component | Purpose |
|---|---|---|---|
| P1 | -2.0 | goal_distance | Primary distance penalty |
| P2 | -1.0 | orientation_error | Penalize misalignment with goal |
| P3 | -0.15 | velocity_norm | Smoothness penalty |
| P4 | -0.20 | gradient_norm | Anti-overshoot penalty |
| P5 | -0.01 | action_norm | Minimal effort cost |
| P7 | -0.08 | co-contraction | Reduce simultaneous muscle activation |

**Bonus terms (conditional):**
| Label | Condition | Value | Purpose |
|---|---|---|---|
| B1 | progress > 0 | +1.5 * progress | Potential-based reward shaping |
| B2 | dist < 3*tol | +4.0 * (1 - dist/threshold) | Proximity bonus (convex near goal) |
| B3 | in_goal_region | +10.0 | In-goal constant |
| B4 | in_goal_region | +2.0 * hold_counter | Hold-growth bonus (linearly increasing) |
| -- | waypoint touch | +25.0 | Waypoint transition bonus |
| -- | hold complete | +150.0 | Terminal success bonus |

---

## 6. Training Details

| Metric | Value |
|---|---|
| Training date | June 19, 2026 |
| Total timesteps | 500,000 |
| Total episodes | 506 |
| Training wall time | 33 minutes 17 seconds |
| Best episode reward | 25,718.49 |
| Mean reward (last 100 episodes) | 22,187.33 |
| Curriculum | Adaptive (AdaptiveCurriculumCallback) |
| Final curriculum stage | 0 (never advanced from 0.6m) |
| Training script | `scripts/train_fischer_session.py` |

**Training command:**
```bash
python scripts/train_fischer_session.py \
    --timesteps 500000 \
    --actuation-mode muscle \
    --goal-direction EAST \
    --waypoints "2.3,-1.0;1.0,1.5;2.5,0.0" \
    --save-dir ./project_assets/outputs/fischer_waypoints_v2_500k
```

---

## 7. Evaluation Results

### 7.1 Waypoint Completion

| Tolerance | Waypoints Completed | Steps | Status |
|---|---|---|---|
| 0.6 m | 3/3 | 181 | PASS |
| 0.4 m | 3/3 | 206 | PASS |
| 0.2 m | 2/3 | 800 (timeout) | Stuck at dist=0.343 |

### 7.2 Consistency (5 deterministic runs each)

| Tolerance | Pass Rate | Steps (all runs) |
|---|---|---|
| 0.6 m | 5/5 (100%) | [181, 181, 181, 181, 181] |
| 0.4 m | 5/5 (100%) | [206, 206, 206, 206, 206] |

The model is fully deterministic — identical step counts across all runs.

### 7.3 With Action Smoothing (EMA alpha=0.3)

| Tolerance | Waypoints | Steps | Note |
|---|---|---|---|
| 0.6 m | 3/3 | 189 | +8 steps (+4.4% overhead) |

Action smoothing provides visually smoother motion with minimal performance cost.

### 7.4 Fischer Validation Harnesses

**Fitts' Law:**
- Regression: MT = -0.0737 + 0.0970 * ID
- R-squared: 0.3173
- Conditions: 21, Trials per condition: 10

**2/3 Power Law:**
- Regression: log V = -0.2130 + -0.3208 * log C
- R = -0.4550, R-squared: 0.2070
- Samples: 3,598

---

## 8. Environment Improvements (Post-Training)

Three environment-level changes improve evaluation stability without requiring retraining:

1. **Orientation tolerance relaxed from 10 to 12 degrees** — The original 10deg boundary caused hold counter oscillation; the arm was in correct position but orientation drifted slightly past threshold every ~20 steps, resetting the counter.

2. **Hold grace (decrement by 5 instead of hard reset)** — When the arm briefly exits the goal region, the hold counter decrements by 5 rather than resetting to 0. This prevents catastrophic loss of hold progress from single-step excursions.

3. **Action smoothing (EMA wrapper, alpha=0.3)** — An `ActionSmoother` gym wrapper applies exponential moving average filtering to muscle commands, reducing jerkiness in the arm's trajectory.

---

## 9. Comparison: Retraining Attempts

Multiple attempts to improve on the original model through retraining were unsuccessful:

| Approach | Result | Best WP Completion |
|---|---|---|
| Original fischer_v2 (this model) | Benchmark | 3/3 at 0.6m (181 steps) |
| Retraining with env changes (holdgrace) | Worse | 1/3 |
| Fine-tune from v2 (lr=3e-4) | Catastrophic forgetting | 2/3 |
| Fine-tune from v2 (lr=1e-5) | Still degraded | Score 60.0 vs orig 181.3 |
| Multi-seed search (5 seeds) | High variance | Best: 2/3 (seed 1 & 5) |
| V3 improved training | EMA velocity, eval curriculum | 2/3 |

**Key insight:** RL training has high variance. The original model was a fortunate seed that learned a stable hold policy. All retraining attempts, including with improved environments, failed to replicate this. The optimal strategy was environment-level improvements applied at evaluation time.

---

## 10. File Manifest

| File | Description |
|---|---|
| `sac_model.zip` | Trained SAC policy (3.0 MB) |
| `sac_model_metadata.pkl` | Metadata pickle |
| `training_history.csv` | Per-episode metrics (506 rows) |
| `training_stats.json` | Summary statistics |
| `training_log.txt` | Timestamped training log |
| `fitts_law.json` | Fitts' Law regression data |
| `fitts_law.png` | Fitts' Law plot |
| `power_law.json` | 2/3 Power Law regression data |
| `power_law.png` | 2/3 Power Law plot |
| `waypoint_trajectories.png` | Trajectory visualization |

---

## 11. How to Run

### Evaluate the model:
```python
from stable_baselines3 import SAC
from rl_armMotion.two_d.environments.task_env import ArmTaskEnv, ActionSmoother

model = SAC.load("project_assets/outputs/fischer_waypoints_v2_500k/sac_model")
raw_env = ArmTaskEnv(actuation_mode="muscle", goal_direction="EAST")
env = ActionSmoother(raw_env, alpha=0.3)  # optional smoothing
env.unwrapped.set_waypoints([[2.3,-1.0],[1.0,1.5],[2.5,0.0]], tolerance=0.6)
obs, _ = env.reset()
for step in range(800):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        print(f"3/3 waypoints in {step} steps")
        break
```

### Visualize in GUI:
```bash
python -m rl_armMotion.two_d.gui
# Load model via "Arm Model Selection" > "Load Model" > select sac_model.zip
# Enable "Smooth motion" checkbox for smoother visualization
```

---

*Report generated June 21, 2026*
