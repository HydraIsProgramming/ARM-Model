# Ablation Study — Terminal-Hold Mechanisms

Generated: 2026-08-14 22:27

- Seeds per condition: 3 (matched across conditions)
- Timesteps per run: 500,000
- Waypoints: [[0.2, 1.2], [2.2, 0.5], [1.0, -1.6]]
- Evaluation: 5 deterministic episodes at 0.6 m tolerance, full 20-step hold

## Results (best checkpoint, mean over seeds)

| Condition | Mean waypoints | Max hold progress | Successes | Mean train reward |
|-----------|----------------|-------------------|-----------|-------------------|
| full | 2.00 | 0.02 | 0 | 1,562 |
| no_hold_curr | 1.00 | 0.28 | 0 | 8,756 |
| decrement_5 | 1.67 | 0.33 | 5 | 7,667 |
| stochastic | 1.33 | 0.38 | 0 | 3,949 |

## Best-checkpoint vs final-model (checkpointing benefit)

| Condition | Final hold | Best hold | Delta |
|-----------|-----------|-----------|-------|
| full | 0.00 | 0.02 | +0.02 |
| no_hold_curr | 0.28 | 0.00 | -0.28 |
| decrement_5 | 0.23 | 0.33 | +0.10 |
| stochastic | - | - | n/a (no checkpoint) |

## Per-run detail

```
{
  "full": [
    {
      "seed": "seed_101",
      "mean_reward": 4547.4467934799995,
      "train_hold": 0.0,
      "final": {
        "mean_wp": 0.0,
        "max_wp": 0,
        "mean_hold": 0.0,
        "max_hold": 0.0,
        "successes": 0,
        "n": 5
      },
      "best": {
        "mean_wp": 2.0,
        "max_wp": 2,
        "mean_hold": 0.05,
        "max_hold": 0.05,
        "successes": 0,
        "n": 5
      }
    },
    {
      "seed": "seed_102",
      "mean_reward": 268.0401164400001,
      "train_hold": 0.0,
      "final": {
        "mean_wp": 0.0,
        "max_wp": 0,
        "mean_hold": 0.0,
        "max_hold": 0.0,
        "successes": 0,
        "n": 5
      },
      "best": {
        "mean_wp": 2.0,
        "max_wp": 2,
        "mean_hold": 0.0,
        "max_hold": 0.0,
        "successes": 0,
        "n": 5
      }
    },
    {
      "seed": "seed_103",
      "mean_reward": -129.54259797000003,
      "train_hold": 0.0,
      "final": {
        "mean_wp": 0.0,
        "max_wp": 0,
        "mean_hold": 0.0,
        "max_hold": 0.0,
        "successes": 0,
        "n": 5
      },
      "best": {
        "mean_wp": 2.0,
        "max_wp": 2,
        "mean_hold": 0.0,
        "max_hold": 0.0,
        "successes": 0,
        "n": 5
      }
    }
  ],
  "no_hold_curr": [
    {
      "seed": "seed_101",
      "mean_reward": 25529.3597811,
      "train_hold": 0.9,
      "final": {
        "mean_wp": 2.0,
        "max_wp": 2,
        "mean_hold": 0.85,
        "max_hold": 0.85,
        "successes": 0,
        "n": 5
      },
      "best": {
        "mean_wp": 2.0,
        "max_wp": 2,
        "mean_hold": 0.0,
        "max_hold": 0.0,
        "successes": 0,
        "n": 5
      }
    },
    {
      "seed": "seed_102",
      "mean_reward": 677.0446071299998,
      "train_hold": 0.0,
      "final": {
        "mean_wp": 0.0,
        "max_wp": 0,
        "mean_hold": 0.0,
        "max_hold": 0.0,
        "successes": 0,
        "n": 5
      },
      "best": {
        "mean_wp": 1.0,
        "max_wp": 1,
        "mean_hold": 0.0,
        "max_hold": 0.0,
        "successes": 0,
        "n": 5
      }
    },
    {
      "seed": "seed_103",
      "mean_reward": 60.665396730000005,
      "train_hold": 0.0,
      "final": {
        "mean_wp": 0.0,
        "max_wp": 0,
        "mean_hold": 0.0,
        "max_hold": 0.0,
        "successes": 0,
        "n": 5
      },
      "best": {
        "mean_wp": 0.0,
        "max_wp": 0,
        "mean_hold": 0.0,
        "max_hold": 0.0,
        "successes": 0,
        "n": 5
      }
    }
  ],
  "decrement_5": [
    {
      "seed": "seed_101",
      "mean_reward": 13133.442521250003,
      "train_hold": 0.2,
      "final": {
        "mean_wp": 2.0,
        "max_wp": 2,
        "mean_hold": 0.1,
        "max_hold": 0.1,
        "successes": 0,
        "n": 5
      },
      "best": {
        "mean_wp": 2.0,
        "max_wp": 2,
        "mean_hold": 0.0,
        "max_hold": 0.0,
        "successes": 0,
        "n": 5
      }
    },
    {
      "seed": "seed_102",
      "mean_reward": 9569.375007600001,
      "train_hold": 0.5333333333333333,
      "final": {
        "mean_wp": 2.0,
        "max_wp": 2,
        "mean_hold": 0.6,
        "max_hold": 0.6,
        "successes": 0,
        "n": 5
      },
      "best": {
        "mean_wp": 2.0,
        "max_wp": 2,
        "mean_hold": 1.0,
        "max_hold": 1.0,
        "successes": 5,
        "n": 5
      }
    },
    {
      "seed": "seed_103",
      "mean_reward": 299.39270385000003,
      "train_hold": 0.0,
      "final": {
        "mean_wp": 0.0,
        "max_wp": 0,
        "mean_hold": 0.0,
        "max_hold": 0.0,
        "successes": 0,
        "n": 5
      },
      "best": {
        "mean_wp": 1.0,
        "max_wp": 1,
        "mean_hold": 0.0,
        "max_hold": 0.0,
        "successes": 0,
        "n": 5
      }
    }
  ],
  "stochastic": [
    {
      "seed": "seed_101",
      "mean_reward": 10948.989053409998,
      "train_hold": 0.6,
      "final": {
        "mean_wp": 2.0,
        "max_wp": 2,
        "mean_hold": 0.7,
        "max_hold": 0.7,
        "successes": 0,
        "n": 5
      },
      "best": {}
    },
    {
      "seed": "seed_102",
      "mean_reward": 729.12481552,
      "train_hold": 0.0,
      "final": {
        "mean_wp": 2.0,
        "max_wp": 2,
        "mean_hold": 0.45,
        "max_hold": 0.45,
        "successes": 0,
        "n": 5
      },
      "best": {}
    },
    {
      "seed": "seed_103",
      "mean_reward": 169.98517166000002,
      "train_hold": 0.0,
      "final": {
        "mean_wp": 0.0,
        "max_wp": 0,
        "mean_hold": 0.0,
        "max_hold": 0.0,
        "successes": 0,
        "n": 5
      },
      "best": {}
    }
  ]
}
```