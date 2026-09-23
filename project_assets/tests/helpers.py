"""Test helpers: an oracle controller that solves the task by reading env state."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_armMotion.two_d.environments.task_env import ArmTaskEnv  # noqa: E402
from rl_armMotion.two_d.utils import dynamics  # noqa: E402


class OraclePolicy:
    """Computed-torque controller that drives the env's arm to the IK solution
    of the active goal (optionally offset), exposing SB3's predict()/save().

    Cheats by reading the env state: it is a test oracle for the task, the
    evaluator and the curriculum, not a learning agent.
    """

    def __init__(self, env, offset=(0.0, 0.0), omega=8.0, gain_scale=1.0):
        self.env = getattr(env, "unwrapped", env)
        self.offset = np.asarray(offset, dtype=float)
        self.kp = (omega ** 2) * gain_scale
        self.kd = 2.0 * omega * gain_scale
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.num_timesteps = 0

    def predict(self, obs, deterministic=True, state=None, episode_start=None):
        env = self.env
        cfg = env.config
        target = env.goal_position + self.offset
        q_t = env.solve_ik(target)
        if q_t is None:
            q_t = env.q
        v = self.kp * (q_t - env.q) - self.kd * env.qd
        tau = (dynamics.mass_matrix(env.q, cfg) @ v + dynamics.bias_vector(env.q, env.qd, cfg)
               - dynamics.passive_torque(env.q, env.qd, cfg))
        if env.actuation_mode == ArmTaskEnv.ACTUATION_TORQUE:
            a = np.clip(tau / env._torque_limits, -1.0, 1.0)
        else:
            # Muscle mode: split the torque between the pos/neg muscle of each joint.
            a = np.zeros(2 * env.num_dof)
            for j, pair in enumerate(env._pairs):
                iso = pair.max_isometric_torque
                frac = float(np.clip(tau[j] / max(iso, 1e-9), -1.0, 1.0))
                a[2 * j] = max(frac, 0.0)
                a[2 * j + 1] = max(-frac, 0.0)
        return a.astype(np.float32), None

    def save(self, path):
        Path(str(path) + ".zip").write_bytes(b"oracle")




class RouteOraclePolicy(OraclePolicy):
    """Oracle that walks a *chosen* route through a free-order task.

    The free-order env never tells the policy which waypoint to visit next, so a
    test policy has to pick for itself. This one follows the route it was given
    and steps to the next leg as each waypoint is marked visited, which is how a
    trained policy is expected to behave - and lets a test assert that an order
    other than 0,1,2 really is allowed.
    """

    def __init__(self, env, route, **kw):
        super().__init__(env, **kw)
        self.route = list(route)

    def _target_index(self):
        for k in self.route:
            if not self.env.visited[k]:
                return k
        return self.route[-1]

    def predict(self, obs, deterministic=True, state=None, episode_start=None):
        env = self.env
        target = env.waypoints[self._target_index()] + self.offset
        q_t = env.solve_ik(target)
        if q_t is None:
            q_t = env.q
        v = self.kp * (q_t - env.q) - self.kd * env.qd
        tau = (dynamics.mass_matrix(env.q, env.config) @ v
               + dynamics.bias_vector(env.q, env.qd, env.config)
               - dynamics.passive_torque(env.q, env.qd, env.config))
        if env.actuation_mode == ArmTaskEnv.ACTUATION_TORQUE:
            a = np.clip(tau / env._torque_limits, -1.0, 1.0)
        else:
            a = np.zeros(2 * env.num_dof)
            for j, pair in enumerate(env._pairs):
                frac = float(np.clip(tau[j] / max(pair.max_isometric_torque, 1e-9), -1.0, 1.0))
                a[2 * j] = max(frac, 0.0)
                a[2 * j + 1] = max(-frac, 0.0)
        return a.astype(np.float32), None
