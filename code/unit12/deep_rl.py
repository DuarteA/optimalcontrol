"""Deep reinforcement learning helpers for Unit 12.

A port of the agent code in Duarte's `lunardeepQSarsaSemiGradient.py`, split
out of the script so that the two lunar-lander notebooks can share it instead
of each carrying a copy.

Convention warning.  Gymnasium hands out **rewards to be maximised**, while
Units 1-11 of this course minimise a **cost**.  The two differ by a sign, and
this module follows Gymnasium: the greedy action is an ``argmax`` here where
every formula in the notes is an ``argmin``.

Both agents approximate the Q-factors with the same small network; they differ
in one line of the update:

    DQN    target = r + gamma * max_v Q_target(x', v)     off-policy
    SARSA  target = r + gamma * Q(x', a')                 on-policy

where ``a'`` is the action the epsilon-greedy policy actually takes next.  That
is the Unit 11 distinction, carried over to the function-approximation case.
"""

from __future__ import annotations

import collections
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:                                    # gymnasium is optional at import time
    import gymnasium as gym
except ImportError:                     # pragma: no cover
    gym = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------
# configuration -- the values are those of the original script
# --------------------------------------------------------------------------

@dataclass
class Config:
    env_id: str = "LunarLander-v3"
    seed: int = 123
    num_episodes: int = 800
    eval_every: int = 50
    eval_episodes: int = 8

    hidden: int = 128

    # DQN
    dqn_batch: int = 64
    dqn_lr: float = 1e-3
    dqn_gamma: float = 0.99
    dqn_replay_size: int = 50_000
    dqn_start_train: int = 1_000
    dqn_target_update: int = 1_000      # in steps
    dqn_eps_start: float = 1.0
    dqn_eps_end: float = 0.02
    dqn_eps_decay: float = 0.995        # per episode

    # SARSA
    sarsa_lr: float = 1e-3
    sarsa_gamma: float = 0.99
    sarsa_eps_start: float = 1.0
    sarsa_eps_end: float = 0.02
    sarsa_eps_decay: float = 0.995

    grad_clip: float = 10.0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_id: str = "LunarLander-v3", seed: int | None = None,
             render_mode: str | None = None, quiet: bool = False):
    """Create the environment, falling back to CartPole-v1 if Box2D is absent.

    ``LunarLander`` needs Box2D (``pip install "gymnasium[box2d]"``), which is a
    compiled dependency and not always available.  CartPole-v1 ships with the
    base package, needs nothing extra, and exercises exactly the same code --
    the script's own diagnostics suggest it as the fallback.  Only the numbers
    change, not the algorithms.
    """
    if gym is None:
        raise ImportError("gymnasium is required: pip install gymnasium")
    try:
        env = gym.make(env_id, render_mode=render_mode)
    except Exception as exc:                                   # noqa: BLE001
        if env_id == "CartPole-v1":
            raise
        if not quiet:
            print(f"[deep_rl] {env_id} unavailable ({type(exc).__name__}); "
                  f"falling back to CartPole-v1.  Install Box2D with "
                  f'`pip install "gymnasium[box2d]"` for the real example.')
        env = gym.make("CartPole-v1", render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


# --------------------------------------------------------------------------
# the shared Q-network
# --------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Three linear layers with ReLU between them -- the deck's 'three-layer NN'."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Experience replay: the first of deep Q-learning's two special features."""

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, done = map(np.array, zip(*batch))
        return s, a, r, ns, done

    def __len__(self):
        return len(self.buffer)


# --------------------------------------------------------------------------
# agents
# --------------------------------------------------------------------------

class DQNAgent:
    """Deep Q-learning: replay buffer + a separate, slowly updated target network."""

    def __init__(self, obs_dim: int, n_actions: int, cfg: Config):
        self.cfg = cfg
        self.n_actions = n_actions
        self.q = QNetwork(obs_dim, n_actions, cfg.hidden).to(DEVICE)
        self.target_q = QNetwork(obs_dim, n_actions, cfg.hidden).to(DEVICE)
        self.target_q.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.dqn_lr)
        self.replay = ReplayBuffer(cfg.dqn_replay_size)
        self.total_steps = 0
        self.eps = cfg.dqn_eps_start

    def select_action(self, state, eval_mode: bool = False) -> int:
        if (not eval_mode) and random.random() < self.eps:
            return random.randrange(self.n_actions)
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            return int(self.q(s).argmax().item())

    def store(self, s, a, r, ns, done):
        self.replay.push(s, a, r, ns, done)
        self.total_steps += 1

    def train_step(self):
        cfg = self.cfg
        if len(self.replay) < cfg.dqn_batch or self.total_steps < cfg.dqn_start_train:
            return
        s, a, r, ns, done = self.replay.sample(cfg.dqn_batch)
        s = torch.tensor(s, dtype=torch.float32, device=DEVICE)
        a = torch.tensor(a, dtype=torch.int64, device=DEVICE).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32, device=DEVICE)
        done = torch.tensor(done, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        q_vals = self.q(s).gather(1, a)
        with torch.no_grad():
            # the maximum is taken over the TARGET network -- off-policy, and
            # the reason the target does not chase the estimate
            next_q = self.target_q(ns).max(1)[0].unsqueeze(1)
            target = r + (1.0 - done) * cfg.dqn_gamma * next_q
        loss = nn.functional.mse_loss(q_vals, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), cfg.grad_clip)
        self.opt.step()

        if self.total_steps % cfg.dqn_target_update == 0:
            self.target_q.load_state_dict(self.q.state_dict())

    def decay_epsilon(self):
        self.eps = max(self.cfg.dqn_eps_end, self.eps * self.cfg.dqn_eps_decay)


class SARSAAgent:
    """SARSA with function approximation: one network, no replay, no target net.

    The update is a **semi-gradient** step: the target contains the parameters
    too, but is computed under ``torch.no_grad()`` and therefore not
    differentiated.  That is exactly the distinction the unit makes between a
    stochastic gradient and a semi-gradient.
    """

    def __init__(self, obs_dim: int, n_actions: int, cfg: Config):
        self.cfg = cfg
        self.n_actions = n_actions
        self.q = QNetwork(obs_dim, n_actions, cfg.hidden).to(DEVICE)
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.sarsa_lr)
        self.eps = cfg.sarsa_eps_start

    def select_action(self, state, eval_mode: bool = False) -> int:
        if (not eval_mode) and random.random() < self.eps:
            return random.randrange(self.n_actions)
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            return int(self.q(s).argmax().item())

    def update(self, s, a, r, ns, na, done):
        s_t = torch.from_numpy(np.asarray(s, dtype=np.float32)).unsqueeze(0).to(DEVICE)
        ns_t = torch.from_numpy(np.asarray(ns, dtype=np.float32)).unsqueeze(0).to(DEVICE)
        q_sa = self.q(s_t)[0, a]
        with torch.no_grad():
            q_next = 0.0 if done else float(self.q(ns_t)[0, na])
            target = torch.tensor(r + self.cfg.sarsa_gamma * q_next,
                                  dtype=torch.float32, device=DEVICE)
        loss = nn.functional.mse_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()

    def decay_epsilon(self):
        self.eps = max(self.cfg.sarsa_eps_end, self.eps * self.cfg.sarsa_eps_decay)


# --------------------------------------------------------------------------
# training / evaluation loops
# --------------------------------------------------------------------------

def evaluate_agent(agent, env, episodes: int = 5):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(agent.select_action(obs, eval_mode=True))
            done = term or trunc
            total += r
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


def train_dqn(agent, env, cfg: Config, num_episodes: int | None = None,
              eval_env=None, verbose: bool = True):
    """One episode loop.  Every step: act, store, one gradient step."""
    n = num_episodes if num_episodes is not None else cfg.num_episodes
    returns, evals = [], []
    for ep in range(1, n + 1):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            a = agent.select_action(obs)
            next_obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.store(obs, a, r, next_obs, float(done))
            agent.train_step()
            obs = next_obs
            total += r
        agent.decay_epsilon()
        returns.append(total)
        if eval_env is not None and (ep % cfg.eval_every == 0 or ep == 1):
            mean, std = evaluate_agent(agent, eval_env, cfg.eval_episodes)
            evals.append((ep, mean, std))
            if verbose:
                print(f"DQN   ep {ep:4d}  return {total:8.1f}  "
                      f"eval_mean {mean:7.1f}  eps {agent.eps:.3f}")
    return returns, evals


def train_sarsa(agent, env, cfg: Config, num_episodes: int | None = None,
                eval_env=None, verbose: bool = True):
    """The SARSA loop: the next action is chosen before the update, not after."""
    n = num_episodes if num_episodes is not None else cfg.num_episodes
    returns, evals = [], []
    for ep in range(1, n + 1):
        obs, _ = env.reset()
        a = agent.select_action(obs)
        done, total = False, 0.0
        while not done:
            next_obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            next_a = agent.select_action(next_obs)      # the second A of SARSA
            agent.update(obs, a, r, next_obs, next_a, done)
            obs, a = next_obs, next_a
            total += r
        agent.decay_epsilon()
        returns.append(total)
        if eval_env is not None and (ep % cfg.eval_every == 0 or ep == 1):
            mean, std = evaluate_agent(agent, eval_env, cfg.eval_episodes)
            evals.append((ep, mean, std))
            if verbose:
                print(f"SARSA ep {ep:4d}  return {total:8.1f}  "
                      f"eval_mean {mean:7.1f}  eps {agent.eps:.3f}")
    return returns, evals


def smooth(x, w: int = 20):
    """Moving average, for plotting on top of the raw per-episode returns."""
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")
