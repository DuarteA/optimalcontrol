"""Policy gradient and actor-critic helpers for Unit 13.

A port of the agents in Duarte's `reinforce_mountaincar.py` and
`actor_critic_mountaincar.py` (github.com/DuarteA/ActorCriticExample), split
out of the scripts so that the two notebooks can share them.

Both agents use the same **Gaussian policy** over a one-dimensional continuous
action: a network outputs a mean and a log standard deviation, and the action
is drawn from that normal distribution.  This is the concrete answer to the
unit's requirement that the policy be *stochastic* -- without randomness the
cost is a discontinuous function of the parameters and there is nothing to
differentiate.

SIGN CONVENTION.  Gymnasium hands out **rewards to be maximised**, while
Units 1-12 minimise a **cost**.  The deck writes the REINFORCE step as a
descent on a cost,

    theta <- theta - gamma * G_k * grad log pi(u_k | x_k, theta),

and the code below minimises the loss  -log pi * R  by gradient descent, which
is the same thing with  g = -R.  Every formula here is therefore the deck's,
read with the sign flipped once.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    import gymnasium as gym
except ImportError:                                     # pragma: no cover
    gym = None

DEVICE = torch.device("cpu")        # these networks are tiny; CPU is faster


@dataclass
class Config:
    env_id: str = "MountainCarContinuous-v0"
    seed: int = 0
    episodes: int = 1000
    max_steps: int = 999
    hidden: int = 128
    gamma: float = 0.99             # discount (alpha in the notes)
    lr_policy: float = 1e-3         # REINFORCE step size
    lr_actor: float = 1e-4          # actor step size (gamma in the notes)
    lr_critic: float = 1e-3         # critic step size (beta in the notes)
    grad_clip: float = 1.0
    normalise_returns: bool = True  # see the note in REINFORCE.update_policy


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(cfg: Config, seed: int | None = None, render_mode: str | None = None):
    if gym is None:
        raise ImportError('gymnasium is required: pip install "gymnasium[classic_control]"')
    env = gym.make(cfg.env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


# --------------------------------------------------------------------------
# networks
# --------------------------------------------------------------------------

class GaussianPolicy(nn.Module):
    """The actor: state -> (mean, log std) of a normal distribution over actions.

    This is the parameterisation the unit calls pi(u | x, theta).  It is
    differentiable in theta, which is the entire reason policy gradient works.
    """

    def __init__(self, state_size: int, action_size: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_mean = nn.Linear(hidden, action_size)
        self.fc_log_std = nn.Linear(hidden, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.clamp(self.fc_log_std(x), min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        """Draw an action and return it with its log-probability."""
        mean, log_std = self(state)
        normal = torch.distributions.Normal(mean, torch.exp(log_std))
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(dim=-1)
        return torch.clamp(action, -1.0, 1.0), log_prob, torch.exp(log_std)


class ValueNetwork(nn.Module):
    """The critic: state -> estimated cost-to-go (here, value)."""

    def __init__(self, state_size: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state)


def _t(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=DEVICE).unsqueeze(0)


# --------------------------------------------------------------------------
# REINFORCE
# --------------------------------------------------------------------------

class REINFORCE:
    """Monte-Carlo policy gradient: one update per episode, using the return."""

    def __init__(self, state_size: int, action_size: int, cfg: Config):
        self.cfg = cfg
        self.policy = GaussianPolicy(state_size, action_size, cfg.hidden).to(DEVICE)
        self.opt = optim.Adam(self.policy.parameters(), lr=cfg.lr_policy)
        self.reset_episode()

    def reset_episode(self):
        self.log_probs, self.rewards = [], []

    def select_action(self, state, store: bool = True):
        action, log_prob, _ = self.policy.sample(_t(state))
        if store:
            self.log_probs.append(log_prob)
        return action.squeeze(0).detach().numpy()

    def store_reward(self, r):
        self.rewards.append(r)

    def returns(self):
        """G_k = sum_{t=k}^{h-1} r_t * gamma^{t-k}, computed backwards."""
        out, G = [], 0.0
        for r in reversed(self.rewards):
            G = r + self.cfg.gamma * G
            out.insert(0, G)
        return torch.tensor(out, dtype=torch.float32, device=DEVICE)

    def update_policy(self):
        G = self.returns()
        raw_std = float(G.std()) if len(G) > 1 else 0.0
        if self.cfg.normalise_returns and len(G) > 1:
            # NOTE: this is a baseline subtraction (and a rescaling) in
            # disguise -- see the notebook.  Subtracting the mean return does
            # not change the gradient in expectation, exactly as the unit's
            # baseline-subtraction slide says; dividing by the standard
            # deviation does change it, by rescaling the step size.
            G = (G - G.mean()) / (G.std() + 1e-8)
        loss = torch.stack([-lp * g for lp, g in zip(self.log_probs, G)]).sum()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self.reset_episode()
        return loss.item(), raw_std


# --------------------------------------------------------------------------
# actor-critic
# --------------------------------------------------------------------------

class ActorCritic:
    """Step-by-step actor-critic: one update per time step, using the TD error.

    The quantity shared by the two updates is

        delta = r + gamma * V(x') - V(x),

    the temporal difference of Unit 11.  The critic fits it (a semi-gradient
    step: the target is detached), and the actor uses it as the signed weight
    on the policy gradient -- which is the unit's baseline subtraction, with
    the baseline learned.
    """

    def __init__(self, state_size: int, action_size: int, cfg: Config):
        self.cfg = cfg
        self.actor = GaussianPolicy(state_size, action_size, cfg.hidden).to(DEVICE)
        self.critic = ValueNetwork(state_size, cfg.hidden).to(DEVICE)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

    def select_action(self, state):
        action, log_prob, _ = self.actor.sample(_t(state))
        return action.squeeze(0).detach().numpy(), log_prob

    def update(self, state, log_prob, reward, next_state, done):
        s, ns = _t(state), _t(next_state)
        value = self.critic(s)
        with torch.no_grad():
            next_value = torch.zeros_like(value) if done else self.critic(ns)
            td_target = reward + self.cfg.gamma * next_value
        delta = (td_target - value).detach()            # the temporal difference

        critic_loss = F.mse_loss(value, td_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.critic_opt.step()

        actor_loss = (-log_prob * delta).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.actor_opt.step()

        return actor_loss.item(), critic_loss.item(), float(delta)


# --------------------------------------------------------------------------
# training loops
# --------------------------------------------------------------------------

def train_reinforce(agent: REINFORCE, env, cfg: Config, episodes: int | None = None,
                    verbose_every: int = 100):
    n = episodes if episodes is not None else cfg.episodes
    scores, losses, grad_scale = [], [], []
    window = deque(maxlen=100)
    for ep in range(1, n + 1):
        state, _ = env.reset()
        total = 0.0
        for _ in range(cfg.max_steps):
            action = agent.select_action(state)
            state, r, term, trunc, _ = env.step(action)
            agent.store_reward(r)
            total += r
            if term or trunc:
                break
        loss, raw_std = agent.update_policy()
        scores.append(total); losses.append(loss); grad_scale.append(raw_std)
        window.append(total)
        if verbose_every and (ep % verbose_every == 0 or ep == 1):
            print(f"REINFORCE ep {ep:4d}  score {total:8.2f}  "
                  f"avg100 {np.mean(window):8.2f}  loss {loss:10.3f}")
    return scores, losses, grad_scale


def train_actor_critic(agent: ActorCritic, env, cfg: Config, episodes: int | None = None,
                       verbose_every: int = 100):
    n = episodes if episodes is not None else cfg.episodes
    scores, a_losses, c_losses, deltas = [], [], [], []
    window = deque(maxlen=100)
    for ep in range(1, n + 1):
        state, _ = env.reset()
        total, al, cl, dl, steps = 0.0, 0.0, 0.0, [], 0
        for _ in range(cfg.max_steps):
            action, log_prob = agent.select_action(state)
            next_state, r, term, trunc, _ = env.step(action)
            done = term or trunc
            a, c, d = agent.update(state, log_prob, r, next_state, done)
            al += a; cl += c; dl.append(d); total += r; steps += 1
            state = next_state
            if done:
                break
        scores.append(total)
        a_losses.append(al / steps); c_losses.append(cl / steps)
        deltas.append(float(np.std(dl)))
        window.append(total)
        if verbose_every and (ep % verbose_every == 0 or ep == 1):
            print(f"actor-critic ep {ep:4d}  score {total:8.2f}  "
                  f"avg100 {np.mean(window):8.2f}  critic loss {cl/steps:10.4f}")
    return scores, a_losses, c_losses, deltas


def evaluate(agent, env, episodes: int = 10):
    """Run the current (still stochastic) policy and report the scores."""
    out = []
    for _ in range(episodes):
        state, _ = env.reset()
        total = 0.0
        while True:
            if isinstance(agent, REINFORCE):
                action = agent.select_action(state, store=False)
            else:
                action, _ = agent.select_action(state)
            state, r, term, trunc, _ = env.step(action)
            total += r
            if term or trunc:
                break
        out.append(total)
    return float(np.mean(out)), float(np.std(out))


def smooth(x, w: int = 50):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")
