"""Policy iteration drivers for Unit 11.

Python translations of the helper functions repeated inside the Unit 11
MATLAB live scripts (`stochinventorycontroPI.mlx`, `statediscPI.mlx`,
`stochinventorycontroSimBasedPI.mlx`, `RobotPlanningSimulationBasedPI.mlx`),
collected here so each notebook shows its problem rather than another copy of
the algorithm.

Costs are MINIMISED, as in Units 1-10. Indices are 0-based throughout, where
the MATLAB originals are 1-based.

An MDP is described exactly as in the live scripts, by three nested lists and
a terminal cost:

    DM[k][i][j] -> list of next states, one per disturbance outcome
    CM[k][i][j] -> list of costs,       one per disturbance outcome
    PM[k][i][j] -> list of probabilities, one per disturbance outcome
    gh[i]       -> terminal cost of state i at stage h

`policy_iteration` alternates `policy_evaluation` (apply the current policy,
sweep backwards, no minimisation) and `policy_improvement` (one greedy sweep,
which is a rollout step with the current policy as base policy) until the
policy stops changing.
"""

import numpy as np


# --------------------------------------------------------------- exact PI
def policy_evaluation(mdp, mu):
    """Cost of policy `mu`: a backward sweep with the action fixed by `mu`."""
    DM, CM, PM, gh = mdp["DM"], mdp["CM"], mdp["PM"], mdp["gh"]
    h = len(DM)
    J = [None]*(h+1)
    J[h] = list(gh)
    for k in range(h-1, -1, -1):
        J[k] = [0.0]*len(DM[k])
        for i in range(len(DM[k])):
            j = mu[k][i]
            J[k][i] = sum(PM[k][i][j][l]*(CM[k][i][j][l] + J[k+1][DM[k][i][j][l]])
                          for l in range(len(DM[k][i][j])))
    return J


def policy_improvement(mdp, J):
    """One greedy sweep against the evaluated cost: a rollout step."""
    DM, CM, PM = mdp["DM"], mdp["CM"], mdp["PM"]
    h = len(DM)
    mu = [None]*h
    for k in range(h):
        mu[k] = [0]*len(DM[k])
        for i in range(len(DM[k])):
            vals = [sum(PM[k][i][j][l]*(CM[k][i][j][l] + J[k+1][DM[k][i][j][l]])
                        for l in range(len(DM[k][i][j])))
                    for j in range(len(DM[k][i]))]
            mu[k][i] = int(np.argmin(vals))
    return mu


def policy_iteration(mdp, verbose=False):
    """Alternate evaluation and improvement until the policy repeats.

    Returns (mu, J, history) with `history` the list of policies visited, so
    the lecture's slide-by-slide picture can be reproduced.
    """
    h = len(mdp["DM"])
    mu = [[0]*len(mdp["DM"][k]) for k in range(h)]      # start: first action everywhere
    history = []
    while True:
        J = policy_evaluation(mdp, mu)
        history.append((mu, J))
        mu_new = policy_improvement(mdp, J)
        if verbose:
            print(f"  iteration {len(history)-1}: J[0][0] = {J[0][0]:.4f}")
        if mu_new == mu:
            return mu, J, history
        mu = mu_new


def dp(M, C, Jh, h):
    """Exact backward DP for a deterministic lookup-table model, for comparison.

    Translation of the `dp` helper in statediscPI.mlx. Ties are broken towards
    the LAST minimiser, as there (`c(end)`).
    """
    n = M.shape[0]
    J = np.array(Jh, dtype=float)
    J_, u_ = np.zeros((n, h)), np.zeros((n, h), dtype=int)
    for k in range(h-1, -1, -1):
        vals = C + J[M]
        J_[:, k] = vals.min(axis=1)
        u_[:, k] = (vals.shape[1] - 1
                    - np.argmin(vals[:, ::-1] != J_[:, k][:, None], axis=1))
        J = J_[:, k]
    return u_, J_


# ------------------------------------------------- simulation-based PI
def sample_dist(prob, rng):
    """Draw an index from a (possibly unnormalised) discrete distribution.

    Translation of the live scripts' `sampledist`.
    """
    prob = np.asarray(prob, dtype=float)
    nonzero = np.flatnonzero(prob > 0)
    cum = np.cumsum(prob[nonzero])
    return int(nonzero[int(np.sum(cum < rng.random()*cum[-1]))])


def sim_based_policy_iteration(num_actions, model_and_cost, *, nsim, eps, alpha,
                               prob_init_state, terminal_states, outer=10,
                               rng=None, terminal_cost=True, greedy_behaviour=False,
                               eps_decay_at=None, broken_exploration=False,
                               max_steps=10_000, verbose=True):
    """Simulation-based policy iteration (generalized policy iteration).

    The inner loop evaluates the current policy from simulated trajectories --
    a SARSA-style update, bootstrapping on ``Q[x'][mu(x')]`` rather than on a
    minimum -- and the outer loop improves the policy greedily.

    The two live scripts differ in three details, exposed here as arguments
    so one driver covers both:

    ``terminal_cost``
        inventory sets ``Q[terminal][0] = g``; the robot script sets it to 0.
    ``greedy_behaviour``
        inventory acts with the current policy ``mu``; the robot script acts
        greedily with respect to the current ``Q``.
    ``eps_decay_at``
        inventory drops ``eps`` after this many simulations; the robot script
        never decays it. Pass None for no decay.

    NOTE on the robot script's exploration. It explores with
    ``sampledist(ones(1,nu))`` -- an UNNORMALISED vector of ones. In
    ``sampledist`` the cumulative sum is then ``[1,2,3]`` and the draw is
    ``rand(1) < 1``, so the test ``sum(cprobdist_ < a)`` is always 0 and the
    exploratory action is always the FIRST one. Exploration there is not
    uniform, it is "always try action 1". The inventory script divides by
    ``nu`` and is correct. This driver always explores uniformly; set
    ``broken_exploration=True`` to reproduce the original behaviour.
    """
    rng = np.random.default_rng() if rng is None else rng
    num_actions = np.asarray(num_actions, dtype=int)
    n = len(num_actions)
    terminal = set(int(t) for t in terminal_states)
    Q = [np.zeros(num_actions[i]) for i in range(n)]
    mu = np.zeros(n, dtype=int)
    history = []

    for ell in range(outer):
        for isim in range(nsim):
            eps_ = eps if (eps_decay_at is None or isim < eps_decay_at) \
                       else eps/(nsim/eps_decay_at)
            x = sample_dist(prob_init_state, rng)
            for _ in range(max_steps):
                if rng.random() < eps_:
                    # see the note above: the robot script's sampledist call is
                    # unnormalised, so its "exploration" always picks action 0
                    u = 0 if broken_exploration else int(rng.integers(num_actions[x]))
                elif greedy_behaviour:
                    u = int(np.argmin(Q[x]))
                else:
                    u = int(mu[x])
                x_next, cost = model_and_cost(x, u)
                if x in terminal:
                    Q[x][0] = cost if terminal_cost else 0.0
                    break
                boot = Q[x_next][0] if x_next in terminal else Q[x_next][mu[x_next]]
                Q[x][u] = (1-alpha)*Q[x][u] + alpha*(cost + boot)
                x = x_next
        J = np.array([Q[i].min() for i in range(n)])
        mu = np.array([int(np.argmin(Q[i])) for i in range(n)])
        history.append((mu.copy(), J.copy()))
        if verbose:
            print(f"  outer iteration {ell+1}/{outer} done")
    return Q, mu, J, history
