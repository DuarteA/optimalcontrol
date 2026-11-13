"""Q-learning drivers for Unit 10.

Python translations of the ``QLearning`` and ``sampledist`` helper functions
that are repeated inside each of the Unit 10 MATLAB live scripts
(``RobotPlanningQLearning.mlx``, ``stochinventorycontroQlearning.mlx``,
``statediscQlearning.mlx``, ``stoppingtimeQlearning.mlx``).  They are
collected here so the notebooks show the *problem*, not four copies of the
same driver.

Two drivers are provided, matching the two variants in the live scripts:

``q_learning_ssp``
    Stationary Q-factors ``Q[i][u]`` for a stochastic shortest path problem.
    Each simulation runs until a terminal state is reached.

``q_learning_finite_horizon``
    Stage-indexed Q-factors ``Q[k][i][u]`` for a finite horizon problem.
    Each simulation runs for exactly ``horizon`` stages.

Both follow Unit 10's convention that the quantity is a COST and is
MINIMISED, so the greedy action is ``argmin_u Q(i,u)``.

Indexing: MATLAB's 1-based state/action indices become 0-based here.  A
``model_and_cost(x, u)`` callable therefore receives 0-based ``x`` and ``u``
and must return ``(x_next, cost)`` with ``x_next`` also 0-based.

TWO QUIRKS OF THE ORIGINAL SCRIPTS ARE PRESERVED DELIBERATELY, because
changing them would change the numbers the lecture shows:

1. The exploration rate decays as ``eps / (nsim / 1000)`` once
   ``isim >= 1000`` -- note ``nsim``, the *total* number of simulations, not
   ``isim``, the current one.  So it is not a decaying schedule at all: it
   drops once, to a constant ``eps * 1000 / nsim``, which for the default
   ``nsim = 300000`` is ``eps / 300``.  Pass ``decay_with_isim=True`` to get
   the schedule that was presumably meant, ``eps / (isim / 1000)``.
2. The step size ``alpha`` is held constant.  The convergence theorem of
   Unit 10 asks for ``sum(gamma_t) = inf`` and ``sum(gamma_t^2) < inf``,
   which a constant step size does not satisfy; in practice a constant step
   is what is used, and the estimates then fluctuate around the optimum
   rather than converging to it.
"""

import numpy as np


def _blend(old, target, alpha):
    """``(1-alpha)*old + alpha*target``, with the alpha == 1 case done exactly.

    Needed because the state-discretization script uses ``alpha = 1`` (the
    problem there is deterministic, so a full replacement is correct) while
    the finite-horizon driver can store ``inf`` for an action that fails to
    reach the terminal state.  Evaluating ``(1 - 1) * inf`` gives ``nan`` in
    both MATLAB and numpy -- but MATLAB's ``min`` silently skips ``nan``
    whereas numpy's propagates it, so the MATLAB script limps on and the
    direct translation does not.  Writing the alpha == 1 case as a plain
    assignment removes the ``nan`` at the source.
    """
    if alpha == 1.0:
        return target
    return (1 - alpha) * old + alpha * target


def sample_dist(prob, rng):
    """Draw an index from a (possibly unnormalised) discrete distribution.

    Translation of the live scripts' ``sampledist``: zero-probability entries
    are removed first, so an index is never returned for an impossible
    outcome.  Returns a 0-based index.
    """
    prob = np.asarray(prob, dtype=float)
    nonzero = np.flatnonzero(prob > 0)
    cum = np.cumsum(prob[nonzero])
    return int(nonzero[int(np.sum(cum < rng.random() * cum[-1]))])


def _explore_rate(eps, isim, nsim, decay_with_isim):
    if isim < 1000:
        return eps
    return eps / (isim / 1000) if decay_with_isim else eps / (nsim / 1000)


def q_learning_ssp(num_actions, model_and_cost, *, nsim, eps, alpha,
                   prob_init_state, terminal_states, rng=None,
                   decay_with_isim=False, max_steps=10_000):
    """Q-learning for a stochastic shortest path problem.

    Parameters
    ----------
    num_actions : array of int, length n
        ``num_actions[i]`` is how many actions are available in state ``i``.
    model_and_cost : callable ``(x, u) -> (x_next, cost)``
        The simulator.  This is the only access to the model.
    nsim : int
        Number of simulated trajectories.
    eps : float
        Initial probability of taking an exploratory (uniformly random)
        action rather than the greedy one.
    alpha : float
        Step size of the Q-factor update.
    prob_init_state : array, length n
        Distribution over initial states.
    terminal_states : iterable of int
        States at which a trajectory stops.
    max_steps : int
        Safety cap on the length of one trajectory.

    Returns
    -------
    Q : list of 1-D arrays
        ``Q[i][u]`` the learned Q-factors.
    mu : array of int, length n
        Greedy policy, ``argmin_u Q[i][u]``.
    J : array of float, length n
        Value function, ``min_u Q[i][u]``.
    """
    rng = np.random.default_rng() if rng is None else rng
    num_actions = np.asarray(num_actions, dtype=int)
    n = len(num_actions)
    terminal = set(int(t) for t in terminal_states)
    Q = [np.zeros(num_actions[i]) for i in range(n)]

    for isim in range(nsim):
        eps_ = _explore_rate(eps, isim, nsim, decay_with_isim)
        x = sample_dist(prob_init_state, rng)
        for _ in range(max_steps):
            if rng.random() < eps_:
                u = int(rng.integers(num_actions[x]))
            else:
                u = int(np.nanargmin(Q[x]))
            x_next, cost = model_and_cost(x, u)
            # NOTE: the original tests the state it has just acted FROM, and
            # on a terminal state overwrites Q[x][0] with the cost incurred
            # there.  Terminal states have a single action, so this is
            # well defined.  Kept as-is.
            if x in terminal:
                Q[x][0] = cost
                break
            Q[x][u] = _blend(Q[x][u], cost + np.nanmin(Q[x_next]), alpha)
            x = x_next

    J = np.array([np.nanmin(Q[i]) for i in range(n)])
    mu = np.array([int(np.nanargmin(Q[i])) for i in range(n)])
    return Q, mu, J


def q_learning_finite_horizon(num_actions, model_and_cost, *, nsim, eps, alpha,
                              prob_init_state, horizon, terminal_index=None,
                              rng=None, decay_with_isim=False):
    """Q-learning for a finite horizon problem, with stage-indexed Q-factors.

    Each simulation runs for exactly ``horizon`` stages.  At the last stage
    the target is the terminal cost rather than a lookahead, as in Unit 10's
    finite-horizon Q-learning algorithm.  If ``terminal_index`` is given, a
    trajectory that does not end there is assigned an infinite Q-factor at
    the final stage -- this is how the state-discretization script enforces a
    hard terminal constraint.

    Returns ``Q`` (list over stages of lists over states), ``mu`` and ``J``,
    both of shape ``(n, horizon)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    num_actions = np.asarray(num_actions, dtype=int)
    n = len(num_actions)
    Q = [[np.zeros(num_actions[i]) for i in range(n)] for _ in range(horizon)]

    for isim in range(nsim):
        eps_ = _explore_rate(eps, isim, nsim, decay_with_isim)
        x = sample_dist(prob_init_state, rng)
        for k in range(horizon):
            if rng.random() < eps_:
                u = int(rng.integers(num_actions[x]))
            else:
                u = int(np.nanargmin(Q[k][x]))
            x_next, cost = model_and_cost(x, u)
            if k == horizon - 1:
                if terminal_index is None or x_next == terminal_index:
                    Q[k][x][u] = cost
                else:
                    Q[k][x][u] = np.inf
            else:
                Q[k][x][u] = _blend(Q[k][x][u],
                                    cost + np.nanmin(Q[k + 1][x_next]), alpha)
            x = x_next

    J = np.array([[np.nanmin(Q[k][i]) for k in range(horizon)] for i in range(n)])
    mu = np.array([[int(np.nanargmin(Q[k][i])) for k in range(horizon)] for i in range(n)])
    return Q, mu, J
