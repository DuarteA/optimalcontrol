"""Stochastic inventory control example, built on stochastic_dp.py.

Extracted from JupyterNotebooks/A_2_StochInventoryControl.ipynb. This is
the same inventory control model as unit1/inventory_control.py --

    x_{k+1} = max(x_k + u_k - d_k, 0),   x_k in {0,...,N}, u_k in {0,...,N-x_k}

minimizing E[ sum_k (c1(x_k) + c2(u_k) - p*min(x_k+u_k, d_k)) + g_h(x_h) ]
with c1(i) = c1_ * i, c2(j) = citem*j + ctr (j>0, else 0) -- but now with
an *uncertain* demand at stage 2: d_2 = 1 with probability 0.5, or d_2 = 0
with probability 0.5 (all other stages have deterministic demand). This
reproduces the example in Unit 2 notes / lecture02-stochastic-dp.qmd
Appendix A ("Stochastic dynamic programming for the inventory control
example").

The construction below builds the (M, C, P) data structures stage by
stage: for stages with a deterministic demand, each (state, action) maps
to a single next state with probability 1; for stage 2, each (state,
action) maps to *two* possible next states (one per demand outcome), each
with probability 0.5.
"""
import numpy as np
from stochastic_dp import stochdp


def stoch_inventory_control(p, citem, ctr, c1_, d, h, N, stoch_stage, d_alt, prob_alt=0.5):
    """Build and solve the stochastic inventory control DP.

    Parameters
    ----------
    p, citem, ctr, c1_ : float
        Selling price, cost per item ordered, transportation cost, and
        storage cost coefficient (c1(i) = c1_ * i).
    d : array of length h
        Nominal (nominal/"otherwise" outcome) demand at each stage.
    h, N : int
        Number of stages and storage capacity.
    stoch_stage : int
        Index (0-based) of the one stage with uncertain demand.
    d_alt : float
        The alternative demand value at `stoch_stage` (occurs with
        probability `1-prob_alt`; `d[stoch_stage]` occurs with `prob_alt`).
    prob_alt : float
        Probability of the *nominal* value at `stoch_stage` (kept at 0.5
        by default, matching the notes' example).

    Returns
    -------
    J, u : the cost-to-go and policy arrays returned by stochdp (states
        are 0-indexed inventory levels 0..N, i.e. row 0 = x=0, ..., row N = x=N).
    """
    n_states = N + 1
    DM = [[[] for _ in range(n_states)] for _ in range(h)]
    CM = [[[] for _ in range(n_states)] for _ in range(h)]
    PM = [[[] for _ in range(n_states)] for _ in range(h)]

    def add_branch(k, i, j, next_state_1indexed, prob, demand):
        if len(DM[k][i]) == 0 or len(DM[k][i]) == j + 1:
            DM[k][i].append([])
            PM[k][i].append([])
            CM[k][i].append([])
        DM[k][i][j + 1] = np.append(DM[k][i][j + 1], next_state_1indexed)
        PM[k][i][j + 1] = np.append(PM[k][i][j + 1], prob)
        order = j + 1  # number of items ordered (u_k), since j starts at -1
        c2 = citem * order + (ctr if order > 0 else 0)
        c1 = i * c1_
        revenue = np.min([demand, order + i]) * p
        CM[k][i][j + 1] = np.append(CM[k][i][j + 1], -1 * (revenue - (c1 + c2)))

    for k in range(h):
        for i in range(n_states):
            # Bound the order so that i + order - (smallest possible demand
            # this stage) never exceeds N, i.e. the next state always stays
            # within the valid range 0..N regardless of which demand
            # realization occurs.
            min_demand_this_stage = min(d[k], d_alt) if k == stoch_stage else d[k]
            max_order = N - i + min_demand_this_stage
            for j in range(-1, max_order):  # j=-1 sentinel, order = j+1, up to max_order
                if k == stoch_stage:
                    next_nom = int(np.max([i + (j + 1) - d[k], 0]))
                    add_branch(k, i, j, next_nom + 1, prob_alt, d[k])
                    next_alt = int(np.max([i + (j + 1) - d_alt, 0]))
                    add_branch(k, i, j, next_alt + 1, 1 - prob_alt, d_alt)
                else:
                    next_state = int(np.max([i + (j + 1) - d[k], 0]))
                    add_branch(k, i, j, next_state + 1, 1.0, d[k])

    Jh = -np.array([m * (citem - c1_) for m in range(n_states)])
    J, u = stochdp(DM, CM, PM, Jh, h)
    return J, u


if __name__ == "__main__":
    # Same numbers as the notes' worked example (Appendix A): h=4, N=4,
    # nominal demand [2,2,1,1], with the stage-2 demand uncertain between
    # its nominal value (1) and 0, each with probability 0.5.
    p = 10
    citem = 5
    ctr = 0.5
    c1_ = 0.2
    d = np.array([2, 2, 1, 1])
    h = 4
    N = 4

    J, u = stoch_inventory_control(
        p, citem, ctr, c1_, d, h, N, stoch_stage=2, d_alt=0, prob_alt=0.5
    )

    print("Expected costs-to-go J[state, stage] (state = inventory level 0..4):")
    print(J)
    print("\nOptimal order quantities u[state, stage]:")
    print(u.astype(int))
    print(
        f"\nE.g. starting empty (x0=0): expected total cost = {J[0,0]:.2f}, "
        f"optimal order u0 = {int(u[0,0])}"
    )
