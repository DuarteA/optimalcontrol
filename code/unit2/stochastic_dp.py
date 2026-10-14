"""General-purpose stochastic dynamic programming solver.

Extracted from JupyterNotebooks/A_2_StochInventoryControl.ipynb. Solves a
finite-horizon Markov decision problem

    min  E[ sum_{k=0}^{h-1} g_k(x_k, u_k, w_k) + g_h(x_h) ]
    s.t. x_{k+1} = f_k(x_k, u_k, w_k)

where k in {0,...,h-1}, x_k in {1,...,n_k}, u_k in {1,...,m_k(x_k)}, and
w_k in {1,...,p_k(x_k,u_k)} is a random disturbance with
Prob[w_k = l | x_k, u_k] given.

This is the stochastic generalization of the deterministic solver in
unit1/dp.py: instead of one next-state/cost per (state, action), each
(state, action) pair maps to a *list* of possible next states, each with
its own cost and probability, and the algorithm minimizes the expected
cost-to-go.

Data format (1-indexed, matching the original MATLAB-style convention):
    M[k][i][j][l] = f_k(i+1, j+1, l+1)   -- next state (1-indexed)
    C[k][i][j][l] = g_k(i+1, j+1, l+1)   -- stage cost
    P[k][i][j][l] = Prob[w_k = l+1 | x_k = i+1, u_k = j+1]
    Jh[i]         = g_h(i+1)             -- terminal cost

  where k indexes stage (0..h-1), i indexes state, j indexes action, and
  l indexes the disturbance outcome for that (state, action) pair.

Returns:
    J[i, k] -- expected cost-to-go at stage k, state i+1 (0-indexed state)
    u[i, k] -- index j of an optimal action at stage k, state i+1
               (if several actions tie, only the first, per np.argmin)
"""
import numpy as np


def stochdp(M, C, P, Jh, h):
    n = int(len(M[0]))  # number of states
    J = Jh
    J_ = np.zeros([n, h])
    u_ = np.zeros([n, h])

    for k in range(h - 1, -1, -1):
        for i in range(0, n):
            nj = len(M[k][i])
            caux = np.zeros(nj)
            for j in range(0, nj):
                caux[j] = 0
                for l in range(0, len(M[k][i][j])):
                    caux[j] = caux[j] + P[k][i][j][l] * (
                        C[k][i][j][l] + J[int(M[k][i][j][l] - 1)]
                    )
            J_[i, k] = np.min(caux)
            u_[i, k] = np.argmin(caux)
        J = J_[:, k]

    return J_, u_


if __name__ == "__main__":
    # Minimal smoke test: a 1-stage, 2-state problem with no real choice,
    # just to confirm the recursion and expectation are wired correctly.
    # State 1 (index 0): only action available leads to state 1 (cost 0)
    #                     w.p. 0.5 or state 2 (cost 10) w.p. 0.5.
    # State 2 (index 1): only action available leads to state 1 for sure,
    #                     cost 2.
    M = [[[[1]], [[1]]]]
    C = [[[[0]], [[2]]]]
    P = [[[[1.0]], [[1.0]]]]
    # Wait -- to actually test the expectation, use a 2-outcome branch:
    M = [[[[1, 2]], [[1]]]]
    C = [[[[0, 10]], [[2]]]]
    P = [[[[0.5, 0.5]], [[1.0]]]]
    Jh = np.array([0.0, 0.0])
    J, u = stochdp(M, C, P, Jh, 1)
    expected_J0 = 0.5 * 0 + 0.5 * 10
    print(f"J(state 1) = {J[0,0]} (expected {expected_J0})")
    print(f"J(state 2) = {J[1,0]} (expected 2.0)")
    assert abs(J[0, 0] - expected_J0) < 1e-9
    assert abs(J[1, 0] - 2.0) < 1e-9
    print("Smoke test passed. See stoch_inventory_control.py for a full example.")
