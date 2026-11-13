"""Dynamic programming algorithm for finite-horizon problems with discrete
state and action spaces (Unit 1: Modeling and the dynamic programming
algorithm).

Extracted from JupyterNotebooks/A_1_dp.ipynb.

Problem data is represented using the convention from the Unit 1 notes
("A more general formalism"):

    M[k][i][j] = f_k(i, j)   -- state reached from state i, stage k, action j
    C[k][i][j] = g_k(i, j)   -- cost of that action
    T[i]       = g_h(i)      -- terminal cost of state i at the final stage

States and actions are indexed starting at 1 (matching the notes'
convention x_k in {1,...,n_k}, u_k in {1,...,m_k(x_k)}).
"""
import numpy as np


def dptv(M, C, T):
    """Run the dynamic programming algorithm on a problem given in (M, C, T)
    form and return the optimal policy and costs-to-go.

    Returns
    -------
    u : u[k][i] is a list of optimal actions (possibly more than one, in
        case of ties) at stage k, state i (0-indexed in the returned lists,
        but action *values* are 1-indexed, matching M/C).
    J : J[k][i] is the cost-to-go at stage k, state i (terminal cost not
        included).
    """
    h = len(M)
    J_ = np.empty((h + 1, 0)).tolist()
    J = np.empty((h, 0)).tolist()
    u = np.empty((h, 0)).tolist()
    for l in range(len(T)):
        J_[h].append(T[l])

    for k in range(h)[::-1]:
        ni = len(M[k])  # number of states at stage k
        for i in range(ni):
            nj = len(M[k][i])  # number of actions at stage k, state i
            caux = np.zeros(nj)
            for j in range(nj):
                caux[j] = C[k][i][j] + J_[k + 1][M[k][i][j] - 1]
            # sort to identify the optimal action(s)
            b = np.argsort(caux)
            a = caux[b]
            J_[k].append(a[0])
            J[k].append(J_[k][i])
            u[k].append([b[0] + 1])
            # actions whose cost-to-go is within a small threshold of the
            # optimum are also considered optimal (ties)
            for ell in range(1, len(a)):
                if np.isinf(a[ell]) or np.isinf(a[0]):
                    break
                if abs(a[ell] - a[0]) < 1e-8:
                    u[k][i].append(b[ell] + 1)
                else:
                    break

    return u, J


if __name__ == "__main__":
    # Worked example from the Unit 1 notes / A_1_dp.ipynb: a 4-stage
    # transition diagram (states/actions 1-indexed).
    stages = 4
    M = np.empty((stages, 0)).tolist()
    C = np.empty((stages, 0)).tolist()

    # stage 0
    C[0] = [[1, 4], [1, 2]]
    M[0] = [[1, 2], [2, 3]]

    # stage 1
    C[1] = [[3, 4], [1, 0], [1, 3, 1]]
    M[1] = [[1, 2], [2, 3], [2, 3, 4]]

    # stage 2
    C[2] = [[1], [5, 2], [3, 1], [3, 4]]
    M[2] = [[1], [1, 2], [2, 3], [2, 3]]

    # stage 3
    C[3] = [[0, 5], [2, 4], [2, 5]]
    M[3] = [[1, 2], [1, 2], [1, 2]]

    # terminal costs g_4(1)=4, g_4(2)=0
    Jh = [4, 0]

    u, J = dptv(M, C, Jh)
    for k in range(stages):
        for i in range(len(J[k])):
            print(f"stage {k}, state {i + 1}: J = {J[k][i]}, optimal action(s) = {[int(a) for a in u[k][i]]}")
