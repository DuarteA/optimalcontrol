"""Inventory control example (Unit 1: Modeling and the dynamic programming
algorithm).

Extracted from JupyterNotebooks/A_1_inventorycontrol.ipynb. Formulates the
inventory control problem from the Unit 1 notes as a (M, C, T) discrete
decision problem (see dp.py) and solves it with the general dptv() solver.
"""
import numpy as np

from dp import dptv


def inventorycontrol(p, citem, ctrans, cstor, d, gh):
    """Solve a deterministic, finite-horizon inventory control problem.

    Parameters
    ----------
    p      : selling price per item
    citem  : purchase price per item
    ctrans : fixed transportation cost per order (paid whenever an order is
             placed, i.e. whenever the supply u_k > 0)
    cstor  : storage cost per item held in stock
    d      : array of demands [d_0, ..., d_{h-1}]
    gh     : terminal cost g_h(i) for i = 0, ..., N (N = len(gh) - 1)

    Returns
    -------
    u, J : optimal policy and costs-to-go, in the same format as dptv().
    """
    N_1 = len(gh)  # N + 1
    h = len(d)
    DM = [[] for _ in range(h)]
    CM = [[] for _ in range(h)]

    for k in range(h):
        DM[k] = [[] for _ in range(N_1)]
        CM[k] = [[] for _ in range(N_1)]
        for i in range(N_1):          # current stock level i in {0,...,N}
            x_range = N_1 - 1 - i     # max supply u = N - i
            DM[k][i] = [[] for _ in range(x_range + 1)]
            CM[k][i] = [[] for _ in range(x_range + 1)]
            for j in range(x_range + 1):  # supply u = j
                i_1 = max(i + j - d[k], 0) + 1
                DM[k][i][j] = i_1
                c2 = citem * j
                if j > 0:
                    c2 = c2 + ctrans
                c1 = i * cstor
                # cost = storage + ordering - revenue from items sold
                CM[k][i][j] = -(min(d[k], j + i) * p - (c1 + c2))

    u, J = dptv(DM, CM, gh)
    return u, J


if __name__ == "__main__":
    # Same numerical instance as in the Unit 1 notes: h=4 stages, N=4,
    # demand d=[2,2,1,1], p=10, purchase price 5, transportation cost 0.5,
    # storage cost 0.2*i.
    p = 10
    citem = 5
    ctrans = 0.5
    cstor = 0.2
    d = np.array([2, 2, 1, 1])
    gh = -np.array([0, 4.8, 9.6, 14.4, 19.2])

    u, J = inventorycontrol(p, citem, ctrans, cstor, d, gh)
    for k in range(len(J)):
        for i in range(len(J[k])):
            actions = [int(a) - 1 for a in u[k][i]]  # convert back to supply u (0-indexed)
            print(f"stage {k}, stock {i}: J = {J[k][i]:.2f}, optimal supply(ies) = {actions}")
