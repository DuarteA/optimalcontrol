"""Exact dynamic programming, base-policy cost evaluation, and rollout for
switched linear systems, as introduced in Unit 8.

Convention (matches the lecture slides): switched linear system
    x_{k+1} = A_{sigma_k} x_k,   sigma_k in {0,...,m-1},
cost sum_{k=0}^{h-1} x_k'Qx_k + x_h'Qh xh (or just a terminal cost, as in
Exercise 8.1). Three ways to pick sigma_k are implemented:
  - dp_cost_matrices: exact backward DP, representing the cost-to-go
    J_k(x)=min_i x'Z_i x as a growing list of candidate quadratic-form
    matrices Z_i (one per possible future mode sequence) -- exact but
    grows as m^(h-k), only usable for small h.
  - base_policy_cost: the (stationary) cost matrix P_bar of always
    applying one fixed mode, via the discrete Lyapunov equation.
  - rollout_policy: the one-step-lookahead (H=1) rollout policy that
    compares each mode's one-step cost plus the base policy's cost-to-go,
    a cheap approximation to the exact DP above that scales to any
    number of modes/stages.
"""
import numpy as np


def dp_cost_matrices(A_list, terminal_Q, h):
    """Exact backward DP over a switched linear system with h stages and
    (only) a terminal cost x_h' terminal_Q x_h (no running cost -- matches
    Exercise 8.1). Returns a list `levels` of length h+1, where
    levels[k] is the list of candidate matrices Z such that
    J_k(x) = min_i x' levels[k][i] x."""
    levels = [[terminal_Q]]
    for k in range(h - 1, -1, -1):
        prev = levels[0]
        new_level = [A.T @ Z @ A for Z in prev for A in A_list]
        levels.insert(0, new_level)
    return levels


def optimal_value_and_mode(x, matrices):
    """Given a list of candidate quadratic-form matrices (as returned per
    stage by dp_cost_matrices), return (min cost, argmin index) at state x."""
    x = np.asarray(x, dtype=float)
    vals = [x @ Z @ x for Z in matrices]
    idx = int(np.argmin(vals))
    return vals[idx], idx


def base_policy_cost(A, Q, tol=1e-12, max_iter=50000):
    """Solve the discrete Lyapunov equation P = A'PA + Q (i.e.
    A'PA - P = -Q) by fixed-point iteration, giving the infinite-horizon
    cost sum_{k=0}^infty x_k'Qx_k = x_0'Px_0 of always applying mode A
    (requires A asymptotically stable; the iteration will fail to
    converge, growing without bound, if it is not -- matching the "no
    positive definite solution" caveat in Exercise 8.2(ii)(b))."""
    P = np.zeros_like(Q)
    for _ in range(max_iter):
        P_new = A.T @ P @ A + Q
        if np.max(np.abs(P_new - P)) < tol:
            return P_new
        if np.max(np.abs(P_new)) > 1e12:
            raise ValueError("Lyapunov iteration diverged -- the base "
                              "policy's mode is not asymptotically stable.")
        P = P_new
    return P


def rollout_policy(x, A_list, Q, P_base):
    """H=1 rollout policy: choose the mode minimizing the one-step cost
    plus the base policy's cost-to-go from the resulting state,
    x'Qx + (A_i x)'P_base(A_i x). Since the x'Qx term does not depend on
    the mode, this is equivalent to minimizing x' A_i' P_base A_i x.
    Returns (best mode index, cost)."""
    x = np.asarray(x, dtype=float)
    vals = [x @ (A.T @ P_base @ A) @ x for A in A_list]
    idx = int(np.argmin(vals))
    return idx, x @ Q @ x + vals[idx]


def simulate_switched(x0, A_list, mode_fn, N):
    """Simulate N steps of a switched linear system, where mode_fn(x, k)
    returns the mode index to apply at state x, stage k. Returns
    (xs, modes) with xs of length N+1."""
    x = np.asarray(x0, dtype=float)
    xs = [x]
    modes = []
    for k in range(N):
        i = mode_fn(x, k)
        modes.append(i)
        x = A_list[i] @ x
        xs.append(x)
    return np.array(xs), modes


if __name__ == "__main__":
    # Exercise 8.1: A_0, A_1 switched system, h=2, terminal cost ||x_h||^2.
    A0 = np.array([[0.8, 0.1], [0.2, 0.8]])
    A1 = np.array([[1.0, 0.1], [0.0, 0.5]])
    levels = dp_cost_matrices([A0, A1], np.eye(2), h=2)
    print("Exercise 8.1: candidate Z matrices for J_0 "
          "(should reproduce Z1..Z4 from the exercise solution):")
    for i, Z in enumerate(levels[0]):
        print(f"  Z_{i+1} =\n{np.round(Z, 4)}")
    x_test = np.array([1.0, -0.5])
    val, idx = optimal_value_and_mode(x_test, levels[0])
    print(f"  at x={x_test}: optimal sigma_0 leads to Z_{idx+1}, J_0={val:.4f}")

    print()
    print("Exercise 8.3: switching between two fixed controllers.")
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    B = np.array([[0.0], [1.0]])
    K1 = np.array([-0.5, -1.0])
    K2 = np.array([-1 / 3, -6 / 5])
    A1c = A + np.outer(B[:, 0], K1)
    A2c = A + np.outer(B[:, 0], K2)
    Q = np.eye(2)
    P_bar = base_policy_cost(A1c, Q)
    print("  P_bar (base policy sigma=1 forever) =\n", np.round(P_bar, 4),
          " -- expected [[3,2],[2,4]]")
    x0 = np.array([1.0, 1.0])
    xs, modes = simulate_switched(
        x0, [A1c, A2c],
        mode_fn=lambda x, k: rollout_policy(x, [A1c, A2c], Q, P_bar)[0],
        N=20,
    )
    print(f"  rollout trajectory norm at k=0,5,10,20: "
          f"{np.linalg.norm(xs[0]):.4f}, {np.linalg.norm(xs[5]):.4f}, "
          f"{np.linalg.norm(xs[10]):.4f}, {np.linalg.norm(xs[20]):.4f} "
          f"(should shrink towards 0, confirming Exercise 8.3(iv))")
