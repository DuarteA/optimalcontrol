"""Discrete-time Riccati recursion for the linear quadratic regulator
(finite-horizon time-varying case and infinite-horizon/algebraic case),
as introduced in Unit 5.

Convention (matches the lecture slides): dynamic model
    x_{k+1} = A x_k + B u_k
cost
    sum_{k=0}^{h-1} [x_k; u_k]' [[Q, S], [S', R]] [x_k; u_k] + x_h' Qh x_h
optimal policy u_k = K_k x_k, with the Riccati recursion (backward)
    P_h = Qh
    K_k = -(B' P_{k+1} B + R)^-1 (S' + B' P_{k+1} A)
    P_k = A' P_{k+1} A + Q - (S + A' P_{k+1} B)(B' P_{k+1} B + R)^-1 (S' + B' P_{k+1} A)
"""
import numpy as np


def riccati_step(A, B, Q, R, S, P_next):
    """One backward Riccati step. Returns (K, P)."""
    n_u = B.shape[1]
    if S is None:
        S = np.zeros((A.shape[0], n_u))
    M = B.T @ P_next @ B + R
    N = S.T + B.T @ P_next @ A
    K = -np.linalg.solve(M, N)
    P = A.T @ P_next @ A + Q - N.T @ np.linalg.solve(M, N)
    return K, P


def finite_horizon_lqr(A, B, Q, R, Qh, h, S=None):
    """Finite-horizon LQR via backward Riccati recursion.

    Returns (Ks, Ps): lists of length h and h+1 respectively, with
    Ks[k] the gain u_k = Ks[k] @ x_k for k in {0,...,h-1} and
    Ps[k] the cost-to-go matrix J_k(x_k) = x_k' Ps[k] x_k for k in {0,...,h}.
    """
    n = A.shape[0]
    Ps = [None] * (h + 1)
    Ks = [None] * h
    Ps[h] = Qh
    for k in range(h - 1, -1, -1):
        K, P = riccati_step(A, B, Q, R, S, Ps[k + 1])
        Ks[k] = K
        Ps[k] = P
    return Ks, Ps


def infinite_horizon_lqr(A, B, Q, R, S=None, tol=1e-10, max_iter=10000):
    """Infinite-horizon LQR by iterating the Riccati recursion to
    convergence (equivalent to solving the algebraic Riccati equation).

    Returns (Kbar, Pbar).
    """
    n = A.shape[0]
    P = Q.copy()
    for _ in range(max_iter):
        K, P_new = riccati_step(A, B, Q, R, S, P)
        if np.max(np.abs(P_new - P)) < tol:
            P = P_new
            break
        P = P_new
    K, P = riccati_step(A, B, Q, R, S, P)
    return K, P


def simulate(A, B, Ks, x0, steps=None):
    """Simulate x_{k+1} = A x_k + B K_k x_k starting from x0.

    Ks can be a single gain matrix (time-invariant) or a list of gains
    (time-varying, as returned by finite_horizon_lqr).
    """
    x = np.asarray(x0, dtype=float).reshape(-1, 1)
    xs = [x.flatten()]
    us = []
    n_steps = steps if steps is not None else len(Ks)
    for k in range(n_steps):
        K = Ks[k] if isinstance(Ks, list) else Ks
        u = K @ x
        us.append(u.flatten())
        x = A @ x + B @ u
        xs.append(x.flatten())
    return np.array(xs), np.array(us)


if __name__ == "__main__":
    # Smoke test: reproduces the double-integrator example of Lecture 5
    # (slides "Example: double integrator" / "Dynamic programming").
    tau = 0.2
    A = np.array([[1, tau], [0, 1]])
    B = np.array([[tau**2 / 2], [tau]])
    Q = np.eye(2)
    R = np.array([[1.0]])
    Qh = 10 * np.eye(2)
    h = 5

    Ks, Ps = finite_horizon_lqr(A, B, Q, R, Qh, h)
    print("K0 =", Ks[0], " expected ~ [-0.918, -1.503]")
    print("P0 =\n", Ps[0], "\n expected ~ [[12.121, 5.702], [5.702, 9.085]]")

    expected_K0 = np.array([-0.918, -1.503])
    assert np.allclose(Ks[0].flatten(), expected_K0, atol=2e-2), Ks[0]
    expected_P0 = np.array([[12.121, 5.702], [5.702, 9.085]])
    assert np.allclose(Ps[0], expected_P0, atol=2e-2), Ps[0]

    xs, us = simulate(A, B, Ks, x0=[1, 0])
    print("Optimal path x0..x5:\n", xs)
    expected_x1 = np.array([0.982, -0.184])
    assert np.allclose(xs[1], expected_x1, atol=2e-2), xs[1]

    # Infinite-horizon check against Lecture 6's double-integrator example
    # (same A, B, Q=I, R=1): K -> [-0.8412, -1.54]
    Kbar, Pbar = infinite_horizon_lqr(A, B, np.eye(2), np.array([[1.0]]))
    print("\nInfinite-horizon Kbar =", Kbar, " expected ~ [-0.8412, -1.54]")
    assert np.allclose(Kbar.flatten(), [-0.8412, -1.54], atol=2e-2), Kbar

    print("\nAll checks passed -- matches Lecture 5/6 double-integrator examples.")
