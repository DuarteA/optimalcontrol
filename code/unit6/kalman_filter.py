"""Discrete-time Kalman filter (prediction + correction) for a linear
Gaussian system, as introduced in Unit 6.

Convention (matches the lecture slides): dynamic model
    x_{k+1} = A x_k + B u_k + w_k,   w_k ~ N(0, W)
output equation
    y_k = C x_k + n_k,               n_k ~ N(0, V)
Then P_{x_k|I_k} = N(xhat_{k|k}, Phi_{k|k}), obtained by iterating
    xhat_{k|k}   = xhat_{k|k-1} + L_k (y_k - C xhat_{k|k-1})
    xhat_{k+1|k} = A xhat_{k|k} + B u_k
    Phi_{k|k}    = Phi_{k|k-1} - Phi_{k|k-1} C' (C Phi_{k|k-1} C' + V)^-1 C Phi_{k|k-1}
    Phi_{k+1|k}  = A Phi_{k|k} A' + W
    L_k          = Phi_{k|k-1} C' (C Phi_{k|k-1} C' + V)^-1
"""
import numpy as np


def correct(x_pred, Phi_pred, y, C, V):
    """Correction/innovation step. Returns (x_upd, Phi_upd, L)."""
    C = np.atleast_2d(C)
    S = C @ Phi_pred @ C.T + V
    L = Phi_pred @ C.T @ np.linalg.inv(S)
    innovation = y - C @ x_pred
    x_upd = x_pred + L @ innovation
    Phi_upd = Phi_pred - L @ C @ Phi_pred
    return x_upd, Phi_upd, L


def predict(x_upd, Phi_upd, A, W, B=None, u=None):
    """Prediction/time-update step. Returns (x_pred_next, Phi_pred_next)."""
    x_next = A @ x_upd
    if B is not None and u is not None:
        x_next = x_next + B @ u
    Phi_next = A @ Phi_upd @ A.T + W
    return x_next, Phi_next


def kalman_filter(ys, A, C, W, V, x0_mean, Phi0, B=None, us=None):
    """Run the full Kalman filter over a sequence of measurements.

    ys: list/array of measurements y_0, y_1, ..., y_{N-1}
    Returns lists of posterior means and covariances
    (x_{0|0}, ..., x_{N-1|N-1}) and (Phi_{0|0}, ..., Phi_{N-1|N-1}), and
    the corresponding gains L_0, ..., L_{N-1}.
    """
    x_pred, Phi_pred = np.asarray(x0_mean, dtype=float).reshape(-1, 1), Phi0
    xs, Phis, Ls = [], [], []
    for k, y in enumerate(ys):
        y = np.atleast_1d(y).reshape(-1, 1)
        x_upd, Phi_upd, L = correct(x_pred, Phi_pred, y, C, V)
        xs.append(x_upd.flatten())
        Phis.append(Phi_upd)
        Ls.append(L)
        u = None if us is None else np.atleast_1d(us[k]).reshape(-1, 1)
        x_pred, Phi_pred = predict(x_upd, Phi_upd, A, W, B=B, u=u)
    return xs, Phis, Ls


def stationary_kalman_filter(A, C, W, V, tol=1e-10, max_iter=10000):
    """Iterate the prediction-covariance Riccati equation to convergence
    (equivalent to solving the algebraic Riccati equation dual to the
    LQR problem, cf. Lecture 6 Appendix E "duality" remark).

    Returns (Phi_bar, L_bar): the steady-state prediction covariance and
    the corresponding steady-state Kalman gain.
    """
    C = np.atleast_2d(C)
    Phi = W.copy()
    for _ in range(max_iter):
        S = C @ Phi @ C.T + V
        L = Phi @ C.T @ np.linalg.inv(S)
        Phi_upd = Phi - L @ C @ Phi
        Phi_next = A @ Phi_upd @ A.T + W
        if np.max(np.abs(Phi_next - Phi)) < tol:
            Phi = Phi_next
            break
        Phi = Phi_next
    S = C @ Phi @ C.T + V
    L_bar = Phi @ C.T @ np.linalg.inv(S)
    return Phi, L_bar


if __name__ == "__main__":
    # Smoke test: reproduces the mass-spring-system example of Lecture 6,
    # Appendix C ("Example: Kalman filter" / "Iterations").
    A = np.array([[0.8090, 0.0935], [-3.6932, 0.8090]])
    C = np.array([[1.0, 0.0]])
    W = np.array([[0.0, 0.0], [0.0, 0.25]])
    V = np.array([[0.01]])
    x0_mean = np.array([1.3, 0.0])
    Phi0 = np.eye(2)

    ys = [1.0979, 0.7542]
    xs, Phis, Ls = kalman_filter(ys, A, C, W, V, x0_mean, Phi0)

    print("L0 =", Ls[0].flatten(), " expected ~ [0.9901, 0]")
    print("Phi_0|0 =\n", Phis[0], "\n expected ~ [[0.0099, 0], [0, 1]]")
    print("x_0|0 =", xs[0], " expected ~ [1.0999, 0]")
    print("x_1|1 =", xs[1], " expected ~ [0.8080, -4.3102]")

    assert np.allclose(Ls[0].flatten(), [0.9901, 0.0], atol=2e-3)
    assert np.allclose(Phis[0], [[0.0099, 0.0], [0.0, 1.0]], atol=2e-3)
    assert np.allclose(xs[0], [1.0999, 0.0], atol=2e-3)
    assert np.allclose(xs[1], [0.8080, -4.3102], atol=2e-3)

    print("\nAll checks passed -- matches Lecture 6, Appendix C's numbers exactly.")
