"""Model predictive control (MPC) with box input constraints, as
introduced in Unit 8.

Convention (matches the lecture's "Quadratic programming formulation"
slides): dynamic model x_{k+1}=A x_k+B u_k, finite-horizon cost
    sum_{k=0}^{H-1} x_k'Qx_k+u_k'Ru_k + x_H'Qh xH,
with x_0 the current (measured) state. Stacking u=(u_0,...,u_{H-1}) and
writing the predicted states X=(x_1,...,x_H) as an affine function of u,
X = Sx x0 + Su u, turns the problem into the batch QP
    min_u  u'(Su'Qbar Su + Rbar)u + 2(Sx x0)'Qbar Su u,  u_min <= u_k <= u_max,
solved here by projected gradient descent (the same iterative-minimization
idea as Unit 7's gradient_method.py, now applied with a projection step
for the box constraints) rather than a general-purpose QP solver, keeping
this module numpy-only. MPC applies only the first control u_0 and
re-solves at the next time step (receding horizon).
"""
import numpy as np


def _block_diag(mats):
    sizes = [m.shape[0] for m in mats]
    n = sum(sizes)
    out = np.zeros((n, n))
    i = 0
    for m, s in zip(mats, sizes):
        out[i:i + s, i:i + s] = m
        i += s
    return out


def prediction_matrices(A, B, H):
    """Sx, Su such that X = Sx @ x0 + Su @ u stacks x_1,...,x_H (each of
    dimension n) as a function of u = (u_0,...,u_{H-1}) (each of dimension m)."""
    n, m = A.shape[0], B.shape[1]
    Sx = np.zeros((H * n, n))
    Su = np.zeros((H * n, H * m))
    Apow = np.eye(n)
    for i in range(H):
        Apow = A @ Apow
        Sx[i * n:(i + 1) * n, :] = Apow
    for i in range(H):
        Apow2 = np.eye(n)
        for j in range(i, -1, -1):
            Su[i * n:(i + 1) * n, j * m:(j + 1) * m] = Apow2 @ B
            Apow2 = A @ Apow2
    return Sx, Su


def mpc_solve(x0, A, B, Q, R, Qh, H, u_min=None, u_max=None, iters=3000):
    """Solve the finite-horizon (possibly box-constrained) QP for the
    control sequence u=(u_0,...,u_{H-1}) minimizing the cost from x0.
    Returns u as an (H, m) array. u_min/u_max: scalars or length-m arrays,
    or None for no constraint on that side."""
    x0 = np.atleast_1d(x0).astype(float)
    n, m = A.shape[0], B.shape[1]
    Sx, Su = prediction_matrices(A, B, H)
    Qbar = _block_diag([Q] * (H - 1) + [Qh])
    Rbar = _block_diag([R] * H)
    Hqp = 2 * (Su.T @ Qbar @ Su + Rbar)
    fqp = 2 * Su.T @ Qbar @ Sx @ x0

    # step size from the largest eigenvalue of Hqp (Lipschitz constant of
    # the gradient), for guaranteed convergence of projected gradient descent
    L = np.max(np.linalg.eigvalsh(Hqp))
    lr = 1.0 / L if L > 0 else 1e-3

    u = np.zeros(H * m)
    lo = -np.inf * np.ones(H * m) if u_min is None else np.tile(u_min, H)
    hi = np.inf * np.ones(H * m) if u_max is None else np.tile(u_max, H)
    for _ in range(iters):
        grad = Hqp @ u + fqp
        u = np.clip(u - lr * grad, lo, hi)
    return u.reshape(H, m)


def mpc_control(x, A, B, Q, R, Qh, H, u_min=None, u_max=None, iters=3000):
    """Receding-horizon control: solve the H-step QP from x and return
    only the first control u_0 (as applied by MPC)."""
    u_seq = mpc_solve(x, A, B, Q, R, Qh, H, u_min, u_max, iters)
    return u_seq[0]


def simulate_mpc(x0, A, B, Q, R, Qh, H, N, u_min=None, u_max=None, iters=1000):
    """Run N steps of receding-horizon MPC in closed loop from x0.
    Returns (xs, us) with xs of length N+1 and us of length N."""
    x0 = np.atleast_1d(x0).astype(float)
    xs = [x0]
    us = []
    x = x0
    for _ in range(N):
        u = mpc_control(x, A, B, Q, R, Qh, H, u_min, u_max, iters)
        x = A @ x + B @ u
        xs.append(x)
        us.append(u)
    return np.array(xs), np.array(us)


if __name__ == "__main__":
    # Double integrator with an input-magnitude constraint, matching
    # B_5_LQinputconstraints.ipynb's setup.
    tau = 0.2
    A = np.array([[1.0, tau], [0.0, 1.0]])
    B = np.array([[tau**2 / 2], [tau]])
    Q = np.eye(2)
    R = np.array([[1.0]])
    Qh = np.eye(2)
    x0 = np.array([1.0, 0.0])
    H = 10

    print("Unconstrained MPC (should match the standard finite-horizon LQR "
          "gain from Unit 5's riccati.py at k=0):")
    u_unconstrained = mpc_solve(x0, A, B, Q, R, Qh, H, u_min=None, u_max=None, iters=3000)
    print("  u_0 =", np.round(u_unconstrained[0], 4))

    print("Box-constrained MPC, |u| <= 0.3:")
    u_constrained = mpc_solve(x0, A, B, Q, R, Qh, H, u_min=-0.3, u_max=0.3, iters=3000)
    print("  u_0 =", np.round(u_constrained[0], 4),
          "-> saturates at the bound" if abs(u_constrained[0, 0]) >= 0.3 - 1e-2 else "")

    print()
    print("Closed-loop simulation, 15 steps, constrained MPC:")
    xs, us = simulate_mpc(x0, A, B, Q, R, Qh, H, N=15, u_min=-0.3, u_max=0.3, iters=800)
    print("  final state:", np.round(xs[-1], 4))
