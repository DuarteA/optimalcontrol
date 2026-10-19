"""Iterative LQR (iLQR), as introduced in Unit 9.

Same running example as shooting.py/collocation.py: minimum-effort swing-up
of a torque-driven pendulum, xdot=[thetadot,-sin(theta)+u], from
x(0)=[0,0] (hanging down) to x(T)=[pi,0] (inverted) over a fixed horizon T,
now with the terminal condition treated as a heavily-weighted quadratic
terminal cost (x_h-xtarget)'Qh(x_h-xtarget), for a direct, apples-to-apples
comparison against the penalized formulations used by shooting/collocation.

Unlike shooting and collocation, which hand the whole trajectory to a
generic (Levenberg-Marquardt) NLP solver, iLQR exploits the problem's
sequential/DP structure directly: at each iteration it linearizes the
dynamics and quadratizes the cost around a nominal trajectory, runs a
backward Riccati-like pass to get a local time-varying feedback policy
(exactly the Unit 5 LQR backward recursion, but re-derived at every
iteration around a moving nominal trajectory instead of solved once for a
globally linear system), and a forward pass (with a line search) to obtain
an improved nominal trajectory. Iterating this converges to a local
optimum of the nonlinear problem -- this is "DDP without the
second-order dynamics terms"; full DDP additionally keeps the
f_xx/f_uu/f_ux terms in the backward pass.
"""
import numpy as np

OMEGA2 = 1.0


def f_cont(x, u):
    theta, thetadot = x
    return np.array([thetadot, -OMEGA2 * np.sin(theta) + u])


def rk4_step(x, u, tau):
    k1 = f_cont(x, u)
    k2 = f_cont(x + 0.5 * tau * k1, u)
    k3 = f_cont(x + 0.5 * tau * k2, u)
    k4 = f_cont(x + tau * k3, u)
    return x + (tau / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def step_jacobians(x, u, tau, eps=1e-6):
    A = np.zeros((2, 2))
    for i in range(2):
        dx = np.zeros(2); dx[i] = eps
        A[:, i] = (rk4_step(x + dx, u, tau) - rk4_step(x - dx, u, tau)) / (2 * eps)
    B = (rk4_step(x, u + eps, tau) - rk4_step(x, u - eps, tau)) / (2 * eps)
    return A, B


def simulate(x0, u, tau):
    h = len(u)
    xs = np.zeros((h + 1, 2))
    xs[0] = x0
    for k in range(h):
        xs[k + 1] = rk4_step(xs[k], u[k], tau)
    return xs


def trajectory_cost(xs, u, tau, xtarget, Qh, R=1.0):
    running = tau * R * np.sum(u**2)
    terminal = (xs[-1] - xtarget) @ Qh @ (xs[-1] - xtarget)
    return running + terminal


def ilqr_solve(x0, xtarget, h, tau, Qh=None, R=1.0, iters=60, u0=None):
    """iLQR: returns (u, xs, cost, cost_history)."""
    if Qh is None:
        Qh = 150.0 * np.eye(2)
    u = np.zeros(h) if u0 is None else np.array(u0, dtype=float)
    xs = simulate(x0, u, tau)
    cost = trajectory_cost(xs, u, tau, xtarget, Qh, R)
    cost_hist = [cost]

    for _ in range(iters):
        # backward pass
        Vx = 2 * Qh @ (xs[-1] - xtarget)
        Vxx = 2 * Qh
        Ks, ks = [None] * h, [None] * h
        for k in range(h - 1, -1, -1):
            A, B = step_jacobians(xs[k], u[k], tau)
            lu = 2 * tau * R * u[k]          # l_u  (l_x=0: no running state cost)
            luu = 2 * tau * R                # scalar l_uu
            Qx = A.T @ Vx
            Qu = lu + B @ Vx
            Qxx = A.T @ Vxx @ A
            Quu = luu + B @ Vxx @ B
            Qux = B @ Vxx @ A
            Quu_reg = Quu + 1e-6  # tiny regularization, Quu is a scalar here
            k_ff = -Qu / Quu_reg
            K_fb = -Qux / Quu_reg
            ks[k], Ks[k] = k_ff, K_fb
            Vx = Qx + K_fb * (Quu * k_ff + Qu) + Qux * k_ff  # (using scalar Quu, Qu)
            Vxx = Qxx + np.outer(K_fb, Qux) + np.outer(K_fb, K_fb) * Quu + np.outer(Qux, K_fb)
            Vxx = 0.5 * (Vxx + Vxx.T)

        # forward pass with backtracking line search
        improved = False
        for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]:
            x_new = np.array(x0, dtype=float)
            xs_new = [x_new.copy()]
            u_new = np.zeros(h)
            for k in range(h):
                du = alpha * ks[k] + Ks[k] @ (x_new - xs[k])
                u_new[k] = u[k] + du
                x_new = rk4_step(x_new, u_new[k], tau)
                xs_new.append(x_new.copy())
            xs_new = np.array(xs_new)
            cost_new = trajectory_cost(xs_new, u_new, tau, xtarget, Qh, R)
            if cost_new < cost:
                xs, u, cost = xs_new, u_new, cost_new
                improved = True
                break
        cost_hist.append(cost)
        if not improved:
            break

    return u, xs, cost, cost_hist


if __name__ == "__main__":
    x0 = np.array([0.0, 0.0])
    xtarget = np.array([np.pi, 0.0])
    T, h = 6.0, 60
    tau = T / h
    Qh = 300.0 * np.eye(2)

    print(f"iLQR: pendulum swing-up (h={h}, tau={tau:.3f})")
    u_star, xs, cost, hist = ilqr_solve(x0, xtarget, h, tau, Qh=Qh, iters=60)
    print(f"  converged in {len(hist)-1} iterations")
    print(f"  final state: theta={xs[-1,0]:.4f} (target pi={np.pi:.4f}), "
          f"thetadot={xs[-1,1]:.4f}")
    print(f"  running (effort) cost = {tau*np.sum(u_star**2):.4f}, "
          f"total (incl. terminal penalty) = {cost:.4f}")
    print(f"  cost history (every 10th iter): "
          f"{[round(c,3) for c in hist[::10]]}")

    print()
    print("Comparing against collocation.py's Hermite-Simpson solution "
          "(same problem, hard terminal constraint via a large penalty):")
    import collocation
    xs_hs, u_hs, cost_hs = collocation.hermite_simpson_collocation_solve(
        x0, xtarget, N=30, tau=T / 30)
    print(f"  Hermite-Simpson: theta_f={xs_hs[-1,0]:.4f}, effort={cost_hs:.4f}")
    print(f"  iLQR:            theta_f={xs[-1,0]:.4f}, effort={tau*np.sum(u_star**2):.4f}")
