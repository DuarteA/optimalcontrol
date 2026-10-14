"""Direct collocation for a nonlinear stage decision problem, as introduced
in Unit 9.

Same running example as shooting.py: minimum-effort swing-up of a
torque-driven pendulum, xdot=[thetadot, -sin(theta)+u], from x(0)=[0,0]
(hanging down) to x(T)=[pi,0] (inverted) over a fixed horizon T.

Unlike shooting (Section 2 / shooting.py), collocation does NOT eliminate
the states by simulation: both states and controls at every grid node are
decision variables, and the dynamics appear only as equality ("defect")
constraints tying consecutive nodes together. This trades a small dense
NLP (shooting) for a much larger but *sparse* and well-conditioned one --
here solved densely for simplicity (the problems are still small), but
in production tools the sparsity is exploited directly.

Two schemes are implemented, both derived directly from a quadrature rule
applied to the ODE x(t+tau)-x(t) = integral_t^{t+tau} f(x,u) dt:
  - trapezoidal: approximate the integral by the trapezoidal rule, giving
    a 2nd-order-accurate defect x_{k+1}-x_k-(tau/2)(f_k+f_{k+1})=0.
  - Hermite-Simpson: interpolate x by a cubic Hermite polynomial over
    each interval, evaluate the dynamics at the interval midpoint of that
    interpolant, and apply Simpson's rule -- 4th-order accurate.

As in shooting.py, all equality constraints (defects, boundary conditions)
are treated as penalized least-squares residuals and solved by
Levenberg-Marquardt, keeping this module numpy-only.
"""
import numpy as np

np.seterr(all="ignore")  # occasional overflow on rejected (too-large)
# Levenberg-Marquardt trial steps is expected and harmless -- the step is
# discarded whenever it does not decrease the cost

OMEGA2 = 1.0


def f_cont(x, u):
    theta, thetadot = x
    return np.array([thetadot, -OMEGA2 * np.sin(theta) + u])


def f_jacobians(x, u, eps=1e-6):
    """Finite-difference Jacobians of the continuous-time dynamics f_cont
    itself (not of an integration step -- collocation only ever calls f,
    never integrates)."""
    Ax = np.zeros((2, 2))
    for i in range(2):
        dx = np.zeros(2); dx[i] = eps
        Ax[:, i] = (f_cont(x + dx, u) - f_cont(x - dx, u)) / (2 * eps)
    Bu = (f_cont(x, u + eps) - f_cont(x, u - eps)) / (2 * eps)
    return Ax, Bu


def _levenberg_marquardt(residual_fn, v0, iters=100, lam0=1e-2):
    v = np.array(v0, dtype=float)
    R, J = residual_fn(v)
    cost = np.sum(R**2)
    lam = lam0
    n = len(v)
    for _ in range(iters):
        JTJ, JTR = J.T @ J, J.T @ R
        for _ in range(30):
            delta = -np.linalg.solve(JTJ + lam * np.eye(n), JTR)
            v_new = v + delta
            R_new, J_new = residual_fn(v_new)
            cost_new = np.sum(R_new**2)
            if cost_new < cost:
                v, R, J, cost = v_new, R_new, J_new, cost_new
                lam = max(lam * 0.5, 1e-8)
                break
            lam *= 2.0
    return v


def _unpack(v, N):
    """v = [x_1,...,x_N (2 each); u_0,...,u_N]; x_0 is fixed, not in v."""
    x = v[:2 * N].reshape(N, 2)
    u = v[2 * N:]
    return x, u


def trapezoidal_collocation_solve(x0, xtarget, N, tau, rho=300.0, iters=100):
    """Trapezoidal direct collocation. N intervals, N+1 nodes (x_0=x0
    fixed; x_1..x_N and u_0..u_N are decision variables). Returns
    (xs, u, effort_cost)."""
    nX, nU = 2 * N, N + 1

    def residual(v):
        x, u = _unpack(v, N)
        xs = np.vstack([x0, x])  # (N+1, 2), xs[0]=x0
        fk = np.array([f_cont(xs[k], u[k]) for k in range(N + 1)])

        # effort quadrature (trapezoidal weights: endpoints get tau/2)
        w = np.full(N + 1, tau); w[0] = w[-1] = tau / 2
        res_eff = np.sqrt(w) * u
        J_eff = np.zeros((N + 1, nX + nU))
        for k in range(N + 1):
            J_eff[k, nX + k] = np.sqrt(w[k])

        # defects
        res_def = np.zeros(2 * N)
        J_def = np.zeros((2 * N, nX + nU))
        for k in range(N):
            Ak, Bk = f_jacobians(xs[k], u[k])
            Ak1, Bk1 = f_jacobians(xs[k + 1], u[k + 1])
            defect = xs[k + 1] - xs[k] - (tau / 2) * (fk[k] + fk[k + 1])
            res_def[2 * k:2 * k + 2] = np.sqrt(rho) * defect
            # d(defect)/dx_k (only if k>0, since x_0 is fixed, not a var)
            if k > 0:
                J_def[2 * k:2 * k + 2, 2 * (k - 1):2 * k] = np.sqrt(rho) * (-np.eye(2) - (tau / 2) * Ak)
            J_def[2 * k:2 * k + 2, 2 * k:2 * k + 2] = np.sqrt(rho) * (np.eye(2) - (tau / 2) * Ak1)
            J_def[2 * k:2 * k + 2, nX + k] = np.sqrt(rho) * (-(tau / 2) * Bk)
            J_def[2 * k:2 * k + 2, nX + k + 1] = np.sqrt(rho) * (-(tau / 2) * Bk1)

        # terminal condition x_N = xtarget
        res_term = np.sqrt(rho) * (xs[-1] - xtarget)
        J_term = np.zeros((2, nX + nU))
        J_term[:, 2 * (N - 1):2 * N] = np.sqrt(rho) * np.eye(2)

        return (np.concatenate([res_eff, res_def, res_term]),
                np.vstack([J_eff, J_def, J_term]))

    v0 = np.zeros(nX + nU)
    for k in range(N):
        v0[2 * k:2 * k + 2] = x0 + (k + 1) / N * (xtarget - x0)  # straight-line guess
    v = _levenberg_marquardt(residual, v0, iters=iters)
    x, u = _unpack(v, N)
    xs = np.vstack([x0, x])
    w = np.full(N + 1, tau); w[0] = w[-1] = tau / 2
    return xs, u, np.sum(w * u**2)


def hermite_simpson_collocation_solve(x0, xtarget, N, tau, rho=300.0, iters=100):
    """Hermite-Simpson direct collocation. Same decision-variable layout as
    trapezoidal; the defect for interval k additionally evaluates the
    dynamics at the interval's Hermite-interpolated midpoint,
        x_mid = (x_k+x_{k+1})/2 + (tau/8)(f_k-f_{k+1}),  u_mid=(u_k+u_{k+1})/2,
    and applies Simpson's rule:
        x_{k+1}-x_k-(tau/6)(f_k+4 f(x_mid,u_mid)+f_{k+1}) = 0.
    Jacobians of the midpoint term are obtained by finite differences on
    the whole map (x_k,u_k,x_{k+1},u_{k+1}) -> f(x_mid,u_mid), to avoid
    hand-deriving the chain rule through the interpolant."""
    nX, nU = 2 * N, N + 1

    def f_mid(xk, uk, xk1, uk1):
        fk, fk1 = f_cont(xk, uk), f_cont(xk1, uk1)
        xmid = 0.5 * (xk + xk1) + (tau / 8) * (fk - fk1)
        umid = 0.5 * (uk + uk1)
        return f_cont(xmid, umid), xmid, umid

    def d_fmid(xk, uk, xk1, uk1, eps=1e-6):
        """Finite-difference Jacobian of f_mid w.r.t. (xk,uk,xk1,uk1), 2x6."""
        base, _, _ = f_mid(xk, uk, xk1, uk1)
        J = np.zeros((2, 6))
        args = [xk[0], xk[1], uk, xk1[0], xk1[1], uk1]

        def eval_at(a):
            return f_mid(np.array([a[0], a[1]]), a[2], np.array([a[3], a[4]]), a[5])[0]

        for i in range(6):
            a_p, a_m = list(args), list(args)
            a_p[i] += eps; a_m[i] -= eps
            J[:, i] = (eval_at(a_p) - eval_at(a_m)) / (2 * eps)
        return base, J

    def residual(v):
        x, u = _unpack(v, N)
        xs = np.vstack([x0, x])
        fk = np.array([f_cont(xs[k], u[k]) for k in range(N + 1)])

        w = np.full(N + 1, tau); w[0] = w[-1] = tau / 2  # same total effort weight bookkeeping as trapezoidal, applied per-node
        res_eff = np.sqrt(w) * u
        J_eff = np.zeros((N + 1, nX + nU))
        for k in range(N + 1):
            J_eff[k, nX + k] = np.sqrt(w[k])

        res_def = np.zeros(2 * N)
        J_def = np.zeros((2 * N, nX + nU))
        for k in range(N):
            Ak, Bk = f_jacobians(xs[k], u[k])
            Ak1, Bk1 = f_jacobians(xs[k + 1], u[k + 1])
            fmid, Jmid = d_fmid(xs[k], u[k], xs[k + 1], u[k + 1])
            defect = xs[k + 1] - xs[k] - (tau / 6) * (fk[k] + 4 * fmid + fk[k + 1])
            res_def[2 * k:2 * k + 2] = np.sqrt(rho) * defect

            dD_dxk = -np.eye(2) - (tau / 6) * (Ak + 4 * Jmid[:, 0:2])
            dD_duk = -(tau / 6) * (Bk + 4 * Jmid[:, 2])
            dD_dxk1 = np.eye(2) - (tau / 6) * (Ak1 + 4 * Jmid[:, 3:5])
            dD_duk1 = -(tau / 6) * (Bk1 + 4 * Jmid[:, 5])
            if k > 0:
                J_def[2 * k:2 * k + 2, 2 * (k - 1):2 * k] = np.sqrt(rho) * dD_dxk
            J_def[2 * k:2 * k + 2, nX + k] = np.sqrt(rho) * dD_duk
            J_def[2 * k:2 * k + 2, 2 * k:2 * k + 2] = np.sqrt(rho) * dD_dxk1
            J_def[2 * k:2 * k + 2, nX + k + 1] = np.sqrt(rho) * dD_duk1

        res_term = np.sqrt(rho) * (xs[-1] - xtarget)
        J_term = np.zeros((2, nX + nU))
        J_term[:, 2 * (N - 1):2 * N] = np.sqrt(rho) * np.eye(2)

        return (np.concatenate([res_eff, res_def, res_term]),
                np.vstack([J_eff, J_def, J_term]))

    v0 = np.zeros(nX + nU)
    for k in range(N):
        v0[2 * k:2 * k + 2] = x0 + (k + 1) / N * (xtarget - x0)
    v = _levenberg_marquardt(residual, v0, iters=iters)
    x, u = _unpack(v, N)
    xs = np.vstack([x0, x])
    w = np.full(N + 1, tau); w[0] = w[-1] = tau / 2
    return xs, u, np.sum(w * u**2)


if __name__ == "__main__":
    x0 = np.array([0.0, 0.0])
    xtarget = np.array([np.pi, 0.0])
    T, N = 6.0, 30
    tau = T / N

    print(f"Trapezoidal collocation: pendulum swing-up (N={N} intervals, tau={tau:.3f})")
    xs_t, u_t, cost_t = trapezoidal_collocation_solve(x0, xtarget, N, tau)
    print(f"  final state: theta={xs_t[-1,0]:.4f} (target pi={np.pi:.4f}), "
          f"thetadot={xs_t[-1,1]:.4f}")
    print(f"  effort cost = {cost_t:.4f}")

    print()
    print(f"Hermite-Simpson collocation: same problem")
    xs_h, u_h, cost_h = hermite_simpson_collocation_solve(x0, xtarget, N, tau)
    print(f"  final state: theta={xs_h[-1,0]:.4f}, thetadot={xs_h[-1,1]:.4f}")
    print(f"  effort cost = {cost_h:.4f}")

    print()
    print("Accuracy comparison as N grows (both should converge to the same "
          "optimal effort; Hermite-Simpson should get there with fewer nodes):")
    for N_ in [6, 10, 20, 40]:
        tau_ = T / N_
        _, _, ct = trapezoidal_collocation_solve(x0, xtarget, N_, tau_, iters=150)
        _, _, ch = hermite_simpson_collocation_solve(x0, xtarget, N_, tau_, iters=150)
        print(f"  N={N_:3d}: trapezoidal cost={ct:.4f}   Hermite-Simpson cost={ch:.4f}")
