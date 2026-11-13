"""Single and multiple shooting for a nonlinear stage decision problem, as
introduced in Unit 9.

Running example throughout this module (and throughout Unit 9's companion
scripts): minimum-effort swing-up of a torque-driven pendulum,
    xdot = [thetadot, -sin(theta) + u],   x=[theta, thetadot],
from x(0)=[0,0] (hanging down, stable) to x(T)=[pi,0] (inverted, unstable)
over a fixed horizon T, minimizing the effort integral(u^2)dt. This is the
nonlinear generalization of Unit 7's static-optimization idea: eliminate the
states by forward simulation ("shoot" the trajectory from an initial guess
of the controls) and minimize over the controls alone.

Both solvers here use Levenberg-Marquardt (a damped Gauss-Newton method) on
a penalized least-squares residual, with the residual's Jacobian obtained
by forward sensitivity propagation -- the same chain-rule idea as Unit 7's
`gradient_method.rollout_grad`, generalized to nonlinear dynamics via
per-step Jacobians A_k=df/dx, B_k=df/du (here computed by finite
differences, to keep this module numpy-only, in the same spirit as Unit
7/8's numpy-only solvers). Levenberg-Marquardt is used instead of plain
gradient descent because the swing-up problem is genuinely nonlinear and
stiff enough (see the conditioning discussion below) that plain gradient
descent converges far too slowly or diverges.
"""
import numpy as np

np.seterr(all="ignore")  # occasional overflow on rejected LM trial steps

OMEGA2 = 1.0  # pendulum natural frequency squared


def f_cont(x, u):
    theta, thetadot = x
    return np.array([thetadot, -OMEGA2 * np.sin(theta) + u])


def rk4_step(x, u, tau):
    k1 = f_cont(x, u)
    k2 = f_cont(x + 0.5 * tau * k1, u)
    k3 = f_cont(x + 0.5 * tau * k2, u)
    k4 = f_cont(x + tau * k3, u)
    return x + (tau / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(x0, u, tau):
    """Forward-simulate the h+1 states under controls u=(u_0,...,u_{h-1})."""
    h = len(u)
    xs = np.zeros((h + 1, 2))
    xs[0] = x0
    for k in range(h):
        xs[k + 1] = rk4_step(xs[k], u[k], tau)
    return xs


def step_jacobians(x, u, tau, eps=1e-6):
    """Finite-difference Jacobians A=dF/dx, B=dF/du of one rk4_step."""
    A = np.zeros((2, 2))
    for i in range(2):
        dx = np.zeros(2)
        dx[i] = eps
        A[:, i] = (rk4_step(x + dx, u, tau) - rk4_step(x - dx, u, tau)) / (2 * eps)
    B = (rk4_step(x, u + eps, tau) - rk4_step(x, u - eps, tau)) / (2 * eps)
    return A, B


def sensitivity_final_state(x0, u, tau):
    """(xs, dxdu): xs are the h+1 states, dxdu is the (2, h) Jacobian of the
    final state x_h w.r.t. u_0,...,u_{h-1}."""
    h = len(u)
    xs = simulate(x0, u, tau)
    dxdu = np.zeros((2, h))
    for k in range(h):
        A, B = step_jacobians(xs[k], u[k], tau)
        dxdu = A @ dxdu
        dxdu[:, k] += B
    return xs, dxdu


def simulate_segment(z, u_seg, tau):
    """One multiple-shooting segment starting at z under controls u_seg:
    returns the end state plus its Jacobians w.r.t. z (2x2) and w.r.t.
    u_seg (2 x m) -- sensitivities only ever propagate across this one
    segment, which is the whole point of multiple shooting."""
    m = len(u_seg)
    x = np.array(z, dtype=float)
    dxdz = np.eye(2)
    dxdu = np.zeros((2, m))
    for k in range(m):
        A, B = step_jacobians(x, u_seg[k], tau)
        dxdz = A @ dxdz
        dxdu = A @ dxdu
        dxdu[:, k] += B
        x = rk4_step(x, u_seg[k], tau)
    return x, dxdz, dxdu


def _levenberg_marquardt(residual_fn, v0, iters=80, lam0=1e-2):
    """Generic damped Gauss-Newton loop shared by both solvers below.
    residual_fn(v) -> (R, J) with R the stacked residual and J its
    Jacobian; returns the converged v."""
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


def single_shooting_solve(x0, xtarget, h, tau, rho=300.0, iters=80):
    """Penalized single shooting: minimize tau*sum(u_k^2) +
    rho*||x_h(u)-xtarget||^2 over the controls alone, the states having
    been eliminated by forward simulation. Returns (u, xs, effort_cost)."""

    def residual(u):
        xs, dxdu = sensitivity_final_state(x0, u, tau)
        R = np.concatenate([np.sqrt(tau) * u, np.sqrt(rho) * (xs[-1] - xtarget)])
        J = np.vstack([np.sqrt(tau) * np.eye(h), np.sqrt(rho) * dxdu])
        return R, J

    u = _levenberg_marquardt(residual, np.zeros(h), iters=iters)
    xs = simulate(x0, u, tau)
    return u, xs, tau * np.sum(u**2)


def multiple_shooting_solve(x0, xtarget, h, S, tau, rho_defect=300.0,
                             rho_term=300.0, iters=80):
    """Multiple shooting: decision vector v=[u (h); z_1,...,z_{S-1} (2
    each)], z_0:=x0 fixed, h divided into S equal segments of m=h/S steps.
    Residuals: effort + inter-segment continuity ("defect") constraints +
    terminal error, all penalized and solved by the same
    Levenberg-Marquardt loop as single shooting -- but now the Jacobian is
    block-structured, each segment's controls only affecting that
    segment's own defect (see simulate_segment). Returns
    (u, xs, segment_starts, effort_cost)."""
    assert h % S == 0
    m = h // S
    nu, nz = h, 2 * (S - 1)
    n = nu + nz

    def unpack(v):
        u = v[:nu]
        z = v[nu:].reshape(S - 1, 2)
        zs = [x0] + [z[i] for i in range(S - 1)]
        return u, zs

    def residual(v):
        u, zs = unpack(v)
        ends, dedz, dedu = [], [], []
        for i in range(S):
            end, dz, du = simulate_segment(zs[i], u[i * m:(i + 1) * m], tau)
            ends.append(end); dedz.append(dz); dedu.append(du)

        res_blocks = [np.sqrt(tau) * u]
        rows = [np.hstack([np.sqrt(tau) * np.eye(nu), np.zeros((nu, nz))])]
        for i in range(S - 1):
            res_blocks.append(np.sqrt(rho_defect) * (ends[i] - zs[i + 1]))
            Jrow = np.zeros((2, n))
            Jrow[:, i * m:(i + 1) * m] = np.sqrt(rho_defect) * dedu[i]
            if i > 0:
                Jrow[:, nu + (i - 1) * 2: nu + i * 2] = np.sqrt(rho_defect) * dedz[i]
            Jrow[:, nu + i * 2: nu + (i + 1) * 2] += -np.sqrt(rho_defect) * np.eye(2)
            rows.append(Jrow)
        res_blocks.append(np.sqrt(rho_term) * (ends[S - 1] - xtarget))
        Jrow = np.zeros((2, n))
        Jrow[:, (S - 1) * m:S * m] = np.sqrt(rho_term) * dedu[S - 1]
        if S - 1 > 0:
            Jrow[:, nu + (S - 2) * 2: nu + (S - 1) * 2] = np.sqrt(rho_term) * dedz[S - 1]
        rows.append(Jrow)
        return np.concatenate(res_blocks), np.vstack(rows)

    v0 = np.zeros(n)
    for i in range(S - 1):
        v0[nu + 2 * i: nu + 2 * i + 2] = x0 + (i + 1) / S * (xtarget - x0)
    v = _levenberg_marquardt(residual, v0, iters=iters)

    u, zs = unpack(v)
    xs = [zs[0]]
    for i in range(S):
        x = zs[i]
        for k in range(m):
            x = rk4_step(x, u[i * m + k], tau)
            xs.append(x)
    return u, np.array(xs), zs, tau * np.sum(u**2)


def sensitivity_condition_number(x0, u, tau):
    """Condition number (sigma_max/sigma_min) of the single-shooting
    sensitivity Jacobian dx_h/du at the given control sequence -- the
    diagnostic behind the "single shooting gets ill-conditioned over long
    horizons" slide."""
    _, dxdu = sensitivity_final_state(x0, u, tau)
    svs = np.linalg.svd(dxdu, compute_uv=False)
    return svs[0] / svs[-1]


if __name__ == "__main__":
    x0 = np.array([0.0, 0.0])
    xtarget = np.array([np.pi, 0.0])
    T, h, S = 6.0, 60, 6
    tau = T / h

    print(f"Single shooting: pendulum swing-up (h={h}, tau={tau:.3f})")
    u_ss, xs_ss, cost_ss = single_shooting_solve(x0, xtarget, h, tau)
    print(f"  final state: theta={xs_ss[-1,0]:.4f} (target pi={np.pi:.4f}), "
          f"thetadot={xs_ss[-1,1]:.4f}")
    print(f"  effort cost tau*sum(u^2) = {cost_ss:.4f}")

    print()
    print(f"Multiple shooting: same problem, S={S} segments")
    u_ms, xs_ms, zs_ms, cost_ms = multiple_shooting_solve(x0, xtarget, h, S, tau)
    print(f"  final state: theta={xs_ms[-1,0]:.4f}, thetadot={xs_ms[-1,1]:.4f}")
    print(f"  effort cost tau*sum(u^2) = {cost_ms:.4f}")

    print()
    print("Conditioning: sigma_max/sigma_min of the sensitivity Jacobian, "
          "single vs. multiple shooting (per segment), as the horizon grows")
    for T_ in [6, 12, 18, 24]:
        h_ = int(10 * T_)
        tau_ = T_ / h_
        u_, xs_, cost_ = single_shooting_solve(x0, xtarget, h_, tau_)
        cond_single = sensitivity_condition_number(x0, u_, tau_)
        u2_, xs2_, zs2_, cost2_ = multiple_shooting_solve(x0, xtarget, h_, S, tau_)
        m_ = h_ // S
        _, _, dedu = simulate_segment(zs2_[0], u2_[:m_], tau_)
        svs = np.linalg.svd(dedu, compute_uv=False)
        cond_multi = svs[0] / svs[-1]
        print(f"  T={T_:4.0f}: single cond={cond_single:8.2f}   "
              f"multi (per segment) cond={cond_multi:6.2f}")
