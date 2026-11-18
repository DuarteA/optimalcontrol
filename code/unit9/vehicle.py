"""Two hard numerical optimal control problems for an electric vehicle, used
as the closing examples of Unit 9.

Both use the same longitudinal vehicle model -- a point mass with
aerodynamic drag, rolling resistance and a road grade -- and differ only in
what is asked of it:

  A. ENERGY-OPTIMAL ("eco-driving").  Cover a fixed route of length L in a
     fixed trip time T, entering and leaving at the average speed, using as
     little battery energy as possible.  The route contains a hill.  Fixed
     horizon, smooth solution, inequality constraints on speed and force.

  B. TIME-OPTIMAL.  Cover the flat opening stretch of the same route from
     rest to rest as quickly as possible, with the same force limit.  FREE
     horizon, and a solution that is bang-bang -- the optimal force jumps
     from +F_MAX to -F_MAX at a single switching instant.

The two stress a numerical method in opposite ways, which is the point of
having both.  Problem A has a smooth optimal trajectory and a cost that is
hard to guess; the methods agree closely and the interest is in the answer.
Problem B has a trivially guessable answer and a NON-SMOOTH one; the
methods disagree, and `time_optimal_reference` below computes the truth
they can be scored against.

A NOTE ON THE SOLVER, which is itself one of the lessons here.  The other
three modules of this unit (shooting.py, collocation.py, ilqr.py) are
deliberately numpy-only: they roll their own Levenberg-Marquardt on a
penalized least-squares residual, so that none of the mechanics is hidden.
That works there because the pendulum's cost is a sum of squares and its
only constraints are equalities.  Neither is true here -- battery energy is
not a sum of squares and can be negative (regeneration), the trip time is
itself a decision variable, and both problems carry genuine inequality
constraints on speed and force.  So this module uses scipy's SLSQP, a real
sequential-quadratic-programming solver.  Every transcription below hands
SLSQP the same problem data and differs only in HOW the dynamics are
discretized, so the comparison is a comparison of transcriptions and not of
solvers.

A NOTE ON SCALING, which is the other lesson.  Written in SI units the
decision variables of these problems span four orders of magnitude --
position ~2000 m, speed ~17 m/s, force ~1500 N -- and SLSQP simply fails to
converge (it exits on its iteration limit with the dynamics still violated).
Every transcription here therefore works in NON-DIMENSIONAL variables,
s/L, v/V_MAX, F/F_MAX, with the defects scaled to match.  This is not a
detail of the implementation; badly scaled transcriptions are one of the
commonest reasons a trajectory optimization "does not work".

Run directly (`python vehicle.py`) to reproduce every number quoted on the
Unit 9 slides.

Requires numpy and scipy.
"""
import numpy as np
from scipy.optimize import minimize, brentq

# --------------------------------------------------------------------------
# Vehicle and route
# --------------------------------------------------------------------------
MASS = 1600.0        # kg
GRAV = 9.81          # m/s^2
C_ROLL = 0.010       # rolling resistance coefficient
RHO_AIR = 1.2        # kg/m^3
CDA = 0.60           # drag area Cd*A, m^2
F_MAX = 3000.0       # N, traction and braking force limit
V_MAX = 30.0         # m/s, speed limit (108 km/h)
V_MIN = 1.0          # m/s, keeps the vehicle moving forwards (problem A)

ETA_DRIVE = 0.90     # battery -> wheels efficiency
ETA_REGEN = 0.65     # wheels -> battery efficiency (asymmetric, on purpose)
P_SMOOTH = 100.0     # W, smoothing width of the max(.,0) in the power split

ROUTE_LENGTH = 2000.0          # m, the eco-driving route
TRIP_TIME = 120.0              # s, the time the trip must take
CRUISE_SPEED = ROUTE_LENGTH / TRIP_TIME   # m/s, the average, used at both ends

HILL_START = 600.0       # m
HILL_LENGTH = 800.0      # m
HILL_HEIGHT = 12.0       # m, up and back down: net zero elevation change

SPRINT_LENGTH = 400.0    # m, the flat opening stretch, for problem B


def elevation(s):
    """Road elevation (m).  One smooth hill, flat before and after, so that
    the route's net elevation change is zero -- any energy the hill costs is
    lost in the battery round trip, not stored."""
    z = np.clip((s - HILL_START) / HILL_LENGTH, 0.0, 1.0)
    return np.where((z > 0.0) & (z < 1.0),
                    0.5 * HILL_HEIGHT * (1.0 - np.cos(2 * np.pi * z)), 0.0)


def grade(s):
    """Road gradient dz/ds.  Small-angle, so sin(alpha) = grade."""
    z = np.clip((s - HILL_START) / HILL_LENGTH, 0.0, 1.0)
    return np.where((z > 0.0) & (z < 1.0),
                    np.pi * HILL_HEIGHT / HILL_LENGTH * np.sin(2 * np.pi * z),
                    0.0)


def resistance(v, s):
    """Total resisting force (N): drag + rolling + grade."""
    return (0.5 * RHO_AIR * CDA * v * np.abs(v)
            + MASS * GRAV * C_ROLL
            + MASS * GRAV * grade(s))


def f_cont(x, F, flat=False):
    """Continuous-time dynamics in SI units.  x = [s, v], input F (N).

    `flat=True` removes the hill, which is the road problem B runs on."""
    s, v = x[..., 0], x[..., 1]
    g = np.zeros_like(v) if flat else MASS * GRAV * grade(s)
    acc = (F - 0.5 * RHO_AIR * CDA * v * np.abs(v)
           - MASS * GRAV * C_ROLL - g) / MASS
    return np.stack([v, acc], axis=-1)


# --------------------------------------------------------------------------
# Battery power, with a smooth split between driving and regenerating
# --------------------------------------------------------------------------
def _softmax0(z):
    """Smooth max(z, 0); the error is O(P_SMOOTH^2/|z|) away from z = 0."""
    return 0.5 * (z + np.sqrt(z * z + P_SMOOTH * P_SMOOTH))


def battery_power(F, v):
    """Power drawn from the battery (W).  Mechanical power at the wheels is
    P = F v; delivering it costs P/eta_d, recovering it returns eta_r |P|.
    The asymmetry eta_d * eta_r < 1 is what makes the hill cost anything."""
    P = F * v
    return _softmax0(P) / ETA_DRIVE - ETA_REGEN * _softmax0(-P)


# --------------------------------------------------------------------------
# Non-dimensional dynamics (see the module docstring on scaling)
# --------------------------------------------------------------------------
def f_scaled(xh, Fh, s_scale, flat=False):
    """Dynamics in the scaled variables sh = s/s_scale, vh = v/V_MAX,
    Fh = F/F_MAX, with time still in seconds."""
    x = np.stack([xh[..., 0] * s_scale, xh[..., 1] * V_MAX], axis=-1)
    f = f_cont(x, Fh * F_MAX, flat=flat)
    return np.stack([f[..., 0] / s_scale, f[..., 1] / V_MAX], axis=-1)


def _defects(xh, Fh, tau, s_scale, scheme, flat=False):
    """Collocation defects in the scaled variables."""
    fk = f_scaled(xh, Fh, s_scale, flat)
    if scheme == "trapezoidal":
        return (xh[1:] - xh[:-1] - (tau / 2) * (fk[1:] + fk[:-1])).ravel()
    if scheme == "hermite-simpson":
        xm = 0.5 * (xh[:-1] + xh[1:]) + (tau / 8) * (fk[:-1] - fk[1:])
        Fm = 0.5 * (Fh[:-1] + Fh[1:])
        fm = f_scaled(xm, Fm, s_scale, flat)
        return (xh[1:] - xh[:-1] - (tau / 6) * (fk[:-1] + 4 * fm + fk[1:])).ravel()
    raise ValueError(scheme)


def _energy(xh, Fh, tau, s_scale, scheme):
    """Battery energy (J) of a scaled trajectory, integrated with the
    quadrature rule that MATCHES the collocation scheme.

    This matching is not cosmetic.  A Hermite-Simpson defect is 4th-order
    accurate, but if the cost is still integrated by the trapezoidal rule
    then the objective is only 2nd-order accurate and the scheme's extra
    accuracy is thrown away -- the solver converges quickly to the wrong
    number.  Integrating with Simpson's rule on the same cubic-Hermite
    midpoint the defect already uses restores it."""
    v = xh[..., 1] * V_MAX
    P = battery_power(Fh * F_MAX, v)
    if scheme == "trapezoidal":
        return float(np.sum(_trapezoid_weights(len(P), tau) * P))
    if scheme == "hermite-simpson":
        fk = f_scaled(xh, Fh, s_scale)
        xm = 0.5 * (xh[:-1] + xh[1:]) + (tau / 8) * (fk[:-1] - fk[1:])
        Fm = 0.5 * (Fh[:-1] + Fh[1:])
        Pm = battery_power(Fm * F_MAX, xm[:, 1] * V_MAX)
        return float(np.sum((tau / 6) * (P[:-1] + 4 * Pm + P[1:])))
    raise ValueError(scheme)


def _central_jac(fun, z, m, eps=1e-6):
    """Central-difference Jacobian.  Accurate to ~1e-10 here because every
    decision variable has been scaled to O(1)."""
    J = np.zeros((m, len(z)))
    for j in range(len(z)):
        zp = z.copy(); zp[j] += eps
        zm = z.copy(); zm[j] -= eps
        J[:, j] = (fun(zp) - fun(zm)) / (2 * eps)
    return J


def _rk4(x, F, tau, flat=False):
    k1 = f_cont(x, F, flat)
    k2 = f_cont(x + 0.5 * tau * k1, F, flat)
    k3 = f_cont(x + 0.5 * tau * k2, F, flat)
    k4 = f_cont(x + tau * k3, F, flat)
    return x + (tau / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _trapezoid_weights(n_nodes, tau):
    w = np.full(n_nodes, tau)
    w[0] = w[-1] = tau / 2
    return w


# ==========================================================================
# Problem A -- energy-optimal speed profile over the route
# ==========================================================================
def eco_solve(N=60, scheme="trapezoidal", maxiter=300, ftol=1e-8):
    """Minimum-battery-energy speed profile, by direct collocation.

    Decision variables (all scaled): s/L and v/V_MAX at the N+1 nodes, and
    F/F_MAX at the N+1 nodes.  The horizon is fixed at TRIP_TIME."""
    tau = TRIP_TIME / N
    n = N + 1
    iS, iV, iF = slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n)
    E_SCALE = 1e6     # report and optimize energy in MJ, so the cost is O(1)

    def unpack(z):
        return np.stack([z[iS], z[iV]], axis=-1), z[iF]

    def cost(z):
        xh, Fh = unpack(z)
        return _energy(xh, Fh, tau, ROUTE_LENGTH, scheme) / E_SCALE

    def defects(z):
        xh, Fh = unpack(z)
        return _defects(xh, Fh, tau, ROUTE_LENGTH, scheme)

    def boundary(z):
        xh, Fh = unpack(z)
        return np.array([xh[0, 0],
                         xh[0, 1] - CRUISE_SPEED / V_MAX,
                         xh[-1, 0] - 1.0,
                         xh[-1, 1] - CRUISE_SPEED / V_MAX])

    # initial guess: the constant-speed trip, which is dynamically feasible
    t = np.linspace(0.0, TRIP_TIME, n)
    s0 = ROUTE_LENGTH * t / TRIP_TIME
    z0 = np.zeros(3 * n)
    z0[iS] = s0 / ROUTE_LENGTH
    z0[iV] = CRUISE_SPEED / V_MAX
    z0[iF] = resistance(CRUISE_SPEED, s0) / F_MAX

    bounds = ([(None, None)] * n
              + [(V_MIN / V_MAX, 1.0)] * n
              + [(-1.0, 1.0)] * n)
    cons = [{"type": "eq", "fun": defects,
             "jac": lambda z: _central_jac(defects, z, 2 * N)},
            {"type": "eq", "fun": boundary,
             "jac": lambda z: _central_jac(boundary, z, 4)}]
    res = minimize(cost, z0, jac=lambda z: _central_jac(
                       lambda y: np.array([cost(y)]), z, 1)[0],
                   bounds=bounds, constraints=cons, method="SLSQP",
                   options={"maxiter": maxiter, "ftol": ftol})

    xh, Fh = unpack(res.x)
    return {"t": t, "s": xh[:, 0] * ROUTE_LENGTH, "v": xh[:, 1] * V_MAX,
            "F": Fh * F_MAX, "N": N, "scheme": scheme,
            "energy": cost(res.x) * E_SCALE,
            "defect_norm": float(np.max(np.abs(defects(res.x)))),
            "boundary_norm": float(np.max(np.abs(boundary(res.x)))),
            "status": res.status, "nit": res.nit, "res": res}


def eco_audit(sol, dt=1e-3):
    """Score a solved speed profile HONESTLY, independently of the scheme
    that produced it.

    The number an NLP reports is its own discretization's *estimate* of the
    cost, evaluated on its own grid.  It is not the cost of the trajectory
    the vehicle would actually follow.  This function takes only the force
    schedule, interpolates it linearly in time, integrates the true dynamics
    with a fine fixed-step RK4, and accumulates the battery energy on that
    fine grid.  It also reports how far the resulting trajectory misses the
    required endpoint.

    Comparing transcriptions on these numbers rather than on their own
    objective values is the only fair comparison, and the gap between the
    two is itself worth looking at."""
    t_nodes, F_nodes = sol["t"], sol["F"]
    n = int(round(TRIP_TIME / dt))
    h = TRIP_TIME / n
    x = np.array([0.0, sol["v"][0]])
    E = 0.0
    P_prev = battery_power(np.interp(0.0, t_nodes, F_nodes), x[1])
    for k in range(n):
        t = k * h
        F = np.interp(t, t_nodes, F_nodes)
        x = _rk4(x, F, h)
        F_next = np.interp(t + h, t_nodes, F_nodes)
        P = battery_power(F_next, x[1])
        E += 0.5 * h * (P_prev + P)      # trapezoid on a very fine grid
        P_prev = P
    return {"energy": float(E),
            "miss_s": float(x[0] - ROUTE_LENGTH),
            "miss_v": float(x[1] - CRUISE_SPEED)}


def eco_constant_speed(N=4000):
    """Battery energy of the baseline: hold the average speed the whole way.

    This is a genuine trajectory of the model -- the force simply cancels
    the resistance at every instant, and the peak force it needs stays well
    inside F_MAX -- so it is the honest 'drive it normally' comparison."""
    t = np.linspace(0.0, TRIP_TIME, N + 1)
    s = ROUTE_LENGTH * t / TRIP_TIME
    v = np.full_like(t, CRUISE_SPEED)
    F = resistance(v, s)
    w = _trapezoid_weights(N + 1, TRIP_TIME / N)
    return {"t": t, "s": s, "v": v, "F": F,
            "energy": float(np.sum(w * battery_power(F, v))),
            "peak_force": float(np.max(np.abs(F)))}


# ==========================================================================
# Problem B -- time-optimal sprint over the flat opening stretch
# ==========================================================================
def sprint_collocation(N=40, scheme="trapezoidal", maxiter=300, T0=25.0,
                       ftol=1e-12):
    """Minimum-time sprint by direct collocation, with the horizon itself a
    decision variable.  The grid is normalized to sigma in [0,1], so the
    step tau = T/N multiplies every defect and T enters the dynamics."""
    n = N + 1
    iS, iV, iF, iT = slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n), 3 * n

    def unpack(z):
        return np.stack([z[iS], z[iV]], axis=-1), z[iF], z[iT] * T0

    def cost(z):
        return float(z[iT])

    def cost_jac(z):
        g = np.zeros(3 * n + 1); g[iT] = 1.0
        return g

    def defects(z):
        xh, Fh, T = unpack(z)
        return _defects(xh, Fh, T / N, SPRINT_LENGTH, scheme, flat=True)

    def boundary(z):
        xh, Fh, T = unpack(z)
        return np.array([xh[0, 0], xh[0, 1], xh[-1, 0] - 1.0, xh[-1, 1]])

    sigma = np.linspace(0.0, 1.0, n)
    z0 = np.zeros(3 * n + 1)
    z0[iS] = sigma
    z0[iV] = 4 * (SPRINT_LENGTH / T0 / V_MAX) * sigma * (1 - sigma)
    z0[iF] = np.where(sigma < 0.5, 1.0, -1.0)
    z0[iT] = 1.0

    bounds = ([(None, None)] * n
              + [(0.0, 1.0)] * n
              + [(-1.0, 1.0)] * n
              + [(1.0 / T0, 200.0 / T0)])
    cons = [{"type": "eq", "fun": defects,
             "jac": lambda z: _central_jac(defects, z, 2 * N)},
            {"type": "eq", "fun": boundary,
             "jac": lambda z: _central_jac(boundary, z, 4)}]
    res = minimize(cost, z0, jac=cost_jac, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"maxiter": maxiter, "ftol": ftol})

    xh, Fh, T = unpack(res.x)
    return {"T": T, "sigma": sigma, "t": T * sigma,
            "s": xh[:, 0] * SPRINT_LENGTH, "v": xh[:, 1] * V_MAX,
            "F": Fh * F_MAX, "N": N, "scheme": scheme,
            "defect_norm": float(np.max(np.abs(defects(res.x)))),
            "status": res.status, "nit": res.nit, "res": res}


def sprint_multiple_shooting(N=40, S=8, maxiter=300, T0=25.0, ftol=1e-12):
    """Minimum-time sprint by multiple shooting: S segments, each integrated
    by RK4 with N/S steps, tied together by continuity constraints on the
    interior states.  Same scaling as the collocation version."""
    assert N % S == 0, "N must be a multiple of S"
    per = N // S
    nZ, nF = 2 * (S - 1), N
    iT = nZ + nF

    def unpack(z):
        return z[:nZ].reshape(S - 1, 2), z[nZ:nZ + nF], z[iT] * T0

    def shoot(z):
        zint, Fh, T = unpack(z)
        tau = T / N
        starts = [np.zeros(2)] + [zint[i] for i in range(S - 1)]
        ends, traj = [], []
        for i in range(S):
            xh = starts[i]
            seg = [xh]
            for k in range(per):
                x = np.array([xh[0] * SPRINT_LENGTH, xh[1] * V_MAX])
                x = _rk4(x, Fh[i * per + k] * F_MAX, tau, flat=True)
                xh = np.array([x[0] / SPRINT_LENGTH, x[1] / V_MAX])
                seg.append(xh)
            ends.append(xh)
            traj.append(np.array(seg))
        return ends, traj

    def cost(z):
        return float(z[iT])

    def cost_jac(z):
        g = np.zeros(iT + 1); g[iT] = 1.0
        return g

    def con(z):
        zint, Fh, T = unpack(z)
        ends, _ = shoot(z)
        c = [ends[i] - zint[i] for i in range(S - 1)]
        c.append(ends[-1] - np.array([1.0, 0.0]))
        return np.concatenate(c)

    sigma = np.linspace(0.0, 1.0, N + 1)
    z0 = np.zeros(iT + 1)
    for i in range(S - 1):
        frac = (i + 1) / S
        z0[2 * i] = frac
        z0[2 * i + 1] = 4 * (SPRINT_LENGTH / T0 / V_MAX) * frac * (1 - frac)
    z0[nZ:nZ + nF] = np.where(sigma[:-1] < 0.5, 1.0, -1.0)
    z0[iT] = 1.0

    bounds = ([(None, None), (0.0, 1.0)] * (S - 1)
              + [(-1.0, 1.0)] * nF
              + [(1.0 / T0, 200.0 / T0)])
    cons = [{"type": "eq", "fun": con,
             "jac": lambda z: _central_jac(con, z, 2 * S)}]
    res = minimize(cost, z0, jac=cost_jac, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"maxiter": maxiter, "ftol": ftol})

    zint, Fh, T = unpack(res.x)
    _, traj = shoot(res.x)
    xs = np.vstack([seg[:-1] for seg in traj] + [traj[-1][-1:]])
    return {"T": T, "s": xs[:, 0] * SPRINT_LENGTH, "v": xs[:, 1] * V_MAX,
            "F": Fh * F_MAX, "N": N, "S": S,
            "con_norm": float(np.max(np.abs(con(res.x)))),
            "status": res.status, "nit": res.nit, "res": res}


def _sim_bang(t_switch, dt):
    """Simulate full acceleration for t_switch seconds, then full braking
    until the vehicle stops.  Returns (T, trajectory)."""
    x = np.zeros(2)
    ts, xs, Fs = [0.0], [x.copy()], []
    n1 = max(int(np.ceil(t_switch / dt)), 1)
    h1 = t_switch / n1
    t = 0.0
    for _ in range(n1):
        x = _rk4(x, F_MAX, h1, flat=True)
        t += h1
        ts.append(t); xs.append(x.copy()); Fs.append(F_MAX)
    while x[1] > 0.0:
        xn = _rk4(x, -F_MAX, dt, flat=True)
        if xn[1] <= 0.0:                      # land exactly on v = 0
            lo, hi = 0.0, dt
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if _rk4(x, -F_MAX, mid, flat=True)[1] > 0.0:
                    lo = mid
                else:
                    hi = mid
            x = _rk4(x, -F_MAX, hi, flat=True)
            t += hi
            ts.append(t); xs.append(x.copy()); Fs.append(-F_MAX)
            break
        x = xn
        t += dt
        ts.append(t); xs.append(x.copy()); Fs.append(-F_MAX)
    xs = np.array(xs)
    return t, {"t": np.array(ts), "s": xs[:, 0], "v": xs[:, 1],
               "F": np.array(Fs)}


def time_optimal_reference(dt=5e-3):
    """Near-exact minimum time, by exploiting the KNOWN structure of the
    solution instead of discretizing the whole problem.

    Pontryagin's minimum principle says that for a minimum-time problem
    whose cost does not depend on the input, with |F| <= F_MAX, the optimal
    force is bang-bang.  Here the switching function is strictly monotone,
    so there is exactly one switch and no singular arc: accelerate flat out,
    then brake flat out.  That collapses the infinite-dimensional problem to
    ONE unknown -- the switching time -- found below by a bracketed root
    find on the distance covered.  The remaining error is the integrator's.

    This is the indirect family of Unit 9's overview earning its keep: when
    the structure of the answer is known, almost nothing has to be
    discretized.  It also gives the yardstick that the direct methods in
    this module are scored against.

    Returns (T, t_switch, trajectory)."""
    def miss(t_switch, h):
        _, tr = _sim_bang(t_switch, h)
        return tr["s"][-1] - SPRINT_LENGTH

    t_switch = brentq(miss, 1.0, 30.0, args=(dt,), xtol=1e-12, rtol=1e-14)
    T, tr = _sim_bang(t_switch, dt)
    return T, t_switch, tr


# ==========================================================================
# Reporting
# ==========================================================================
def report_eco():
    print("=" * 76)
    print("A.  ENERGY-OPTIMAL SPEED PROFILE ('eco-driving')")
    print("=" * 76)
    base = eco_constant_speed()
    print(f"  route {ROUTE_LENGTH:.0f} m in {TRIP_TIME:.0f} s "
          f"(average {CRUISE_SPEED:.2f} m/s = {3.6*CRUISE_SPEED:.0f} km/h); "
          f"hill {HILL_HEIGHT:.0f} m over {HILL_LENGTH:.0f} m")
    print(f"  efficiencies: eta_drive = {ETA_DRIVE}, eta_regen = {ETA_REGEN} "
          f"(round trip {ETA_DRIVE*ETA_REGEN:.3f})")
    print()
    print(f"  baseline, constant speed throughout : "
          f"{base['energy']/1e6:7.4f} MJ   (peak |F| {base['peak_force']:.0f} N)")
    out = {}
    for scheme in ("trapezoidal", "hermite-simpson"):
        r = eco_solve(N=60, scheme=scheme)
        out[scheme] = r
        saving = 100.0 * (base["energy"] - r["energy"]) / base["energy"]
        print(f"  {scheme:20s} N = 60        : {r['energy']/1e6:7.4f} MJ   "
              f"({saving:4.1f}% saved)  v in [{r['v'].min():5.2f}, "
              f"{r['v'].max():5.2f}] m/s   defect {r['defect_norm']:.1e}  "
              f"[status {r['status']}, {r['nit']} it]")
    r = out["hermite-simpson"]
    print()
    print("  the optimal profile, read off at four points on the route:")
    for label, s_at in (("start (flat)", 0.0),
                        ("foot of the climb", HILL_START),
                        ("crest", HILL_START + HILL_LENGTH / 2),
                        ("foot of the descent", HILL_START + HILL_LENGTH)):
        print(f"      {label:22s} s = {s_at:6.0f} m : "
              f"v = {np.interp(s_at, r['s'], r['v']):5.2f} m/s")
    return base, out


def report_sprint():
    print()
    print("=" * 76)
    print("B.  TIME-OPTIMAL SPRINT OVER THE FLAT OPENING STRETCH")
    print("=" * 76)
    T_ref, t_sw, ref = time_optimal_reference()
    print(f"  {SPRINT_LENGTH:.0f} m from rest to rest, |F| <= {F_MAX:.0f} N, "
          f"flat road")
    print(f"  reference (bang-bang structure + one root find):")
    print(f"      T* = {T_ref:.6f} s,  switch at t = {t_sw:.6f} s "
          f"({100*t_sw/T_ref:.1f}% of the way),  peak speed "
          f"{ref['v'].max():.3f} m/s")
    print()
    print(f"  {'method':26s} {'N':>4s} {'T [s]':>10s} {'T - T* [s]':>12s} "
          f"{'defect':>9s}")
    rows = []
    for scheme in ("trapezoidal", "hermite-simpson"):
        for N in (10, 20, 40, 80):
            r = sprint_collocation(N=N, scheme=scheme)
            rows.append((scheme, N, r["T"], r["T"] - T_ref))
            print(f"  {scheme:26s} {N:4d} {r['T']:10.5f} "
                  f"{r['T']-T_ref:+12.5f} {r['defect_norm']:9.1e}")
    r = sprint_multiple_shooting(N=40, S=8)
    rows.append(("multiple shooting (S=8)", 40, r["T"], r["T"] - T_ref))
    print(f"  {'multiple shooting (S=8)':26s} {40:4d} {r['T']:10.5f} "
          f"{r['T']-T_ref:+12.5f} {r['con_norm']:9.1e}")
    return T_ref, t_sw, ref, rows


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    report_eco()
    report_sprint()
