"""Zero-order-hold discretization of a continuous-time linear system, and
exact (grid) dynamic programming over discretized state/action spaces, as
introduced in Unit 7.

Convention (matches the lecture slides): continuous-time model
    xdot(t) = A x(t) + B u(t),   u(t) = u_k for t in [k*tau, (k+1)*tau)
discretized model, x_k := x(k*tau),
    x_{k+1} = Ad x_k + Bd u_k
where
    Ad = e^{A*tau},   Bd = int_0^tau e^{A*s} ds  B.
Both are obtained in one shot from the augmented-matrix identity
    expm([[A, B], [0, 0]] * tau) = [[Ad, Bd], [0, I]].
"""
import numpy as np


def matrix_exp(M, terms=200):
    """Matrix exponential via a truncated power series (numpy-only, no
    scipy.linalg.expm dependency -- 200 terms is far more than enough for
    the well-scaled matrices used in this course)."""
    R = np.eye(M.shape[0])
    term = np.eye(M.shape[0])
    for k in range(1, terms):
        term = term @ M / k
        R = R + term
    return R


def discretize(A, B, tau):
    """Zero-order-hold discretization with sampling period tau. Returns (Ad, Bd)."""
    A = np.atleast_2d(A).astype(float)
    B = np.atleast_2d(B).astype(float)
    if B.shape[0] != A.shape[0]:
        B = B.reshape(A.shape[0], -1)
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A * tau
    M[:n, n:] = B * tau
    Md = matrix_exp(M)
    return Md[:n, :n], Md[:n, n:]


def dp_grid(X, U, dynamics, stage_cost, terminal_cost, h):
    """Exact backward dynamic programming over finite state/action grids.

    X, U: iterables of admissible state/action values (states reached by
        applying an action need not lie in X -- only the states for which
        a policy is requested must).
    dynamics(x, u) -> x_next
    stage_cost(x, u) -> float
    terminal_cost(x) -> float
    h: horizon.

    Returns (policy, value): policy[k][x] = argmin action at stage k,
    value[k][x] = optimal cost-to-go, for k in {0,...,h} and x in X. Ties
    are broken by keeping the first minimizer found (in the order U is
    iterated); the calling code can re-scan for ties if that matters, as
    in Unit 7, Problem 7.2.
    """
    # value function is looked up on-the-fly for any x reachable from X
    # under any u in U, so we memoize lazily rather than precomputing a
    # fixed grid of reachable states.
    memo_value = {h: {}}
    memo_policy = {}

    def V(k, x):
        if k == h:
            if x not in memo_value[h]:
                memo_value[h][x] = terminal_cost(x)
            return memo_value[h][x]
        if k not in memo_value:
            memo_value[k] = {}
            memo_policy[k] = {}
        if x in memo_value[k]:
            return memo_value[k][x]
        best_cost, best_u = None, None
        for u in U:
            xn = dynamics(x, u)
            c = stage_cost(x, u) + V(k + 1, xn)
            if best_cost is None or c < best_cost - 1e-12:
                best_cost, best_u = c, u
        memo_value[k][x] = best_cost
        memo_policy[k][x] = best_u
        return best_cost

    for x in X:
        V(0, x)
    # also populate stages 1..h-1 for the requested grid X (states at later
    # stages that a policy is wanted for)
    for k in range(1, h):
        for x in X:
            V(k, x)
    return memo_policy, memo_value


if __name__ == "__main__":
    print("Discretizing the five (A, B, tau) triples from Problem 7.3:")
    cases = [
        (np.array([[1.0]]), np.array([[2.0]]), 0.1),
        (np.array([[1.0, 6.0], [0.0, 0.0]]), np.array([[5.0], [7.0]]), 0.05),
        (np.array([[0.0, -4.0], [4.0, 0.0]]), np.array([[0.0], [1.0]]), 0.2),
        (np.array([[2.0, 1.0], [0.0, 2.0]]), np.array([[0.0], [1.0]]), 1.0),
        (
            np.array([[2.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([[1.0], [2.0], [1.0]]),
            0.5,
        ),
    ]
    for i, (A, B, tau) in enumerate(cases, start=1):
        Ad, Bd = discretize(A, B, tau)
        print(f"  (i={i}) Ad=\n{np.round(Ad, 4)}\n       Bd=\n{np.round(Bd, 4)}")

    print()
    print("Grid DP for Problem 7.2 (x_{k+1}=x_k+u_k, h=3, "
          "cost sum x_k^2+u_k^2+x_h^2, X={0,...,5}, U={-5,...,5}):")
    X = list(range(0, 6))
    U = list(range(-5, 6))
    h = 3
    policy, value = dp_grid(
        X, U,
        dynamics=lambda x, u: x + u,
        stage_cost=lambda x, u: x**2 + u**2,
        terminal_cost=lambda x: x**2,
        h=h,
    )
    for k in range(h):
        row = ", ".join(f"x={x}: u*={policy[k][x]}, J={value[k][x]}" for x in X)
        print(f"  stage {k}: {row}")
