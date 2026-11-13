"""Discrete Bayes filter (prediction + correction) for a finite-state Markov
chain observed through a noisy sensor, as introduced in Unit 4.

State transition matrix P and sensor/output matrix R follow the convention
used throughout the lecture and exercises:

    P[i, j] = Prob[x_{k+1} = i | x_k = j]      (n x n, columns sum to 1)
    R[i, j] = Prob[y_k = i | x_k = j]           (m x n, columns sum to 1)

where states and measurements are 0-indexed integers 0..n-1, 0..m-1.
"""

import numpy as np


def correct(p_bar, y, R):
    """Correction (measurement update) step: p = D(y) p_bar / (1^T D(y) p_bar).

    D(y) is the diagonal matrix with the y-th row of R on the diagonal, i.e.
    Prob[y | x=i] for each state i.
    """
    d = R[y, :]
    q = d * p_bar
    total = q.sum()
    if total == 0:
        raise ValueError(f"measurement y={y} has zero likelihood under p_bar={p_bar}")
    return q / total


def predict(p, P):
    """Prediction (time update) step: p_bar_{k+1} = P p_k."""
    return P @ p


def bayes_filter(p0_prior, y0, ys, P, R):
    """Run the full Bayes filter given an initial prior and a sequence of
    measurements y0, y1, ..., y_{T-1} (ys holds y1 onward; y0 is the first
    measurement, used to correct the prior before any prediction step).

    Returns a list [p_0, p_1, ..., p_T] of posterior belief vectors.
    """
    p = correct(p0_prior, y0, R)
    beliefs = [p]
    for y in ys:
        p_bar = predict(p, P)
        p = correct(p_bar, y, R)
        beliefs.append(p)
    return beliefs


if __name__ == "__main__":
    # Smoke test: reproduces Problem 4.1 (angle estimation of a spinning
    # wheel with a noisy hue sensor). States 0,1,2,3 <-> angles 0, pi/2, pi,
    # 3pi/2; measurements 0,1,2,3 <-> green, orange, red, blue.
    P = np.array([
        [0.2, 0.0, 0.2, 0.6],
        [0.6, 0.2, 0.0, 0.2],
        [0.2, 0.6, 0.2, 0.0],
        [0.0, 0.2, 0.6, 0.2],
    ])
    R = np.array([
        [0.8, 0.1, 0.0, 0.1],
        [0.1, 0.8, 0.1, 0.0],
        [0.0, 0.1, 0.8, 0.1],
        [0.1, 0.0, 0.1, 0.8],
    ])
    p_tilde0 = np.array([0.25, 0.25, 0.25, 0.25])
    # y0=green(0), y1=orange(1), y2=orange(1), y3=blue(3), y4=blue(3)
    measurements = [0, 1, 1, 3, 3]
    beliefs = bayes_filter(p_tilde0, measurements[0], measurements[1:], P, R)
    names = ["0", "pi/2", "pi", "3pi/2"]
    for k, p in enumerate(beliefs):
        print(f"p_{k} =", np.round(p, 4))

    expected_p1 = np.array([0.0478, 0.9043, 0.0478, 0.0])
    expected_p4 = np.array([0.1962, 0.0, 0.0140, 0.7899])
    assert np.allclose(beliefs[1], expected_p1, atol=1e-3), beliefs[1]
    assert np.allclose(beliefs[4], expected_p4, atol=1e-3), beliefs[4]
    print("Matches Problem 4.1 solution (p_1 and p_4 checked).")
