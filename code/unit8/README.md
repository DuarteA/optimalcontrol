# Unit 8 code

Companion code for Unit 8 (Approximate Dynamic Programming: MPC, Rollout,
and Switched Systems), extracted from and linking to the notebooks in
`JupyterNotebooks/` (there filed under the `B_5_` prefix -- note that
Lecture8.pdf itself internally mislabels this material "Lecture 9", but
the file name and the user's own numbering make it Unit 8; see the
lecture qmd's header comment), by the same naming convention used for
Units 1-7.

- **`mpc.py`** -- finite-horizon model predictive control with optional
  box input constraints: `prediction_matrices` (builds the batch
  $X=S_x x_0+S_u u$ prediction), `mpc_solve` (solves the resulting QP by
  projected gradient descent -- no external QP solver required, in the
  same spirit as Unit 7's `gradient_method.py`), `mpc_control` (returns
  just the receding-horizon $u_0$), and `simulate_mpc` (closed-loop
  rollout). Run directly (`python mpc.py`) to reproduce
  `A_8_LQinputconstraints.ipynb`'s double-integrator setup: the
  unconstrained case matches Unit 5's Riccati-based LQR gain exactly,
  and the constrained case visibly saturates at the input bound.
- **`rollout.py`** -- dynamic programming, base-policy cost evaluation,
  and rollout for switched linear systems: `dp_cost_matrices` (exact
  backward DP, representing the cost-to-go as a growing list of
  candidate quadratic-form matrices), `base_policy_cost` (solves the
  discrete Lyapunov equation for a fixed mode's infinite-horizon cost),
  `rollout_policy` (the $H=1$ lookahead policy), and `simulate_switched`.
  Run directly (`python rollout.py`) to reproduce Exercise 8.1's $Z_1,
  \ldots,Z_4$ matrices and Exercise 8.3's stabilizing rollout policy
  ($\bar P=\begin{bmatrix}3&2\\2&4\end{bmatrix}$, trajectory norm decaying
  to $0$).
- **`A_8_SLS.ipynb`** -- the central live script for this unit: switched
  linear systems, approximate dynamic programming, and the resource-aware
  control application, matching most of the lecture directly (`SLS.mlx`).
- **`A_8_LQinputconstraints.ipynb`** -- LQR with input constraints on a
  double integrator, the worked example behind `mpc.py`
  (`LQinputconstraints.mlx`).
- **`A_8_A_MPC_quad_tracking.ipynb`** -- MPC trajectory tracking for a
  linearized quadcopter (longitudinal channel) (`MPC_quad_tracking.mlx`).
- **`A_8_K_MPCRollout.ipynb`** -- MPC and rollout for a switched linear
  system with random disturbances (`MPCRollout.mlx`).

All files require only `numpy` (and `matplotlib`/`scipy` for the notebook
demos).
