# Unit 7 code

Companion code for Unit 7 (Discretization and Static Optimization),
extracted from and linking to the notebooks in `JupyterNotebooks/` (there
filed under the `B_3_` and `B_4_` prefixes), by the same naming convention
used for Units 1-6.

- **`discretization.py`** -- zero-order-hold discretization of a
  continuous-time model: `matrix_exp` (a numpy-only truncated-series matrix
  exponential) and `discretize` (builds $A_d=e^{A\tau}$, $B_d=\int_0^\tau
  e^{As}ds\,B$ in one shot via the augmented-matrix trick), plus `dp_grid`,
  an exact backward-DP solver over finite state/action grids. Run directly
  (`python discretization.py`) to reproduce both the five $(A,B,\tau)$
  discretization cases from Exercise 7.3 and the full policy/cost table
  from Exercise 7.2 -- along the way this independently caught (and fixed)
  a rounding slip in the original solutions manual for case (iv), where
  $A_d=e^2\begin{bmatrix}1&1\\0&1\end{bmatrix}$, not the "7.384" the source
  states (the true value is $e^2=7.3891$; see the note in
  `exercises/unit7-exercises.qmd`).
- **`gradient_method.py`** -- plain gradient descent applied to the
  static-optimization view of a stage decision problem (`rollout_cost`,
  `rollout_grad`, `gradient_descent`). Run directly (`python
  gradient_method.py`) to reproduce `A_7_gradsearch.ipynb`'s example
  exactly ($u^*=[-0.6,-0.2]$, $J(u^*)=1.6$), matching the closed-form
  Riccati solution to the same problem.
- **`A_7_discexample.ipynb`** -- the scalar discretization example
  motivating the lecture, comparing an exponential vs. quadratic terminal
  cost (`discexample.mlx`).
- **`A_7_statedisc.ipynb`** -- state- and input-space discretization of a
  double-integrator positioning problem, worked as a grid DP (matches
  `discretization.py`'s `dp_grid`) (`statedisc.mlx`).
  Corrected on 2026-10-01: the port from `statedisc.mlx` had kept MATLAB's
  1-based indexing in five places while building the transition table
  `M` 0-based, so the rollout started from $(d,v)=(0,0.1)$ instead of the
  origin, applied the input one level above the one it reported, reached
  $d=1$ only by saturating at the grid edge, and ended at $v=0.1$ rather
  than $0$ -- which the 1-based read-back then hid. The terminal cost was
  also set to $1$ at the target instead of $0$, inflating the reported
  energy by $\tau\delta_u^2$. With all six corrected the notebook returns
  the energy $16.0$ and the input sequence $(4,-4)$ for its own defaults
  $\tau=0.5$, $\delta_u=0.2$, matching both `discretization.py` and the
  closed form $12K^2/(K^2-1)$ of the notes.
- **`A_7_gradsearch.ipynb`** -- the gradient-descent solution to the same
  toy problem as `A_7_discexample.ipynb`, this time via the
  static-optimization route (`gradsearch.mlx`).
- **`A_7_K_MinEnergy.ipynb`** -- minimum-energy control of a hanging crane,
  a worked application of the static-optimization machinery
  (`MinEnergy.mlx`).
- **`A_7_K_Brachistochrone.ipynb`** -- the classical brachistochrone
  problem, included as an "additional live script" example of optimal
  control beyond the linear-quadratic setting (`Brachistochrone.mlx`).

All files require only `numpy` (and `matplotlib`/`scipy` for the notebook
demos).
