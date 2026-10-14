# Unit 9 code

Companion code for Unit 9 (Numerical Methods in Optimal Control), new
material not sourced from an existing lecture PDF or notebook (unlike
Units 1-8). All three scripts solve variants of the same running example
-- minimum-effort swing-up of a torque-driven pendulum,
$\dot x=[\dot\theta,\,-\sin\theta+u]$, from $x(0)=[0,0]$ (hanging down) to
$x(T)=[\pi,0]$ (inverted) over a fixed horizon $T$ -- so their solutions
can be directly compared against each other (see the lecture qmd's
trajectory-comparison figure). Following the convention set in Units 7-8,
everything here is `numpy`-only: rather than a general-purpose NLP/QP
solver, the shooting and collocation scripts implement their own
Levenberg-Marquardt (damped Gauss-Newton) solver on a penalized
least-squares formulation of the equality-constrained problem, and the
iLQR script implements its own backward/forward pass.

- **`shooting.py`** -- direct single and multiple shooting.
  `single_shooting_solve` eliminates the states by forward simulation and
  optimizes over the controls alone; `multiple_shooting_solve` breaks the
  horizon into segments with their own free initial states tied together
  by continuity ("defect") constraints. Both use forward sensitivity
  propagation (`sensitivity_final_state`, `simulate_segment` -- the same
  chain-rule idea as Unit 7's `gradient_method.rollout_grad`, generalized
  to nonlinear dynamics via per-step Jacobians) to build the
  Levenberg-Marquardt Jacobian. `sensitivity_condition_number` is the
  diagnostic behind the lecture's conditioning figure: run directly
  (`python shooting.py`) to reproduce it (single shooting's sensitivity
  Jacobian becomes increasingly ill-conditioned as the horizon grows,
  multiple shooting's stays near 1 since sensitivities only ever
  propagate within one segment).
- **`collocation.py`** -- direct collocation.
  `trapezoidal_collocation_solve` and `hermite_simpson_collocation_solve`
  parametrize both states and controls at every grid node and enforce the
  dynamics only as equality constraints between consecutive nodes (a
  trapezoidal-quadrature defect and a Hermite-Simpson defect
  respectively), rather than eliminating the states by simulation as
  shooting does. Run directly (`python collocation.py`) to compare their
  accuracy as the number of intervals grows.
- **`ilqr.py`** -- iterative LQR. `ilqr_solve` iterates a backward
  Riccati-like pass (linearizing the dynamics and quadratizing the cost
  around the current nominal trajectory -- the direct nonlinear
  generalization of Unit 5's LQR backward recursion) and a forward pass
  with a backtracking line search. Run directly (`python ilqr.py`) to see
  the fast (near-quadratic) convergence typical of Newton-type methods,
  and a comparison against `collocation.py`'s solution -- note that iLQR
  here converges to a *different* local optimum (it dips backward before
  swinging up, rather than swinging up directly), a genuine illustration
  of local methods' sensitivity to their initial guess/trajectory, not a
  bug.

Figures used in the lecture (`lectures/figures/lecture9/`) were generated
by a one-off script that calls into all three modules; see the header
comment in `lectures/lecture09-numerical-methods.qmd` for what each
figure shows.

All files require only `numpy` (`matplotlib` additionally for regenerating
the figures).
