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

- **`vehicle.py`** -- the two closing case studies, on a different system:
  an electric vehicle modelled longitudinally as a point mass with drag,
  rolling resistance and road grade, with an *asymmetric* battery
  (`eta_drive=0.90` delivering, `eta_regen=0.65` recovering).
  `eco_solve` is the energy-optimal problem -- cover a 2000 m route with a
  12 m hill in a fixed 120 s using as little battery energy as possible --
  by trapezoidal or Hermite-Simpson collocation. `eco_constant_speed` is
  the cruise-control baseline it is measured against, and `eco_audit`
  re-simulates any solution through the true dynamics to score it honestly
  (the objective an NLP reports is computed on its own grid, by its own
  rule, and on a coarse grid it can be a good number attached to a
  trajectory that never finishes the route). `sprint_collocation` and
  `sprint_multiple_shooting` are the time-optimal problem -- 400 m from
  rest to rest, flat out -- with the horizon itself a decision variable,
  handled by normalizing time to [0,1]. `time_optimal_reference` solves the
  same problem the *indirect* way: Pontryagin's minimum principle makes the
  answer bang-bang with a single switch, so one root find on the switching
  time gives a reference the direct methods can be scored against. Run
  directly (`python vehicle.py`) to reproduce every number on the closing
  slides.

  TWO THINGS ABOUT THIS MODULE DIFFER FROM THE THREE ABOVE, and both are
  deliberate. First, it needs **scipy** (SLSQP and brentq), not just numpy:
  battery energy is not a sum of squares and can be negative, the trip time
  is a decision variable, and both problems carry genuine inequality
  constraints, so the hand-rolled Levenberg-Marquardt used for the pendulum
  does not apply. Second, every transcription works in **non-dimensional**
  variables (s/L, v/V_MAX, F/F_MAX). In SI units the decision variables of
  these problems span four orders of magnitude and SLSQP fails outright --
  it exits on its iteration limit with the dynamics still violated. Badly
  scaled transcriptions are one of the commonest reasons a trajectory
  optimization "does not work".
- **`make_case_study_figures.py`** -- runs everything in `vehicle.py`,
  prints the full comparison tables, writes `results.json`, and regenerates
  the two case-study figures (`ev_eco_profile.png`, `ev_time_optimal.png`).

## Notebooks

Four notebooks, one per section of the lecture, written 2026-10-26. Each
imports the matching module above rather than repeating its code, so the
notebooks and the slides run exactly the same solvers. See
`notebooks.qmd` for the landing page. They are stored without output.

- `A_9_ShootingPendulum.ipynb` -- single and multiple shooting; the
  conditioning diagnostic; and the experiment showing that the two
  solvers return *different local optima* only because they start from
  different guesses (warm-starting single shooting from the
  multiple-shooting solution moves it onto the better one, 2.2371 against
  2.3117).
- `A_9_CollocationPendulum.ipynb` -- trapezoidal and Hermite-Simpson; the
  grid-refinement sweep; and an audit that simulates each solution's
  control through the true dynamics. See the caveat below, which this
  notebook is built around.
- `A_9_ILQRPendulum.ipynb` -- the backward/forward passes, the
  convergence history, the local optimum, and the bridges to Units 5, 8
  and 10.
- `A_9_EVCaseStudies.ipynb` -- both closing case studies.

## A caveat about the penalty weight, found 2026-10-26

Writing the notebooks surfaced three related things that the slides do
not yet say. Nothing in the August code or in the committed figures has
been changed; the notebooks describe the situation honestly instead.

1. **The backward "dip" is not an iLQR quirk.** `single_shooting_solve`
   finds the same local optimum, to the same cost (2.3117). On
   `trajectory_comparison.png` the two curves lie exactly on top of one
   another, which is why only one dipping curve is visible. It is an
   initial-guess effect: `single_shooting_solve` and `ilqr_solve` both
   start from u = 0, while `multiple_shooting_solve` and the collocation
   solvers additionally seed a state trajectory from x(0) to x(T).
2. **`collocation.py`'s default rho = 300 is loose.** The defects are
   penalized, not enforced, so the optimizer can buy a lower effort by
   violating the dynamics. Simulating the resulting control open-loop
   leaves the pendulum at theta = 1.38 rather than pi, and the reported
   effort of 2.0300 is the cost of a trajectory that does not swing up.
   At rho = 3000 the max defect falls from 3.5e-3 to 3.6e-4 and the
   reported effort rises to 2.2177 -- which is where
   `multiple_shooting_solve` independently lands (2.2194).
3. **`collocation_accuracy.png` is therefore not a convergence study.**
   The curve falls steadily instead of levelling off, and the two schemes
   are indistinguishable, because at fixed rho the error floor is set by
   the penalty weight rather than by the order of the quadrature. Redone
   at rho = 3000 the sweep settles towards 2.22 as it should. (Order does
   show through in the EV case study, where the defects reach SLSQP as
   hard equality constraints and Hermite-Simpson at N = 20 matches
   trapezoidal at N = 80.)

If the defaults are raised, `collocation_accuracy.png` and the effort
numbers quoted on the collocation slides need regenerating with them.

Figures used in the lecture (`lectures/figures/lecture9/`) were generated
by a one-off script that calls into all three modules; see the header
comment in `lectures/lecture09-numerical-methods.qmd` for what each
figure shows.

`shooting.py`, `collocation.py` and `ilqr.py` require only `numpy`;
`vehicle.py` additionally requires `scipy`, and regenerating the figures
also needs `matplotlib`.
