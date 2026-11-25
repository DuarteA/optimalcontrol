# Unit 10 code

Companion code for Unit 10 (Reinforcement Learning: Infinite-Horizon
Problems and Q-Learning). Unlike Units 1-9, whose notebooks came from the
`JupyterNotebooks/` folder, these are **Python translations of the MATLAB
live scripts** `AssetSellingValueIteration.mlx`,
`stoppingtimeQlearning.mlx`, `stochinventorycontroQlearning.mlx`,
`statediscQlearning.mlx` and `RobotPlanningQLearning.mlx`, made on
2026-10-21. They keep the `A_10_` prefix used for every other unit.

- **`q_learning.py`** -- the two Q-learning drivers that the live scripts
  repeat inside each file, collected here so the notebooks show the
  problem rather than four copies of the algorithm. `q_learning_ssp` keeps
  stationary Q-factors `Q[i][u]` and runs each simulation to a terminal
  state; `q_learning_finite_horizon` keeps stage-indexed Q-factors
  `Q[k][i][u]` and runs each simulation for a fixed number of stages, with
  the terminal cost as the target at the last one. `sample_dist` is the
  live scripts' `sampledist`. Costs are minimised throughout, matching the
  unit.

## The notebooks

| Notebook | What it does |
|---|---|
| `A_10_AssetSellingValueIteration.ipynb` | Value iteration for the asset selling problem; reproduces the two figures in the unit. |
| `A_10_stoppingtimeQLearning.ipynb` | The same problem **without** the model, twice: as a scalar stochastic approximation, and as Q-learning under an arbitrary exploration policy. |
| `A_10_StochInventoryControlQLearning.ipynb` | Unit 2's stochastic inventory problem by Q-learning, compared against exact DP. |
| `A_10_statediscQLearning.ipynb` | Unit 7's minimum-energy double integrator on a discretized grid, by Q-learning. |
| `A_10_RobotPlanningQLearning.ipynb` | Units 2-3's noisy grid path planning by Q-learning, compared against exact stochastic DP. |

Each needs only `numpy` and `matplotlib`, and each runs in well under a
minute.

## Verified against exact solutions

Every notebook was run and checked against a model-based answer computed
independently, rather than being trusted because it looked plausible:

- **Asset selling.** Converges in 63 iterations to a wait value of
  $16\,857.9$ and a threshold of $16\,900$ euros. **Note for the slides
  and notes:** the figure in the lecture plots the first *ten* iterates,
  and it is the tenth whose threshold is "around $16\,700$". The
  converged threshold is $16\,900$. Both numbers are printed by the
  notebook so the distinction is visible.
- **Stopping time.** Stochastic approximation gives $16\,893.5$ and
  Q-learning $16\,879.4$, against the exact fixed point $16\,857.9$; both
  recover the $16\,900$ threshold. Hundreds of thousands of samples to
  pin down one number -- the data inefficiency of the unit, made concrete.
- **Inventory control.** Matches exact DP to within a few hundredths at
  stage 0 and exactly at stage 3, but leaves **five states at zero**: high
  stock at stages 1 and 2 is never visited, so nothing is ever learned
  there. This is the convergence theorem's first condition ("every
  state-action pair infinitely often") failing in plain sight, and the
  notebook says so.
- **State discretization.** Recovers Unit 7's grid DP answer *exactly*:
  $u=[4,-4]$, positions $[0,0.5,1]$, cost $16$. The problem is
  deterministic, so the step size $\gamma=1$ of the original script makes
  each update exact.
- **Robot planning.** Within a few tenths of exact stochastic DP across
  the grid (max error $2.3$, in rarely-visited cells), and agrees on the
  best starting row.

## Two quirks of the original scripts, preserved

Both are documented in `q_learning.py` and can be switched off there.

1. **The exploration rate does not decay.** The scripts compute
   `eps/(nsim/1000)` once `isim >= 1000` -- dividing by `nsim`, the
   *total* number of simulations, not `isim`, the current one. So it drops
   once to a constant `eps*1000/nsim`, which at the default
   `nsim = 300000` is `eps/300`. This is almost certainly meant to be
   `isim`; pass `decay_with_isim=True` to get that. It is the direct cause
   of the unvisited states in the inventory notebook.
2. **The step size is constant.** Unit 10's convergence theorem asks for
   $\sum\gamma^t=\infty$ and $\sum(\gamma^t)^2<\infty$, which a constant
   step size fails. In practice a constant step is what is used, and the
   estimates then fluctuate around the optimum instead of converging to
   it -- visible in the Q-trace plotted by the stopping-time notebook.

## One translation fix

With `alpha = 1` the MATLAB update `(1-alpha)*Q + alpha*target` evaluates
`0*Inf` whenever a Q-factor is infinite, giving `NaN`. MATLAB's `min` and
`[~,i] = min(...)` silently skip `NaN`, so the script limps on; numpy's
propagate it, and the direct translation returns `NaN` everywhere. The
`alpha == 1` case is therefore written as a plain assignment (which is
what it means), and the drivers use `nanmin`/`nanargmin` to match MATLAB's
selection semantics.
