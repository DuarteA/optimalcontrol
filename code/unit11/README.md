# Unit 11 code

Companion code for Unit 11 (Policy Iteration, Simulation-Based Policy
Evaluation and SARSA). Seven notebooks, from two sources.

**Five ports of the MATLAB live scripts** (supplied 2026-10-25), sharing one
module, `policy_iteration.py`:

| live script | notebook |
|---|---|
| `stochinventorycontroPI.mlx` | `A_11_StochInventoryControlPI.ipynb` |
| `statediscPI.mlx` | `A_11_statediscPI.ipynb` |
| `stoppingtimeSimPI.mlx` | `A_11_stoppingtimeSimPI.ipynb` |
| `stochinventorycontroSimBasedPI.mlx` | `A_11_StochInventoryControlSimBasedPI.ipynb` |
| `RobotPlanningSimulationBasedPI.mlx` | `A_11_RobotPlanningSimulationBasedPI.ipynb` |

**Two ports of the Python scripts** (supplied 2026-10-22) -- the code behind
the result plots on slides 33 and 43:

- `frozenlakeQlearningSarsa.py` -> `A_11_FrozenLakeQLearningSARSA.ipynb`
- `blackjackOnOffPolicy.py`     -> `A_11_BlackjackOnOffPolicy.ipynb`

## Running them

The five live-script ports need only `numpy` and `matplotlib`, like every
other unit's notebooks. The two Gymnasium ones use `FrozenLake-v1` and
`Blackjack-v1` exactly as the originals do, but each also carries a ~25-line
stand-in for the environment it needs, used automatically if `gymnasium` is
not installed, so they run anywhere too. Both paths were tested and agree.

## `policy_iteration.py`

The shared module, so the same loop serves all five live-script ports rather
than being retyped in each:

- `policy_evaluation(mdp, mu)` -- backward sweep with the action fixed by
  `mu`; no minimisation anywhere.
- `policy_improvement(mdp, J)` -- one greedy sweep, which is exactly a
  rollout step with `mu` as the base policy.
- `policy_iteration(mdp)` -> `(mu, J, history)`, where `history` is the
  `(mu, J)` pair after each improvement step. That history is what makes the
  slide sequence reproducible.
- `dp(M, C, Jh, h)` -- exact backward DP for the discretization example, so
  policy iteration can be compared against it in the same notebook. Ties
  break to the *last* minimiser, matching MATLAB's `min`.
- `sim_based_policy_iteration(...)` -- the simulation-based loop, with the
  three differences between the two original scripts exposed as arguments
  (`terminal_cost`, `greedy_behaviour`, `eps_decay_at`) instead of forked
  into two copies.
- `sample_dist(prob, rng)` -- the port of `sampledist.m`.

The MDP format the exact routines take is the one the scripts use:
`DM[k][i][j]`, `CM[k][i][j]`, `PM[k][i][j]` are lists over the disturbance
outcomes of stage `k`, state `i`, action `j`; `gh[i]` is the terminal cost.

## What each notebook was checked against

**`A_11_StochInventoryControlPI.ipynb`** reproduces the lecture's lattice
sequence exactly. `J(stock 0, stage 0)` after each improvement step:
$0 \to -25.30 \to -25.70 \to -25.75 \to -25.95$, converging in five
iterations -- one lattice diagram per number. The optimal stage-0 column
$-25.95, -30.75, -36.05, -40.65, -45.45$ agrees to the last digit with two
independent calculations elsewhere in the course: the stochastic DP of Unit 2
and the exact-DP reference inside
`code/unit10/A_10_StochInventoryControlQLearning.ipynb`. Three routes, same
five numbers.

**`A_11_statediscPI.ipynb`** -- policy iteration and backward DP both give
`J[0] = 801.0`, policy `u = [4, -4]`, positions `[0, 0.5, 1]`: the same
trajectory Unit 7's DP found and the same one Unit 10's Q-learning notebook
recovered without a model.

One number needs a word of explanation. The cost here is $16.02$ while Unit
7 reports $16.00$. The terminal cost is `Jh[target] = 1` with $+\infty$
everywhere else -- the $1$ is a marker saying "this state is admissible", not
a real cost -- and it contributes $dt \cdot 1 \cdot du^2 = 0.02$. Setting it
to $0$ reproduces $16.00$ exactly. The notebook says so rather than leaving
the reader to wonder which unit is wrong.

**`A_11_stoppingtimeSimPI.ipynb`** -- because the policy is fixed, the value
of waiting satisfies a *linear* relation and has a closed form
$\theta^\star = \bar\alpha\bar\theta/(1-\bar\alpha\gamma)$. Stochastic
approximation gives $15\,152.3$ against the closed form $15\,163.4$, a
$0.07\%$ error. Worth noting that this fixed policy (threshold
$c_{25} = 12\,400$) is worth about $15\,160$, while Unit 10's *optimal*
threshold of $16\,900$ is worth $16\,858$ -- evaluating a policy and
improving it are separate operations, and this notebook does only the first.

**`A_11_StochInventoryControlSimBasedPI.ipynb`** -- the same problem as the
first notebook with the model removed, so every entry can be checked against
the exact answer. Stage 0 comes out accurate to $0.15$--$0.63$ and the last
stage is exact (its Q-factors come straight from the salvage value, with no
bootstrapping). The entries still sitting at zero are the instructive ones:
high stock at stages 1 and 2 is never visited, with errors of $31.35$,
$21.10$, $35.90$, $25.50$, $19.20$. That is the convergence theorem's first
condition -- every state-action pair infinitely often -- visibly not holding,
and it matters less than it looks, because those states are unreachable under
any good policy. Learned policy
`[[2,4,2,1],[3,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]`. Runs in about 50 s at
the original's `nsim = 300_000`, which is kept so the numbers match.

**`A_11_RobotPlanningSimulationBasedPI.ipynb`** -- the noisy grid of Units 2
and 3, solved by generalized policy iteration from simulation and compared
against the exact stochastic DP in the same notebook. See the bug below.

**`A_11_FrozenLakeQLearningSARSA.ipynb`** -- both algorithms reach a win rate
near $0.5$ on the slippery lake and settle on similar policies. Several
arrows in the learned policies point *into a wall*, which is correct:
commanding into a wall makes the intended move a no-op and leaves only the
two perpendicular slips, which is how you move sideways with no chance of
sliding into a hole.

**`A_11_BlackjackOnOffPolicy.ipynb`** -- blackjack is small enough to solve
exactly by value iteration, so every estimate is checked against the truth.
The optimal action at $(20, 10, \text{False})$ is **stick**:
$Q(\text{stick}) = +0.4350$ against $Q(\text{hit}) = -0.8547$.

## A confirmed bug in the robot script's exploration

`RobotPlanningSimulationBasedPI.mlx` explores with

```matlab
u(k) = sampledist(ones(1,nu(x(k))));
```

an **unnormalised** vector of ones. Inside `sampledist` the cumulative sum is
then `[1,2,3]` and the draw is `a = rand(1)`, always below $1$, so
`sum(cprobdist_ < a)` is always $0$ and the "random" exploratory action is
always the **first** one -- up. Exploration is not uniform over the three
actions; it is "always try up".

`stochinventorycontroSimBasedPI.mlx` does this correctly, with
`ones(1,nu(x(k)))/nu(x(k))`. The fix is one `/nu`.

The notebook measures what the bug costs rather than just describing it:

| | max error, first column | max error, grid |
|---|---|---|
| as written | **6.39** | 6.39 |
| exploration fixed | **1.89** | 1.89 |

Fixing one character more than halves the error; with the bug the middle rows
are overestimated by about six, because the robot never discovers that going
*down* is sometimes right. A good illustration of the unit's point: the
algorithm, the step size and the number of simulations are all fine here, and
the answer is still wrong because one branch of the action space was never
tried.

## A confirmed bug in the ordinary-IS estimator

The slide shows the ordinary-IS curve settling near **0.60** while on-policy
and weighted IS settle near **0.45**. Ordinary importance sampling is
unbiased, so all three should agree -- and in the notebook they do.

The cause is one index in `mc_off_policy_eval_trace_ordinary`:

```python
for k in range(t, T):       # should be range(t+1, T)
    W_t *= ratios[k]
```

For a **Q**-factor the conditioning is on having taken action `a` in state
`s`, so the weight product must start at `t+1`. Starting at `t` multiplies in
an extra factor `pi(a|s)/b(a|s)`. With `behavior_eps=0.6` and two actions,
`b(pi(s)|s) = 0.3 + 0.4 = 0.7`, so each surviving visit is inflated by
`1/0.7 = 1.43`.

Verified numerically in the notebook, 20000 episodes:

| estimator | result |
|---|---|
| exact, by value iteration | **+0.4350** |
| on-policy MC | +0.4364 |
| weighted IS | +0.4329 |
| ordinary IS, product from `t+1` | +0.4200 |
| ordinary IS, product from `t` (as written) | **+0.6076** |

That last line reproduces the slide's curve. The weighted version in the same
file does *not* have the problem, because it updates `W` **after** the
Q-update, so its product effectively starts at `t+1` -- the asymmetry between
the two functions is the tell. The notebook plots both variants against the
exact value so the difference is visible.

## Other notes on the ports

- MATLAB's `min` and `[~,i]=min()` silently skip `NaN`; numpy propagates it.
  Relevant wherever a $0\cdot\infty$ can arise, as it does with step size
  $1$ -- the module uses plain assignment when `alpha == 1` rather than
  blending, and `nanmin`/`nanargmin` elsewhere.
- Indices are 0-based throughout, so action labels and state numbers are one
  lower than the MATLAB printouts.
- The frozen-lake animations use `anim.to_jshtml()` so they play inline
  instead of opening a window; `eps_start=0.5`, the value in the second
  version supplied.
- The blackjack script's `next_player_state`, `usable_ace`, `sum_hand`,
  `is_bust` and `draw_card` helpers are unused by `main()` and were dropped.
  (`next_player_state` in particular has an unreachable branch and a
  `1 in [1]` test that is always true -- worth deleting in the original too.)
- Both Gymnasium notebooks state the convention clash explicitly: Gymnasium
  hands out **rewards to maximise**, while Units 1-10 minimise a **cost**, so
  the greedy action there is an `argmax`.
