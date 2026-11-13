# Unit 12 code

Companion code for Unit 12 (Approximation in Value Space: Neural Networks,
Deep Q-Learning and LSTD). Four notebooks, ported from three of Duarte's own
scripts, which the deck links to on slides 9, 11, 16 and 25:

| deck slide | source script | notebook |
|---|---|---|
| 9 | `NNplotexample.py` | `A_12_NNRegression.ipynb` |
| 11 | `compareNNBatchStochGrad.py` | `A_12_StochasticVsFullGradient.ipynb` |
| 16 | `lunardeepQSarsaSemiGradient.py` | `A_12_LunarLanderDQN.ipynb` |
| 25 | `lunardeepQSarsaSemiGradient.py` | `A_12_LunarLanderDQNvsSARSA.ipynb` |

The last two come from one script, which produces both of the deck's result
plots (`lunar_dqn_returns.png` and `lunar_compare_returns.png`). Rather than
duplicate three hundred lines of agent code across two notebooks, the agents
live in `deep_rl.py` and the notebooks are thin -- the same arrangement as
`q_learning.py` in `code/unit10/` and `policy_iteration.py` in `code/unit11/`.

## Running them

**These notebooks need PyTorch**, unlike every other unit's, which need only
`numpy` and `matplotlib`:

```
pip install torch
```

The two lunar-lander notebooks also need Gymnasium, and the lunar lander
itself needs Box2D:

```
pip install "gymnasium[box2d]"
```

If Box2D is missing, `deep_rl.make_env` falls back to `CartPole-v1`, which
ships with the base Gymnasium package and exercises exactly the same code --
the original script's own diagnostics suggest it as the fallback. Only the
numbers change, not the algorithms.

The lunar-lander notebooks default to the deck's `NUM_EPISODES = 800` and take
a few minutes each; 150-200 episodes already show the shape of the curve.

## `deep_rl.py`

The shared module: `QNetwork` (three linear layers of 128 units with ReLU
between them -- the deck's "three-layer NN"), `ReplayBuffer`, `DQNAgent`,
`SARSAAgent`, `train_dqn`, `train_sarsa`, `evaluate_agent`, and a `Config`
dataclass holding the original script's hyperparameters. `make_env` adds the
CartPole fallback described above.

The two agents differ in one line, which is the whole point of Unit 11
restated with function approximation:

```python
# DQN     -- off-policy: maximise over the next action, using the target network
next_q = self.target_q(ns).max(1)[0]
# SARSA   -- on-policy: the action the epsilon-greedy policy actually takes
q_next = self.q(ns_t)[0, na]
```

**Convention warning**, the same one as Unit 11's Gymnasium notebooks:
Gymnasium hands out *rewards to be maximised*, while Units 1-11 minimise a
*cost*. The greedy action in this module is an `argmax` where every formula in
the notes is an `argmin`.

## What each notebook was checked against

**`A_12_NNRegression.ipynb`** reproduces the slide-9 figure: the target
$\sin(2\pi x)+0.3x$ on $[-1.5,1.5]$, 300 noisy samples, a $[64,64]$ ReLU
network, Adam, 400 epochs. Final training loss around $0.0136$ against the
script's $0.0131$ -- the difference is PyTorch version, not method.

The notebook adds one line the script does not have: the **noise floor**.
The samples carry noise of standard deviation $0.08$, so no fit can beat a
mean squared error of $0.0064$ against the noisy targets. Printing it turns
the loss from a number into a diagnostic -- near the floor means the network
has extracted what there is, far above means underfitting, well below means
it is fitting noise.

**`A_12_StochasticVsFullGradient.ipynb`** settles two things about the deck's
table on slide 11.

*The epoch count is a typo.* The script computes `epochs2 = int(epochs*300)`,
which with 400 epochs is **120 000**, not the 12 000 the slide prints. The
notebook runs both: at 12 000 epochs plain gradient descent reaches a training
loss of $0.2796$, nowhere near converged, while 120 000 epochs reproduces the
slide's $0.011145$ **exactly**. The losses and the 400 epochs in that table
are right; only the 12 000 is wrong.

*The comparison varies two things at once.* The stochastic run uses **Adam**
and the full-gradient run uses **plain SGD**, so the twenty-fold timing gap is
the product of the batch size and the optimiser. The notebook separates them
by running the full gradient with Adam at an equal number of parameter
updates (1600 each). On this problem that run is about as fast as the
stochastic one and reaches a *lower* training loss -- $0.0058$ against
$0.0130$ -- which is below the noise floor, i.e. it has started fitting the
noise.

So with 255 training points the full gradient is not the expensive option,
and the deck's gap should not be read as the cost of the full gradient. The
argument for stochastic gradient is about large $N$: one full-batch step costs
$O(N)$ and one minibatch step does not. The lunar-lander notebooks are the
realistic case, with a replay buffer of 50 000 transitions.

**`A_12_LunarLanderDQN.ipynb`** trains deep Q-learning for 800 episodes and
reproduces the slide-16 plot, raw returns with a moving average on top. It
then runs a short **ablation**, which the script does not: the same agent with
the replay buffer shrunk to one minibatch, and with the target network copied
every step. Both devices exist to stop the regression target from moving in
step with the estimate being fitted, and the ablation is the quickest way to
see that this is what they are for. The notebook also saves `dqn_policy.gif`,
the animation the deck links to.

**`A_12_LunarLanderDQNvsSARSA.ipynb`** trains both agents and reproduces the
slide-25 plot. The deck's claim is that they perform similarly, and the
notebook gives the reason rather than just the picture: with $\epsilon$
decaying to $0.02$, the action SARSA bootstraps on *is* the maximising action
$98.5\%$ of the time, so the two targets agree on all but one step in seventy.
Raising the exploration floor to $0.2$ drops that to $85\%$ and the two
separate -- the lunar-lander version of Unit 11's frozen-lake observation.

It also records why SARSA has no replay buffer, which is not a simplification:
a stored transition carries the action an *older* policy chose, and SARSA's
target needs the action the *current* policy would take. That is Problem
11.4's "the data table is one column short", restated as a property of the
algorithm.

## Notes on the ports

- The agents are faithful to the script. Two changes: `soft_update` was
  unused and is dropped, and `SARSAAgent.update` built its target with
  `torch.tensor(...)` around an existing tensor, which warns on modern
  PyTorch -- rewritten to build the scalar target directly.
- The script's `main` trains DQN and SARSA in sequence in one process; the
  notebooks do the same but let each be run on its own.
- Numbers will not match the deck exactly. Both lunar agents are seeded, but
  the result depends on the PyTorch version and on CPU threading, and the
  environment is stochastic. The *shape* is what reproduces.
