# Unit 13 code

Companion code for Unit 13 (Policy Gradient and Actor-Critic Methods). Two
notebooks over a shared module, ported from the two scripts in
[DuarteA/ActorCriticExample](https://github.com/DuarteA/ActorCriticExample) --
the repository the deck links to on slide 19.

| deck | source script | notebook |
|---|---|---|
| slide 11 (REINFORCE) | `reinforce_mountaincar.py` | `A_13_MountainCarREINFORCE.ipynb` |
| slides 17--19 (actor-critic) | `actor_critic_mountaincar.py` | `A_13_MountainCarActorCritic.ipynb` |

The agents live in `policy_gradient.py`, the same arrangement as
`q_learning.py` (Unit 10), `policy_iteration.py` (Unit 11) and `deep_rl.py`
(Unit 12). A local copy of the repository is in
`Code/VScodeexample/actorcriticexample2/`.

## Running them

Like Unit 12's, these need PyTorch. Unlike Unit 12's, they need **no Box2D** --
`MountainCarContinuous-v0` is in Gymnasium's classic-control set:

```
pip install torch "gymnasium[classic_control]"
```

Both default to the script's `EPISODES = 1000`. REINFORCE takes about
20 minutes at that setting and actor-critic considerably longer, because it
takes a gradient step per *time step* and episodes run to the 999-step limit.
200--300 episodes already show everything described below.

**Convention warning**, as in Units 11 and 12: Gymnasium hands out *rewards to
be maximised* while the course minimises a *cost*. The deck's descent on a
cost and the code's descent on the loss $-\log\pi\cdot R$ are the same thing
with $g=-R$.

## Why this environment, and what actually happens

`MountainCarContinuous-v0` has a continuous action, which is the reason it
belongs to this unit rather than Unit 12: a Q-factor method would have to
minimise over a continuous set at every step, while a policy gradient method
simply outputs the action.

It also has a reward of $-0.1u^2$ per step plus $+100$ on reaching the flag.
That shape produces a trap, and both notebooks fall into it:

**A policy that does nothing scores about $0$, and a policy that burns fuel
without reaching the goal scores about $-50$.** So until the car has reached
the flag *at least once*, the gradient has no evidence that acting is ever
worthwhile — it only sees fuel being spent. The policy improves by shrinking
its actions towards zero, and since the Gaussian's standard deviation is
itself a learned parameter, exploration shuts down with it.

Measured here, 200 episodes each, seed 0:

| | first 20 episodes | last 50 | best | past the flag | final policy std |
|:--|--:|--:|--:|--:|--:|
| REINFORCE | $-23.08$ | $-6.03$ | $-0.01$ | $0$ | $0.0057$ |
| actor-critic | $-1.45$ | $-0.89$ | $-0.00$ | $0$ | $0.0011$ |

The last column is the whole story. The Gaussian policy starts with a standard
deviation near $1.0$; both runs end with it essentially at zero, so the policy
has stopped exploring and can never discover the goal.

**And the actor-critic gets there faster.** REINFORCE takes 29 episodes to
stop acting; actor-critic has effectively stopped after **one**. That is not a
failure of the critic — it is the critic working. Lower-variance gradient
estimates make the method commit to the direction it is being pushed, and here
that direction leads to the local optimum. Variance reduction is about the
*estimator*, not about the *objective*.

This is not a defect in the port. It is the exploration failure that policy
gradient methods are prone to and value-based methods are not: there is no
$\epsilon$-greedy floor here as there was in Units 11 and 12, and the only
exploration available is the width of the policy's own output distribution —
which the objective is actively pushing to zero.

The REINFORCE notebook includes an optional rerun with the standard deviation
floored at $0.5$, which is the cheapest of the standard remedies. Whether it
rescues the run depends on the seed, which is itself the lesson.

## What the notebooks add to the scripts

**The variance claim is measured, not asserted.** The unit says a critic
reduces the variance of the policy gradient. The actor-critic notebook
collects, along the same trajectories under one fixed policy, the two
quantities that multiply $\nabla_\theta\log\pi$ — the Monte-Carlo return $G_k$
for REINFORCE and the TD error $\delta_k$ for actor-critic — and compares
their spreads directly, with histograms. The structural reason is that $G_k$
accumulates every reward to the end of the episode while $\delta_k$ involves
one reward and two critic evaluations, so the spread of the first grows with
the horizon and the second does not.

Measured over the runs above, the within-episode spread of the weight is

| | spread of the weight on $\nabla_\theta\log\pi$, last 20 episodes |
|:--|--:|
| REINFORCE, raw return $G_k$ | $0.278$ |
| actor-critic, TD error $\delta_k$ | $0.003$ |

a factor of about **109**. (The notebook repeats this properly, on the same
trajectories under one fixed policy, since by the end of training the two
agents are following different policies and some of that gap is a difference
in behaviour rather than in estimator.)

Two caveats are stated with it: $\delta_k$ is biased where $G_k$ is not
(the Unit 11 trade, reappearing), and $\delta_k$'s spread is only small once
the critic is any good.

**The return normalisation is named for what it is.** The REINFORCE script
normalises the returns, $(G_k-\bar G)/\mathrm{std}(G)$, before forming the
loss. Subtracting the mean is exactly the **baseline subtraction** of the
deck's slide 13 — it leaves the gradient unchanged in expectation because
$\sum_u\nabla_\theta\pi=\nabla_\theta(1)=0$. So the script's "REINFORCE" is
already REINFORCE *with a baseline*, a crude constant one, and the actor-critic
comparison is really constant baseline against learned state-dependent
baseline. Dividing by the standard deviation is *not* a baseline and does
change the update, by rescaling the step size per episode.
`cfg.normalise_returns` turns both off.

**What the critic does not fix is said plainly.** A critic reduces the
variance of the gradient estimate; it does not change what the gradient points
at. The trap above is in the objective, and the remedies are the ones on the
deck's variants slide — an entropy bonus to keep the policy wide (Soft
Actor-Critic), or a different exploration mechanism altogether.

## Notes on the ports

- The two scripts duplicate `ActorNetwork`/`PolicyNetwork` almost exactly;
  they are one `GaussianPolicy` class here.
- The scripts are driven by `input()` menus under `if __name__ == "__main__"`,
  which cannot run in a notebook; the entry points are plain function calls.
- `actor_critic_mountaincar.py` also implements an **episodic** actor-critic
  (Monte-Carlo advantage instead of the TD error). It is not ported — it is
  the Unit 11 Monte-Carlo-versus-TD choice reappearing inside the critic, and
  is worth a third notebook if you want it.
- The critic's update is a **semi-gradient** step in exactly the Unit 12
  sense: the TD target is detached, so the parameters appear in the target and
  are not differentiated.
- `tqdm` is used by the scripts for progress bars; the notebooks print
  periodic summaries instead, so it is not a dependency.
