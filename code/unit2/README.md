# Unit 2 code

Companion code for Unit 2 (Stochastic dynamic programming), extracted from
and linking to the notebooks in `JupyterNotebooks/`. All four `A_2_*`
notebooks (and `BuzwireChallenge.ipynb`) are the Python versions of the
MATLAB LiveScripts already referenced in `lectures/lecture02-stochastic-dp.qmd`
("Additional live scripts" section and Appendix A).

- **`stochastic_dp.py`** -- general-purpose stochastic dynamic programming
  solver (`stochdp`), the stochastic generalization of `unit1/dp.py`: each
  (state, action) pair can lead to several possible next states, each with
  its own cost and probability, and the algorithm minimizes the *expected*
  cost-to-go. Run directly (`python stochastic_dp.py`) for a minimal smoke
  test.
- **`stoch_inventory_control.py`** -- the inventory control example with
  uncertain demand from the notes / lecture02 Appendix A, built on top of
  `stochastic_dp.py`. Run directly (`python stoch_inventory_control.py`)
  to reproduce the same numbers shown there (h=4, N=4, stochastic demand
  at stage 2).
- **`A_2_StochInventoryControl.ipynb`** -- the original annotated notebook
  `stochastic_dp.py`/`stoch_inventory_control.py` were extracted from.
- **`A_2_A_Stochastic_DP_policy_pacman.ipynb`** -- simplified Pac-Man
  pursuit-evasion game solved with stochastic DP (the "simplified pac-man
  game" live script). Visualization-heavy (matplotlib + `clear_output`
  animation); kept as a notebook rather than extracted to a bare script.
- **`A_2_K_TempControl.ipynb`** -- thermostat/temperature control with
  stochastic DP (the "simple temperature control model" live script).
- **`A_2_RobotPlanningStochastic.ipynb`** -- the Unit 1 robot grid-planning
  example revisited with a stochastic disturbance, comparing deterministic
  vs. stochastic DP policies (the "RobotPlanningStochastic" live script).
- **`BuzwireChallenge.ipynb`** -- the buzz-wire game example (the
  "Buzz wire" live script).

Only the inventory control example was extracted to clean, tested `.py`
files: it is the one directly worked out in the notes, and the general
`stochdp` solver it's built on is reusable for other problems too (see
[[feedback-pyodide-live-code-snippets]] convention: linked real files for
anything package-heavy or long, not inline Pyodide snippets). The other
three notebooks are lecture demos with matplotlib animation/visualization
that are more naturally explored interactively in Jupyter than as
standalone scripts -- kept as-is, with the full explained version and
figures, for students who want to run or adapt them.

All files require only `numpy` (and `matplotlib` for the notebook demos).
