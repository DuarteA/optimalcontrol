# Unit 1 code

Companion code for Unit 1 (Modeling and the dynamic programming algorithm),
extracted from `JupyterNotebooks/A_1_dp.ipynb` and
`JupyterNotebooks/A_1_inventorycontrol.ipynb`.

- **`dp.py`** -- general-purpose implementation of the dynamic programming
  algorithm (`dptv`) for any finite-horizon problem with discrete state and
  action spaces, using the `(M, C, T)` data format described in the notes'
  "A more general formalism" section. Run directly (`python dp.py`) to see
  it solve the worked example from the notes.
- **`inventory_control.py`** -- the inventory control example from the
  notes, built on top of `dp.py`. Run directly (`python inventory_control.py`)
  to reproduce the same numbers used in the notes (including the
  open-loop/closed-loop appendix figures).
- **`A_1_A_Viterbi.ipynb`** -- dynamic programming (Viterbi) algorithm for
  estimating the angle of a spinning wheel from noisy camera hue
  measurements, i.e. most-likely-state-sequence estimation for a hidden
  Markov chain (`A_Viterbi.mlx`). Two fixes were applied to this copy
  relative to the original in `JupyterNotebooks/`: (i) the uniqueness
  check in `viterbiangleestimation` indexed `u[k][x[k]]` while `x` is
  1-based, which raised `IndexError` whenever the optimal path visited
  the last angle index `N` -- corrected to `u[k][x[k]-1]`, matching the
  transition line right below it; (ii) two markdown sentences still
  described `viterbiangleestimation` as a "MATLAB function ... given
  below" although it is Python and appears above -- reworded. Known
  remaining issue, left as-is: when every candidate path has infinite
  cost (easy to trigger with point-mass `beta`/`gamma` and a measurement
  inconsistent with the dynamics), the function returns an arbitrary path
  instead of reporting infeasibility.
- **`A_1_dp.ipynb`**, **`A_1_inventorycontrol.ipynb`** -- the original
  annotated Jupyter notebooks these scripts were extracted from, with the
  full mathematical write-up and figures. Kept here as-is for students who
  want the complete, explained version rather than the bare `.py` files.

These are real, runnable files (require only `numpy`), meant to be
downloaded and run locally (or in Jupyter) -- they are too
package/computation-heavy for the in-browser Pyodide widgets used for the
short snippets embedded directly in `unit1-notes.qmd`.
