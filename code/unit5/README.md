# Unit 5 code

Companion code for Unit 5 (Linear Quadratic Regulator), extracted from and
linking to the notebooks in `JupyterNotebooks/` (there filed under the
`B_1_` prefix), by the same naming convention used for Units 1-4.

- **`riccati.py`** -- the core recursion of the lecture: `finite_horizon_lqr`
  (backward Riccati recursion for a time-varying/finite-horizon problem,
  returns the gains $K_k$ and cost-to-go matrices $P_k$), `infinite_horizon_lqr`
  (iterates the recursion to convergence, i.e. solves the algebraic Riccati
  equation), and `simulate` (propagates the closed-loop trajectory). Run
  directly (`python riccati.py`) to reproduce the double-integrator example
  from the lecture exactly: $K_0$, $P_0$, the optimal path, and the
  infinite-horizon gain all match the slide numbers.
- **`A_5_LQfinhorizondoubleintegrator.ipynb`** -- the double-integrator
  finite-horizon LQR example worked through in the lecture (`LQfinhorizondoubleintegrator.mlx`).
- **`A_5_Riccatiequations.ipynb`** -- the double-integrator infinite-horizon
  example, showing $P_k \to \bar P$ and $K_k \to \bar K$ (`Riccatiequations.mlx`).
- **`A_5_K_LQRinvertedpendulum.ipynb`** -- LQR applied to the linearized
  inverted-pendulum-on-a-cart model (`LQRinvertedpendulum.mlx`).
- **`A_5_N_referencetrackinglqr.ipynb`**, **`A_5_K_Tracking_lqr.ipynb`**,
  **`A_5_A_LQR_Smoothing.ipynb`** -- the three "additional live scripts"
  on reference tracking (as opposed to the stabilization problem the
  lecture focuses on): `referencetrackingLQR.mlx`, `TrackingLQR.mlx`,
  `LQRSmoothing.mlx`.
- **`A_5_K_PositioningHeadDisk.ipynb`** -- LQR applied to a disk-drive
  read/write head positioning problem (`LQRPositioningHeadDisk.mlx`).

All files require only `numpy` (and `matplotlib`/`scipy` for the notebook
demos).
