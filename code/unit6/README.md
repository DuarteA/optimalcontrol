# Unit 6 code

Companion code for Unit 6 (Kalman filter and LQG), extracted from and
linking to the notebooks in `JupyterNotebooks/` (there filed under the
`B_2_` prefix, plus one `B_1_` file that is actually Lecture 6 material --
see the note below), by the same naming convention used for Units 1-5.

- **`kalman_filter.py`** -- a general-purpose discrete-time Kalman filter:
  `predict` (time update, $\hat x_{k+1|k}=A\hat x_{k|k}+Bu_k$,
  $\Phi_{k+1|k}=A\Phi_{k|k}A^\top+W$), `correct` (measurement update with
  gain $L_k=\Phi_{k|k-1}C^\top(C\Phi_{k|k-1}C^\top+V)^{-1}$),
  `kalman_filter` (runs the full loop over a sequence of measurements),
  and `stationary_kalman_filter` (iterates the prediction-covariance
  Riccati equation to convergence, i.e. the steady-state/stationary
  filter used for average-cost LQG design). Run directly
  (`python kalman_filter.py`) to reproduce the mass-spring-system example
  from Lecture 6, Appendix C -- the printed $L_0$, $\Phi_{0|0}$,
  $\hat x_{0|0}$, $\hat x_{1|1}$ match the slide numbers exactly.
- **`A_6_LQCstochastic.ipynb`** -- the stochastic (disturbance-affected)
  linear quadratic control example from the start of the lecture
  (`LQCstochastic.mlx`).
- **`A_6_lqg.ipynb`** -- the full LQG design example (inverted pendulum
  with noisy position measurement) from the lecture (`lqg.mlx`).
- **`A_6_Kalman.ipynb`** -- the mass-spring-system Kalman filter example
  worked by hand in Appendix C, matching `kalman_filter.py`'s smoke test
  (`Kalman.mlx`).
- **`A_6_K_KalmanIMU.ipynb`** -- Kalman filtering for IMU-based angle
  estimation (`KalmanIMU.mlx`).
- **`A_6_K_SensorFusionAndLQG.ipynb`** -- combining multiple sensors via
  the Kalman filter ahead of LQG control (`SensorFusionAndLQG.mlx`).
- **`A_6_K_quadraticgame.ipynb`** -- the min-max/worst-case-disturbance
  quadratic game referenced as an "additional live script"
  (`quadraticgame.mlx`); filed as `B_1_K_game.ipynb` in the source
  `JupyterNotebooks/` folder. The `B_1_` prefix is a leftover from an
  earlier lecture numbering -- the content is Lecture 6 material, which
  the instructor confirmed on 2026-10-01 after it had briefly been moved
  to Unit 5 and back.

Note: two further live scripts referenced in the lecture, `MultirateLQG.mlx`
and `Kalmansumofgaussians.mlx`, have no matching Python notebook in
`JupyterNotebooks/` -- flag if you'd like either written from scratch.

All files require only `numpy` (and `matplotlib`/`scipy` for the notebook
demos).
