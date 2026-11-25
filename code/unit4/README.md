# Unit 4 code

Companion code for Unit 4 (Bayes' filter and POMDP), extracted from and
linking to the notebooks in `JupyterNotebooks/`, by the same `A_1_`/`A_2_`/
`A_3_` naming convention used for Units 1-3.

- **`bayes_filter.py`** -- a general-purpose discrete Bayes filter:
  `predict` (time update, $\bar p_{k+1} = Pp_k$), `correct` (measurement
  update, $p_{k+1} = D(y_{k+1})\bar p_{k+1} / \mathbb{1}^\top D(y_{k+1})\bar
  p_{k+1}$), and `bayes_filter` (runs the full prediction/correction loop
  over a sequence of measurements). States and measurements are 0-indexed;
  `P` and `R` follow the column-stochastic convention `P[i,j] =
  Prob[x_{k+1}=i|x_k=j]`, `R[i,j] = Prob[y_k=i|x_k=j]`. Run directly
  (`python bayes_filter.py`) to reproduce Problem 4.1's numbers (angle
  estimation of a spinning wheel from noisy hue measurements) -- the belief
  vectors $p_0,\dots,p_4$ printed match the exercise solution exactly.
- **`A_4_angleestimationbayes.ipynb`** -- the original annotated notebook
  version of the Problem 4.1 example, with plotting of the belief
  distribution over the color wheel at each step (referenced from the
  Lecture 4 slides as `LiveScript/Lecture4/angleestimationbayes.mlx`).
- **`A_4_K_gridworldBayes.ipynb`** -- the grid-world robot tracking example
  from the lecture (Bayes filter over a 2D position grid with a
  disturbance/sensor model), referenced as
  `LiveScript/Lecture4/gridworldBayes.mlx`.
- **`A_4_K_RemoteTempSensingBayes.ipynb`** -- the remote temperature sensing
  example, referenced as `LiveScript/Lecture4/RemoteTempSensingBayes.mlx`.

Note: the lecture slides also reference a fourth live script, `BayesIMU.mlx`
(IMU angle estimation); no matching Python notebook was found in
`JupyterNotebooks/` (the closest file, `B_2_K_KalmanIMU.ipynb`, is a
Kalman-filter example for a different unit) -- flag if you'd like this
written from scratch.

All files require only `numpy` (and `matplotlib` for the notebook demos).
