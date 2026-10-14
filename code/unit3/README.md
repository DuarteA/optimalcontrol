# Unit 3 code

Companion code for Unit 3 (Search methods in graphs), extracted from and
linking to the notebooks in `JupyterNotebooks/`. All five `A_3_*`
notebooks are the Python versions of the MATLAB LiveScripts (originally
referenced from the Lecture 3 slides), by the same `A_1_`/`A_2_` naming
convention used for Units 1 and 2.

- **`graph_search.py`** -- general-purpose shortest-path solvers:
  `dijkstra` (Dijkstra's algorithm), `astar` (A* with a user-supplied
  heuristic), and `shortestpath_dp` (the DP algorithm applied to the
  transition diagram associated with a graph). All three take a weighted
  adjacency matrix `W` (1-indexed nodes, `np.inf` for absent edges),
  matching the convention used throughout the Unit 3 exercises. Run
  directly (`python graph_search.py`) to reproduce Problem 3.1's
  numbers (shortest path from node 1 to node 3 in the 5-node example
  graph, by both Dijkstra and DP).
- **`A_3_shortestpathgraphs.ipynb`** -- the original annotated notebook
  `graph_search.py` was extracted from; also includes plotting helpers
  for transition diagrams and a runnable comparison of DP/Dijkstra/A* on
  a 6-node example.
- **`A_3_RobotPlanningDeterministic.ipynb`** -- the robot grid-planning
  example from the lecture (shortest path around obstacles in a grid,
  1 for horizontal/vertical moves, $\sqrt{2}$ for diagonal moves), solved
  with the DP algorithm.
- **`A_3_K_Puzzle8.ipynb`** -- the 8-puzzle-on-a-torus example, solved
  with both Dijkstra's and the A* algorithm.
- **`A_3_A_Shortest_path_policy_pacman.ipynb`** -- a simplified Pac-Man
  pursuit-evasion game: computing a shortest-path pursuit policy for a
  ghost in a grid world.
- **`A_3_A_Intersection_policy_pacman.ipynb`** -- a variant of the same
  Pac-Man game, computing a policy at grid intersections.

Only Problem 3.1 (the smallest worked example, directly checkable by
hand against the exercise solution) was extracted to a clean, tested
`.py` module; the notebooks are kept as-is, with the full explained
version, plots, and (for the Pac-Man/puzzle examples) visualization code
that is more naturally explored interactively in Jupyter than as a
standalone script.

All files require only `numpy` (and `matplotlib` for the notebook demos).
