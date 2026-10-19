"""Shortest-path search algorithms for graphs (Unit 3: Search methods in
graphs).

Inspired by JupyterNotebooks/A_3_shortestpathgraphs.ipynb, rewritten as
clean, reusable functions.

Graphs are represented by a weighted adjacency matrix W (numpy array),
where W[i-1, j-1] is the cost of the directed edge from node i to node j
(nodes are 1-indexed, matching the notes and PS1 exercises), and
`np.inf` marks the absence of an edge. Diagonal entries are ignored (no
self-loops).
"""
import numpy as np


def _reconstruct_path(prev, inode, fnode):
    """Walk the `prev` back-pointers (0-indexed) from fnode to inode and
    return the path as a list of 1-indexed nodes, or None if fnode is
    unreachable from inode."""
    if inode == fnode:
        return [inode]
    node = fnode
    path = [node]
    while node != inode:
        p = prev[node - 1]
        if p is None:
            return None
        node = p + 1
        path.append(node)
    path.reverse()
    return path


def dijkstra(W, inode, fnode):
    """Dijkstra's algorithm: shortest path from node `inode` to node
    `fnode` in a graph with non-negative edge weights `W`.

    Returns
    -------
    path : list of 1-indexed nodes from inode to fnode (None if
        unreachable).
    cost : the cost of that path (np.inf if unreachable).
    """
    n = W.shape[0]
    dist = np.full(n, np.inf)
    dist[inode - 1] = 0.0
    prev = [None] * n
    open_nodes = set(range(n))

    while open_nodes:
        i = min(open_nodes, key=lambda x: dist[x])
        if np.isinf(dist[i]):
            break  # remaining open nodes are unreachable
        open_nodes.remove(i)
        if i == fnode - 1:
            break
        for j in range(n):
            if j == i or np.isinf(W[i, j]):
                continue
            alt = dist[i] + W[i, j]
            if alt < dist[j]:
                dist[j] = alt
                prev[j] = i

    return _reconstruct_path(prev, inode, fnode), dist[fnode - 1]


def astar(W, h, inode, fnode):
    """A* search from node `inode` to node `fnode`, using heuristic `h`
    (array of length n; h[i] estimates the cost from node i+1 to fnode).
    For the algorithm to return a shortest path, `h` must be admissible
    (never overestimate the true remaining cost).

    Returns
    -------
    path : list of 1-indexed nodes from inode to fnode (None if
        unreachable).
    cost : the true cost of that path (np.inf if unreachable).
    """
    n = W.shape[0]
    dist = np.full(n, np.inf)
    dist[inode - 1] = 0.0
    prev = [None] * n
    open_nodes = {inode - 1}
    closed = set()

    while open_nodes:
        i = min(open_nodes, key=lambda x: dist[x] + h[x])
        open_nodes.remove(i)
        if i == fnode - 1:
            break
        closed.add(i)
        for j in range(n):
            if j == i or np.isinf(W[i, j]) or j in closed:
                continue
            alt = dist[i] + W[i, j]
            if alt < dist[j]:
                dist[j] = alt
                prev[j] = i
                open_nodes.add(j)

    return _reconstruct_path(prev, inode, fnode), dist[fnode - 1]


def shortestpath_dp(W, fnode):
    """Dynamic programming algorithm applied to the transition diagram
    associated with the graph: the shortest path to `fnode` using at most
    k hops is computed for k=1,...,n-1 (n-1 hops always suffice in a graph
    with n nodes and no negative cycles), by relaxing every edge at each
    iteration -- the graph-search analogue of the DP recursion used
    throughout the notes.

    Returns
    -------
    policy : policy[i-1] is the next node to move to from node i along a
        shortest path to fnode (policy[fnode-1] = fnode).
    J : J[i-1] is the shortest-path cost from node i to fnode (np.inf if
        unreachable).
    """
    n = W.shape[0]
    J = np.full(n, np.inf)
    J[fnode - 1] = 0.0
    policy = np.arange(1, n + 1)

    for _ in range(n - 1):
        updated = False
        for i in range(n):
            if i == fnode - 1:
                continue
            best_cost, best_j = J[i], None
            for j in range(n):
                if j == i or np.isinf(W[i, j]):
                    continue
                alt = W[i, j] + J[j]
                if alt < best_cost:
                    best_cost, best_j = alt, j
            if best_j is not None:
                J[i] = best_cost
                policy[i] = best_j + 1
                updated = True
        if not updated:
            break

    return policy, J


if __name__ == "__main__":
    # Smoke test / worked example: Problem 3.1 from the Unit 3 exercises
    # (5-node graph, shortest path from node 1 to node 3).
    W = np.array([
        [0, 0, np.inf, np.inf, 1],
        [3, 0, 5, 3, 1],
        [np.inf, 5, 0, 1, 3],
        [np.inf, np.inf, 1, 0, 1],
        [1, 1, 3, 1, 0],
    ], dtype=float)

    path, cost = dijkstra(W, 1, 3)
    print(f"Dijkstra 1->3: path={path}, cost={cost}")

    policy, J = shortestpath_dp(W, 3)
    print(f"DP costs-to-go to node 3: {J}")
    print(f"DP policy (next node from each node): {policy}")
