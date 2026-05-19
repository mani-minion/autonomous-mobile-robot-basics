"""
a_star_path_planning.py
──────────────────────────────────────────────────────────────────────────────
A* (A-Star) path planning algorithm for Autonomous Mobile Robots.
Finds the optimal (shortest) collision-free path on a 2D occupancy grid
from a start position to a goal position.

Used in: ROS Navigation Stack (global planner), AMR warehouse robots,
         autonomous vehicles, drone path planning.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Grid Map ──────────────────────────────────────────────────────────────────
# 0 = free space, 1 = obstacle
GRID = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

# ── Node Class ────────────────────────────────────────────────────────────────
class Node:
    def __init__(self, row, col, g=0, h=0, parent=None):
        self.row    = row
        self.col    = col
        self.g      = g          # cost from start
        self.h      = h          # heuristic to goal
        self.f      = g + h      # total cost
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

# ── Heuristic — Euclidean Distance ───────────────────────────────────────────
def heuristic(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# ── A* Algorithm ──────────────────────────────────────────────────────────────
def a_star(grid, start, goal):
    """
    Find shortest path from start to goal on grid using A*.

    Args:
        grid  : 2D numpy array (0=free, 1=obstacle)
        start : (row, col) tuple
        goal  : (row, col) tuple

    Returns:
        path: list of (row, col) tuples from start to goal, or None if no path
    """
    rows, cols = grid.shape

    # 8-directional movement (including diagonals)
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),     # cardinal
        (-1,-1), (-1, 1), (1,-1), (1, 1)        # diagonal
    ]
    step_costs = [1.0, 1.0, 1.0, 1.0,
                  1.414, 1.414, 1.414, 1.414]   # diagonal = √2

    open_heap = []
    start_node = Node(start[0], start[1], g=0, h=heuristic(start, goal))
    heapq.heappush(open_heap, start_node)

    open_dict  = {(start[0], start[1]): start_node}
    closed_set = set()

    while open_heap:
        current = heapq.heappop(open_heap)
        pos = (current.row, current.col)

        if pos in closed_set:
            continue
        closed_set.add(pos)

        # Goal reached
        if pos == goal:
            path = []
            node = current
            while node:
                path.append((node.row, node.col))
                node = node.parent
            return list(reversed(path))

        # Expand neighbors
        for i, (dr, dc) in enumerate(directions):
            nr, nc = current.row + dr, current.col + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue
            if (nr, nc) in closed_set:
                continue

            g_new = current.g + step_costs[i]
            h_new = heuristic((nr, nc), goal)
            neighbor = Node(nr, nc, g=g_new, h=h_new, parent=current)

            if (nr, nc) not in open_dict or open_dict[(nr,nc)].g > g_new:
                heapq.heappush(open_heap, neighbor)
                open_dict[(nr, nc)] = neighbor

    return None  # No path found

# ── Visualize ─────────────────────────────────────────────────────────────────
def visualize(grid, path, start, goal):
    display = np.copy(grid).astype(float)

    cmap = mcolors.ListedColormap(["white", "black", "skyblue", "lime", "red", "orange"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    if path:
        for (r, c) in path:
            if (r, c) != start and (r, c) != goal:
                display[r, c] = 2   # path = skyblue

    display[start[0], start[1]] = 3   # start = lime
    display[goal[0],  goal[1]]  = 4   # goal  = red

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(display, cmap=cmap, norm=norm)

    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0 and display[r, c] not in [2, 3, 4]:
                ax.text(c, r, f"{heuristic((r,c), goal):.1f}",
                        ha="center", va="center", fontsize=5, color="gray")

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.5)
    ax.set_title(f"A* Path Planning  |  Path length: {len(path) if path else 'N/A'} steps  |  "
                 f"Cost: {sum(1.414 if abs(path[i][0]-path[i-1][0])==1 and abs(path[i][1]-path[i-1][1])==1 else 1 for i in range(1,len(path))):.2f}" if path else "A* — No path found",
                 fontsize=11)

    from matplotlib.patches import Patch
    legend = [Patch(color="lime",    label=f"Start {start}"),
              Patch(color="red",     label=f"Goal  {goal}"),
              Patch(color="skyblue", label="Planned path"),
              Patch(color="black",   label="Obstacle"),
              Patch(color="white",   label="Free space")]
    ax.legend(handles=legend, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig("a_star_path.png", dpi=150)
    plt.show()
    print("Saved: a_star_path.png")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    START = (9, 0)
    GOAL  = (0, 14)

    print("A* Path Planning — Autonomous Mobile Robot")
    print(f"Grid size : {GRID.shape[0]} × {GRID.shape[1]}")
    print(f"Start     : {START}")
    print(f"Goal      : {GOAL}")
    print("Searching...\n")

    path = a_star(GRID, START, GOAL)

    if path:
        print(f"✅ Path found! {len(path)} steps")
        print("Path:", " → ".join(str(p) for p in path))
        visualize(GRID, path, START, GOAL)
    else:
        print("❌ No path found — goal is unreachable.")
