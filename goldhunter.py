
"""
Gold Hunter - AI Search & Optimization Algorithms
===================================================
Solves a 6x6 grid-based gold collection problem using:
  1. A* Search  – optimal, with admissible MST heuristic
  2. Greedy Descent – steepest-descent hill climbing with restarts
  3. Genetic Algorithm – evolutionary optimization with OX crossover

Usage:
  python goldhunter.py --alg a_star   --layout input.txt
  python goldhunter.py --alg greedy   --layout input.txt
  python goldhunter.py --alg genetic  --layout input.txt
  python goldhunter.py --compare      --layout input.txt
"""

import argparse
import heapq
import random
import time
import sys
from collections import deque

# ── Grid constants ─────────────────────────────────────────────
ROWS, COLS = 6, 6
# Movement directions: (delta_row, delta_col, action_letter)
DIRECTIONS = [(-1, 0, 'U'), (1, 0, 'D'), (0, -1, 'L'), (0, 1, 'R')]


# ══════════════════════════════════════════════════════════════
#  SECTION 1: GRID PARSING
# ══════════════════════════════════════════════════════════════

def parse_grid(filepath):
    """
    Read and parse a 6x6 grid layout from a text file.

    Each cell is one of:  S (start), G (gold), # (wall), . (empty)

    Returns
    -------
    grid  : list[list[str]]   – 6x6 character grid
    start : tuple(int,int)    – (row, col) of the start cell
    golds : list[tuple(int,int)] – positions of all gold cells
    Returns (None, None, None) on any parsing error.
    """
    try:
        grid = []
        with open(filepath, 'r') as f:
            for line in f:
                stripped = line.rstrip('\n').rstrip('\r')
                if stripped:                       # skip blank lines
                    grid.append(list(stripped))

        # --- validate dimensions ---
        if len(grid) != ROWS:
            return None, None, None
        for row in grid:
            if len(row) != COLS:
                return None, None, None

        # --- locate start and golds ---
        start = None
        golds = []
        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c] == 'S':
                    start = (r, c)
                elif grid[r][c] == 'G':
                    golds.append((r, c))

        if start is None or len(golds) == 0:
            return None, None, None

        return grid, start, golds

    except FileNotFoundError:
        return None, None, None


# ══════════════════════════════════════════════════════════════
#  SECTION 2: BFS SHORTEST PATH UTILITIES
# ══════════════════════════════════════════════════════════════

def bfs_from(grid, source):
    """
    BFS from *source* to every reachable cell on the grid.

    Returns
    -------
    dist : dict  –  (r,c) → shortest distance from source
    prev : dict  –  (r,c) → ( (parent_r, parent_c), action )
                     used for path reconstruction
    """
    dist = {source: 0}
    prev = {}                       # source has no predecessor
    queue = deque([source])

    while queue:
        r, c = queue.popleft()
        d = dist[(r, c)]
        for dr, dc, action in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < ROWS and 0 <= nc < COLS
                    and grid[nr][nc] != '#'
                    and (nr, nc) not in dist):
                dist[(nr, nc)] = d + 1
                prev[(nr, nc)] = ((r, c), action)
                queue.append((nr, nc))

    return dist, prev


def reconstruct_actions(prev, source, target):
    """
    Rebuild the action sequence from *source* to *target*
    using the predecessor map returned by bfs_from().
    """
    if source == target:
        return []

    actions = []
    cur = target
    while cur != source:
        parent, act = prev[cur]
        actions.append(act)
        cur = parent
    actions.reverse()
    return actions


# ══════════════════════════════════════════════════════════════
#  SECTION 3: PRECOMPUTATION
# ══════════════════════════════════════════════════════════════

def precompute(grid, key_positions):
    """
    Run BFS from every key position (start + golds).

    Returns
    -------
    bfs_maps    : dict  pos → (dist_map, prev_map)
    dist_matrix : dict  (posA, posB) → shortest distance
    path_matrix : dict  (posA, posB) → list of actions
    """
    bfs_maps = {}
    for pos in key_positions:
        if pos not in bfs_maps:
            bfs_maps[pos] = bfs_from(grid, pos)

    dist_matrix = {}
    path_matrix = {}
    for src in key_positions:
        src_dist, src_prev = bfs_maps[src]
        for dst in key_positions:
            d = src_dist.get(dst, float('inf'))
            dist_matrix[(src, dst)] = d
            if d < float('inf'):
                path_matrix[(src, dst)] = reconstruct_actions(src_prev, src, dst)
            else:
                path_matrix[(src, dst)] = []

    return bfs_maps, dist_matrix, path_matrix


# ══════════════════════════════════════════════════════════════
#  SECTION 4: MST (MINIMUM SPANNING TREE) — for A* heuristic    
# ══════════════════════════════════════════════════════════════

def mst_cost_prim(nodes, edge_cost_fn):
    """
    Prim's algorithm for MST cost on a complete graph defined
    by *nodes* and *edge_cost_fn(a, b)*.

    Returns the total MST edge weight, or inf if disconnected.
    """
    n = len(nodes)
    if n <= 1:
        return 0

    in_mst = [False] * n
    # min_edge[i] = cheapest edge from node i to any MST node
    min_edge = [float('inf')] * n

    in_mst[0] = True                              # seed with node 0
    for i in range(1, n):
        min_edge[i] = edge_cost_fn(nodes[0], nodes[i])

    total = 0
    for _ in range(n - 1):
        # pick the cheapest fringe node
        u, u_cost = -1, float('inf')
        for i in range(n):
            if not in_mst[i] and min_edge[i] < u_cost:
                u_cost = min_edge[i]
                u = i
        if u == -1 or u_cost == float('inf'):
            return float('inf')                    # graph disconnected

        in_mst[u] = True
        total += u_cost

        # relax edges from newly added node
        for i in range(n):
            if not in_mst[i]:
                c = edge_cost_fn(nodes[u], nodes[i])
                if c < min_edge[i]:
                    min_edge[i] = c
    return total


# ══════════════════════════════════════════════════════════════
#  SECTION 5: A* SEARCH
# ══════════════════════════════════════════════════════════════

def a_star_search(grid, start, golds):
    """
    A* Search for the Gold Hunter problem.

    State
    -----
    (position, bitmask)   where bit i set ⟹ gold i collected

    Heuristic (admissible)
    ----------------------
    h(p, M) = MST({p} ∪ uncollected ∪ {start})  +  |uncollected|

    • MST ≤ any spanning connected subgraph  ⟹  lower-bounds movement
    • |uncollected| = exact count of remaining grab actions (cost 1 each)
    ⟹ h never overestimates ⟹ admissible ⟹ A* optimal.
    """
    num_golds = len(golds)
    all_collected = (1 << num_golds) - 1          # all bits set

    # --- precompute BFS maps for heuristic lookups ---
    key_positions = [start] + list(golds)
    bfs_maps, dist_matrix, _ = precompute(grid, key_positions)

    # --- verify every gold is reachable from start ---
    for g in golds:
        if dist_matrix.get((start, g), float('inf')) == float('inf'):
            return -1, "None"

    # --- heuristic with memoisation ---
    h_cache = {}

    def heuristic(pos, mask):
        key = (pos, mask)
        if key in h_cache:
            return h_cache[key]

        uncollected = [golds[i] for i in range(num_golds)
                       if not (mask & (1 << i))]

        if not uncollected:
            # all gold grabbed – just return to start
            h_val = bfs_maps[start][0].get(pos, float('inf'))
        else:
            # MST of {pos, uncollected golds, start}
            node_set = list(set([pos] + uncollected + [start]))

            def edge_cost(a, b):
                if a == b:
                    return 0
                if a in bfs_maps:
                    return bfs_maps[a][0].get(b, float('inf'))
                if b in bfs_maps:
                    return bfs_maps[b][0].get(a, float('inf'))
                return float('inf')

            h_val = mst_cost_prim(node_set, edge_cost) + len(uncollected)

        h_cache[key] = h_val
        return h_val

    # --- A* main loop ---
    start_state = (start, 0)
    h0 = heuristic(start, 0)

    counter = 0                                    # tiebreaker
    open_set = [(h0, 0, counter, start_state)]     # (f, g, tie, state)
    g_score = {start_state: 0}
    came_from = {}                                 # state → (parent, action)

    while open_set:
        f, g, _, state = heapq.heappop(open_set)
        pos, mask = state

        if g > g_score.get(state, float('inf')):
            continue                               # outdated entry

        # ── goal check ──
        if mask == all_collected and pos == start:
            actions = []
            cur = state
            while cur in came_from:
                parent, act = came_from[cur]
                actions.append(act)
                cur = parent
            actions.reverse()
            return g, ' '.join(actions)

        r, c = pos

        # ── expand: Grab ──
        for i in range(num_golds):
            if golds[i] == pos and not (mask & (1 << i)):
                new_mask = mask | (1 << i)
                ns = (pos, new_mask)
                ng = g + 1
                if ng < g_score.get(ns, float('inf')):
                    g_score[ns] = ng
                    came_from[ns] = (state, 'G')
                    counter += 1
                    heapq.heappush(open_set,
                                   (ng + heuristic(pos, new_mask),
                                    ng, counter, ns))

        # ── expand: Move U / D / L / R ──
        for dr, dc, act in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and grid[nr][nc] != '#':
                np_ = (nr, nc)
                ns = (np_, mask)
                ng = g + 1
                if ng < g_score.get(ns, float('inf')):
                    g_score[ns] = ng
                    came_from[ns] = (state, act)
                    counter += 1
                    heapq.heappush(open_set,
                                   (ng + heuristic(np_, mask),
                                    ng, counter, ns))

    return -1, "None"


# ══════════════════════════════════════════════════════════════
#  SECTION 6: PERMUTATION EVALUATION  (shared helpers)
# ══════════════════════════════════════════════════════════════

def evaluate_permutation(perm, golds, start, dist_matrix):
    """
    Cost of visiting golds in the order given by *perm*,
    following BFS shortest paths between consecutive waypoints.

    cost = Σ movement_distances + n  (one grab per gold)
    """
    total = 0
    current = start
    for idx in perm:
        d = dist_matrix.get((current, golds[idx]), float('inf'))
        if d == float('inf'):
            return float('inf')
        total += d + 1                             # move + grab
        current = golds[idx]
    # return to start
    d = dist_matrix.get((current, start), float('inf'))
    if d == float('inf'):
        return float('inf')
    total += d
    return total


def build_action_sequence(perm, golds, start, path_matrix):
    """
    Build the full action string for a given visit-order permutation.
    """
    actions = []
    current = start
    for idx in perm:
        gp = golds[idx]
        actions.extend(path_matrix.get((current, gp), []))
        actions.append('G')                        # grab
        current = gp
    actions.extend(path_matrix.get((current, start), []))
    return actions


def nearest_neighbor_perm(golds, start, dist_matrix):
    """
    Nearest-neighbour heuristic: greedily visit the closest
    unvisited gold at every step.
    """
    n = len(golds)
    visited = set()
    perm = []
    current = start
    for _ in range(n):
        best, best_d = -1, float('inf')
        for i in range(n):
            if i not in visited:
                d = dist_matrix.get((current, golds[i]), float('inf'))
                if d < best_d:
                    best_d = d
                    best = i
        if best == -1 or best_d == float('inf'):
            break
        perm.append(best)
        visited.add(best)
        current = golds[best]
    return perm


# ══════════════════════════════════════════════════════════════
#  SECTION 7: GREEDY DESCENT  (Steepest-Descent Hill Climbing)
# ══════════════════════════════════════════════════════════════

def greedy_descent(grid, start, golds, num_restarts=30):
    """
    Greedy Descent for Gold Hunter.

    Representation
    --------------
    π = (π₁, π₂, …, πₙ)  –  a permutation of gold indices
    giving the order in which golds are visited.

    Neighbourhood operators
    -----------------------
    1. Swap   – exchange two elements
    2. 2-opt  – reverse a contiguous sub-sequence

    Strategy
    --------
    Steepest descent: evaluate *all* neighbours, move to the best
    improving one.  Repeat with random restarts to escape local optima.
    """
    random.seed(42)
    n = len(golds)

    key_positions = [start] + list(golds)
    _, dist_matrix, path_matrix = precompute(grid, key_positions)

    # reachability check
    for g in golds:
        if dist_matrix.get((start, g), float('inf')) == float('inf'):
            return -1, "None"

    best_perm = None
    best_cost = float('inf')

    for restart in range(num_restarts):
        # initialise permutation
        if restart == 0:
            cur_perm = nearest_neighbor_perm(golds, start, dist_matrix)
        else:
            cur_perm = list(range(n))
            random.shuffle(cur_perm)

        cur_cost = evaluate_permutation(cur_perm, golds, start, dist_matrix)
        if cur_cost == float('inf'):
            continue

        # ── Phase 1: swap-based steepest descent ──
        improved = True
        while improved:
            improved = False
            nb_cost, nb_swap = cur_cost, None
            for i in range(n):
                for j in range(i + 1, n):
                    trial = cur_perm[:]
                    trial[i], trial[j] = trial[j], trial[i]
                    c = evaluate_permutation(trial, golds, start, dist_matrix)
                    if c < nb_cost:
                        nb_cost = c
                        nb_swap = (i, j)
            if nb_swap:
                i, j = nb_swap
                cur_perm[i], cur_perm[j] = cur_perm[j], cur_perm[i]
                cur_cost = nb_cost
                improved = True

        # ── Phase 2: 2-opt reversal descent ──
        improved = True
        while improved:
            improved = False
            nb_cost, nb_rev = cur_cost, None
            for i in range(n):
                for j in range(i + 2, n + 1):
                    trial = cur_perm[:i] + cur_perm[i:j][::-1] + cur_perm[j:]
                    c = evaluate_permutation(trial, golds, start, dist_matrix)
                    if c < nb_cost:
                        nb_cost = c
                        nb_rev = (i, j)
            if nb_rev:
                i, j = nb_rev
                cur_perm = cur_perm[:i] + cur_perm[i:j][::-1] + cur_perm[j:]
                cur_cost = nb_cost
                improved = True

        if cur_cost < best_cost:
            best_cost = cur_cost
            best_perm = cur_perm[:]

    if best_perm is None or best_cost == float('inf'):
        return -1, "None"

    actions = build_action_sequence(best_perm, golds, start, path_matrix)
    return best_cost, ' '.join(actions)


# ══════════════════════════════════════════════════════════════
#  SECTION 8: GENETIC ALGORITHM
# ══════════════════════════════════════════════════════════════

def genetic_algorithm(grid, start, golds,
                      pop_size=100, generations=500,
                      mutation_rate=0.15, elite_count=5,
                      tournament_k=3):
    """
    Genetic Algorithm for Gold Hunter.

    Chromosome  –  permutation of gold indices (visit order)
    Selection   –  tournament (size k=3)
    Crossover   –  Order Crossover (OX)
    Mutation    –  swap mutation (prob = 0.15)
    Elitism     –  top 5 survive unchanged
    """
    random.seed(42)
    n = len(golds)

    key_positions = [start] + list(golds)
    _, dist_matrix, path_matrix = precompute(grid, key_positions)

    for g in golds:
        if dist_matrix.get((start, g), float('inf')) == float('inf'):
            return -1, "None"

    # ── helper closures ──

    def fitness(perm):
        return evaluate_permutation(perm, golds, start, dist_matrix)

    def tournament(pop, fits):
        idxs = random.sample(range(len(pop)), min(tournament_k, len(pop)))
        winner = min(idxs, key=lambda i: fits[i])
        return pop[winner][:]

    def ox_crossover(p1, p2):
        """Order Crossover (OX): preserves relative ordering."""
        sz = len(p1)
        if sz <= 1:
            return p1[:]
        c1 = random.randint(0, sz - 1)
        c2 = random.randint(0, sz - 1)
        if c1 > c2:
            c1, c2 = c2, c1
        child = [None] * sz
        seg = set()
        for i in range(c1, c2 + 1):
            child[i] = p1[i]
            seg.add(p1[i])
        fill = [x for x in p2 if x not in seg]
        fi = 0
        for i in range(sz):
            if child[i] is None:
                child[i] = fill[fi]
                fi += 1
        return child

    def swap_mutate(perm):
        if len(perm) <= 1:
            return perm[:]
        ch = perm[:]
        i, j = random.sample(range(len(ch)), 2)
        ch[i], ch[j] = ch[j], ch[i]
        return ch

    # ── initialise population ──
    pop = []
    nn = nearest_neighbor_perm(golds, start, dist_matrix)
    if len(nn) == n:
        pop.append(nn)
    while len(pop) < pop_size:
        p = list(range(n))
        random.shuffle(p)
        pop.append(p)

    fits = [fitness(ind) for ind in pop]

    # ── evolution loop ──
    for gen in range(generations):
        paired = sorted(zip(fits, pop), key=lambda x: x[0])
        fits = [p[0] for p in paired]
        pop  = [p[1] for p in paired]

        nxt_pop, nxt_fit = [], []

        # elitism
        for i in range(min(elite_count, len(pop))):
            nxt_pop.append(pop[i][:])
            nxt_fit.append(fits[i])

        # breed offspring
        while len(nxt_pop) < pop_size:
            p1 = tournament(pop, fits)
            p2 = tournament(pop, fits)
            child = ox_crossover(p1, p2)
            if random.random() < mutation_rate:
                child = swap_mutate(child)
            nxt_pop.append(child)
            nxt_fit.append(fitness(child))

        pop, fits = nxt_pop, nxt_fit

    # best in final population
    best_i = min(range(len(pop)), key=lambda i: fits[i])
    best_perm = pop[best_i]
    best_cost = fits[best_i]

    if best_cost == float('inf'):
        return -1, "None"

    actions = build_action_sequence(best_perm, golds, start, path_matrix)
    return best_cost, ' '.join(actions)


# ══════════════════════════════════════════════════════════════
#  SECTION 9: COMPARISON MODE
# ══════════════════════════════════════════════════════════════

def compare_algorithms(grid, start, golds):
    """Run all three algorithms and print a comparison table."""
    algos = [
        ('A* Search',         lambda: a_star_search(grid, start, golds)),
        ('Greedy Descent',    lambda: greedy_descent(grid, start, golds)),
        ('Genetic Algorithm', lambda: genetic_algorithm(grid, start, golds)),
    ]

    print(f"\n{'Algorithm':<22} {'Cost':>6} {'Time (s)':>10}  Actions")
    print('-' * 80)

    for name, fn in algos:
        t0 = time.time()
        cost, actions = fn()
        elapsed = time.time() - t0
        print(f"{name:<22} {cost:>6} {elapsed:>10.4f}  {actions}")


# ══════════════════════════════════════════════════════════════
#  SECTION 10: MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Gold Hunter – AI Search & Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python goldhunter.py --alg a_star  --layout input.txt
  python goldhunter.py --alg greedy  --layout input.txt
  python goldhunter.py --alg genetic --layout input.txt
  python goldhunter.py --compare     --layout input.txt
        """)
    parser.add_argument('--alg', choices=['a_star', 'greedy', 'genetic'],
                        help='Algorithm to run')
    parser.add_argument('--layout', required=True,
                        help='Path to input grid file')
    parser.add_argument('--compare', action='store_true',
                        help='Run all algorithms and compare')
    args = parser.parse_args()

    grid, start, golds = parse_grid(args.layout)
    if grid is None:
        print("-1")
        print("None")
        return

    if args.compare:
        compare_algorithms(grid, start, golds)
        return

    if args.alg is None:
        parser.print_help()
        return

    if args.alg == 'a_star':
        cost, actions = a_star_search(grid, start, golds)
    elif args.alg == 'greedy':
        cost, actions = greedy_descent(grid, start, golds)
    elif args.alg == 'genetic':
        cost, actions = genetic_algorithm(grid, start, golds)

    print(cost)
    print(actions)


if __name__ == '__main__':
    main()






