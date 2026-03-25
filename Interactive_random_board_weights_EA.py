"""
Random Interactive Catan Board Placer — Competitive EA

EA uses indirect encoding: individuals are weight vectors over vertex features.
The evolved strategy is board-agnostic and competes against a random opponent
using alternating placement (EA → opponent → EA → opponent).

Controls:
  'r'  — new random board
  'c'  — clear settlements
  'e'  — run Evolutionary Algorithm
  'f'  — cycle fitness mode  (difference | ratio | absolute)
"""

import math
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# =============================
# Board Data
# =============================

PIPS = {2:1, 3:2, 4:3, 5:4, 6:5, 8:5, 9:4, 10:3, 11:2, 12:1}

LAND_HEXES = (["Forest"] * 4 + ["Pasture"] * 4 + ["Field"] * 4 +
              ["Hills"] * 3 + ["Mountains"] * 3 + ["Desert"] * 1)

TOKEN_VALUES = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]

LAND_HEX_LAYOUT = [
    [(0,0),(0,1),(0,2)],
    [(1,0),(1,1),(1,2),(1,3)],
    [(2,0),(2,1),(2,2),(2,3),(2,4)],
    [(3,0),(3,1),(3,2),(3,3)],
    [(4,0),(4,1),(4,2)],
]
LAND_HEX_INDEX = [h for row in LAND_HEX_LAYOUT for h in row]

RESOURCE_COLORS = {
    "Forest":"#00af00","Pasture":"#03f500","Field":"#ffda00",
    "Hills":"#b04a1a","Mountains":"#888888","Desert":"#ccba6a",
}
RESOURCE_NAMES = {k: k for k in RESOURCE_COLORS}
SETTLEMENT_COLORS = ["#f43c45", "#5b8cff", "#c9c9c9", "#7600A5"]

MAX_SETTLEMENTS = 2      # settlements per player

# =============================
# EA Hyperparameters
# =============================

EA_POP_SIZE       = 60
EA_GENERATIONS    = 100
EA_NUM_BOARDS     = 30   # training boards per fitness evaluation
EA_MUTATION_SIGMA = 0.15
EA_TEST_BOARDS    = 100  # fresh boards for final comparison plot

# =============================
# Fitness Mode  (cycle with 'f')
# =============================
# "difference" : EA_pips - opponent_pips
# "ratio"      : EA_pips / (EA_pips + opponent_pips)
# "absolute"   : EA_pips  (ignores opponent)

FITNESS_MODES = ["difference", "ratio", "absolute"]
_fitness_mode = [0]   # mutable index

def current_fitness_mode():
    return FITNESS_MODES[_fitness_mode[0]]


# =============================
# Board Generation
# =============================

def generate_board():
    tiles  = LAND_HEXES[:]
    random.shuffle(tiles)
    tokens = TOKEN_VALUES[:]
    random.shuffle(tokens)
    board  = {}
    ti     = 0
    for pos, res in zip(LAND_HEX_INDEX, tiles):
        if res == "Desert":
            board[pos] = {"Resource": res, "Token": None, "Pips": 0}
        else:
            t = tokens[ti]; ti += 1
            board[pos] = {"Resource": res, "Token": t, "Pips": PIPS[t]}
    return board


# =============================
# Geometry
# =============================

LAND_HEX_SIZE = 1.0
ROW_OFFSET    = [1.0, 0.5, 0.0, 0.5, 1.0]

def land_hex_center(row, col):
    x = col * math.sqrt(3) * LAND_HEX_SIZE + ROW_OFFSET[row] * math.sqrt(3) * LAND_HEX_SIZE
    y = -row * 1.5 * LAND_HEX_SIZE
    return x, y

def land_hex_vertices(cx, cy):
    return [(cx + LAND_HEX_SIZE * math.cos(math.radians(60*i - 30)),
             cy + LAND_HEX_SIZE * math.sin(math.radians(60*i - 30))) for i in range(6)]

def generate_vertex_map():
    coord_to_id = {}
    id_to_coord = {}
    vtx_to_hex  = {}
    hex_to_vtx  = {}
    ctr = [0]

    def get_id(x, y):
        key = (round(x, 4), round(y, 4))
        if key not in coord_to_id:
            coord_to_id[key] = ctr[0]
            id_to_coord[ctr[0]] = (x, y)
            ctr[0] += 1
        return coord_to_id[key]

    for r, row in enumerate(LAND_HEX_LAYOUT):
        for c, pos in enumerate(row):
            cx, cy = land_hex_center(r, c)
            hex_to_vtx[pos] = []
            for vx, vy in land_hex_vertices(cx, cy):
                vid = get_id(vx, vy)
                hex_to_vtx[pos].append(vid)
                vtx_to_hex.setdefault(vid, [])
                if pos not in vtx_to_hex[vid]:
                    vtx_to_hex[vid].append(pos)

    return vtx_to_hex, hex_to_vtx, id_to_coord


# =============================
# Settlement Analysis
# =============================

def pip_color(p):
    return {5:"#e84040",4:"#f4a93c",3:"#f4e040",2:"#7ec8f4",1:"#aaaaaa",0:"#555555"}.get(p,"#555555")

def settlement_analysis(settlements, board, vtx_to_hex):
    per = []
    for vid in settlements:
        hexes = vtx_to_hex.get(vid, [])
        sub   = sum(board[h]["Pips"] for h in hexes)
        tiles = [(board[h]["Resource"], board[h]["Token"], board[h]["Pips"]) for h in hexes]
        per.append({"Vertex ID": vid, "Subtotal of Pips": sub, "Tiles": tiles})
    total_pips = sum(
        sum(board[h]["Pips"] for h in vtx_to_hex.get(vid, []))
        for vid in settlements
    )
    observed  = {h for vid in settlements for h in vtx_to_hex.get(vid, [])}
    resources = {board[h]["Resource"] for h in observed if board[h]["Resource"] != "Desert"}
    return per, total_pips, resources


# =============================
# Vertex Features  (indirect encoding)
# =============================
#
# Each vertex → 13 normalised features:
#   pip_sum           - total pip of adjacent tiles
#   max_pip           - highest pip among tiles
#   count_high        - number of 6s and 8s
#   num_hexes         - number of adjacent tiles
#   resource_variety  - number of different resources
#   pip_variance      - variance of pip numbers
#   min_pip           - smallest pip
#   isolation         - number of valid empty adjacent vertices (expansion space)
#   has_wood          - touches a Forest tile (1/0)
#   has_brick         - touches a Hills tile (1/0)
#   has_scarce        - touches a scarce resource (Hills or Mountains) (1/0)
#   is_edge_vertex    - on the edge/corner of the board, num_hexes < 3 (1/0)
#   neighbor_opponents- number of nearby opponent settlements

FEATURE_MAX = [15.0, 5.0, 3.0, 3.0, 3.0, 6.25, 5.0, 6.0, 1.0, 1.0, 1.0, 1.0, 2.0]
NUM_WEIGHTS  = 13

SCARCE_RESOURCES = {"Hills", "Mountains"}   # only 3 tiles each vs 4 for others

def vertex_features(vid, board, vtx_to_hex, hex_to_vtx=None, placed=None, opponent_placed=None):
    hexes     = vtx_to_hex.get(vid, [])
    pips      = [board[h]["Pips"] for h in hexes]
    res_list  = [board[h]["Resource"] for h in hexes]

    pip_sum          = sum(pips)
    max_pip          = max(pips) if pips else 0
    min_pip          = min(pips) if pips else 0
    count_high       = sum(1 for p in pips if p >= 5)
    num_hexes        = len(hexes)
    resource_variety = len({r for r in res_list if r != "Desert"})

    # pip variance
    if len(pips) > 1:
        mean     = pip_sum / len(pips)
        variance = sum((p - mean) ** 2 for p in pips) / len(pips)
    else:
        variance = 0.0

    # isolation: number of valid adjacent vertices not yet placed
    isolation = 0
    adjacent  = set()
    if hex_to_vtx is not None:
        for h in hexes:
            for v in hex_to_vtx.get(h, []):
                if v != vid:
                    adjacent.add(v)
        if placed is not None:
            placed_set = set(placed)
            isolation  = sum(1 for v in adjacent if v not in placed_set)

    # binary resource features
    has_wood   = 1 if any(r == "Forest" for r in res_list) else 0
    has_brick  = 1 if any(r == "Hills"  for r in res_list) else 0
    has_scarce = 1 if any(r in SCARCE_RESOURCES for r in res_list) else 0

    # edge vertex: touches fewer than 3 hexes
    is_edge = 1 if num_hexes < 3 else 0

    # number of opponent settlements sharing an adjacent hex
    neighbor_opp = 0
    if opponent_placed is not None and hex_to_vtx is not None:
        opp_set = set(opponent_placed)
        neighbor_opp = sum(1 for v in adjacent if v in opp_set)

    raw = [pip_sum, max_pip, count_high, num_hexes,
           resource_variety, variance, min_pip, isolation,
           has_wood, has_brick, has_scarce, is_edge, neighbor_opp]
    return [r / m for r, m in zip(raw, FEATURE_MAX)]

def score_vertex(vid, board, vtx_to_hex, weights,
                 hex_to_vtx=None, placed=None, opponent_placed=None):
    return sum(w * f for w, f in zip(weights,
               vertex_features(vid, board, vtx_to_hex, hex_to_vtx, placed, opponent_placed)))


# =============================
# Validity
# =============================

def is_valid(vertex, placed, vtx_to_hex, hex_to_vtx):
    if vertex in placed:
        return False
    for p in placed:
        for h in vtx_to_hex.get(p, []):
            corners = hex_to_vtx.get(h, [])
            for i, c in enumerate(corners):
                if c == p:
                    if vertex in (corners[i-1], corners[(i+1) % len(corners)]):
                        return False
    return True


# =============================
# Competitive Placement
# =============================

def decode_competitive(weights, board, vtx_to_hex, hex_to_vtx, id_to_coord):
    """
    Alternating placement with random starting order.
    Each game randomly decides whether EA or opponent goes first.
    EA uses greedy weighted scoring; opponent places randomly.
    Returns (ea_settlements, opponent_settlements).
    """
    all_placed  = []
    ea_s        = []
    opp_s       = []

    ea_first = random.random() < 0.5   # randomly decide who goes first

    for turn in range(MAX_SETTLEMENTS * 2):
        ea_turn = (turn % 2 == 0) if ea_first else (turn % 2 == 1)

        if ea_turn:                                # EA's turn — greedy
            ranked = sorted(id_to_coord.keys(),
                            key=lambda v: score_vertex(v, board, vtx_to_hex, weights,
                                                       hex_to_vtx, all_placed, opp_s),
                            reverse=True)
            for v in ranked:
                if is_valid(v, all_placed, vtx_to_hex, hex_to_vtx):
                    ea_s.append(v)
                    all_placed.append(v)
                    break
        else:                                      # Opponent's turn — random
            valid = [v for v in id_to_coord if is_valid(v, all_placed, vtx_to_hex, hex_to_vtx)]
            if valid:
                v = random.choice(valid)
                opp_s.append(v)
                all_placed.append(v)

    return ea_s, opp_s

def competitive_with_human_opponent(weights, board, vtx_to_hex, hex_to_vtx, id_to_coord, event):
    """
     ADD STUFF TO THIS FUNCTION TO SIMULATE PLACING CATAN SETTLEMENTS WITH ALGORITHM
     Alternating placement with random starting order.
     Each game randomly decides whether EA or opponent goes first.
     EA uses greedy weighted scoring; opponent places manually on board.
     Returns (ea_settlements, opponent_settlements).
     """
    all_placed = []
    ea_s = []
    opp_s = []

    ea_first = random.random() < 0.5  # randomly decide who goes first

    for turn in range(MAX_SETTLEMENTS * 2):
        ea_turn = (turn % 2 == 0) if ea_first else (turn % 2 == 1)

        if ea_turn:  # EA's turn — greedy
            ranked = sorted(id_to_coord.keys(),
                            key=lambda v: score_vertex(v, board, vtx_to_hex, weights,
                                                       hex_to_vtx, all_placed, opp_s),
                            reverse=True)
            for v in ranked:
                if is_valid(v, all_placed, vtx_to_hex, hex_to_vtx):
                    ea_s.append(v)
                    all_placed.append(v)
                    break

        else:  # Opponent's turn — manual
            valid = [v for v in id_to_coord if is_valid(v, all_placed, vtx_to_hex, hex_to_vtx)]
            if event is not None:
                x, y = event
                v = nearest_vertex(x, y, id_to_coord)
                if v and valid:
                    opp_s.append(v)
                    all_placed.append(v)
                    return ea_s, opp_s  # Return after each manual placement to wait for next click
                # If no click_data provided or invalid, return current state
            return ea_s, opp_s

    return ea_s, opp_s


# =============================
# Fitness Function  (3 modes)
# =============================

def fitness_competitive(ea_s, opp_s, board, vtx_to_hex, mode=None):
    if mode is None:
        mode = current_fitness_mode()
    _, ea_pips,  _ = settlement_analysis(ea_s,  board, vtx_to_hex)
    _, opp_pips, _ = settlement_analysis(opp_s, board, vtx_to_hex)
    if mode == "difference":
        return ea_pips - opp_pips
    elif mode == "ratio":
        return ea_pips / (ea_pips + opp_pips) if (ea_pips + opp_pips) > 0 else 0.0
    else:                                           # absolute
        return float(ea_pips)

def fitness_multi_board(weights, boards, vtx_to_hex, hex_to_vtx, id_to_coord, mode=None):
    """Average fitness across multiple boards (used as EA objective)."""
    return sum(
        fitness_competitive(*decode_competitive(weights, b, vtx_to_hex, hex_to_vtx, id_to_coord),
                            b, vtx_to_hex, mode)
        for b in boards
    ) / len(boards)


# =============================
# EA Operators
# =============================

def tournament_selection(population, fitnesses, k=3):
    """k-way tournament selection."""
    candidates = random.sample(range(len(population)), min(k, len(population)))
    return population[max(candidates, key=lambda i: fitnesses[i])][:]

def recombine(pa, pb):
    """Uniform crossover on weight vectors."""
    return [a if random.random() < 0.5 else b for a, b in zip(pa, pb)]

def mutate(weights, sigma=EA_MUTATION_SIGMA):
    """Gaussian mutation, weights clipped to non-negative."""
    return [max(0.0, w + random.gauss(0, sigma)) for w in weights]


# =============================
# Evolutionary Algorithm
# =============================

def run_ea(vtx_to_hex, hex_to_vtx, id_to_coord,
           pop_size=EA_POP_SIZE, generations=EA_GENERATIONS, num_boards=EA_NUM_BOARDS):
    """
    Evolve a weight vector [w_pip_sum, w_max_pip, w_count_high, w_num_hexes] that
    maximises average competitive fitness across num_boards random training boards.

    Tracks per generation:
      best_per_gen, avg_per_gen     — convergence
      diversity_per_gen             — mean std-dev of weights (population spread)
      crossover_delta_per_gen       — avg(child_fitness - mean_parent_fitness) before mutation
      mutation_delta_per_gen        — avg(after_mutation_fitness - before_mutation_fitness)

    Returns (best_weights, stats_dict).
    """
    mode   = current_fitness_mode()
    boards = [generate_board() for _ in range(num_boards)]
    # Smaller board sample for operator-effect tracking (keeps runtime reasonable)
    sample_boards = boards[:5]

    population = [[random.uniform(0, 1) for _ in range(NUM_WEIGHTS)]
                  for _ in range(pop_size)]

    best_weights = None
    best_fitness = -1e9

    best_per_gen          = []
    avg_per_gen           = []
    diversity_per_gen     = []
    crossover_delta_per_gen = []
    mutation_delta_per_gen  = []

    for gen in range(generations):
        fitnesses = [fitness_multi_board(ind, boards, vtx_to_hex, hex_to_vtx, id_to_coord, mode)
                     for ind in population]

        gen_best = max(fitnesses)
        gen_avg  = sum(fitnesses) / len(fitnesses)
        best_per_gen.append(gen_best)
        avg_per_gen.append(gen_avg)

        # Progress bar
        pct  = (gen + 1) / generations
        bar  = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
        print(f"\r[{bar}] {gen+1}/{generations}  best={gen_best:.3f}  avg={gen_avg:.3f}", end="", flush=True)
        if gen == generations - 1:
            print()

        # Diversity: mean of per-dimension std-dev across population
        diversity = 0.0
        for dim in range(NUM_WEIGHTS):
            vals = [ind[dim] for ind in population]
            mean = sum(vals) / len(vals)
            diversity += math.sqrt(sum((v - mean)**2 for v in vals) / len(vals))
        diversity_per_gen.append(diversity / NUM_WEIGHTS)

        best_idx = fitnesses.index(gen_best)
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_weights = population[best_idx][:]

        # Build next generation + track operator effects
        new_population   = [population[best_idx][:]]
        xover_deltas     = []
        mut_deltas       = []

        while len(new_population) < pop_size:
            pa     = tournament_selection(population, fitnesses)
            pb     = tournament_selection(population, fitnesses)
            child  = recombine(pa, pb)

            # Crossover delta: child fitness vs average parent fitness (on sample boards)
            fa     = fitness_multi_board(pa,    sample_boards, vtx_to_hex, hex_to_vtx, id_to_coord, mode)
            fb     = fitness_multi_board(pb,    sample_boards, vtx_to_hex, hex_to_vtx, id_to_coord, mode)
            fc     = fitness_multi_board(child, sample_boards, vtx_to_hex, hex_to_vtx, id_to_coord, mode)
            xover_deltas.append(fc - (fa + fb) / 2)

            child_before = child[:]
            child        = mutate(child)

            # Mutation delta: fitness after vs before mutation (on sample boards)
            fm = fitness_multi_board(child, sample_boards, vtx_to_hex, hex_to_vtx, id_to_coord, mode)
            mut_deltas.append(fm - fc)

            new_population.append(child)

        crossover_delta_per_gen.append(sum(xover_deltas) / len(xover_deltas))
        mutation_delta_per_gen.append(sum(mut_deltas)   / len(mut_deltas))
        population = new_population

    stats = {
        "best_per_gen"           : best_per_gen,
        "avg_per_gen"            : avg_per_gen,
        "diversity_per_gen"      : diversity_per_gen,
        "crossover_delta_per_gen": crossover_delta_per_gen,
        "mutation_delta_per_gen" : mutation_delta_per_gen,
    }
    return best_weights, stats


# =============================
# Results Figure
# =============================

def show_results_figure(stats, best_weights, vtx_to_hex, hex_to_vtx, id_to_coord):
    """
    Pop up a separate figure with 6 panels:
      1. Convergence (best + avg fitness per generation)
      2. Population diversity per generation
      3. Crossover effect (avg child - avg_parent per generation)
      4. Mutation effect (avg fitness delta per generation)
      5. EA vs random opponent pip comparison (box plot over EA_TEST_BOARDS fresh boards)
      6. Best weight vector values
    """
    mode = current_fitness_mode()

    # ── Test on fresh boards ──────────────────────────────────────────────
    test_boards  = [generate_board() for _ in range(EA_TEST_BOARDS)]
    ea_pips_list  = []
    opp_pips_list = []
    for b in test_boards:
        ea_s, opp_s = decode_competitive(best_weights, b, vtx_to_hex, hex_to_vtx, id_to_coord)
        _, ep, _ = settlement_analysis(ea_s,  b, vtx_to_hex)
        _, op, _ = settlement_analysis(opp_s, b, vtx_to_hex)
        ea_pips_list.append(ep)
        opp_pips_list.append(op)

    # ── Figure layout ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor="white")
    fig.suptitle(f"EA Results  |  fitness mode: {mode}  |  {EA_TEST_BOARDS} test boards",
                 fontsize=13, fontweight="bold", color="black")

    DARK   = "white"
    TICK_C = "black"

    def style(ax, title, xlabel, ylabel):
        ax.set_facecolor(DARK)
        ax.set_title(title, color="black", fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, color=TICK_C, fontsize=8)
        ax.set_ylabel(ylabel, color=TICK_C, fontsize=8)
        ax.tick_params(colors=TICK_C, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#aaaaaa")

    gens = range(len(stats["best_per_gen"]))

    # 1. Convergence
    ax = axes[0, 0]
    style(ax, "Convergence", "Generation", f"Fitness ({mode})")
    ax.plot(gens, stats["best_per_gen"], color="#f4a93c", lw=1.5, label="Best")
    ax.plot(gens, stats["avg_per_gen"],  color="#7ec8f4", lw=1.0, alpha=0.8, label="Average")
    ax.legend(fontsize=7, facecolor="white", labelcolor="black")

    # 2. Population diversity
    ax = axes[0, 1]
    style(ax, "Population Diversity", "Generation", "Mean weight std-dev")
    ax.plot(gens, stats["diversity_per_gen"], color="#88ff88", lw=1.2)

    # 3. Crossover effect
    ax = axes[0, 2]
    style(ax, "Crossover Effect", "Generation", "Avg(child − avg_parents)")
    ax.plot(gens, stats["crossover_delta_per_gen"], color="#ff88ff", lw=1.2)
    ax.axhline(0, color="#aaaaaa", lw=0.8, linestyle="--")

    # 4. Mutation effect
    ax = axes[1, 0]
    style(ax, "Mutation Effect", "Generation", "Avg fitness Δ from mutation")
    ax.plot(gens, stats["mutation_delta_per_gen"], color="#cc7700", lw=1.2)
    ax.axhline(0, color="#aaaaaa", lw=0.8, linestyle="--")

    # 5. EA vs opponent pip comparison
    ax = axes[1, 1]
    style(ax, f"EA vs Random Opponent  ({EA_TEST_BOARDS} boards)", "Player", "Total pip sum")
    bp = ax.boxplot([ea_pips_list, opp_pips_list],
                    labels=["EA", "Random"],
                    patch_artist=True,
                    medianprops=dict(color="black", lw=2))
    bp["boxes"][0].set_facecolor("#5b8cff")
    bp["boxes"][1].set_facecolor("#aaaaaa")
    for elem in ["whiskers", "caps", "fliers"]:
        for item in bp[elem]:
            item.set_color("black")
    ea_mean  = sum(ea_pips_list)  / len(ea_pips_list)
    opp_mean = sum(opp_pips_list) / len(opp_pips_list)
    ax.text(0.5, 0.95,
            f"EA mean={ea_mean:.1f}   Opponent mean={opp_mean:.1f}",
            transform=ax.transAxes, ha="center", va="top",
            color="black", fontsize=8)






    # 6. Best weight vector
    ax = axes[1, 2]
    style(ax, "Best Weight Vector", "Feature", "Weight value")
    labels = ["pip_sum", "max_pip", "count_high", "num_hexes",
              "res_variety", "pip_var", "min_pip", "isolation",
              "has_wood", "has_brick", "has_scarce", "is_edge", "nbr_opp"]
    colors = ["#f4a93c", "#7ec8f4", "#88ff88", "#ff88ff",
              "#ff7744", "#44ffcc", "#aa88ff", "#ffcc44",
              "#55cc55", "#cc5555", "#cc55cc", "#5555cc", "#cccccc"]
    bars   = ax.bar(labels, best_weights, color=colors)
    for bar, val in zip(bars, best_weights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", color="black", fontsize=8)
    ax.tick_params(axis="x", labelrotation=15)

    plt.tight_layout()
    plt.show()


# =============================
# Board Visualisation
# =============================

def draw_board(ax, board, id_to_coord, ea_settlements, opp_settlements):
    ax.cla()
    ax.set_facecolor("#0f1117")
    ax.set_aspect("equal")
    ax.axis("off")

    all_placed = len(ea_settlements) + len(opp_settlements)
    total      = MAX_SETTLEMENTS * 2
    if all_placed < total:
        turn   = all_placed % 2
        player = "EA" if turn == 0 else "Opponent"
        ax.set_title(f"Turn {all_placed+1}/{total} — {player}'s placement  |  'e'=run EA  'f'=cycle fitness",
                     fontsize=10, color="white", pad=8)
    else:
        ax.set_title(f"All settlements placed  |  Press 'c' to clear  |  fitness: {current_fitness_mode()}",
                     fontsize=10, color="white", pad=8)

    for r, row in enumerate(LAND_HEX_LAYOUT):
        for c, pos in enumerate(row):
            cx, cy   = land_hex_center(r, c)
            tile     = board[pos]
            corners  = land_hex_vertices(cx, cy)
            xs = [p[0] for p in corners] + [corners[0][0]]
            ys = [p[1] for p in corners] + [corners[0][1]]
            ax.fill(xs, ys, color=RESOURCE_COLORS[tile["Resource"]], zorder=1, alpha=0.92)
            ax.plot(xs, ys, color="#eeeeee", lw=1.2, zorder=2)
            ax.text(cx, cy+0.50, RESOURCE_NAMES[tile["Resource"]], ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold", zorder=3)
            if tile["Token"]:
                nc = "#ff4444" if tile["Pips"] >= 4 else "#eeeeee"
                ax.add_patch(Circle((cx, cy-0.10), 0.35, facecolor="#1a1a1a", edgecolor=nc, lw=1.5, zorder=4))
                ax.text(cx, cy-0.05, str(tile["Token"]), ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color=nc, zorder=5)
                for di in range(tile["Pips"]):
                    ox = (di - (tile["Pips"]-1)/2) * 0.15
                    ax.add_patch(Circle((cx+ox, cy-0.35), 0.035, facecolor=nc, zorder=5))

    placed_set = set(ea_settlements) | set(opp_settlements)
    for vid, (vx, vy) in id_to_coord.items():
        if vid not in placed_set:
            ax.add_patch(Circle((vx, vy), 0.09, facecolor="#444444", edgecolor="#888888", lw=0.8, zorder=6))

    # EA settlements (blue tones)
    for i, vid in enumerate(ea_settlements):
        vx, vy = id_to_coord[vid]
        ax.add_patch(Circle((vx, vy), 0.25, facecolor=SETTLEMENT_COLORS[i], edgecolor="white", lw=2, zorder=10))
        ax.text(vx, vy, f"E{i+1}", ha="center", va="center", fontsize=8, fontweight="bold", color="black", zorder=11)

    # Opponent settlements (grey/purple)
    for i, vid in enumerate(opp_settlements):
        vx, vy = id_to_coord[vid]
        ax.add_patch(Circle((vx, vy), 0.25, facecolor=SETTLEMENT_COLORS[2+i], edgecolor="white", lw=2, zorder=10))
        ax.text(vx, vy, f"O{i+1}", ha="center", va="center", fontsize=8, fontweight="bold", color="black", zorder=11)

    xs_all = [land_hex_center(r, c)[0] for r, row in enumerate(LAND_HEX_LAYOUT) for c in range(len(row))]
    ys_all = [land_hex_center(r, c)[1] for r, row in enumerate(LAND_HEX_LAYOUT) for c in range(len(row))]
    m = 1.2
    ax.set_xlim(min(xs_all)-m, max(xs_all)+m)
    ax.set_ylim(min(ys_all)-m, max(ys_all)+m)


def draw_analysis(ax, board, ea_s, opp_s, vtx_to_hex, ea_best_fitness=None):
    ax.cla()
    ax.set_facecolor("#111111")
    ax.axis("off")
    ax.set_title("Settlement Analysis", fontsize=12, fontweight="bold", color="white", pad=8)

    if not ea_s and not opp_s:
        ax.text(0.5, 0.5, "Press 'e' to run EA\nor click vertices manually.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="#888888", style="italic")
        return

    lines = []

    for label, settlements, colors in [("EA", ea_s, SETTLEMENT_COLORS[:2]),
                                        ("Opponent", opp_s, SETTLEMENT_COLORS[2:])]:
        if not settlements:
            continue
        _, upips, ures = settlement_analysis(settlements, board, vtx_to_hex)
        lines.append((f"── {label} ──────────────────────", "#aaaaaa", 9, "normal"))
        per, _, _ = settlement_analysis(settlements, board, vtx_to_hex)
        for i, sd in enumerate(per):
            lines.append((f"Settlement {i+1}  (v{sd['Vertex ID']})", colors[i % len(colors)], 9.5, "bold"))
            for res, num, pips in sd["Tiles"]:
                nt  = str(num) if num else "—"
                bar = "●"*pips + "○"*(5-pips)
                lines.append((f"  {res:9s}  #{nt:>2}  {bar}", pip_color(pips), 8, "normal"))
            lines.append((f"  sub: {sd['Subtotal of Pips']} pips", "#dddddd", 8.5, "bold"))
        lines.append((f"  Unique pips: {upips}   Resources: {len(ures)}", "#f4a93c", 9, "bold"))
        lines.append(("", "white", 8, "normal"))

    # Competitive summary
    if ea_s and opp_s:
        _, ep, _ = settlement_analysis(ea_s,  board, vtx_to_hex)
        _, op, _ = settlement_analysis(opp_s, board, vtx_to_hex)
        diff  = ep - op
        ratio = ep / (ep + op) if (ep + op) > 0 else 0
        lines.append(("─" * 36, "#444444", 8, "normal"))
        col = "#88ff88" if diff >= 0 else "#e84040"
        lines.append((f"Pip advantage: {diff:+d}   ({ratio*100:.1f}%)", col, 10, "bold"))
        lines.append((f"Fitness mode: {current_fitness_mode()}", "#aaaaaa", 8, "normal"))

    n     = len(lines)
    scale = 0.5 if n > 32 else (0.72 if n > 22 else 1.0)
    y     = 1.00
    for text, color, fs, fw in lines:
        ax.text(0.04, y, text, transform=ax.transAxes,
                fontsize=fs*scale, color=color, fontweight=fw,
                fontfamily="monospace", va="top")
        y -= (0.048 if fs >= 9.5 else 0.042) * scale


def draw_legend(ax):
    ax.cla()
    ax.set_facecolor("#111111")
    ax.axis("off")
    ax.set_title("Pip Legend", fontsize=11, fontweight="bold", color="white", pad=8)
    items = [(5,"6/8 — 5/36"),(4,"5/9 — 4/36"),(3,"4/10 — 3/36"),
             (2,"3/11 — 2/36"),(1,"2/12 — 1/36"),(0,"Desert — 0")]
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal")
    for i, (p, label) in enumerate(items):
        vp = 0.90 - i*0.15
        ax.add_patch(Circle((0.07, vp), 0.04, color=pip_color(p),
                             transform=ax.transAxes, clip_on=False))
        ax.text(0.15, vp, label, transform=ax.transAxes, va="center",
                fontsize=8.5, color="#cccccc", fontfamily="monospace")
    ax.text(0.5, 0.04, "'r'=new board  'c'=clear  'e'=EA  'f'=fitness",
            transform=ax.transAxes, ha="center", fontsize=7.5, color="#666666")


def draw_convergence(ax, stats):
    ax.cla()
    ax.set_facecolor("#111111")
    ax.set_title(f"EA Convergence  [{current_fitness_mode()}]",
                 fontsize=10, fontweight="bold", color="white", pad=6)
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")
    if stats is None:
        ax.text(0.5, 0.5, "Press 'e' to run EA",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#888888", style="italic")
        return
    gens = range(len(stats["best_per_gen"]))
    ax.plot(gens, stats["best_per_gen"], color="#f4a93c", lw=1.5, label="Best")
    ax.plot(gens, stats["avg_per_gen"],  color="#7ec8f4", lw=1.0, alpha=0.8, label="Average")
    ax.set_xlabel("Generation", fontsize=7, color="#aaaaaa")
    ax.set_ylabel("Fitness", fontsize=7, color="#aaaaaa")
    ax.legend(fontsize=7, facecolor="#222222", labelcolor="white", framealpha=0.8)


# =============================
# Click Helper
# =============================

CLICK_RADIUS = 0.25

def nearest_vertex(cx, cy, id_to_coord):
    best_id, best_d = None, float("inf")
    for vid, (vx, vy) in id_to_coord.items():
        d = math.sqrt((cx-vx)**2 + (cy-vy)**2)
        if d < best_d:
            best_d, best_id = d, vid
    return best_id if best_d <= CLICK_RADIUS else None


# =============================
# Main
# =============================

def main():
    state = {
        "Board"      : generate_board(),
        "EA_s"       : [],
        "Opp_s"      : [],
        "EA_weights" : None,
        "EA_stats"   : None,
        "EA_fitness" : None,
        "Manual_opponent_active" : False,
        "EA_first" : None,
        "Human_session_active"   : False,   # True while 30-board session is running
        "Human_session_boards"   : [],      # list of 30 pre-generated boards
        "Human_session_index"    : 0,       # which board we are on (0-29)
        "Human_pip_scores"       : [],      # human pip total per board
        "EA_pip_scores_session"  : [],      # EA pip total per board (same boards)
    }

    vtx_to_hex, hex_to_vtx, id_to_coord = generate_vertex_map()

    fig = plt.figure(figsize=(14, 8), facecolor="#0f1117")
    fig.suptitle("Catan — Competitive EA  (indirect encoding vs random opponent)",
                 fontsize=13, fontweight="bold", color="white", y=0.99)

    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1],
                          height_ratios=[1.8, 0.55, 0.75],
                          hspace=0.35, wspace=0.05,
                          left=0.01, right=0.99, top=0.95, bottom=0.02)

    ax_board = fig.add_subplot(gs[:, 0])
    ax_ana   = fig.add_subplot(gs[0, 1])
    ax_leg   = fig.add_subplot(gs[1, 1])
    ax_conv  = fig.add_subplot(gs[2, 1])

    def refresh():
        draw_board(ax_board, state["Board"], id_to_coord, state["EA_s"], state["Opp_s"])
        draw_analysis(ax_ana, state["Board"], state["EA_s"], state["Opp_s"], vtx_to_hex, state["EA_fitness"])
        draw_legend(ax_leg)
        draw_convergence(ax_conv, state["EA_stats"])
        fig.canvas.draw_idle()

    refresh()

    def ea_place(state, vtx_to_hex, hex_to_vtx, id_to_coord):
        all_placed = state["EA_s"] + state["Opp_s"]

        ranked = sorted(
            id_to_coord.keys(),
            key=lambda v: score_vertex(
                v,
                state["Board"],
                vtx_to_hex,
                state["EA_weights"],
                hex_to_vtx,
                all_placed,
                state["Opp_s"],
            ),
            reverse=True,
        )

        for v in ranked:
            if is_valid(v, all_placed, vtx_to_hex, hex_to_vtx):
                state["EA_s"].append(v)
                print(f"EA placed at vertex {v}")
                return

    def show_human_vs_ea_chart():
        """Show a box plot comparing human pip scores vs EA pip scores across the 30-board session."""
        human_scores = state["Human_pip_scores"]
        ea_scores    = state["EA_pip_scores_session"]

        fig_results, ax_results = plt.subplots(figsize=(7, 5), facecolor="white")
        fig_results.suptitle(
            f"Human vs EA — {len(human_scores)} board session",
            fontsize=13, fontweight="bold", color="black"
        )

        box_plot = ax_results.boxplot(
            [ea_scores, human_scores],
            labels=["EA", "You (Human)"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2)
        )
        box_plot["boxes"][0].set_facecolor("#5b8cff")
        box_plot["boxes"][1].set_facecolor("#f43c45")
        for element in ["whiskers", "caps", "fliers"]:
            for item in box_plot[element]:
                item.set_color("black")

        ea_mean    = sum(ea_scores)    / len(ea_scores)
        human_mean = sum(human_scores) / len(human_scores)

        ax_results.text(
            0.5, 0.95,
            f"EA mean = {ea_mean:.1f}     You mean = {human_mean:.1f}",
            transform=ax_results.transAxes,
            ha="center", va="top", fontsize=10, color="black"
        )
        ax_results.set_ylabel("Total pip score", fontsize=10)
        ax_results.set_facecolor("white")
        ax_results.tick_params(colors="black")
        for spine in ax_results.spines.values():
            spine.set_edgecolor("#aaaaaa")

        plt.tight_layout()
        plt.show()

    def start_human_session():
        """Begin a 30-board human vs EA session."""
        if state["EA_weights"] is None:
            print("Run the EA first with 'e' before starting a human session.")
            return

        number_of_session_boards = 30
        state["Human_session_boards"]  = [generate_board() for _ in range(number_of_session_boards)]
        state["Human_session_index"]   = 0
        state["Human_pip_scores"]      = []
        state["EA_pip_scores_session"] = []
        state["Human_session_active"]  = True

        # Pre-compute EA score for every board using the evolved weights
        for session_board in state["Human_session_boards"]:
            ea_s, _ = decode_competitive(
                state["EA_weights"], session_board, vtx_to_hex, hex_to_vtx, id_to_coord
            )
            _, ea_pips, _ = settlement_analysis(ea_s, session_board, vtx_to_hex)
            state["EA_pip_scores_session"].append(ea_pips)

        # Load the first board
        state["Board"]  = state["Human_session_boards"][0]
        state["EA_s"]   = []
        state["Opp_s"]  = []
        state["EA_first"] = False   # Human always places first in session mode
        print(f"Human session started — place your {MAX_SETTLEMENTS} settlements on each of the {number_of_session_boards} boards.")
        print("Board 1 / 30")
        refresh()

    def advance_human_session():
        """Record score for current board and move to the next, or end the session."""
        _, human_pips, _ = settlement_analysis(state["Opp_s"], state["Board"], vtx_to_hex)
        state["Human_pip_scores"].append(human_pips)

        board_number = state["Human_session_index"] + 1
        ea_pips_this_board = state["EA_pip_scores_session"][state["Human_session_index"]]
        print(f"Board {board_number}: You = {human_pips} pips  |  EA = {ea_pips_this_board} pips")

        state["Human_session_index"] += 1

        # Check if session is complete
        if state["Human_session_index"] >= len(state["Human_session_boards"]):
            state["Human_session_active"] = False
            print("Session complete! Showing results...")
            show_human_vs_ea_chart()
            return

        # Load the next board
        state["Board"]  = state["Human_session_boards"][state["Human_session_index"]]
        state["EA_s"]   = []
        state["Opp_s"]  = []
        print(f"Board {state['Human_session_index'] + 1} / {len(state['Human_session_boards'])}")
        refresh()


    def on_click(event):

        if event.inaxes != ax_board or event.xdata is None:
            return

        # ── Human session mode (30-board comparison) ──────────────────────
        if state["Human_session_active"]:
            if len(state["Opp_s"]) >= MAX_SETTLEMENTS:
                return   # Already placed all settlements for this board

            all_placed = state["EA_s"] + state["Opp_s"]
            nearest = nearest_vertex(event.xdata, event.ydata, id_to_coord)
            if nearest is None:
                return
            if not is_valid(nearest, all_placed, vtx_to_hex, hex_to_vtx):
                print("Invalid placement — too close to another settlement.")
                return

            state["Opp_s"].append(nearest)
            print(f"You placed at vertex {nearest}  ({len(state['Opp_s'])} / {MAX_SETTLEMENTS})")
            refresh()

            # Once human has placed all settlements, record and advance
            if len(state["Opp_s"]) >= MAX_SETTLEMENTS:
                advance_human_session()
            return

        if not state["Manual_opponent_active"]:
            return

        all_placed = state["EA_s"] + state["Opp_s"]

        if len(all_placed) >= MAX_SETTLEMENTS * 2:
            _, ea_pips, _ = settlement_analysis(state["EA_s"], state["Board"], vtx_to_hex)
            _, opp_pips, _ = settlement_analysis(state["Opp_s"], state["Board"], vtx_to_hex)
            print(f"Final score — EA: {ea_pips} pips | You: {opp_pips} pips")
            state["Manual_opponent_active"] = False
            return

        turn = len(all_placed)

        if state["EA_first"]:
            ea_turn = (turn % 2 == 0)
        else:
            ea_turn = (turn % 2 == 1)

        # EA TURN
        if ea_turn:
            ea_place(state, vtx_to_hex, hex_to_vtx, id_to_coord)
            refresh()
            return

        # HUMAN TURN
        v = nearest_vertex(event.xdata, event.ydata, id_to_coord)
        if v is None:
            return

        if not is_valid(v, all_placed, vtx_to_hex, hex_to_vtx):
            print("Invalid placement.")
            return

        state["Opp_s"].append(v)
        print(f"You placed at vertex {v}")

        refresh()

        # If EA still has settlements left, place immediately
        all_placed = state["EA_s"] + state["Opp_s"]
        if len(all_placed) < MAX_SETTLEMENTS * 2:

            if state["EA_first"]:
                ea_turn = (len(all_placed) % 2 == 0)
            else:
                ea_turn = (len(all_placed) % 2 == 1)

            if ea_turn and len(state["EA_s"]) < MAX_SETTLEMENTS:
                ea_place(state, vtx_to_hex, hex_to_vtx, id_to_coord)
                refresh()

    def on_key(event):

        default_weights = [4.373, 0.791, 0.030, 1.025]
        if event.key == 'r':
            state["Board"] = generate_board()
            if state["EA_weights"] is not None:
                ea_s, opp_s = decode_competitive(state["EA_weights"], state["Board"],
                                                 vtx_to_hex, hex_to_vtx, id_to_coord)
                state["EA_s"], state["Opp_s"] = ea_s, opp_s
                print("New board — EA strategy re-applied.")
            else:
                state["EA_s"], state["Opp_s"] = [], []
                print("New board generated.")
            refresh()

        elif event.key == 'c':
            state["EA_s"], state["Opp_s"] = [], []
            print("Settlements cleared.")
            refresh()

        elif event.key == 'f':
            _fitness_mode[0] = (_fitness_mode[0] + 1) % len(FITNESS_MODES)
            print(f"Fitness mode → {current_fitness_mode()}")
            refresh()

        elif event.key == 'e':
            mode = current_fitness_mode()
            print(f"Running EA  ({EA_GENERATIONS} gen, pop {EA_POP_SIZE}, "
                  f"{EA_NUM_BOARDS} boards, mode={mode})...")
            best_weights, stats = run_ea(vtx_to_hex, hex_to_vtx, id_to_coord)

            state["EA_weights"] = best_weights
            state["EA_stats"]   = stats
            state["EA_fitness"] = max(stats["best_per_gen"])

            ea_s, opp_s = decode_competitive(best_weights, state["Board"],
                                             vtx_to_hex, hex_to_vtx, id_to_coord)
            state["EA_s"], state["Opp_s"] = ea_s, opp_s

            w = best_weights
            print(f"EA done | weights: pip_sum={w[0]:.3f}  max_pip={w[1]:.3f}  "
                  f"count_high={w[2]:.3f}  num_hexes={w[3]:.3f}")
            print(f"Best avg fitness ({mode}): {state['EA_fitness']:.3f}")
            refresh()

            # Show detailed results figure
            show_results_figure(stats, best_weights, vtx_to_hex, hex_to_vtx, id_to_coord)

        elif event.key == 'v':

            if state["EA_weights"] is None:
                print("Using default EA weights")
                state["EA_weights"] = [4.373, 0.791, 0.030, 1.025]

            state["Manual_opponent_active"] = True
            state["EA_s"] = []
            state["Opp_s"] = []

            state["EA_first"] = random.random() < 0.5

            if state["EA_first"]:
                print("EA goes first")
                ea_place(state, vtx_to_hex, hex_to_vtx, id_to_coord)
            else:
                print("You go first. Click a vertex.")

            refresh()

            show_results_figure(stats, state["EA_weights"], vtx_to_hex, hex_to_vtx, id_to_coord)

        elif event.key == 'h':
            start_human_session()



    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event",    on_key)
    plt.show()


if __name__ == "__main__":
    main()
