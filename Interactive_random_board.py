"""
Random Interactive Catan Board Placer

The following code generates a random Catan board
You are able the click on any vertex on the board to place a settlement
You can place from 2 to 4 settlements or change the value (MAX_SETTLEMENTS) in the code
There is a 'pip' analysis panel that indicates how many pips your settlement achieves

How to use:
Press 'r' to generate a new random board
Press 'c' to clear the settlements on the board
Right click with mouse on a hexagonal corner dot to place the settlement
- Settlements cannot be placed adjacent to each other
"""

import math
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# =============================
# Important Board Variable Data
# =============================

PIPS = {2:1, 3:2, 4:3, 5:4, 6:5, 8:5, 9:4, 10:3, 11:2, 12:1}

LAND_HEXES = (["Forest"] * 4 + ["Pasture"] * 4 + ["Field"] * 4 + ["Hills"] * 3 + ["Mountains"] * 3 + ["Desert"] * 1)

TOKEN_VALUES = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]



# This makes the shape of the board and gives positions for each row to place the hexes
LAND_HEX_LAYOUT = [
                    [(0,0),(0,1),(0,2)],                    # row 1 has 3 hexes
                    [(1,0),(1,1),(1,2),(1,3)],              # row 2 has 4 hexes
                    [(2,0),(2,1),(2,2),(2,3),(2,4)],        # row 3 has 5 hexes
                    [(3,0),(3,1),(3,2),(3,3)],              # row 4 has 4 hexes
                    [(4,0),(4,1),(4,2)]                     # row 5 has 3 hexes
]

LAND_HEX_INDEX = [hex for row in LAND_HEX_LAYOUT for hex in row]


RESOURCE_COLORS = {
    "Forest" : "#00af00",
    "Pasture" : "#03f500",
    "Field" : "#ffda00",
    "Hills" : "#b04a1a",
    "Mountains" : "#888888",
    "Desert" : "#ccba6a",
}

RESOURCE_NAMES = {
    "Forest" : "Forest",
    "Pasture" : "Pasture",
    "Field" : "Field",
    "Hills" : "Hills",
    "Mountains" : "Mountains",
    "Desert" : "Desert",
}

SETTLEMENT_COLORS = ["#f43c45", "#5b8cff", "#c9c9c9", "#7600A5"]


# VARIABLE TO CHANGE THE MAX NUMBER OF SETTLEMENTS (prefer either 2 or 4 depending on 1 or two players)

MAX_SETTLEMENTS = 4








# =============================
# Random Catan Board Generation
# =============================

def generate_board():
    tiles = LAND_HEXES[:]
    random.shuffle(tiles)
    tokens = TOKEN_VALUES[:]
    random.shuffle(tokens)
    board = {}
    token_index = 0
    for hex_position, resource in zip(LAND_HEX_INDEX, tiles):
        if resource == "Desert":
            board[hex_position] = {"Resource": resource, "Token": None, "Pips": 0}
        else:
            t = tokens[token_index]; token_index += 1
            board[hex_position] = {"Resource": resource, "Token": t, "Pips": PIPS[t]}
    return board









# ================
# Land Hex Drawing
# ================


LAND_HEX_SIZE = 1.0
ROW_OFFSET = [1.0, 0.5, 0.0, 0.5, 1.0]

def land_hex_center(row, column):
    x = column * math.sqrt(3) * LAND_HEX_SIZE + ROW_OFFSET[row] * math.sqrt(3) * LAND_HEX_SIZE
    y = -row * 1.5 * LAND_HEX_SIZE
    return x, y

def land_hex_verticies(cx, cy):
    points = []
    for i in range(6):
        angle = math.radians(60 * i - 30)
        points.append((cx + LAND_HEX_SIZE * math.cos(angle),
                    cy + LAND_HEX_SIZE * math.sin(angle)))
    return points



def generate_vertex_map():
    coordinate_to_vertex_id = {}                # Maps board coordinate to vertex id
    vertex_id_to_coordinate = {}                # Maps vertex id to coordinate on board
    vertex_to_hexes = {}                        # List of hex positions touching the hex
    hex_to_vertices = {}                        # Maps hex position to its 6 vertex id's
    counter = [0]                               

    def get_vertex_id(x, y):
        key = (round(x, 4), round(y, 4))
        if key not in coordinate_to_vertex_id:
            coordinate_to_vertex_id[key] = counter[0]                   # Assign unique ID
            vertex_id_to_coordinate[counter[0]] = (x, y)                # Store vertex position
            counter[0] += 1                 
        return coordinate_to_vertex_id[key]
 
    for r, row in enumerate(LAND_HEX_LAYOUT):
        for c, hex_position in enumerate(row):
            cx, cy = land_hex_center(r, c)
            hex_to_vertices[hex_position] = []
            for (vx, vy) in land_hex_verticies(cx, cy):
                vertex_id = get_vertex_id(vx, vy)
                hex_to_vertices[hex_position].append(vertex_id)         # Store vertex id to hex
                vertex_to_hexes.setdefault(vertex_id, [])               # Create an empty list for the vertex ID if not seen before
                if hex_position not in vertex_to_hexes[vertex_id]:
                    vertex_to_hexes[vertex_id].append(hex_position)
 
    return vertex_to_hexes, hex_to_vertices, vertex_id_to_coordinate










# ===============
# Pip Identifiers
# ===============

def pip_color(pips):
    return {5:"#e84040", 4:"#f4a93c", 3:"#f4e040", 2:"#7ec8f4", 1:"#aaaaaa", 0:"#555555"}.get(pips, "#555555")

def Settlement_Analysis(settlements, board, vertex_to_hexes):
    per_settlement = []
    for vertex_id in settlements:
        hexes = vertex_to_hexes.get(vertex_id, [])
        subTotal_pips = sum(board[hex_position]["Pips"] for hex_position in hexes)
        tiles = [(board[hex_position]["Resource"], board[hex_position]["Token"], board[hex_position]["Pips"]) for hex_position in hexes]
        per_settlement.append({"Vertex ID": vertex_id, "Subtotal of Pips": subTotal_pips, "Tiles": tiles})
 
    # Unique pip total
    observed = set()
    for vertex_id in settlements:
        for hex_position in vertex_to_hexes.get(vertex_id, []):
            observed.add(hex_position)
    unique_pips = sum(board[hex_position]["Pips"] for hex_position in observed)
    resources   = set(board[hex_position]["Resource"] for hex_position in observed if board[hex_position]["Resource"] != "Desert")
 
    return per_settlement, unique_pips, resources








# ===========================
# TODO Fitness Function
# ===========================



# ===========================
# TODO Evolutionary Algorithm
# ===========================




# =========================
# TODO Tournament Selection
# =========================



# ==================
# TODO Recombination
# ==================



# =============
# TODO Mutation
# =============











# ==============
# Visualizer Tab
# ==============

def draw_board(board_axes, board, vertex_id_to_coordinate, settlements):
    """Draw the hex board, number tokens, vertex dots, and settlement markers."""
    board_axes.cla()
    board_axes.set_facecolor("#0f1117")
    board_axes.set_aspect("equal")
    board_axes.axis("off")
 
    # Title changes depending on how many settlements have been placed
    number_placed = len(settlements)
    if number_placed < MAX_SETTLEMENTS:
        title_text = f"Click a vertex to place settlement {number_placed + 1} of {MAX_SETTLEMENTS}"
    else:
        title_text = f"All {MAX_SETTLEMENTS} settlements placed  |  Press 'c' to clear"
    board_axes.set_title(title_text, fontsize=11, color="white", pad=8)
 
    # Draw each hex tile
    for row_index, row in enumerate(LAND_HEX_LAYOUT):
        for column_index, hex_position in enumerate(row):
            center_x, center_y = land_hex_center(row_index, column_index)
            tile = board[hex_position]
            corners = land_hex_verticies(center_x, center_y)
 
            # Close the polygon by repeating the first point at the end
            x_points = [point[0] for point in corners] + [corners[0][0]]
            y_points = [point[1] for point in corners] + [corners[0][1]]
 
            # Fill the hex with its resource colour
            board_axes.fill(x_points, y_points, color=RESOURCE_COLORS[tile["Resource"]], zorder=1, alpha=0.92)
 
            # Draw the hex border
            border_color     = "#eeeeee"
            border_thickness = 1.2
            board_axes.plot(x_points, y_points, color=border_color, linewidth=border_thickness, zorder=2)
 
            # Resource name label at top of hex
            board_axes.text(center_x, center_y + 0.50, RESOURCE_NAMES[tile["Resource"]], ha="center", va="center", fontsize=6.5, color="white", fontweight="bold", zorder=3)
 
            # Number token circle and number (skip desert)
            if tile["Token"]:
                number_color = "#ff4444" if tile["Pips"] >= 4 else "#eeeeee"
                token_circle = Circle((center_x, center_y - 0.10), radius=0.35, facecolor="#1a1a1a", edgecolor=number_color, linewidth=1.5, zorder=4)
                board_axes.add_patch(token_circle)
                board_axes.text(center_x, center_y - 0.05, str(tile["Token"]), ha="center", va="center", fontsize=8.5, fontweight="bold", color=number_color, zorder=5)
 
                # Pip dots underneath the number
                for dot_index in range(tile["Pips"]):
                    dot_x_offset = (dot_index - (tile["Pips"] - 1) / 2) * 0.15
                    pip_dot = Circle((center_x + dot_x_offset, center_y - 0.35), radius=0.035, facecolor=number_color, zorder=5)
                    board_axes.add_patch(pip_dot)
 
    # Draw vertex dots (skip vertices that already have a settlement)
    placed_vertices = set(settlements)
    for vertex_id, coordinates in vertex_id_to_coordinate.items():
        vertex_x, vertex_y = coordinates
        if vertex_id in placed_vertices:
            continue
        vertex_dot = Circle((vertex_x, vertex_y), radius=0.09, facecolor="#444444", edgecolor="#888888", linewidth=0.8, zorder=6)
        board_axes.add_patch(vertex_dot)
 
    # Draw settlement markers on top
    for settlement_index, vertex_id in enumerate(settlements):
        vertex_x, vertex_y = vertex_id_to_coordinate[vertex_id]
        settlement_color = SETTLEMENT_COLORS[settlement_index % len(SETTLEMENT_COLORS)]
        settlement_marker = Circle((vertex_x, vertex_y), radius=0.25, facecolor=settlement_color, edgecolor="white", linewidth=2, zorder=10)
        board_axes.add_patch(settlement_marker)
        board_axes.text(vertex_x, vertex_y, str(settlement_index + 1), ha="center", va="center", fontsize=9, fontweight="bold", color="black", zorder=11)
 
    # Set the axis boundaries to fit the board with a small margin
    all_x_centers = [land_hex_center(row_index, column_index)[0]
                     for row_index, row in enumerate(LAND_HEX_LAYOUT)
                     for column_index in range(len(row))]
    all_y_centers = [land_hex_center(row_index, column_index)[1]
                     for row_index, row in enumerate(LAND_HEX_LAYOUT)
                     for column_index in range(len(row))]
    margin = 1.2
    board_axes.set_xlim(min(all_x_centers) - margin, max(all_x_centers) + margin)
    board_axes.set_ylim(min(all_y_centers) - margin, max(all_y_centers) + margin)
 
 
def draw_analysis_panel(analysis_axes, board, settlements, vertex_to_hexes):
    """Draw the settlement pip breakdown panel on the right."""
    analysis_axes.cla()
    analysis_axes.set_facecolor("#111111")
    analysis_axes.axis("off")
    analysis_axes.set_title("Settlement Analysis", fontsize=12, fontweight="bold", color="white", pad=8)
 
    # Show a placeholder message if no settlements have been placed yet
    if not settlements:
        analysis_axes.text(0.5, 0.5, "No settlements placed yet.\nClick the dots on the board.", transform=analysis_axes.transAxes, ha="center", va="center", fontsize=10, color="#888888", style="italic")
        return
 
    per_settlement, unique_pips, resource_types = Settlement_Analysis(
        settlements, board, vertex_to_hexes)
 
    # Build a list of (text, color, font_size, font_weight) lines to display
    display_lines = []
 
    for settlement_index, settlement_data in enumerate(per_settlement):
        header_color = SETTLEMENT_COLORS[settlement_index % len(SETTLEMENT_COLORS)]
        display_lines.append((f"Settlement {settlement_index + 1}  (vertex {settlement_data['Vertex ID']})", header_color, 10, "bold"))
        for (resource, number, pips) in settlement_data["Tiles"]:
            number_text  = str(number) if number else "—"
            pip_bar      = "●" * pips + "○" * (5 - pips)
            display_lines.append(( f"  {resource:9s}  #{number_text:>2}  {pip_bar}  ({pips} Pips)", pip_color(pips), 8.5, "normal"))
        display_lines.append((f"  Sub-total: {settlement_data['Subtotal of Pips']} Pips", "#dddddd", 9, "bold"))
        display_lines.append(("", "white", 8, "normal"))    # blank spacer line
 
    # Summary totals at the bottom
    display_lines.append(("─" * 38, "#444444", 8, "normal"))
    display_lines.append((f"Total unique pips:     {unique_pips}",         "#f4a93c", 10, "bold"))
    display_lines.append((f"Resources per 36 rolls: {unique_pips/36:.3f}", "#dddddd", 9,  "normal"))
    display_lines.append((f"Resource types:        {len(resource_types)}",  "#dddddd", 9,  "normal"))
 
    # Alex Note: Add % of Optimal Line Here TODO
    #
    #
    #
    #
    #




    # Render each line top to bottom
    number_of_lines = len(display_lines)
    if number_of_lines > 30:
        scale = 0.5
    elif number_of_lines > 20:
        scale = 0.75
    else:
        scale = 1.0

    current_y = 1.00
    for (text, color, font_size, font_weight) in display_lines:
        scaled_font_size = font_size * scale
        analysis_axes.text(0.05, current_y, text, transform=analysis_axes.transAxes, fontsize=scaled_font_size, color=color, fontweight=font_weight, fontfamily="monospace", va="top")
        current_y -= (0.05 if font_size >= 10 else 0.045) * scale
 
 
def draw_legend_panel(legend_axes):
    """Draw the pip color legend panel."""
    legend_axes.cla()
    legend_axes.set_facecolor("#111111")
    legend_axes.axis("off")
    legend_axes.set_title("Pip Legend", fontsize=11, fontweight="bold",color="white", pad=8)
 
    legend_items = [
        (5, "6 or 8   — 5/36 rolls"),
        (4, "5 or 9   — 4/36 rolls"),
        (3, "4 or 10  — 3/36 rolls"),
        (2, "3 or 11  — 2/36 rolls"),
        (1, "2 or 12  — 1/36 rolls"),
        (0, "Desert   — never rolled"),
    ]

    legend_axes.set_xlim(0, 1)
    legend_axes.set_ylim(0, 1)
    legend_axes.set_aspect("equal")

    for item_index, (pips, label_text) in enumerate(legend_items):
        vertical_position = 0.90 - item_index * 0.15
        color_dot = Circle((0.07, vertical_position), 0.04, color=pip_color(pips), transform=legend_axes.transAxes, clip_on=False)
        legend_axes.add_patch(color_dot)
        legend_axes.text(0.15, vertical_position, label_text, transform=legend_axes.transAxes, va="center", fontsize=8.5, color="#cccccc", fontfamily="monospace")
 
    legend_axes.text(0.5, 0.05,"    'r' = new board     'c' = clear settlements     ",transform=legend_axes.transAxes, ha="center", va="center", fontsize=8, color="#666666")











# ====================
# Mouse Right Click Function
# ====================

CLICK_DETECTION_RADIUS = 0.25
 
def find_nearest_vertex(click_x, click_y, vertex_id_to_coordinate):
    """
    Returns the vertex ID closest to the click position,
    but only if it is within CLICK_DETECTION_RADIUS units.
    Returns None if no vertex is close enough.
    """
    nearest_vertex_id = None
    nearest_distance  = float('inf')
 
    for vertex_id, coordinates in vertex_id_to_coordinate.items():
        vertex_x, vertex_y = coordinates
        distance = math.sqrt((click_x - vertex_x)**2 + (click_y - vertex_y)**2)
        if distance < nearest_distance:
            nearest_distance  = distance
            nearest_vertex_id = vertex_id
 
    if nearest_distance <= CLICK_DETECTION_RADIUS:
        return nearest_vertex_id
    return None


# ======================
# Convergence Chart TODO
# ======================






# ====
# Main
# ====
 
def main():
    # Shared state stored in a dictionary so the event handlers can modify it
    program_state = {
        "Board"       : generate_board(),
        "Settlements" : [],
    }
 
    vertex_to_hexes, hex_to_vertices, vertex_id_to_coordinate = generate_vertex_map()
 
    # Set up the figure with three panels:
    # left column  — board (spans both rows)
    # top right    — settlement analysis
    # bottom right — pip legend
    figure = plt.figure(figsize=(14, 8), facecolor="#0f1117")
    figure.suptitle("Catan — Interactive Settlement Placer", fontsize=14, fontweight="bold", color="white", y=0.99)
    
    # Alex Note: Convergence chart will affect the grid layout
    grid_layout = figure.add_gridspec(2, 2, width_ratios=[1.5, 1],  height_ratios=[1.5, 0.5],  hspace=0.25, wspace=0.05,  left=0.01, right=0.99,  top=0.95, bottom=0.02)
 
    board_axes    = figure.add_subplot(grid_layout[:, 0])   # board — full left column
    analysis_axes = figure.add_subplot(grid_layout[0, 1])   # analysis — top right
    legend_axes   = figure.add_subplot(grid_layout[1, 1])   # legend — bottom right
    # Alex Note: add convergence_axes subplot
    #
    #
    #
    #
 
    def refresh_display():
        """Redraw all three panels with the current program state."""
        draw_board(board_axes, program_state["Board"], vertex_id_to_coordinate, program_state["Settlements"])
        draw_analysis_panel(analysis_axes, program_state["Board"], program_state["Settlements"], vertex_to_hexes)
        draw_legend_panel(legend_axes)
        # Alex Note: we can add a convergence chart here (I can implement myself)
        #
        #
        #
        #
        figure.canvas.draw_idle()
 
    refresh_display()
 
    # Mouse Click Controls
    def on_mouse_click(click_event):
        # Ignore clicks outside the board panel
        if click_event.inaxes != board_axes:
            return
        if click_event.xdata is None or click_event.ydata is None:
            return
        if len(program_state["Settlements"]) >= MAX_SETTLEMENTS:
            print(f"Already placed {MAX_SETTLEMENTS} settlements. Press 'c' to clear.")
            return
 
        nearest_vertex = find_nearest_vertex(click_event.xdata, click_event.ydata, vertex_id_to_coordinate)
        if nearest_vertex is None:
            return
        if nearest_vertex in program_state["Settlements"]:
            print(f"Vertex {nearest_vertex} already has a settlement — pick another.")
            return
        
        # Settlements must be 1 vertex away from eachother
        adjacent_to_placed = set()
        for placed_vertex in program_state["Settlements"]:
            for hex_position in vertex_to_hexes.get(placed_vertex, []):
                hex_corners = hex_to_vertices.get(hex_position, [])
                for i in range(len(hex_corners)):
                    if hex_corners[i] == placed_vertex:
                        previous_corner = hex_corners[i - 1]
                        next_corner = hex_corners[(i + 1) if (i + 1) < len(hex_corners) else 0]
                        adjacent_to_placed.add(previous_corner)
                        adjacent_to_placed.add(next_corner)

        if nearest_vertex in adjacent_to_placed:
            print(f"Vertex {nearest_vertex} is too close — must be at least 1 vertex away.")
            return


 
        program_state["Settlements"].append(nearest_vertex)
        print(f"Placed settlement {len(program_state['Settlements'])} at vertex {nearest_vertex}")
        refresh_display()
 
    # Keyboard Controls
    def on_key_press(key_event):
        if key_event.key == 'r':
            program_state["Board"]       = generate_board()
            program_state["Settlements"] = []
            print("New board generated.")
            refresh_display()
        elif key_event.key == 'c':
            program_state["Settlements"] = []
            print("Settlements cleared.")
            refresh_display()
    # We can add later a key bind 'e' where the EA runns manually
    #
    #
    #
    #
    #



 
    figure.canvas.mpl_connect("button_press_event", on_mouse_click)
    figure.canvas.mpl_connect("key_press_event",    on_key_press)
 
    plt.show()
 
 
if __name__ == "__main__":
    main()