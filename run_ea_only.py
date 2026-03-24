"""Run EA only — no GUI, just terminal progress bar and results plot."""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, no window

from Interactive_random_board import (
    generate_vertex_map, run_ea, decode_competitive,
    settlement_analysis, show_results_figure
)
import matplotlib.pyplot as plt

print("Building vertex map...")
vtx_to_hex, hex_to_vtx, id_to_coord = generate_vertex_map()

print("Starting EA...\n")
best_weights, stats = run_ea(vtx_to_hex, hex_to_vtx, id_to_coord)

print(f"\nBest weights:")
labels = ["pip_sum", "max_pip", "count_high", "num_hexes",
          "resource_variety", "pip_variance", "min_pip", "isolation",
          "has_wood", "has_brick", "has_scarce", "is_edge", "neighbor_opponents"]
for label, w in zip(labels, best_weights):
    print(f"  {label:20s} = {w:.4f}")

print(f"\nBest fitness: {max(stats['best_per_gen']):.4f}")

# Save results figure to file instead of showing
matplotlib.use("Agg")
show_results_figure(stats, best_weights, vtx_to_hex, hex_to_vtx, id_to_coord)
plt.savefig("ea_results.png", dpi=150, bbox_inches="tight", facecolor="white")
print("\nResults saved to ea_results.png")
