import sys

sys.path.insert(0, ".")

import osmnx as ox
import matplotlib.pyplot as plt
from app.services.graph_service import load_or_download_graph
from collections import Counter


if __name__ == "__main__":
    G = load_or_download_graph("lyon", "Lyon, France")

    # Count highway types present
    highway_types = []
    for u, v, data in G.edges(data=True):
        hw = data.get("highway", "unknown")
        if isinstance(hw, list):
            highway_types.extend(hw)
        else:
            highway_types.append(hw)

    counts = Counter(highway_types)
    print("Highway types in graph:")
    for hw_type, count in sorted(counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {hw_type:<25} {count:>6}")

    # Check runnable edge types are present
    runnable = ["footway", "path", "track", "cycleway", "pedestrian", "steps"]
    print("\nRunnable path types confirmed:")
    for r in runnable:
        present = counts.get(r, 0)
        status = "YES" if present > 0 else "MISSING"
        print(f"  {r:<20} {status} ({present} edges)")

    # Plot the graph coloured by highway type
    print("\nGenerating map plot (this may take 20-30 seconds)...")
    fig, ax = ox.plot_graph(
        G,
        figsize=(14, 14),
        node_size=0,
        edge_linewidth=0.4,
        edge_color="#1D9E75",
        bgcolor="black",
        show=False,
        close=False,
    )
    ax.set_title("Lyon OSM Graph — all path types", color="white", fontsize=14)
    plt.savefig("data/lyon_graph.png", dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    print("Map saved to data/lyon_graph.png")
