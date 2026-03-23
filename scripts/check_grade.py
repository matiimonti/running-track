import sys

sys.path.insert(0, ".")

from app.services.graph_service import load_or_download_graph


if __name__ == "__main__":
    G = load_or_download_graph("lyon", "Lyon, France")

    # Check if grade survived the save/load cycle
    grades = [
        d.get("grade") for u, v, d in G.edges(data=True) if d.get("grade") is not None
    ]
    elevations = [
        d.get("elevation") for n, d in G.nodes(data=True) if d.get("elevation") is not None
    ]

    print(f"Edges with grade: {len(grades)} / {len(G.edges)}")
    print(f"Nodes with elevation: {len(elevations)} / {len(G.nodes)}")

    if grades:
        import numpy as np

        grades_f = [float(g) for g in grades]
        print(f"Grade range: {min(grades_f):.1f}% to {max(grades_f):.1f}%")
        print(f"Avg absolute grade: {np.mean([abs(g) for g in grades_f]):.1f}%")
        print(f"Edges > 8% grade: {sum(1 for g in grades_f if abs(g) > 8)}")
        print(f"Edges > 12% grade: {sum(1 for g in grades_f if abs(g) > 12)}")
