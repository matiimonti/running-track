import sys

sys.path.insert(0, ".")

from app.services.graph_service import load_or_download_graph


if __name__ == "__main__":
    G = load_or_download_graph("lyon", "Lyon, France")

    print(f"\nNodes: {len(G.nodes)}")
    print(f"Edges: {len(G.edges)}")
    print("\nSample edge attributes:")
    u, v, data = next(iter(G.edges(data=True)))
    for key, val in data.items():
        print(f"  {key}: {val}")
