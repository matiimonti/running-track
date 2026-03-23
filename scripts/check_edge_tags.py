import sys

sys.path.insert(0, ".")

from app.services.graph_service import load_or_download_graph
from collections import Counter

G = load_or_download_graph("lyon", "Lyon, France")

surfaces = Counter()
sources = Counter()
runnable = 0

for u, v, data in G.edges(data=True):
    surfaces[data.get("surface_type", "missing")] += 1
    sources[data.get("surface_source", "missing")] += 1
    if data.get("is_runnable"):
        runnable += 1

total = len(G.edges)

print(f"\nSurface type breakdown ({total} total edges):")
for s, count in sorted(surfaces.items(), key=lambda x: -x[1]):
    print(f"  {s:<15} {count:>6}  ({round(count/total*100,1)}%)")

print("\nSurface source:")
for s, count in sorted(sources.items(), key=lambda x: -x[1]):
    print(f"  {s:<20} {count:>6}")

print(f"\nRunnable edges: {runnable} / {total} ({round(runnable/total*100,1)}%)")

# Show a sample edge with all attributes
print("\nSample fully-tagged edge:")
for u, v, data in G.edges(data=True):
    if data.get("surface_source") == "osm":
        for k, val in data.items():
            print(f"  {k}: {val}")
        break
