import sys

sys.path.insert(0, ".")

from app.services.graph_service import load_or_download_graph
from app.services.graph_cache import get_cache_status
from collections import Counter

CITIES_TO_CHECK = [("lyon", "Lyon, France")]

for city_slug, place_name in CITIES_TO_CHECK:
    print(f"\n{'='*50}")
    print(f"Data quality report: {city_slug}")
    print(f"{'='*50}")

    G = load_or_download_graph(city_slug, place_name)

    total_edges = len(G.edges)
    total_nodes = len(G.nodes)

    # Check node attributes
    nodes_missing_elevation = sum(
        1 for n, d in G.nodes(data=True) if d.get("elevation") is None
    )

    # Check edge attributes
    missing = {
        "grade": 0,
        "surface_type": 0,
        "highway_type": 0,
        "elevation_start": 0,
        "elevation_end": 0,
    }
    surface_sources = Counter()
    surface_types = Counter()

    for u, v, data in G.edges(data=True):
        for attr in missing:
            if data.get(attr) is None:
                missing[attr] += 1
        surface_sources[data.get("surface_source", "missing")] += 1
        surface_types[data.get("surface_type", "missing")] += 1

    print(f"\nNodes: {total_nodes}")
    print(
        f"  Missing elevation: {nodes_missing_elevation} ({round(nodes_missing_elevation/total_nodes*100,1)}%)"
    )

    print(f"\nEdges: {total_edges}")
    for attr, count in missing.items():
        pct = round(count / total_edges * 100, 1)
        status = "OK" if count == 0 else "WARN" if pct < 5 else "FAIL"
        print(f"  [{status}] Missing {attr}: {count} ({pct}%)")

    print("\nSurface source breakdown:")
    for source, count in sorted(surface_sources.items(), key=lambda x: -x[1]):
        print(f"  {source:<20} {count:>6} ({round(count/total_edges*100,1)}%)")

    print("\nSurface type breakdown:")
    for surface, count in sorted(surface_types.items(), key=lambda x: -x[1]):
        print(f"  {surface:<15} {count:>6} ({round(count/total_edges*100,1)}%)")

    print("\nCache status:")
    for city, info in get_cache_status().items():
        print(
            f"  {city}: built {info['built_at']}, age {info['age_days']} days, stale={info['stale']}"
        )
