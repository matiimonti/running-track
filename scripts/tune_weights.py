"""
Weight tuning script — generates routes for all 4 profiles from a fixed start
point and prints a comparison table.

Usage:
    python scripts/tune_weights.py
    python scripts/tune_weights.py --city paris --distance 5000
    python scripts/tune_weights.py --city lyon --distance 3000 --bearing 90

Output: one row per profile showing method, distance, surface breakdown, score.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import osmnx as ox
from app.services.edge_tagger import tag_edges
from app.services.graph_service import compute_edge_grades
from app.services.routing_engine import (
    snap_to_nearest_node,
    generate_loop_with_fallback,
    compute_route_metadata,
    score_route,
    _path_length_m,
)
from app.services.run_profiles import get_profile

CITIES = {
    "lyon": (45.764, 4.835),
    "paris": (48.856, 2.352),
    "london": (51.507, -0.128),
}

_GRAPH_CACHE = {}


def load_graph(city: str):
    if city in _GRAPH_CACHE:
        return _GRAPH_CACHE[city]

    lat, lng = CITIES[city]
    print(f"Downloading graph for {city}...")
    G = ox.graph_from_point((lat, lng), dist=3000, network_type="all", retain_all=False)
    for node_id, data in G.nodes(data=True):
        data["elevation"] = 100.0
    G = compute_edge_grades(G)
    G = tag_edges(G)
    _GRAPH_CACHE[city] = G
    print(f"Graph ready: {len(G.nodes)} nodes, {len(G.edges)} edges\n")
    return G


def run_profile(G, lat, lng, profile_name, target_m, base_bearing):
    profile = get_profile(profile_name)
    start = snap_to_nearest_node(G, lat, lng)
    try:
        path, method = generate_loop_with_fallback(
            G, start, target_m, profile, base_bearing=base_bearing
        )
        actual_m = _path_length_m(G, path)
        meta = compute_route_metadata(G, path, profile)
        score = score_route(G, path, profile)
        return {
            "profile": profile_name,
            "method": method,
            "actual_m": round(actual_m),
            "target_m": target_m,
            "pct_of_target": round(actual_m / target_m * 100, 1),
            "score": score,
            "time_min": meta.estimated_time_min,
            "gain_m": round(meta.elevation_gain_m),
            "surfaces": meta.surface_breakdown,
        }
    except ValueError as e:
        return {
            "profile": profile_name,
            "method": "FAILED",
            "error": str(e),
        }


def print_table(results, target_m):
    # Collect all surface types seen across all results
    all_surfaces = sorted(
        {s for r in results if "surfaces" in r for s in r["surfaces"]}
    )

    # Header
    surf_w = 9
    header = (
        f"{'Profile':<10} {'Method':<12} {'Actual m':>8} {'% target':>8} "
        f"{'Score':>6} {'Time':>6} {'Gain':>6}"
    )
    for s in all_surfaces:
        header += f"  {s[:surf_w]:>{surf_w}}"
    print(header)
    print("-" * len(header))

    for r in results:
        if r["method"] == "FAILED":
            print(f"{r['profile']:<10} FAILED — {r.get('error', '')}")
            continue

        row = (
            f"{r['profile']:<10} {r['method']:<12} {r['actual_m']:>8} "
            f"{r['pct_of_target']:>7}% {r['score']:>6} "
            f"{r['time_min']:>5}m {r['gain_m']:>5}m"
        )
        for s in all_surfaces:
            pct = r["surfaces"].get(s, 0.0)
            row += f"  {pct:>{surf_w}.1f}%"
        print(row)

    print()
    print("Surface key:", ", ".join(all_surfaces))
    print()
    print("What to look for:")
    print("  - trail should have more dirt/unpaved/gravel than city")
    print("  - interval should have the most asphalt/paved")
    print("  - scenic should have low primary/secondary road %")
    print("  - score should vary meaningfully across profiles (not all ~75)")
    print("  - method should ideally be loop_10 or loop_20, not out_and_back")


def main():
    parser = argparse.ArgumentParser(description="Tune routing weights")
    parser.add_argument("--city", default="lyon", choices=list(CITIES.keys()))
    parser.add_argument(
        "--distance", type=float, default=3000, help="Target distance in metres"
    )
    parser.add_argument(
        "--bearing",
        type=float,
        default=45.0,
        help="Base bearing in degrees (0=N, 90=E)",
    )
    args = parser.parse_args()

    lat, lng = CITIES[args.city]
    G = load_graph(args.city)

    print(f"City: {args.city} ({lat}, {lng})")
    print(f"Target: {args.distance:.0f}m | Base bearing: {args.bearing}°\n")

    profiles = ["city", "trail", "scenic", "interval"]
    results = [
        run_profile(G, lat, lng, p, args.distance, args.bearing) for p in profiles
    ]

    print_table(results, args.distance)


if __name__ == "__main__":
    main()
