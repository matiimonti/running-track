# app/services/routing_engine.py

import networkx as nx
from app.logging_config import logger
from app.services.validators import haversine_km
from app.services.run_profiles import RunProfile

_DEFAULT_SURFACE_WEIGHT = 2.0  # cost for unknown surface type
_DEFAULT_HIGHWAY_WEIGHT = 1.5  # cost for unknown highway type
_REPETITION_PENALTY = 5.0  # edges already used cost 5× more on return path


def snap_to_nearest_node(G: nx.MultiDiGraph, lat: float, lng: float) -> int:
    """
    Snap a lat/lng coordinate to the nearest connected node in the graph.

    Filters out isolated nodes (degree == 0) since A* cannot route from them.
    Uses haversine distance so accuracy holds at any scale.

    Returns the node_id (int) of the nearest connected node.
    Raises ValueError if the graph has no connected nodes at all.
    """
    best_node = None
    best_dist = float("inf")

    for node_id, data in G.nodes(data=True):
        node_lat = data.get("y")
        node_lng = data.get("x")

        # Skip nodes with missing coordinates
        if node_lat is None or node_lng is None:
            continue

        # Skip isolated nodes — A* can't start or end there
        if G.degree(node_id) == 0:
            continue

        dist = haversine_km(lat, lng, node_lat, node_lng)
        if dist < best_dist:
            best_dist = dist
            best_node = node_id

    if best_node is None:
        raise ValueError(
            f"No connected node found in graph near ({lat}, {lng}). "
            "The graph may be empty or all nodes are isolated."
        )

    logger.info(
        "snapped_to_node",
        node=best_node,
        dist_m=round(best_dist * 1000, 1),
        lat=lat,
        lng=lng,
    )
    return best_node


def compute_edge_cost(edge_data: dict, profile: RunProfile) -> float:
    """
    Compute the routing cost for a single edge given a run type profile.

    Returns float("inf") for non-runnable edges so A* never traverses them.
    Cost = length × surface_weight × highway_weight × grade_multiplier
    """
    # Hard block on non-runnable edges
    if not edge_data.get("is_runnable", False):
        return float("inf")

    # Hard block on steps if profile doesn't allow them
    highway_type = edge_data.get("highway_type", "unknown")
    if highway_type == "steps" and not profile.allow_steps:
        return float("inf")

    length = edge_data.get("length", 1.0)
    surface_type = edge_data.get("surface_type", "unknown")
    grade = abs(edge_data.get("grade", 0.0))  # use absolute grade — both up and downhill add cost

    surface_weight = profile.surface_weights.get(surface_type, _DEFAULT_SURFACE_WEIGHT)
    highway_weight = profile.highway_weights.get(highway_type, _DEFAULT_HIGHWAY_WEIGHT)

    # Grade multiplier: flat = 1.0, rises sharply above max_comfortable_grade
    if grade <= profile.max_comfortable_grade:
        grade_multiplier = 1.0 + (profile.grade_penalty_per_pct * grade)
    else:
        # Steep penalty above comfort threshold
        grade_multiplier = (
            1.0
            + (profile.grade_penalty_per_pct * profile.max_comfortable_grade)
            + (profile.grade_penalty_per_pct * 3.0 * (grade - profile.max_comfortable_grade))
        )

    return length * surface_weight * highway_weight * grade_multiplier



def build_used_edges(path: list[int]) -> set[tuple[int, int]]:
    """
    Build a set of (u, v) edge tuples from a node path.
    Used to penalise re-traversal on the return leg.
    """
    return {(path[i], path[i + 1]) for i in range(len(path) - 1)}


def make_cost_fn(profile: RunProfile, used_edges: set[tuple[int, int]] | None = None):
    """
    Return a weight function compatible with nx.astar_path(weight=...).

    nx calls it as weight(u, v, edge_data) for each candidate edge.
    If used_edges is provided, edges already in the set get a ×5 penalty.
    """
    def cost_fn(u: int, v: int, edge_data: dict) -> float:
        base = compute_edge_cost(edge_data, profile)
        if used_edges and (u, v) in used_edges:
            return base * _REPETITION_PENALTY
        return base

    return cost_fn






