import math
import random
from dataclasses import dataclass
import networkx as nx
from app.logging_config import logger
from app.services.validators import haversine_km
from app.services.run_profiles import RunProfile

_DEFAULT_SURFACE_WEIGHT = 2.0  # cost for unknown surface type
_DEFAULT_HIGHWAY_WEIGHT = 1.5  # cost for unknown highway type
_REPETITION_PENALTY = 5.0  # edges already used cost 5× more on return path
_EARTH_RADIUS_KM = 6371.0
_FALLBACK_ATTEMPTS_PER_TOLERANCE = 3

# Highway types that indicate a dangerous road crossing
_CROSSING_HIGHWAY_TYPES = {"motorway", "trunk", "motorway_link", "trunk_link"}


@dataclass
class CrossingWarning:
    node_id: int
    lat: float
    lng: float
    highway_type: str  # the dangerous road type adjacent to this node


def detect_crossings(G: nx.MultiDiGraph, path: list[int]) -> list[CrossingWarning]:
    """
    Flag nodes in the path that are adjacent to motorway or trunk edges.

    In OSM, a node shared between a footway/path and a motorway is a crossing
    point — the runner physically crosses or joins a dangerous road there.

    Returns a list of CrossingWarning (one per flagged node, deduplicated).
    """
    warnings: list[CrossingWarning] = []
    seen: set[int] = set()

    for node in path:
        if node in seen:
            continue

        for neighbor in G.successors(node):
            edge_data = min(
                G[node][neighbor].values(),
                key=lambda d: d.get("length", float("inf")),
            )
            highway_type = edge_data.get("highway_type", "unknown")
            if highway_type in _CROSSING_HIGHWAY_TYPES:
                warnings.append(
                    CrossingWarning(
                        node_id=node,
                        lat=G.nodes[node]["y"],
                        lng=G.nodes[node]["x"],
                        highway_type=highway_type,
                    )
                )
                seen.add(node)
                break  # one warning per node is enough

    if warnings:
        logger.warning(
            "crossing_detected",
            count=len(warnings),
            nodes=[w.node_id for w in warnings],
        )
    else:
        logger.info("crossing_check_clean", path_nodes=len(path))

    return warnings


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


def _project_coordinate(lat: float, lng: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    """
    Project a coordinate distance_km away from (lat, lng) along bearing_deg.
    bearing_deg: 0 = north, 90 = east, 180 = south, 270 = west.
    Returns (new_lat, new_lng).
    """
    bearing = math.radians(bearing_deg)
    lat_r = math.radians(lat)
    lng_r = math.radians(lng)
    d = distance_km / _EARTH_RADIUS_KM

    new_lat_r = math.asin(
        math.sin(lat_r) * math.cos(d)
        + math.cos(lat_r) * math.sin(d) * math.cos(bearing)
    )
    new_lng_r = lng_r + math.atan2(
        math.sin(bearing) * math.sin(d) * math.cos(lat_r),
        math.cos(d) - math.sin(lat_r) * math.sin(new_lat_r),
    )
    return math.degrees(new_lat_r), math.degrees(new_lng_r)

def _haversine_cost_heuristic(G: nx.MultiDiGraph, goal: int):
    """
    Return an A* heuristic function: estimated cost from any node to goal.
    Uses haversine distance scaled by minimum edge cost (1.0) so it's admissible.
    """
    goal_lat = G.nodes[goal]["y"]
    goal_lng = G.nodes[goal]["x"]

    def heuristic(u: int, v: int) -> float:
        node_lat = G.nodes[u]["y"]
        node_lng = G.nodes[u]["x"]
        return haversine_km(node_lat, node_lng, goal_lat, goal_lng) * 1000  # convert to metres

    return heuristic

def _path_length_m(G: nx.MultiDiGraph, path: list[int]) -> float:
    """Sum edge lengths along a node path in metres."""
    total = 0.0
    for i in range(len(path) - 1):
        edge_data = min(G[path[i]][path[i + 1]].values(), key=lambda d: d.get("length", 0))
        total += edge_data.get("length", 0.0)
    return total

def generate_loop(
    G: nx.MultiDiGraph,
    start_node: int,
    target_distance_m: float,
    profile: RunProfile,
    bearing_deg: float | None = None,
) -> list[int]:
    """
    Generate a loop route using two A* passes.

    Pass 1: start → midpoint (no repetition penalty)
    Pass 2: midpoint → start (with repetition penalty on outbound edges)

    Returns the combined node path (start node appears at both ends).
    Raises ValueError if no valid loop is found within ±10% of target distance.
    """
    if bearing_deg is None:
        bearing_deg = random.uniform(0, 360)

    start_lat = G.nodes[start_node]["y"]
    start_lng = G.nodes[start_node]["x"]

    # Project midpoint coordinate D/2 away along bearing
    mid_lat, mid_lng = _project_coordinate(
        start_lat, start_lng, bearing_deg, (target_distance_m / 1000) / 2
    )
    mid_node = snap_to_nearest_node(G, mid_lat, mid_lng)

    if mid_node == start_node:
        raise ValueError("Midpoint snapped to start node — graph may be too sparse for this distance.")

    # Pass 1: outbound (no penalty)
    outbound_cost_fn = make_cost_fn(profile, used_edges=None)
    try:
        outbound = nx.astar_path(
            G,
            start_node,
            mid_node,
            heuristic=_haversine_cost_heuristic(G, mid_node),
            weight=outbound_cost_fn,
        )
    except nx.NetworkXNoPath:
        raise ValueError(f"No path found from start to midpoint node {mid_node}.")

    # Pass 2: return with anti-repetition penalty
    used = build_used_edges(outbound)
    return_cost_fn = make_cost_fn(profile, used_edges=used)
    try:
        return_path = nx.astar_path(
            G,
            mid_node,
            start_node,
            heuristic=_haversine_cost_heuristic(G, start_node),
            weight=return_cost_fn,
        )
    except nx.NetworkXNoPath:
        raise ValueError(f"No return path found from midpoint node {mid_node} to start.")

    # Combine: outbound + return (drop duplicate midpoint node at the join)
    full_path = outbound + return_path[1:]

    # Validate distance within ±10%
    actual_m = _path_length_m(G, full_path)
    lower = target_distance_m * 0.9
    upper = target_distance_m * 1.1

    if not (lower <= actual_m <= upper):
        raise ValueError(
            f"Loop distance {actual_m:.0f}m is outside ±10% of target {target_distance_m:.0f}m."
        )

    logger.info(
        "loop_generated",
        start=start_node,
        mid=mid_node,
        nodes=len(full_path),
        actual_m=round(actual_m),
        target_m=round(target_distance_m),
        bearing=round(bearing_deg, 1),
    )
    return full_path

def _out_and_back(
    G: nx.MultiDiGraph,
    start_node: int,
    target_distance_m: float,
    profile: RunProfile,
) -> list[int]:
    """
    Generate an out-and-back route as a last resort.
    Finds the node closest to D/4 away (so the full out-and-back ≈ D/2 * 2 = D),
    then returns outbound + reversed outbound.
    """
    start_lat = G.nodes[start_node]["y"]
    start_lng = G.nodes[start_node]["x"]

    bearing = random.uniform(0, 360)
    mid_lat, mid_lng = _project_coordinate(
        start_lat, start_lng, bearing, (target_distance_m / 1000) / 4
    )
    mid_node = snap_to_nearest_node(G, mid_lat, mid_lng)

    cost_fn = make_cost_fn(profile, used_edges=None)
    try:
        outbound = nx.astar_path(G, start_node, mid_node, weight=cost_fn)
    except nx.NetworkXNoPath:
        raise ValueError("Out-and-back fallback failed: no path found.")

    return outbound + list(reversed(outbound[:-1]))

def generate_loop_with_fallback(
    G: nx.MultiDiGraph,
    start_node: int,
    target_distance_m: float,
    profile: RunProfile,
    base_bearing: float | None = None,
) -> tuple[list[int], str]:
    """
    Attempt loop generation with a progressive fallback ladder:
      1. ±10% tolerance, 3 bearing attempts
      2. ±20% tolerance, 3 bearing attempts
      3. Out-and-back
      4. Raise with a clear message

    base_bearing: if provided, profile.bearing_offset is added to it so that
    different run types fan out in different directions from the same start.
    If None, a random base bearing is used each attempt.

    Returns (path, method) where method is one of:
      "loop_10", "loop_20", "out_and_back"
    """
    start_lat = G.nodes[start_node]["y"]
    start_lng = G.nodes[start_node]["x"]

    def _pick_bearing() -> float:
        """Pick a bearing: apply profile offset to base (or random if no base)."""
        b = base_bearing if base_bearing is not None else random.uniform(0, 360)
        return (b + profile.bearing_offset) % 360

    def _try_loop(tolerance: float) -> list[int] | None:
        bearing = _pick_bearing()
        mid_dist_km = (target_distance_m / 1000) * profile.midpoint_factor

        mid_lat, mid_lng = _project_coordinate(start_lat, start_lng, bearing, mid_dist_km)
        mid_node = snap_to_nearest_node(G, mid_lat, mid_lng)
        if mid_node == start_node:
            return None

        try:
            outbound = nx.astar_path(
                G, start_node, mid_node,
                heuristic=_haversine_cost_heuristic(G, mid_node),
                weight=make_cost_fn(profile),
            )
            used = build_used_edges(outbound)
            return_path = nx.astar_path(
                G, mid_node, start_node,
                heuristic=_haversine_cost_heuristic(G, start_node),
                weight=make_cost_fn(profile, used_edges=used),
            )
        except nx.NetworkXNoPath:
            return None

        full_path = outbound + return_path[1:]
        actual_m = _path_length_m(G, full_path)
        lower = target_distance_m * (1 - tolerance)
        upper = target_distance_m * (1 + tolerance)

        return full_path if lower <= actual_m <= upper else None

    # Stage 1: ±10%
    for _ in range(_FALLBACK_ATTEMPTS_PER_TOLERANCE):
        path = _try_loop(0.10)
        if path:
            logger.info("loop_found", method="loop_10", target_m=round(target_distance_m), profile=profile.name)
            return path, "loop_10"

    # Stage 2: ±20%
    for _ in range(_FALLBACK_ATTEMPTS_PER_TOLERANCE):
        path = _try_loop(0.20)
        if path:
            logger.info("loop_found", method="loop_20", target_m=round(target_distance_m), profile=profile.name)
            return path, "loop_20"

    # Stage 3: out-and-back
    try:
        path = _out_and_back(G, start_node, target_distance_m, profile)
        logger.info("loop_found", method="out_and_back", target_m=round(target_distance_m), profile=profile.name)
        return path, "out_and_back"
    except ValueError:
        pass

    raise ValueError(
        f"Could not generate a route of {target_distance_m / 1000:.1f} km "
        f"from node {start_node}. The graph may be too sparse or the area "
        f"too constrained (dead ends, water, etc.)."
    )


def extend_path_to_target(
    G: nx.MultiDiGraph,
    path: list[int],
    target_distance_m: float,
    profile: RunProfile,
) -> list[int]:
    """
    Extend a short path by appending low-cost edges until it reaches
    the lower bound of the target distance range (target × 0.9).

    The extension walks greedily from the last node, picking the
    cheapest outgoing edge not already in the path. Then re-routes
    back to start via A*.

    Returns the extended path, or the original if no extension is needed
    or possible.
    """
    lower = target_distance_m * 0.9
    actual_m = _path_length_m(G, path)

    if actual_m >= lower:
        return path  # already long enough, nothing to do

    start_node = path[0]
    visited_nodes = set(path)
    extended = list(path)

    while _path_length_m(G, extended) < lower:
        current = extended[-1]
        best_next = None
        best_cost = float("inf")

        for neighbor in G.successors(current):
            if neighbor in visited_nodes:
                continue
            edge_data = min(
                G[current][neighbor].values(),
                key=lambda d: d.get("length", float("inf")),
            )
            cost = compute_edge_cost(edge_data, profile)
            if cost < best_cost:
                best_cost = cost
                best_next = neighbor

        if best_next is None:
            # No unvisited neighbors — can't extend further
            break

        extended.append(best_next)
        visited_nodes.add(best_next)

    # Re-route from current end back to start
    if extended[-1] != start_node:
        used = build_used_edges(extended)
        return_cost_fn = make_cost_fn(profile, used_edges=used)
        try:
            return_leg = nx.astar_path(
                G,
                extended[-1],
                start_node,
                heuristic=_haversine_cost_heuristic(G, start_node),
                weight=return_cost_fn,
            )
            extended = extended + return_leg[1:]
        except nx.NetworkXNoPath:
            logger.warning(
                "extension_return_failed",
                node=extended[-1],
                start=start_node,
            )
            return path  # return original if we can't close the loop

    logger.info(
        "path_extended",
        original_m=round(actual_m),
        extended_m=round(_path_length_m(G, extended)),
        target_m=round(target_distance_m),
    )
    return extended


@dataclass
class Waypoint:
    lat: float
    lng: float
    street_name: str        # OSM name of the outgoing edge, or "" if unnamed
    turn: str               # "start" | "straight" | "slight_right" | "right" |
                            # "sharp_right" | "u_turn" | "sharp_left" | "left" |
                            # "slight_left" | "end"
    distance_from_prev_m: float   # length of the edge arriving at this node
    elevation_m: float      # metres above sea level


def _bearing(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Compute the forward bearing in degrees [0, 360) from point 1 to point 2."""
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    d_lng = math.radians(lng2 - lng1)
    x = math.sin(d_lng) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lng)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _turn_direction(incoming_bearing: float, outgoing_bearing: float) -> str:
    """Classify a turn from the change in bearing."""
    delta = (outgoing_bearing - incoming_bearing + 360) % 360
    if delta < 22.5 or delta >= 337.5:
        return "straight"
    elif delta < 67.5:
        return "slight_right"
    elif delta < 112.5:
        return "right"
    elif delta < 157.5:
        return "sharp_right"
    elif delta < 202.5:
        return "u_turn"
    elif delta < 247.5:
        return "sharp_left"
    elif delta < 292.5:
        return "left"
    else:
        return "slight_left"


def _edge_name(G: nx.MultiDiGraph, u: int, v: int) -> str:
    """Return the OSM name of the cheapest (shortest) edge between u and v."""
    edge_data = min(G[u][v].values(), key=lambda d: d.get("length", float("inf")))
    name = edge_data.get("name", "")
    if isinstance(name, list):
        name = name[0]
    return str(name) if name else ""


def extract_waypoints(G: nx.MultiDiGraph, path: list[int]) -> list[Waypoint]:
    """
    Extract turn-by-turn waypoints from a node path.

    A waypoint is emitted at:
      - The start node
      - Any node where the turn angle exceeds 22.5° (not "straight")
      - Any node where the street name changes
      - The end node

    Each waypoint carries: lat/lng, street name of the outgoing edge,
    turn direction, cumulative distance from the previous waypoint, and elevation.
    """
    if len(path) < 2:
        raise ValueError("Path must have at least 2 nodes to extract waypoints.")

    waypoints: list[Waypoint] = []

    # Accumulated distance since the last emitted waypoint
    dist_since_last = 0.0

    # --- Start node ---
    start = path[0]
    waypoints.append(Waypoint(
        lat=G.nodes[start]["y"],
        lng=G.nodes[start]["x"],
        street_name=_edge_name(G, path[0], path[1]),
        turn="start",
        distance_from_prev_m=0.0,
        elevation_m=G.nodes[start].get("elevation", 0.0),
    ))

    # Walk interior nodes
    for i in range(1, len(path) - 1):
        prev_node = path[i - 1]
        curr_node = path[i]
        next_node = path[i + 1]

        # Accumulate distance of edge arriving at curr_node
        edge_in = min(G[prev_node][curr_node].values(), key=lambda d: d.get("length", float("inf")))
        dist_since_last += edge_in.get("length", 0.0)

        # Compute bearings
        in_bearing = _bearing(
            G.nodes[prev_node]["y"], G.nodes[prev_node]["x"],
            G.nodes[curr_node]["y"], G.nodes[curr_node]["x"],
        )
        out_bearing = _bearing(
            G.nodes[curr_node]["y"], G.nodes[curr_node]["x"],
            G.nodes[next_node]["y"], G.nodes[next_node]["x"],
        )

        turn = _turn_direction(in_bearing, out_bearing)
        outgoing_name = _edge_name(G, curr_node, next_node)
        incoming_name = _edge_name(G, prev_node, curr_node)

        # Emit if turn is meaningful or street name changes
        if turn != "straight" or outgoing_name != incoming_name:
            waypoints.append(Waypoint(
                lat=G.nodes[curr_node]["y"],
                lng=G.nodes[curr_node]["x"],
                street_name=outgoing_name,
                turn=turn,
                distance_from_prev_m=round(dist_since_last, 1),
                elevation_m=G.nodes[curr_node].get("elevation", 0.0),
            ))
            dist_since_last = 0.0

    # --- End node ---
    end = path[-1]
    last_edge = min(G[path[-2]][end].values(), key=lambda d: d.get("length", float("inf")))
    dist_since_last += last_edge.get("length", 0.0)

    waypoints.append(Waypoint(
        lat=G.nodes[end]["y"],
        lng=G.nodes[end]["x"],
        street_name="",
        turn="end",
        distance_from_prev_m=round(dist_since_last, 1),
        elevation_m=G.nodes[end].get("elevation", 0.0),
    ))

    logger.info("waypoints_extracted", count=len(waypoints), path_nodes=len(path))
    return waypoints


@dataclass
class ElevationPoint:
    distance_m: float   # cumulative distance along the route at this point
    elevation_m: float  # metres above sea level


def extract_elevation_profile(G: nx.MultiDiGraph, path: list[int]) -> list[ElevationPoint]:
    """
    Build an elevation profile as a list of (distance_along_route, elevation) pairs.

    Every node in the path is included so the chart has full resolution.
    Cumulative distance is computed from edge lengths.
    """
    if len(path) < 1:
        raise ValueError("Path must have at least 1 node.")

    profile: list[ElevationPoint] = []
    cumulative_m = 0.0

    profile.append(ElevationPoint(
        distance_m=0.0,
        elevation_m=G.nodes[path[0]].get("elevation", 0.0),
    ))

    for i in range(1, len(path)):
        edge_data = min(
            G[path[i - 1]][path[i]].values(),
            key=lambda d: d.get("length", float("inf")),
        )
        cumulative_m += edge_data.get("length", 0.0)
        profile.append(ElevationPoint(
            distance_m=round(cumulative_m, 1),
            elevation_m=G.nodes[path[i]].get("elevation", 0.0),
        ))

    logger.info(
        "elevation_profile_extracted",
        points=len(profile),
        total_m=round(cumulative_m),
    )
    return profile


@dataclass
class RouteMetadata:
    total_distance_m: float
    elevation_gain_m: float          # sum of all uphill segments
    elevation_loss_m: float          # sum of all downhill segments (positive value)
    estimated_time_min: float        # based on profile pace
    surface_breakdown: dict          # surface_type -> % of total distance (0–100)


def compute_route_metadata(
    G: nx.MultiDiGraph,
    path: list[int],
    profile: RunProfile,
) -> RouteMetadata:
    """
    Compute summary statistics for a generated route.

    Elevation gain/loss uses per-edge elevation_start/elevation_end attributes
    attached by graph_service.compute_edge_grades.
    Surface breakdown sums edge lengths by surface_type and converts to %.
    Estimated time uses profile.pace_min_per_km × total distance.
    """
    total_distance_m = 0.0
    elevation_gain_m = 0.0
    elevation_loss_m = 0.0
    surface_lengths: dict[str, float] = {}

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = min(G[u][v].values(), key=lambda d: d.get("length", float("inf")))

        length = edge_data.get("length", 0.0)
        total_distance_m += length

        # Elevation gain/loss
        elev_start = edge_data.get("elevation_start", G.nodes[u].get("elevation", 0.0))
        elev_end = edge_data.get("elevation_end", G.nodes[v].get("elevation", 0.0))
        delta = elev_end - elev_start
        if delta > 0:
            elevation_gain_m += delta
        else:
            elevation_loss_m += abs(delta)

        # Surface breakdown
        surface = edge_data.get("surface_type", "unknown")
        surface_lengths[surface] = surface_lengths.get(surface, 0.0) + length

    # Convert surface lengths to percentages
    surface_breakdown = {
        surface: round((length / total_distance_m) * 100, 1) if total_distance_m > 0 else 0.0
        for surface, length in surface_lengths.items()
    }

    estimated_time_min = round((total_distance_m / 1000) * profile.pace_min_per_km, 1)

    logger.info(
        "route_metadata_computed",
        distance_m=round(total_distance_m),
        gain_m=round(elevation_gain_m),
        loss_m=round(elevation_loss_m),
        time_min=estimated_time_min,
        profile=profile.name,
    )

    return RouteMetadata(
        total_distance_m=round(total_distance_m, 1),
        elevation_gain_m=round(elevation_gain_m, 1),
        elevation_loss_m=round(elevation_loss_m, 1),
        estimated_time_min=estimated_time_min,
        surface_breakdown=surface_breakdown,
    )


def score_route(G: nx.MultiDiGraph, path: list[int], profile: RunProfile) -> int:
    """
    Score a route from 0 to 100 based on three components:

      Surface match  (40 pts) — how well edge surfaces align with the profile.
                                 Weight 1.0 = perfect, higher = worse.
      Grade comfort  (40 pts) — fraction of route within max_comfortable_grade.
      Strava popularity (20 pts) — edge-level popularity score if available,
                                    otherwise neutral (10 pts).

    Returns an integer in [0, 100].
    """
    _SURFACE_WEIGHT_CAP = 5.0   # weights above this are treated as worst-case
    _WEIGHT_SURFACE = 40
    _WEIGHT_GRADE = 40
    _WEIGHT_POPULARITY = 20

    min_surface_w = min(profile.surface_weights.values())

    surface_score_sum = 0.0
    grade_score_sum = 0.0
    popularity_score_sum = 0.0
    total_length = 0.0
    popularity_data_present = False

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = min(G[u][v].values(), key=lambda d: d.get("length", float("inf")))
        length = edge_data.get("length", 0.0)
        if length <= 0:
            continue
        total_length += length

        # Surface score: normalize weight to [0, 1] where 1 = best match
        surface_type = edge_data.get("surface_type", "unknown")
        surface_w = profile.surface_weights.get(surface_type, _DEFAULT_SURFACE_WEIGHT)
        surface_w = min(surface_w, _SURFACE_WEIGHT_CAP)
        surface_score = 1.0 - (surface_w - min_surface_w) / (_SURFACE_WEIGHT_CAP - min_surface_w)
        surface_score_sum += surface_score * length

        # Grade score: 1.0 at or below comfort threshold, decays above it
        grade = abs(edge_data.get("grade", 0.0))
        if grade <= profile.max_comfortable_grade:
            grade_score = 1.0
        else:
            overage = grade - profile.max_comfortable_grade
            grade_score = max(0.0, 1.0 - overage / profile.max_comfortable_grade)
        grade_score_sum += grade_score * length

        # Strava popularity: normalised to [0, 1], capped at 1000 hits
        popularity = edge_data.get("popularity")
        if popularity is not None:
            popularity_data_present = True
            popularity_score_sum += min(float(popularity) / 1000.0, 1.0) * length

    if total_length == 0:
        return 0

    surface_component = (surface_score_sum / total_length) * _WEIGHT_SURFACE
    grade_component = (grade_score_sum / total_length) * _WEIGHT_GRADE

    if popularity_data_present:
        popularity_component = (popularity_score_sum / total_length) * _WEIGHT_POPULARITY
    else:
        popularity_component = _WEIGHT_POPULARITY * 0.5  # neutral when no data

    raw = surface_component + grade_component + popularity_component
    score = max(0, min(100, round(raw)))

    logger.info(
        "route_scored",
        score=score,
        surface_pts=round(surface_component, 1),
        grade_pts=round(grade_component, 1),
        popularity_pts=round(popularity_component, 1),
        profile=profile.name,
    )
    return score
