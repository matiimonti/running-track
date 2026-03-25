"""
Integration tests — require a real OSM graph download.

Run with:
    python -m pytest -m integration -v

Skip with:
    python -m pytest -m "not integration"
"""

import pytest
import osmnx as ox

from app.services.edge_tagger import tag_edges
from app.services.graph_service import compute_edge_grades
from app.services.routing_engine import (
    snap_to_nearest_node,
    generate_loop_with_fallback,
    detect_crossings,
    extract_waypoints,
    extract_elevation_profile,
    compute_route_metadata,
    score_route,
    _path_length_m,
)
from app.services.run_profiles import get_profile


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _build_graph(lat: float, lng: float, dist: int = 2000):
    """Download and fully tag a small graph around a point."""
    G = ox.graph_from_point((lat, lng), dist=dist, network_type="all", retain_all=False)
    # Attach flat elevation (avoids SRTM network call in tests)
    for node_id, data in G.nodes(data=True):
        data["elevation"] = 100.0
    G = compute_edge_grades(G)
    G = tag_edges(G)
    return G


@pytest.fixture(scope="session")
def lyon_graph():
    return _build_graph(lat=45.764, lng=4.835)


@pytest.fixture(scope="session")
def paris_graph():
    return _build_graph(lat=48.856, lng=2.352)


# ── Helpers ───────────────────────────────────────────────────────────────────

_TARGET_M = 2000  # 2 km — achievable in a 2 km radius graph
_BASE_BEARING = 45.0  # fixed bearing so tests are deterministic
_ALL_PROFILES = ["city", "trail", "scenic", "interval"]


def _run_loop(G, lat, lng, profile_name, target_m=_TARGET_M):
    profile = get_profile(profile_name)
    start = snap_to_nearest_node(G, lat, lng)
    path, method = generate_loop_with_fallback(
        G, start, target_m, profile, base_bearing=_BASE_BEARING
    )
    return path, method, profile


# ── Loop generation — Lyon ────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.parametrize("profile_name", _ALL_PROFILES)
def test_loop_generated_for_all_profiles_lyon(lyon_graph, profile_name):
    path, method, _ = _run_loop(lyon_graph, 45.764, 4.835, profile_name)
    assert len(path) >= 2
    assert method in ("loop_10", "loop_20", "out_and_back")


@pytest.mark.integration
@pytest.mark.parametrize("profile_name", _ALL_PROFILES)
def test_loop_distance_within_20_pct_lyon(lyon_graph, profile_name):
    path, method, _ = _run_loop(lyon_graph, 45.764, 4.835, profile_name)
    actual = _path_length_m(lyon_graph, path)
    if method in ("loop_10", "loop_20"):
        assert (
            _TARGET_M * 0.80 <= actual <= _TARGET_M * 1.20
        ), f"{profile_name}: {actual:.0f}m is outside ±20% of {_TARGET_M}m"
    else:
        assert actual > 0, f"{profile_name} out_and_back returned empty path"


@pytest.mark.integration
def test_routes_differ_by_profile_lyon(lyon_graph):
    """Same start + distance must produce different paths for different profiles."""
    paths = {}
    for name in _ALL_PROFILES:
        path, _, _ = _run_loop(lyon_graph, 45.764, 4.835, name)
        paths[name] = path
    # At least two profiles should produce different paths
    unique_paths = {tuple(p) for p in paths.values()}
    assert len(unique_paths) > 1, "All profiles produced identical routes"


# ── Output extraction — Lyon ──────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.parametrize("profile_name", _ALL_PROFILES)
def test_waypoints_non_empty_lyon(lyon_graph, profile_name):
    path, _, _ = _run_loop(lyon_graph, 45.764, 4.835, profile_name)
    waypoints = extract_waypoints(lyon_graph, path)
    assert len(waypoints) >= 2  # at minimum start + end


@pytest.mark.integration
@pytest.mark.parametrize("profile_name", _ALL_PROFILES)
def test_waypoints_start_end_present_lyon(lyon_graph, profile_name):
    path, _, _ = _run_loop(lyon_graph, 45.764, 4.835, profile_name)
    waypoints = extract_waypoints(lyon_graph, path)
    assert waypoints[0].turn == "start"
    assert waypoints[-1].turn == "end"


@pytest.mark.integration
@pytest.mark.parametrize("profile_name", _ALL_PROFILES)
def test_elevation_profile_lyon(lyon_graph, profile_name):
    path, _, _ = _run_loop(lyon_graph, 45.764, 4.835, profile_name)
    profile_points = extract_elevation_profile(lyon_graph, path)
    assert len(profile_points) == len(path)
    distances = [p.distance_m for p in profile_points]
    assert all(distances[i] <= distances[i + 1] for i in range(len(distances) - 1))


@pytest.mark.integration
@pytest.mark.parametrize("profile_name", _ALL_PROFILES)
def test_metadata_lyon(lyon_graph, profile_name):
    path, _, profile = _run_loop(lyon_graph, 45.764, 4.835, profile_name)
    meta = compute_route_metadata(lyon_graph, path, profile)
    assert meta.total_distance_m > 0
    assert meta.elevation_gain_m >= 0
    assert meta.elevation_loss_m >= 0
    assert meta.estimated_time_min > 0
    total_pct = sum(meta.surface_breakdown.values())
    assert total_pct == pytest.approx(100.0, abs=0.5)


@pytest.mark.integration
@pytest.mark.parametrize("profile_name", _ALL_PROFILES)
def test_score_in_bounds_lyon(lyon_graph, profile_name):
    path, _, profile = _run_loop(lyon_graph, 45.764, 4.835, profile_name)
    score = score_route(lyon_graph, path, profile)
    assert 0 <= score <= 100


@pytest.mark.integration
@pytest.mark.parametrize("profile_name", _ALL_PROFILES)
def test_crossing_detection_runs_lyon(lyon_graph, profile_name):
    path, _, _ = _run_loop(lyon_graph, 45.764, 4.835, profile_name)
    warnings = detect_crossings(lyon_graph, path)
    assert isinstance(warnings, list)  # no crash, any result is valid


# ── City-specific surface preferences ────────────────────────────────────────


@pytest.mark.integration
def test_city_avoids_dirt_more_than_trail_lyon(lyon_graph):
    city_path, _, city_profile = _run_loop(lyon_graph, 45.764, 4.835, "city")
    trail_path, _, trail_profile = _run_loop(lyon_graph, 45.764, 4.835, "trail")
    city_meta = compute_route_metadata(lyon_graph, city_path, city_profile)
    trail_meta = compute_route_metadata(lyon_graph, trail_path, trail_profile)
    city_dirt = city_meta.surface_breakdown.get("dirt", 0)
    trail_dirt = trail_meta.surface_breakdown.get("dirt", 0)
    # City profile penalises dirt heavily — should have less dirt than trail
    assert city_dirt <= trail_dirt


# ── Two-city coverage — Paris ─────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.parametrize("profile_name", _ALL_PROFILES)
def test_loop_generated_for_all_profiles_paris(paris_graph, profile_name):
    path, method, _ = _run_loop(paris_graph, 48.856, 2.352, profile_name)
    assert len(path) >= 2
    assert method in ("loop_10", "loop_20", "out_and_back")


@pytest.mark.integration
@pytest.mark.parametrize("profile_name", _ALL_PROFILES)
def test_loop_distance_within_20_pct_paris(paris_graph, profile_name):
    path, method, _ = _run_loop(paris_graph, 48.856, 2.352, profile_name)
    actual = _path_length_m(paris_graph, path)
    if method in ("loop_10", "loop_20"):
        assert (
            _TARGET_M * 0.80 <= actual <= _TARGET_M * 1.20
        ), f"{profile_name}: {actual:.0f}m is outside ±20% of {_TARGET_M}m"
    else:
        assert actual > 0, f"{profile_name} out_and_back returned empty path"


@pytest.mark.integration
@pytest.mark.parametrize("profile_name", _ALL_PROFILES)
def test_metadata_paris(paris_graph, profile_name):
    path, _, profile = _run_loop(paris_graph, 48.856, 2.352, profile_name)
    meta = compute_route_metadata(paris_graph, path, profile)
    assert meta.total_distance_m > 0
    total_pct = sum(meta.surface_breakdown.values())
    assert total_pct == pytest.approx(100.0, abs=0.5)
