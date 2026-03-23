import pytest
import networkx as nx
from unittest.mock import patch
from app.services.graph_service import (
    attach_elevation,
    compute_edge_grades,
    tag_edges,
)
from app.services.run_profiles import get_profile
from app.services.validators import validate_route_request
from app.services.city_registry import find_nearest_city, get_city


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_simple_graph() -> nx.MultiDiGraph:
    """Create a minimal graph for testing — 3 nodes, 2 edges."""
    G = nx.MultiDiGraph()
    G.add_node(1, x=4.835, y=45.764)  # Lyon centre
    G.add_node(2, x=4.836, y=45.765)
    G.add_node(3, x=4.837, y=45.763)
    G.add_edge(1, 2, highway="footway", length=150.0)
    G.add_edge(2, 3, highway="path", length=80.0)
    return G


# ── Elevation tests ───────────────────────────────────────────────────────────


def test_attach_elevation_adds_to_all_nodes():
    G = make_simple_graph()
    with patch("app.services.graph_service.elevation_data") as mock_elev:
        mock_elev.get_elevation.return_value = 200.0
        G = attach_elevation(G, "test")
    for node_id, data in G.nodes(data=True):
        assert "elevation" in data
        assert data["elevation"] == 200.0


def test_attach_elevation_handles_none():
    """Nodes with no SRTM data should get elevation=0.0, not crash."""
    G = make_simple_graph()
    with patch("app.services.graph_service.elevation_data") as mock_elev:
        mock_elev.get_elevation.return_value = None
        G = attach_elevation(G, "test")
    for node_id, data in G.nodes(data=True):
        assert data["elevation"] == 0.0


# ── Grade tests ───────────────────────────────────────────────────────────────


def test_compute_edge_grades_flat():
    """Flat edges (same elevation) should have grade 0."""
    G = make_simple_graph()
    for node_id in G.nodes:
        G.nodes[node_id]["elevation"] = 100.0
    G = compute_edge_grades(G)
    for u, v, data in G.edges(data=True):
        assert data["grade"] == 0.0


def test_compute_edge_grades_uphill():
    G = make_simple_graph()
    G.nodes[1]["elevation"] = 100.0
    G.nodes[2]["elevation"] = 115.0  # 15m rise over 150m = 10% grade
    G.nodes[3]["elevation"] = 115.0
    G = compute_edge_grades(G)
    grades = [d["grade"] for u, v, d in G.edges(data=True)]
    assert any(g > 0 for g in grades)


def test_grade_clamped_to_30():
    """Extremely short edges should not produce grades beyond ±30%."""
    G = nx.MultiDiGraph()
    G.add_node(1, x=4.835, y=45.764, elevation=0.0)
    G.add_node(2, x=4.836, y=45.765, elevation=500.0)
    G.add_edge(1, 2, highway="footway", length=1.0)  # 1m length, 500m rise
    G = compute_edge_grades(G)
    for u, v, data in G.edges(data=True):
        assert data["grade"] == 0.0  # too short, gets 0


def test_grade_has_elevation_start_end():
    G = make_simple_graph()
    for node_id in G.nodes:
        G.nodes[node_id]["elevation"] = 180.0
    G = compute_edge_grades(G)
    for u, v, data in G.edges(data=True):
        assert "elevation_start" in data
        assert "elevation_end" in data


# ── Surface tagging tests ─────────────────────────────────────────────────────


def test_tag_edges_adds_surface_type():
    G = make_simple_graph()
    for node_id in G.nodes:
        G.nodes[node_id]["elevation"] = 100.0
    G = compute_edge_grades(G)
    G = tag_edges(G)
    for u, v, data in G.edges(data=True):
        assert "surface_type" in data
        assert "highway_type" in data
        assert "surface_source" in data
        assert "is_runnable" in data


def test_tag_edges_osm_surface_used_when_present():
    G = nx.MultiDiGraph()
    G.add_node(1, x=4.835, y=45.764, elevation=100.0)
    G.add_node(2, x=4.836, y=45.765, elevation=100.0)
    G.add_edge(1, 2, highway="residential", surface="asphalt", length=100.0)
    G = compute_edge_grades(G)
    G = tag_edges(G)
    u, v, data = next(iter(G.edges(data=True)))
    assert data["surface_type"] == "asphalt"
    assert data["surface_source"] == "osm"


def test_tag_edges_fallback_when_no_surface():
    G = make_simple_graph()  # no surface tags
    for node_id in G.nodes:
        G.nodes[node_id]["elevation"] = 100.0
    G = compute_edge_grades(G)
    G = tag_edges(G)
    for u, v, data in G.edges(data=True):
        assert data["surface_source"] == "fallback"
        assert data["surface_type"] != "missing"


def test_tag_edges_runnable_flag():
    G = nx.MultiDiGraph()
    G.add_node(1, x=4.835, y=45.764, elevation=100.0)
    G.add_node(2, x=4.836, y=45.765, elevation=100.0)
    G.add_node(3, x=4.837, y=45.763, elevation=100.0)
    G.add_edge(1, 2, highway="footway", length=100.0)
    G.add_edge(2, 3, highway="motorway", length=100.0)  # not runnable
    G = compute_edge_grades(G)
    G = tag_edges(G)
    runnability = {
        d["highway_type"]: d["is_runnable"] for u, v, d in G.edges(data=True)
    }
    assert runnability["footway"] is True
    assert runnability["motorway"] is False


# ── Run profile tests ─────────────────────────────────────────────────────────


def test_all_profiles_exist():
    for run_type in ["city", "trail", "scenic", "interval"]:
        profile = get_profile(run_type)
        assert profile.name == run_type


def test_invalid_profile_raises():
    with pytest.raises(ValueError):
        get_profile("marathon")


def test_trail_tolerates_hills_more_than_interval():
    trail = get_profile("trail")
    interval = get_profile("interval")
    assert trail.grade_penalty_per_pct < interval.grade_penalty_per_pct
    assert trail.max_comfortable_grade > interval.max_comfortable_grade


def test_city_prefers_asphalt_over_dirt():
    city = get_profile("city")
    assert city.surface_weights["asphalt"] < city.surface_weights["dirt"]


def test_trail_prefers_dirt_over_asphalt():
    trail = get_profile("trail")
    assert trail.surface_weights["dirt"] < trail.surface_weights["asphalt"]


# ── Validator tests ───────────────────────────────────────────────────────────


def test_valid_request_passes():
    validate_route_request(45.764, 4.835, 10.0, "city")


def test_invalid_lat_raises():
    with pytest.raises(ValueError, match="Latitude"):
        validate_route_request(999, 4.835, 10.0, "city")


def test_invalid_lng_raises():
    with pytest.raises(ValueError, match="Longitude"):
        validate_route_request(45.764, 999, 10.0, "city")


def test_distance_too_short_raises():
    with pytest.raises(ValueError, match="at least 1 km"):
        validate_route_request(45.764, 4.835, 0.5, "city")


def test_distance_too_long_raises():
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_route_request(45.764, 4.835, 150.0, "city")


def test_invalid_run_type_raises():
    with pytest.raises(ValueError, match="Invalid run type"):
        validate_route_request(45.764, 4.835, 10.0, "sprint")


def test_bounding_box_too_large_raises():
    with pytest.raises(ValueError, match="radius"):
        validate_route_request(45.764, 4.835, 10.0, "city", max_radius_km=3.0)


# ── City registry tests ───────────────────────────────────────────────────────


def test_get_city_valid():
    city = get_city("lyon")
    assert city.slug == "lyon"
    assert city.lat == 45.7640


def test_get_city_invalid_raises():
    with pytest.raises(ValueError):
        get_city("atlantis")


def test_find_nearest_city_lyon_coords():
    city = find_nearest_city(45.764, 4.835)
    assert city.slug == "lyon"


def test_find_nearest_city_paris_coords():
    city = find_nearest_city(48.856, 2.352)
    assert city.slug == "paris"
