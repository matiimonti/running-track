import pytest
import networkx as nx

from app.services.routing_engine import (
    compute_edge_cost,
    make_cost_fn,
    snap_to_nearest_node,
    detect_crossings,
    extract_waypoints,
    extract_elevation_profile,
    compute_route_metadata,
    score_route,
    extend_path_to_target,
    build_used_edges,
    _REPETITION_PENALTY,
)
from app.services.run_profiles import get_profile, CITY, TRAIL, INTERVAL


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_edge(
    length=100.0,
    surface_type="asphalt",
    highway_type="footway",
    is_runnable=True,
    grade=0.0,
    elevation_start=100.0,
    elevation_end=100.0,
) -> dict:
    return {
        "length": length,
        "surface_type": surface_type,
        "highway_type": highway_type,
        "is_runnable": is_runnable,
        "grade": grade,
        "elevation_start": elevation_start,
        "elevation_end": elevation_end,
    }


def make_loop_graph() -> nx.MultiDiGraph:
    """
    Four-node loop: 1 → 2 → 3 → 4 → 1, fully tagged.
    Positions form a rough square around Lyon centre so bearing logic is realistic.
    """
    G = nx.MultiDiGraph()
    G.add_node(1, x=4.835, y=45.764, elevation=100.0)
    G.add_node(2, x=4.836, y=45.765, elevation=110.0)
    G.add_node(3, x=4.837, y=45.764, elevation=105.0)
    G.add_node(4, x=4.836, y=45.763, elevation=100.0)
    edges = [
        (1, 2, "footway", "asphalt", 200.0, 0.0, 100.0, 110.0),
        (2, 3, "path", "dirt", 150.0, -3.0, 110.0, 105.0),
        (3, 4, "residential", "asphalt", 180.0, -2.0, 105.0, 100.0),
        (4, 1, "footway", "asphalt", 190.0, 0.0, 100.0, 100.0),
    ]
    for u, v, hw, surf, length, grade, elev_s, elev_e in edges:
        G.add_edge(
            u,
            v,
            highway_type=hw,
            surface_type=surf,
            length=length,
            grade=grade,
            elevation_start=elev_s,
            elevation_end=elev_e,
            is_runnable=True,
        )
    return G


def make_turn_graph() -> nx.MultiDiGraph:
    """
    Five nodes arranged to produce known turn directions.

    Path A: 1 → 2 → 3  (straight north)
    Path B: 1 → 2 → 4  (right turn — east then south)
    Path C: 1 → 2 → 5  (left turn — east then north)

    Node layout:
      5 (north of 4)
      2 (east of 1)
      4 (south of 2)
      1 (origin)
      3 (north of 1, for straight test)
    """
    G = nx.MultiDiGraph()
    # Straight: 1 → 2 → 3, all heading north
    G.add_node(1, x=4.000, y=45.000, elevation=100.0)
    G.add_node(2, x=4.000, y=45.001, elevation=100.0)
    G.add_node(3, x=4.000, y=45.002, elevation=100.0)
    # Right turn from 2: heading east then south
    G.add_node(4, x=4.001, y=45.001, elevation=100.0)  # east of 2
    G.add_node(5, x=4.001, y=45.002, elevation=100.0)  # north of 4 (left turn)

    for u, v, name in [
        (1, 2, "Main Street"),
        (2, 3, "Main Street"),
        (2, 4, "Side Road"),
        (4, 5, "Side Road"),
        (2, 5, "Park Lane"),
    ]:
        G.add_edge(
            u,
            v,
            highway_type="footway",
            surface_type="asphalt",
            length=110.0,
            grade=0.0,
            elevation_start=100.0,
            elevation_end=100.0,
            is_runnable=True,
            name=name,
        )
    return G


# ── compute_edge_cost ─────────────────────────────────────────────────────────


def test_cost_inf_for_non_runnable():
    edge = make_edge(is_runnable=False)
    assert compute_edge_cost(edge, CITY) == float("inf")


def test_cost_inf_for_steps_when_disallowed():
    edge = make_edge(highway_type="steps", is_runnable=True)
    assert CITY.allow_steps is False
    assert compute_edge_cost(edge, CITY) == float("inf")


def test_cost_steps_allowed_for_trail():
    edge = make_edge(highway_type="steps", surface_type="paved", is_runnable=True)
    assert TRAIL.allow_steps is True
    cost = compute_edge_cost(edge, TRAIL)
    assert cost < float("inf")


def test_cost_scales_with_length():
    short = make_edge(length=100.0)
    long = make_edge(length=200.0)
    assert compute_edge_cost(long, CITY) == pytest.approx(
        compute_edge_cost(short, CITY) * 2, rel=1e-6
    )


def test_cost_surface_weight_applied():
    asphalt_edge = make_edge(surface_type="asphalt")
    dirt_edge = make_edge(surface_type="dirt")
    assert compute_edge_cost(asphalt_edge, CITY) < compute_edge_cost(dirt_edge, CITY)


def test_cost_highway_weight_applied():
    footway = make_edge(highway_type="footway")
    primary = make_edge(highway_type="primary")
    assert compute_edge_cost(footway, CITY) < compute_edge_cost(primary, CITY)


def test_cost_flat_grade_no_extra_penalty():
    flat = make_edge(grade=0.0)
    cost = compute_edge_cost(flat, CITY)
    # grade_multiplier = 1.0 + 0.3 * 0 = 1.0
    assert cost == pytest.approx(
        flat["length"]
        * CITY.surface_weights["asphalt"]
        * CITY.highway_weights["footway"]
        * 1.0,
        rel=1e-6,
    )


def test_cost_grade_within_comfort_gentle_penalty():
    edge = make_edge(grade=3.0)  # within CITY comfort of 5%
    cost_flat = compute_edge_cost(make_edge(grade=0.0), CITY)
    cost_graded = compute_edge_cost(edge, CITY)
    assert cost_graded > cost_flat


def test_cost_grade_above_comfort_steep_penalty():
    within = make_edge(grade=5.0)  # at CITY limit
    above = make_edge(grade=10.0)  # above limit
    assert compute_edge_cost(above, CITY) > compute_edge_cost(within, CITY)


def test_cost_grade_uses_absolute_value():
    uphill = make_edge(grade=5.0)
    downhill = make_edge(grade=-5.0)
    assert compute_edge_cost(uphill, CITY) == pytest.approx(
        compute_edge_cost(downhill, CITY), rel=1e-6
    )


def test_cost_unknown_surface_uses_default():
    edge = make_edge(surface_type="lunar_dust")
    cost = compute_edge_cost(edge, CITY)
    assert cost < float("inf")
    assert cost > 0


# ── make_cost_fn ──────────────────────────────────────────────────────────────


def test_cost_fn_no_penalty_when_not_in_used():
    edge = make_edge()
    fn = make_cost_fn(CITY, used_edges=set())
    assert fn(1, 2, edge) == pytest.approx(compute_edge_cost(edge, CITY), rel=1e-6)


def test_cost_fn_repetition_penalty_applied():
    edge = make_edge()
    fn = make_cost_fn(CITY, used_edges={(1, 2)})
    base = compute_edge_cost(edge, CITY)
    assert fn(1, 2, edge) == pytest.approx(base * _REPETITION_PENALTY, rel=1e-6)


def test_cost_fn_penalty_not_applied_to_different_edge():
    edge = make_edge()
    fn = make_cost_fn(CITY, used_edges={(1, 2)})
    assert fn(2, 3, edge) == pytest.approx(compute_edge_cost(edge, CITY), rel=1e-6)


def test_cost_fn_inf_still_inf_with_penalty():
    edge = make_edge(is_runnable=False)
    fn = make_cost_fn(CITY, used_edges={(1, 2)})
    assert fn(1, 2, edge) == float("inf")


def test_cost_fn_no_used_edges_param():
    edge = make_edge()
    fn = make_cost_fn(CITY)
    assert fn(1, 2, edge) == pytest.approx(compute_edge_cost(edge, CITY), rel=1e-6)


# ── build_used_edges ──────────────────────────────────────────────────────────


def test_build_used_edges_correct_pairs():
    path = [1, 2, 3, 4]
    used = build_used_edges(path)
    assert used == {(1, 2), (2, 3), (3, 4)}


def test_build_used_edges_single_edge():
    assert build_used_edges([1, 2]) == {(1, 2)}


def test_build_used_edges_empty_path():
    assert build_used_edges([1]) == set()


# ── snap_to_nearest_node ──────────────────────────────────────────────────────


def test_snap_returns_nearest_node():
    G = make_loop_graph()
    # Node 1 is at (y=45.764, x=4.835) — query very close to it
    result = snap_to_nearest_node(G, lat=45.7641, lng=4.8351)
    assert result == 1


def test_snap_skips_isolated_node():
    G = make_loop_graph()
    # Add an isolated node closer to query point
    G.add_node(99, x=4.8351, y=45.7641, elevation=100.0)  # isolated, no edges
    result = snap_to_nearest_node(G, lat=45.7641, lng=4.8351)
    assert result != 99  # should skip isolated node


def test_snap_raises_on_empty_graph():
    G = nx.MultiDiGraph()
    with pytest.raises(ValueError, match="No connected node"):
        snap_to_nearest_node(G, lat=45.764, lng=4.835)


def test_snap_raises_on_all_isolated():
    G = nx.MultiDiGraph()
    G.add_node(1, x=4.835, y=45.764)
    G.add_node(2, x=4.836, y=45.765)
    # No edges — both isolated
    with pytest.raises(ValueError, match="No connected node"):
        snap_to_nearest_node(G, lat=45.764, lng=4.835)


# ── detect_crossings ──────────────────────────────────────────────────────────


def test_detect_crossings_clean_path():
    G = make_loop_graph()
    warnings = detect_crossings(G, [1, 2, 3, 4])
    assert warnings == []


def test_detect_crossings_flags_motorway_adjacent_node():
    G = make_loop_graph()
    G.add_node(99, x=4.840, y=45.764, elevation=100.0)
    G.add_edge(2, 99, highway_type="motorway", length=500.0, is_runnable=False)
    warnings = detect_crossings(G, [1, 2, 3])
    assert len(warnings) == 1
    assert warnings[0].node_id == 2
    assert warnings[0].highway_type == "motorway"


def test_detect_crossings_flags_trunk_link():
    G = make_loop_graph()
    G.add_node(99, x=4.840, y=45.764, elevation=100.0)
    G.add_edge(3, 99, highway_type="trunk_link", length=300.0, is_runnable=False)
    warnings = detect_crossings(G, [1, 2, 3])
    assert any(w.highway_type == "trunk_link" for w in warnings)


def test_detect_crossings_deduplicates_node():
    G = make_loop_graph()
    G.add_node(98, x=4.840, y=45.764, elevation=100.0)
    G.add_node(99, x=4.841, y=45.764, elevation=100.0)
    G.add_edge(2, 98, highway_type="motorway", length=500.0, is_runnable=False)
    G.add_edge(2, 99, highway_type="trunk", length=400.0, is_runnable=False)
    # Node 2 has two dangerous neighbors — should still produce only one warning
    warnings = detect_crossings(G, [1, 2, 3])
    flagged_nodes = [w.node_id for w in warnings]
    assert flagged_nodes.count(2) == 1


def test_detect_crossings_warning_has_correct_coords():
    G = make_loop_graph()
    G.add_node(99, x=4.840, y=45.764, elevation=100.0)
    G.add_edge(1, 99, highway_type="motorway", length=500.0, is_runnable=False)
    warnings = detect_crossings(G, [1, 2])
    assert warnings[0].lat == G.nodes[1]["y"]
    assert warnings[0].lng == G.nodes[1]["x"]


# ── extract_waypoints ─────────────────────────────────────────────────────────


def test_waypoints_start_and_end_always_present():
    G = make_turn_graph()
    waypoints = extract_waypoints(G, [1, 2, 3])
    assert waypoints[0].turn == "start"
    assert waypoints[-1].turn == "end"


def test_waypoints_raises_on_single_node_path():
    G = make_turn_graph()
    with pytest.raises(ValueError):
        extract_waypoints(G, [1])


def test_waypoints_straight_not_emitted():
    G = make_turn_graph()
    # 1 → 2 → 3 is straight north — no interior waypoint expected
    waypoints = extract_waypoints(G, [1, 2, 3])
    interior = [w for w in waypoints if w.turn not in ("start", "end")]
    assert interior == []


def test_waypoints_turn_emitted_on_direction_change():
    G = make_turn_graph()
    # 1 → 2 → 4: going north then east → a turn should be detected at node 2
    waypoints = extract_waypoints(G, [1, 2, 4])
    interior = [w for w in waypoints if w.turn not in ("start", "end")]
    assert len(interior) >= 1


def test_waypoints_emitted_on_street_name_change():
    G = make_loop_graph()
    # Add name attributes so that edges 1→2 and 2→3 have different names
    for key in G[1][2]:
        G[1][2][key]["name"] = "Oak Street"
    for key in G[2][3]:
        G[2][3][key]["name"] = "River Road"
    waypoints = extract_waypoints(G, [1, 2, 3])
    # Node 2 should be emitted because street name changes
    interior = [w for w in waypoints if w.turn not in ("start", "end")]
    assert any(w.lat == G.nodes[2]["y"] for w in interior)


def test_waypoints_distance_from_prev_positive():
    G = make_turn_graph()
    waypoints = extract_waypoints(G, [1, 2, 4, 5])
    # All waypoints after start should have positive distance
    for w in waypoints[1:]:
        assert w.distance_from_prev_m > 0


def test_waypoints_end_node_coords_match_graph():
    G = make_loop_graph()
    waypoints = extract_waypoints(G, [1, 2, 3, 4])
    end = waypoints[-1]
    assert end.lat == G.nodes[4]["y"]
    assert end.lng == G.nodes[4]["x"]


def test_waypoints_elevation_present():
    G = make_loop_graph()
    waypoints = extract_waypoints(G, [1, 2, 3, 4])
    for w in waypoints:
        assert isinstance(w.elevation_m, float)


# ── extract_elevation_profile ─────────────────────────────────────────────────


def test_elevation_profile_first_point_distance_zero():
    G = make_loop_graph()
    profile = extract_elevation_profile(G, [1, 2, 3, 4])
    assert profile[0].distance_m == 0.0


def test_elevation_profile_point_count_equals_node_count():
    G = make_loop_graph()
    path = [1, 2, 3, 4]
    profile = extract_elevation_profile(G, path)
    assert len(profile) == len(path)


def test_elevation_profile_distances_monotonically_increasing():
    G = make_loop_graph()
    profile = extract_elevation_profile(G, [1, 2, 3, 4])
    distances = [p.distance_m for p in profile]
    assert all(distances[i] < distances[i + 1] for i in range(len(distances) - 1))


def test_elevation_profile_total_matches_path_length():
    G = make_loop_graph()
    path = [1, 2, 3]
    profile = extract_elevation_profile(G, path)
    expected = sum(
        min(G[path[i]][path[i + 1]].values(), key=lambda d: d.get("length", 0))[
            "length"
        ]
        for i in range(len(path) - 1)
    )
    assert profile[-1].distance_m == pytest.approx(expected, rel=1e-6)


def test_elevation_profile_elevations_match_nodes():
    G = make_loop_graph()
    path = [1, 2, 3]
    profile = extract_elevation_profile(G, path)
    assert profile[0].elevation_m == G.nodes[1]["elevation"]
    assert profile[1].elevation_m == G.nodes[2]["elevation"]
    assert profile[2].elevation_m == G.nodes[3]["elevation"]


def test_elevation_profile_raises_on_empty_path():
    G = make_loop_graph()
    with pytest.raises((ValueError, KeyError, StopIteration)):
        extract_elevation_profile(G, [])


# ── compute_route_metadata ────────────────────────────────────────────────────


def test_metadata_total_distance_correct():
    G = make_loop_graph()
    path = [1, 2, 3, 4]
    meta = compute_route_metadata(G, path, CITY)
    expected = 200.0 + 150.0 + 180.0
    assert meta.total_distance_m == pytest.approx(expected, rel=1e-6)


def test_metadata_elevation_gain_only_uphill():
    G = make_loop_graph()
    # Edges: 1→2 +10m, 2→3 -5m, 3→4 -5m
    meta = compute_route_metadata(G, [1, 2, 3, 4], CITY)
    assert meta.elevation_gain_m == pytest.approx(10.0, rel=1e-3)


def test_metadata_elevation_loss_only_downhill():
    G = make_loop_graph()
    meta = compute_route_metadata(G, [1, 2, 3, 4], CITY)
    assert meta.elevation_loss_m == pytest.approx(10.0, rel=1e-3)


def test_metadata_surface_breakdown_sums_to_100():
    G = make_loop_graph()
    meta = compute_route_metadata(G, [1, 2, 3, 4], CITY)
    total = sum(meta.surface_breakdown.values())
    assert total == pytest.approx(100.0, abs=0.2)


def test_metadata_estimated_time_uses_profile_pace():
    G = make_loop_graph()
    path = [1, 2, 3, 4]
    city_meta = compute_route_metadata(G, path, CITY)
    trail_meta = compute_route_metadata(G, path, TRAIL)
    # Trail pace is slower → more time for the same distance
    assert trail_meta.estimated_time_min > city_meta.estimated_time_min


def test_metadata_estimated_time_calculation():
    G = make_loop_graph()
    path = [1, 2, 3, 4]
    meta = compute_route_metadata(G, path, CITY)
    expected = round((meta.total_distance_m / 1000) * CITY.pace_min_per_km, 1)
    assert meta.estimated_time_min == pytest.approx(expected, abs=0.05)


# ── score_route ───────────────────────────────────────────────────────────────


def test_score_within_bounds():
    G = make_loop_graph()
    score = score_route(G, [1, 2, 3, 4], CITY)
    assert 0 <= score <= 100


def test_score_is_integer():
    G = make_loop_graph()
    assert isinstance(score_route(G, [1, 2, 3, 4], CITY), int)


def test_score_city_prefers_asphalt_route():
    G = nx.MultiDiGraph()
    G.add_node(1, x=4.835, y=45.764, elevation=100.0)
    G.add_node(2, x=4.836, y=45.765, elevation=100.0)
    G.add_node(3, x=4.837, y=45.764, elevation=100.0)
    G.add_edge(
        1,
        2,
        highway_type="footway",
        surface_type="asphalt",
        length=200.0,
        grade=0.0,
        elevation_start=100.0,
        elevation_end=100.0,
        is_runnable=True,
    )
    G.add_edge(
        2,
        3,
        highway_type="footway",
        surface_type="asphalt",
        length=200.0,
        grade=0.0,
        elevation_start=100.0,
        elevation_end=100.0,
        is_runnable=True,
    )

    G2 = nx.MultiDiGraph()
    G2.add_node(1, x=4.835, y=45.764, elevation=100.0)
    G2.add_node(2, x=4.836, y=45.765, elevation=100.0)
    G2.add_node(3, x=4.837, y=45.764, elevation=100.0)
    G2.add_edge(
        1,
        2,
        highway_type="track",
        surface_type="dirt",
        length=200.0,
        grade=0.0,
        elevation_start=100.0,
        elevation_end=100.0,
        is_runnable=True,
    )
    G2.add_edge(
        2,
        3,
        highway_type="track",
        surface_type="dirt",
        length=200.0,
        grade=0.0,
        elevation_start=100.0,
        elevation_end=100.0,
        is_runnable=True,
    )

    asphalt_score = score_route(G, [1, 2, 3], CITY)
    dirt_score = score_route(G2, [1, 2, 3], CITY)
    assert asphalt_score > dirt_score


def test_score_flat_route_better_for_interval():
    G_flat = nx.MultiDiGraph()
    G_flat.add_node(1, x=4.835, y=45.764, elevation=100.0)
    G_flat.add_node(2, x=4.836, y=45.765, elevation=100.0)
    G_flat.add_edge(
        1,
        2,
        highway_type="footway",
        surface_type="asphalt",
        length=200.0,
        grade=0.0,
        elevation_start=100.0,
        elevation_end=100.0,
        is_runnable=True,
    )

    G_hilly = nx.MultiDiGraph()
    G_hilly.add_node(1, x=4.835, y=45.764, elevation=100.0)
    G_hilly.add_node(2, x=4.836, y=45.765, elevation=120.0)
    G_hilly.add_edge(
        1,
        2,
        highway_type="footway",
        surface_type="asphalt",
        length=200.0,
        grade=10.0,
        elevation_start=100.0,
        elevation_end=120.0,
        is_runnable=True,
    )

    flat_score = score_route(G_flat, [1, 2], INTERVAL)
    hilly_score = score_route(G_hilly, [1, 2], INTERVAL)
    assert flat_score > hilly_score


def test_score_neutral_when_no_popularity_data():
    G = make_loop_graph()
    score = score_route(G, [1, 2, 3, 4], CITY)
    # With neutral popularity (10/20) + good asphalt surface + flat grades,
    # score should be comfortably above 50
    assert score > 50


# ── extend_path_to_target ─────────────────────────────────────────────────────


def make_extendable_graph() -> nx.MultiDiGraph:
    """
    Linear chain 1 → 2 → 3 → 4 → 5, plus return edge 5 → 1.
    Used to test path extension.
    """
    G = nx.MultiDiGraph()
    coords = [
        (1, 4.835, 45.764),
        (2, 4.836, 45.764),
        (3, 4.837, 45.764),
        (4, 4.838, 45.764),
        (5, 4.839, 45.764),
    ]
    for node_id, x, y in coords:
        G.add_node(node_id, x=x, y=y, elevation=100.0)
    for u, v in [(1, 2), (2, 3), (3, 4), (4, 5)]:
        G.add_edge(
            u,
            v,
            highway_type="footway",
            surface_type="asphalt",
            length=100.0,
            grade=0.0,
            elevation_start=100.0,
            elevation_end=100.0,
            is_runnable=True,
        )
    # Return edge so A* can close the loop
    G.add_edge(
        5,
        1,
        highway_type="footway",
        surface_type="asphalt",
        length=400.0,
        grade=0.0,
        elevation_start=100.0,
        elevation_end=100.0,
        is_runnable=True,
    )
    return G


def test_extend_returns_original_if_long_enough():
    G = make_extendable_graph()
    path = [1, 2, 3, 4, 5, 1]  # 500m total
    result = extend_path_to_target(G, path, target_distance_m=400.0, profile=CITY)
    assert result == path


def test_extend_lengthens_short_path():
    G = make_extendable_graph()
    short_path = [1, 2]  # 100m — well below any target
    result = extend_path_to_target(G, short_path, target_distance_m=300.0, profile=CITY)
    # Extended path must be longer than original
    original_len = sum(
        min(
            G[short_path[i]][short_path[i + 1]].values(),
            key=lambda d: d.get("length", 0),
        )["length"]
        for i in range(len(short_path) - 1)
    )
    result_len = sum(
        min(G[result[i]][result[i + 1]].values(), key=lambda d: d.get("length", 0))[
            "length"
        ]
        for i in range(len(result) - 1)
    )
    assert result_len > original_len


# ── run profile fields ────────────────────────────────────────────────────────


def test_all_profiles_have_bearing_offset():
    for name in ["city", "trail", "scenic", "interval"]:
        profile = get_profile(name)
        assert isinstance(profile.bearing_offset, float)


def test_all_profiles_have_midpoint_factor():
    for name in ["city", "trail", "scenic", "interval"]:
        profile = get_profile(name)
        assert 0.0 < profile.midpoint_factor < 1.0


def test_all_profiles_have_pace():
    for name in ["city", "trail", "scenic", "interval"]:
        profile = get_profile(name)
        assert profile.pace_min_per_km > 0


def test_bearing_offsets_are_distinct():
    offsets = {
        get_profile(n).bearing_offset for n in ["city", "trail", "scenic", "interval"]
    }
    assert len(offsets) == 4


def test_interval_pace_faster_than_trail():
    assert INTERVAL.pace_min_per_km < TRAIL.pace_min_per_km


def test_city_midpoint_factor_standard():
    assert CITY.midpoint_factor == pytest.approx(0.50, rel=1e-3)
