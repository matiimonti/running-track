import osmnx as ox
import networkx as nx
import srtm
from pathlib import Path
from app.logging_config import logger

from app.services.graph_cache import record_graph_built, is_graph_stale

GRAPH_DIR = Path("data/graphs")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# SRTM elevation data (cached locally after first download)
elevation_data = srtm.get_data()


def get_graph_path(city_slug: str) -> Path:
    return GRAPH_DIR / f"{city_slug}.graphml"


def download_graph(city_slug: str, place_name: str) -> nx.MultiDiGraph:
    """Download walkable/runnable path graph for a city from OSM."""
    logger.info("downloading_graph", city=city_slug, place=place_name)
    G = ox.graph_from_place(
        place_name,
        network_type="all",
        retain_all=False,
        simplify=True,
    )
    logger.info(
        "graph_downloaded", city=city_slug, nodes=len(G.nodes), edges=len(G.edges)
    )
    return G


def attach_elevation(G: nx.MultiDiGraph, city_slug: str) -> nx.MultiDiGraph:
    """Attach elevation in metres to every node using SRTM data."""
    logger.info("attaching_elevation", city=city_slug)

    missing = 0
    for node_id, data in G.nodes(data=True):
        lat = data.get("y")
        lon = data.get("x")
        elev = elevation_data.get_elevation(lat, lon)
        if elev is None:
            elev = 0.0
            missing += 1
        G.nodes[node_id]["elevation"] = float(elev)

    logger.info(
        "elevation_attached", city=city_slug, total_nodes=len(G.nodes), missing=missing
    )
    return G


def compute_edge_grades(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Compute grade (slope %) for every edge from endpoint elevation difference."""
    for u, v, data in G.edges(data=True):
        elev_u = G.nodes[u].get("elevation", 0.0)
        elev_v = G.nodes[v].get("elevation", 0.0)
        length = data.get("length", 1.0)

        if length >= 5.0:  # ignore edges shorter than 5m — grades unreliable
            grade = (elev_v - elev_u) / length
            grade_pct = round(max(-30.0, min(30.0, grade * 100)), 2)  # clamp to ±30%
        else:
            grade_pct = 0.0

        data["grade"] = grade_pct
        data["elevation_start"] = elev_u
        data["elevation_end"] = elev_v

    return G


def load_or_download_graph(
    city_slug: str, place_name: str, force_refresh: bool = False
) -> nx.MultiDiGraph:
    """Load graph from disk if fresh, otherwise download, enrich, and save."""
    graph_path = get_graph_path(city_slug)

    if graph_path.exists() and not force_refresh and not is_graph_stale(city_slug):
        logger.info("loading_graph_from_disk", city=city_slug)
        G = ox.load_graphml(graph_path)
        logger.info(
            "graph_loaded", city=city_slug, nodes=len(G.nodes), edges=len(G.edges)
        )
        return G

    G = download_graph(city_slug, place_name)
    G = attach_elevation(G, city_slug)
    G = compute_edge_grades(G)
    G = tag_edges(G)
    ox.save_graphml(G, graph_path)
    record_graph_built(city_slug)  # ← record in manifest
    logger.info("graph_saved", city=city_slug, path=str(graph_path))
    return G


# Fallback surface by highway type when OSM surface tag is missing
HIGHWAY_SURFACE_FALLBACK = {
    "footway": "paved",
    "pedestrian": "paved",
    "corridor": "paved",
    "steps": "paved",
    "residential": "asphalt",
    "living_street": "asphalt",
    "primary": "asphalt",
    "secondary": "asphalt",
    "tertiary": "asphalt",
    "service": "asphalt",
    "cycleway": "asphalt",
    "path": "unpaved",
    "track": "dirt",
    "unclassified": "unpaved",
}

# Normalise raw OSM surface values to clean categories
SURFACE_NORMALISE = {
    "asphalt": "asphalt",
    "concrete": "asphalt",
    "paved": "paved",
    "cobblestone": "paved",
    "sett": "paved",
    "paving_stones": "paved",
    "compacted": "gravel",
    "gravel": "gravel",
    "fine_gravel": "gravel",
    "unpaved": "unpaved",
    "dirt": "dirt",
    "ground": "dirt",
    "grass": "grass",
    "mud": "dirt",
    "sand": "dirt",
    "wood": "paved",
    "metal": "paved",
}

# Runnable highway types — anything not in this set gets flagged
RUNNABLE_HIGHWAY_TYPES = {
    "footway",
    "path",
    "track",
    "cycleway",
    "pedestrian",
    "residential",
    "living_street",
    "unclassified",
    "tertiary",
    "secondary",
    "primary",
    "service",
    "steps",
    "corridor",
}


def tag_edges(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Tag each edge with normalised surface type, highway class, and runnability."""
    missing_surface = 0
    total = 0

    for u, v, data in G.edges(data=True):
        total += 1

        # Normalise highway type
        highway = data.get("highway", "unknown")
        if isinstance(highway, list):
            highway = highway[0]
        data["highway_type"] = str(highway)
        data["is_runnable"] = highway in RUNNABLE_HIGHWAY_TYPES

        # Normalise surface tag
        raw_surface = data.get("surface", None)
        if isinstance(raw_surface, list):
            raw_surface = raw_surface[0]

        if raw_surface and raw_surface in SURFACE_NORMALISE:
            data["surface_type"] = SURFACE_NORMALISE[raw_surface]
            data["surface_source"] = "osm"
        elif raw_surface:
            # OSM has a value but we don't recognise it — treat as unpaved
            data["surface_type"] = "unpaved"
            data["surface_source"] = "osm_unknown"
        else:
            # No surface tag — use fallback from highway type
            fallback = HIGHWAY_SURFACE_FALLBACK.get(highway, "unknown")
            data["surface_type"] = fallback
            data["surface_source"] = "fallback"
            missing_surface += 1

    logger.info(
        "edges_tagged",
        total=total,
        missing_surface=missing_surface,
        missing_pct=round(missing_surface / total * 100, 1),
    )
    return G
