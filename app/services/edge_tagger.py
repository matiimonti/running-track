"""
Edge tagging — attaches normalised surface type, highway class, and runnability
to every edge in a graph. Separated from graph_service to keep concerns distinct.
"""

import networkx as nx
from app.logging_config import logger


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
