"""
GraphHopper routing client — round-trip fallback when OSMnx fails.

Calls the GraphHopper Routing API with algorithm="round_trip".
Returns a list of (lat, lng) coordinate pairs.

Docs: https://docs.graphhopper.com/#operation/postRoute
"""

import httpx
from app.logging_config import logger

_BASE_URL = "https://graphhopper.com/api/1/route"
_TIMEOUT_S = 10.0

# Map internal profile names to GraphHopper vehicle profiles
_PROFILE_MAP = {
    "city": "foot",
    "trail": "hike",
    "scenic": "foot",
    "interval": "foot",
}


def fetch_round_trip(
    lat: float,
    lng: float,
    distance_m: float,
    profile_name: str,
    api_key: str,
) -> list[tuple[float, float]]:
    """
    Request a round-trip route from GraphHopper.

    Returns a list of (lat, lng) tuples.
    Raises ValueError if the API returns no route or an error.
    Raises httpx.HTTPError on network failures.
    """
    gh_profile = _PROFILE_MAP.get(profile_name, "foot")

    payload = {
        "points": [[lng, lat]],  # GraphHopper uses [lng, lat] order
        "profile": gh_profile,
        "algorithm": "round_trip",
        "round_trip.distance": distance_m,
        "points_encoded": False,  # ask for GeoJSON coordinates
        "locale": "en",
    }

    logger.info(
        "graphhopper_request",
        lat=lat,
        lng=lng,
        distance_m=distance_m,
        profile=gh_profile,
    )

    response = httpx.post(
        _BASE_URL,
        params={"key": api_key},
        json=payload,
        timeout=_TIMEOUT_S,
    )

    if response.status_code != 200:
        logger.warning(
            "graphhopper_error",
            status=response.status_code,
            body=response.text[:200],
        )
        raise ValueError(
            f"GraphHopper returned HTTP {response.status_code}: {response.text[:200]}"
        )

    data = response.json()
    paths = data.get("paths", [])

    if not paths:
        raise ValueError("GraphHopper returned no paths.")

    # Coordinates are GeoJSON: [[lng, lat, elev], ...]
    raw_coords = paths[0].get("points", {}).get("coordinates", [])

    if not raw_coords:
        raise ValueError("GraphHopper path has no coordinates.")

    # Convert [lng, lat, ...] → (lat, lng)
    coords = [(c[1], c[0]) for c in raw_coords]

    logger.info(
        "graphhopper_route_received",
        points=len(coords),
        distance_m=paths[0].get("distance"),
    )

    return coords
