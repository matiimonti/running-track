"""
OpenRouteService routing client — round-trip fallback when OSMnx and GraphHopper fail.

Calls the ORS Directions API with round_trip options.
Returns a list of (lat, lng) coordinate pairs.

Docs: https://openrouteservice.org/dev/#/api-docs/v2/directions/{profile}/geojson/post
"""

import httpx
from app.logging_config import logger

_BASE_URL = "https://api.openrouteservice.org/v2/directions/{profile}/geojson"
_TIMEOUT_S = 10.0

# Map internal profile names to ORS pedestrian profiles
_PROFILE_MAP = {
    "city": "foot-walking",
    "trail": "foot-hiking",
    "scenic": "foot-walking",
    "interval": "foot-walking",
}


def fetch_round_trip(
    lat: float,
    lng: float,
    distance_m: float,
    profile_name: str,
    api_key: str,
) -> list[tuple[float, float]]:
    """
    Request a round-trip route from OpenRouteService.

    Returns a list of (lat, lng) tuples.
    Raises ValueError if the API returns no route or an error.
    Raises httpx.HTTPError on network failures.
    """
    ors_profile = _PROFILE_MAP.get(profile_name, "foot-walking")
    url = _BASE_URL.format(profile=ors_profile)

    payload = {
        "coordinates": [[lng, lat]],  # ORS uses [lng, lat] order
        "options": {
            "round_trip": {
                "length": distance_m,
                "seed": 0,  # deterministic route for same inputs
            }
        },
    }

    logger.info(
        "ors_request",
        lat=lat,
        lng=lng,
        distance_m=distance_m,
        profile=ors_profile,
    )

    response = httpx.post(
        url,
        headers={"Authorization": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=_TIMEOUT_S,
    )

    if response.status_code != 200:
        logger.warning(
            "ors_error",
            status=response.status_code,
            body=response.text[:200],
        )
        raise ValueError(
            f"OpenRouteService returned HTTP {response.status_code}: {response.text[:200]}"
        )

    data = response.json()
    features = data.get("features", [])

    if not features:
        raise ValueError("OpenRouteService returned no features.")

    # Coordinates are GeoJSON: [[lng, lat, elev], ...]
    raw_coords = features[0].get("geometry", {}).get("coordinates", [])

    if not raw_coords:
        raise ValueError("OpenRouteService feature has no coordinates.")

    # Convert [lng, lat, ...] → (lat, lng)
    coords = [(c[1], c[0]) for c in raw_coords]

    logger.info(
        "ors_route_received",
        points=len(coords),
    )

    return coords
