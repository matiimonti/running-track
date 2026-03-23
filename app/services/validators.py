"""
Input validators for route generation requests.
"""

import math
from app.logging_config import logger


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance in km between two coordinates."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def validate_coordinates(lat: float, lng: float) -> None:
    """Validate lat/lng are within real-world bounds."""
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude {lat} out of range. Must be between -90 and 90.")
    if not (-180 <= lng <= 180):
        raise ValueError(f"Longitude {lng} out of range. Must be between -180 and 180.")


def validate_distance(distance_km: float) -> None:
    """Validate requested route distance is within acceptable range."""
    if distance_km < 1.0:
        raise ValueError("Distance must be at least 1 km.")
    if distance_km > 100.0:
        raise ValueError("Distance cannot exceed 100 km.")


def validate_bounding_box(
    lat: float, lng: float, distance_km: float, max_radius_km: float = 50.0
) -> None:
    """
    Validate that the route won't require a graph area too large to process.
    A route of distance D km can extend at most D km from the start in any direction,
    so D is the safe upper bound on required graph radius.
    """
    if distance_km > max_radius_km:
        raise ValueError(
            f"Requested distance {distance_km}km exceeds the maximum supported "
            f"graph radius of {max_radius_km}km. Try a shorter distance."
        )


def validate_route_request(
    lat: float,
    lng: float,
    distance_km: float,
    run_type: str,
    max_radius_km: float = 50.0,
) -> None:
    """Run all validations for a route generation request."""
    valid_run_types = ["city", "trail", "scenic", "interval"]

    validate_coordinates(lat, lng)
    validate_distance(distance_km)
    validate_bounding_box(lat, lng, distance_km, max_radius_km)

    if run_type not in valid_run_types:
        raise ValueError(
            f"Invalid run type '{run_type}'. Must be one of: {valid_run_types}"
        )

    logger.info(
        "route_request_valid",
        lat=lat,
        lng=lng,
        distance_km=distance_km,
        run_type=run_type,
    )
