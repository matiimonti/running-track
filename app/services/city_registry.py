"""
City registry — defines supported cities and their OSM place names.

Adding a new city: add an entry to CITIES with a slug, OSM place name,
and approximate bounding box. The graph will be downloaded on first request.
"""

from dataclasses import dataclass
from app.logging_config import logger
from app.services.validators import haversine_km


@dataclass
class City:
    slug: str  # used as filename and API param e.g. "lyon"
    osm_name: str  # passed to OSMnx e.g. "Lyon, France"
    display_name: str  # shown to users e.g. "Lyon, France"
    lat: float  # city centre latitude
    lng: float  # city centre longitude
    max_radius_km: float = 15.0  # max route radius for this city


CITIES = {
    "lyon": City(
        slug="lyon",
        osm_name="Lyon, France",
        display_name="Lyon, France",
        lat=45.7640,
        lng=4.8357,
        max_radius_km=15.0,
    ),
    "paris": City(
        slug="paris",
        osm_name="Paris, France",
        display_name="Paris, France",
        lat=48.8566,
        lng=2.3522,
        max_radius_km=15.0,
    ),
    "london": City(
        slug="london",
        osm_name="London, England",
        display_name="London, UK",
        lat=51.5074,
        lng=-0.1278,
        max_radius_km=15.0,
    ),
}


def get_city(slug: str) -> City:
    if slug not in CITIES:
        raise ValueError(f"Unsupported city '{slug}'. Supported: {list(CITIES.keys())}")
    return CITIES[slug]


def find_nearest_city(lat: float, lng: float) -> City:
    """Find the closest supported city to a given coordinate."""
    nearest = min(CITIES.values(), key=lambda city: haversine_km(lat, lng, city.lat, city.lng))
    dist_km = haversine_km(lat, lng, nearest.lat, nearest.lng)
    logger.info("nearest_city_found", city=nearest.slug, dist_km=round(dist_km, 1))
    return nearest
