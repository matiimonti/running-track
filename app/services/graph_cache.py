"""
Graph cache management.

Tracks when each city graph was last built and provides
invalidation logic — graphs older than MAX_AGE_DAYS are
considered stale and will be rebuilt on next request.
"""

import fcntl
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from app.logging_config import logger

GRAPH_DIR = Path("data/graphs")
CACHE_MANIFEST = GRAPH_DIR / "manifest.json"
MAX_AGE_DAYS = 7


def _load_manifest() -> dict:
    if CACHE_MANIFEST.exists():
        with open(CACHE_MANIFEST) as f:
            return json.load(f)
    return {}


def _save_manifest(manifest: dict) -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)


def record_graph_built(city_slug: str) -> None:
    """Record that a graph was just built for a city."""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = GRAPH_DIR / "manifest.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = _load_manifest()
        manifest[city_slug] = {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "built_at_ts": time.time(),
        }
        _save_manifest(manifest)
    logger.info("graph_cache_recorded", city=city_slug)


def is_graph_stale(city_slug: str) -> bool:
    """Return True if graph is older than MAX_AGE_DAYS or has never been built."""
    manifest = _load_manifest()
    if city_slug not in manifest:
        return True

    built_at_ts = manifest[city_slug].get("built_at_ts", 0)
    age_days = (time.time() - built_at_ts) / 86400

    if age_days > MAX_AGE_DAYS:
        logger.info(
            "graph_cache_stale",
            city=city_slug,
            age_days=round(age_days, 1),
            max_age_days=MAX_AGE_DAYS,
        )
        return True

    logger.info(
        "graph_cache_fresh",
        city=city_slug,
        age_days=round(age_days, 1),
        max_age_days=MAX_AGE_DAYS,
    )
    return False


def invalidate_graph(city_slug: str) -> None:
    """Force a graph to be rebuilt on next request."""
    graph_path = GRAPH_DIR / f"{city_slug}.graphml"

    if graph_path.exists():
        graph_path.unlink()
        logger.info("graph_file_deleted", city=city_slug)

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = GRAPH_DIR / "manifest.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = _load_manifest()
        if city_slug in manifest:
            del manifest[city_slug]
            _save_manifest(manifest)
    logger.info("graph_manifest_cleared", city=city_slug)


def get_cache_status() -> dict:
    """Return cache status for all cities."""
    manifest = _load_manifest()
    status = {}
    for city_slug, info in manifest.items():
        built_at_ts = info.get("built_at_ts", 0)
        age_days = (time.time() - built_at_ts) / 86400
        status[city_slug] = {
            "built_at": info.get("built_at"),
            "age_days": round(age_days, 1),
            "stale": age_days > MAX_AGE_DAYS,
        }
    return status
