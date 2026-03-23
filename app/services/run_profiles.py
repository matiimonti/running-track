"""
Run type profiles — define edge cost weights for each run type.

Each profile is a dict of weights applied during edge scoring.
Higher weight = higher cost = less preferred.
Lower weight = lower cost = more preferred.
"""

from dataclasses import dataclass


@dataclass
class RunProfile:
    name: str
    description: str

    # Surface weights (multipliers on base edge cost)
    surface_weights: dict

    # Grade penalties
    grade_penalty_per_pct: float  # cost added per % of grade (absolute)
    max_comfortable_grade: float  # above this grade, penalty increases sharply

    # Highway type weights
    highway_weights: dict

    # Whether to prefer popular (Strava) segments
    prefer_popular: bool

    # Whether to allow steps
    allow_steps: bool

    # Degrees added to the shared base bearing so each run type fans out in a
    # different direction when routes are generated from the same start point.
    bearing_offset: float

    # Fraction of target distance used to project the midpoint node.
    # Smaller = tighter loop, larger = more direct out-and-back shape.
    midpoint_factor: float


CITY = RunProfile(
    name="city",
    description="Urban streets, flat, well-paved — good for tempo and easy runs",
    surface_weights={
        "asphalt": 1.0,  # most preferred
        "paved": 1.1,
        "gravel": 2.5,
        "unpaved": 3.0,
        "dirt": 4.0,
        "grass": 4.0,
        "unknown": 2.0,
    },
    grade_penalty_per_pct=0.3,
    max_comfortable_grade=5.0,
    highway_weights={
        "footway": 1.0,
        "pedestrian": 1.0,
        "residential": 1.2,
        "living_street": 1.1,
        "cycleway": 1.3,
        "tertiary": 1.5,
        "secondary": 2.0,
        "primary": 3.0,  # avoid busy roads
        "path": 2.0,
        "track": 3.0,
        "steps": 2.5,
        "service": 1.4,
        "unclassified": 1.6,
        "corridor": 1.0,
        "unknown": 2.5,
    },
    prefer_popular=True,
    allow_steps=False,
    bearing_offset=0.0,    # reference direction — other types fan out from here
    midpoint_factor=0.50,  # standard loop shape
)

TRAIL = RunProfile(
    name="trail",
    description="Dirt paths, forest tracks, natural surfaces — for trail runners",
    surface_weights={
        "dirt": 1.0,  # most preferred
        "unpaved": 1.1,
        "gravel": 1.2,
        "grass": 1.3,
        "paved": 2.5,
        "asphalt": 3.0,  # avoid roads
        "unknown": 1.5,
    },
    grade_penalty_per_pct=0.1,  # trail runners tolerate hills
    max_comfortable_grade=15.0,
    highway_weights={
        "track": 1.0,
        "path": 1.0,
        "footway": 1.2,
        "cycleway": 1.5,
        "pedestrian": 1.3,
        "living_street": 2.0,
        "residential": 3.0,
        "tertiary": 3.5,
        "secondary": 4.0,
        "primary": 5.0,
        "steps": 1.5,  # trail runners handle steps
        "service": 2.5,
        "unclassified": 1.8,
        "corridor": 3.0,
        "unknown": 1.5,
    },
    prefer_popular=True,
    allow_steps=True,
    bearing_offset=90.0,   # explores 90° offset from city — different quadrant
    midpoint_factor=0.45,  # tighter midpoint = more winding return, more trail coverage
)

SCENIC = RunProfile(
    name="scenic",
    description="Parks, waterfronts, green spaces — beautiful and low traffic",
    surface_weights={
        "paved": 1.0,
        "asphalt": 1.1,
        "gravel": 1.3,
        "unpaved": 1.5,
        "dirt": 1.8,
        "grass": 1.4,
        "unknown": 1.8,
    },
    grade_penalty_per_pct=0.2,
    max_comfortable_grade=8.0,
    highway_weights={
        "footway": 1.0,
        "pedestrian": 1.0,
        "path": 1.1,
        "cycleway": 1.2,
        "track": 1.3,
        "living_street": 1.5,
        "residential": 2.0,
        "service": 2.0,
        "unclassified": 2.0,
        "tertiary": 2.5,
        "secondary": 3.5,
        "primary": 5.0,
        "steps": 2.0,
        "corridor": 3.0,
        "unknown": 2.0,
    },
    prefer_popular=True,
    allow_steps=False,
    bearing_offset=180.0,  # explores opposite direction from city
    midpoint_factor=0.40,  # closer midpoint = more time exploring the area
)

INTERVAL = RunProfile(
    name="interval",
    description="Flat, consistent surface — for speed work and interval training",
    surface_weights={
        "asphalt": 1.0,
        "paved": 1.0,
        "gravel": 3.0,
        "unpaved": 4.0,
        "dirt": 5.0,
        "grass": 4.0,
        "unknown": 2.5,
    },
    grade_penalty_per_pct=0.8,  # interval runners want flat
    max_comfortable_grade=2.0,
    highway_weights={
        "footway": 1.0,
        "pedestrian": 1.0,
        "cycleway": 1.1,
        "residential": 1.3,
        "living_street": 1.2,
        "tertiary": 1.8,
        "secondary": 2.5,
        "primary": 4.0,
        "path": 2.0,
        "track": 3.5,
        "steps": 10.0,  # never use steps for intervals
        "service": 1.5,
        "unclassified": 2.0,
        "corridor": 1.0,
        "unknown": 2.5,
    },
    prefer_popular=True,
    allow_steps=False,
    bearing_offset=270.0,  # explores 270° offset — last of the four quadrants
    midpoint_factor=0.55,  # farther midpoint = straighter, more direct loop
)

# Registry — look up profile by name
PROFILES = {
    "city": CITY,
    "trail": TRAIL,
    "scenic": SCENIC,
    "interval": INTERVAL,
}


def get_profile(run_type: str) -> RunProfile:
    if run_type not in PROFILES:
        raise ValueError(
            f"Unknown run type '{run_type}'. Must be one of: {list(PROFILES.keys())}"
        )
    return PROFILES[run_type]
