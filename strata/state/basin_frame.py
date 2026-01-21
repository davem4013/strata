"""Geometry-only BasinFrame v0."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class BasinFrame:
    """
    Geometry-only frame describing residual field shape for STRATA v0.

    Basins are intentionally empty; this frame narrates geometry only.
    """

    timestamp: str  # ISO timestamp of the underlying residual geometry
    symbol: str
    domain_dim: int
    residual_energy: float
    residual_max: float
    centroid: List[float]
    fit_quality: float
    basins: List[dict] = field(default_factory=list)

    # Optional per-frame coordinates (do not alter legacy geometry fields).
    residual_coordinate: Optional[float] = None
    response: Optional[float] = None

    # Legacy fields retained for compatibility with older tests/consumers.
    basin_center: Optional[List[float]] = None
    basin_radius: Optional[float] = None
    basin_velocity: Optional[float] = None
    compression: Optional[float] = None
    residual_norm: Optional[float] = None
    stability_score: Optional[float] = None
