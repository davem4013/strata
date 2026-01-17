"""Interpreted STRATA basin snapshot."""
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class BasinFrame:
    """
    Lightweight, JSON-friendly basin state.

    All fields are numeric or list-of-numeric to keep serialization simple.
    """

    timestamp: float
    basin_center: List[float]
    basin_radius: Optional[float]
    basin_velocity: Optional[float]
    compression: Optional[float]
    residual_norm: Optional[float]
    stability_score: Optional[float] = None
