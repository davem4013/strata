"""Lightweight immutable STRATA state snapshot."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StrataState:
    timestamp: float
    symbol: str
    spot: float

    surface_timestamp: Optional[float]
    atm_iv: Optional[float]

    residual: float
    normalized_residual: float

    basin_center: float
    basin_width: float
    basin_velocity: float

    position_state: str
    normalized_distance: float
    risk_score: float
