"""
Rolling buffer for interpreted STRATA frames.
"""

from collections import deque
from datetime import datetime, timezone
from typing import Deque, Generic, List, Optional, TypeVar, Union

from strata.analysis.residuals import ResidualGeometry
from .basin_frame import BasinFrame
from .basin_geometry import BasinGeometryBuilder, BasinGeometryFrame, EVENT_RESET
from .strata_state import StrataState

T = TypeVar("T")


class StrataBuffer(Generic[T]):
    """Fixed-size append-only buffer preserving insertion order."""

    def __init__(self, maxlen: int = 500):
        self._frames: Deque[T] = deque(maxlen=maxlen)
        self._maxlen = maxlen

    def append(self, frame: T) -> None:
        self._frames.append(frame)

    def latest(self) -> Optional[T]:
        return self._frames[-1] if self._frames else None

    def history(self, n: Optional[int] = None) -> List[T]:
        if not self._frames:
            return []
        if n is None or n >= len(self._frames):
            return list(self._frames)
        if n <= 0:
            return []
        return list(self._frames)[-n:]

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def size(self) -> int:
        """Return current number of frames."""
        return len(self._frames)

    def clear(self) -> None:
        """Remove all frames."""
        self._frames.clear()


# ---------------------------------------------------------------------
# Shared buffers owned by STRATA
# ---------------------------------------------------------------------

STRATA_BUFFER: StrataBuffer[BasinFrame] = StrataBuffer()
strata_buffer = STRATA_BUFFER  # legacy alias

GEOMETRY_BUFFER: StrataBuffer[BasinGeometryFrame] = StrataBuffer()
geometry_buffer = GEOMETRY_BUFFER

_GEOMETRY_BUILDER = BasinGeometryBuilder(radius=2.0, trail_length=30)


# ---------------------------------------------------------------------
# Legacy frame builders
# ---------------------------------------------------------------------

def _legacy_frame_from_geometry(symbol: str, geometry: ResidualGeometry) -> BasinFrame:
    ts = geometry.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)

    centroid = (
        geometry.centroid.tolist()
        if hasattr(geometry.centroid, "tolist")
        else list(geometry.centroid)
    )

    return BasinFrame(
        timestamp=ts.isoformat(),
        symbol=symbol,
        domain_dim=geometry.domain_dim,
        residual_energy=float(geometry.residual_energy),
        residual_max=float(geometry.residual_max),
        centroid=[float(x) for x in centroid],
        fit_quality=float(geometry.fit_quality),
        basins=[],
    )


def _legacy_frame_from_state(state: StrataState) -> BasinFrame:
    ts = datetime.fromtimestamp(float(state.timestamp), tz=timezone.utc)

    residual_energy = abs(state.residual) if state.residual is not None else 0.0
    residual_max = (
        max(abs(state.normalized_residual), residual_energy)
        if state.normalized_residual is not None
        else residual_energy
    )

    residual_coord = state.normalized_residual
    response = state.response
    domain_dim = 2 if response is not None else 1  # 2D when response is present (including 0), else legacy 1D

    return BasinFrame(
        timestamp=ts.isoformat(),
        symbol=state.symbol,
        domain_dim=domain_dim,
        residual_energy=float(residual_energy),
        residual_max=float(residual_max),
        centroid=[float(state.basin_center)],
        fit_quality=1.0,
        basins=[],
        residual_coordinate=float(residual_coord) if residual_coord is not None else None,
        response=response if response is not None else None,
    )


# ---------------------------------------------------------------------
# Geometry frame builders
# ---------------------------------------------------------------------

def _geometry_frame_from_geometry(symbol: str, geometry: ResidualGeometry) -> BasinGeometryFrame:
    ts = geometry.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)

    centroid = (
        geometry.centroid.tolist()
        if hasattr(geometry.centroid, "tolist")
        else list(geometry.centroid)
    )
    center_hint = float(centroid[0]) if centroid else 0.0

    return _GEOMETRY_BUILDER.from_geometry(
        symbol=symbol,
        timestamp=ts,
        center_hint=center_hint,
        residual_energy=float(geometry.residual_energy),
        residual_max=float(geometry.residual_max),
    )


def _geometry_frame_from_state(state: StrataState) -> BasinGeometryFrame:
    residual_energy = abs(state.residual) if state.residual is not None else 0.0
    normalized_residual = state.normalized_residual or 0.0
    residual_max = max(abs(normalized_residual), residual_energy)

    return _GEOMETRY_BUILDER.from_state(
        timestamp=float(state.timestamp),
        symbol=state.symbol,
        normalized_residual=float(normalized_residual),
        basin_center=float(state.basin_center),
        basin_width=float(state.basin_width),
        basin_velocity=float(state.basin_velocity),
        residual_energy=float(residual_energy),
        residual_max=float(residual_max),
    )


# ---------------------------------------------------------------------
# Public update helpers
# ---------------------------------------------------------------------

def update_from_residual_geometry(
    symbol: str,
    geometry: ResidualGeometry,
    buffer: Optional[StrataBuffer[BasinFrame]] = None,
) -> BasinFrame:
    target = buffer or STRATA_BUFFER
    frame = _legacy_frame_from_geometry(symbol, geometry)
    target.append(frame)

    try:
        update_geometry_from_residual(symbol, geometry)
    except Exception:
        pass

    return frame


def update_from_state(
    state: StrataState,
    buffer: Optional[StrataBuffer[BasinFrame]] = None,
) -> BasinFrame:
    target = buffer or STRATA_BUFFER
    frame = _legacy_frame_from_state(state)
    target.append(frame)

    try:
        update_geometry_from_state(state)
    except Exception:
        pass

    return frame


def append_interpreted_state(
    state_or_geometry: Union[StrataState, ResidualGeometry],
    buffer: Optional[StrataBuffer[BasinFrame]] = None,
) -> None:
    target = buffer or STRATA_BUFFER

    if isinstance(state_or_geometry, ResidualGeometry):
        update_from_residual_geometry(
            symbol=getattr(state_or_geometry, "symbol", "UNKNOWN"),
            geometry=state_or_geometry,
            buffer=target,
        )
        return

    update_from_state(state_or_geometry, buffer=target)


def update_geometry_from_residual(
    symbol: str,
    geometry: ResidualGeometry,
    buffer: Optional[StrataBuffer[BasinGeometryFrame]] = None,
) -> BasinGeometryFrame:
    target = buffer or GEOMETRY_BUFFER
    frame = _geometry_frame_from_geometry(symbol, geometry)
    target.append(frame)
    return frame


def update_geometry_from_state(
    state: StrataState,
    buffer: Optional[StrataBuffer[BasinGeometryFrame]] = None,
) -> BasinGeometryFrame:
    target = buffer or GEOMETRY_BUFFER
    frame = _geometry_frame_from_state(state)
    target.append(frame)
    return frame


def append_geometry(
    state_or_geometry: Union[StrataState, ResidualGeometry],
    buffer: Optional[StrataBuffer[BasinGeometryFrame]] = None,
) -> None:
    target = buffer or GEOMETRY_BUFFER

    if isinstance(state_or_geometry, ResidualGeometry):
        update_geometry_from_residual(
            symbol=getattr(state_or_geometry, "symbol", "UNKNOWN"),
            geometry=state_or_geometry,
            buffer=target,
        )
        return

    update_geometry_from_state(state_or_geometry, buffer=target)


# ---------------------------------------------------------------------
# Backwards compatibility (MUST be last)
# ---------------------------------------------------------------------

StateBuffer = StrataBuffer

__all__ = [
    "StrataBuffer",
    "StateBuffer",
    "STRATA_BUFFER",
    "GEOMETRY_BUFFER",
]
