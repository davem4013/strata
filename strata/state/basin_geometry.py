"""Canonical basin geometry frame and builder for STRATA."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, List, Optional, Sequence, Tuple
import math


# Event labels emitted alongside geometry frames
EVENT_BOUNDARY_CROSS = "boundary_cross"
EVENT_RESET = "reset"
EVENT_NONE = "none"


@dataclass(frozen=True)
class BasinGeometryFrame:
    """
    Analysis-complete basin geometry snapshot.

    All geometry is expressed in normalized space (sigma units), projected into a
    stable 2D basis with a fixed radius for rendering. Velocity is derived from
    successive projections; the trail preserves recent motion for smooth drawing.
    """

    timestamp: str
    symbol: str

    proj_2d: List[float]
    vel_2d: List[float]
    speed: float

    radius: float
    dist: float
    distance_to_edge: float
    outside: bool
    trail_2d: List[List[float]]

    residual_energy: float
    residual_max: float

    event: str = EVENT_NONE

    # Additional context retained for debugging/consumers
    basin_center: float = 0.0
    basin_width: float = 0.0
    normalized_residual: float = 0.0


class BasinGeometryBuilder:
    """Stateful builder that maintains projection stability and kinematics."""

    def __init__(self, radius: float = 2.0, trail_length: int = 30) -> None:
        self.radius = float(radius)
        self._trail: Deque[Tuple[float, float]] = deque(maxlen=int(trail_length))
        self._prev_proj: Optional[Tuple[float, float]] = None
        self._prev_ts: Optional[float] = None
        self._prev_outside: Optional[bool] = None
        self._velocity_scale: float = 1.0

    def reset(self) -> None:
        """Clear internal buffers and emit a reset event on next build."""
        self._trail.clear()
        self._prev_proj = None
        self._prev_ts = None
        self._prev_outside = None
        self._velocity_scale = 1.0

    def from_state(
        self,
        *,
        timestamp: float,
        symbol: str,
        normalized_residual: float,
        basin_center: float,
        basin_width: float,
        basin_velocity: float,
        residual_energy: float,
        residual_max: float,
        event_hint: Optional[str] = None,
    ) -> BasinGeometryFrame:
        """
        Build a BasinGeometryFrame from scalar basin state.

        Inputs are expected in sigma space for stability; the projection maps the
        scalar basin offset + velocity into a fixed 2D basis with diagonal scaling.
        """
        ts_epoch = float(timestamp)
        ts = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).isoformat()

        offset = float(normalized_residual) - float(basin_center)
        half_width = float(basin_width) / 2.0
        boundary = half_width if half_width > 0 else self.radius

        distance_to_edge = boundary - abs(offset)
        outside = distance_to_edge < 0

        # Update velocity scaling with gentle smoothing to prevent oscillations.
        abs_vel = abs(float(basin_velocity))
        self._velocity_scale = max(1e-3, 0.9 * self._velocity_scale + 0.1 * abs_vel)
        vel_scale = self._velocity_scale

        # Stable 2D projection: fixed axes [sigma offset, basin velocity].
        proj_x = (offset / boundary) * self.radius if boundary > 0 else 0.0
        proj_y = (float(basin_velocity) / vel_scale) * self.radius if vel_scale > 0 else 0.0
        proj_2d = (float(proj_x), float(proj_y))

        # Append to trail before velocity so we preserve the current point.
        self._trail.append(proj_2d)
        trail_snapshot = [[float(x), float(y)] for x, y in self._trail]

        if self._prev_proj is None or self._prev_ts is None:
            vel_2d = (0.0, 0.0)
            speed = 0.0
            event = event_hint or EVENT_RESET
        else:
            dt = max(ts_epoch - self._prev_ts, 1e-6)
            vel_2d = (
                (proj_2d[0] - self._prev_proj[0]) / dt,
                (proj_2d[1] - self._prev_proj[1]) / dt,
            )
            speed = math.hypot(*vel_2d)
            if event_hint:
                event = event_hint
            else:
                event = EVENT_BOUNDARY_CROSS if outside != self._prev_outside else EVENT_NONE

        dist = abs(offset)

        frame = BasinGeometryFrame(
            timestamp=ts,
            symbol=symbol,
            proj_2d=[proj_2d[0], proj_2d[1]],
            vel_2d=[vel_2d[0], vel_2d[1]],
            speed=float(speed),
            radius=self.radius,
            dist=float(dist),
            distance_to_edge=float(distance_to_edge),
            outside=bool(outside),
            trail_2d=trail_snapshot,
            residual_energy=float(residual_energy),
            residual_max=float(residual_max),
            event=event,
            basin_center=float(basin_center),
            basin_width=float(basin_width),
            normalized_residual=float(normalized_residual),
        )

        # Update state for next step.
        self._prev_proj = proj_2d
        self._prev_ts = ts_epoch
        self._prev_outside = outside

        return frame

    def from_geometry(
        self,
        *,
        symbol: str,
        timestamp: datetime,
        center_hint: float = 0.0,
        residual_energy: float = 0.0,
        residual_max: float = 0.0,
    ) -> BasinGeometryFrame:
        """
        Fallback builder for ResidualGeometry summaries when full state is absent.

        Uses centroid as a proxy for position and assumes a fixed basin width
        anchored to the canonical radius.
        """
        ts_epoch = timestamp.timestamp()
        return self.from_state(
            timestamp=ts_epoch,
            symbol=symbol,
            normalized_residual=center_hint,
            basin_center=0.0,
            basin_width=self.radius * 2.0,
            basin_velocity=0.0,
            residual_energy=residual_energy,
            residual_max=residual_max,
            event_hint=EVENT_RESET,
        )

