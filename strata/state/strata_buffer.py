"""Rolling buffer for interpreted STRATA frames."""
from collections import deque
from typing import Deque, Generic, List, Optional, TypeVar

from .basin_frame import BasinFrame
from .strata_state import StrataState

T = TypeVar("T", bound=BasinFrame)


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


# Shared buffer instance owned by STRATA
STRATA_BUFFER: StrataBuffer[BasinFrame] = StrataBuffer()


def to_basin_frame(state: StrataState) -> BasinFrame:
    """Interpret a StrataState into a BasinFrame without altering underlying math."""
    center_val = state.basin_center
    center = [center_val] if center_val is not None else []

    radius = state.basin_width / 2.0 if state.basin_width is not None else None
    velocity = state.basin_velocity if state.basin_velocity is not None else None

    # Simple compression proxy: narrower basins imply higher compression; avoid division by zero.
    compression = None
    if state.basin_width:
        compression = 1.0 / float(state.basin_width) if state.basin_width != 0 else None

    residual_norm = abs(state.normalized_residual) if state.normalized_residual is not None else None

    return BasinFrame(
        timestamp=float(state.timestamp),
        basin_center=center,
        basin_radius=radius,
        basin_velocity=velocity,
        compression=compression,
        residual_norm=residual_norm,
        stability_score=None,
    )


def append_interpreted_state(state: StrataState, buffer: Optional[StrataBuffer[BasinFrame]] = None) -> None:
    """Append an interpreted BasinFrame to the provided or shared buffer."""
    target = buffer or STRATA_BUFFER
    frame = to_basin_frame(state)
    target.append(frame)
