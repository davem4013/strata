"""Ring buffer for RegimeState snapshots."""
from collections import deque
from typing import List, Optional
import logging

from .regime_tracker import RegimeState

logger = logging.getLogger(__name__)


class RegimeBuffer:
    """Bounded append-only buffer for RegimeState objects."""

    def __init__(self, maxlen: int = 2000):
        self._buffer: deque[RegimeState] = deque(maxlen=maxlen)
        self._maxlen = maxlen

    def append(self, state: RegimeState) -> None:
        if self._buffer and state.timestamp <= self._buffer[-1].timestamp:
            logger.warning(
                "Non-monotonic regime timestamp detected: new=%s last=%s",
                state.timestamp,
                self._buffer[-1].timestamp,
            )
        self._buffer.append(state)

    def latest(self) -> Optional[RegimeState]:
        return self._buffer[-1] if self._buffer else None

    def history(self, n: Optional[int] = None) -> List[RegimeState]:
        if not self._buffer:
            return []
        if n is None:
            return list(self._buffer)
        if n <= 0:
            return []
        return list(self._buffer)[-n:]

    def size(self) -> int:
        return len(self._buffer)

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def clear(self) -> None:
        """Remove all stored regime states."""
        self._buffer.clear()
