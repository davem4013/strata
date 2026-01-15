"""Simple in-memory ring buffer for STRATA state."""
from collections import deque
from typing import List, Optional
import logging

from .strata_state import StrataState

logger = logging.getLogger(__name__)


class StrataStateBuffer:
    """Bounded append-only buffer for StrataState objects."""

    def __init__(self, maxlen: int = 5000):
        self._buffer: deque[StrataState] = deque(maxlen=maxlen)
        self._maxlen = maxlen

    def append(self, state: StrataState) -> None:
        """Append a new state, warning on non-monotonic timestamps."""
        if self._buffer and state.timestamp <= self._buffer[-1].timestamp:
            logger.warning(
                "Non-monotonic timestamp detected: new=%s last=%s",
                state.timestamp,
                self._buffer[-1].timestamp,
            )
        self._buffer.append(state)

    def latest(self) -> Optional[StrataState]:
        """Return the most recent state (or None)."""
        return self._buffer[-1] if self._buffer else None

    def history(self, n: int) -> List[StrataState]:
        """Return the last n states (oldest→newest)."""
        if n <= 0:
            return []
        return list(self._buffer)[-n:]

    def size(self) -> int:
        """Current buffer length."""
        return len(self._buffer)

    @property
    def maxlen(self) -> int:
        return self._maxlen
