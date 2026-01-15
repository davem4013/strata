"""Simple in-memory ring buffer for STRATA state."""
from collections import deque
from threading import Lock
from typing import List, Optional
import logging

from .strata_state import StrataState

logger = logging.getLogger(__name__)


class StrataStateBuffer:
    """Bounded append-only buffer for StrataState objects."""

    def __init__(self, maxlen: int = 5000):
        self._buffer: deque[StrataState] = deque(maxlen=maxlen)
        self._maxlen = maxlen
        self._lock = Lock()

    def append(self, state: StrataState) -> None:
        """Append a new state; drop non-monotonic snapshots."""
        with self._lock:
            if self._buffer and state.timestamp <= self._buffer[-1].timestamp:
                logger.warning(
                    "Non-monotonic timestamp detected: new=%s last=%s — dropping snapshot",
                    state.timestamp,
                    self._buffer[-1].timestamp,
                )
                return
            self._buffer.append(state)

    def latest(self) -> Optional[StrataState]:
        """Return the most recent state (or None)."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def history(self, n: Optional[int] = None) -> List[StrataState]:
        """
        Return a copy of the last n states (oldest→newest).

        Args:
            n: Number of items to return. None returns full buffer.
        """
        with self._lock:
            if not self._buffer:
                return []
            if n is None:
                return list(self._buffer)
            if n <= 0:
                return []
            return list(self._buffer)[-n:]

    def is_empty(self) -> bool:
        """True if buffer has no entries."""
        with self._lock:
            return len(self._buffer) == 0

    def size(self) -> int:
        """Current buffer length."""
        with self._lock:
            return len(self._buffer)

    @property
    def maxlen(self) -> int:
        return self._maxlen
