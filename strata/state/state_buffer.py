"""Lightweight rolling window buffer for state frames."""
from collections import deque
from threading import Lock
from typing import Deque, Generic, List, Optional, TypeVar

from .basin_state import ResidualBasin

T = TypeVar("T")


class StrataStateBuffer(Generic[T]):
    """
    Fixed-size buffer that preserves insertion order and overwrites the oldest frame when full.

    Provides simple accessors for pushing frames, retrieving the latest frame, and getting a
    window of recent frames without any domain-specific logic.
    """

    def __init__(self, maxlen: int = 5000):
        self._frames: Deque[T] = deque(maxlen=maxlen)
        self._maxlen = maxlen
        self._lock = Lock()

    def push(self, frame: T) -> None:
        """Insert a frame; evicts the oldest when capacity is exceeded."""
        with self._lock:
            self._frames.append(frame)

    def get_window(self, n: Optional[int] = None) -> List[T]:
        """
        Return up to the last n frames in time order (oldest→newest).

        Args:
            n: Optional limit. None returns the full window.
        """
        with self._lock:
            if not self._frames:
                return []
            if n is None or n >= len(self._frames):
                return list(self._frames)
            if n <= 0:
                return []
            return list(self._frames)[-n:]

    def get_latest(self) -> Optional[T]:
        """Return the most recent frame, or None if empty."""
        with self._lock:
            return self._frames[-1] if self._frames else None

    # Compatibility shims for existing callers
    def append(self, frame: T) -> None:
        self.push(frame)

    def history(self, n: Optional[int] = None) -> List[T]:
        return self.get_window(n)

    def latest(self) -> Optional[T]:
        return self.get_latest()

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._frames) == 0

    def size(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def maxlen(self) -> int:
        return self._maxlen

# Public API alias
StateBuffer = StrataStateBuffer
# Basin geometry buffer alias (stores ResidualBasin objects in memory)
BasinBuffer = StrataStateBuffer[ResidualBasin]
