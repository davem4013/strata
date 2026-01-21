"""
Rolling buffers for STRATA state.
"""

from collections import deque
from typing import Deque, Generic, List, Optional, TypeVar

from .strata_state import StrataState

T = TypeVar("T")


class StrataBuffer(Generic[T]):
    """Generic fixed-size append-only ring buffer."""

    def __init__(self, maxlen: int = 500):
        self._frames: Deque[T] = deque(maxlen=maxlen)
        self._maxlen = maxlen

    def append(self, item: T) -> None:
        self._frames.append(item)

    def latest(self) -> Optional[T]:
        return self._frames[-1] if self._frames else None

    def history(self, n: Optional[int] = None) -> List[T]:
        if not self._frames:
            return []
        if n is None or n >= len(self._frames):
            return list(self._frames)
        return list(self._frames)[-n:]

    def size(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def push(self, item):
        """Legacy alias for append()."""
        self.append(item)


class StrataStateBuffer(StrataBuffer[StrataState]):
    """Ring buffer for interpreted StrataState objects."""
    def get_window(self, n: int):
        """
        Return the last n+1 elements as a list.
        Legacy helper used by regression window logic.
        """
        if n < 0:
            return []
        if n >= len(self._frames):
            return list(self._frames)
        return list(self._frames)[-(n + 1):]


StateBuffer = StrataStateBuffer
# Compatibility alias for legacy imports/tests; use StrataStateBuffer going forward.
BasinBuffer = StrataStateBuffer
