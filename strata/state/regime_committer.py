"""Adapter that maps StrataState buffer updates to RegimeState buffer."""
from __future__ import annotations

import logging
from typing import Optional

from .regime_buffer import RegimeBuffer
from .regime_tracker import RegimeState, infer_regime
from .state_buffer import StrataStateBuffer

logger = logging.getLogger(__name__)


class RegimeCommitter:
    """Lightweight committer that derives RegimeState from StrataState history."""

    def __init__(
        self,
        symbol: str,
        state_buffer: StrataStateBuffer,
        regime_buffer: RegimeBuffer,
        window: int = 20,
    ):
        self.symbol = symbol
        self.state_buffer = state_buffer
        self.regime_buffer = regime_buffer
        self.window = window
        self._last_timestamp: Optional[float] = None

    def step(self) -> Optional[RegimeState]:
        """Process the latest StrataState if new; return RegimeState if appended."""
        latest = self.state_buffer.latest()
        if latest is None:
            return None

        if self._last_timestamp is not None and latest.timestamp <= self._last_timestamp:
            return None

        history = self.state_buffer.history(self.window)
        regime_state = infer_regime(history, self.symbol)
        if regime_state:
            self.regime_buffer.append(regime_state)
            self._last_timestamp = latest.timestamp
            logger.info(
                "Regime update %s label=%s conf=%.2f buffer=%d",
                self.symbol,
                regime_state.label.value,
                regime_state.confidence,
                self.regime_buffer.size(),
            )
        return regime_state
