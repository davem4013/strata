"""Adapter that maps StrataState buffer updates to RegimeState buffer."""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Optional

from strata.state.regime_buffer import RegimeBuffer
from strata.state.regime_tracker import RegimeState, infer_regime
from strata.state.basin_frame import BasinFrame
from strata.state.strata_buffer import strata_buffer
from strata.state.state_buffer import StrataStateBuffer
from strata.state.strata_state import StrataState

logger = logging.getLogger(__name__)


class RegimeCommitter:
    """Lightweight committer that derives RegimeState from StrataState history."""

    def __init__(
        self,
        symbol: str,
        state_buffer: StrataStateBuffer,
        regime_buffer: RegimeBuffer,
        window: int = 20,
        geometry_cadence: int = 0,
    ):
        self.symbol = symbol
        self.state_buffer = state_buffer
        self.regime_buffer = regime_buffer
        self.window = window
        self.geometry_cadence = max(int(geometry_cadence or 0), 0)
        self.geometry_cadence = 1
        self._last_timestamp: Optional[float] = None
        self._step_index: int = 0

    def step(self) -> Optional[RegimeState]:
        """Process the latest StrataState if new; return RegimeState if appended."""
        logger.warning("REGIME STEPPER CALLED: %s", self.symbol)
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
            self._step_index += 1
            logger.info(
                "Regime update %s label=%s conf=%.2f buffer=%d",
                self.symbol,
                regime_state.label.value,
                regime_state.confidence,
                self.regime_buffer.size(),
            )
            self._emit_geometry_frame(latest, regime_state)
        return regime_state

    def _emit_geometry_frame(self, latest_state: StrataState, regime_state: Optional[RegimeState]) -> None:
        """Emit a minimal BasinFrame to the shared buffer on cadence."""
        if self.geometry_cadence <= 0:
            return
        if self._step_index % self.geometry_cadence != 0:
            return

        try:
            logger.warning("GEOMETRY EMIT %s step=%d", self.symbol, self._step_index)
            basin_frame = self._build_basin_frame(latest_state, regime_state)
            strata_buffer.append(basin_frame)
            logger.warning("STRATA BUFFER SIZE NOW %d", strata_buffer.size())
        except Exception as exc:
            logger.debug("Geometry emission skipped: %s", exc)

    def _build_basin_frame(self, latest_state: StrataState, regime_state: Optional[RegimeState]) -> BasinFrame:
        """
        Build a BasinFrame from the interpreted StrataState, including regime metadata when available.
        """
        frame_factory = getattr(BasinFrame, "from_state", None)
        if callable(frame_factory):
            try:
                return frame_factory(latest_state, regime=regime_state)
            except TypeError:
                return frame_factory(latest_state)

        ts = datetime.fromtimestamp(float(latest_state.timestamp), tz=timezone.utc)
        residual_energy = abs(latest_state.residual) if latest_state.residual is not None else 0.0
        normalized_residual = abs(latest_state.normalized_residual) if latest_state.normalized_residual is not None else 0.0
        residual_max = max(normalized_residual, residual_energy)
        domain_dim = 2 if latest_state.response is not None else 1  # 2D when response is present (including 0), otherwise legacy 1D

        basin_entry = {
            "center": latest_state.basin_center,
            "width": latest_state.basin_width,
            "velocity": latest_state.basin_velocity,
            "normalized_residual": latest_state.normalized_residual,
            "position_state": latest_state.position_state,
            "risk_score": latest_state.risk_score,
        }
        if regime_state:
            basin_entry["regime"] = regime_state.label.value
            basin_entry["regime_confidence"] = regime_state.confidence

        return BasinFrame(
            timestamp=ts.isoformat(),
            symbol=latest_state.symbol,
            domain_dim=domain_dim,
            residual_energy=float(residual_energy),
            residual_max=float(residual_max),
            centroid=[float(latest_state.basin_center)],
            fit_quality=1.0,
            basins=[basin_entry],
            residual_coordinate=float(latest_state.normalized_residual) if latest_state.normalized_residual is not None else None,
            response=latest_state.response if latest_state.response is not None else None,
        )
