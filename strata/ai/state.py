"""Structured market state used for regime and prompt context."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from strata.analysis.vol_spine import ForwardVolPoint


@dataclass
class MarketState:
    """
    Core STRATA market state vector.

    forward_vol_spine and spine_residuals elevate term-structure stress into
    the regime engine alongside legacy surface/greeks fields.
    """

    iv_surface: Optional[Dict[str, Any]] = None
    delta_pressure: Optional[float] = None
    gamma_exposure: Optional[float] = None
    sd_bands: Optional[Dict[str, Any]] = None
    forward_vol_spine: List[ForwardVolPoint] = field(default_factory=list)
    spine_residuals: List[Dict[str, float]] = field(default_factory=list)

    def to_compact_dict(self) -> Dict[str, Any]:
        """Flatten for prompts/telemetry."""
        return {
            "iv_surface": self.iv_surface,
            "delta_pressure": self.delta_pressure,
            "gamma_exposure": self.gamma_exposure,
            "sd_bands": self.sd_bands,
            "forward_vol_spine": [
                p.to_dict() for p in self.forward_vol_spine
            ],
            "spine_residuals": self.spine_residuals,
        }
