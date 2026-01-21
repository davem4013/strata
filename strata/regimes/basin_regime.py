"""Deterministic basin regime classification from basin dynamics."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

from strata.dynamics.basin_dynamics import BasinDynamics


class BasinRegime(str, Enum):
    STABLE = "stable"
    COMPRESSING = "compressing"
    EXPANDING = "expanding"
    DRIFTING = "drifting"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RegimeClassification:
    regime: BasinRegime
    confidence: float
    rationale: str


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def classify_basin_regime(dynamics: BasinDynamics) -> RegimeClassification:
    """
    Classify basin regime using ordered rules for determinism and explainability.
    """
    checks: List[RegimeClassification] = []

    # Strong compression signal
    if dynamics.compression_rate > 0.05:
        confidence = _clamp01(0.6 + min(dynamics.compression_rate, 1.0) * 0.4)
        checks.append(
            RegimeClassification(
                regime=BasinRegime.COMPRESSING,
                confidence=confidence,
                rationale=f"Widths shrinking (compression_rate={dynamics.compression_rate:.3f})",
            )
        )

    # Strong expansion signal
    if dynamics.compression_rate < -0.05:
        confidence = _clamp01(0.6 + min(abs(dynamics.compression_rate), 1.0) * 0.4)
        checks.append(
            RegimeClassification(
                regime=BasinRegime.EXPANDING,
                confidence=confidence,
                rationale=f"Widths widening (compression_rate={dynamics.compression_rate:.3f})",
            )
        )

    # Drift dominates
    if abs(dynamics.center_drift_rate) > 0.05:
        confidence = _clamp01(0.5 + min(abs(dynamics.center_drift_rate), 1.0) * 0.5)
        direction = "up" if dynamics.center_drift_rate > 0 else "down"
        checks.append(
            RegimeClassification(
                regime=BasinRegime.DRIFTING,
                confidence=confidence,
                rationale=f"Center drifting {direction} (center_drift_rate={dynamics.center_drift_rate:.3f})",
            )
        )

    # Volatility in stability scores
    if dynamics.stability_volatility > 0.1:
        confidence = _clamp01(0.4 + min(dynamics.stability_volatility, 1.0) * 0.6)
        checks.append(
            RegimeClassification(
                regime=BasinRegime.VOLATILE,
                confidence=confidence,
                rationale=f"Stability volatility elevated (stability_volatility={dynamics.stability_volatility:.3f})",
            )
        )

    # Baseline stable when low movement and low noise
    if (
        abs(dynamics.compression_rate) <= 0.05
        and abs(dynamics.center_drift_rate) <= 0.05
        and dynamics.stability_volatility <= 0.1
    ):
        confidence = _clamp01(0.5 + max(0.0, 0.1 - dynamics.stability_volatility) * 5.0)
        checks.append(
            RegimeClassification(
                regime=BasinRegime.STABLE,
                confidence=confidence,
                rationale="Widths and center stable with low stability volatility",
            )
        )

    if not checks:
        return RegimeClassification(
            regime=BasinRegime.UNKNOWN,
            confidence=0.2,
            rationale="Insufficient signal for classification",
        )

    # Deterministic selection: highest confidence, then first-in-rule-order.
    checks.sort(key=lambda c: c.confidence, reverse=True)
    best = checks[0]
    return RegimeClassification(
        regime=best.regime,
        confidence=_clamp01(best.confidence),
        rationale=best.rationale,
    )
