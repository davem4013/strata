"""Deterministic regime tracking over in-memory StrataState history."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import logging
import math

from .strata_state import StrataState

logger = logging.getLogger(__name__)

# Thresholds (tunable constants, deterministic)
RESIDUAL_VEL_LOW = 0.002
RESIDUAL_VEL_HIGH = 0.01
CENTER_VEL_LOW = 0.002
WIDTH_CHANGE_SMALL = 0.02  # 2% change
WIDTH_CHANGE_MED = 0.1     # 10% change
RISK_DELTA_SMALL = 0.02
RISK_DELTA_MED = 0.05


class RegimeLabel(Enum):
    STABLE = "stable"
    DRIFTING = "drifting"
    COMPRESSING = "compressing"
    EXPANDING = "expanding"
    TRANSITION = "transition"


@dataclass(frozen=True)
class RegimeState:
    timestamp: float
    symbol: str
    label: RegimeLabel
    confidence: float
    metrics: Dict[str, float]


def _safe_delta(a: float, b: float) -> float:
    return float(b - a)


def _pct_change(a: float, b: float) -> float:
    if abs(a) < 1e-9:
        return 0.0
    return float((b - a) / abs(a))


def _window_stats(states: List[StrataState]) -> Tuple[float, float, float, float, float, float]:
    first, last = states[0], states[-1]
    time_delta = max(last.timestamp - first.timestamp, 1e-6)

    residual_vel = _safe_delta(first.normalized_residual, last.normalized_residual) / time_delta
    center_vel = _safe_delta(first.basin_center, last.basin_center) / time_delta
    width_change_pct = _pct_change(first.basin_width, last.basin_width)
    risk_delta = _safe_delta(first.risk_score, last.risk_score)
    risk_rate = risk_delta / time_delta

    width_rate = _safe_delta(first.basin_width, last.basin_width) / time_delta

    return residual_vel, center_vel, width_change_pct, risk_delta, risk_rate, width_rate


def infer_regime(states: List[StrataState], symbol: str) -> Optional[RegimeState]:
    if len(states) < 2:
        return None

    residual_vel, center_vel, width_change_pct, risk_delta, risk_rate, width_rate = _window_stats(states)

    # Base metrics for transparency
    metrics = {
        "residual_velocity": residual_vel,
        "center_velocity": center_vel,
        "width_change_pct": width_change_pct,
        "width_rate": width_rate,
        "risk_delta": risk_delta,
        "risk_rate": risk_rate,
    }

    labels = []

    if (
        abs(residual_vel) < RESIDUAL_VEL_LOW
        and abs(center_vel) < CENTER_VEL_LOW
        and abs(width_change_pct) < WIDTH_CHANGE_SMALL
        and abs(risk_delta) < RISK_DELTA_SMALL
    ):
        labels.append(RegimeLabel.STABLE)

    if abs(residual_vel) >= RESIDUAL_VEL_HIGH and abs(width_change_pct) < WIDTH_CHANGE_MED:
        labels.append(RegimeLabel.DRIFTING)

    if width_change_pct < -WIDTH_CHANGE_MED and risk_delta > RISK_DELTA_SMALL:
        labels.append(RegimeLabel.COMPRESSING)

    if width_change_pct > WIDTH_CHANGE_MED:
        labels.append(RegimeLabel.EXPANDING)

    # Divergent signals or sign flips between residual and center movement → transition
    if math.copysign(1.0, residual_vel or 1.0) != math.copysign(1.0, center_vel or 1.0):
        labels.append(RegimeLabel.TRANSITION)

    if not labels:
        labels.append(RegimeLabel.TRANSITION)

    # Resolve conflicts: multiple distinct labels => transition
    final_label = labels[0]
    if len(set(labels)) > 1:
        final_label = RegimeLabel.TRANSITION

    # Confidence: degrade as velocity/width/risk move away from stable thresholds
    if final_label == RegimeLabel.STABLE:
        confidence = 0.9
    elif final_label in (RegimeLabel.DRIFTING, RegimeLabel.EXPANDING, RegimeLabel.COMPRESSING):
        confidence = 0.8
    else:
        confidence = 0.6

    return RegimeState(
        timestamp=states[-1].timestamp,
        symbol=symbol,
        label=final_label,
        confidence=confidence,
        metrics=metrics,
    )
