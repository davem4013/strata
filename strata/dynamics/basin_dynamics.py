"""Immutable basin dynamics interpretation built from in-memory buffers."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev
from typing import List

from strata.state.basin_state import BasinEvolution, ResidualBasin, compare_basins
from strata.state.state_buffer import BasinBuffer


@dataclass(frozen=True)
class BasinDynamics:
    stability_trend: float
    compression_rate: float
    center_drift_rate: float
    stability_volatility: float
    regime_risk_score: float
    samples: int


def _stability_scores(history: List[ResidualBasin]) -> List[float]:
    return [b.stability_score() for b in history]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_basin_dynamics(buffer: BasinBuffer) -> BasinDynamics:
    """
    Interpret basin evolution over time using immutable ResidualBasin snapshots.

    Uses compare_basins for pairwise evolution without mutating buffer contents.
    """
    history = buffer.history()
    sample_count = len(history)

    if sample_count < 2:
        return BasinDynamics(
            stability_trend=0.0,
            compression_rate=0.0,
            center_drift_rate=0.0,
            stability_volatility=0.0,
            regime_risk_score=0.0,
            samples=sample_count,
        )

    stability_scores = _stability_scores(history)
    stability_trend = float(stability_scores[-1] - stability_scores[0])
    stability_volatility = float(pstdev(stability_scores)) if len(stability_scores) > 1 else 0.0

    compression_changes: List[float] = []
    drift_changes: List[float] = []

    for before, after in zip(history[:-1], history[1:]):
        evolution: BasinEvolution = compare_basins(before, after)
        compression_effect = evolution.compression - evolution.expansion
        compression_changes.append(compression_effect)
        drift_changes.append(evolution.center_delta)

    compression_rate = float(sum(compression_changes) / len(compression_changes)) if compression_changes else 0.0
    center_drift_rate = float(sum(drift_changes) / len(drift_changes)) if drift_changes else 0.0

    # Risk increases with absolute compression/expansion, center drift, and stability noise.
    compression_risk = min(abs(compression_rate), 1.0)
    drift_risk = min(abs(center_drift_rate), 1.0)
    volatility_risk = min(stability_volatility, 1.0)

    regime_risk_score = _clamp01(0.4 * compression_risk + 0.4 * drift_risk + 0.2 * volatility_risk)

    return BasinDynamics(
        stability_trend=stability_trend,
        compression_rate=compression_rate,
        center_drift_rate=center_drift_rate,
        stability_volatility=stability_volatility,
        regime_risk_score=regime_risk_score,
        samples=sample_count,
    )
