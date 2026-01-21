"""Multi-timescale fusion of basin regimes with deterministic weighting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .basin_regime import BasinRegime, RegimeClassification

_TIMESCALE_WEIGHTS: Dict[str, float] = {
    "1m": 1.0,
    "5m": 1.5,
    "15m": 2.0,
    "30m": 2.5,
    "1h": 3.0,
}


def _weight(timescale: str) -> float:
    return _TIMESCALE_WEIGHTS.get(timescale, 1.0)


@dataclass(frozen=True)
class RegimeStack:
    regimes: Dict[str, RegimeClassification]
    dominant_regime: BasinRegime
    alignment_score: float
    conflict: bool
    rationale: Dict[str, str]


def fuse_regime_stack(regimes_by_timescale: Dict[str, RegimeClassification]) -> RegimeStack:
    if not regimes_by_timescale:
        return RegimeStack(
            regimes={},
            dominant_regime=BasinRegime.UNKNOWN,
            alignment_score=0.0,
            conflict=True,
            rationale={},
        )

    weighted_scores: Dict[BasinRegime, float] = {}
    rationale: Dict[str, str] = {}

    for ts, classification in regimes_by_timescale.items():
        rationale[ts] = classification.rationale
        weight = _weight(ts)
        weighted = weight * classification.confidence
        weighted_scores[classification.regime] = weighted_scores.get(classification.regime, 0.0) + weighted

    # Deterministic dominant: max weighted score, tie-broken by regime name
    dominant = sorted(weighted_scores.items(), key=lambda kv: (-kv[1], kv[0].value))[0][0]

    total_weighted_conf = sum(weighted_scores.values()) or 1.0
    alignment_score = weighted_scores.get(dominant, 0.0) / total_weighted_conf

    # Conflict if low alignment or top-2 regimes are close
    sorted_scores = sorted(weighted_scores.values(), reverse=True)
    top = sorted_scores[0]
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    disagreement = second >= 0.85 * top
    conflict = alignment_score <= 0.6 or disagreement

    return RegimeStack(
        regimes=dict(regimes_by_timescale),
        dominant_regime=dominant,
        alignment_score=max(0.0, min(1.0, alignment_score)),
        conflict=conflict,
        rationale=rationale,
    )
