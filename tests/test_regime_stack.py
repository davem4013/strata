"""Tests for multi-timescale regime fusion."""
from strata.regimes.basin_regime import BasinRegime, RegimeClassification
from strata.regimes.regime_stack import RegimeStack, fuse_regime_stack


def _rc(regime, confidence, rationale=""):
    return RegimeClassification(regime=regime, confidence=confidence, rationale=rationale)


def test_dominant_regime_by_weighted_confidence():
    regimes = {
        "1m": _rc(BasinRegime.STABLE, 0.9),
        "1h": _rc(BasinRegime.COMPRESSING, 0.4),  # higher weight should dominate
    }

    stack = fuse_regime_stack(regimes)

    assert isinstance(stack, RegimeStack)
    assert stack.dominant_regime == BasinRegime.COMPRESSING
    assert 0.0 <= stack.alignment_score <= 1.0


def test_alignment_and_conflict_flags():
    regimes = {
        "1m": _rc(BasinRegime.STABLE, 0.9),
        "5m": _rc(BasinRegime.STABLE, 0.8),
        "15m": _rc(BasinRegime.STABLE, 0.7),
    }
    stack = fuse_regime_stack(regimes)
    assert stack.dominant_regime == BasinRegime.STABLE
    assert stack.conflict is False
    assert stack.alignment_score > 0.6


def test_conflict_when_alignment_low():
    regimes = {
        "1m": _rc(BasinRegime.STABLE, 0.6),
        "5m": _rc(BasinRegime.EXPANDING, 0.6),
    }
    stack = fuse_regime_stack(regimes)
    assert stack.conflict is True
    assert 0.0 <= stack.alignment_score <= 1.0


def test_rationale_is_preserved():
    regimes = {
        "1m": _rc(BasinRegime.DRIFTING, 0.5, "short-term drift"),
        "1h": _rc(BasinRegime.DRIFTING, 0.6, "long-term drift"),
    }
    stack = fuse_regime_stack(regimes)
    assert stack.rationale["1m"] == "short-term drift"
    assert stack.rationale["1h"] == "long-term drift"
    assert stack.dominant_regime == BasinRegime.DRIFTING
