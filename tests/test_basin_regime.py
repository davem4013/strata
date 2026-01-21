"""Tests for basin regime classification."""
from strata.dynamics.basin_dynamics import BasinDynamics
from strata.regimes.basin_regime import BasinRegime, classify_basin_regime


def _dyn(
    stability_trend=0.0,
    compression_rate=0.0,
    center_drift_rate=0.0,
    stability_volatility=0.0,
    regime_risk_score=0.0,
    samples=10,
):
    return BasinDynamics(
        stability_trend=stability_trend,
        compression_rate=compression_rate,
        center_drift_rate=center_drift_rate,
        stability_volatility=stability_volatility,
        regime_risk_score=regime_risk_score,
        samples=samples,
    )


def test_compressing_rule_wins_with_confidence():
    dynamics = _dyn(compression_rate=0.2)
    result = classify_basin_regime(dynamics)

    assert result.regime == BasinRegime.COMPRESSING
    assert 0.0 <= result.confidence <= 1.0
    assert "compression" in result.rationale.lower()


def test_expanding_rule():
    dynamics = _dyn(compression_rate=-0.3)
    result = classify_basin_regime(dynamics)

    assert result.regime == BasinRegime.EXPANDING
    assert result.confidence > 0.6


def test_drifting_rule():
    dynamics = _dyn(center_drift_rate=0.4)
    result = classify_basin_regime(dynamics)

    assert result.regime == BasinRegime.DRIFTING
    assert "drift" in result.rationale.lower()


def test_volatile_rule():
    dynamics = _dyn(stability_volatility=0.3)
    result = classify_basin_regime(dynamics)

    assert result.regime == BasinRegime.VOLATILE
    assert result.confidence >= 0.4


def test_stable_rule_when_low_motion_and_noise():
    dynamics = _dyn(compression_rate=0.01, center_drift_rate=0.0, stability_volatility=0.05)
    result = classify_basin_regime(dynamics)

    assert result.regime == BasinRegime.STABLE
    assert result.confidence >= 0.5


def test_unknown_when_no_signals():
    dynamics = _dyn(samples=1)  # minimal data but no signals
    result = classify_basin_regime(dynamics)

    assert result.regime in {BasinRegime.UNKNOWN, BasinRegime.STABLE}
    assert 0.0 <= result.confidence <= 1.0
