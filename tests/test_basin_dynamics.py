"""Unit tests for BasinDynamics engine."""
from strata.dynamics.basin_dynamics import BasinDynamics, compute_basin_dynamics
from strata.state.basin_state import ResidualBasin
from strata.state.state_buffer import BasinBuffer


def _basin(center: float, width: float, curvature: float = 0.5, sample_count: int = 10) -> ResidualBasin:
    return ResidualBasin(
        center=center,
        width=width,
        boundary_upper=center + width / 2.0,
        boundary_lower=center - width / 2.0,
        curvature=curvature,
        sample_count=sample_count,
        variance=(width / 4.0) ** 2,
    )


def test_stability_trend_sign():
    buf = BasinBuffer()
    buf.push(_basin(0.0, width=2.0, curvature=0.3, sample_count=5))
    buf.push(_basin(0.0, width=1.5, curvature=0.5, sample_count=10))
    buf.push(_basin(0.0, width=1.0, curvature=0.8, sample_count=20))

    dynamics = compute_basin_dynamics(buf)

    assert dynamics.stability_trend > 0


def test_compression_and_drift_detection():
    buf = BasinBuffer()
    buf.push(_basin(0.0, width=2.0))
    buf.push(_basin(0.5, width=1.0))
    buf.push(_basin(1.0, width=0.5))

    dynamics = compute_basin_dynamics(buf)

    assert dynamics.compression_rate > 0  # widths shrink → compression
    assert dynamics.center_drift_rate > 0  # centers increase


def test_stability_volatility_sensitivity():
    buf = BasinBuffer()
    buf.push(_basin(0.0, width=2.0, curvature=0.2))
    buf.push(_basin(0.1, width=1.0, curvature=0.8))
    buf.push(_basin(0.0, width=1.8, curvature=0.3))

    dynamics = compute_basin_dynamics(buf)

    assert dynamics.stability_volatility > 0


def test_risk_score_bounds():
    buf = BasinBuffer()
    buf.push(_basin(0.0, width=2.0))
    buf.push(_basin(1.0, width=0.5))

    dynamics = compute_basin_dynamics(buf)

    assert isinstance(dynamics, BasinDynamics)
    assert 0.0 <= dynamics.regime_risk_score <= 1.0
