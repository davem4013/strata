import time

from strata.state.regime_tracker import RegimeLabel, infer_regime
from strata.state.strata_state import StrataState


def make_state(ts, spot, res, nres, center, width, risk):
    return StrataState(
        timestamp=ts,
        symbol="GLD",
        spot=spot,
        surface_timestamp=None,
        atm_iv=None,
        residual=res,
        normalized_residual=nres,
        basin_center=center,
        basin_width=width,
        basin_velocity=0.0,
        position_state="centered",
        normalized_distance=0.0,
        risk_score=risk,
    )


def test_stable_regime():
    base_ts = time.time()
    states = [
        make_state(base_ts + i, 100.0, 0.0, 0.001 * i, 0.0, 1.0, 0.1)
        for i in range(3)
    ]
    regime = infer_regime(states, "GLD")
    assert regime is not None
    assert regime.label == RegimeLabel.STABLE


def test_drifting_regime():
    base_ts = time.time()
    states = [
        make_state(base_ts + i, 100.0 + i, 0.0, 0.05 * i, 0.0, 1.0, 0.1)
        for i in range(5)
    ]
    regime = infer_regime(states, "GLD")
    assert regime is not None
    assert regime.label == RegimeLabel.DRIFTING


def test_compressing_regime():
    base_ts = time.time()
    states = [
        make_state(base_ts + i, 100.0, 0.0, 0.02, 0.0, 1.0 - 0.1 * i, 0.1 + 0.05 * i)
        for i in range(3)
    ]
    regime = infer_regime(states, "GLD")
    assert regime is not None
    assert regime.label == RegimeLabel.COMPRESSING


def test_transition_regime_conflict():
    base_ts = time.time()
    # Residual drifting up, center drifting down => conflict
    states = [
        make_state(base_ts + i, 100.0, 0.0, 0.02 * i, 0.1 - 0.05 * i, 1.0, 0.1)
        for i in range(4)
    ]
    regime = infer_regime(states, "GLD")
    assert regime is not None
    assert regime.label == RegimeLabel.TRANSITION
