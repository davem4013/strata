from strata.state.strata_buffer import STRATA_BUFFER, append_interpreted_state
from strata.state.strata_state import StrataState


def _state(ts: float) -> StrataState:
    return StrataState(
        timestamp=ts,
        symbol="TEST",
        spot=100.0,
        surface_timestamp=None,
        atm_iv=None,
        residual=0.5,
        normalized_residual=1.2,
        basin_center=0.1,
        basin_width=2.0,
        basin_velocity=0.05,
        position_state="centered",
        normalized_distance=0.2,
        risk_score=0.3,
    )


def test_append_interpreted_state_populates_buffer():
    STRATA_BUFFER.history()  # ensure buffer exists
    # Clear shared buffer by recreating for test isolation
    from strata.state import strata_buffer as sb

    sb.STRATA_BUFFER = sb.StrataBuffer()

    state = _state(1234.5)
    append_interpreted_state(state, buffer=sb.STRATA_BUFFER)

    latest = sb.STRATA_BUFFER.latest()
    assert latest is not None
    assert latest.timestamp == state.timestamp
    assert latest.basin_center == [state.basin_center]
    assert latest.basin_radius == state.basin_width / 2.0


def test_append_interpreted_state_respects_provided_buffer():
    from strata.state.strata_buffer import StrataBuffer

    custom_buffer = StrataBuffer(maxlen=2)
    state = _state(2000.0)
    append_interpreted_state(state, buffer=custom_buffer)

    latest = custom_buffer.latest()
    assert latest is not None
    assert latest.timestamp == 2000.0
