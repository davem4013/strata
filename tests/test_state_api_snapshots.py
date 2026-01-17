from strata.state.state_api import get_basin_history, get_latest_basin_frame
from strata.state.strata_buffer import StrataBuffer, STRATA_BUFFER, append_interpreted_state
from strata.state.strata_state import StrataState


def _state(ts: float, center: float) -> StrataState:
    return StrataState(
        timestamp=ts,
        symbol="TST",
        spot=100.0,
        surface_timestamp=None,
        atm_iv=None,
        residual=0.1,
        normalized_residual=center,  # reuse center for uniqueness
        basin_center=center,
        basin_width=2.0,
        basin_velocity=0.05,
        position_state="centered",
        normalized_distance=0.2,
        risk_score=0.3,
    )


def test_latest_basin_frame_reflects_last_append(monkeypatch):
    # isolate shared buffer
    from strata.state import strata_buffer as sb

    new_buf = StrataBuffer(maxlen=5)
    monkeypatch.setattr(sb, "STRATA_BUFFER", new_buf, raising=True)

    append_interpreted_state(_state(1.0, 0.1), buffer=new_buf)
    append_interpreted_state(_state(2.0, 0.2), buffer=new_buf)

    latest = get_latest_basin_frame()
    assert latest is not None
    assert latest.timestamp == 2.0
    assert latest.basin_center == [0.2]


def test_basin_history_ordering(monkeypatch):
    from strata.state import strata_buffer as sb

    new_buf = StrataBuffer(maxlen=5)
    monkeypatch.setattr(sb, "STRATA_BUFFER", new_buf, raising=True)

    append_interpreted_state(_state(1.0, 0.1), buffer=new_buf)
    append_interpreted_state(_state(2.0, 0.2), buffer=new_buf)
    append_interpreted_state(_state(3.0, 0.3), buffer=new_buf)

    history = get_basin_history(2)
    assert len(history) == 2
    assert history[0].timestamp == 2.0
    assert history[1].timestamp == 3.0
