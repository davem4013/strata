from strata.state.basin_frame import BasinFrame
from strata.state.basin_state import ResidualBasin
from strata.state.strata_buffer import StrataBuffer


def _frame(ts: float, center: float) -> BasinFrame:
    basin = ResidualBasin(
        center=center,
        width=1.0,
        boundary_upper=center + 0.5,
        boundary_lower=center - 0.5,
        curvature=0.5,
        sample_count=10,
        variance=0.0,
    ).to_dict()

    return BasinFrame(
        timestamp=str(ts),
        symbol="TEST",
        domain_dim=1,
        residual_energy=abs(center),
        residual_max=abs(center),
        centroid=[center],
        fit_quality=1.0,
        basins=[basin],
        residual_coordinate=center,
        response=None,
    )


def test_append_and_latest():
    buf = StrataBuffer(maxlen=3)
    assert buf.latest() is None

    f1 = _frame(1.0, 0.0)
    buf.append(f1)
    assert buf.latest() == f1

    f2 = _frame(2.0, 1.0)
    buf.append(f2)
    assert buf.latest() == f2


def test_history_order_and_length():
    buf = StrataBuffer(maxlen=5)
    frames = [_frame(i, i * 0.1) for i in range(3)]
    for f in frames:
        buf.append(f)

    hist = buf.history(2)
    assert len(hist) == 2
    assert hist[0] == frames[1]
    assert hist[1] == frames[2]

    full = buf.history()
    assert full == frames


def test_maxlen_eviction():
    buf = StrataBuffer(maxlen=2)
    buf.append(_frame(1.0, 0.0))
    buf.append(_frame(2.0, 0.1))
    buf.append(_frame(3.0, 0.2))  # evicts first

    hist = buf.history()
    assert len(hist) == 2
    assert hist[0].timestamp == "2.0"
    assert hist[1].timestamp == "3.0"
