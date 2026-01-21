from datetime import datetime

import numpy as np

from strata.analysis.residuals import ResidualGeometry
from strata.state.state_api import (
    get_basin_geometry_history,
    get_basin_history,
    get_latest_basin_frame,
    get_latest_basin_geometry,
)
from strata.state.strata_buffer import StrataBuffer, STRATA_BUFFER, update_from_residual_geometry
from strata.state.strata_state import StrataState


def test_latest_basin_frame_reflects_last_append(monkeypatch):
    # isolate shared buffer
    from strata.state import strata_buffer as sb

    new_buf = StrataBuffer(maxlen=5)
    monkeypatch.setattr(sb, "STRATA_BUFFER", new_buf, raising=True)

    geom1 = ResidualGeometry(
        timestamp=datetime.utcfromtimestamp(1.0),
        domain_dim=1,
        residual_energy=0.1,
        residual_max=0.1,
        centroid=np.array([0.1]),
        fit_quality=0.9,
        sample_count=3,
    )
    geom2 = ResidualGeometry(
        timestamp=datetime.utcfromtimestamp(2.0),
        domain_dim=1,
        residual_energy=0.2,
        residual_max=0.2,
        centroid=np.array([0.2]),
        fit_quality=0.9,
        sample_count=3,
    )

    update_from_residual_geometry("TST", geom1, buffer=new_buf)
    update_from_residual_geometry("TST", geom2, buffer=new_buf)

    latest = get_latest_basin_frame()
    assert latest is not None
    assert latest.timestamp.startswith("1970-01-01T00:00:02")
    assert latest.centroid == [0.2]


def test_basin_history_ordering(monkeypatch):
    from strata.state import strata_buffer as sb

    new_buf = StrataBuffer(maxlen=5)
    monkeypatch.setattr(sb, "STRATA_BUFFER", new_buf, raising=True)

    for ts, center in [(1.0, 0.1), (2.0, 0.2), (3.0, 0.3)]:
        geom = ResidualGeometry(
            timestamp=datetime.utcfromtimestamp(ts),
            domain_dim=1,
            residual_energy=center,
            residual_max=center,
            centroid=np.array([center]),
            fit_quality=0.95,
            sample_count=3,
        )
        update_from_residual_geometry("TST", geom, buffer=new_buf)

    history = get_basin_history(2)
    assert len(history) == 2
    assert history[0].timestamp.startswith("1970-01-01T00:00:02")
    assert history[1].timestamp.startswith("1970-01-01T00:00:03")


def test_geometry_buffer_receives_parallel_updates(monkeypatch):
    from strata.state import strata_buffer as sb

    legacy_buf = StrataBuffer(maxlen=5)
    geometry_buf = StrataBuffer(maxlen=5)

    monkeypatch.setattr(sb, "STRATA_BUFFER", legacy_buf, raising=True)
    monkeypatch.setattr(sb, "GEOMETRY_BUFFER", geometry_buf, raising=True)

    state = StrataState(
        timestamp=5.0,
        symbol="GEO",
        spot=10.0,
        surface_timestamp=None,
        atm_iv=None,
        residual=0.2,
        normalized_residual=0.25,
        basin_center=0.1,
        basin_width=0.6,
        basin_velocity=0.05,
        position_state="centered",
        normalized_distance=0.1,
        risk_score=0.2,
    )

    sb.append_interpreted_state(state, buffer=legacy_buf)

    latest_geo = get_latest_basin_geometry()
    assert latest_geo is not None
    assert latest_geo.symbol == "GEO"
    assert len(latest_geo.proj_2d) == 2
    assert latest_geo.radius == 2.0
    assert latest_geo.trail_2d  # at least one point

    history = get_basin_geometry_history(2)
    assert history
