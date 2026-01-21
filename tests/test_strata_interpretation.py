from datetime import datetime

import numpy as np

from strata.analysis.residuals import ResidualGeometry
from strata.state.strata_buffer import STRATA_BUFFER, update_from_residual_geometry


def test_append_interpreted_state_populates_buffer():
    STRATA_BUFFER.history()  # ensure buffer exists
    # Clear shared buffer by recreating for test isolation
    from strata.state import strata_buffer as sb

    sb.STRATA_BUFFER = sb.StrataBuffer()

    geom = ResidualGeometry(
        timestamp=datetime.utcfromtimestamp(1234.5),
        domain_dim=2,
        residual_energy=0.0,
        residual_max=0.0,
        centroid=np.array([0.1, -0.2]),
        fit_quality=1.0,
        sample_count=10,
    )
    update_from_residual_geometry("TEST", geom, buffer=sb.STRATA_BUFFER)

    latest = sb.STRATA_BUFFER.latest()
    assert latest is not None
    assert latest.timestamp.startswith("1970-01-01T00:20:34")
    assert latest.symbol == "TEST"
    assert latest.domain_dim == 2
    assert latest.centroid == [0.1, -0.2]
    assert latest.basins == []


def test_append_interpreted_state_respects_provided_buffer():
    from strata.state.strata_buffer import StrataBuffer

    custom_buffer = StrataBuffer(maxlen=2)
    geom = ResidualGeometry(
        timestamp=datetime.utcfromtimestamp(2000.0),
        domain_dim=1,
        residual_energy=0.5,
        residual_max=0.5,
        centroid=np.array([0.5]),
        fit_quality=0.9,
        sample_count=5,
    )
    update_from_residual_geometry("TEST", geom, buffer=custom_buffer)

    latest = custom_buffer.latest()
    assert latest is not None
    assert latest.symbol == "TEST"
    assert latest.domain_dim == 1
    assert latest.centroid == [0.5]
