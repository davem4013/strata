"""State buffer + committer integration for ResidualBasin storage/serialization."""
from strata.state import state_committer as state_committer_module
from strata.state.basin_state import ResidualBasin
from strata.state.state_buffer import BasinBuffer, StrataStateBuffer
from strata.state.state_committer import StrataStateCommitter


class _StubState:
    def __init__(self, normalized_residual: float, timestamp: float = 0.0):
        self.normalized_residual = normalized_residual
        self.timestamp = timestamp


def test_basin_buffer_stores_objects_not_dicts():
    buffer = BasinBuffer()
    basin = ResidualBasin(
        center=0.1,
        width=0.4,
        boundary_upper=0.3,
        boundary_lower=-0.1,
        curvature=0.6,
        sample_count=5,
        variance=0.02,
    )

    buffer.push(basin)

    latest = buffer.latest()
    assert isinstance(latest, ResidualBasin)
    assert latest.center == basin.center


def test_basin_dict_round_trip_fidelity():
    basin = ResidualBasin(
        center=0.25,
        width=1.0,
        boundary_upper=0.75,
        boundary_lower=-0.25,
        curvature=0.4,
        sample_count=12,
        variance=0.16,
    )

    data = basin.to_dict()
    assert data["center"] == 0.25
    assert data["width"] == 1.0
    assert data["boundary_upper"] == 0.75
    assert data["boundary_lower"] == -0.25
    assert data["curvature"] == 0.4
    assert data["sample_count"] == 12

    rebuilt = ResidualBasin(
        center=data["center"],
        width=data["width"],
        boundary_upper=data["boundary_upper"],
        boundary_lower=data["boundary_lower"],
        curvature=data["curvature"],
        sample_count=data["sample_count"],
        variance=basin.variance,
    )
    assert rebuilt == basin


def test_committer_compute_basin_uses_objects_and_preserves_outputs(monkeypatch):
    """
    _compute_basin should return a ResidualBasin, push it to the basin buffer,
    and expose the same center/width values legacy callers expect.
    """
    state_buffer = StrataStateBuffer()
    state_buffer.push(_StubState(normalized_residual=0.1, timestamp=0.0))

    # Provide deterministic clustering output
    target_basin = ResidualBasin(
        center=0.2,
        width=0.5,
        boundary_upper=0.45,
        boundary_lower=-0.05,
        curvature=0.7,
        sample_count=2,
        variance=0.01,
    )

    def _fake_cluster_residuals(residuals):
        return target_basin

    monkeypatch.setattr(state_committer_module, "cluster_residuals", _fake_cluster_residuals)

    committer = StrataStateCommitter(
        analytics_url="http://example.com",
        symbol="TEST",
        buffer=state_buffer,
        poll_seconds=0.1,
        surface_poll_seconds=0.1,
    )

    basin, basin_is_2d = committer._compute_basin(0.3, response=None)

    assert isinstance(basin, ResidualBasin)
    assert basin.center == target_basin.center
    assert basin.width == target_basin.width
    assert basin_is_2d is False

    latest = committer.basin_buffer.latest()
    assert latest is basin

    # Serialization boundary: dict matches prior numeric shape
    serialized = basin.to_dict()
    assert serialized["center"] == target_basin.center
    assert serialized["width"] == target_basin.width
    assert serialized["boundary_upper"] == target_basin.boundary_upper
    assert serialized["boundary_lower"] == target_basin.boundary_lower
