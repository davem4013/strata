"""Unit tests for ResidualBasin geometry container."""
from strata.state.basin_state import BasinEvolution, ResidualBasin, compare_basins


def test_residual_basin_to_dict_and_radius():
    basin = ResidualBasin(
        center=0.5,
        width=2.0,
        boundary_upper=1.5,
        boundary_lower=-0.5,
        curvature=0.25,
        sample_count=42,
        variance=0.36,
    )

    data = basin.to_dict()

    assert data["center"] == 0.5
    assert data["boundary_upper"] == 1.5
    assert data["boundary_lower"] == -0.5
    assert data["width"] == 2.0
    assert data["curvature"] == 0.25
    assert data["sample_count"] == 42
    assert basin.radius == 1.0


def test_residual_basin_from_geometry_matches_width_and_boundaries():
    basin = ResidualBasin.from_geometry(
        center=1.0,
        std=0.5,
        boundary_sigma=2.0,
        sample_count=10,
        curvature=0.75,
    )

    assert basin.center == 1.0
    assert basin.boundary_upper == 1.0 + (2.0 * 0.5)
    assert basin.boundary_lower == 1.0 - (2.0 * 0.5)
    assert basin.width == basin.boundary_upper - basin.boundary_lower
    assert basin.variance == 0.25
    assert basin.sample_count == 10
    assert basin.curvature == 0.75


def test_distance_monotonicity_and_normalization():
    basin = ResidualBasin(
        center=0.0,
        width=2.0,
        boundary_upper=1.0,
        boundary_lower=-1.0,
        curvature=0.5,
        sample_count=10,
        variance=0.25,
    )

    inner = basin.normalized_distance(0.2)
    outer = basin.normalized_distance(0.8)

    assert outer > inner
    assert basin.distance(0.0) == 0.0
    assert basin.contains(0.5)
    assert basin.contains(-1.0)
    assert not basin.contains(1.5)
    assert basin.boundary_margin(0.0) == 1.0
    assert basin.boundary_margin(0.5) == 0.5
    assert basin.boundary_margin(1.5) == 0.5


def test_stability_score_prefers_tight_supported_curved_basins():
    loose = ResidualBasin(
        center=0.0,
        width=4.0,
        boundary_upper=2.0,
        boundary_lower=-2.0,
        curvature=0.2,
        sample_count=2,
        variance=1.0,
    )
    tight = ResidualBasin(
        center=0.0,
        width=1.0,
        boundary_upper=0.5,
        boundary_lower=-0.5,
        curvature=0.8,
        sample_count=30,
        variance=0.04,
    )

    assert 0.0 <= loose.stability_score() <= 1.0
    assert 0.0 <= tight.stability_score() <= 1.0
    assert tight.stability_score() > loose.stability_score()


def test_compare_basins_reports_diagnostics():
    before = ResidualBasin(
        center=0.0,
        width=2.0,
        boundary_upper=1.0,
        boundary_lower=-1.0,
        curvature=0.3,
        sample_count=10,
        variance=0.25,
    )
    after = ResidualBasin(
        center=0.5,
        width=1.0,
        boundary_upper=1.0,
        boundary_lower=0.0,
        curvature=0.6,
        sample_count=25,
        variance=0.04,
    )

    diag = compare_basins(before, after)

    assert isinstance(diag, BasinEvolution)
    assert diag.center_delta == 0.5
    assert diag.radius_delta == -0.5
    assert diag.width_ratio == 0.5
    assert diag.compression > 0.0
    assert diag.expansion == 0.0
    assert diag.stability_delta > 0.0
