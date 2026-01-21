import numpy as np
import pytest

from strata.analysis.residuals import ResidualFitter


def test_residual_geometry_1d_affine():
    """1D affine baseline should yield ~zero residual geometry."""
    # Simple line: v = 2t + 1 over four samples.
    coords = np.arange(4, dtype=float).reshape(-1, 1)
    values = 2.0 * coords[:, 0] + 1.0
    feature_map = lambda c: np.column_stack([np.ones(len(c)), c[:, 0]])

    fitter = ResidualFitter(feature_map)
    field, geom = fitter.fit(coords=coords, values=values)

    assert np.allclose(field.residuals, 0.0, atol=1e-10)
    assert geom.residual_energy == pytest.approx(0.0, abs=1e-10)
    # When residual weights are zero, centroid falls back to mean of coords.
    assert geom.centroid == pytest.approx(np.mean(coords, axis=0))
    assert geom.fit_quality == pytest.approx(1.0, rel=1e-6)
    assert geom.domain_dim == 1
    assert geom.sample_count == len(coords)


def test_residual_geometry_2d_quadratic_surface():
    """2D quadratic bowl recovered exactly with matching feature map."""
    rng = np.random.default_rng(42)
    # Scattered samples; no grid structure assumed.
    coords = rng.uniform(low=-1.0, high=1.0, size=(50, 2))
    x, y = coords[:, 0], coords[:, 1]

    def quad_surface(c):
        m, t = c[:, 0], c[:, 1]
        return np.column_stack([np.ones_like(m), m, t, m**2, t**2, m * t])

    # Ground-truth coefficients for [1, x, y, x^2, y^2, x*y]
    coefs = np.array([0.5, 0.8, -0.3, 0.2, 0.1, -0.05])
    values = quad_surface(coords) @ coefs

    fitter = ResidualFitter(quad_surface)
    field, geom = fitter.fit(coords=coords, values=values)

    assert np.allclose(field.residuals, 0.0, atol=1e-10)
    assert geom.residual_energy == pytest.approx(0.0, abs=1e-10)
    assert geom.residual_max == pytest.approx(0.0, abs=1e-10)
    # Residual weights are zero, so centroid is the mean of scattered samples.
    assert geom.centroid == pytest.approx(np.mean(coords, axis=0))
    assert geom.fit_quality == pytest.approx(1.0, rel=1e-6)
    assert geom.domain_dim == 2
    assert geom.sample_count == len(coords)


def test_residual_geometry_scatter_order_invariance():
    """Geometry should not depend on sample ordering."""
    rng = np.random.default_rng(7)
    coords = rng.normal(size=(30, 2))

    def quad_surface(c):
        m, t = c[:, 0], c[:, 1]
        return np.column_stack([np.ones_like(m), m, t, m**2, t**2, m * t])

    coefs = np.array([1.0, -0.2, 0.4, 0.05, -0.03, 0.02])
    values = quad_surface(coords) @ coefs

    fitter = ResidualFitter(quad_surface)
    field_a, geom_a = fitter.fit(coords=coords, values=values)

    perm = rng.permutation(len(coords))
    coords_b = coords[perm]
    values_b = values[perm]
    field_b, geom_b = fitter.fit(coords=coords_b, values=values_b)

    assert np.allclose(field_a.residuals, field_b.residuals, atol=1e-10)
    assert geom_a.residual_energy == pytest.approx(geom_b.residual_energy, abs=1e-10)
    assert geom_a.residual_max == pytest.approx(geom_b.residual_max, abs=1e-10)
    assert geom_a.centroid == pytest.approx(geom_b.centroid, abs=1e-12)
    assert geom_a.fit_quality == pytest.approx(geom_b.fit_quality, rel=1e-9)
