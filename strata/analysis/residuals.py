"""Residual geometry engine.

This module now treats residuals as geometric objects rather than decisions.
The core `ResidualFitter` fits a linear model (via arbitrary feature maps) over
an N-dimensional domain (1D or 2D today), returns the raw residual field, and
summarizes that field into lightweight geometry for STRATA v0.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Callable, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from strata.config import RESIDUAL_LOOKBACK_PERIODS
from strata.db.queries import get_recent_market_data, write_residual_state

logger = logging.getLogger(__name__)


@dataclass
class ResidualField:
    """Scattered residual samples over an arbitrary domain."""

    coords: np.ndarray
    values: np.ndarray
    predictions: np.ndarray
    residuals: np.ndarray
    coefficients: np.ndarray


@dataclass
class ResidualGeometry:
    """Lightweight geometry summary consumed by STRATA v0."""

    timestamp: datetime
    domain_dim: int
    residual_energy: float
    residual_max: float
    centroid: np.ndarray
    fit_quality: float
    sample_count: int

    def as_dict(self) -> dict:
        """Convert to serializable dict for downstream components."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "domain_dim": self.domain_dim,
            "residual_energy": self.residual_energy,
            "residual_max": self.residual_max,
            "centroid": self.centroid.tolist(),
            "fit_quality": self.fit_quality,
            "sample_count": self.sample_count,
        }


class ResidualFitter:
    """
    Dimension-agnostic residual geometry engine.

    - Inputs: `coords` with shape (N, D) and `values` with shape (N,)
    - Feature map: callable that builds the regression design matrix from coords
    - Outputs: a scattered `ResidualField` and a `ResidualGeometry` summary

    Geometric interpretation:
        ResidualField  → "where and how large are the deviations" (raw samples)
        ResidualGeometry → "how much mass, where is its center, how well is the
        baseline explaining the field" (no decisions, just shape descriptors)

    Minimal usage:
        >>> feature_map = lambda c: np.stack([np.ones(len(c)), c[:, 0], c[:, 0] ** 2], axis=1)
        >>> fitter = ResidualFitter(feature_map)
        >>> field, geom = fitter.fit(coords=np.arange(10).reshape(-1, 1), values=prices)

    2D IV surface:
        >>> def quad_surface(coords):
        ...     m, t = coords[:, 0], coords[:, 1]
        ...     return np.column_stack([np.ones_like(m), m, t, m**2, t**2, m * t])
        >>> fitter = ResidualFitter(quad_surface)
        >>> field, geom = fitter.fit(coords=iv_coords, values=iv_values)
    """

    def __init__(
        self,
        feature_map: Callable[[np.ndarray], np.ndarray],
        timestamp: Optional[datetime] = None,
    ) -> None:
        self.feature_map = feature_map
        self.timestamp = timestamp

    def fit(self, coords: np.ndarray, values: np.ndarray) -> Tuple[ResidualField, ResidualGeometry]:
        coords = np.asarray(coords, dtype=float)
        values = np.asarray(values, dtype=float)

        if coords.ndim != 2:
            raise ValueError("coords must have shape (N, D)")
        if values.ndim != 1:
            raise ValueError("values must be a 1D array")
        if coords.shape[0] != values.shape[0]:
            raise ValueError("coords and values must have the same number of samples")

        features = np.asarray(self.feature_map(coords), dtype=float)
        if features.ndim != 2 or features.shape[0] != coords.shape[0]:
            raise ValueError("feature_map must return array with shape (N, K)")

        # Linear regression over feature space; feature_map controls basis.
        model = LinearRegression(fit_intercept=False)
        model.fit(features, values)
        predictions = model.predict(features)
        residuals = values - predictions

        residual_energy = float(np.linalg.norm(residuals))
        residual_max = float(np.max(np.abs(residuals))) if residuals.size else 0.0

        weights = np.abs(residuals)
        weight_sum = float(np.sum(weights))
        # If residual energy is effectively zero, fall back to the mean location.
        if weight_sum <= 1e-12:
            centroid = np.mean(coords, axis=0)
        else:
            centroid = np.sum(coords * weights[:, None], axis=0) / weight_sum

        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((values - np.mean(values)) ** 2))
        fit_quality = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        geometry = ResidualGeometry(
            timestamp=self.timestamp or datetime.utcnow(),
            domain_dim=coords.shape[1],
            residual_energy=residual_energy,
            residual_max=residual_max,
            centroid=centroid,
            fit_quality=fit_quality,
            sample_count=len(values),
        )

        field = ResidualField(
            coords=coords,
            values=values,
            predictions=predictions,
            residuals=residuals,
            coefficients=model.coef_,
        )

        return field, geometry


def compute_residuals(
    asset: str,
    timescale: str,
    lookback_periods: Optional[int] = None
) -> None:
    """
    Calculate least squares residuals and write to residual_state table.

    This function:
    1. Fetches recent market data
    2. Fits a linear regression model
    3. Calculates residuals (deviations from the fitted line)
    4. Normalizes residuals by standard error
    5. Writes results to database

    Args:
        asset: Asset symbol
        timescale: Time resolution
        lookback_periods: Rolling window size for regression (default from config)
    """
    if lookback_periods is None:
        lookback_periods = RESIDUAL_LOOKBACK_PERIODS

    # Fetch recent market data
    data = get_recent_market_data(asset, timescale, n=lookback_periods + 1)

    if len(data) < lookback_periods:
        logger.warning(
            f"Insufficient data for {asset} {timescale}: "
            f"{len(data)} < {lookback_periods}"
        )
        return

    # Convert to DataFrame and sort chronologically
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp')

    # Convert price to float for regression
    prices = df['price'].astype(float).values
    times = np.arange(len(prices)).reshape(-1, 1)

    geometry = None
    # Fit least squares model
    try:
        fair_value, residual, normalized_residual, std_error, r_squared = (
            fit_least_squares(prices, times)
        )
        # Geometry frame (1D time domain) for STRATA consumption.
        try:
            timestamp = pd.to_datetime(df.iloc[-1]["timestamp"]).to_pydatetime()
            feature_map = lambda c: np.column_stack([np.ones(len(c)), c[:, 0]])
            geometry = ResidualFitter(feature_map, timestamp=timestamp).fit(times, prices)[1]
        except Exception as geo_exc:  # pragma: no cover - geometry emission is best-effort
            logger.debug("Residual geometry emission skipped: %s", geo_exc)
            geometry = None
    except Exception as e:
        logger.error(f"Error fitting least squares for {asset} {timescale}: {e}")
        return

    # Write to database (most recent point)
    current_data = {
        'asset': asset,
        'timestamp': df.iloc[-1]['timestamp'],
        'timescale': timescale,
        'price': Decimal(str(prices[-1])),
        'fair_value': Decimal(str(round(fair_value, 4))),
        'residual': Decimal(str(round(residual, 6))),
        'normalized_residual': Decimal(str(round(normalized_residual, 4))),
        'std_error': Decimal(str(round(std_error, 6))) if std_error is not None else None,
        'r_squared': Decimal(str(round(r_squared, 4))) if r_squared is not None else None
    }

    write_residual_state(current_data)

    # Emit geometry frame to STRATA buffer
    if geometry is not None:
        try:
            from strata.state.strata_buffer import update_from_residual_geometry

            update_from_residual_geometry(symbol=asset, geometry=geometry)
        except Exception as exc:  # pragma: no cover - do not block residual persistence
            logger.debug("Unable to update STRATA buffer with geometry: %s", exc)

    logger.info(
        f"Computed residual for {asset} {timescale}: "
        f"normalized_residual={normalized_residual:.4f}, "
        f"r_squared={r_squared:.4f}"
    )

    logger.debug(
        f"  price={prices[-1]:.2f}, fair_value={fair_value:.2f}, "
        f"residual={residual:.4f}, std_error={std_error:.6f}"
    )


def fit_least_squares(
    prices: np.ndarray,
    times: np.ndarray
) -> Tuple[float, float, float, float, float]:
    """
    Fit linear regression model and calculate residual geometry for 1D time.

    The feature map is a simple affine basis [1, t] so the baseline matches the
    legacy time-trend fit while still flowing through the geometry engine.

    Returns a tuple kept for backward compatibility with callers that expect:
        (fair_value, residual, normalized_residual, std_error, r_squared)
    """
    if len(prices) < 2:
        raise ValueError("Need at least 2 data points for regression")

    if len(prices) != len(times):
        raise ValueError("prices and times must have same length")

    coords = np.asarray(times, dtype=float)
    coords = coords.reshape(len(prices), -1)
    prices_arr = np.asarray(prices, dtype=float)

    feature_map = lambda c: np.column_stack([np.ones(len(c)), c[:, 0]])
    fitter = ResidualFitter(feature_map)
    field, geometry = fitter.fit(coords=coords, values=prices_arr)

    fair_value = float(field.predictions[-1])
    residual = float(field.residuals[-1])
    std_error = float(np.std(field.residuals, ddof=1)) if len(field.residuals) > 1 else 0.0
    normalized_residual = residual / std_error if std_error > 1e-10 else 0.0
    r_squared = float(geometry.fit_quality)

    logger.debug(
        "Least squares fit (geom): slope=%.4f intercept=%.2f r_squared=%.4f",
        field.coefficients[1] if len(field.coefficients) > 1 else 0.0,
        field.coefficients[0] if len(field.coefficients) > 0 else 0.0,
        r_squared,
    )

    return fair_value, residual, normalized_residual, std_error, r_squared


def compute_residuals_bulk(asset: str, timescale: str, n_periods: int) -> None:
    """
    Compute residuals for multiple historical periods.

    Useful for backfilling residual_state table.

    Args:
        asset: Asset symbol
        timescale: Time resolution
        n_periods: Number of periods to process
    """
    logger.info(
        f"Computing bulk residuals for {asset} {timescale}: {n_periods} periods"
    )

    # Fetch all data
    all_data = get_recent_market_data(asset, timescale, n=n_periods + RESIDUAL_LOOKBACK_PERIODS)

    if len(all_data) < RESIDUAL_LOOKBACK_PERIODS + 1:
        logger.warning(
            f"Insufficient data for bulk residual computation: "
            f"{len(all_data)} < {RESIDUAL_LOOKBACK_PERIODS + 1}"
        )
        return

    df = pd.DataFrame(all_data)
    df = df.sort_values('timestamp')

    # Compute residuals for each window
    computed_count = 0
    for i in range(RESIDUAL_LOOKBACK_PERIODS, len(df)):
        # Get window
        window_df = df.iloc[i - RESIDUAL_LOOKBACK_PERIODS:i + 1]

        prices = window_df['price'].astype(float).values
        times = np.arange(len(prices)).reshape(-1, 1)
        geometry = None

        try:
            fair_value, residual, normalized_residual, std_error, r_squared = (
                fit_least_squares(prices, times)
            )
            try:
                timestamp = pd.to_datetime(window_df.iloc[-1]["timestamp"]).to_pydatetime()
                feature_map = lambda c: np.column_stack([np.ones(len(c)), c[:, 0]])
                geometry = ResidualFitter(feature_map, timestamp=timestamp).fit(times, prices)[1]
            except Exception as geo_exc:  # pragma: no cover - best-effort geometry
                logger.debug("Residual geometry emission skipped in bulk: %s", geo_exc)

            # Write to database
            current_data = {
                'asset': asset,
                'timestamp': window_df.iloc[-1]['timestamp'],
                'timescale': timescale,
                'price': Decimal(str(prices[-1])),
                'fair_value': Decimal(str(round(fair_value, 4))),
                'residual': Decimal(str(round(residual, 6))),
                'normalized_residual': Decimal(str(round(normalized_residual, 4))),
                'std_error': Decimal(str(round(std_error, 6))),
                'r_squared': Decimal(str(round(r_squared, 4)))
            }

            write_residual_state(current_data)
            if geometry is not None:
                try:
                    from strata.state.strata_buffer import update_from_residual_geometry

                    update_from_residual_geometry(symbol=asset, geometry=geometry)
                except Exception as exc:  # pragma: no cover
                    logger.debug("Unable to update STRATA buffer with geometry (bulk): %s", exc)
            computed_count += 1

        except Exception as e:
            logger.error(f"Error computing residual at index {i}: {e}")
            continue

    logger.info(f"Computed {computed_count} residuals for {asset} {timescale}")


def analyze_residual_statistics(asset: str, timescale: str) -> dict:
    """
    Analyze statistical properties of residuals.

    Useful for diagnostics and validation.

    Args:
        asset: Asset symbol
        timescale: Time resolution

    Returns:
        Dict with statistics: mean, std, min, max, autocorrelation, etc.
    """
    from strata.db.queries import get_recent_residuals

    data = get_recent_residuals(asset, timescale, n=100)



    if len(data) < 10:
        logger.warning(f"Insufficient residual data for statistics: {len(data)}")
        return None



    df = pd.DataFrame(data)
    df = df.sort_values('timestamp')

    residuals = df['normalized_residual'].astype(float).values

    # Calculate statistics
    stats = {
        'count': len(residuals),
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'min': float(np.min(residuals)),
        'max': float(np.max(residuals)),
        'median': float(np.median(residuals)),
        'q25': float(np.percentile(residuals, 25)),
        'q75': float(np.percentile(residuals, 75))
    }

    # Autocorrelation (lag 1)
    if len(residuals) > 1:
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        stats['autocorr_lag1'] = float(autocorr)

    logger.info(
        f"Residual statistics for {asset} {timescale}: "
        f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
        f"range=[{stats['min']:.4f}, {stats['max']:.4f}]"
    )

    return stats
