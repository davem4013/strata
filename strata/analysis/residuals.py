"""Residual calculation using least squares regression."""
from datetime import datetime
from decimal import Decimal
from typing import Tuple, Optional
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from strata.config import RESIDUAL_LOOKBACK_PERIODS
from strata.db.queries import get_recent_market_data, write_residual_state

logger = logging.getLogger(__name__)


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

    # Fit least squares model
    try:
        fair_value, residual, normalized_residual, std_error, r_squared = (
            fit_least_squares(prices, times)
        )
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
    Fit linear regression model and calculate residuals.

    The model fits a straight line through the price series:
        fair_value(t) = slope * t + intercept

    Residuals are deviations from this line:
        residual(t) = actual_price(t) - fair_value(t)

    Args:
        prices: Array of prices
        times: Array of time indices

    Returns:
        Tuple of (fair_value, residual, normalized_residual, std_error, r_squared)
        - fair_value: Predicted price from linear fit at current time
        - residual: Actual price - fair_value
        - normalized_residual: residual / std_error (dimensionless)
        - std_error: Standard deviation of all residuals
        - r_squared: Goodness of fit (0 to 1)

    Raises:
        ValueError: If inputs are invalid
    """
    if len(prices) < 2:
        raise ValueError("Need at least 2 data points for regression")

    if len(prices) != len(times):
        raise ValueError("prices and times must have same length")

    # Fit model
    model = LinearRegression()
    model.fit(times, prices)

    # Predict fair value at current time (last point)
    fair_value = model.predict(times[-1:].reshape(-1, 1))[0]

    # Calculate residual at current time
    residual = prices[-1] - fair_value

    # Calculate standard error from all residuals
    predictions = model.predict(times)
    residuals_all = prices - predictions
    std_error = np.std(residuals_all, ddof=1)  # Use sample std deviation

    # Avoid division by zero
    if std_error < 1e-10:
        logger.warning("Standard error near zero, setting normalized_residual to 0")
        normalized_residual = 0.0
    else:
        # Normalized residual (dimensionless z-score)
        normalized_residual = residual / std_error

    # R-squared (coefficient of determination)
    r_squared = model.score(times, prices)

    logger.debug(
        f"Least squares fit: slope={model.coef_[0]:.4f}, "
        f"intercept={model.intercept_:.2f}, r_squared={r_squared:.4f}"
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

        try:
            fair_value, residual, normalized_residual, std_error, r_squared = (
                fit_least_squares(prices, times)
            )

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
        return {}

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
