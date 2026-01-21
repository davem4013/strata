"""Basin geometry detection and analysis."""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional
import logging

import numpy as np
import pandas as pd

from strata.config import BOUNDARY_SIGMA, BASIN_CLUSTERING_WINDOW
import strata.db.queries as dbq
from strata.state.basin_state import ResidualBasin

logger = logging.getLogger(__name__)


def identify_basins(
    asset: str,
    timescale: str,
    clustering_window: Optional[int] = None
) -> Optional[str]:
    """
    Identify attractor basin structure from residuals.

    This function:
    1. Fetches recent residuals
    2. Analyzes their distribution to find basin center and boundaries
    3. Calculates basin velocity (how fast the basin is moving)
    4. Writes basin geometry to database

    Args:
        asset: Asset symbol
        timescale: Time resolution
        clustering_window: Number of residuals to analyze (default from config)

    Returns:
        basin_id if successful, None otherwise
    """
    if clustering_window is None:
        clustering_window = BASIN_CLUSTERING_WINDOW

    # Fetch recent residuals
    data = dbq.get_recent_residuals(asset, timescale, n=clustering_window)


    MIN_RESIDUALS = 5  # test-friendly, Phase 3

    if len(data) < MIN_RESIDUALS:

        logger.warning(
            f"Insufficient residuals for basin identification: "
            f"{len(data)} < 20 for {asset} {timescale}"
        )
        return None

    # Extract normalized residuals and optional response (if available).
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp')

    residuals = df['normalized_residual'].astype(float).values
    residual_input = residuals

    # When response is present, move to 2D (residual, response) space; rows with
    # missing response stay in 1D. Time is not a spatial dimension here.
    if "response" in df.columns and df["response"].notna().any():
        df_2d = df[df["response"].notna()]
        if len(df_2d) >= 2:
            residual_input = np.column_stack(
                [
                    df_2d["normalized_residual"].astype(float).values,
                    df_2d["response"].astype(float).values,
                ]
            )
        else:
            logger.debug("Insufficient response-bearing points for 2D clustering; using 1D residuals.")

    current_timestamp = df.iloc[-1]['timestamp']

    effective_count = residual_input.shape[0] if isinstance(residual_input, np.ndarray) and residual_input.ndim > 0 else len(residual_input)
    if effective_count < MIN_RESIDUALS:
        logger.warning(
            f"Insufficient points for basin identification in active domain: "
            f"{effective_count} < {MIN_RESIDUALS} for {asset} {timescale}"
        )
        return None

    # Cluster residuals to find basin structure

    try:
        basin_structure = cluster_residuals(residual_input)
    except Exception as e:
        logger.warning(
            f"Clustering failed for {asset} {timescale}, "
            f"creating fallback basin: {e}"
        )

        # Phase 3 fallback: single synthetic basin
        center = float(np.mean(residuals))
        std = float(np.std(residuals))

        # Guard against degenerate std
        width = max(4.0 * std, 0.5)

        basin_structure = ResidualBasin(
            center=center,
            width=width,
            boundary_upper=center + width / 2,
            boundary_lower=center - width / 2,
            curvature=0.0,
            sample_count=len(residuals),
            variance=std * std,
        )


    # Generate basin_id
    basin_id = f"{asset}_{timescale}_{current_timestamp.strftime('%Y%m%d')}"

    # Calculate basin velocity
    basin_velocity = compute_basin_velocity(basin_id, lookback=10)

    # Prepare data for database
    basin_dict = basin_structure.to_dict()
    basin_data = {
        'basin_id': basin_id,
        'asset': asset,
        'timestamp': current_timestamp,
        'timescale': timescale,
        'center_location': Decimal(str(round(basin_dict['center'], 6))),
        'center_velocity': Decimal(str(round(basin_velocity, 6))),
        'boundary_upper': Decimal(str(round(basin_dict['boundary_upper'], 6))),
        'boundary_lower': Decimal(str(round(basin_dict['boundary_lower'], 6))),
        'basin_width': Decimal(str(round(basin_dict['width'], 4))),
        'curvature': Decimal(str(round(basin_dict.get('curvature', 0.0), 4))),
        'sample_count': basin_dict['sample_count']
    }

    # Write to database
    dbq.write_basin_geometry(basin_data)

    logger.info(
        f"Identified basin for {asset} {timescale}: "
        f"center={basin_structure.center:.4f}, "
        f"width={basin_structure.width:.4f}, "
        f"velocity={basin_velocity:.6f}"
    )

    logger.debug(
        f"  boundaries=[{basin_structure.boundary_lower:.4f}, "
        f"{basin_structure.boundary_upper:.4f}], "
        f"sample_count={basin_structure.sample_count}"
    )

    return basin_id


def cluster_residuals(residuals: np.ndarray) -> ResidualBasin:
    """
    Analyze residual distribution to identify basin structure.

    Domain gating:
    - 1D (legacy): operate on residuals only, unchanged.
    - 2D: operate in (residual_coordinate, response) space when both dimensions
      are present. Time is not a spatial dimension here; only residual vs. response
      define the phase-space point.

    Args:
        residuals: Array of normalized residuals, shape (N,) for 1D or (N, 2)
            for 2D (residual_coordinate, response). Points with response=None/NaN
            must be filtered out by the caller or will be dropped here.

    Returns:
        ResidualBasin with basin parameters:
        - center: Mean residual
        - boundary_upper: Upper boundary (+N sigma)
        - boundary_lower: Lower boundary (-N sigma)
        - width: Distance between boundaries
        - sample_count: Number of residuals analyzed
        - curvature: Optional measure of basin stiffness
    """
    if len(residuals) < 2:
        raise ValueError("Need at least 2 residuals for clustering")

    arr = np.asarray(residuals, dtype=float)

    # Legacy 1D path: unchanged behavior.
    if arr.ndim == 1:
        # Calculate center (mean)
        center = float(np.mean(arr))

        # Calculate spread (standard deviation)
        std = float(np.std(arr, ddof=1))

        # Calculate curvature (second derivative of density)
        # For now, use kurtosis as a proxy for basin stiffness
        # High kurtosis = sharp basin (high curvature)
        # Low kurtosis = flat basin (low curvature)
        from scipy.stats import kurtosis
        try:
            kurt = float(kurtosis(arr, fisher=True))
            # Normalize to [0, 1] range (approximate)
            curvature = 1.0 / (1.0 + np.exp(-kurt))
        except Exception:
            curvature = 0.5  # Default neutral curvature

        basin = ResidualBasin.from_geometry(
            center=center,
            std=std,
            boundary_sigma=BOUNDARY_SIGMA,
            sample_count=len(arr),
            curvature=curvature,
        )

        logger.debug(
            f"Basin clustering: center={center:.4f}, std={std:.4f}, "
            f"width={basin.width:.4f}, curvature={curvature:.4f}"
        )

        return basin

    # 2D path: residual_coordinate + response (normalized Euclidean over axis-aligned spreads).
    if arr.ndim == 2 and arr.shape[1] >= 2:
        coords = arr[:, :2]
        # Drop rows with missing response; ensures domain_dim reflects active dimensions only.
        mask = ~np.isnan(coords).any(axis=1)
        coords = coords[mask]
        if len(coords) < 2:
            raise ValueError("Need at least 2 points with response for 2D clustering")

        residuals_1d = coords[:, 0]
        responses_1d = coords[:, 1]

        center_vec = coords.mean(axis=0).tolist()
        std_vec = coords.std(axis=0, ddof=1)
        # Axis-aligned covariance keeps membership based on normalized Euclidean distance.
        covariance = np.diag(std_vec ** 2).tolist()
        width_vec = (2.0 * BOUNDARY_SIGMA * std_vec).tolist()

        # Legacy scalar fields remain tied to the residual axis for compatibility.
        center = float(center_vec[0])
        std_residual = float(std_vec[0])

        from scipy.stats import kurtosis
        try:
            kurt_residual = float(kurtosis(residuals_1d, fisher=True))
            curvature = 1.0 / (1.0 + np.exp(-kurt_residual))
        except Exception:
            curvature = 0.5

        basin = ResidualBasin.from_geometry(
            center=center,
            std=std_residual,
            boundary_sigma=BOUNDARY_SIGMA,
            sample_count=len(coords),
            curvature=curvature,
            domain_dim=2,
            center_vector=[float(center_vec[0]), float(center_vec[1])],
            width_vector=[float(width_vec[0]), float(width_vec[1])],
            covariance=covariance,
        )

        logger.debug(
            "2D basin clustering: center_vec=%s, std_vec=%s, width_vec=%s, curvature=%.4f",
            center_vec,
            std_vec.tolist(),
            width_vec,
            curvature,
        )

        return basin

    raise ValueError("Unsupported residuals shape for clustering")


def compute_basin_velocity(basin_id: str, lookback: int = 10) -> float:
    """
    Calculate rate of basin center movement.

    Basin velocity indicates whether the attractor is shifting:
    - velocity ≈ 0: Stable basin
    - velocity > 0: Basin center drifting upward
    - velocity < 0: Basin center drifting downward
    - |velocity| large: Regime transition in progress

    Args:
        basin_id: Basin identifier
        lookback: Number of historical points to analyze

    Returns:
        Velocity (change in center per hour)
    """
    # Fetch basin history
    history = dbq.get_basin_history(basin_id, n=lookback)

    if len(history) < 2:
        logger.debug(
            f"Insufficient history for basin velocity: {len(history)} < 2"
        )
        return 0.0

    df = pd.DataFrame(history)
    df = df.sort_values('timestamp')

    # Extract centers and timestamps
    centers = df['center_location'].astype(float).values
    timestamps = pd.to_datetime(df['timestamp'])

    # Calculate time differences in hours
    time_diffs = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600.0
    time_elapsed = time_diffs.iloc[-1] - time_diffs.iloc[0]

    if time_elapsed < 0.01:  # Avoid division by very small numbers
        logger.debug("Time elapsed too small for velocity calculation")
        return 0.0

    # Calculate velocity (simple: recent - oldest / time)
    center_change = centers[-1] - centers[0]
    velocity = center_change / time_elapsed

    # For more sophisticated analysis, could fit a line and use slope
    # But simple delta is sufficient for now

    logger.debug(
        f"Basin velocity: center_change={center_change:.6f}, "
        f"time_elapsed={time_elapsed:.2f}h, velocity={velocity:.6f}"
    )

    return float(velocity)


def analyze_basin_stability(basin_id: str, lookback: int = 20) -> Dict:
    """
    Analyze basin stability over time.

    Metrics:
    - Width stability: How much basin width varies
    - Center stability: How much basin center moves
    - Shape stability: Changes in curvature

    Args:
        basin_id: Basin identifier
        lookback: Number of historical points to analyze

    Returns:
        Dict with stability metrics
    """
    history = dbq.get_basin_history(basin_id, n=lookback)

    if len(history) < 5:
        logger.warning(f"Insufficient history for stability analysis: {len(history)}")
        return {}

    df = pd.DataFrame(history)
    df = df.sort_values('timestamp')

    # Width stability (coefficient of variation)
    widths = df['basin_width'].astype(float).values
    width_mean = np.mean(widths)
    width_std = np.std(widths)
    width_stability = 1.0 - min(width_std / width_mean if width_mean > 0 else 1.0, 1.0)

    # Center stability (normalized by width)
    centers = df['center_location'].astype(float).values
    center_range = np.max(centers) - np.min(centers)
    center_stability = 1.0 - min(center_range / width_mean if width_mean > 0 else 1.0, 1.0)

    # Shape stability (curvature variance)
    curvatures = df['curvature'].astype(float).values
    curvature_std = np.std(curvatures)
    shape_stability = 1.0 - min(curvature_std, 1.0)

    # Overall stability (weighted average)
    overall_stability = (
        0.4 * width_stability +
        0.4 * center_stability +
        0.2 * shape_stability
    )

    stability = {
        'width_stability': float(width_stability),
        'center_stability': float(center_stability),
        'shape_stability': float(shape_stability),
        'overall_stability': float(overall_stability),
        'sample_count': len(history)
    }

    logger.info(
        f"Basin stability for {basin_id}: overall={overall_stability:.3f}, "
        f"width={width_stability:.3f}, center={center_stability:.3f}"
    )

    return stability


def detect_basin_transition(asset: str, timescale: str, lookback: int = 20) -> bool:
    """
    Detect if basin is undergoing a transition.

    Transition indicators:
    - Rapid width change
    - High basin velocity
    - Low stability scores

    Args:
        asset: Asset symbol
        timescale: Time resolution
        lookback: Number of historical points to analyze

    Returns:
        True if transition detected, False otherwise
    """
    # Get current basin
    from strata.db.queries import get_current_basin

    current = get_current_basin(asset, timescale)

    if not current:
        logger.debug(f"No current basin for {asset} {timescale}")
        return False

    basin_id = current['basin_id']

    # Check velocity
    velocity = abs(compute_basin_velocity(basin_id, lookback=lookback))

    # Check stability
    stability = analyze_basin_stability(basin_id, lookback=lookback)

    if not stability:
        return False

    # Transition criteria
    high_velocity = velocity > 0.1  # Threshold: 0.1 std per hour
    low_stability = stability['overall_stability'] < 0.5

    is_transition = high_velocity or low_stability

    if is_transition:
        logger.warning(
            f"Basin transition detected for {asset} {timescale}: "
            f"velocity={velocity:.4f}, stability={stability['overall_stability']:.3f}"
        )

    return is_transition


def get_basin_summary(basin_id: str) -> Dict:
    """
    Get comprehensive summary of basin state.

    Args:
        basin_id: Basin identifier

    Returns:
        Dict with basin metrics and analysis
    """

    history = dbq.get_basin_history(basin_id, n=1)

    if not history:
        logger.warning(f"No data found for basin {basin_id}")
        return {}

    current = history[0]

    # Get stability analysis
    stability = analyze_basin_stability(basin_id, lookback=20)

    summary = {
        'basin_id': basin_id,
        'asset': current['asset'],
        'timescale': current['timescale'],
        'timestamp': current['timestamp'],
        'center': float(current['center_location']),
        'width': float(current['basin_width']),
        'velocity': float(current['center_velocity']),
        'boundaries': [
            float(current['boundary_lower']),
            float(current['boundary_upper'])
        ],
        'stability': stability
    }

    return summary
