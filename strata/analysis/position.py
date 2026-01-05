"""Basin position analysis - price position relative to attractor basin."""
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Tuple
import logging

import numpy as np
import pandas as pd

from strata.db.queries import (
    get_current_basin,
    get_recent_residuals,
    get_basin_history,
    write_basin_position
)

logger = logging.getLogger(__name__)


def compute_basin_position(
    asset: str,
    timescale: str,
    timestamp: Optional[datetime] = None
) -> Optional[str]:
    """
    Compute price position relative to basin.

    This function analyzes where the current price sits within the basin:
    - Distance to center
    - Distance to boundaries
    - How well price is tracking basin movement
    - Position state classification

    Args:
        asset: Asset symbol
        timescale: Time resolution
        timestamp: Analysis timestamp (defaults to now)

    Returns:
        Position state string if successful, None otherwise
    """
    if timestamp is None:
        timestamp = datetime.now()

    # Get current basin geometry
    basin = get_current_basin(asset, timescale)

    if not basin:
        logger.warning(f"No basin found for {asset} {timescale}")
        return None

    basin_id = basin['basin_id']

    # Get current residual state
    residuals = get_recent_residuals(asset, timescale, n=1)

    if not residuals:
        logger.warning(f"No residual data for {asset} {timescale}")
        return None

    current_residual = float(residuals[0]['normalized_residual'])

    # Extract basin parameters
    basin_center = float(basin['center_location'])
    basin_width = float(basin['basin_width'])
    boundary_upper = float(basin['boundary_upper'])
    boundary_lower = float(basin['boundary_lower'])
    basin_velocity = float(basin['center_velocity'])

    # Calculate distance to center
    distance_to_center = abs(current_residual - basin_center)

    # Calculate distance to nearest boundary
    distance_to_upper = abs(current_residual - boundary_upper)
    distance_to_lower = abs(current_residual - boundary_lower)
    distance_to_boundary = min(distance_to_upper, distance_to_lower)

    # Calculate normalized distance (0 = center, 1 = boundary)
    # Use half-width as normalizer
    half_width = basin_width / 2.0
    if half_width > 0:
        normalized_distance = min(distance_to_center / half_width, 1.5)  # Cap at 1.5 (beyond boundary)
    else:
        normalized_distance = 0.0

    # Calculate price velocity in residual space
    price_velocity = compute_price_velocity(asset, timescale, lookback=5)

    # Calculate lag score (how well price tracks basin movement)
    lag_score = compute_lag_score(basin_velocity, price_velocity)

    # Classify position state
    position_state = calculate_position_state(
        normalized_distance, distance_to_boundary, lag_score
    )

    # Prepare data for database
    position_data = {
        'basin_id': basin_id,
        'asset': asset,
        'timestamp': timestamp,
        'timescale': timescale,
        'distance_to_center': Decimal(str(round(distance_to_center, 4))),
        'distance_to_boundary': Decimal(str(round(distance_to_boundary, 4))),
        'normalized_distance': Decimal(str(round(min(normalized_distance, 1.0), 4))),
        'basin_velocity': Decimal(str(round(basin_velocity, 6))),
        'price_velocity': Decimal(str(round(price_velocity, 6))),
        'lag_score': Decimal(str(round(lag_score, 3))),
        'position_state': position_state
    }

    # Write to database
    write_basin_position(position_data)

    logger.info(
        f"Basin position for {asset} {timescale}: state={position_state}, "
        f"normalized_distance={normalized_distance:.3f}, lag_score={lag_score:.3f}"
    )

    logger.debug(
        f"  residual={current_residual:.4f}, center={basin_center:.4f}, "
        f"distance_to_center={distance_to_center:.4f}, "
        f"distance_to_boundary={distance_to_boundary:.4f}"
    )

    return position_state


def calculate_position_state(
    normalized_distance: float,
    distance_to_boundary: float,
    lag_score: float
) -> str:
    """
    Classify position state based on metrics.

    Position states (in order of severity):
    - 'detaching': Price breaking away from basin (critical)
    - 'lagging': Price not tracking basin movement
    - 'edge': Price near basin boundary
    - 'tracking': Price following basin center
    - 'centered': Price near basin center (ideal)

    Args:
        normalized_distance: Distance from center normalized by half-width (0-1+)
        distance_to_boundary: Absolute distance to nearest boundary
        lag_score: How well price tracks basin (0-1, higher is better)

    Returns:
        Position state string
    """
    # Critical: Price beyond or very near boundary AND not tracking
    if normalized_distance >= 0.9 and lag_score < 0.3:
        return 'detaching'

    # Warning: Price not tracking basin movement
    if lag_score < 0.5:
        return 'lagging'

    # Caution: Price near edge of basin
    if normalized_distance >= 0.7:
        return 'edge'

    # Normal: Price tracking basin
    if normalized_distance >= 0.3:
        return 'tracking'

    # Ideal: Price centered in basin
    return 'centered'


def compute_lag_score(basin_velocity: float, price_velocity: float) -> float:
    """
    Calculate how well price is tracking basin movement.

    Lag score interpretation:
    - 1.0: Perfect tracking (velocities aligned)
    - 0.5: Orthogonal movement
    - 0.0: Opposite directions or completely detached

    Args:
        basin_velocity: Rate of basin center movement
        price_velocity: Rate of price movement in residual space

    Returns:
        Lag score (0 to 1)
    """
    # If both velocities are very small, consider it perfect tracking (stationary)
    if abs(basin_velocity) < 1e-4 and abs(price_velocity) < 1e-4:
        return 1.0

    # If only one is moving, tracking is poor
    if abs(basin_velocity) < 1e-4 or abs(price_velocity) < 1e-4:
        return 0.3

    # Calculate cosine similarity (both are scalar velocities)
    # In 1D, this is just: sign(v1) * sign(v2) gives ±1
    # Magnitude ratio also matters
    direction_match = np.sign(basin_velocity) * np.sign(price_velocity)

    # If opposite directions, very low score
    if direction_match < 0:
        return 0.1

    # If same direction, score based on magnitude similarity
    # Use ratio of smaller to larger
    ratio = min(abs(basin_velocity), abs(price_velocity)) / max(abs(basin_velocity), abs(price_velocity))

    # Convert ratio to score (1.0 = perfect match, 0.5 = one twice as fast as other)
    lag_score = 0.5 + 0.5 * ratio  # Range: 0.5 to 1.0 for same direction

    return float(np.clip(lag_score, 0.0, 1.0))


def compute_price_velocity(
    asset: str,
    timescale: str,
    lookback: int = 5
) -> float:
    """
    Calculate rate of price movement in residual space.

    Args:
        asset: Asset symbol
        timescale: Time resolution
        lookback: Number of recent residuals to analyze

    Returns:
        Velocity (change in residual per hour)
    """
    residuals = get_recent_residuals(asset, timescale, n=lookback + 1)

    if len(residuals) < 2:
        logger.debug(f"Insufficient residuals for velocity: {len(residuals)}")
        return 0.0

    df = pd.DataFrame(residuals)
    df = df.sort_values('timestamp')

    # Extract residuals and timestamps
    residual_values = df['normalized_residual'].astype(float).values
    timestamps = pd.to_datetime(df['timestamp'])

    # Calculate time differences in hours
    time_diffs = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600.0
    time_elapsed = time_diffs.iloc[-1] - time_diffs.iloc[0]

    if time_elapsed < 0.01:
        logger.debug("Time elapsed too small for velocity calculation")
        return 0.0

    # Calculate velocity (simple delta)
    residual_change = residual_values[-1] - residual_values[0]
    velocity = residual_change / time_elapsed

    logger.debug(
        f"Price velocity: residual_change={residual_change:.6f}, "
        f"time_elapsed={time_elapsed:.2f}h, velocity={velocity:.6f}"
    )

    return float(velocity)


def analyze_position_risk(asset: str, timescale: str) -> Dict:
    """
    Analyze risk based on basin position.

    Risk factors:
    - Distance to boundary (closer = higher risk)
    - Velocity toward boundary
    - Lag score (low = higher risk)

    Args:
        asset: Asset symbol
        timescale: Time resolution

    Returns:
        Dict with risk metrics
    """
    from strata.db.queries import get_basin_position_current

    position = get_basin_position_current(asset, timescale)

    if not position:
        logger.warning(f"No position data for {asset} {timescale}")
        return {}

    # Extract metrics
    normalized_distance = float(position['normalized_distance'])
    distance_to_boundary = float(position['distance_to_boundary'])
    lag_score = float(position['lag_score'])
    position_state = position['position_state']

    # Calculate risk scores (0 = low risk, 1 = high risk)

    # Boundary risk: closer to edge = higher risk
    boundary_risk = normalized_distance

    # Tracking risk: low lag score = higher risk
    tracking_risk = 1.0 - lag_score

    # Velocity risk: high velocity toward boundary
    basin_velocity = float(position['basin_velocity'])
    price_velocity = float(position['price_velocity'])

    # If velocities point in same direction and away from center, risk increases
    velocity_away = (basin_velocity + price_velocity) / 2.0
    velocity_risk = np.clip(abs(velocity_away) * 10.0, 0.0, 1.0)  # Scale to 0-1

    # Overall risk (weighted combination)
    overall_risk = (
        0.4 * boundary_risk +
        0.3 * tracking_risk +
        0.3 * velocity_risk
    )

    # Map position state to risk level
    state_risk_map = {
        'centered': 0.1,
        'tracking': 0.3,
        'edge': 0.6,
        'lagging': 0.7,
        'detaching': 0.95
    }
    state_risk = state_risk_map.get(position_state, 0.5)

    # Take maximum of calculated risk and state risk
    overall_risk = max(overall_risk, state_risk)

    risk = {
        'overall_risk': float(np.clip(overall_risk, 0.0, 1.0)),
        'boundary_risk': float(boundary_risk),
        'tracking_risk': float(tracking_risk),
        'velocity_risk': float(velocity_risk),
        'state_risk': float(state_risk),
        'position_state': position_state,
        'recommendation': get_risk_recommendation(overall_risk, position_state)
    }

    logger.info(
        f"Position risk for {asset} {timescale}: overall={overall_risk:.3f}, "
        f"state={position_state}, recommendation={risk['recommendation']}"
    )

    return risk


def get_risk_recommendation(risk_score: float, position_state: str) -> str:
    """
    Get trading recommendation based on risk.

    Args:
        risk_score: Overall risk score (0-1)
        position_state: Position state string

    Returns:
        Recommendation string
    """
    if position_state == 'detaching':
        return 'EXIT_IMMEDIATELY'

    if risk_score > 0.8:
        return 'EXIT_POSITION'

    if risk_score > 0.6:
        return 'REDUCE_EXPOSURE'

    if risk_score > 0.4:
        return 'MONITOR_CLOSELY'

    if risk_score > 0.2:
        return 'NORMAL_OPERATION'

    return 'LOW_RISK'


def get_position_summary(asset: str, timescale: str) -> Dict:
    """
    Get comprehensive position summary.

    Args:
        asset: Asset symbol
        timescale: Time resolution

    Returns:
        Dict with position metrics and analysis
    """
    from strata.db.queries import get_basin_position_current

    position = get_basin_position_current(asset, timescale)

    if not position:
        logger.warning(f"No position data for {asset} {timescale}")
        return {}

    # Get risk analysis
    risk = analyze_position_risk(asset, timescale)

    summary = {
        'asset': asset,
        'timescale': timescale,
        'timestamp': position['timestamp'],
        'position_state': position['position_state'],
        'normalized_distance': float(position['normalized_distance']),
        'distance_to_boundary': float(position['distance_to_boundary']),
        'lag_score': float(position['lag_score']),
        'basin_velocity': float(position['basin_velocity']),
        'price_velocity': float(position['price_velocity']),
        'risk': risk
    }

    return summary
