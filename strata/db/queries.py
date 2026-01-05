"""Database query functions for STRATA."""
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional
import logging

from strata.db.connection import get_cursor

logger = logging.getLogger(__name__)

# ============================================================================
# WRITE OPERATIONS
# ============================================================================

def write_market_state(data: Dict) -> None:
    """
    Insert market data.

    Args:
        data: {
            'asset': str,
            'timestamp': datetime,
            'timescale': str,
            'price': Decimal,
            'implied_vol': Optional[Decimal],
            'skew': Optional[Decimal],
            'volume': Optional[int],
            'bid_ask_spread': Optional[Decimal]
        }
    """
    query = """
        INSERT INTO market_state
        (asset, timestamp, timescale, price, implied_vol, skew, volume, bid_ask_spread)
        VALUES (%(asset)s, %(timestamp)s, %(timescale)s, %(price)s,
                %(implied_vol)s, %(skew)s, %(volume)s, %(bid_ask_spread)s)
        ON CONFLICT (asset, timestamp, timescale) DO UPDATE SET
            price = EXCLUDED.price,
            implied_vol = EXCLUDED.implied_vol,
            skew = EXCLUDED.skew,
            volume = EXCLUDED.volume,
            bid_ask_spread = EXCLUDED.bid_ask_spread;
    """
    with get_cursor() as cursor:
        cursor.execute(query, data)
    logger.debug(f"Wrote market_state: {data['asset']} {data['timescale']} {data['timestamp']}")


def write_residual_state(data: Dict) -> None:
    """Insert residual calculations."""
    query = """
        INSERT INTO residual_state
        (asset, timestamp, timescale, price, fair_value, residual,
         normalized_residual, std_error, r_squared)
        VALUES (%(asset)s, %(timestamp)s, %(timescale)s, %(price)s,
                %(fair_value)s, %(residual)s, %(normalized_residual)s,
                %(std_error)s, %(r_squared)s)
        ON CONFLICT (asset, timestamp, timescale) DO UPDATE SET
            price = EXCLUDED.price,
            fair_value = EXCLUDED.fair_value,
            residual = EXCLUDED.residual,
            normalized_residual = EXCLUDED.normalized_residual,
            std_error = EXCLUDED.std_error,
            r_squared = EXCLUDED.r_squared;
    """
    with get_cursor() as cursor:
        cursor.execute(query, data)
    logger.debug(f"Wrote residual_state: {data['asset']} {data['timescale']}")


def write_basin_geometry(data: Dict) -> None:
    """Insert basin structure."""
    query = """
        INSERT INTO basin_geometry
        (basin_id, asset, timestamp, timescale, center_location, center_velocity,
         boundary_upper, boundary_lower, basin_width, curvature, sample_count)
        VALUES (%(basin_id)s, %(asset)s, %(timestamp)s, %(timescale)s,
                %(center_location)s, %(center_velocity)s, %(boundary_upper)s,
                %(boundary_lower)s, %(basin_width)s, %(curvature)s, %(sample_count)s)
        ON CONFLICT (basin_id, timestamp) DO UPDATE SET
            center_location = EXCLUDED.center_location,
            center_velocity = EXCLUDED.center_velocity,
            boundary_upper = EXCLUDED.boundary_upper,
            boundary_lower = EXCLUDED.boundary_lower,
            basin_width = EXCLUDED.basin_width,
            curvature = EXCLUDED.curvature,
            sample_count = EXCLUDED.sample_count;
    """
    with get_cursor() as cursor:
        cursor.execute(query, data)
    logger.debug(f"Wrote basin_geometry: {data['basin_id']}")


def write_basin_position(data: Dict) -> None:
    """Insert basin position analysis."""
    query = """
        INSERT INTO basin_position
        (basin_id, asset, timestamp, timescale, distance_to_center, distance_to_boundary,
         normalized_distance, basin_velocity, price_velocity, lag_score, position_state)
        VALUES (%(basin_id)s, %(asset)s, %(timestamp)s, %(timescale)s,
                %(distance_to_center)s, %(distance_to_boundary)s, %(normalized_distance)s,
                %(basin_velocity)s, %(price_velocity)s, %(lag_score)s, %(position_state)s)
        ON CONFLICT (basin_id, timestamp) DO UPDATE SET
            distance_to_center = EXCLUDED.distance_to_center,
            distance_to_boundary = EXCLUDED.distance_to_boundary,
            normalized_distance = EXCLUDED.normalized_distance,
            basin_velocity = EXCLUDED.basin_velocity,
            price_velocity = EXCLUDED.price_velocity,
            lag_score = EXCLUDED.lag_score,
            position_state = EXCLUDED.position_state;
    """
    with get_cursor() as cursor:
        cursor.execute(query, data)
    logger.debug(f"Wrote basin_position: {data['basin_id']}")


def write_model_interpretation(data: Dict) -> None:
    """Insert model analysis."""
    query = """
        INSERT INTO model_interpretation
        (model_id, basin_id, timestamp, timescale, regime_type, center_estimate,
         boundary_upper_estimate, boundary_lower_estimate, stability_score,
         confidence, interpretation_text, embedding)
        VALUES (%(model_id)s, %(basin_id)s, %(timestamp)s, %(timescale)s,
                %(regime_type)s, %(center_estimate)s, %(boundary_upper_estimate)s,
                %(boundary_lower_estimate)s, %(stability_score)s, %(confidence)s,
                %(interpretation_text)s, %(embedding)s)
        ON CONFLICT (model_id, basin_id, timestamp) DO UPDATE SET
            regime_type = EXCLUDED.regime_type,
            center_estimate = EXCLUDED.center_estimate,
            boundary_upper_estimate = EXCLUDED.boundary_upper_estimate,
            boundary_lower_estimate = EXCLUDED.boundary_lower_estimate,
            stability_score = EXCLUDED.stability_score,
            confidence = EXCLUDED.confidence,
            interpretation_text = EXCLUDED.interpretation_text,
            embedding = EXCLUDED.embedding;
    """
    with get_cursor() as cursor:
        cursor.execute(query, data)
    logger.debug(f"Wrote model_interpretation: {data['model_id']} for {data['basin_id']}")


def write_agreement_metrics(data: Dict) -> None:
    """Insert agreement scores."""
    query = """
        INSERT INTO agreement_metrics
        (basin_id, timestamp, timescale, agreement_score, disagreement_type,
         semantic_distance, variance_center, variance_boundary,
         directional_divergence, model_count)
        VALUES (%(basin_id)s, %(timestamp)s, %(timescale)s, %(agreement_score)s,
                %(disagreement_type)s, %(semantic_distance)s, %(variance_center)s,
                %(variance_boundary)s, %(directional_divergence)s, %(model_count)s)
        ON CONFLICT (basin_id, timestamp) DO UPDATE SET
            agreement_score = EXCLUDED.agreement_score,
            disagreement_type = EXCLUDED.disagreement_type,
            semantic_distance = EXCLUDED.semantic_distance,
            variance_center = EXCLUDED.variance_center,
            variance_boundary = EXCLUDED.variance_boundary,
            directional_divergence = EXCLUDED.directional_divergence,
            model_count = EXCLUDED.model_count;
    """
    with get_cursor() as cursor:
        cursor.execute(query, data)
    logger.debug(f"Wrote agreement_metrics: {data['basin_id']} score={data['agreement_score']}")


def write_cross_scale_coherence(data: Dict) -> None:
    """Insert cascade detection."""
    query = """
        INSERT INTO cross_scale_coherence
        (basin_id, timestamp, parent_timescale, child_timescale, coherence_score,
         cascade_flag, persistence_count, direction)
        VALUES (%(basin_id)s, %(timestamp)s, %(parent_timescale)s, %(child_timescale)s,
                %(coherence_score)s, %(cascade_flag)s, %(persistence_count)s, %(direction)s)
        ON CONFLICT (basin_id, timestamp, parent_timescale, child_timescale) DO UPDATE SET
            coherence_score = EXCLUDED.coherence_score,
            cascade_flag = EXCLUDED.cascade_flag,
            persistence_count = EXCLUDED.persistence_count,
            direction = EXCLUDED.direction;
    """
    with get_cursor() as cursor:
        cursor.execute(query, data)
    logger.debug(f"Wrote cross_scale_coherence: {data['basin_id']}")


def write_regime_event(data: Dict) -> None:
    """Insert regime event."""
    query = """
        INSERT INTO regime_event
        (asset, timescale, timestamp, event_type, severity, source, metadata)
        VALUES (%(asset)s, %(timescale)s, %(timestamp)s, %(event_type)s,
                %(severity)s, %(source)s, %(metadata)s);
    """
    with get_cursor() as cursor:
        cursor.execute(query, data)
    logger.info(f"Regime event: {data['event_type']} for {data['asset']} severity={data['severity']}")


def write_system_state(key: str, value: dict) -> None:
    """
    Insert or update system state configuration.

    Args:
        key: Configuration key
        value: JSON-serializable dict
    """
    query = """
        INSERT INTO system_state (key, value, updated_at)
        VALUES (%(key)s, %(value)s, NOW())
        ON CONFLICT (key) DO UPDATE SET
            value = EXCLUDED.value,
            updated_at = NOW();
    """
    with get_cursor() as cursor:
        cursor.execute(query, {'key': key, 'value': value})
    logger.debug(f"Wrote system_state: {key}")


# ============================================================================
# READ OPERATIONS
# ============================================================================

def get_recent_market_data(asset: str, timescale: str, n: int = 100) -> List[Dict]:
    """Retrieve recent market data."""
    query = """
        SELECT * FROM market_state
        WHERE asset = %s AND timescale = %s
        ORDER BY timestamp DESC
        LIMIT %s;
    """
    with get_cursor() as cursor:
        cursor.execute(query, (asset, timescale, n))
        return cursor.fetchall()


def get_recent_residuals(asset: str, timescale: str, n: int = 100) -> List[Dict]:
    """Retrieve recent residuals for basin identification."""
    query = """
        SELECT * FROM residual_state
        WHERE asset = %s AND timescale = %s
        ORDER BY timestamp DESC
        LIMIT %s;
    """
    with get_cursor() as cursor:
        cursor.execute(query, (asset, timescale, n))
        return cursor.fetchall()


def get_current_basin(asset: str, timescale: str) -> Optional[Dict]:
    """Get most recent basin geometry."""
    query = """
        SELECT * FROM basin_geometry
        WHERE asset = %s AND timescale = %s
        ORDER BY timestamp DESC
        LIMIT 1;
    """
    with get_cursor() as cursor:
        cursor.execute(query, (asset, timescale))
        return cursor.fetchone()


def get_basin_history(basin_id: str, n: int = 10) -> List[Dict]:
    """Get basin evolution over time."""
    query = """
        SELECT * FROM basin_geometry
        WHERE basin_id = %s
        ORDER BY timestamp DESC
        LIMIT %s;
    """
    with get_cursor() as cursor:
        cursor.execute(query, (basin_id, n))
        return cursor.fetchall()


def get_basin_position_current(asset: str, timescale: str) -> Optional[Dict]:
    """Get most recent basin position."""
    query = """
        SELECT * FROM basin_position
        WHERE asset = %s AND timescale = %s
        ORDER BY timestamp DESC
        LIMIT 1;
    """
    with get_cursor() as cursor:
        cursor.execute(query, (asset, timescale))
        return cursor.fetchone()


def get_model_interpretations(basin_id: str, timestamp: datetime) -> List[Dict]:
    """Get all model interpretations for a basin at timestamp."""
    query = """
        SELECT * FROM model_interpretation
        WHERE basin_id = %s AND timestamp = %s;
    """
    with get_cursor() as cursor:
        cursor.execute(query, (basin_id, timestamp))
        return cursor.fetchall()


def get_agreement_metrics_current(asset: str, timescale: str) -> Optional[Dict]:
    """Get most recent agreement metrics."""
    query = """
        SELECT am.* FROM agreement_metrics am
        JOIN basin_geometry bg ON am.basin_id = bg.basin_id AND am.timestamp = bg.timestamp
        WHERE bg.asset = %s AND bg.timescale = %s
        ORDER BY am.timestamp DESC
        LIMIT 1;
    """
    with get_cursor() as cursor:
        cursor.execute(query, (asset, timescale))
        return cursor.fetchone()


def get_active_cascades(lookback_hours: int = 24) -> List[Dict]:
    """Retrieve recent cascade detections."""
    query = """
        SELECT * FROM cross_scale_coherence
        WHERE cascade_flag = TRUE
          AND timestamp > NOW() - INTERVAL '%s hours'
        ORDER BY timestamp DESC, persistence_count DESC;
    """
    with get_cursor() as cursor:
        cursor.execute(query, (lookback_hours,))
        return cursor.fetchall()


def get_active_alerts(lookback_hours: int = 24) -> List[Dict]:
    """Retrieve recent high-severity events."""
    query = """
        SELECT * FROM regime_event
        WHERE timestamp > NOW() - INTERVAL '%s hours'
          AND severity > 0.7
        ORDER BY severity DESC, timestamp DESC;
    """
    with get_cursor() as cursor:
        cursor.execute(query, (lookback_hours,))
        return cursor.fetchall()


def get_system_state(key: str) -> Optional[dict]:
    """
    Get system configuration value.

    Args:
        key: Configuration key

    Returns:
        Configuration value as dict, or None if not found
    """
    query = """
        SELECT value FROM system_state
        WHERE key = %s;
    """
    with get_cursor() as cursor:
        cursor.execute(query, (key,))
        result = cursor.fetchone()
        return result['value'] if result else None


def get_all_system_state() -> Dict[str, dict]:
    """
    Get all system configuration.

    Returns:
        Dict mapping keys to values
    """
    query = """
        SELECT key, value FROM system_state;
    """
    with get_cursor() as cursor:
        cursor.execute(query)
        results = cursor.fetchall()
        return {row['key']: row['value'] for row in results}


# ============================================================================
# BULK OPERATIONS
# ============================================================================

def bulk_write_market_state(data_list: List[Dict]) -> None:
    """
    Bulk insert market data for efficiency.

    Args:
        data_list: List of market_state dicts
    """
    if not data_list:
        return

    query = """
        INSERT INTO market_state
        (asset, timestamp, timescale, price, implied_vol, skew, volume, bid_ask_spread)
        VALUES (%(asset)s, %(timestamp)s, %(timescale)s, %(price)s,
                %(implied_vol)s, %(skew)s, %(volume)s, %(bid_ask_spread)s)
        ON CONFLICT (asset, timestamp, timescale) DO UPDATE SET
            price = EXCLUDED.price,
            implied_vol = EXCLUDED.implied_vol,
            skew = EXCLUDED.skew,
            volume = EXCLUDED.volume,
            bid_ask_spread = EXCLUDED.bid_ask_spread;
    """
    with get_cursor() as cursor:
        cursor.executemany(query, data_list)
    logger.info(f"Bulk wrote {len(data_list)} market_state records")


def bulk_write_residual_state(data_list: List[Dict]) -> None:
    """
    Bulk insert residual data.

    Args:
        data_list: List of residual_state dicts
    """
    if not data_list:
        return

    query = """
        INSERT INTO residual_state
        (asset, timestamp, timescale, price, fair_value, residual,
         normalized_residual, std_error, r_squared)
        VALUES (%(asset)s, %(timestamp)s, %(timescale)s, %(price)s,
                %(fair_value)s, %(residual)s, %(normalized_residual)s,
                %(std_error)s, %(r_squared)s)
        ON CONFLICT (asset, timestamp, timescale) DO UPDATE SET
            price = EXCLUDED.price,
            fair_value = EXCLUDED.fair_value,
            residual = EXCLUDED.residual,
            normalized_residual = EXCLUDED.normalized_residual,
            std_error = EXCLUDED.std_error,
            r_squared = EXCLUDED.r_squared;
    """
    with get_cursor() as cursor:
        cursor.executemany(query, data_list)
    logger.info(f"Bulk wrote {len(data_list)} residual_state records")
