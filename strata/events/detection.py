"""Regime event detection and alerting."""
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Optional
import logging
import json

from strata.db.queries import (
    get_current_basin,
    get_basin_history,
    get_basin_position_current,
    get_agreement_metrics_current,
    write_regime_event,
    get_active_alerts as db_get_active_alerts
)

logger = logging.getLogger(__name__)


def detect_regime_events(
    asset: str,
    timescale: str,
    timestamp: Optional[datetime] = None
) -> List[Dict]:
    """
    Scan for regime events and write to regime_event table.

    Checks for:
    1. High disagreement (agreement_score < 0.5)
    2. Directional divergence (directional_divergence > 0.7)
    3. Position detachment (position_state = 'detaching')
    4. Position at edge (position_state = 'edge')
    5. Basin collapse (basin_width shrinking rapidly)

    Args:
        asset: Asset symbol
        timescale: Time resolution
        timestamp: Event timestamp (defaults to now)

    Returns:
        List of detected events
    """
    if timestamp is None:
        timestamp = datetime.now()

    logger.info(f"Detecting regime events for {asset} {timescale}")

    # Get current basin
    basin = get_current_basin(asset, timescale)
    if not basin:
        logger.warning(f"No basin found for {asset} {timescale}")
        return []

    basin_id = basin['basin_id']

    # Get agreement metrics
    agreement = get_agreement_metrics_current(asset, timescale)

    # Get position
    position = get_basin_position_current(asset, timescale)

    # Get basin history
    basin_history = get_basin_history(basin_id, n=5)

    # Collect detected events
    events = []

    # Check 1: High disagreement
    if agreement and float(agreement['agreement_score']) < 0.5:
        severity = 1.0 - float(agreement['agreement_score'])
        events.append({
            'event_type': 'high_disagreement',
            'severity': severity,
            'source': 'agreement',
            'metadata': {
                'agreement_score': float(agreement['agreement_score']),
                'disagreement_type': agreement['disagreement_type']
            }
        })
        logger.warning(
            f"High disagreement detected for {asset}: "
            f"score={agreement['agreement_score']:.3f}"
        )

    # Check 2: Directional divergence (CRITICAL)
    if agreement and float(agreement['directional_divergence']) > 0.7:
        severity = float(agreement['directional_divergence'])
        events.append({
            'event_type': 'directional_divergence',
            'severity': severity,
            'source': 'agreement',
            'metadata': {
                'directional_divergence': float(agreement['directional_divergence']),
                'disagreement_type': agreement['disagreement_type']
            }
        })
        logger.warning(
            f"Directional divergence detected for {asset}: "
            f"divergence={agreement['directional_divergence']:.3f}"
        )

    # Check 3: Position detachment (CRITICAL)
    if position and position['position_state'] == 'detaching':
        severity = min(float(position['normalized_distance']), 1.0)
        events.append({
            'event_type': 'position_detaching',
            'severity': severity,
            'source': 'position',
            'metadata': {
                'position_state': position['position_state'],
                'normalized_distance': float(position['normalized_distance']),
                'lag_score': float(position['lag_score'])
            }
        })
        logger.error(
            f"CRITICAL: Position detaching for {asset}: "
            f"distance={position['normalized_distance']:.3f}"
        )

    # Check 4: Position at edge (WARNING)
    if position and position['position_state'] == 'edge':
        severity = 0.7
        events.append({
            'event_type': 'exit_warning',
            'severity': severity,
            'source': 'position',
            'metadata': {
                'position_state': position['position_state'],
                'distance_to_boundary': float(position['distance_to_boundary'])
            }
        })
        logger.warning(
            f"Exit warning for {asset}: position at edge"
        )

    # Check 5: Basin collapse
    if len(basin_history) >= 3:
        widths = [float(b['basin_width']) for b in basin_history]

        # Calculate rate of change
        width_change_rate = (widths[0] - widths[-1]) / widths[-1]

        # Check for rapid shrinkage (30% or more)
        if width_change_rate < -0.3:
            severity = min(abs(width_change_rate), 1.0)
            events.append({
                'event_type': 'basin_collapse',
                'severity': severity,
                'source': 'geometry',
                'metadata': {
                    'width_change_rate': width_change_rate,
                    'current_width': widths[0],
                    'previous_width': widths[-1],
                    'periods': len(widths)
                }
            })
            logger.error(
                f"Basin collapse detected for {asset}: "
                f"width change={width_change_rate:.1%}"
            )

    # Write events to database
    for event in events:
        create_event(
            asset=asset,
            timescale=timescale,
            timestamp=timestamp,
            **event
        )

    if events:
        logger.info(f"Detected {len(events)} regime events for {asset}")
    else:
        logger.debug(f"No regime events detected for {asset}")

    return events


def create_event(
    asset: str,
    timescale: str,
    timestamp: datetime,
    event_type: str,
    severity: float,
    source: str,
    metadata: Dict
) -> None:
    """
    Write regime event to database.

    Args:
        asset: Asset symbol
        timescale: Time resolution
        timestamp: Event timestamp
        event_type: Event type identifier
        severity: Severity score (0-1)
        source: Source of detection (agreement/position/geometry/cascade)
        metadata: Event-specific metadata
    """
    # Prepare data
    data = {
        'asset': asset,
        'timescale': timescale,
        'timestamp': timestamp,
        'event_type': event_type,
        'severity': Decimal(str(round(severity, 3))),
        'source': source,
        'metadata': json.dumps(metadata)
    }

    # Write to database
    write_regime_event(data)

    # Log based on severity
    if severity > 0.8:
        logger.critical(
            f"CRITICAL EVENT: {event_type} for {asset} - severity={severity:.2f}"
        )
    elif severity > 0.7:
        logger.error(
            f"HIGH SEVERITY EVENT: {event_type} for {asset} - severity={severity:.2f}"
        )
    elif severity > 0.5:
        logger.warning(
            f"Medium severity event: {event_type} for {asset} - severity={severity:.2f}"
        )
    else:
        logger.info(
            f"Low severity event: {event_type} for {asset} - severity={severity:.2f}"
        )


def get_active_alerts(lookback_hours: int = 24) -> List[Dict]:
    """
    Retrieve recent high-severity events.

    Args:
        lookback_hours: Hours to look back (default 24)

    Returns:
        List of high-severity events
    """
    return db_get_active_alerts(lookback_hours)


def get_critical_alerts(lookback_hours: int = 24) -> List[Dict]:
    """
    Retrieve critical alerts (severity > 0.8).

    Args:
        lookback_hours: Hours to look back

    Returns:
        List of critical events
    """
    all_alerts = get_active_alerts(lookback_hours)

    # Filter for critical (severity > 0.8)
    critical = [
        alert for alert in all_alerts
        if float(alert['severity']) > 0.8
    ]

    return critical


def summarize_events(events: List[Dict]) -> Dict:
    """
    Summarize detected events for reporting.

    Args:
        events: List of event dicts

    Returns:
        Summary dict
    """
    if not events:
        return {
            'total_events': 0,
            'max_severity': 0.0,
            'critical_count': 0,
            'event_types': []
        }

    severities = [e['severity'] for e in events]
    event_types = [e['event_type'] for e in events]

    summary = {
        'total_events': len(events),
        'max_severity': max(severities),
        'avg_severity': sum(severities) / len(severities),
        'critical_count': sum(1 for s in severities if s > 0.8),
        'high_count': sum(1 for s in severities if 0.7 < s <= 0.8),
        'medium_count': sum(1 for s in severities if 0.5 < s <= 0.7),
        'event_types': list(set(event_types)),
        'event_type_counts': {
            et: event_types.count(et) for et in set(event_types)
        }
    }

    return summary


def check_cascade_across_timescales(
    asset: str,
    timescales: List[str],
    timestamp: Optional[datetime] = None
) -> bool:
    """
    Check if disagreement is cascading across timescales.

    Cascade indicator: disagreement appears first on fast timescales,
    then propagates to slower timescales. This signals a regime
    transition in progress.

    Args:
        asset: Asset symbol
        timescales: List of timescales (ordered fast to slow)
        timestamp: Check timestamp (defaults to now)

    Returns:
        bool: True if cascade detected
    """
    if timestamp is None:
        timestamp = datetime.now()

    logger.info(f"Checking for disagreement cascade in {asset}")

    # Get agreement scores for each timescale
    agreement_scores = []

    for ts in timescales:
        agreement = get_agreement_metrics_current(asset, ts)

        if agreement:
            score = float(agreement['agreement_score'])
            agreement_scores.append((ts, score))
        else:
            agreement_scores.append((ts, None))

    # Check for cascade pattern:
    # - Fast timescales have low agreement
    # - Slow timescales have progressively lower agreement
    # - Indicates disagreement propagating upward

    # Filter out None values
    valid_scores = [(ts, score) for ts, score in agreement_scores if score is not None]

    if len(valid_scores) < 2:
        logger.debug("Insufficient data for cascade detection")
        return False

    # Check if scores are decreasing (getting worse) as we go to slower timescales
    # and if fastest timescale has low agreement
    if valid_scores[0][1] < 0.5:  # Fast timescale has disagreement
        # Check if trend continues
        cascade = True
        for i in range(len(valid_scores) - 1):
            # Allow for small increases, but trend should be downward
            if valid_scores[i + 1][1] > valid_scores[i][1] + 0.2:
                cascade = False
                break

        if cascade:
            logger.warning(
                f"Cascade detected for {asset}: "
                f"scores={[f'{s:.2f}' for _, s in valid_scores]}"
            )
            return True

    return False


def generate_alert_message(event: Dict) -> str:
    """
    Generate human-readable alert message.

    Args:
        event: Event dict

    Returns:
        Alert message string
    """
    asset = event['asset']
    event_type = event['event_type']
    severity = float(event['severity'])
    metadata = json.loads(event['metadata']) if isinstance(event['metadata'], str) else event['metadata']

    # Event-specific messages
    if event_type == 'high_disagreement':
        agreement_score = metadata.get('agreement_score', 'N/A')
        msg = (
            f"HIGH DISAGREEMENT for {asset}: "
            f"Models cannot agree on regime state. "
            f"Agreement score: {agreement_score:.3f}. "
            f"Recommendation: REDUCE EXPOSURE."
        )

    elif event_type == 'directional_divergence':
        divergence = metadata.get('directional_divergence', 'N/A')
        msg = (
            f"DIRECTIONAL DIVERGENCE for {asset}: "
            f"Models predicting opposing directions - chaotic regime. "
            f"Divergence: {divergence:.3f}. "
            f"Recommendation: EXIT POSITIONS."
        )

    elif event_type == 'position_detaching':
        msg = (
            f"POSITION DETACHING for {asset}: "
            f"Price breaking away from basin - regime transition likely. "
            f"Recommendation: EXIT IMMEDIATELY."
        )

    elif event_type == 'exit_warning':
        msg = (
            f"EXIT WARNING for {asset}: "
            f"Price at edge of basin. "
            f"Recommendation: MONITOR CLOSELY, prepare to exit."
        )

    elif event_type == 'basin_collapse':
        width_change = metadata.get('width_change_rate', 'N/A')
        msg = (
            f"BASIN COLLAPSE for {asset}: "
            f"Basin width shrinking rapidly ({width_change:.1%}). "
            f"Recommendation: EXIT POSITIONS."
        )

    else:
        msg = f"Event {event_type} for {asset} - severity {severity:.2f}"

    return msg
