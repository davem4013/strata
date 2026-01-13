"""
Model agreement/disagreement analysis - THE CORE INNOVATION.

This module implements the disagreement engine: measuring model divergence
as a primary signal for regime uncertainty. When models agree, regime is stable.
When models disagree, regime is uncertain or transitioning.
"""
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional
import logging

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from strata.db.queries import (
    get_model_interpretations,
    write_agreement_metrics
)


logger = logging.getLogger(__name__)


def compute_agreement(basin_id: str, timestamp: datetime) -> Optional[Decimal]:
    """
    Calculate agreement metrics across all model interpretations.

    This is the core of STRATA's innovation: disagreement as signal.

    Steps:
    1. Fetch all model_interpretation records for basin_id at timestamp
    2. Check model_count >= 2 (skip if insufficient models)
    3. Calculate semantic_distance via calculate_semantic_distance()
    4. Calculate variance_center via calculate_variance_metrics()
    5. Calculate variance_boundary via calculate_variance_metrics()
    6. Calculate directional_divergence via calculate_directional_divergence()
    7. Calculate overall agreement_score via compute_agreement_score()
    8. Classify disagreement_type via classify_disagreement_type()
    9. Write to agreement_metrics table

    Args:
        basin_id: Basin identifier
        timestamp: Analysis timestamp

    Returns:
        Agreement score (0-1) if successful, None otherwise
    """
    logger.info(f"Computing agreement for {basin_id} at {timestamp}")

    # Fetch model interpretations
    interpretations_list = get_model_interpretations(basin_id, timestamp)

    if len(interpretations_list) < 2:
        logger.warning(
            f"Insufficient models for agreement calculation: "
            f"{len(interpretations_list)} < 2"
        )
        return None

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(interpretations_list)

    logger.debug(
        f"Analyzing {len(df)} model interpretations: "
        f"{df['model_id'].tolist()}"
    )

    # Extract embeddings
    embeddings = [np.array(row['embedding']) for _, row in df.iterrows()]

    # Calculate semantic distance
    semantic_distance = calculate_semantic_distance(embeddings)

    # Calculate variance metrics
    variance_metrics = calculate_variance_metrics(df)
    variance_center = variance_metrics['variance_center']
    variance_boundary = variance_metrics['variance_boundary']

    # Calculate directional divergence
        
    from strata.analysis.basins import get_basin_summary

    basin_summary = get_basin_summary(basin_id)

    if not basin_summary:
        raise ValueError(f"No basin summary available for basin {basin_id}")

    basin_center = basin_summary["center"]

    directional_divergence = calculate_directional_divergence(
        df,
        basin_center
    )


    # Calculate overall agreement score
    agreement_score = compute_agreement_score(
        df, semantic_distance, variance_center, variance_boundary
    )

    # Classify disagreement type
    disagreement_type = classify_disagreement_type(
        variance_center, variance_boundary, directional_divergence
    )

    # Extract timescale (all should be same)
    timescale = df.iloc[0]['timescale']

    # Prepare data for database
    data = {
        'basin_id': basin_id,
        'timestamp': timestamp,
        'timescale': timescale,
        'agreement_score': Decimal(str(round(agreement_score, 3))),
        'disagreement_type': disagreement_type,
        'semantic_distance': Decimal(str(round(semantic_distance, 4))),
        'variance_center': Decimal(str(round(variance_center, 6))),
        'variance_boundary': Decimal(str(round(variance_boundary, 6))),
        'directional_divergence': Decimal(str(round(directional_divergence, 3))),
        'model_count': len(df)
    }

    # Write to database
    write_agreement_metrics(data)

    logger.info(
        f"Agreement for {basin_id}: score={agreement_score:.3f}, "
        f"type={disagreement_type}, divergence={directional_divergence:.3f}"
    )

    logger.debug(
        f"  semantic_distance={semantic_distance:.4f}, "
        f"variance_center={variance_center:.6f}, "
        f"variance_boundary={variance_boundary:.6f}"
    )

    return Decimal(str(agreement_score))


def calculate_semantic_distance(embeddings: List[np.ndarray]) -> float:
    """
    Calculate average pairwise cosine distance between embeddings.

    Semantic distance measures how different the model interpretations are
    at a linguistic/conceptual level. High distance = models describing
    different scenarios.

    Args:
        embeddings: List of np.ndarray, each shape (384,)

    Returns:
        float: Average cosine distance (0 = identical, 2 = opposite)
               Typically ranges 0-1 for similar text
    """
    if len(embeddings) < 2:
        return 0.0

    distances = []

    # Calculate all pairwise distances
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            try:
                # Cosine distance = 1 - cosine_similarity
                dist = cosine(embeddings[i], embeddings[j])
                distances.append(dist)
            except Exception as e:
                logger.warning(f"Error calculating cosine distance: {e}")
                continue

    if not distances:
        return 0.0

    avg_distance = float(np.mean(distances))

    logger.debug(
        f"Semantic distance: {avg_distance:.4f} "
        f"(range: {np.min(distances):.4f} - {np.max(distances):.4f})"
    )

    return avg_distance


def calculate_variance_metrics(interpretations: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate variance in model estimates.

    High variance = models disagree on basin location/boundaries.

    Args:
        interpretations: DataFrame with columns:
            - center_estimate
            - boundary_upper_estimate
            - boundary_lower_estimate

    Returns:
        dict: {
            'variance_center': float,
            'variance_boundary': float (avg of upper and lower)
        }
    """
    # Convert to float for variance calculation
    centers = interpretations['center_estimate'].astype(float)
    uppers = interpretations['boundary_upper_estimate'].astype(float)
    lowers = interpretations['boundary_lower_estimate'].astype(float)

    # Calculate variances
    variance_center = float(np.var(centers, ddof=1))
    variance_upper = float(np.var(uppers, ddof=1))
    variance_lower = float(np.var(lowers, ddof=1))

    # Average boundary variance
    variance_boundary = (variance_upper + variance_lower) / 2.0

    logger.debug(
        f"Variance metrics: center={variance_center:.6f}, "
        f"boundary={variance_boundary:.6f}"
    )

    return {
        'variance_center': variance_center,
        'variance_boundary': variance_boundary
    }



def calculate_directional_divergence(
    df: pd.DataFrame,
    basin_center: float | None = None
) -> float:
    """
    Directional divergence = epistemic conflict score.

    0.0 → full agreement
    1.0 → perfectly balanced split
    """

    if df.empty:
        return 0.0

    if "center_estimate" not in df.columns:
        raise ValueError(
            f"Expected 'center_estimate' column, got {df.columns.tolist()}"
        )

    centers = df["center_estimate"].astype(float)


    if basin_center is None:
        # Test mode: assume zero-centered equilibrium
        reference = 0.0
    else:
        reference = basin_center


    # Direction only: -1 / +1
    directions = np.sign(centers - reference)

    # Remove exact-zero entries (perfectly on center)
    directions = directions[directions != 0]

    if len(directions) == 0:
        return 0.0

    # Agreement = magnitude of mean direction
    agreement = abs(directions.mean())

    # Divergence = lack of agreement
    divergence = 1.0 - agreement

    return float(divergence)


def compute_agreement_score(
    interpretations: pd.DataFrame,
    semantic_distance: float,
    variance_center: float,
    variance_boundary: float
) -> float:
    """
    Aggregate agreement score (0-1) based on all metrics.

    Agreement score interpretation:
    - 1.0: Perfect agreement (stable regime)
    - 0.7-1.0: High agreement (confident assessment)
    - 0.5-0.7: Moderate agreement (some uncertainty)
    - 0.3-0.5: Low agreement (significant uncertainty)
    - 0.0-0.3: Very low agreement (chaotic regime)

    Scoring weights:
    - semantic_distance: 40% (conceptual similarity)
    - variance_center: 30% (agreement on basin center)
    - variance_boundary: 20% (agreement on basin size)
    - avg_confidence: 10% (models' self-assessed confidence)

    Args:
        interpretations: DataFrame with model interpretations
        semantic_distance: Average pairwise embedding distance
        variance_center: Variance in center estimates
        variance_boundary: Variance in boundary estimates

    Returns:
        float: Agreement score (0-1)
    """
    # Normalize semantic distance (typical range 0-1 for cosine)
    # High distance = low agreement
    semantic_score = max(0.0, 1.0 - semantic_distance)

    # Normalize variances (use exponential decay)
    # High variance = low agreement
    variance_center_score = np.exp(-variance_center)
    variance_boundary_score = np.exp(-variance_boundary)

    # Average confidence across models
    avg_confidence = interpretations['confidence'].astype(float).mean()

    # Weighted combination
    agreement_score = (
        0.4 * semantic_score +
        0.3 * variance_center_score +
        0.2 * variance_boundary_score +
        0.1 * avg_confidence
    )

    # Clip to [0, 1]
    agreement_score = float(np.clip(agreement_score, 0.0, 1.0))

    logger.debug(
        f"Agreement score components: "
        f"semantic={semantic_score:.3f}, "
        f"var_center={variance_center_score:.3f}, "
        f"var_boundary={variance_boundary_score:.3f}, "
        f"confidence={avg_confidence:.3f}, "
        f"total={agreement_score:.3f}"
    )

    return agreement_score


def classify_disagreement_type(
    variance_center: float,
    variance_boundary: float,
    directional_divergence: float
) -> Optional[str]:
    """
    Classify type of disagreement.

    Disagreement types:
    - 'center': Models disagree on basin center location
    - 'boundary': Models disagree on basin size/boundaries
    - 'both': Models disagree on both center and boundaries
    - 'chaotic': Models predict opposing directions (critical!)
    - None: Low disagreement (agreement)

    Args:
        variance_center: Variance in center estimates
        variance_boundary: Variance in boundary estimates
        directional_divergence: Directional divergence score

    Returns:
        str or None: Disagreement type
    """
    # Thresholds for "high" variance
    CENTER_VARIANCE_THRESHOLD = 0.5
    BOUNDARY_VARIANCE_THRESHOLD = 0.5
    DIRECTIONAL_DIVERGENCE_THRESHOLD = 0.7

    center_high = variance_center > CENTER_VARIANCE_THRESHOLD
    boundary_high = variance_boundary > BOUNDARY_VARIANCE_THRESHOLD
    chaotic = directional_divergence > DIRECTIONAL_DIVERGENCE_THRESHOLD

    if chaotic:
        disagreement_type = 'chaotic'
    elif center_high and boundary_high:
        disagreement_type = 'both'
    elif center_high:
        disagreement_type = 'center'
    elif boundary_high:
        disagreement_type = 'boundary'
    else:
        disagreement_type = None  # Low disagreement

    if disagreement_type:
        logger.debug(
            f"Disagreement classified as '{disagreement_type}': "
            f"center_var={variance_center:.4f}, "
            f"boundary_var={variance_boundary:.4f}, "
            f"directional_div={directional_divergence:.3f}"
        )

    return disagreement_type


def analyze_disagreement_trend(
    basin_id: str,
    lookback: int = 10
) -> Dict:
    """
    Analyze trend in model agreement over time.

    Useful for detecting regime transitions:
    - Decreasing agreement = regime becoming unstable
    - Increasing disagreement = potential phase transition

    Args:
        basin_id: Basin identifier
        lookback: Number of historical periods to analyze

    Returns:
        Dict with trend metrics
    """
    # TODO: Implement after adding get_agreement_history query
    logger.warning("analyze_disagreement_trend not yet implemented")
    return {}


def get_most_divergent_models(
    basin_id: str,
    timestamp: datetime
) -> List[Dict]:
    """
    Identify which models are most divergent from consensus.

    Useful for understanding source of disagreement.

    Args:
        basin_id: Basin identifier
        timestamp: Analysis timestamp

    Returns:
        List of dicts with model_id and divergence score
    """
    interpretations_list = get_model_interpretations(basin_id, timestamp)

    if len(interpretations_list) < 2:
        return []

    df = pd.DataFrame(interpretations_list)

    # Calculate consensus (median)
    consensus_center = df['center_estimate'].astype(float).median()

    # Calculate each model's divergence from consensus
    divergences = []
    for _, row in df.iterrows():
        center = float(row['center_estimate'])
        divergence = abs(center - consensus_center)

        divergences.append({
            'model_id': row['model_id'],
            'center_estimate': center,
            'consensus_center': consensus_center,
            'divergence': divergence,
            'stability_score': float(row['stability_score']),
            'confidence': float(row['confidence'])
        })

    # Sort by divergence (highest first)
    divergences.sort(key=lambda x: x['divergence'], reverse=True)

    return divergences
