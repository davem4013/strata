"""Integration tests for STRATA Phase 3 AI pipeline."""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from strata.config import DB_CONFIG, setup_logging
from strata.db.connection import init_pool, close_pool
from strata.db.queries import (
    get_model_interpretations,
    get_agreement_metrics_current
)
from strata.ingestion.market_data import MockMarketDataIngester
from strata.analysis.residuals import compute_residuals
from strata.analysis.basins import identify_basins
from strata.analysis.position import compute_basin_position
from strata.ai.model_analysis import ModelAnalyzer
from strata.ai.agreement import compute_agreement, calculate_semantic_distance
from strata.events.detection import detect_regime_events, summarize_events

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def db_connection():
    """Initialize database connection for tests."""
    init_pool(**DB_CONFIG)
    yield
    close_pool()


def test_model_analysis(db_connection):
    """Test model analysis with mock responses."""
    logger.info("=" * 80)
    logger.info("AI TEST: Model Analysis")
    logger.info("=" * 80)

    asset = 'TEST_AI'
    timescale = '1d'

    # Generate test basin data
    logger.info("\n[1/5] Generating test data...")
    ingester = MockMarketDataIngester(seed=100)

    for i in range(50):
        timestamp = datetime.now() - timedelta(days=50 - i)
        ingester.ingest_market_data(asset, timescale, timestamp=timestamp)

    # Compute residuals
    logger.info("[2/5] Computing residuals...")
    for i in range(20, 50):
        compute_residuals(asset, timescale)

    # Identify basin
    logger.info("[3/5] Identifying basin...")
    basin_id = identify_basins(asset, timescale)
    assert basin_id is not None, "Failed to identify basin"

    # Compute position
    compute_basin_position(asset, timescale)

    # Run model analysis
    logger.info("[4/5] Running model analysis...")
    analyzer = ModelAnalyzer(use_mock=True)
    timestamp = datetime.now()

    count = analyzer.trigger_model_analysis(basin_id, timestamp, asset, timescale)

    logger.info(f"✓ Generated {count} model interpretations")
    assert count >= 2, f"Expected at least 2 model interpretations, got {count}"

    # Verify interpretations
    logger.info("[5/5] Verifying interpretations...")
    interpretations = get_model_interpretations(basin_id, timestamp)

    assert len(interpretations) == count, "Interpretation count mismatch"

    for interp in interpretations:
        logger.info(f"  Model {interp['model_id']}:")
        logger.info(f"    Regime: {interp['regime_type']}")
        logger.info(f"    Center: {interp['center_estimate']:.4f}")
        logger.info(f"    Stability: {interp['stability_score']:.3f}")
        logger.info(f"    Confidence: {interp['confidence']:.3f}")

        # Check required fields
        assert interp['regime_type'] in ['stable', 'transitional', 'bifurcating', 'collapsing']
        assert interp['center_estimate'] is not None
        assert interp['boundary_upper_estimate'] is not None
        assert interp['boundary_lower_estimate'] is not None
        assert 0 <= float(interp['stability_score']) <= 1
        assert 0 <= float(interp['confidence']) <= 1
        assert interp['interpretation_text'] is not None
        assert interp['embedding'] is not None
        assert len(interp['embedding']) == 384, "Embedding dimension mismatch"

    logger.info("\n✓ Model analysis test PASSED")


def test_agreement_calculation(db_connection):
    """Test disagreement engine."""
    logger.info("\n" + "=" * 80)
    logger.info("AI TEST: Agreement Calculation (CORE INNOVATION)")
    logger.info("=" * 80)

    asset = 'TEST_AGREE'
    timescale = '1d'

    # Setup test data
    logger.info("\n[1/4] Setting up test scenario...")
    ingester = MockMarketDataIngester(seed=200)

    for i in range(50):
        timestamp = datetime.now() - timedelta(days=50 - i)
        ingester.ingest_market_data(asset, timescale, timestamp=timestamp)

    for i in range(20, 50):
        compute_residuals(asset, timescale)

    basin_id = identify_basins(asset, timescale)
    compute_basin_position(asset, timescale)

    # Run model analysis
    logger.info("[2/4] Running model analysis...")
    analyzer = ModelAnalyzer(use_mock=True)
    timestamp = datetime.now()

    count = analyzer.trigger_model_analysis(basin_id, timestamp, asset, timescale)
    assert count >= 2, "Need at least 2 models for agreement test"

    # Compute agreement
    logger.info("[3/4] Computing agreement metrics...")
    agreement_score = compute_agreement(basin_id, timestamp)

    assert agreement_score is not None, "Agreement score should not be None"
    assert 0 <= float(agreement_score) <= 1, "Agreement score out of range"

    # Verify agreement metrics
    logger.info("[4/4] Verifying agreement metrics...")
    agreement = get_agreement_metrics_current(asset, timescale)

    assert agreement is not None, "No agreement metrics found"

    logger.info(f"\nAgreement Metrics:")
    logger.info(f"  Score: {agreement['agreement_score']:.3f}")
    logger.info(f"  Type: {agreement['disagreement_type']}")
    logger.info(f"  Semantic Distance: {agreement['semantic_distance']:.4f}")
    logger.info(f"  Variance Center: {agreement['variance_center']:.6f}")
    logger.info(f"  Variance Boundary: {agreement['variance_boundary']:.6f}")
    logger.info(f"  Directional Divergence: {agreement['directional_divergence']:.3f}")
    logger.info(f"  Model Count: {agreement['model_count']}")

    # Validate ranges
    assert 0 <= float(agreement['agreement_score']) <= 1
    assert 0 <= float(agreement['semantic_distance']) <= 2
    assert float(agreement['variance_center']) >= 0
    assert float(agreement['variance_boundary']) >= 0
    assert 0 <= float(agreement['directional_divergence']) <= 1
    assert int(agreement['model_count']) == count

    logger.info("\n✓ Agreement calculation test PASSED")


def test_semantic_distance():
    """Test semantic distance calculation."""
    logger.info("\n" + "=" * 80)
    logger.info("UNIT TEST: Semantic Distance")
    logger.info("=" * 80)

    # Create test embeddings
    logger.info("\nTesting semantic distance with synthetic embeddings...")

    # Test 1: Identical embeddings -> distance = 0
    emb1 = np.random.randn(384)
    emb2 = emb1.copy()

    distance_identical = calculate_semantic_distance([emb1, emb2])
    logger.info(f"[1/3] Identical embeddings: distance={distance_identical:.4f}")
    assert distance_identical < 0.01, "Identical embeddings should have ~0 distance"

    # Test 2: Similar embeddings -> small distance
    emb3 = emb1 + np.random.randn(384) * 0.1
    distance_similar = calculate_semantic_distance([emb1, emb3])
    logger.info(f"[2/3] Similar embeddings: distance={distance_similar:.4f}")
    assert distance_similar < 0.5, "Similar embeddings should have small distance"

    # Test 3: Different embeddings -> larger distance
    emb4 = np.random.randn(384)
    distance_different = calculate_semantic_distance([emb1, emb4])
    logger.info(f"[3/3] Different embeddings: distance={distance_different:.4f}")
    assert distance_different > distance_similar, "Different should have larger distance"

    logger.info("\n✓ Semantic distance test PASSED")


def test_event_detection(db_connection):
    """Test event generation."""
    logger.info("\n" + "=" * 80)
    logger.info("AI TEST: Event Detection")
    logger.info("=" * 80)

    asset = 'TEST_EVENTS'
    timescale = '1d'

    # Setup scenario with high disagreement
    logger.info("\n[1/4] Creating high disagreement scenario...")
    ingester = MockMarketDataIngester(seed=300)

    # Generate volatile data to trigger disagreement
    for i in range(50):
        timestamp = datetime.now() - timedelta(days=50 - i)
        ingester.ingest_market_data(asset, timescale, timestamp=timestamp)

    for i in range(20, 50):
        compute_residuals(asset, timescale)

    basin_id = identify_basins(asset, timescale)
    compute_basin_position(asset, timescale)

    # Run model analysis
    logger.info("[2/4] Running model analysis...")
    analyzer = ModelAnalyzer(use_mock=True)
    timestamp = datetime.now()

    analyzer.trigger_model_analysis(basin_id, timestamp, asset, timescale)

    # Compute agreement
    logger.info("[3/4] Computing agreement...")
    compute_agreement(basin_id, timestamp)

    # Detect events
    logger.info("[4/4] Detecting regime events...")
    events = detect_regime_events(asset, timescale, timestamp)

    logger.info(f"\n✓ Detected {len(events)} events")

    # Summarize events
    summary = summarize_events(events)

    logger.info(f"\nEvent Summary:")
    logger.info(f"  Total Events: {summary['total_events']}")
    if summary['total_events'] > 0:
        logger.info(f"  Max Severity: {summary['max_severity']:.3f}")
        logger.info(f"  Avg Severity: {summary['avg_severity']:.3f}")
        logger.info(f"  Critical: {summary['critical_count']}")
        logger.info(f"  High: {summary['high_count']}")
        logger.info(f"  Event Types: {summary['event_types']}")

        for event in events:
            logger.info(
                f"    - {event['event_type']}: severity={event['severity']:.2f}, "
                f"source={event['source']}"
            )

    logger.info("\n✓ Event detection test PASSED")


def test_directional_divergence():
    """Test directional divergence calculation."""
    logger.info("\n" + "=" * 80)
    logger.info("UNIT TEST: Directional Divergence")
    logger.info("=" * 80)

    from strata.ai.agreement import calculate_directional_divergence
    import pandas as pd

    # Test 1: All models agree (same direction)
    logger.info("\n[1/3] Testing same direction (low divergence)...")
    df_agree = pd.DataFrame({
        'center_estimate': [1.0, 1.1, 1.2, 0.9, 1.05]
    })
    divergence_low = calculate_directional_divergence(df_agree)
    logger.info(f"  Same direction divergence: {divergence_low:.3f}")
    assert divergence_low < 0.3, "Same direction should have low divergence"

    # Test 2: Models split (opposing directions)
    logger.info("[2/3] Testing opposing directions (high divergence)...")
    df_oppose = pd.DataFrame({
        'center_estimate': [1.0, 1.2, -1.0, -1.1, -0.9]
    })
    divergence_high = calculate_directional_divergence(df_oppose)
    logger.info(f"  Opposing directions divergence: {divergence_high:.3f}")
    assert divergence_high > 0.7, "Opposing directions should have high divergence"

    # Test 3: Balanced split
    logger.info("[3/3] Testing balanced split...")
    df_balanced = pd.DataFrame({
        'center_estimate': [0.5, -0.5, 0.6, -0.6]
    })
    divergence_balanced = calculate_directional_divergence(df_balanced)
    logger.info(f"  Balanced split divergence: {divergence_balanced:.3f}")
    assert divergence_balanced > 0.8, "Balanced split should have very high divergence"

    logger.info("\n✓ Directional divergence test PASSED")


if __name__ == '__main__':
    """Run tests standalone."""
    # Initialize DB
    init_pool(**DB_CONFIG)

    try:
        # Run tests
        test_model_analysis(None)
        test_agreement_calculation(None)
        test_semantic_distance()
        test_event_detection(None)
        test_directional_divergence()

        print("\n" + "=" * 80)
        print("ALL AI TESTS PASSED ✓")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise

    finally:
        close_pool()
