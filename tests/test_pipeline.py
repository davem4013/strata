"""Integration tests for STRATA Phase 2 data pipeline."""
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
    get_recent_market_data,
    get_recent_residuals,
    get_current_basin,
    get_basin_position_current
)
from strata.ingestion.market_data import MockMarketDataIngester, generate_historical_data
from strata.analysis.residuals import compute_residuals, analyze_residual_statistics
from strata.analysis.basins import identify_basins, get_basin_summary
from strata.analysis.position import compute_basin_position, get_position_summary

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def db_connection():
    """Initialize database connection for tests."""
    init_pool(**DB_CONFIG)
    yield
    close_pool()


def test_full_pipeline(db_connection):
    """
    Test complete data pipeline: ingestion -> residuals -> basins -> position.

    This test:
    1. Generates synthetic market data
    2. Computes residuals
    3. Identifies basin structure
    4. Analyzes position
    5. Verifies all components work together
    """
    logger.info("=" * 80)
    logger.info("INTEGRATION TEST: Full Pipeline")
    logger.info("=" * 80)

    asset = 'TEST'
    timescale = '1d'
    n_periods = 100

    # Step 1: Generate synthetic data
    logger.info(f"\n[1/5] Generating {n_periods} periods of synthetic data...")

    ingester = MockMarketDataIngester(seed=42)
    start_time = datetime.now() - timedelta(days=n_periods)

    for i in range(n_periods):
        timestamp = start_time + timedelta(days=i)
        ingester.ingest_market_data(asset, timescale, timestamp=timestamp)

    # Verify data written
    market_data = get_recent_market_data(asset, timescale, n=n_periods)
    assert len(market_data) == n_periods, f"Expected {n_periods} market data points, got {len(market_data)}"
    logger.info(f"✓ Generated {len(market_data)} market data points")

    # Step 2: Compute residuals
    logger.info(f"\n[2/5] Computing residuals...")

    for i in range(20, n_periods):
        compute_residuals(asset, timescale, lookback_periods=20)

    residual_data = get_recent_residuals(asset, timescale, n=100)
    assert len(residual_data) > 0, "No residuals computed"
    logger.info(f"✓ Computed {len(residual_data)} residuals")

    # Analyze residual statistics
    stats = analyze_residual_statistics(asset, timescale)
    logger.info(f"  Residual statistics:")
    logger.info(f"    Mean: {stats['mean']:.4f}")
    logger.info(f"    Std: {stats['std']:.4f}")
    logger.info(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    # Verify residuals are reasonable
    assert abs(stats['mean']) < 2.0, f"Residual mean too large: {stats['mean']}"
    assert stats['std'] > 0, "Residual std should be positive"

    # Step 3: Identify basin structure
    logger.info(f"\n[3/5] Identifying basin structure...")

    basin_id = identify_basins(asset, timescale, clustering_window=80)
    assert basin_id is not None, "Failed to identify basin"
    logger.info(f"✓ Identified basin: {basin_id}")

    # Verify basin geometry
    basin = get_current_basin(asset, timescale)
    assert basin is not None, "No basin geometry found"

    basin_summary = get_basin_summary(basin_id)
    logger.info(f"  Basin summary:")
    logger.info(f"    Center: {basin_summary['center']:.4f}")
    logger.info(f"    Width: {basin_summary['width']:.4f}")
    logger.info(f"    Boundaries: [{basin_summary['boundaries'][0]:.4f}, {basin_summary['boundaries'][1]:.4f}]")
    logger.info(f"    Velocity: {basin_summary['velocity']:.6f}")

    # Verify basin parameters are reasonable
    assert abs(basin_summary['center']) < 5.0, f"Basin center too far from origin: {basin_summary['center']}"
    assert basin_summary['width'] > 0, "Basin width should be positive"

    # Step 4: Compute basin position
    logger.info(f"\n[4/5] Computing basin position...")

    position_state = compute_basin_position(asset, timescale)
    assert position_state is not None, "Failed to compute position"
    logger.info(f"✓ Position state: {position_state}")

    # Verify position data
    position = get_basin_position_current(asset, timescale)
    assert position is not None, "No position data found"

    position_summary = get_position_summary(asset, timescale)
    logger.info(f"  Position summary:")
    logger.info(f"    State: {position_summary['position_state']}")
    logger.info(f"    Normalized distance: {position_summary['normalized_distance']:.3f}")
    logger.info(f"    Lag score: {position_summary['lag_score']:.3f}")
    logger.info(f"    Risk: {position_summary['risk']['overall_risk']:.3f}")
    logger.info(f"    Recommendation: {position_summary['risk']['recommendation']}")

    # Verify position metrics are in valid ranges
    assert 0 <= position_summary['normalized_distance'] <= 1.5, "Normalized distance out of range"
    assert 0 <= position_summary['lag_score'] <= 1.0, "Lag score out of range"

    # Step 5: Summary
    logger.info(f"\n[5/5] Pipeline Summary")
    logger.info(f"=" * 80)
    logger.info(f"Asset: {asset}, Timescale: {timescale}")
    logger.info(f"Data points: {len(market_data)}")
    logger.info(f"Residuals: {len(residual_data)}")
    logger.info(f"Basin: {basin_id}")
    logger.info(f"Position: {position_state}")
    logger.info(f"Risk: {position_summary['risk']['overall_risk']:.3f} ({position_summary['risk']['recommendation']})")
    logger.info(f"=" * 80)
    logger.info("✓ Full pipeline test PASSED")


def test_basin_clustering():
    """Test basin clustering with different data patterns."""
    logger.info("\n" + "=" * 80)
    logger.info("UNIT TEST: Basin Clustering")
    logger.info("=" * 80)

    from strata.analysis.basins import cluster_residuals

    # Test 1: Mean-reverting data (should have narrow basin)
    logger.info("\n[1/3] Testing mean-reverting data...")
    mean_reverting = np.random.normal(0, 0.5, 100)

    basin1 = cluster_residuals(mean_reverting)
    logger.info(f"  Center: {basin1['center']:.4f}")
    logger.info(f"  Width: {basin1['width']:.4f}")
    logger.info(f"  Boundaries: [{basin1['boundary_lower']:.4f}, {basin1['boundary_upper']:.4f}]")

    assert abs(basin1['center']) < 0.5, "Center should be near 0 for mean-reverting data"
    assert basin1['width'] < 3.0, "Width should be narrow for mean-reverting data"

    # Test 2: Trending data (should have wider basin or off-center)
    logger.info("\n[2/3] Testing trending data...")
    trending = np.linspace(-1, 1, 100) + np.random.normal(0, 0.3, 100)

    basin2 = cluster_residuals(trending)
    logger.info(f"  Center: {basin2['center']:.4f}")
    logger.info(f"  Width: {basin2['width']:.4f}")
    logger.info(f"  Boundaries: [{basin2['boundary_lower']:.4f}, {basin2['boundary_upper']:.4f}]")

    assert basin2['width'] > 0, "Width should be positive"

    # Test 3: High volatility data (should have wide basin)
    logger.info("\n[3/3] Testing high volatility data...")
    high_vol = np.random.normal(0, 2.0, 100)

    basin3 = cluster_residuals(high_vol)
    logger.info(f"  Center: {basin3['center']:.4f}")
    logger.info(f"  Width: {basin3['width']:.4f}")
    logger.info(f"  Boundaries: [{basin3['boundary_lower']:.4f}, {basin3['boundary_upper']:.4f}]")

    assert basin3['width'] > basin1['width'], "High vol should have wider basin than low vol"

    logger.info("\n✓ Basin clustering test PASSED")


def test_position_states():
    """Test position state classification."""
    logger.info("\n" + "=" * 80)
    logger.info("UNIT TEST: Position State Classification")
    logger.info("=" * 80)

    from strata.analysis.position import calculate_position_state

    # Test different position scenarios
    test_cases = [
        (0.1, 0.5, 0.9, 'centered'),    # Near center, tracking well
        (0.4, 0.3, 0.8, 'tracking'),    # Mid-range, tracking
        (0.75, 0.1, 0.7, 'edge'),       # Near boundary
        (0.5, 0.3, 0.3, 'lagging'),     # Not tracking
        (0.95, 0.05, 0.2, 'detaching'), # Beyond boundary, not tracking
    ]

    for i, (norm_dist, dist_bound, lag, expected) in enumerate(test_cases, 1):
        state = calculate_position_state(norm_dist, dist_bound, lag)
        logger.info(
            f"[{i}/5] norm_dist={norm_dist:.2f}, lag={lag:.2f} -> "
            f"state='{state}' (expected: '{expected}')"
        )
        assert state == expected, f"Expected {expected}, got {state}"

    logger.info("\n✓ Position state classification test PASSED")


def test_mock_ingester_consistency():
    """Test that mock ingester produces consistent data with same seed."""
    logger.info("\n" + "=" * 80)
    logger.info("UNIT TEST: Mock Ingester Consistency")
    logger.info("=" * 80)

    from strata.ingestion.market_data import MockMarketDataIngester

    # Create two ingesters with same seed
    ingester1 = MockMarketDataIngester(seed=123)
    ingester2 = MockMarketDataIngester(seed=123)

    # Generate prices
    price1 = ingester1.fetch_latest_price('GLD')
    price2 = ingester2.fetch_latest_price('GLD')

    logger.info(f"Ingester 1 price: {price1}")
    logger.info(f"Ingester 2 price: {price2}")

    # Should be identical with same seed
    assert price1 == price2, "Prices should be identical with same seed"

    # Test different seed gives different result
    ingester3 = MockMarketDataIngester(seed=456)
    price3 = ingester3.fetch_latest_price('GLD')

    logger.info(f"Ingester 3 price (different seed): {price3}")
    assert price3 != price1, "Different seed should give different price"

    logger.info("\n✓ Mock ingester consistency test PASSED")


if __name__ == '__main__':
    """Run tests standalone."""
    # Initialize DB
    init_pool(**DB_CONFIG)

    try:
        # Run tests
        test_full_pipeline(None)
        test_basin_clustering()
        test_position_states()
        test_mock_ingester_consistency()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise

    finally:
        close_pool()
