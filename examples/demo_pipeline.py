#!/usr/bin/env python3
"""
STRATA Phase 2 Demo: Complete Data Pipeline

This script demonstrates the full STRATA data pipeline:
1. Market data ingestion (using mock data)
2. Residual calculation
3. Basin identification
4. Position analysis
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strata.config import DB_CONFIG, ASSETS
from strata.db.connection import init_pool, close_pool
from strata.ingestion.market_data import MockMarketDataIngester
from strata.analysis.residuals import compute_residuals
from strata.analysis.basins import identify_basins, get_basin_summary
from strata.analysis.position import compute_basin_position, get_position_summary

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_pipeline():
    """Run complete pipeline demo."""
    logger.info("=" * 80)
    logger.info("STRATA Phase 2 Pipeline Demo")
    logger.info("=" * 80)

    # Initialize database
    logger.info("\nInitializing database connection...")
    init_pool(**DB_CONFIG)

    try:
        # Configuration
        asset = 'GLD'  # Use first asset from config
        timescale = '1d'
        n_historical_periods = 100

        logger.info(f"\nConfiguration:")
        logger.info(f"  Asset: {asset}")
        logger.info(f"  Timescale: {timescale}")
        logger.info(f"  Historical periods: {n_historical_periods}")

        # Step 1: Generate historical data
        logger.info("\n" + "-" * 80)
        logger.info("STEP 1: Generating Historical Market Data")
        logger.info("-" * 80)

        ingester = MockMarketDataIngester(seed=42)
        start_time = datetime.now() - timedelta(days=n_historical_periods)

        logger.info(f"Generating {n_historical_periods} days of synthetic data...")
        for i in range(n_historical_periods):
            timestamp = start_time + timedelta(days=i)
            ingester.ingest_market_data(asset, timescale, timestamp=timestamp)

        logger.info(f"✓ Generated {n_historical_periods} market data points")

        # Step 2: Compute residuals
        logger.info("\n" + "-" * 80)
        logger.info("STEP 2: Computing Least Squares Residuals")
        logger.info("-" * 80)

        logger.info("Computing residuals for each period...")
        for i in range(20, n_historical_periods):
            compute_residuals(asset, timescale, lookback_periods=20)

        logger.info("✓ Residuals computed")

        # Analyze residual statistics
        from strata.analysis.residuals import analyze_residual_statistics
        stats = analyze_residual_statistics(asset, timescale)

        logger.info("\nResidual Statistics:")
        logger.info(f"  Count: {stats['count']}")
        logger.info(f"  Mean: {stats['mean']:.4f}")
        logger.info(f"  Std Dev: {stats['std']:.4f}")
        logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        logger.info(f"  Median: {stats['median']:.4f}")
        logger.info(f"  Autocorr (lag 1): {stats.get('autocorr_lag1', 0):.4f}")

        # Step 3: Identify basin
        logger.info("\n" + "-" * 80)
        logger.info("STEP 3: Identifying Attractor Basin")
        logger.info("-" * 80)

        basin_id = identify_basins(asset, timescale, clustering_window=80)

        if basin_id:
            logger.info(f"✓ Basin identified: {basin_id}")

            basin_summary = get_basin_summary(basin_id)

            logger.info("\nBasin Geometry:")
            logger.info(f"  Center: {basin_summary['center']:.4f}")
            logger.info(f"  Width: {basin_summary['width']:.4f}")
            logger.info(f"  Boundaries: [{basin_summary['boundaries'][0]:.4f}, "
                       f"{basin_summary['boundaries'][1]:.4f}]")
            logger.info(f"  Velocity: {basin_summary['velocity']:.6f}")

            if basin_summary.get('stability'):
                stability = basin_summary['stability']
                logger.info(f"\nBasin Stability:")
                logger.info(f"  Overall: {stability['overall_stability']:.3f}")
                logger.info(f"  Width: {stability['width_stability']:.3f}")
                logger.info(f"  Center: {stability['center_stability']:.3f}")
                logger.info(f"  Shape: {stability['shape_stability']:.3f}")
        else:
            logger.error("✗ Failed to identify basin")
            return

        # Step 4: Analyze position
        logger.info("\n" + "-" * 80)
        logger.info("STEP 4: Analyzing Price Position in Basin")
        logger.info("-" * 80)

        position_state = compute_basin_position(asset, timescale)

        if position_state:
            logger.info(f"✓ Position state: {position_state}")

            position_summary = get_position_summary(asset, timescale)

            logger.info("\nPosition Metrics:")
            logger.info(f"  State: {position_summary['position_state']}")
            logger.info(f"  Normalized Distance: {position_summary['normalized_distance']:.3f}")
            logger.info(f"  Distance to Boundary: {position_summary['distance_to_boundary']:.4f}")
            logger.info(f"  Lag Score: {position_summary['lag_score']:.3f}")
            logger.info(f"  Basin Velocity: {position_summary['basin_velocity']:.6f}")
            logger.info(f"  Price Velocity: {position_summary['price_velocity']:.6f}")

            risk = position_summary['risk']
            logger.info(f"\nRisk Assessment:")
            logger.info(f"  Overall Risk: {risk['overall_risk']:.3f}")
            logger.info(f"  Boundary Risk: {risk['boundary_risk']:.3f}")
            logger.info(f"  Tracking Risk: {risk['tracking_risk']:.3f}")
            logger.info(f"  Velocity Risk: {risk['velocity_risk']:.3f}")
            logger.info(f"  Recommendation: {risk['recommendation']}")
        else:
            logger.error("✗ Failed to compute position")
            return

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Asset: {asset}")
        logger.info(f"Timescale: {timescale}")
        logger.info(f"Basin: {basin_id}")
        logger.info(f"Position State: {position_state}")
        logger.info(f"Risk Level: {risk['overall_risk']:.3f}")
        logger.info(f"Recommendation: {risk['recommendation']}")
        logger.info("=" * 80)
        logger.info("✓ Pipeline demo completed successfully!")

    except Exception as e:
        logger.error(f"Error in pipeline demo: {e}", exc_info=True)
        raise

    finally:
        close_pool()


if __name__ == '__main__':
    demo_pipeline()
