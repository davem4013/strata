#!/usr/bin/env python3
"""
STRATA Phase 3 Demo: Intelligence Layer

This script demonstrates the complete STRATA intelligence pipeline:
1. Market data ingestion
2. Residual calculation
3. Basin identification
4. Position analysis
5. AI model analysis (multi-model interpretation)
6. Agreement/disagreement metrics (THE CORE INNOVATION)
7. Event detection and alerting
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strata.config import DB_CONFIG
from strata.db.connection import init_pool, close_pool
from strata.db.queries import get_model_interpretations, get_agreement_metrics_current
from strata.ingestion.market_data import MockMarketDataIngester
from strata.analysis.residuals import compute_residuals
from strata.analysis.basins import identify_basins, get_basin_summary
from strata.analysis.position import compute_basin_position, get_position_summary
from strata.ai.model_analysis import ModelAnalyzer
from strata.ai.agreement import compute_agreement, get_most_divergent_models
from strata.events.detection import (
    detect_regime_events,
    summarize_events,
    generate_alert_message,
    get_active_alerts
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_intelligence_pipeline():
    """Demonstrate complete AI analysis pipeline."""

    print("\n" + "=" * 80)
    print("STRATA Phase 3: Intelligence Layer Demo")
    print("=" * 80)
    print("\nThis demo showcases STRATA's core innovation:")
    print("Using model DISAGREEMENT as a signal for regime uncertainty")
    print("=" * 80)

    # Initialize database
    logger.info("\nInitializing database connection...")
    init_pool(**DB_CONFIG)

    try:
        # Configuration
        asset = 'QQQ'
        timescale = '1d'
        n_periods = 60

        print(f"\nConfiguration:")
        print(f"  Asset: {asset}")
        print(f"  Timescale: {timescale}")
        print(f"  Periods: {n_periods}")

        # ========================================================================
        # STEP 1: Data Pipeline
        # ========================================================================

        print("\n" + "-" * 80)
        print("STEP 1: Market Data Pipeline")
        print("-" * 80)

        print("\n[1.1] Generating market data...")

        from strata.ingestion.market_data import AnalyticsAPIMarketDataIngester

        ingester = AnalyticsAPIMarketDataIngester(
            base_url="http://127.0.0.1:8000"
        )


        print(f"✓ Generated {n_periods} days of market data")

        print("\n[1.2] Computing residuals...")
        for i in range(20, n_periods):
            compute_residuals(asset, timescale)
        print("✓ Residuals computed")

        print("\n[1.3] Identifying basin structure...")
        basin_id = identify_basins(asset, timescale)
        print(f"✓ Basin identified: {basin_id}")

        basin_summary = get_basin_summary(basin_id)
        print(f"\nBasin Geometry:")
        print(f"  Center: {basin_summary['center']:.4f}")
        print(f"  Width: {basin_summary['width']:.4f}")
        print(f"  Boundaries: [{basin_summary['boundaries'][0]:.4f}, {basin_summary['boundaries'][1]:.4f}]")

        print("\n[1.4] Computing position...")
        position_state = compute_basin_position(asset, timescale)
        print(f"✓ Position state: {position_state}")

        position_summary = get_position_summary(asset, timescale)
        print(f"\nPosition Metrics:")
        print(f"  State: {position_summary['position_state']}")
        print(f"  Normalized Distance: {position_summary['normalized_distance']:.3f}")
        print(f"  Lag Score: {position_summary['lag_score']:.3f}")

        # ========================================================================
        # STEP 2: AI Model Analysis (Multi-Model Interpretation)
        # ========================================================================

        print("\n" + "-" * 80)
        print("STEP 2: AI Model Analysis (Multi-Model Interpretation)")
        print("-" * 80)
        print("\nQuerying multiple AI models to interpret basin state...")
        print("Models: Router + 4 Specialized Heads (Vol, Corr, Temporal, Stability)")

        analyzer = ModelAnalyzer(use_mock=True)
        timestamp = datetime.now()

        count = analyzer.trigger_model_analysis(basin_id, timestamp, asset, timescale)
        print(f"\n✓ Generated {count} model interpretations")

        # Show individual model interpretations
        interpretations = get_model_interpretations(basin_id, timestamp)

        print("\nModel Interpretations:")
        print("-" * 80)
        for interp in interpretations:
            print(f"\n{interp['model_id']}:")
            print(f"  Regime Type: {interp['regime_type']}")
            print(f"  Center Estimate: {interp['center_estimate']:.4f}")
            print(f"  Boundaries: [{float(interp['boundary_lower_estimate']):.4f}, {float(interp['boundary_upper_estimate']):.4f}]")
            print(f"  Stability Score: {interp['stability_score']:.3f}")
            print(f"  Confidence: {interp['confidence']:.3f}")
            print(f"  Reasoning: {interp['interpretation_text'][:80]}...")

        # ========================================================================
        # STEP 3: Agreement/Disagreement Metrics (CORE INNOVATION)
        # ========================================================================

        print("\n" + "-" * 80)
        print("STEP 3: DISAGREEMENT ENGINE (CORE INNOVATION)")
        print("-" * 80)
        print("\nAnalyzing model agreement/disagreement...")
        print("Key insight: Disagreement = Regime Uncertainty")

        agreement_score = compute_agreement(basin_id, timestamp)
        print(f"\n✓ Agreement score calculated: {agreement_score:.3f}")

        agreement = get_agreement_metrics_current(asset, timescale)

        print("\nAgreement Metrics:")
        print("-" * 80)
        print(f"  Overall Agreement: {agreement['agreement_score']:.3f}")
        print(f"  Disagreement Type: {agreement['disagreement_type'] or 'None (High Agreement)'}")
        print(f"  Semantic Distance: {agreement['semantic_distance']:.4f}")
        print(f"  Variance (Center): {agreement['variance_center']:.6f}")
        print(f"  Variance (Boundary): {agreement['variance_boundary']:.6f}")
        print(f"  Directional Divergence: {agreement['directional_divergence']:.3f}")
        print(f"  Model Count: {agreement['model_count']}")

        # Interpret agreement score
        score = float(agreement['agreement_score'])
        if score > 0.7:
            interpretation = "HIGH AGREEMENT - Stable regime, models converge"
        elif score > 0.5:
            interpretation = "MODERATE AGREEMENT - Some uncertainty"
        elif score > 0.3:
            interpretation = "LOW AGREEMENT - Significant uncertainty, caution advised"
        else:
            interpretation = "VERY LOW AGREEMENT - Chaotic regime, high risk"

        print(f"\nInterpretation: {interpretation}")

        # Show most divergent models
        divergent = get_most_divergent_models(basin_id, timestamp)
        if divergent:
            print("\nMost Divergent from Consensus:")
            for i, model in enumerate(divergent[:3], 1):
                print(f"  {i}. {model['model_id']}: divergence={model['divergence']:.4f}")

        # ========================================================================
        # STEP 4: Event Detection and Alerting
        # ========================================================================

        print("\n" + "-" * 80)
        print("STEP 4: Event Detection and Alerting")
        print("-" * 80)
        print("\nScanning for regime events...")

        events = detect_regime_events(asset, timescale, timestamp)

        if events:
            print(f"\n⚠️  Detected {len(events)} regime events")

            summary = summarize_events(events)
            print("\nEvent Summary:")
            print(f"  Total Events: {summary['total_events']}")
            print(f"  Max Severity: {summary['max_severity']:.3f}")
            print(f"  Critical Events: {summary['critical_count']}")
            print(f"  High Severity Events: {summary['high_count']}")
            print(f"  Event Types: {', '.join(summary['event_types'])}")

            print("\nDetailed Events:")
            print("-" * 80)
            for event in events:
                # Reconstruct event dict for alert message
                event_dict = {
                    'asset': asset,
                    'event_type': event['event_type'],
                    'severity': event['severity'],
                    'metadata': event['metadata']
                }
                msg = generate_alert_message(event_dict)
                print(f"\n[{event['severity']:.2f}] {msg}")
        else:
            print("\n✓ No regime events detected - all systems normal")

        # ========================================================================
        # FINAL SUMMARY
        # ========================================================================

        print("\n" + "=" * 80)
        print("INTELLIGENCE LAYER SUMMARY")
        print("=" * 80)

        print(f"\nAsset: {asset}")
        print(f"Timescale: {timescale}")
        print(f"Analysis Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        print("\nBasin State:")
        print(f"  ID: {basin_id}")
        print(f"  Center: {basin_summary['center']:.4f}")
        print(f"  Width: {basin_summary['width']:.4f}")

        print("\nPosition:")
        print(f"  State: {position_summary['position_state']}")
        print(f"  Risk: {position_summary['risk']['overall_risk']:.3f}")
        print(f"  Recommendation: {position_summary['risk']['recommendation']}")

        print("\nModel Analysis:")
        print(f"  Models Queried: {count}")
        print(f"  Agreement Score: {agreement_score:.3f}")
        print(f"  Disagreement Type: {agreement['disagreement_type'] or 'None'}")
        print(f"  Directional Divergence: {agreement['directional_divergence']:.3f}")

        print("\nEvents:")
        if events:
            print(f"  Detected: {len(events)} events")
            print(f"  Max Severity: {summary['max_severity']:.3f}")
            if summary['critical_count'] > 0:
                print(f"  ⚠️  CRITICAL ALERTS: {summary['critical_count']}")
        else:
            print(f"  Detected: 0 events (normal operation)")

        print("\n" + "=" * 80)
        print("STRATA Intelligence Layer Demo Complete!")
        print("=" * 80)
        print("\nKey Innovation Demonstrated:")
        print("  ✓ Multi-model basin interpretation")
        print("  ✓ Semantic disagreement measurement")
        print("  ✓ Directional divergence detection")
        print("  ✓ Regime event generation")
        print("  ✓ Risk-based alerting")
        print("\nDisagreement as Signal: When models can't agree, regime is uncertain.")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Error in intelligence demo: {e}", exc_info=True)
        raise

    finally:
        close_pool()


if __name__ == '__main__':
    demo_intelligence_pipeline()
