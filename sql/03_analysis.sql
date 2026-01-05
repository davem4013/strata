-- ============================================================================
-- STRATA Analysis Schema
-- ============================================================================
-- Tables: cross_scale_coherence, regime_event
-- Dependencies: PostgreSQL 14+
-- ============================================================================

-- ============================================================================
-- 1. cross_scale_coherence - Disagreement propagation across timescales
-- ============================================================================

CREATE TABLE IF NOT EXISTS cross_scale_coherence (
    id SERIAL PRIMARY KEY,
    basin_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    parent_timescale VARCHAR(10) NOT NULL,
    child_timescale VARCHAR(10) NOT NULL,

    coherence_score DECIMAL(4,3) NOT NULL CHECK (coherence_score BETWEEN 0 AND 1),
    cascade_flag BOOLEAN DEFAULT FALSE,
    persistence_count INT DEFAULT 1,
    direction VARCHAR(20),

    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(basin_id, timestamp, parent_timescale, child_timescale)
);

CREATE INDEX IF NOT EXISTS idx_cascade_detection
    ON cross_scale_coherence(basin_id, timestamp DESC)
    WHERE cascade_flag = TRUE;

CREATE INDEX IF NOT EXISTS idx_coherence_lookup
    ON cross_scale_coherence(basin_id, parent_timescale, child_timescale, timestamp DESC);

COMMENT ON TABLE cross_scale_coherence IS 'Disagreement propagation across timescales - detects phase transitions';
COMMENT ON COLUMN cross_scale_coherence.basin_id IS 'Reference basin (typically the parent/slower timescale)';
COMMENT ON COLUMN cross_scale_coherence.parent_timescale IS 'SLOWER timescale (e.g., weekly)';
COMMENT ON COLUMN cross_scale_coherence.child_timescale IS 'FASTER timescale (e.g., daily)';
COMMENT ON COLUMN cross_scale_coherence.coherence_score IS '1=high disagreement alignment, 0=independent disagreement';
COMMENT ON COLUMN cross_scale_coherence.cascade_flag IS 'TRUE when disagreement cascades from fast to slow timescale';
COMMENT ON COLUMN cross_scale_coherence.persistence_count IS 'Number of consecutive periods cascade has been active';
COMMENT ON COLUMN cross_scale_coherence.direction IS 'Values: upward (fast→slow cascade), stable, dampening';

-- ============================================================================
-- 2. regime_event - Regime transition events for alerting and backtesting
-- ============================================================================

CREATE TABLE IF NOT EXISTS regime_event (
    id SERIAL PRIMARY KEY,
    asset VARCHAR(10) NOT NULL,
    timescale VARCHAR(10),
    timestamp TIMESTAMPTZ NOT NULL,

    event_type VARCHAR(30) NOT NULL,
    severity DECIMAL(4,3) NOT NULL CHECK (severity BETWEEN 0 AND 1),
    source VARCHAR(30) NOT NULL,
    metadata JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_regime_events
    ON regime_event(asset, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_high_severity_events
    ON regime_event(severity DESC, timestamp DESC)
    WHERE severity > 0.7;

CREATE INDEX IF NOT EXISTS idx_event_type
    ON regime_event(event_type, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_event_source
    ON regime_event(source, timestamp DESC);

COMMENT ON TABLE regime_event IS 'Regime transition events for alerting and backtesting';
COMMENT ON COLUMN regime_event.asset IS 'Asset symbol where event occurred';
COMMENT ON COLUMN regime_event.timescale IS 'Timescale where event was detected (null if multi-scale)';
COMMENT ON COLUMN regime_event.event_type IS 'Values: exit_warning, cascade_detected, basin_collapse, high_disagreement, directional_divergence, position_detaching';
COMMENT ON COLUMN regime_event.severity IS '0=informational, 1=critical (exit positions immediately)';
COMMENT ON COLUMN regime_event.source IS 'Values: agreement, geometry, cascade, position';
COMMENT ON COLUMN regime_event.metadata IS 'JSON payload with event-specific details';

-- ============================================================================
-- Event Type Examples:
-- ============================================================================
--
-- exit_warning: Price approaching basin boundary
--   metadata: {"distance_to_boundary": 0.05, "velocity": -0.02}
--
-- cascade_detected: Disagreement propagating upward through timescales
--   metadata: {"cascade_path": ["1h", "4h", "1d"], "coherence_scores": [0.85, 0.82, 0.79]}
--
-- basin_collapse: Basin width shrinking rapidly
--   metadata: {"basin_width_change": -0.45, "time_periods": 3}
--
-- high_disagreement: Model agreement below threshold
--   metadata: {"agreement_score": 0.32, "disagreement_type": "both"}
--
-- directional_divergence: Models predicting opposing directions
--   metadata: {"divergence": 0.91, "bullish_count": 2, "bearish_count": 3}
--
-- position_detaching: Price no longer tracking basin
--   metadata: {"lag_score": 0.15, "position_state": "detaching"}
-- ============================================================================
