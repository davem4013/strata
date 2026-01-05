-- ============================================================================
-- STRATA System Schema
-- ============================================================================
-- Tables: system_state
-- Dependencies: PostgreSQL 14+
-- ============================================================================

-- ============================================================================
-- 1. system_state - System configuration and runtime state
-- ============================================================================

CREATE TABLE IF NOT EXISTS system_state (
    id SERIAL PRIMARY KEY,
    key VARCHAR(100) NOT NULL UNIQUE,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_system_state_key
    ON system_state(key);

COMMENT ON TABLE system_state IS 'System configuration and runtime state';
COMMENT ON COLUMN system_state.key IS 'Configuration key (unique identifier)';
COMMENT ON COLUMN system_state.value IS 'JSON value - flexible schema for various config types';
COMMENT ON COLUMN system_state.updated_at IS 'Last update timestamp';

-- ============================================================================
-- Example system_state entries:
-- ============================================================================
--
-- Key: 'assets'
-- Value: {"symbols": ["GLD", "QQQ", "XRT", "TLT", "NVDA"]}
--
-- Key: 'timescales'
-- Value: {"scales": ["1h", "4h", "1d", "1w", "1m"], "active": ["1d", "1w"]}
--
-- Key: 'analysis_parameters'
-- Value: {
--   "residual_lookback_periods": 20,
--   "basin_clustering_window": 100,
--   "boundary_sigma": 2.0
-- }
--
-- Key: 'model_config'
-- Value: {
--   "router": "youtu_router",
--   "heads": ["head_vol", "head_corr", "head_temporal", "head_stability"],
--   "embedding_model": "all-MiniLM-L6-v2"
-- }
--
-- Key: 'thresholds'
-- Value: {
--   "agreement_low": 0.5,
--   "directional_divergence_high": 0.7,
--   "cascade_coherence_min": 0.75,
--   "position_detach_lag": 0.25
-- }
--
-- Key: 'last_run'
-- Value: {
--   "timestamp": "2025-01-05T12:00:00Z",
--   "pipeline": "full_analysis",
--   "status": "success",
--   "duration_seconds": 45.3
-- }
--
-- Key: 'data_health'
-- Value: {
--   "last_market_data_timestamp": "2025-01-05T12:00:00Z",
--   "data_gaps": [],
--   "assets_with_issues": []
-- }
-- ============================================================================

-- Insert default configuration
INSERT INTO system_state (key, value)
VALUES
    ('assets', '{"symbols": ["GLD", "QQQ", "XRT", "TLT", "NVDA"]}'::jsonb),
    ('timescales', '{"scales": ["1h", "4h", "1d", "1w", "1m"], "active": ["1d", "1w"]}'::jsonb),
    ('analysis_parameters', '{"residual_lookback_periods": 20, "basin_clustering_window": 100, "boundary_sigma": 2.0}'::jsonb),
    ('thresholds', '{"agreement_low": 0.5, "directional_divergence_high": 0.7, "cascade_coherence_min": 0.75, "position_detach_lag": 0.25}'::jsonb)
ON CONFLICT (key) DO NOTHING;
