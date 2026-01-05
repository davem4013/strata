-- ============================================================================
-- STRATA Core Schema
-- ============================================================================
-- Tables: market_state, residual_state, basin_geometry, basin_position
-- Dependencies: PostgreSQL 14+, pgvector extension
-- ============================================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- 1. market_state - Raw market data from IBKR API
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_state (
    id SERIAL PRIMARY KEY,
    asset VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    timescale VARCHAR(10) NOT NULL,
    price DECIMAL(12,4) NOT NULL,
    implied_vol DECIMAL(8,6),
    skew DECIMAL(8,6),
    volume BIGINT,
    bid_ask_spread DECIMAL(8,6),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(asset, timestamp, timescale)
);






COMMENT ON TABLE market_state IS 'Raw market observables from IBKR API';
COMMENT ON COLUMN market_state.asset IS 'Asset symbol (GLD, QQQ, XRT, TLT, NVDA)';
COMMENT ON COLUMN market_state.timescale IS 'Values: 1h, 4h, 1d, 1w, 1m';
COMMENT ON COLUMN market_state.price IS 'Last traded price';
COMMENT ON COLUMN market_state.implied_vol IS 'ATM implied volatility from options';
COMMENT ON COLUMN market_state.skew IS 'Volatility skew (OTM put vol - ATM vol)';
COMMENT ON COLUMN market_state.volume IS 'Trading volume for the period';
COMMENT ON COLUMN market_state.bid_ask_spread IS 'Current bid-ask spread';

-- ============================================================================
-- 2. residual_state - Deviations from fair value (least squares)
-- ============================================================================

CREATE TABLE IF NOT EXISTS residual_state (
    id SERIAL PRIMARY KEY,
    asset VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    timescale VARCHAR(10) NOT NULL,
    price DECIMAL(12,4) NOT NULL,
    fair_value DECIMAL(12,4) NOT NULL,
    residual DECIMAL(12,6) NOT NULL,
    normalized_residual DECIMAL(8,4) NOT NULL,
    std_error DECIMAL(8,6),
    r_squared DECIMAL(6,4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(asset, timestamp, timescale)
);

CREATE INDEX IF NOT EXISTS idx_residual_lookup
    ON residual_state(asset, timescale, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_residual_normalized
    ON residual_state(asset, timescale, normalized_residual);

COMMENT ON TABLE residual_state IS 'Least squares residuals - distance from equilibrium';
COMMENT ON COLUMN residual_state.fair_value IS 'Least squares fitted value from rolling regression';
COMMENT ON COLUMN residual_state.residual IS 'Price - fair_value (raw deviation)';
COMMENT ON COLUMN residual_state.normalized_residual IS 'Residual / std_error (dimensionless)';
COMMENT ON COLUMN residual_state.std_error IS 'Standard error of regression residuals';
COMMENT ON COLUMN residual_state.r_squared IS 'Goodness of fit for regression model';

-- ============================================================================
-- 3. basin_geometry - Attractor basin structure
-- ============================================================================

CREATE TABLE IF NOT EXISTS basin_geometry (
    id SERIAL PRIMARY KEY,
    basin_id VARCHAR(50) NOT NULL,
    asset VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    timescale VARCHAR(10) NOT NULL,

    -- Phase 1: 1D geometry in residual space
    center_location DECIMAL(12,6) NOT NULL,
    center_velocity DECIMAL(8,6),
    boundary_upper DECIMAL(12,6) NOT NULL,
    boundary_lower DECIMAL(12,6) NOT NULL,
    basin_width DECIMAL(8,4) NOT NULL,
    curvature DECIMAL(8,4),
    sample_count INT,

    -- Phase 2: Multi-dimensional geometry (nullable, future-proof)
    center_vector vector(8),
    covariance_matrix JSONB,
    principal_axes JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(basin_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_basin_lookup
    ON basin_geometry(basin_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_basin_asset
    ON basin_geometry(asset, timescale, timestamp DESC);

COMMENT ON TABLE basin_geometry IS 'Attractor basin structure in residual space';
COMMENT ON COLUMN basin_geometry.basin_id IS 'Format: {asset}_{timescale}_{epoch}';
COMMENT ON COLUMN basin_geometry.center_location IS 'Mean of residual cluster (basin center)';
COMMENT ON COLUMN basin_geometry.center_velocity IS 'Rate of basin center drift (d/dt center_location)';
COMMENT ON COLUMN basin_geometry.boundary_upper IS 'Upper basin boundary (+N sigma from center)';
COMMENT ON COLUMN basin_geometry.boundary_lower IS 'Lower basin boundary (-N sigma from center)';
COMMENT ON COLUMN basin_geometry.basin_width IS 'Distance between upper and lower boundaries';
COMMENT ON COLUMN basin_geometry.curvature IS 'Second derivative of residual density (basin stiffness)';
COMMENT ON COLUMN basin_geometry.sample_count IS 'Number of residuals used to define basin';

-- ============================================================================
-- 4. basin_position - Price position relative to basin
-- ============================================================================

CREATE TABLE IF NOT EXISTS basin_position (
    id SERIAL PRIMARY KEY,
    basin_id VARCHAR(50) NOT NULL,
    asset VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    timescale VARCHAR(10) NOT NULL,

    distance_to_center DECIMAL(8,4) NOT NULL,
    distance_to_boundary DECIMAL(8,4) NOT NULL,
    normalized_distance DECIMAL(8,4) CHECK (normalized_distance BETWEEN 0 AND 1),

    basin_velocity DECIMAL(8,6),
    price_velocity DECIMAL(8,6),
    lag_score DECIMAL(4,3) CHECK (lag_score BETWEEN 0 AND 1),

    position_state VARCHAR(20),

    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(basin_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_basin_position_lookup
    ON basin_position(asset, timescale, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_position_state_alerts
    ON basin_position(asset, position_state, timestamp DESC)
    WHERE position_state IN ('edge', 'lagging', 'detaching');

COMMENT ON TABLE basin_position IS 'Price position relative to basin - critical for trading decisions';
COMMENT ON COLUMN basin_position.distance_to_center IS 'Absolute distance: |residual - basin_center|';
COMMENT ON COLUMN basin_position.distance_to_boundary IS 'Distance to nearest boundary';
COMMENT ON COLUMN basin_position.normalized_distance IS '0=center, 1=boundary (position within basin)';
COMMENT ON COLUMN basin_position.basin_velocity IS 'Rate of basin center movement';
COMMENT ON COLUMN basin_position.price_velocity IS 'Rate of price movement in residual space';
COMMENT ON COLUMN basin_position.lag_score IS '1=perfectly tracking basin, 0=completely detached';
COMMENT ON COLUMN basin_position.position_state IS 'Values: centered, tracking, edge, lagging, detaching';
