-- ============================================================================
-- STRATA Database Setup
-- ============================================================================
-- Run this file to set up the complete STRATA database schema
-- Usage: psql -U strata_user -d strata -f sql/00_setup.sql
-- ============================================================================

-- Set client encoding
SET client_encoding = 'UTF8';

-- Set timezone
SET timezone = 'UTC';

-- Enable extension
CREATE EXTENSION IF NOT EXISTS vector;

\echo '============================================================================'
\echo 'STRATA Database Setup'
\echo '============================================================================'
\echo ''

\echo 'Loading core schema (market_state, residual_state, basin_geometry, basin_position)...'
\i sql/01_core.sql
\echo 'Core schema loaded.'
\echo ''

\echo 'Loading AI schema (model_interpretation, agreement_metrics)...'
\i sql/02_ai.sql
\echo 'AI schema loaded.'
\echo ''

\echo 'Loading analysis schema (cross_scale_coherence, regime_event)...'
\i sql/03_analysis.sql
\echo 'Analysis schema loaded.'
\echo ''

\echo 'Loading system schema (system_state)...'
\i sql/04_system.sql
\echo 'System schema loaded.'
\echo ''

\echo '============================================================================'
\echo 'STRATA Database Setup Complete'
\echo '============================================================================'
\echo ''
\echo 'Tables created:'
\echo '  - market_state'
\echo '  - residual_state'
\echo '  - basin_geometry'
\echo '  - basin_position'
\echo '  - model_interpretation'
\echo '  - agreement_metrics'
\echo '  - cross_scale_coherence'
\echo '  - regime_event'
\echo '  - system_state'
\echo ''
\echo 'Extensions enabled:'
\echo '  - vector (pgvector)'
\echo ''
