-- ============================================================================
-- STRATA AI Schema
-- ============================================================================
-- Tables: model_interpretation, agreement_metrics
-- Dependencies: PostgreSQL 14+, pgvector extension
-- ============================================================================

-- ============================================================================
-- 1. model_interpretation - AI model assessments of basin state
-- ============================================================================

CREATE TABLE IF NOT EXISTS model_interpretation (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) NOT NULL,
    basin_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    timescale VARCHAR(10) NOT NULL,

    regime_type VARCHAR(50),
    center_estimate DECIMAL(12,6),
    boundary_upper_estimate DECIMAL(8,4),
    boundary_lower_estimate DECIMAL(8,4),
    stability_score DECIMAL(4,3) CHECK (stability_score BETWEEN 0 AND 1),
    confidence DECIMAL(4,3) CHECK (confidence BETWEEN 0 AND 1),

    interpretation_text TEXT,
    embedding vector(384),

    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(model_id, basin_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_model_interp
    ON model_interpretation(basin_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_model_id
    ON model_interpretation(model_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_embedding_similarity
    ON model_interpretation
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

COMMENT ON TABLE model_interpretation IS 'Youtu-LLM model interpretations of basin state';
COMMENT ON COLUMN model_interpretation.model_id IS 'Values: youtu_router, head_vol, head_corr, head_temporal, head_stability';
COMMENT ON COLUMN model_interpretation.regime_type IS 'Values: stable, transitional, bifurcating, collapsing';
COMMENT ON COLUMN model_interpretation.center_estimate IS 'Model''s estimate of basin center location';
COMMENT ON COLUMN model_interpretation.boundary_upper_estimate IS 'Model''s estimate of upper boundary';
COMMENT ON COLUMN model_interpretation.boundary_lower_estimate IS 'Model''s estimate of lower boundary';
COMMENT ON COLUMN model_interpretation.stability_score IS '1=highly stable, 0=collapsing/transitioning';
COMMENT ON COLUMN model_interpretation.confidence IS 'Model confidence in its assessment';
COMMENT ON COLUMN model_interpretation.interpretation_text IS 'Free-form textual interpretation for semantic comparison';
COMMENT ON COLUMN model_interpretation.embedding IS 'Sentence embedding of interpretation_text for semantic distance';

-- ============================================================================
-- 2. agreement_metrics - Model disagreement as regime uncertainty signal
-- ============================================================================

CREATE TABLE IF NOT EXISTS agreement_metrics (
    id SERIAL PRIMARY KEY,
    basin_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    timescale VARCHAR(10) NOT NULL,

    agreement_score DECIMAL(4,3) NOT NULL CHECK (agreement_score BETWEEN 0 AND 1),
    disagreement_type VARCHAR(20),

    semantic_distance DECIMAL(8,4),
    variance_center DECIMAL(8,6),
    variance_boundary DECIMAL(8,6),
    directional_divergence DECIMAL(4,3) CHECK (directional_divergence BETWEEN 0 AND 1),

    model_count INT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(basin_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_agreement
    ON agreement_metrics(basin_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_dangerous_disagreement
    ON agreement_metrics(timescale, agreement_score, directional_divergence)
    WHERE agreement_score < 0.5 AND directional_divergence > 0.7;

COMMENT ON TABLE agreement_metrics IS 'Model disagreement as regime uncertainty signal';
COMMENT ON COLUMN agreement_metrics.agreement_score IS '1=perfect agreement, 0=complete disagreement';
COMMENT ON COLUMN agreement_metrics.disagreement_type IS 'Values: center, boundary, both, chaotic';
COMMENT ON COLUMN agreement_metrics.semantic_distance IS 'Average pairwise cosine distance of embeddings';
COMMENT ON COLUMN agreement_metrics.variance_center IS 'Variance of center_estimate across models';
COMMENT ON COLUMN agreement_metrics.variance_boundary IS 'Variance of boundary estimates across models';
COMMENT ON COLUMN agreement_metrics.directional_divergence IS '0=same direction, 1=opposing directions (critical!)';
COMMENT ON COLUMN agreement_metrics.model_count IS 'Number of models that provided interpretations';
