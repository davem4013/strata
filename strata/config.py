"""STRATA system configuration loaded from environment variables."""
import os
from typing import List
from decimal import Decimal
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

# ============================================================================
# Database Configuration
# ============================================================================

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': os.getenv('DB_NAME', 'strata'),
    'user': os.getenv('DB_USER', 'strata_user'),
    'password': os.getenv('DB_PASSWORD', '9928')
}

DB_MIN_CONNECTIONS = int(os.getenv('DB_MIN_CONNECTIONS', '1'))
DB_MAX_CONNECTIONS = int(os.getenv('DB_MAX_CONNECTIONS', '10'))

# ============================================================================
# IBKR API Configuration
# ============================================================================

IBKR_CONFIG = {
    'host': os.getenv('IBKR_HOST', '127.0.0.1'),
    'port': int(os.getenv('IBKR_PORT', '7497')),
    'client_id': int(os.getenv('IBKR_CLIENT_ID', '1'))
}

IBKR_PAPER_TRADING = bool(int(os.getenv('IBKR_PAPER_TRADING', '1')))

# ============================================================================
# Asset Configuration
# ============================================================================

ASSETS: List[str] = os.getenv('ASSETS', 'GLD,QQQ,XRT,TLT,NVDA').split(',')
TIMESCALES: List[str] = os.getenv('TIMESCALES', '1h,4h,1d,1w,1m').split(',')
ACTIVE_TIMESCALES: List[str] = os.getenv('ACTIVE_TIMESCALES', '1d,1w').split(',')

# ============================================================================
# Analysis Parameters
# ============================================================================

RESIDUAL_LOOKBACK_PERIODS = int(os.getenv('RESIDUAL_LOOKBACK_PERIODS', '20'))
BASIN_CLUSTERING_WINDOW = int(os.getenv('BASIN_CLUSTERING_WINDOW', '100'))
BOUNDARY_SIGMA = float(os.getenv('BOUNDARY_SIGMA', '2.0'))

# ============================================================================
# Agreement & Disagreement Thresholds
# ============================================================================

AGREEMENT_THRESHOLD_LOW = Decimal(os.getenv('AGREEMENT_THRESHOLD_LOW', '0.5'))
DIRECTIONAL_DIVERGENCE_HIGH = Decimal(os.getenv('DIRECTIONAL_DIVERGENCE_HIGH', '0.7'))
CASCADE_COHERENCE_MIN = Decimal(os.getenv('CASCADE_COHERENCE_MIN', '0.75'))
POSITION_DETACH_LAG = Decimal(os.getenv('POSITION_DETACH_LAG', '0.25'))

# ============================================================================
# AI Model Configuration
# ============================================================================

MODEL_ROUTER = os.getenv('MODEL_ROUTER', 'youtu_router')
MODEL_HEADS: List[str] = os.getenv(
    'MODEL_HEADS',
    'head_vol,head_corr,head_temporal,head_stability'
).split(',')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

# Combined model list
MODELS = {
    'router': MODEL_ROUTER,
    'heads': MODEL_HEADS
}

# Optional model API configuration
MODEL_API_URL = os.getenv('MODEL_API_URL', '')
MODEL_API_KEY = os.getenv('MODEL_API_KEY', '')

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_FILE = os.getenv('LOG_FILE', '')
LOG_FORMAT = os.getenv('LOG_FORMAT', 'detailed')

# ============================================================================
# Pipeline Configuration
# ============================================================================

PIPELINE_INTERVAL = int(os.getenv('PIPELINE_INTERVAL', '3600'))
PIPELINE_CONTINUOUS = bool(int(os.getenv('PIPELINE_CONTINUOUS', '0')))
PIPELINE_BACKFILL = bool(int(os.getenv('PIPELINE_BACKFILL', '0')))
PIPELINE_BACKFILL_PERIODS = int(os.getenv('PIPELINE_BACKFILL_PERIODS', '100'))

# ============================================================================
# Alert Configuration
# ============================================================================

ALERTS_EMAIL_ENABLED = bool(int(os.getenv('ALERTS_EMAIL_ENABLED', '0')))
ALERTS_EMAIL_CONFIG = {
    'from': os.getenv('ALERTS_EMAIL_FROM', 'strata@example.com'),
    'to': os.getenv('ALERTS_EMAIL_TO', 'trader@example.com'),
    'smtp_host': os.getenv('ALERTS_EMAIL_SMTP_HOST', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('ALERTS_EMAIL_SMTP_PORT', '587')),
    'smtp_user': os.getenv('ALERTS_EMAIL_SMTP_USER', ''),
    'smtp_password': os.getenv('ALERTS_EMAIL_SMTP_PASSWORD', '')
}
ALERTS_SEVERITY_THRESHOLD = Decimal(os.getenv('ALERTS_SEVERITY_THRESHOLD', '0.7'))

# ============================================================================
# Development & Testing
# ============================================================================

MOCK_IBKR_DATA = bool(int(os.getenv('MOCK_IBKR_DATA', '0')))
MOCK_DATA_SEED = int(os.getenv('MOCK_DATA_SEED', '42'))
DEBUG_MODE = bool(int(os.getenv('DEBUG_MODE', '0')))

# ============================================================================
# Validation
# ============================================================================

def validate_config() -> None:
    """Validate configuration values."""
    errors = []

    # Validate timescales
    if not set(ACTIVE_TIMESCALES).issubset(set(TIMESCALES)):
        errors.append(f"ACTIVE_TIMESCALES must be subset of TIMESCALES")

    # Validate thresholds
    if not (0 <= AGREEMENT_THRESHOLD_LOW <= 1):
        errors.append(f"AGREEMENT_THRESHOLD_LOW must be between 0 and 1")

    if not (0 <= DIRECTIONAL_DIVERGENCE_HIGH <= 1):
        errors.append(f"DIRECTIONAL_DIVERGENCE_HIGH must be between 0 and 1")

    if not (0 <= CASCADE_COHERENCE_MIN <= 1):
        errors.append(f"CASCADE_COHERENCE_MIN must be between 0 and 1")

    if not (0 <= POSITION_DETACH_LAG <= 1):
        errors.append(f"POSITION_DETACH_LAG must be between 0 and 1")

    # Validate analysis parameters
    if RESIDUAL_LOOKBACK_PERIODS < 2:
        errors.append(f"RESIDUAL_LOOKBACK_PERIODS must be at least 2")

    if BASIN_CLUSTERING_WINDOW < RESIDUAL_LOOKBACK_PERIODS:
        errors.append(f"BASIN_CLUSTERING_WINDOW must be >= RESIDUAL_LOOKBACK_PERIODS")

    if BOUNDARY_SIGMA <= 0:
        errors.append(f"BOUNDARY_SIGMA must be positive")

    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging() -> None:
    """Configure logging based on config settings."""
    import logging
    import sys

    # Map log level string to logging constant
    level = getattr(logging, LOG_LEVEL, logging.INFO)

    # Configure format
    if LOG_FORMAT == 'json':
        # JSON format for structured logging
        format_string = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
    elif LOG_FORMAT == 'detailed':
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    else:  # simple
        format_string = '%(levelname)s: %(message)s'

    # Configure handlers
    handlers = []

    # Always log to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(format_string))
    handlers.append(stdout_handler)

    # Optionally log to file
    if LOG_FILE:
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )

    # Set specific loggers
    logging.getLogger('strata').setLevel(level)

    # Reduce noise from third-party libraries
    logging.getLogger('ib_insync').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


# ============================================================================
# Initialization
# ============================================================================

# Validate config on import
validate_config()

# Setup logging
setup_logging()
