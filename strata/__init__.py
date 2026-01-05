"""
STRATA: State-Tracking Regime Analysis Through AI

A state-based market regime detection system that uses disagreement between
multiple AI models as a primary signal for regime uncertainty. The system treats
markets as dynamical systems with moving attractor basins, detecting phase
transitions before they complete.
"""

__version__ = '0.1.0'

# Make key imports available at package level
from strata.config import (
    ASSETS,
    TIMESCALES,
    ACTIVE_TIMESCALES,
    DB_CONFIG,
    IBKR_CONFIG
)

__all__ = [
    '__version__',
    'ASSETS',
    'TIMESCALES',
    'ACTIVE_TIMESCALES',
    'DB_CONFIG',
    'IBKR_CONFIG'
]
