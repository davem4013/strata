"""Read-only FastAPI endpoints for regime state."""
from dataclasses import asdict
from typing import Dict, Optional
import logging

try:
    from fastapi import APIRouter, HTTPException, Query
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "FastAPI is required for strata.state.regime_api; install fastapi to use these endpoints"
    ) from exc

from .regime_buffer import RegimeBuffer

logger = logging.getLogger(__name__)

REGIME_BUFFERS: Dict[str, RegimeBuffer] = {}

router = APIRouter()


def register_regime_buffer(symbol: str, buffer: RegimeBuffer) -> None:
    REGIME_BUFFERS[symbol.upper()] = buffer
    logger.info("Registered regime buffer for %s", symbol.upper())


def get_regime_buffer(symbol: str) -> Optional[RegimeBuffer]:
    return REGIME_BUFFERS.get(symbol.upper())


@router.get("/state/regime/latest")
def get_latest_regime(symbol: str = Query(..., description="Asset symbol")):
    buffer = get_regime_buffer(symbol)
    if buffer is None:
        raise HTTPException(status_code=404, detail="No regime buffer for symbol")
    state = buffer.latest()
    if state is None:
        return None
    return asdict(state)


@router.get("/state/regime/history")
def get_regime_history(
    symbol: str = Query(..., description="Asset symbol"),
    limit: int = Query(200, ge=1, le=5000),
):
    buffer = get_regime_buffer(symbol)
    if buffer is None:
        raise HTTPException(status_code=404, detail="No regime buffer for symbol")
    states = buffer.history(limit)
    return [asdict(state) for state in states]


def attach_regime_routes(app) -> None:
    app.include_router(router)
