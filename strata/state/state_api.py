"""Read-only STRATA state APIs (pure Python and FastAPI adapters)."""
from dataclasses import asdict
from typing import Dict, List, Optional
import logging

try:
    from fastapi import APIRouter, HTTPException, Query
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "FastAPI is required for strata.state.state_api; install fastapi to use these endpoints"
    ) from exc

from .state_buffer import StrataStateBuffer
from . import strata_buffer as sb
from .strata_buffer import BasinFrame

logger = logging.getLogger(__name__)

STATE_BUFFERS: Dict[str, StrataStateBuffer] = {}

router = APIRouter()


# -------------------------
# Pure Python snapshot API
# -------------------------
def get_latest_basin_frame() -> Optional[BasinFrame]:
    """Return the most recent interpreted basin frame (or None)."""
    return sb.STRATA_BUFFER.latest()


def get_basin_history(n: int = 100) -> List[BasinFrame]:
    """Return up to the last n basin frames (oldest → newest)."""
    if n <= 0:
        return []
    return sb.STRATA_BUFFER.history(n)


# -------------------------
# FastAPI adapter (legacy)
# -------------------------
def register_buffer(symbol: str, buffer: StrataStateBuffer) -> None:
    """Register a buffer for a symbol (uppercased)."""
    STATE_BUFFERS[symbol.upper()] = buffer
    logger.info("Registered state buffer for %s", symbol.upper())


def get_buffer(symbol: str) -> Optional[StrataStateBuffer]:
    """Return registered buffer for symbol."""
    return STATE_BUFFERS.get(symbol.upper())


@router.get("/state/strata/latest")
def get_latest_state(symbol: str = Query(..., description="Asset symbol")):
    buffer = get_buffer(symbol)
    if buffer is None:
        raise HTTPException(status_code=404, detail="No state buffer for symbol")
    state = buffer.latest()
    if state is None or buffer.is_empty():
        raise HTTPException(status_code=404, detail="No state available")
    return asdict(state)


@router.get("/state/strata/history")
def get_state_history(
    symbol: str = Query(..., description="Asset symbol"),
    limit: int = Query(500, ge=1, le=5000),
):
    buffer = get_buffer(symbol)
    if buffer is None:
        raise HTTPException(status_code=404, detail="No state buffer for symbol")
    if buffer.is_empty():
        raise HTTPException(status_code=404, detail="No state available")
    states = buffer.history(limit)
    return [asdict(state) for state in states]


@router.get("/state/strata/status")
def get_state_status(symbol: str = Query(..., description="Asset symbol")):
    buffer = get_buffer(symbol)
    if buffer is None:
        raise HTTPException(status_code=404, detail="No state buffer for symbol")
    if buffer.is_empty():
        raise HTTPException(status_code=404, detail="No state available")

    latest = buffer.latest()
    last_ts = latest.timestamp if latest else None
    age = None
    if last_ts is not None:
        import time
        age = max(time.time() - float(last_ts), 0.0)

    return {
        "symbol": symbol,
        "size": buffer.size(),
        "max_size": buffer.maxlen,
        "last_timestamp": last_ts,
        "age_seconds": age,
    }


def attach_to_app(app) -> None:
    """Include state routes on an existing FastAPI app."""
    app.include_router(router)
