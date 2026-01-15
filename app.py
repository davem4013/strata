"""FastAPI bootstrap wiring in-memory STRATA state buffers."""
from fastapi import FastAPI

from strata.config import ASSETS
from strata.state.state_buffer import StrataStateBuffer
from strata.state.regime_buffer import RegimeBuffer
from strata.state import state_api, regime_api


def create_app(symbols=None) -> FastAPI:
    app = FastAPI(title="STRATA State API", version="0.1.0")

    symbols = symbols or ASSETS
    for symbol in symbols:
        # Fresh buffers per symbol; committers should register their own or reuse these
        state_api.register_buffer(symbol, StrataStateBuffer())
        regime_api.register_regime_buffer(symbol, RegimeBuffer())

    state_api.attach_to_app(app)
    regime_api.attach_regime_routes(app)

    return app


app = create_app()
