"""CLI entrypoint to run live STRATA state + regime committers."""
import argparse
import logging
import threading
import time

from .state_buffer import StrataStateBuffer
from .state_committer import StrataStateCommitter
from .regime_buffer import RegimeBuffer
from .regime_committer import RegimeCommitter

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live STRATA state buffer")
    parser.add_argument(
        "--symbol",
        default="GLD",
        help="Asset symbol to track",
    )
    parser.add_argument(
        "--analytics-url",
        default="http://127.0.0.1:8000",
        help="Base URL for Analytics API",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="Tick polling interval seconds",
    )
    parser.add_argument(
        "--surface-poll-seconds",
        type=float,
        default=10.0,
        help="Surface polling interval seconds",
    )
    parser.add_argument(
        "--regime",
        action="store_true",
        help="Enable regime tracking from in-memory state",
    )
    parser.add_argument(
        "--regime-poll-seconds",
        type=float,
        default=1.0,
        help="Regime commit interval seconds",
    )
    args = parser.parse_args()

    state_buffer = StrataStateBuffer()
    regime_buffer = RegimeBuffer()

    # Register buffers with FastAPI router if available
    try:
        from . import state_api, regime_api

        state_api.register_buffer(args.symbol, state_buffer)
        regime_api.register_regime_buffer(args.symbol, regime_buffer)
    except Exception:
        logger.debug("FastAPI not available; skipping state/regime API registration")

    state_committer = StrataStateCommitter(
        analytics_url=args.analytics_url,
        symbol=args.symbol,
        buffer=state_buffer,
        poll_seconds=args.poll_seconds,
        surface_poll_seconds=args.surface_poll_seconds,
    )

    stop_event = threading.Event()

    def run_regime_loop():
        regime_committer = RegimeCommitter(
            symbol=args.symbol,
            state_buffer=state_buffer,
            regime_buffer=regime_buffer,
            window=20,
        )
        while not stop_event.is_set():
            regime_committer.step()
            time.sleep(args.regime_poll_seconds)

    regime_thread = None
    if args.regime:
        regime_thread = threading.Thread(target=run_regime_loop, daemon=True)
        regime_thread.start()

    try:
        state_committer.run()
    except KeyboardInterrupt:
        logger.info("Shutting down live state engine...")
        stop_event.set()
        if regime_thread:
            regime_thread.join(timeout=2.0)


if __name__ == "__main__":  # pragma: no cover
    main()
