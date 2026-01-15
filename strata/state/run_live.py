"""CLI entrypoint to run live STRATA state committer."""
import argparse
import logging

from .state_buffer import StrataStateBuffer
from .state_committer import StrataStateCommitter

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
    args = parser.parse_args()

    buffer = StrataStateBuffer()

    # Register buffer with FastAPI router if available
    try:
        from . import state_api
        state_api.register_buffer(args.symbol, buffer)
    except Exception:
        logger.debug("FastAPI not available; skipping state API registration")

    committer = StrataStateCommitter(
        analytics_url=args.analytics_url,
        symbol=args.symbol,
        buffer=buffer,
        poll_seconds=args.poll_seconds,
        surface_poll_seconds=args.surface_poll_seconds,
    )

    committer.run()


if __name__ == "__main__":  # pragma: no cover
    main()
