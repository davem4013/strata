"""Live STRATA state committer using in-memory ring buffer."""
from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests

from strata.analysis.basins import cluster_residuals
from strata.analysis.position import calculate_position_state, compute_lag_score
from strata.analysis.residuals import fit_least_squares
from strata.config import BASIN_CLUSTERING_WINDOW, RESIDUAL_LOOKBACK_PERIODS
from .basin_state import ResidualBasin
from .state_buffer import BasinBuffer, StrataStateBuffer
from .state_frame import StateFrame, StateFrameBuffer
from .strata_buffer import append_interpreted_state
from .strata_state import StrataState

logger = logging.getLogger(__name__)


class StrataStateCommitter:
    """
    Poll Analytics API, compute STRATA state, and push into an in-memory buffer.
    """

    def __init__(
        self,
        analytics_url: str,
        symbol: str,
        buffer: StrataStateBuffer,
        frame_buffer: Optional[StateFrameBuffer] = None,
        frame_timescale: float = 0.0,
        poll_seconds: float = 1.0,
        surface_poll_seconds: float = 10.0,
        basin_buffer: Optional[BasinBuffer] = None,
    ):
        self.analytics_url = analytics_url.rstrip("/")
        self.symbol = symbol
        self.buffer = buffer
        self.frame_buffer = frame_buffer
        self.frame_timescale = float(frame_timescale)
        self.poll_seconds = poll_seconds
        self.surface_poll_seconds = surface_poll_seconds
        self.basin_buffer = basin_buffer or BasinBuffer()

        self._last_price: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._last_surface: Optional[Dict] = None
        self._last_surface_ts: Optional[float] = None
        self._last_surface_poll: float = 0.0

    def run(self) -> None:
        """Run commit loop indefinitely."""
        logger.info(
            "Starting StrataStateCommitter for %s (poll=%.2fs, surface_poll=%.2fs)",
            self.symbol,
            self.poll_seconds,
            self.surface_poll_seconds,
        )
        try:
            while True:
                try:
                    self._step()
                except Exception as exc:  # pragma: no cover - resilience path
                    logger.error("Commit step failed: %s", exc, exc_info=True)
                time.sleep(self.poll_seconds)
        except KeyboardInterrupt:
            logger.info("StrataStateCommitter stopped by user")

    def _step(self) -> None:
        now = time.time()

        if now - self._last_surface_poll >= self.surface_poll_seconds:
            surface = self._fetch_surface()
            if surface is not None:
                self._last_surface = surface
                surface_ts = surface.get("timestamp") or surface.get("surface_timestamp")
                self._last_surface_ts = (
                    self._to_timestamp(surface_ts) if surface_ts is not None else self._last_surface_ts
                )
            self._last_surface_poll = now

        price_tick = self._fetch_price()
        if price_tick is None:
            return

        price = price_tick["price"]
        tick_ts = price_tick["timestamp"]

        new_surface_ts = self._last_surface_ts

        if self._is_new_tick(price, tick_ts, new_surface_ts):
            state = self._build_state(price_tick, self._last_surface, new_surface_ts)
            if state:
                self.buffer.append(state)
                append_interpreted_state(state)
                self._push_frame(state, price_tick, self._last_surface)
                self._last_price = price
                self._last_timestamp = tick_ts
                self._last_surface_ts = new_surface_ts
                logger.info(
                    "Committed state %s price=%.4f residual=%.4f center=%.4f width=%.4f buffer=%d",
                    self.symbol,
                    state.spot,
                    state.residual,
                    state.basin_center,
                    state.basin_width,
                    self.buffer.size(),
                )

    def _is_new_tick(self, price: float, timestamp: float, surface_ts: Optional[float]) -> bool:
        if self._last_timestamp is None:
            return True
        if timestamp > self._last_timestamp:
            return True
        if surface_ts is not None and self._last_surface_ts is not None and surface_ts > self._last_surface_ts:
            return True
        if price != self._last_price:
            if timestamp <= self._last_timestamp:
                logger.warning(
                    "Non-monotonic tick for %s (ts=%s last=%s)",
                    self.symbol,
                    timestamp,
                    self._last_timestamp,
                )
            return True
        return False

    def _fetch_price(self) -> Optional[Dict]:
        try:
            resp = requests.get(
                f"{self.analytics_url}/get_last_price",
                params={"symbol": self.symbol},
                timeout=3,
            )
            resp.raise_for_status()
            data = resp.json()
            price = data.get("price") or data.get("spot")
            if price is None:
                logger.warning("Price response missing price field: %s", data)
                return None
            ts_raw = data.get("timestamp") or data.get("ts") or time.time()
            return {
                "price": float(price),
                "timestamp": self._to_timestamp(ts_raw),
            }
        except requests.RequestException as exc:
            logger.warning("Price poll failed for %s: %s", self.symbol, exc)
            return None
        except Exception as exc:
            logger.error("Unexpected price response error: %s", exc)
            return None

    def _fetch_surface(self) -> Optional[Dict]:
        try:
            resp = requests.get(
                f"{self.analytics_url}/state/surface/latest",
                params={"symbol": self.symbol},
                timeout=5,
            )
            if not resp.ok:
                logger.debug("Surface poll not ok (%s): %s", resp.status_code, resp.text)
                return None
            return resp.json()
        except requests.RequestException as exc:
            logger.debug("Surface poll failed for %s: %s", self.symbol, exc)
            return None

    def _build_state(
        self,
        price_tick: Dict,
        surface_snapshot: Optional[Dict],
        surface_ts: Optional[float],
    ) -> Optional[StrataState]:
        price = float(price_tick["price"])
        timestamp = float(price_tick["timestamp"])

        surface_ts, atm_iv = self._extract_surface(surface_snapshot, price, surface_ts)

        residual, normalized_residual = self._compute_residual(price, timestamp)
        basin = self._compute_basin(normalized_residual)
        basin_center = basin.center
        basin_width = basin.width
        basin_velocity = self._compute_basin_velocity(basin_center, timestamp)
        price_velocity = self._compute_price_velocity(normalized_residual, timestamp)

        half_width = basin_width / 2.0 if basin_width is not None else 0.0
        distance_to_center = abs(normalized_residual - basin_center)
        if half_width > 0:
            upper = basin_center + half_width
            lower = basin_center - half_width
            distance_to_boundary = min(
                abs(normalized_residual - upper),
                abs(normalized_residual - lower),
            )
        else:
            distance_to_boundary = 0.0

        if half_width > 0:
            normalized_distance = min(distance_to_center / half_width, 1.5)
        else:
            normalized_distance = 0.0

        lag_score = compute_lag_score(basin_velocity, price_velocity)
        position_state = calculate_position_state(
            normalized_distance, abs(distance_to_boundary), lag_score
        )
        risk_score = self._compute_risk(
            position_state,
            normalized_distance,
            lag_score,
            basin_velocity,
            price_velocity,
        )

        return StrataState(
            timestamp=timestamp,
            symbol=self.symbol,
            spot=price,
            surface_timestamp=surface_ts,
            atm_iv=atm_iv,
            residual=residual,
            normalized_residual=normalized_residual,
            basin_center=basin_center,
            basin_width=basin_width,
            basin_velocity=basin_velocity,
            position_state=position_state,
            normalized_distance=normalized_distance,
            risk_score=risk_score,
        )

    def _compute_residual(self, price: float, timestamp: float) -> Tuple[float, float]:
        prices, times = self._regression_window(price, timestamp)

        if len(prices) < 2:
            return 0.0, 0.0

        time_array = np.array(times, dtype=float)
        time_array = time_array - time_array[0]
        time_array = time_array.reshape(-1, 1)

        fair_value, residual, normalized_residual, _, _ = fit_least_squares(
            np.array(prices, dtype=float),
            time_array,
        )

        return float(residual), float(normalized_residual)

    def _regression_window(self, price: float, timestamp: float) -> Tuple[list, list]:
        """
        Build regression series using rolling state frames if available.

        Falls back to StrataState history when no frame buffer is configured.
        """
        if self.frame_buffer is not None:
            window = self.frame_buffer.get_window(RESIDUAL_LOOKBACK_PERIODS - 1)
            prices = [frame.price for frame in window] + [price]
            times = [frame.timestamp for frame in window] + [timestamp]
        else:
            history = self.buffer.history(RESIDUAL_LOOKBACK_PERIODS - 1)
            prices = [state.spot for state in history] + [price]
            times = [state.timestamp for state in history] + [timestamp]

        return prices, times

    def _compute_basin(self, normalized_residual: float) -> ResidualBasin:
        residuals = [s.normalized_residual for s in self.buffer.history(BASIN_CLUSTERING_WINDOW - 1)]
        residuals.append(normalized_residual)

        if len(residuals) < 2:
            basin = ResidualBasin(
                center=float(normalized_residual),
                width=1.0,
                boundary_upper=float(normalized_residual) + 0.5,
                boundary_lower=float(normalized_residual) - 0.5,
                curvature=0.5,
                sample_count=len(residuals),
                variance=0.0,
            )
            self.basin_buffer.push(basin)
            return basin

        residual_array = np.array(residuals[-BASIN_CLUSTERING_WINDOW:], dtype=float)

        try:
            basin = cluster_residuals(residual_array)
        except Exception as exc:
            logger.debug("Basin clustering fallback for %s: %s", self.symbol, exc)
            center = float(np.mean(residual_array))
            std = float(np.std(residual_array, ddof=1)) if len(residual_array) > 1 else 0.0
            width = max(4.0 * std, 0.5)
            basin = ResidualBasin(
                center=center,
                width=width,
                boundary_upper=center + width / 2.0,
                boundary_lower=center - width / 2.0,
                curvature=0.5,
                sample_count=len(residual_array),
                variance=float(std * std),
            )

        # Keep downstream calculations safe and deterministic even if width is degenerate.
        if basin.width <= 0:
            safe_width = max(basin.width, 1e-6)
            basin = ResidualBasin(
                center=basin.center,
                width=safe_width,
                boundary_upper=basin.center + safe_width / 2.0,
                boundary_lower=basin.center - safe_width / 2.0,
                curvature=basin.curvature,
                sample_count=basin.sample_count,
                variance=basin.variance,
            )

        self.basin_buffer.push(basin)
        return basin

    def _compute_basin_velocity(self, center: float, timestamp: float) -> float:
        history = self.buffer.history(10)
        if not history:
            return 0.0

        centers = [s.basin_center for s in history] + [center]
        times = [s.timestamp for s in history] + [timestamp]

        if len(centers) < 2:
            return 0.0

        time_elapsed_hours = (times[-1] - times[0]) / 3600.0
        if time_elapsed_hours <= 1e-6:
            return 0.0

        return float((centers[-1] - centers[0]) / time_elapsed_hours)

    def _compute_price_velocity(
        self, normalized_residual: float, timestamp: float, lookback: int = 5
    ) -> float:
        history = self.buffer.history(lookback)
        if not history:
            return 0.0

        residuals = [s.normalized_residual for s in history] + [normalized_residual]
        times = [s.timestamp for s in history] + [timestamp]

        if len(residuals) < 2:
            return 0.0

        time_elapsed_hours = (times[-1] - times[0]) / 3600.0
        if time_elapsed_hours <= 1e-6:
            return 0.0

        return float((residuals[-1] - residuals[0]) / time_elapsed_hours)

    def _compute_risk(
        self,
        position_state: str,
        normalized_distance: float,
        lag_score: float,
        basin_velocity: float,
        price_velocity: float,
    ) -> float:
        boundary_risk = normalized_distance
        tracking_risk = max(0.0, 1.0 - lag_score)
        velocity_away = (basin_velocity + price_velocity) / 2.0
        velocity_risk = min(max(abs(velocity_away) * 10.0, 0.0), 1.0)

        overall = (
            0.4 * boundary_risk + 0.3 * tracking_risk + 0.3 * velocity_risk
        )

        state_risk_map = {
            "centered": 0.1,
            "tracking": 0.3,
            "edge": 0.6,
            "lagging": 0.7,
            "detaching": 0.95,
        }
        overall = max(overall, state_risk_map.get(position_state, 0.5))
        return float(max(0.0, min(overall, 1.0)))

    def _extract_surface(
        self, surface: Optional[Dict], spot: float, existing_ts: Optional[float]
    ) -> Tuple[Optional[float], Optional[float]]:
        if not surface:
            return existing_ts, None

        ts_raw = surface.get("timestamp") or surface.get("surface_timestamp")
        surface_ts = self._to_timestamp(ts_raw) if ts_raw else existing_ts

        atm_iv = surface.get("atm_iv")
        if atm_iv is not None:
            try:
                return surface_ts, float(atm_iv)
            except (TypeError, ValueError):
                atm_iv = None

        iv_surface = surface.get("iv_surface")
        if iv_surface:
            atm_iv = self._extract_atm_iv_from_surface(iv_surface, spot)

        return surface_ts, atm_iv

    def _extract_atm_iv_from_surface(self, iv_surface, spot: float) -> Optional[float]:
        try:
            if isinstance(iv_surface, dict):
                expiries = sorted(iv_surface.keys())
                if not expiries:
                    return None
                front_expiry = expiries[0]
                strikes = iv_surface.get(front_expiry, {})
                if isinstance(strikes, dict) and strikes:
                    closest = min(
                        strikes.keys(),
                        key=lambda strike: abs(float(strike) - float(spot)),
                    )
                    return float(strikes[closest])
            elif isinstance(iv_surface, list) and iv_surface:
                atm_row = iv_surface[0]
                if atm_row:
                    mid_idx = len(atm_row) // 2
                    return float(atm_row[mid_idx])
        except Exception as exc:
            logger.debug("ATM IV extraction failed: %s", exc)
        return None

    def _push_frame(
        self,
        state: StrataState,
        price_tick: Dict[str, Any],
        surface_snapshot: Optional[Dict[str, Any]],
    ) -> None:
        """Build and push a StateFrame into the frame buffer if configured."""
        if self.frame_buffer is None:
            return

        frame = self._build_state_frame(state, price_tick, surface_snapshot)
        if frame:
            self.frame_buffer.push(frame)

    def _build_state_frame(
        self,
        state: StrataState,
        price_tick: Dict[str, Any],
        surface_snapshot: Optional[Dict[str, Any]],
    ) -> Optional[StateFrame]:
        surface = surface_snapshot or {}

        def _as_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        sd_bands = surface.get("sd_bands")
        sd_upper = _as_float(sd_bands.get("upper"), 0.0) if isinstance(sd_bands, dict) else 0.0
        sd_lower = _as_float(sd_bands.get("lower"), 0.0) if isinstance(sd_bands, dict) else 0.0

        return StateFrame(
            timestamp=state.timestamp,
            asset=self.symbol,
            timescale=self.frame_timescale,
            price=state.spot,
            atm_iv=_as_float(state.atm_iv, 0.0) if state.atm_iv is not None else 0.0,
            iv_25d_call=_as_float(surface.get("iv_25d_call"), 0.0),
            iv_25d_put=_as_float(surface.get("iv_25d_put"), 0.0),
            iv_slope=_as_float(surface.get("iv_slope"), 0.0),
            sd_band_upper=sd_upper,
            sd_band_lower=sd_lower,
            delta_pressure=_as_float(surface.get("delta_pressure"), 0.0),
            gamma_exposure=_as_float(surface.get("gamma_exposure"), 0.0),
            vanna=_as_float(surface.get("vanna"), 0.0),
            vol_of_vol=_as_float(surface.get("vol_of_vol"), 0.0),
        )

    @staticmethod
    def _to_timestamp(value) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        try:
            from datetime import datetime

            return datetime.fromisoformat(value).timestamp()
        except Exception:
            return time.time()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live STRATA state committer")
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
