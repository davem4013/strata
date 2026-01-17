"""Normalized, numeric state frame ready for buffering and regression."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .state_buffer import StrataStateBuffer


@dataclass(frozen=True)
class StateFrame:
    """
    Lightweight container for a single state observation.

    Values are numeric-only to stay friendly to numpy and regression workflows.
    """

    timestamp: float  # epoch seconds
    asset: str  # symbol or identifier
    timescale: float  # seconds represented by this frame
    price: float  # reference price used for derived measures

    atm_iv: float  # at-the-money implied volatility
    iv_25d_call: float  # 25-delta call IV
    iv_25d_put: float  # 25-delta put IV
    iv_slope: float  # skew measure

    sd_band_upper: float  # upper standard deviation band around price
    sd_band_lower: float  # lower standard deviation band around price
    delta_pressure: float  # directional flow proxy
    gamma_exposure: float  # aggregated gamma exposure
    vanna: float  # slope of delta with respect to IV
    vol_of_vol: float  # variability of IV surface

    def feature_names(self) -> List[str]:
        """Ordered feature names matching feature_values order."""
        return [
            "timestamp",
            "timescale",
            "price",
            "atm_iv",
            "iv_25d_call",
            "iv_25d_put",
            "iv_slope",
            "sd_band_upper",
            "sd_band_lower",
            "delta_pressure",
            "gamma_exposure",
            "vanna",
            "vol_of_vol",
        ]

    def feature_values(self) -> List[float]:
        """Numeric feature vector (asset excluded for encoding flexibility)."""
        return [
            self.timestamp,
            self.timescale,
            self.price,
            self.atm_iv,
            self.iv_25d_call,
            self.iv_25d_put,
            self.iv_slope,
            self.sd_band_upper,
            self.sd_band_lower,
            self.delta_pressure,
            self.gamma_exposure,
            self.vanna,
            self.vol_of_vol,
        ]

    def to_numpy(self):
        """Return feature vector as a numpy array of floats."""
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("numpy is required for StateFrame.to_numpy") from exc
        return np.asarray(self.feature_values(), dtype=float)


StateFrameBuffer = StrataStateBuffer[StateFrame]
