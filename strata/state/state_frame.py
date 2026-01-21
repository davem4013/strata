"""
StateFrame definitions and buffers for STRATA.
"""

from typing import Optional

from .state_buffer import StrataStateBuffer


class StateFrame:
    """
    Lightweight wrapper representing a single interpreted STRATA state frame.
    """

    def __init__(
        self,
        timestamp: float,
        symbol: str = "",
        asset: str = "",
        value: Optional[float] = None,
        residual: Optional[float] = None,
        timescale: str = "",
        price: float = 0.0,
        price_pressure: float = 0.0,
        atm_iv: float = 0.0,
        iv_25d_call: float = 0.0,
        iv_25d_put: float = 0.0,
        iv_slope: float = 0.0,
        sd_band_upper: float = 0.0,
        sd_band_lower: float = 0.0,
        delta_pressure: float = 0.0,
        gamma_exposure: float = 0.0,
        vanna: float = 0.0,
        vol_of_vol: float = 0.0,
        response: Optional[float] = None,
    ):
        self.timestamp = timestamp
        self.symbol = symbol
        self.asset = asset
        self.value = value
        self.residual = residual
        self.timescale = timescale
        self.price = price
        self.price_pressure = price_pressure
        self.atm_iv = atm_iv
        self.iv_25d_call = iv_25d_call
        self.iv_25d_put = iv_25d_put
        self.iv_slope = iv_slope
        self.sd_band_upper = sd_band_upper
        self.sd_band_lower = sd_band_lower
        self.delta_pressure = delta_pressure
        self.gamma_exposure = gamma_exposure
        self.vanna = vanna
        self.vol_of_vol = vol_of_vol
        self.response = response


# Canonical buffer for StateFrame
StateFrameBuffer = StrataStateBuffer

__all__ = [
    "StateFrame",
    "StateFrameBuffer",
]
