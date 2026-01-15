"""Market data ingestion from IBKR API, Analytics API, and mock data generation."""
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List
import logging
import random
import math
import requests
import numpy as np
from ib_insync import IB, Stock

from strata.analysis.vol_spine import (
    analyze_spine_field,
    build_forward_vol_spine,
    extract_residual_field,
    serialize_spine,
)
from strata.config import IBKR_CONFIG, MOCK_IBKR_DATA, MOCK_DATA_SEED
from strata.db.queries import write_market_state, write_system_state

logger = logging.getLogger(__name__)


# ============================================================================
# IBKR Market Data Ingester
# ============================================================================

class MarketDataIngester:
    """Fetch and store market data from IBKR."""

    def __init__(self):
        self.ib = IB()
        self._connected = False

    def connect(self) -> None:
        """Connect to IBKR API."""
        if not self._connected:
            try:
                self.ib.connect(
                    IBKR_CONFIG['host'],
                    IBKR_CONFIG['port'],
                    clientId=IBKR_CONFIG['client_id']
                )
                self._connected = True
                logger.info(
                    f"Connected to IBKR API at {IBKR_CONFIG['host']}:{IBKR_CONFIG['port']}"
                )
            except Exception as e:
                logger.error(f"Failed to connect to IBKR API: {e}")
                raise

    def disconnect(self) -> None:
        """Disconnect from IBKR API."""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR API")

    def ingest_market_data(
        self,
        asset: str,
        timescale: str,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Fetch market data from IBKR and write to market_state table.

        Args:
            asset: Asset symbol (GLD, QQQ, XRT, TLT, NVDA)
            timescale: Time resolution (1h, 4h, 1d, 1w, 1m)
        """
        if not self._connected:
            self.connect()

        if timestamp is None:
            timestamp = datetime.utcnow()

        try:
            # Create contract
            contract = Stock(asset, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            # Get market data
            ticker = self.ib.reqMktData(contract)
            self.ib.sleep(1)  # Wait for data

            if ticker.last and ticker.last > 0:
                data = {
                    'asset': asset,
                    'timestamp': timestamp,
                    'timescale': timescale,
                    'price': Decimal(str(ticker.last)),
                    'implied_vol': None,  # TODO: Get from options
                    'skew': None,  # TODO: Calculate from options
                    'volume': int(ticker.volume) if ticker.volume else None,
                    'bid_ask_spread': (
                        Decimal(str(ticker.ask - ticker.bid))
                        if ticker.ask and ticker.bid
                        else None
                    )
                }

                write_market_state(data)
                logger.info(
                    f"Ingested {asset} {timescale}: price={data['price']}, "
                    f"volume={data['volume']}"
                )
            else:
                logger.warning(f"No market data for {asset}")

        except Exception as e:
            logger.error(f"Error ingesting market data for {asset}: {e}")
            raise

    def fetch_latest_price(self, asset: str) -> Optional[Decimal]:
        """
        Get most recent price for asset.

        Args:
            asset: Asset symbol

        Returns:
            Latest price as Decimal, or None if unavailable
        """
        if not self._connected:
            self.connect()

        try:
            contract = Stock(asset, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            ticker = self.ib.reqMktData(contract)
            self.ib.sleep(1)

            if ticker.last and ticker.last > 0:
                return Decimal(str(ticker.last))
            return None

        except Exception as e:
            logger.error(f"Error fetching price for {asset}: {e}")
            return None

    def fetch_volatility_surface(self, asset: str) -> Dict:
        """
        Get implied vol and skew data from options.

        TODO: Implement options chain analysis
        - Query options at multiple strikes
        - Calculate ATM implied vol
        - Calculate skew (OTM put vol - ATM vol)
        - Return as dict

        Args:
            asset: Asset symbol

        Returns:
            Dict with 'implied_vol' and 'skew' (currently None)
        """
        logger.warning(f"fetch_volatility_surface not yet implemented for {asset}")
        return {
            'implied_vol': None,
            'skew': None
        }


# ============================================================================
# Mock Market Data Ingester (for testing without IBKR connection)
# ============================================================================

class MockMarketDataIngester:
    """
    Generate realistic synthetic market data for testing.

    Uses sine wave + random walk + noise to simulate price movements.
    """

    # Base prices for each asset
    BASE_PRICES = {
        'GLD': 175.0,
        'QQQ': 385.0,
        'XRT': 62.0,
        'TLT': 93.0,
        'NVDA': 475.0
    }

    # Typical volatilities
    VOLATILITY = {
        'GLD': 0.015,  # ~1.5% daily vol
        'QQQ': 0.020,  # ~2.0% daily vol
        'XRT': 0.018,
        'TLT': 0.012,
        'NVDA': 0.035  # Higher vol for tech
    }

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize mock data generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is None:
            seed = MOCK_DATA_SEED

        self._seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

        # Track state for each asset
        self._step = {}
        self._drift = {}

        logger.info(f"Mock data ingester initialized with seed {seed}")

    def connect(self) -> None:
        logger.info("Mock ingester ready (no connection needed)")

    def disconnect(self) -> None:
        logger.info("Mock ingester disconnected")

    def ingest_market_data(
        self,
        asset: str,
        timescale: str,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Generate and store synthetic market data.

        Args:
            asset: Asset symbol
            timescale: Time resolution
            timestamp: Timestamp for this data point
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        if asset not in self.BASE_PRICES:
            logger.warning(f"Unknown asset {asset}, skipping")
            return

        base_price = self.BASE_PRICES[asset]
        vol = self.VOLATILITY[asset]

        price = self._generate_price(asset, base_price, vol)
        vol_data = self.fetch_volatility_surface(asset)
        volume = self._generate_volume(asset)

        data = {
            'asset': asset,
            'timestamp': timestamp,
            'timescale': timescale,
            'price': Decimal(str(round(price, 2))),
            'implied_vol': vol_data['implied_vol'],
            'skew': vol_data['skew'],
            'volume': volume,
            'bid_ask_spread': Decimal(str(round(self._rng.uniform(0.01, 0.05), 4)))
        }

        write_market_state(data)
        logger.debug(
            f"Mock data for {asset}: price={data['price']}, vol={data['implied_vol']}"
        )

    def _generate_price(self, asset: str, base_price: float, vol: float) -> float:
        """
        Generate realistic price with trends and noise.

        Args:
            asset: Asset symbol
            base_price: Starting price
            vol: Volatility level

        Returns:
            Generated price
        """
        if asset not in self._step:
            self._step[asset] = 0

        step = self._step[asset]
        self._step[asset] += 1

        # Synthetic time (deterministic)
        period = 30
        sine_component = math.sin(2 * math.pi * step / period) * 0.05

        # Random walk drift (seeded)
        drift_change = self._np_rng.normal(0, vol / 10)

        if asset not in self._drift:
            self._drift[asset] = 0.0

        self._drift[asset] = max(
            min(self._drift[asset] + drift_change, 0.2),
            -0.2
        )

        noise = self._np_rng.normal(0, vol / 2)

        price = base_price * (1 + sine_component + self._drift[asset] + noise)

        return max(price, 0.01)

    def _generate_price_at_step(self, asset, base_price, vol, step):
        period = 30
        sine_component = math.sin(2 * math.pi * step / period) * 0.05

        drift = self._drift.get(asset, 0.0)
        noise = self._np_rng.normal(0, vol / 2)

        return max(base_price * (1 + sine_component + drift + noise), 0.01)

    def _generate_volume(self, asset: str) -> int:
        """
        Generate realistic volume.

        Args:
            asset: Asset symbol

        Returns:
            Volume (integer)
        """
        # Base volume varies by asset
        base_volumes = {
            'GLD': 5_000_000,
            'QQQ': 50_000_000,
            'XRT': 3_000_000,
            'TLT': 4_000_000,
            'NVDA': 40_000_000
        }

        base = base_volumes.get(asset, 1_000_000)

        # Add randomness (±30%)
        multiplier = self._rng.uniform(0.7, 1.3)

        return int(base * multiplier)

    def fetch_latest_price(self, asset: str) -> Optional[Decimal]:
        if asset not in self.BASE_PRICES:
            return None

        base_price = self.BASE_PRICES[asset]
        vol = self.VOLATILITY[asset]

        # Snapshot step instead of advancing
        step = self._step.get(asset, 0)

        price = self._generate_price_at_step(
            asset, base_price, vol, step
        )

        return Decimal(str(round(price, 2)))

    def fetch_volatility_surface(self, asset: str) -> Dict:
        """
        Get mock volatility data.

        Args:
            asset: Asset symbol

        Returns:
            Dict with mock implied_vol and skew
        """
        vol = self.VOLATILITY.get(asset, 0.02)
        return {
            'implied_vol': Decimal(str(vol)),
            'skew': Decimal(str(round(self._rng.uniform(-0.05, 0.05), 6)))
        }


# ============================================================================
# Analytics API Ingester (pulls from Analytics FastAPI service)
# ============================================================================

class AnalyticsAPIIngester:
    """
    Ingest market state from Analytics FastAPI service.
    
    This pulls synthetic regime data generated by Analytics, providing
    rich, varied data with known regime labels for STRATA validation.
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        logger.info(f"Analytics API ingester using {self.base_url}")
    
    def connect(self) -> None:
        """Test connection to Analytics API."""
        try:
            response = requests.get(f"{self.base_url}/state/surface/symbols", timeout=5)
            response.raise_for_status()
            logger.info("Analytics API ingester ready")
        except Exception as e:
            logger.error(f"Failed to connect to Analytics API: {e}")
            raise
    
    def disconnect(self) -> None:
        logger.info("Analytics API ingester disconnected")
    
    def ingest_market_data(
        self,
        asset: str,
        timescale: str,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Pull latest surface state from Analytics and store in STRATA DB.
        
        Args:
            asset: Asset symbol
            timescale: Time resolution
            timestamp: Timestamp to use (defaults to surface timestamp)
        """
        try:
            # Pull latest IV surface state
            surface_resp = requests.get(
                f"{self.base_url}/state/surface/latest",
                params={"symbol": asset},
                timeout=5
            )
            surface_resp.raise_for_status()
            surface_data = surface_resp.json()
            
            # Pull current regime for validation
            regime_resp = requests.get(
                f"{self.base_url}/regime/current",
                timeout=5
            )
            regime_data = regime_resp.json() if regime_resp.ok else {}
            
            # Extract metrics
            price = surface_data.get("spot")
            iv_surface = surface_data.get("iv_surface", {})
            
            # Calculate ATM IV from surface
            implied_vol = None
            if iv_surface:
                if isinstance(iv_surface, dict):
                    front_expiry = sorted(iv_surface.keys())[0]
                    strikes = iv_surface.get(front_expiry, {})
                    if isinstance(strikes, dict) and strikes:
                        closest_strike = min(
                            strikes.keys(),
                            key=lambda strike: abs(float(strike) - float(price))
                        )
                        implied_vol = strikes[closest_strike]
                elif isinstance(iv_surface, list) and len(iv_surface) > 0:
                    atm_row = iv_surface[0]
                    if len(atm_row) > 0:
                        mid_idx = len(atm_row) // 2
                        implied_vol = atm_row[mid_idx]
            
            if price is None:
                logger.warning(f"No price returned for {asset}")
                return
            
            # Use surface timestamp if available, otherwise provided timestamp
            if timestamp is None:
                ts = surface_data.get("timestamp")
                timestamp = datetime.fromisoformat(ts) if ts else datetime.utcnow()
            
            data = {
                "asset": asset,
                "timestamp": timestamp,
                "timescale": timescale,
                "price": Decimal(str(price)),
                "implied_vol": (
                    Decimal(str(implied_vol)) if implied_vol is not None else None
                ),
                "skew": None,
                "volume": None,
                "bid_ask_spread": None,
            }

            
            # Store regime label for validation
            if regime_data.get("regime_type"):
                write_system_state(
                    f"regime_label:{asset}:{timescale}:{timestamp.isoformat()}",
                    {
                        "regime_type": regime_data.get("regime_type"),
                        "regime_label": regime_data.get("regime_label"),
                        "feedback_boost": regime_data.get("feedback_boost"),
                        "vol_skew": regime_data.get("vol_skew"),
                        "as_of": timestamp.isoformat(),
                    }
                )
            
            write_market_state(data)
            logger.debug(
                f"Analytics-ingested {asset}: price={price}, iv={implied_vol}, "
                f"regime={regime_data.get('regime_type')}"
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Analytics API request failed for {asset}: {e}")
        except Exception as e:
            logger.error(f"Analytics API ingestion failed for {asset}: {e}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def ingest_all_assets(timescale: str, source: str = "mock") -> None:
    """
    Ingest market data for all configured assets.
    
    Args:
        timescale: Time resolution (1h, 4h, 1d, 1w, 1m)
        source: Data source ("mock", "ibkr", or "analytics")
    """
    from strata.config import ASSETS

    if source == "mock":
        ingester = MockMarketDataIngester()
    elif source == "ibkr":
        ingester = MarketDataIngester()
    elif source == "analytics":
        ingester = AnalyticsAPIIngester()
    else:
        raise ValueError(f"Unknown ingestion source: {source}")

    ingester.connect()
    try:
        for asset in ASSETS:
            ingester.ingest_market_data(asset, timescale)
    finally:
        ingester.disconnect()


def generate_historical_data(
    asset: str,
    timescale: str,
    n_periods: int,
    start_time: Optional[datetime] = None,
    source: str = "analytics"
) -> None:
    """
    Generate or fetch historical data for backtesting.
    
    Args:
        asset: Asset symbol
        timescale: Time resolution
        n_periods: Number of historical periods to generate/fetch
        start_time: Starting timestamp (for mock data only)
        source: Data source ("mock" or "analytics")
    """
    
    if source == "analytics":
        # Fetch real historical surface data from Analytics
        ingester = AnalyticsAPIIngester()
        
        try:
            logger.info(
                f"Fetching {n_periods} historical periods for {asset} from Analytics"
            )
            
            response = requests.get(
                f"{ingester.base_url}/state/surface/history",
                params={"symbol": asset, "n": n_periods},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            items = data.get("items", [])
            if not items:
                logger.warning(f"No historical data available for {asset}")
                return
            
            # Ingest each historical snapshot
            for item in items:
                # Parse timestamp (handles multiple formats)
                ts = item["timestamp"]
                if isinstance(ts, str):
                    timestamp = datetime.fromisoformat(ts)
                elif isinstance(ts, (int, float)):
                    timestamp = datetime.fromtimestamp(ts)
                else:
                    timestamp = ts if isinstance(ts, datetime) else datetime.utcnow()
                
                price = item["spot"]
                iv_surface = item.get("iv_surface", {})
                
                # Extract ATM IV
                implied_vol = None
                if iv_surface:
                    if isinstance(iv_surface, dict):
                        front_expiry = sorted(iv_surface.keys())[0]
                        strikes = iv_surface.get(front_expiry, {})
                        if isinstance(strikes, dict) and strikes:
                            closest_strike = min(
                                strikes.keys(),
                                key=lambda strike: abs(float(strike) - float(price))
                            )
                            implied_vol = strikes[closest_strike]
                    elif isinstance(iv_surface, list) and len(iv_surface) > 0:
                        atm_row = iv_surface[0]
                        if len(atm_row) > 0:
                            mid_idx = len(atm_row) // 2
                            implied_vol = atm_row[mid_idx]
                
                write_market_state({
                    "asset": asset,
                    "timestamp": timestamp,
                    "timescale": timescale,
                    "price": Decimal(str(price)),
                    "implied_vol": (
                        Decimal(str(implied_vol)) if implied_vol is not None else None
                    ),
                    "skew": None,
                    "volume": None,
                    "bid_ask_spread": None,
                })
            
            logger.info(
                f"Ingested {len(items)} historical periods for {asset} from Analytics"
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch Analytics history for {asset}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing Analytics history for {asset}: {e}")
            raise
            
    else:  # source == "mock"
        # Generate synthetic data using MockMarketDataIngester
        ingester = MockMarketDataIngester()

        # Calculate time delta per period
        timescale_hours = {
            '1h': 1,
            '4h': 4,
            '1d': 24,
            '1w': 168,
            '1m': 720
        }

        hours_per_period = timescale_hours.get(timescale, 24)

        if start_time is None:
            start_time = datetime.now() - timedelta(hours=hours_per_period * n_periods)

        logger.info(
            f"Generating {n_periods} mock historical periods for {asset} {timescale} "
            f"starting from {start_time}"
        )

        for i in range(n_periods):
            timestamp = start_time + timedelta(hours=hours_per_period * i)
            ingester.ingest_market_data(asset, timescale, timestamp=timestamp)

        logger.info(f"Generated {n_periods} mock historical data points for {asset}")
