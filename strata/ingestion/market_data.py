"""Market data ingestion from IBKR API and mock data generation."""
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict
import logging
import random
import math

from ib_insync import IB, Stock
import numpy as np

from strata.config import IBKR_CONFIG, MOCK_IBKR_DATA, MOCK_DATA_SEED
from strata.db.queries import write_market_state

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

    def ingest_market_data(self, asset: str, timescale: str) -> None:
        """
        Fetch market data from IBKR and write to market_state table.

        Args:
            asset: Asset symbol (GLD, QQQ, XRT, TLT, NVDA)
            timescale: Time resolution (1h, 4h, 1d, 1w, 1m)
        """
        if not self._connected:
            self.connect()

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
                    'timestamp': datetime.now(),
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
        'GLD': 180.0,
        'QQQ': 450.0,
        'XRT': 75.0,
        'TLT': 95.0,
        'NVDA': 140.0
    }

    # Volatility per asset (daily std deviation as %)
    VOLATILITY = {
        'GLD': 0.01,   # 1% daily vol (low)
        'QQQ': 0.015,  # 1.5% daily vol (moderate)
        'XRT': 0.012,  # 1.2% daily vol
        'TLT': 0.008,  # 0.8% daily vol (low)
        'NVDA': 0.03   # 3% daily vol (high)
    }

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize mock ingester.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed if seed is not None else MOCK_DATA_SEED
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Track state for random walk
        self._drift = {asset: 0.0 for asset in self.BASE_PRICES}
        self._last_timestamp = {}

        logger.info(f"MockMarketDataIngester initialized with seed={self.seed}")

    def connect(self) -> None:
        """Mock connect (no-op)."""
        logger.info("Mock ingester connected (simulated)")

    def disconnect(self) -> None:
        """Mock disconnect (no-op)."""
        logger.info("Mock ingester disconnected (simulated)")

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
            timestamp: Optional timestamp (defaults to now)
        """
        if asset not in self.BASE_PRICES:
            logger.warning(f"Unknown asset {asset}, using default price 100.0")
            base_price = 100.0
            vol = 0.02
        else:
            base_price = self.BASE_PRICES[asset]
            vol = self.VOLATILITY[asset]

        if timestamp is None:
            timestamp = datetime.now()

        # Generate price
        price = self._generate_price(asset, base_price, vol, timestamp)

        # Generate volume
        volume = self._generate_volume(asset)

        # Generate bid-ask spread (proportional to volatility)
        bid_ask_spread = base_price * vol * 0.1

        data = {
            'asset': asset,
            'timestamp': timestamp,
            'timescale': timescale,
            'price': Decimal(str(round(price, 2))),
            'implied_vol': Decimal(str(round(vol, 6))),
            'skew': Decimal(str(round(random.uniform(-0.05, 0.05), 6))),
            'volume': volume,
            'bid_ask_spread': Decimal(str(round(bid_ask_spread, 4)))
        }

        write_market_state(data)
        logger.debug(
            f"Mock ingested {asset} {timescale}: price={data['price']}, "
            f"volume={data['volume']}"
        )


    def _generate_price(
        self,
        asset: str,
        base_price: float,
        vol: float,
        timestamp: datetime
    ) -> float:
        """
        Generate realistic price using sine wave + random walk + noise.
        """

        # Convert timestamp to time index (hours since epoch)
        t = timestamp.timestamp() / 3600.0

        # Sine wave component (30-day cycle)
        period = 30 * 24  # 30 days in hours
        sine_component = math.sin(2 * math.pi * t / period) * 0.05

        # Random walk drift
        drift_change = np.random.normal(0, vol / 10)

        # Lazy-init drift for unknown assets
        if asset not in self._drift:
            self._drift[asset] = 0.0

        self._drift[asset] = max(
            min(self._drift[asset] + drift_change, 0.2),
            -0.2
        )

        # Noise component
        noise = np.random.normal(0, vol / 2)

        # Final price
        price = base_price * (
            1
            + sine_component
            + self._drift[asset]
            + noise
        )

        return max(price, 0.01)

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
        multiplier = random.uniform(0.7, 1.3)

        return int(base * multiplier)

    def fetch_latest_price(self, asset: str) -> Optional[Decimal]:
        """
        Get synthetic current price.

        Args:
            asset: Asset symbol

        Returns:
            Synthetic price
        """
        if asset not in self.BASE_PRICES:
            return None

        base_price = self.BASE_PRICES[asset]
        vol = self.VOLATILITY[asset]
        price = self._generate_price(asset, base_price, vol, datetime.now())

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
            'skew': Decimal(str(round(random.uniform(-0.05, 0.05), 6)))
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def ingest_all_assets(timescale: str, use_mock: Optional[bool] = None) -> None:
    """
    Ingest data for all configured assets at given timescale.

    Args:
        timescale: Time resolution (1h, 4h, 1d, 1w, 1m)
        use_mock: Use mock data (defaults to MOCK_IBKR_DATA config)
    """
    from strata.config import ASSETS

    if use_mock is None:
        use_mock = MOCK_IBKR_DATA

    if use_mock:
        ingester = MockMarketDataIngester()
        logger.info(f"Using MockMarketDataIngester for timescale {timescale}")
    else:
        ingester = MarketDataIngester()
        logger.info(f"Using MarketDataIngester for timescale {timescale}")

    ingester.connect()

    try:
        for asset in ASSETS:
            ingester.ingest_market_data(asset, timescale)
    finally:
        ingester.disconnect()

    logger.info(f"Ingested data for {len(ASSETS)} assets at timescale {timescale}")


def generate_historical_data(
    asset: str,
    timescale: str,
    n_periods: int,
    start_time: Optional[datetime] = None
) -> None:
    """
    Generate historical synthetic data for backtesting.

    Args:
        asset: Asset symbol
        timescale: Time resolution
        n_periods: Number of historical periods to generate
        start_time: Starting timestamp (defaults to n_periods ago)
    """
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
        f"Generating {n_periods} historical periods for {asset} {timescale} "
        f"starting from {start_time}"
    )

    for i in range(n_periods):
        timestamp = start_time + timedelta(hours=hours_per_period * i)
        ingester.ingest_market_data(asset, timescale, timestamp=timestamp)

    logger.info(f"Generated {n_periods} historical data points for {asset}")
