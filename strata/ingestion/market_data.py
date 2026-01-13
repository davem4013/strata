"""Market data ingestion from IBKR API and mock data generation."""
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

class AnalyticsAPIMarketDataIngester:
    """
    Ingest market state from Analytics FastAPI service.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        logger.info(f"Analytics API ingester using {self.base_url}")

    def connect(self) -> None:
        # No persistent connection needed
        logger.info("Analytics API ingester ready")

    def disconnect(self) -> None:
        logger.info("Analytics API ingester disconnected")

    def ingest_market_data(self, asset: str, timescale: str) -> None:
        """
        Pull derived market state from Analytics and store in STRATA DB.
        """

        try:
            # 1️⃣ Pull IV / Greeks summary
            iv_resp = requests.post(
                f"{self.base_url}/analytics/compute/iv",
                json={"symbol": asset}
            )
            iv_resp.raise_for_status()
            iv_data = iv_resp.json()

            # 2️⃣ Pull current regime (optional but powerful)
            regime_resp = requests.get(
                f"{self.base_url}/regime/current",
                params={"symbol": asset}
            )
            regime_data = regime_resp.json() if regime_resp.ok else {}

            # 3️⃣ Pull underlying price
            price = iv_data.get("spot")
            implied_vol = iv_data.get("atm_iv")
            term_structure = (
                iv_data.get("forward_vol_spine")
                or iv_data.get("term_structure")
                or iv_data.get("atm_term_structure")
                or []
            )

            if price is None:
                logger.warning(f"No price returned for {asset}")
                return

            data = {
                "asset": asset,
                "timestamp": datetime.utcnow(),
                "timescale": timescale,
                "price": Decimal(str(price)),
                "implied_vol": (
                    Decimal(str(implied_vol)) if implied_vol is not None else None
                ),
                "skew": None,  # optional: derive later
                "volume": None,
                "bid_ask_spread": None,
            }

            write_market_state(data)

            logger.info(
                f"Analytics-ingested {asset}: price={price}, iv={implied_vol}, "
                f"regime={regime_data.get('regime_type')}"
            )

        except Exception as e:
            logger.error(f"Analytics API ingestion failed for {asset}: {e}")

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
        self._rng = random.Random(self.seed)
        self._np_rng = np.random.default_rng(self.seed)


        # Track state for random walk
        self._drift = {asset: 0.0 for asset in self.BASE_PRICES}
        self._last_timestamp = {}
        self._step = {}
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
            'skew': Decimal(str(round(self._rng.uniform(-0.05, 0.05), 6))),
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

        # Initialize step counter
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
# Synthetic Market Data Ingester (for testing without IBKR connection)
# ============================================================================


    """
    Ingest market state from Analytics FastAPI service.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        logger.info(f"Analytics API ingester using {self.base_url}")

    def connect(self) -> None:
        # No persistent connection needed
        logger.info("Analytics API ingester ready")

    def disconnect(self) -> None:
        logger.info("Analytics API ingester disconnected")

    def ingest_market_data(self, asset: str, timescale: str) -> None:
        """
        Pull derived market state from Analytics and store in STRATA DB.
        """

        try:
            # 1️⃣ Pull IV / Greeks summary
            iv_resp = requests.post(
                f"{self.base_url}/analytics/compute/iv",
                json={"symbol": asset}
            )
            iv_resp.raise_for_status()
            iv_data = iv_resp.json()

            # 2️⃣ Pull current regime (optional but powerful)
            regime_resp = requests.get(
                f"{self.base_url}/regime/current",
                params={"symbol": asset}
            )
            regime_data = regime_resp.json() if regime_resp.ok else {}

            # 3️⃣ Pull underlying price
            price = iv_data.get("spot")
            implied_vol = iv_data.get("atm_iv")

            if price is None:
                logger.warning(f"No price returned for {asset}")
                return

            data = {
                "asset": asset,
                "timestamp": datetime.utcnow(),
                "timescale": timescale,
                "price": Decimal(str(price)),
                "implied_vol": (
                    Decimal(str(implied_vol)) if implied_vol is not None else None
                ),
                "skew": None,  # optional: derive later
                "volume": None,
                "bid_ask_spread": None,
            }

            # Build and persist forward volatility spine + residual field
            if term_structure:
                try:
                    spine = build_forward_vol_spine(term_structure)
                    residual_field = extract_residual_field(spine)
                    spine_metrics = analyze_spine_field(spine)

                    write_system_state(
                        f"forward_vol_spine:{asset}:{timescale}",
                        {
                            "spine": serialize_spine(spine),
                            "residual_field": residual_field,
                            "metrics": spine_metrics,
                            "as_of": datetime.utcnow().isoformat(),
                        },
                    )
                    logger.debug(
                        "Updated forward vol spine for %s %s: front_residual=%.4f, "
                        "curvature=%.4f",
                        asset,
                        timescale,
                        spine_metrics["front_residual"],
                        spine_metrics["max_abs_curvature"],
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to construct forward vol spine for %s: %s",
                        asset,
                        exc,
                    )

            write_market_state(data)

            logger.info(
                f"Analytics-ingested {asset}: price={price}, iv={implied_vol}, "
                f"regime={regime_data.get('regime_type')}"
            )

        except Exception as e:
            logger.error(f"Analytics API ingestion failed for {asset}: {e}")



# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def ingest_all_assets(timescale: str, source: str = "mock") -> None:
    from strata.config import ASSETS

    if source == "mock":
        ingester = MockMarketDataIngester()
    elif source == "ibkr":
        ingester = MarketDataIngester()
    elif source == "analytics":
        ingester = AnalyticsAPIMarketDataIngester()
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
