# STRATA

**State-Tracking Regime Analysis Through AI**

STRATA models financial markets as non-autonomous dynamical systems, detecting regime stability and phase transitions via moving basins in statistical residual space.

## Core Concept

- **Markets as attractor basins**: Price movements exist within dynamical regime states
- **Residuals as coordinates**: Deviations from fair value define position in basin
- **Basin boundaries**: Limits of mean-reversion behavior
- **Model disagreement as signal**: AI model divergence indicates regime uncertainty
- **Cross-scale cascades**: Detect regime transitions propagating through timescales

## Project Status

**Phase 1: Foundation** ✅ **COMPLETE**
- ✅ Project structure and dependencies
- ✅ Database schema (9 tables, pgvector support)
- ✅ Database connection pooling
- ✅ Configuration management (environment variables)
- ✅ Query interface (read/write operations)

**Phase 2: Data Pipeline** ✅ **COMPLETE**
- ✅ Market data ingestion (IBKR API + Mock)
- ✅ Residual calculation (least squares)
- ✅ Basin geometry detection
- ✅ Position analysis

**Phase 3: Intelligence Layer** ✅ **COMPLETE**
- ✅ AI model integration (Mock mode + Youtu-LLM ready)
- ✅ Agreement metrics computation (disagreement engine)
- ✅ Event detection and alerting

**Phase 4: Advanced Features** 🔄 **NEXT**
- ⏳ Cross-scale cascade detection
- ⏳ Pipeline orchestration
- ⏳ Real Youtu-LLM integration

## Quick Start

### 1. Prerequisites

- Python 3.9+
- PostgreSQL 14+
- pgvector extension

### 2. Installation

```bash
# Clone repository
git clone <repository-url>
cd strata

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env
```

Key configuration:
- `DB_*`: Database connection settings
- `IBKR_*`: Interactive Brokers API settings
- `ASSETS`: Comma-separated list of assets to track
- `TIMESCALES`: Analysis timescales (1h, 4h, 1d, 1w, 1m)

### 4. Database Setup

**Option A: Automated**
```bash
python scripts/setup_database.py
```

**Option B: Manual**
```bash
# Create database and user
sudo -u postgres psql
CREATE USER strata_user WITH PASSWORD 'changeme';
CREATE DATABASE strata OWNER strata_user;
\q

# Run schema
psql -U strata_user -d strata -f sql/00_setup.sql
```

See [sql/README.md](sql/README.md) for detailed database setup instructions.

### 5. Verify Installation

```bash
# Test database connection
python -c "from strata.db.connection import init_pool, get_cursor; from strata.config import DB_CONFIG; init_pool(**DB_CONFIG); print('✓ Database connection successful')"

# Check configuration
python -c "from strata import ASSETS, TIMESCALES; print(f'Assets: {ASSETS}'); print(f'Timescales: {TIMESCALES}')"
```

## Usage

### Running Demos

**Phase 2 Demo - Data Pipeline:**
```bash
python examples/demo_pipeline.py
```
This demonstrates:
1. Generate 100 days of synthetic market data
2. Compute least squares residuals
3. Identify basin structure
4. Analyze price position
5. Display risk assessment and trading recommendation

**Phase 3 Demo - Intelligence Layer (THE CORE INNOVATION):**
```bash
python examples/demo_intelligence.py
```
This demonstrates:
1. Complete data pipeline (above)
2. Multi-model AI basin interpretation
3. **Disagreement engine** - semantic distance, variance, directional divergence
4. Agreement score calculation (0-1 scale)
5. Regime event detection and alerting
6. Risk-based trading recommendations

**The key insight: Model disagreement = regime uncertainty**

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=strata

# Run Phase 2 tests (data pipeline)
python tests/test_pipeline.py

# Run Phase 3 tests (AI intelligence layer)
python tests/test_ai_pipeline.py
```

### Using the Pipeline in Code

```python
from strata.config import DB_CONFIG
from strata.db.connection import init_pool
from strata.ingestion.market_data import MockMarketDataIngester
from strata.analysis.residuals import compute_residuals
from strata.analysis.basins import identify_basins
from strata.analysis.position import compute_basin_position, get_position_summary

# Initialize database
init_pool(**DB_CONFIG)

# Generate mock data
ingester = MockMarketDataIngester()
for i in range(100):
    ingester.ingest_market_data('GLD', '1d')

# Compute residuals
compute_residuals('GLD', '1d', lookback_periods=20)

# Identify basin
basin_id = identify_basins('GLD', '1d')

# Analyze position
position_state = compute_basin_position('GLD', '1d')
summary = get_position_summary('GLD', '1d')

print(f"Position: {position_state}")
print(f"Risk: {summary['risk']['overall_risk']:.3f}")
print(f"Recommendation: {summary['risk']['recommendation']}")
```

## Architecture

### Database Schema

**Core Tables:**
- `market_state` - Raw market data from IBKR
- `residual_state` - Least squares residuals
- `basin_geometry` - Attractor basin structure
- `basin_position` - Price-basin relationship

**AI Tables:**
- `model_interpretation` - Model assessments
- `agreement_metrics` - Disagreement metrics

**Analysis Tables:**
- `cross_scale_coherence` - Cascade detection
- `regime_event` - Events and alerts

**System Tables:**
- `system_state` - Configuration

### Module Structure

```
strata/
├── config.py              # Configuration management
├── db/
│   ├── connection.py      # Database connection pool
│   └── queries.py         # Query functions
├── ingestion/             # Market data ingestion
├── analysis/              # Residuals, basins, cascades
├── ai/                    # Model integration
├── events/                # Event detection
└── orchestration/         # Pipeline management
```

## Development

### Running Tests

```bash
pytest
```

### Code Style

```bash
# Format code
black strata/

# Lint
ruff check strata/
```

### Database Management

```bash
# Reset database (WARNING: destroys all data)
psql -U strata_user -d strata -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
python scripts/setup_database.py

# View tables
psql -U strata_user -d strata -c "\dt"

# Check data
psql -U strata_user -d strata -c "SELECT COUNT(*) FROM market_state;"
```

## Design Philosophy

1. **Not predicting direction** - Detecting regime uncertainty
2. **State-based memory** - Context-embedded recall like human episodic memory
3. **Disagreement as signal** - Model divergence is information, not noise
4. **Multi-resolution analysis** - Hourly to monthly timescales
5. **Dynamical systems framing** - Basins, attractors, phase transitions

## Configuration

All configuration via environment variables (`.env` file):

```bash
# Database
DB_HOST=localhost
DB_NAME=strata
DB_USER=strata_user
DB_PASSWORD=changeme

# Assets
ASSETS=GLD,QQQ,XRT,TLT,NVDA
TIMESCALES=1h,4h,1d,1w,1m

# Analysis Parameters
RESIDUAL_LOOKBACK_PERIODS=20
BASIN_CLUSTERING_WINDOW=100
BOUNDARY_SIGMA=2.0

# Thresholds
AGREEMENT_THRESHOLD_LOW=0.5
DIRECTIONAL_DIVERGENCE_HIGH=0.7
```

See `.env.example` for complete configuration options.

## License

MIT

## Contributing

This is a research/trading system in active development.

**Phase 1 (Foundation)** ✅ Complete
**Phase 2 (Data Pipeline)** ✅ Complete
**Phase 3 (Intelligence Layer)** ✅ Complete
**Phase 4 (Advanced Features)** 🔄 Next
