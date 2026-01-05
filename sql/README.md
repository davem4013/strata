# STRATA Database Schema

This directory contains SQL schema files for the STRATA database.

## Prerequisites

- PostgreSQL 14 or higher
- `pgvector` extension installed

### Installing pgvector

```bash
# On Ubuntu/Debian
sudo apt-get install postgresql-14-pgvector

# On macOS with Homebrew
brew install pgvector

# Or install from source
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

## Database Setup

### Option 1: Automated Setup

Run the setup script from the repository root:

```bash
python scripts/setup_database.py
```

### Option 2: Manual Setup

1. **Create the database and user:**

```bash
# Connect to PostgreSQL as superuser
sudo -u postgres psql

# Create user and database
CREATE USER strata_user WITH PASSWORD 'changeme';
CREATE DATABASE strata OWNER strata_user;

# Grant permissions
GRANT ALL PRIVILEGES ON DATABASE strata TO strata_user;

# Exit
\q
```

2. **Run the schema setup:**

```bash
# From the repository root
psql -U strata_user -d strata -f sql/00_setup.sql
```

Or run individual schema files in order:

```bash
psql -U strata_user -d strata -f sql/01_core.sql
psql -U strata_user -d strata -f sql/02_ai.sql
psql -U strata_user -d strata -f sql/03_analysis.sql
psql -U strata_user -d strata -f sql/04_system.sql
```

## Schema Files

- **00_setup.sql** - Master setup file that loads all schemas
- **01_core.sql** - Core tables: market_state, residual_state, basin_geometry, basin_position
- **02_ai.sql** - AI tables: model_interpretation, agreement_metrics
- **03_analysis.sql** - Analysis tables: cross_scale_coherence, regime_event
- **04_system.sql** - System table: system_state

## Schema Overview

### Core Tables

1. **market_state** - Raw market data from IBKR API
2. **residual_state** - Least squares residuals (price deviations from fair value)
3. **basin_geometry** - Attractor basin structure in residual space
4. **basin_position** - Price position relative to basin

### AI Tables

5. **model_interpretation** - AI model assessments of basin state
6. **agreement_metrics** - Model disagreement metrics (primary signal)

### Analysis Tables

7. **cross_scale_coherence** - Disagreement propagation across timescales
8. **regime_event** - Regime transition events and alerts

### System Tables

9. **system_state** - System configuration and runtime state

## Verification

After setup, verify the schema:

```bash
psql -U strata_user -d strata -c "\dt"
```

You should see all 9 tables listed.

Check that pgvector is enabled:

```bash
psql -U strata_user -d strata -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

## Resetting the Database

To drop all tables and start fresh:

```bash
psql -U strata_user -d strata -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
psql -U strata_user -d strata -f sql/00_setup.sql
```

## Security Notes

- Change the default password for `strata_user` in production
- Use SSL connections for remote database access
- Restrict network access to the database port (5432)
- Consider using role-based access control for different components
