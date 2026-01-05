#!/usr/bin/env python3
"""
STRATA Database Setup Script

This script initializes the STRATA database schema.
Run this before first use or to reset the database.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging

from strata.config import DB_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_postgres_connection():
    """Check if we can connect to PostgreSQL."""
    try:
        # Try to connect to the default 'postgres' database
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database='postgres',
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        conn.close()
        return True
    except psycopg2.Error as e:
        logger.error(f"Cannot connect to PostgreSQL: {e}")
        return False


def create_database():
    """Create the STRATA database if it doesn't exist."""
    try:
        # Connect to default database
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database='postgres',
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (DB_CONFIG['database'],)
        )
        exists = cursor.fetchone()

        if exists:
            logger.info(f"Database '{DB_CONFIG['database']}' already exists")
        else:
            # Create database
            cursor.execute(f"CREATE DATABASE {DB_CONFIG['database']}")
            logger.info(f"Created database '{DB_CONFIG['database']}'")

        cursor.close()
        conn.close()
        return True

    except psycopg2.Error as e:
        logger.error(f"Error creating database: {e}")
        return False


def run_sql_file(filepath: Path):
    """Run a SQL file against the STRATA database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        logger.info(f"Running {filepath.name}...")

        with open(filepath, 'r') as f:
            sql = f.read()

        cursor.execute(sql)
        conn.commit()

        cursor.close()
        conn.close()

        logger.info(f"✓ {filepath.name} completed")
        return True

    except psycopg2.Error as e:
        logger.error(f"Error running {filepath.name}: {e}")
        return False


def setup_schema():
    """Set up the complete database schema."""
    sql_dir = Path(__file__).parent.parent / 'sql'

    # Schema files in order
    schema_files = [
        '01_core.sql',
        '02_ai.sql',
        '03_analysis.sql',
        '04_system.sql'
    ]

    logger.info("Setting up STRATA database schema...")

    for filename in schema_files:
        filepath = sql_dir / filename
        if not filepath.exists():
            logger.error(f"Schema file not found: {filepath}")
            return False

        if not run_sql_file(filepath):
            return False

    return True


def verify_setup():
    """Verify that all tables were created successfully."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Expected tables
        expected_tables = [
            'market_state',
            'residual_state',
            'basin_geometry',
            'basin_position',
            'model_interpretation',
            'agreement_metrics',
            'cross_scale_coherence',
            'regime_event',
            'system_state'
        ]

        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)

        tables = [row[0] for row in cursor.fetchall()]

        logger.info(f"\nVerifying schema...")
        logger.info(f"Found {len(tables)} tables:")

        all_present = True
        for table in expected_tables:
            if table in tables:
                logger.info(f"  ✓ {table}")
            else:
                logger.error(f"  ✗ {table} - MISSING")
                all_present = False

        # Check for pgvector extension
        cursor.execute(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        )
        has_vector = cursor.fetchone() is not None

        if has_vector:
            logger.info(f"  ✓ pgvector extension")
        else:
            logger.warning(f"  ✗ pgvector extension - NOT INSTALLED")
            all_present = False

        cursor.close()
        conn.close()

        return all_present

    except psycopg2.Error as e:
        logger.error(f"Error verifying setup: {e}")
        return False


def main():
    """Main setup routine."""
    logger.info("=" * 60)
    logger.info("STRATA Database Setup")
    logger.info("=" * 60)

    logger.info(f"\nDatabase configuration:")
    logger.info(f"  Host: {DB_CONFIG['host']}")
    logger.info(f"  Port: {DB_CONFIG['port']}")
    logger.info(f"  Database: {DB_CONFIG['database']}")
    logger.info(f"  User: {DB_CONFIG['user']}")

    # Check PostgreSQL connection
    logger.info("\nChecking PostgreSQL connection...")
    if not check_postgres_connection():
        logger.error("Cannot connect to PostgreSQL. Please check:")
        logger.error("  1. PostgreSQL is running")
        logger.error("  2. Database credentials in .env are correct")
        logger.error("  3. User has necessary permissions")
        sys.exit(1)

    logger.info("✓ PostgreSQL connection successful")

    # Create database
    logger.info("\nCreating database if needed...")
    if not create_database():
        logger.error("Failed to create database")
        sys.exit(1)

    # Set up schema
    logger.info("\nSetting up schema...")
    if not setup_schema():
        logger.error("Failed to set up schema")
        sys.exit(1)

    # Verify setup
    if not verify_setup():
        logger.error("\nSetup verification failed")
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("Database setup completed successfully!")
    logger.info("=" * 60)
    logger.info("\nYou can now run STRATA applications.")


if __name__ == '__main__':
    main()
