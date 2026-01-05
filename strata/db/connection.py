"""Database connection management with connection pooling."""
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
from typing import Optional
import logging

logger = logging.getLogger(__name__)

_pool: Optional[SimpleConnectionPool] = None

import os


DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("PIPELINE_DB", "strata"),  # ✅ FIX
    "user": os.getenv("DB_USER", "trading_user"),
    "password": os.getenv("DB_PASS"),
}
if DB_CONFIG["password"] is None:
    raise RuntimeError(
        "DB_PASS is not set in the environment. "
        "Database connection cannot be initialized."
    )



def init_pool(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    minconn: int = 1,
    maxconn: int = 10
) -> None:
    """
    Initialize database connection pool.

    Args:
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password
        minconn: Minimum number of connections in pool
        maxconn: Maximum number of connections in pool
    """
    global _pool
    if _pool is None:
        _pool = SimpleConnectionPool(
            minconn=minconn,
            maxconn=maxconn,
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        logger.info(f"Database connection pool initialized: {user}@{host}:{port}/{database}")
    else:
        logger.warning("Connection pool already initialized")


def close_pool() -> None:
    """Close all connections in the pool."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        logger.info("Database connection pool closed")


@contextmanager
def get_connection():
    """
    Get database connection from pool with automatic cleanup.

    Yields:
        psycopg2.connection: Database connection

    Raises:
        RuntimeError: If pool is not initialized
    """
    if _pool is None:
        raise RuntimeError(
            "Connection pool not initialized. Call init_pool() first."
        )

    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


@contextmanager
def get_cursor(cursor_factory=RealDictCursor):
    """
    Get cursor with automatic connection management.

    Args:
        cursor_factory: Cursor factory class (default: RealDictCursor)

    Yields:
        psycopg2.cursor: Database cursor

    Example:
        >>> with get_cursor() as cursor:
        ...     cursor.execute("SELECT * FROM market_state LIMIT 1")
        ...     result = cursor.fetchone()
    """
    with get_connection() as conn:
        cursor = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cursor
        finally:
            cursor.close()
