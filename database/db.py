"""
SQLite database initialization and connection management.
"""
import os
import sqlite3
import threading
from typing import Optional

from config import DB_PATH


_local = threading.local()


def get_connection() -> sqlite3.Connection:
    """Get a thread-local database connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return _local.conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tiles (
            x INTEGER NOT NULL,
            y INTEGER NOT NULL,
            week_key TEXT NOT NULL,
            image BLOB NOT NULL,
            checksum TEXT NOT NULL,
            PRIMARY KEY (x, y, week_key)
        );

        CREATE TABLE IF NOT EXISTS maps (
            week_key TEXT PRIMARY KEY,
            image BLOB NOT NULL,
            created_at TEXT NOT NULL
        );


        -- Composite index covers all three query patterns:
        --   get_tile:         WHERE x=? AND y=? AND week_key=?
        --   get_archive_tile: WHERE x=? AND y=? AND week_key<=? ORDER BY week_key DESC
        --   get_region:       WHERE x BETWEEN ? AND ? AND y BETWEEN ? AND ? AND week_key<=?
        -- A single (x, y, week_key) index satisfies all of them.
        CREATE INDEX IF NOT EXISTS idx_tiles_xyz ON tiles(x, y, week_key);
    """)
    conn.commit()


def close_db():
    """Close the connection for the current thread."""
    if hasattr(_local, "conn") and _local.conn is not None:
        _local.conn.close()
        _local.conn = None