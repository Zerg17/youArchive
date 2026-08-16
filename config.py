"""
Configuration for youArchive server.
"""
import os
from datetime import datetime

# Server
HOST = "0.0.0.0"
PORT = 8080

# Tile dimensions
TILE_SIZE = 1024
MAP_SIZE = 4096  # tiles per axis (4096x4096 tiles)
MAX_REGION_TILES = 128  # max tiles in a region request

# Download settings
BASE_URL = "https://backend.youplace.live/tile/px"
DOWNLOAD_INTERVAL = 0.7  # seconds between downloads
MAX_BACKOFF = 120.0  # max backoff time in seconds

INTEREST_ZONES = []


# Database
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "youarchive.db")

# Archiver
WEEK_CHECK_INTERVAL = 600  # check week change every 10 minutes

# Region request rate limit (seconds)
REGION_RATE_LIMIT = 60.0

def get_current_week() -> str:
    """Return current week key like '26W25'."""
    iso = datetime.now().isocalendar()
    year_short = iso[0] % 100
    return f"{year_short}W{iso[1]:02d}"