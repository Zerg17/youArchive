"""
Asynchronous tile downloader.
Downloads tiles from youplace.live with rate limiting and backoff.
"""
import asyncio
import logging
from typing import Optional, List, Tuple

import aiohttp
import numpy as np

from config import (
    BASE_URL, DOWNLOAD_INTERVAL, MAX_BACKOFF,
    MAP_SIZE, TILE_SIZE, INTEREST_ZONES
)
from core.tile_manager import TileManager
from core.image_utils import check_tile_from_download, compress_tile
from database.db import get_connection

logger = logging.getLogger(__name__)


def _write_gone_tombstone(x: int, y: int, week: str):
    """
    Record a tile's disappearance directly in `tiles` as an empty row
    (image=b'', checksum='') — a "tombstone". Once this exists, any read
    path that already falls back to "most recent row <= requested week"
    will naturally return it, and an empty/falsy image is enough signal to
    treat the tile as gone. No separate table, no need to consult the
    week's map PNG at read time.

    Only called once, exactly at the (x, y) transition where the tile was
    filled last week and is confirmed empty this week — not on every empty
    week thereafter (was_filled_last_week() already returns False for
    subsequent weeks once the tombstone is the most recent row, so this
    naturally doesn't get re-triggered).
    """
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO tiles (x, y, week_key, image, checksum)"
        " VALUES (?, ?, ?, ?, ?)",
        (x, y, week, b"", ""),
    )
    conn.commit()


class TileDownloader:
    """
    Background tile downloader.

    Maintains a priority list of tiles to download.  Each iteration of the
    download loop picks the first tile from the list; if the list is empty it
    falls back to a random unloaded tile so new painted areas are discovered
    over time.
    """

    def __init__(self, tile_manager: TileManager):
        self.tile_manager = tile_manager
        self._list: List[Tuple[int, int]] = []
        self._in_list: set = set()
        self._lock = asyncio.Lock()
        self._running = False
        self._backoff = DOWNLOAD_INTERVAL
        self._session: Optional[aiohttp.ClientSession] = None
        self._task: Optional[asyncio.Task] = None
        self._current_week: Optional[str] = None
        self._scan_iter = None   # coarse-to-fine scan iterator

    @property
    def queue_size(self) -> int:
        """Return the number of tiles currently waiting in the queue."""
        return len(self._list)

    def set_current_week(self, week: str):
        self._current_week = week
        self._scan_iter = self.tile_manager.make_scan_iterator(week)

    async def start(self):
        self._running = True
        self._session = aiohttp.ClientSession()
        for x, y in INTEREST_ZONES:
            self.add_to_list(x, y)
        self._task = asyncio.create_task(self._download_loop())
        logger.info("Downloader started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        if self._session:
            await self._session.close()
        logger.info("Downloader stopped")


    def add_to_list(self, x: int, y: int):
        """Append a tile to the download list (skip if already present or downloaded)."""
        key = (x, y)
        if key not in self._in_list and not self.tile_manager.is_downloaded(x, y):
            self._in_list.add(key)
            self._list.append(key)

    def add_to_list_force(self, x: int, y: int):
        """Prepend a tile to the download list, bypassing the is_downloaded check.

        Used during week transitions when tiles must be re-fetched even though
        they are already marked as downloaded in the previous week's map.
        """
        key = (x, y)
        if key not in self._in_list:
            self._in_list.add(key)
            self._list.insert(0, key)


    async def _download_loop(self):
        """
        Every DOWNLOAD_INTERVAL seconds: pick the next tile and download it.

        Priority order:
          1. First item in self._list  (tiles added explicitly, e.g. week transition
             or neighbors of a just-discovered filled tile)
          2. A random unloaded tile from get_random_unloaded()  (discovery)

        If neither source has a tile, sleep and retry.
        """
        while self._running:
            try:
                # Pick next tile
                x = y = None
                while self._list:
                    cx, cy = self._list.pop(0)
                    self._in_list.discard((cx, cy))
                    if not self.tile_manager.is_downloaded(cx, cy):
                        x, y = cx, cy
                        break
                if x is None:
                    # List empty or all entries already downloaded — use scan
                    if self._scan_iter is None and self._current_week:
                        self._scan_iter = self.tile_manager.make_scan_iterator(
                            self._current_week
                        )
                    tile = next(self._scan_iter, None) if self._scan_iter else None
                    if tile is None:
                        await asyncio.sleep(self._backoff)
                        continue
                    x, y = tile

                await self._download_tile(x, y)
                await asyncio.sleep(self._backoff)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in download loop: {e}")
                await asyncio.sleep(1)


    async def _download_tile(self, x: int, y: int):
        url = f"{BASE_URL}/{x}/{y}"
        logger.debug(f"Downloading tile {x},{y}")

        try:
            async with self._session.get(
                url, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.read()
                    await self._process_tile_data(x, y, data)
                    self._backoff = DOWNLOAD_INTERVAL

                elif response.status in (429, 503, 502, 500):
                    logger.warning(
                        f"Server error {response.status} for tile {x},{y}, backing off"
                    )
                    self._backoff = min(self._backoff * 2, MAX_BACKOFF)
                    self.add_to_list(x, y)

                elif response.status == 404:
                    logger.debug(f"Tile {x},{y} not found (404)")
                    was_filled_prev = self.tile_manager.was_filled_last_week(x, y)
                    self.tile_manager.set_tile_empty(x, y, was_filled_prev=was_filled_prev)
                    if was_filled_prev and self._current_week:
                        _write_gone_tombstone(x, y, self._current_week)
                    self._backoff = DOWNLOAD_INTERVAL

                else:
                    logger.warning(f"Unexpected status {response.status} for {x},{y}")
                    self._backoff = min(self._backoff * 2, MAX_BACKOFF)

        except asyncio.TimeoutError:
            logger.warning(f"Timeout downloading tile {x},{y}")
            self._backoff = min(self._backoff * 2, MAX_BACKOFF)
            self.add_to_list(x, y)

        except aiohttp.ClientError as e:
            logger.error(f"Connection error for tile {x},{y}: {e}")
            self._backoff = min(self._backoff * 2, MAX_BACKOFF)
            self.add_to_list(x, y)

    async def _process_tile_data(self, x: int, y: int, data: bytes):
        """
        Process a downloaded tile.
        Heavy PIL/numpy/SQLite work is offloaded to a thread executor
        so the event loop stays responsive to HTTP requests.
        """
        loop = asyncio.get_event_loop()
        week = self._current_week
        if week is None:
            logger.error("Current week not set, cannot store tile")
            return

        processed = await loop.run_in_executor(
            None, _process_tile_sync, self.tile_manager, x, y, data, week
        )

        if processed is None:
            logger.warning(f"Failed to process tile {x},{y}")
            return

        is_empty, fill_pct, filled_pixels, changed, was_filled_prev, neighbors = processed

        if is_empty:
            self.tile_manager.set_tile_empty(x, y, was_filled_prev=was_filled_prev)
            if was_filled_prev:
                _write_gone_tombstone(x, y, week)
            logger.debug(f"Tile {x},{y} is empty (gone:{was_filled_prev})")
        else:
            if not was_filled_prev:
                status = "new"
            elif changed:
                status = "changed"
            else:
                status = "unchanged"
            self.tile_manager.set_tile_filled(x, y, fill_pct=fill_pct, status=status)
            logger.info(
                f"Downloaded tile {x},{y} (fill:{fill_pct}% / {filled_pixels}px "
                f"status:{status})"
            )
            # Queue neighbors so clusters are fully covered — back on main thread
            for nx, ny in neighbors:
                self.add_to_list(nx, ny)


def _process_tile_sync(
    tile_manager: "TileManager", x: int, y: int, data: bytes, week: str
) -> Optional[tuple]:
    """
    Synchronous tile processing — runs in a thread executor.
    Returns (is_empty, fill_pct, filled_pixels, changed, was_filled_prev,
    neighbor_list) or None on error.
    """
    try:
        is_empty, img, checksum = check_tile_from_download(x, y, data)
    except Exception:
        return None

    was_filled_prev = tile_manager.was_filled_last_week(x, y)

    if is_empty:
        return (True, 0, 0, False, was_filled_prev, [])

    if img is None:
        return None

    import numpy as np

    alpha_arr = np.array(img.convert("RGBA").getchannel("A"))
    filled_pixels = int(np.count_nonzero(alpha_arr))
    fill_pct = max(1, filled_pixels * 100 // (TILE_SIZE * TILE_SIZE))

    # Build neighbours first — always scan a 5×5 window around a found tile
    # so new content that appeared next to unchanged tiles is still discovered.
    neighbors = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
                neighbors.append((nx, ny))

    conn = get_connection()

    # Changed relative to last known version? Note: if the tile just
    # reappeared after being gone, `prev` here may be the empty tombstone
    # row written by _write_gone_tombstone() — its checksum ('') will never
    # match a real image's checksum, so this naturally reports changed=True
    # and a fresh row gets inserted below. That's fine either way: the
    # actual status shown on the map ("new" vs "changed") is decided by
    # was_filled_prev in _process_tile_data, not by this flag, whenever the
    # tile wasn't filled last week.
    changed = False
    prev_row = conn.execute(
        "SELECT week_key FROM tiles"
        " WHERE x=? AND y=? AND week_key<?"
        " ORDER BY week_key DESC LIMIT 1",
        (x, y, week),
    ).fetchone()
    if prev_row:
        prev = conn.execute(
            "SELECT checksum FROM tiles WHERE x=? AND y=? AND week_key=?",
            (x, y, str(prev_row["week_key"])),
        ).fetchone()
        if prev and str(prev["checksum"]) != checksum:
            changed = True
        elif prev:
            # Same as previous week — no need to store a copy in the DB,
            # but neighbours are still needed to discover new content.
            return (False, fill_pct, filled_pixels, False, was_filled_prev, neighbors)

    compressed = compress_tile(img)
    conn.execute(
        "INSERT OR REPLACE INTO tiles (x, y, week_key, image, checksum)"
        " VALUES (?, ?, ?, ?, ?)",
        (x, y, week, compressed, checksum),
    )
    conn.commit()

    return (False, fill_pct, filled_pixels, changed, was_filled_prev, neighbors)