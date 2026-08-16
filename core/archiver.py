"""
Archiver: handles week transitions and archival of tile maps.
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional

from config import WEEK_CHECK_INTERVAL, get_current_week
from core.tile_manager import TileManager
from core.downloader import TileDownloader
from database.db import get_connection

logger = logging.getLogger(__name__)


class Archiver:
    """
    Manages week transitions:
    - Checks periodically if the ISO week has changed.
    - On change: seals the old map (with yellow preserved) into the archive,
      resets the in-memory map to empty, saves a blank snapshot for the new
      week, then re-queues all previously-filled tiles for re-download so
      changes can be detected.
    - Handles catching up if the server was down for multiple weeks.
    """

    def __init__(self, tile_manager: TileManager, downloader: TileDownloader):
        self.tile_manager = tile_manager
        self.downloader = downloader
        self.current_week: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """Start the archiver loop."""
        self._running = True
        self.current_week = get_current_week()
        await self._catch_up()
        self._task = asyncio.create_task(self._archiver_loop())
        logger.info(f"Archiver started, current week: {self.current_week}")

    async def stop(self):
        """Stop the archiver."""
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("Archiver stopped")

    async def _queue_historical_tiles(self, current_week: str):
        """
        Queue every tile that was ever filled (according to the `tiles`
        table), EXCEPT those belonging to the current week, so they get
        re-downloaded to detect changes.

        The `tiles` table stores a row per (x, y, week_key) only when a tile
        is new or changed in that week, but a tile's first appearance is always
        recorded. Therefore SELECT DISTINCT x, y across all weeks captures
        every tile that was ever filled — including ones that have been
        unchanged for many weeks. Excluding the current week avoids
        re-queueing tiles that are already present in the loaded current-week
        map.

        Called at startup (so an interrupted scan is resumed on restart) and
        on every week transition.
        """
        conn = get_connection()
        rows = conn.execute(
            "SELECT DISTINCT x, y FROM tiles WHERE week_key != ?",
            (current_week,),
        ).fetchall()
        for row in rows:
            self.downloader.add_to_list_force(row["x"], row["y"])
        logger.info(
            f"Queued {len(rows)} historical tiles for re-download "
            f"(excluding current week {current_week})"
        )

    async def _catch_up(self):
        """
        Called once at startup. Three cases:

        A) Current week map exists in DB → load it and restore tile statuses.
        B) Older weeks exist but not current → seal the latest old week
           (archive=True preserves yellow, strips empty noise), reset map,
           save blank snapshot for the new week.
        C) Nothing in DB at all → just create blank snapshot for current week.

        In all cases we then re-queue every historically-filled tile so a
        scan interrupted by a crash/restart is resumed.
        """
        current = self.current_week
        if current is None:
            logger.warning("Cannot catch up: current_week is None")
            return

        try:
            conn = get_connection()
            rows = conn.execute("SELECT week_key FROM maps").fetchall()
            existing_weeks = {str(row["week_key"]) for row in rows}

            # --- Case A: current week already exists ---
            if current in existing_weeks:
                logger.info(f"Loading existing map for week {current}")
                found = self.tile_manager.load_from_db(current)
                if found:
                    self.downloader.set_current_week(current)
                await self._queue_historical_tiles(current)
                return

            # --- Cases B / C ---
            old_weeks = sorted(w for w in existing_weeks if w < current)

            if old_weeks:
                latest_old = old_weeks[-1]
                logger.info(f"Loading old week {latest_old} for finalization")
                found = self.tile_manager.load_from_db(latest_old)
                if found:
                    # Seal: yellow preserved in archive, empty stripped.
                    self.tile_manager.save_to_db(latest_old, archive=True)
                    logger.info(f"Finalized archive for week {latest_old}")

            # Reset to empty BEFORE saving the new-week snapshot so the DB
            # record is a clean blank, not a copy of the old week.
            self.tile_manager.reset_to_empty()
            self.tile_manager.save_to_db(current, archive=False)
            self.downloader.set_current_week(current)
            logger.info(f"Created blank snapshot for new week {current}")

            await self._queue_historical_tiles(current)

        except Exception as e:
            logger.error(f"Catch-up failed: {e}", exc_info=True)


    async def _archiver_loop(self):
        """Periodically check if the week has changed."""
        while self._running:
            try:
                await asyncio.sleep(WEEK_CHECK_INTERVAL)
                await self._check_week_change()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in archiver loop: {e}")

    async def _check_week_change(self):
        """Detect ISO week rollover and handle the transition."""
        new_week = get_current_week()

        if self.current_week is None:
            self.current_week = new_week
            return

        if new_week == self.current_week:
            return

        logger.info(f"Week changed: {self.current_week} -> {new_week}")
        old_week = self.current_week

        # 1. Seal the old week: keep yellow to record change history,
        #    strip empty-tile noise.
        self.tile_manager.save_to_db(old_week, archive=True)
        logger.info(f"Sealed archive for week {old_week}")

        # 2. Reset in-memory map to empty so:
        #    a) the new-week DB snapshot is genuinely blank, and
        #    b) is_downloaded() returns False for all tiles, allowing the
        #       download loop to actually re-fetch them.
        self.tile_manager.reset_to_empty()

        # 3. Save blank snapshot for the new week.
        self.current_week = new_week
        self.tile_manager.save_to_db(new_week, archive=False)
        self.downloader.set_current_week(new_week)

        # 4. Re-queue all tiles that have ever been filled (except the new
        #    current week, already blank) so changes are detected.
        await self._queue_historical_tiles(new_week)

        logger.info(
            f"Week transition complete. Historical tiles queued for re-download."
        )