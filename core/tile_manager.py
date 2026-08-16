"""
Tile manager: manages the current tile map in memory (numpy array),
synchronizes with the database.
"""
import io
from typing import Optional, Set, Tuple, List

import numpy as np
from PIL import Image

from config import MAP_SIZE
from database.db import get_connection


UNLOADED = 0     # gray    — never downloaded / not (yet) rescanned this week
EMPTY    = 1     # black   — downloaded and confirmed empty (current week only;
                 #           stripped to UNLOADED when a week is archived)
# 2..102 = green   (filled, unchanged since last time seen — fill_pct 0-100%)
YELLOW   = 103   # bright yellow — filled, content changed vs. previous version
NEW      = 104   # malinovy      — filled this week, was NOT filled last week
                 #                 (first appearance, or reappearance after
                 #                 having disappeared)
GONE     = 105   # red           — was filled last week, confirmed empty/gone
                 #                 this week (a "disappeared" transition)

# Values that represent "this tile currently has content" for the purposes of
# week-over-week comparisons (used to decide NEW vs "still there" and to
# decide GONE for the following week).
FILLED_VALUES = frozenset([YELLOW, NEW]) | frozenset(range(2, 103))


def fill_green(fill_pct: int) -> int:
    return 2 + min(max(fill_pct, 0), 100)


GRAD_MIN = 40   # intensity at fill_pct = 1  (clearly visible vs black)
GRAD_MAX = 230  # intensity at fill_pct = 100

# malinovy (new) / red (gone) — moderate-contrast palette
_COLOR_YELLOW = (255, 230, 0)   # bright, no gradient — must "jump out"
_COLOR_NEW    = (219, 39, 119)  # malinovy — appeared this week
_COLOR_GONE   = (211, 47, 47)   # red — disappeared this week

_PALETTE_FLAT = np.zeros(256 * 3, dtype=np.uint8)
_PALETTE_FLAT[0 * 3: 0 * 3 + 3] = [128, 128, 128]   # UNLOADED
_PALETTE_FLAT[1 * 3: 1 * 3 + 3] = [0,   0,   0  ]   # EMPTY
for _v in range(2, 103):   # green pct 0..100
    _pct = _v - 2
    _g = 0 if _pct == 0 else GRAD_MIN + _pct * (GRAD_MAX - GRAD_MIN) // 100
    _PALETTE_FLAT[_v * 3: _v * 3 + 3] = [0, _g, 0]
_PALETTE_FLAT[YELLOW * 3: YELLOW * 3 + 3] = _COLOR_YELLOW
_PALETTE_FLAT[NEW * 3: NEW * 3 + 3]       = _COLOR_NEW
_PALETTE_FLAT[GONE * 3: GONE * 3 + 3]     = _COLOR_GONE


def _map_to_palette_png(arr: np.ndarray, compress_level: int) -> bytes:
    """Encode a map array (dtype=uint8) as an 8-bit palette PNG."""
    img = Image.fromarray(arr, "P")
    img.putpalette(_PALETTE_FLAT.tolist())
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=compress_level)
    return buf.getvalue()



class TileManager:
    """
    Manages the current tile map in memory as a numpy array.
    map[y, x] holds a map value (0-202); the value doubles as a palette index.
    """

    def __init__(self):
        self.map: np.ndarray = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        self.week_key: Optional[str] = None
        self._downloaded: Set[Tuple[int, int]] = set()
        # Read-only reference to the *previous* week's sealed map, used only
        # to decide NEW vs "still there" / GONE for tiles downloaded this
        # week. None means "no previous week known" — everything found this
        # week is then treated as NEW.
        self.prev_map: Optional[np.ndarray] = None

    def load_from_db(self, week_key: str) -> bool:
        """
        Load the map for a given week from the database.
        Returns True if found, False if not.
        Maps are stored as 8-bit palette PNGs where pixel value == map value.
        """
        conn = get_connection()
        row = conn.execute(
            "SELECT image FROM maps WHERE week_key = ?", (week_key,)
        ).fetchone()

        if row is None:
            return False

        img = Image.open(io.BytesIO(row["image"]))

        if img.mode != "P":
            raise ValueError(
                f"Map image for week '{week_key}' is mode '{img.mode}', expected 'P'. "
                "Delete the database and restart to rebuild from scratch."
            )

        self.map = np.array(img, dtype=np.uint8)
        self.week_key = week_key
        self._rebuild_downloaded_set()
        return True

    def save_to_db(self, week_key: str, archive: bool = False):
        """
        Save the current map to the database as an 8-bit palette PNG.

        archive=False: compress_level=1 — fast (live map / new-week snapshot).
        archive=True:  compress_level=7 — smaller (week being sealed).
          Also strips EMPTY pixels (value=1 → 0) to remove scan noise, since
          "black" is only meaningful while a week is still live/in-progress.
          Yellow (changed), malinovy (new) and red (gone) pixels are all kept
          so the archived history shows when things changed, appeared, or
          disappeared.
        """
        arr = self.map
        if archive:
            arr = arr.copy()
            arr[arr == EMPTY] = UNLOADED

        from datetime import datetime
        conn = get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO maps (week_key, image, created_at) VALUES (?, ?, ?)",
            (week_key, _map_to_palette_png(arr, 7 if archive else 1),
             datetime.now().isoformat()),
        )
        conn.commit()
        self.week_key = week_key

    def load_prev_map(self, week_key: str) -> bool:
        """
        Load a (typically just-sealed) week's map into self.prev_map, purely
        as a read-only reference for this week's NEW/GONE determination.
        Does NOT touch self.map or self.week_key.
        Returns True if found, False if not (prev_map is set to None then).
        """
        conn = get_connection()
        row = conn.execute(
            "SELECT image FROM maps WHERE week_key = ?", (week_key,)
        ).fetchone()

        if row is None:
            self.prev_map = None
            return False

        img = Image.open(io.BytesIO(row["image"]))
        if img.mode != "P":
            self.prev_map = None
            return False

        self.prev_map = np.array(img, dtype=np.uint8)
        return True

    def was_filled_last_week(self, x: int, y: int) -> bool:
        """
        True if (x, y) held content at the end of the previous week, per
        self.prev_map. If no previous week is known, everything counts as
        "not filled last week" so newly found tiles are marked NEW.
        """
        if self.prev_map is None:
            return False
        if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
            return False
        return int(self.prev_map[y, x]) in FILLED_VALUES

    def reset_to_empty(self):
        """
        Reset the map to fully unloaded state (all zeros).
        Called at the start of a new week so the fresh snapshot is blank
        and the download loop sees every tile as not-downloaded.
        """
        self.map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        self._downloaded.clear()

    def get_tile_status(self, x: int, y: int) -> int:
        if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
            return int(self.map[y, x])
        return UNLOADED

    def set_tile_status(self, x: int, y: int, value: int):
        if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
            self.map[y, x] = value
            if value != UNLOADED:
                self._downloaded.add((x, y))

    def set_tile_filled(self, x: int, y: int, fill_pct: int, status: str = "unchanged"):
        """
        status:
          "unchanged" -> green gradient by fill_pct (default)
          "changed"   -> flat bright yellow (content differs from what was
                         there before, tile was present both weeks)
          "new"       -> flat malinovy (tile was NOT present last week)
        """
        if status == "new":
            value = NEW
        elif status == "changed":
            value = YELLOW
        else:
            value = fill_green(fill_pct)
        self.set_tile_status(x, y, value)

    def set_tile_empty(self, x: int, y: int, was_filled_prev: bool = False):
        """
        was_filled_prev=True marks this as a GONE (red) transition — the tile
        held content last week and is confirmed empty this week. Otherwise it
        is a plain EMPTY (black) scan result.
        """
        self.set_tile_status(x, y, GONE if was_filled_prev else EMPTY)

    def is_downloaded(self, x: int, y: int) -> bool:
        return int(self.map[y, x]) != UNLOADED

    def make_scan_iterator(self, week_key: str):
        """
        Return a generator that yields (x, y) coordinates to probe for new
        content using a Z-order curve (Morton code) with reversed bit order.

        The curve visits points in a coarse-to-fine order: first iteration
        covers the 4 quadrants of the map, then halves each quadrant, etc.
        This gives excellent spatial coverage even if the scan is interrupted
        mid-way — after K iterations the visited points are spread evenly
        across the entire map rather than clustered in one corner.

        MAP_SIZE = 4096 = 2^12, so bits_per_axis = 12 and the full scan
        covers 2^24 = 16,777,216 points (one per tile).

        Week offset: week_num is decoded through a direct Z-order curve to
        produce (offset_x, offset_y) in 1-pixel steps. This distributes
        weekly offsets evenly in 2D: neighbouring weeks get nearby but
        distinct offsets.
        week_num = YY * 53 + WW  (e.g. "26W26" → 1404)
        """
        BITS = 12  # log2(MAP_SIZE) = log2(4096)

        try:
            yy, ww = week_key.split("W")
            week_num = (int(yy) * 53 + int(ww)) % 64
        except Exception:
            week_num = 0

        # Decode week_num through direct Z-order curve to get (offset_x, offset_y).
        # This distributes offsets evenly in 2D space: neighbouring weeks get
        # nearby but distinct offsets.
        ox = oy = 0
        for k in range(BITS):
            ox |= ((week_num >> (2 * k))     & 1) << k
            oy |= ((week_num >> (2 * k + 1)) & 1) << k

        total = 1 << (2 * BITS)   # 16_777_216
        for i in range(total):
            # Deinterleave bits of i into x (even bits) and y (odd bits),
            # with most-significant bits first so coarse positions come first.
            x = y = 0
            for k in range(BITS):
                x |= ((i >> (2 * BITS - 2 - 2 * k)) & 1) << k
                y |= ((i >> (2 * BITS - 1 - 2 * k)) & 1) << k

            sx = (x + ox) % MAP_SIZE
            sy = (y + oy) % MAP_SIZE

            if self.map[sy, sx] == UNLOADED:
                yield (sx, sy)

    def _rebuild_downloaded_set(self):
        mask = self.map != UNLOADED
        ys, xs = np.where(mask)
        self._downloaded = set(zip(xs.tolist(), ys.tolist()))

    def get_scanned_count(self) -> int:
        """Number of tiles that have been scanned (either filled or empty)."""
        return len(self._downloaded)

    def get_filled_count(self) -> int:
        """Number of tiles that currently have non-empty content
        (green, yellow, or malinovy — NOT the red 'gone' marker)."""
        return int(np.count_nonzero((self.map >= 2) & (self.map <= NEW)))

    def to_png_bytes(self, compress_level: int = 1) -> bytes:
        """Encode the current map as an 8-bit palette PNG."""
        return _map_to_palette_png(self.map, compress_level)
