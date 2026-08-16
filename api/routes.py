import asyncio
import io
import logging
import os
import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, HTMLResponse

from config import MAP_SIZE, MAX_REGION_TILES, REGION_RATE_LIMIT
from core.tile_manager import TileManager
from core.downloader import TileDownloader
from core.image_utils import composite_tiles_to_png
from database.db import get_connection

logger = logging.getLogger(__name__)

router = APIRouter()

tile_manager: Optional[TileManager] = None
downloader: Optional[TileDownloader] = None

MAP_CACHE_TTL    = 5.0
_cached_map_png: Optional[bytes] = None
_cache_expires_at: float = 0.0
_map_lock        = asyncio.Lock()
_last_region_time: float = 0.0

_STATIC_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
_HTML_PATH    = os.path.join(_STATIC_DIR, "map_viewer.html")
_FAVICON_PATH = os.path.join(_STATIC_DIR, "favicon.ico")
_ROBOTS_PATH  = os.path.join(_STATIC_DIR, "robots.txt")


def setup_routes(tm: TileManager, dl: TileDownloader):
    global tile_manager, downloader
    tile_manager = tm
    downloader   = dl


@router.get("/")
async def get_frontend():
    try:
        with open(_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving frontend: {e}")


@router.get("/test")
async def get_test_page():
    test_html = os.path.join(_STATIC_DIR, "region_test.html")
    try:
        with open(test_html, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Test page not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving test page: {e}")


@router.get("/favicon.ico")
async def get_favicon():
    try:
        with open(_FAVICON_PATH, "rb") as f:
            return Response(content=f.read(), media_type="image/x-icon")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Favicon not found")


@router.get("/robots.txt")
async def get_robots():
    try:
        with open(_ROBOTS_PATH, "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="text/plain")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="robots.txt not found")


@router.get("/tile/{x}/{y}")
async def get_tile(x: int, y: int):
    """Current tile. Returns 204 if not downloaded or empty."""
    if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
        raise HTTPException(status_code=404, detail="Coordinates out of bounds")
    tm = tile_manager
    if tm is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    if tm.get_tile_status(x, y) >= 2:
        week = tm.week_key
        if week:
            row = get_connection().execute(
                "SELECT image, checksum FROM tiles WHERE x=? AND y=? AND week_key<=?"
                " ORDER BY week_key DESC LIMIT 1",
                (x, y, week),
            ).fetchone()
            if row:
                return Response(
                    content=row["image"],
                    media_type="image/png",
                    headers={
                        # Current week's tiles can change as new downloads
                        # land throughout the week — keep this short, in
                        # line with the client's live-refresh interval, so
                        # browsers/proxies don't serve a stale tile for long.
                        "Cache-Control": "public, max-age=60",
                        "ETag": f'"{row["checksum"]}"',
                    }
                )
    return Response(status_code=204)


@router.get("/tile/{x}/{y}/{week}")
async def get_archive_tile(x: int, y: int, week: str):
    """
    Tile as it appeared during (or most recently before) a given week.
    Falls back to earlier weeks when the tile wasn't re-downloaded that week.
    """
    if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
        raise HTTPException(status_code=404, detail="Coordinates out of bounds")
    row = get_connection().execute(
        "SELECT image, checksum FROM tiles WHERE x=? AND y=? AND week_key<=?"
        " ORDER BY week_key DESC LIMIT 1",
        (x, y, week),
    ).fetchone()
    if row:
        return Response(
            content=row["image"],
            media_type="image/png",
            headers={
                # Archive weeks are immutable — this exact (x, y, week)
                # will never resolve to different bytes again, so it's safe
                # to let browsers cache it "forever" and skip revalidation
                # entirely.
                "Cache-Control": "public, max-age=31536000, immutable",
                "ETag": f'"{row["checksum"]}"',
            }
        )
    return Response(status_code=204)


@router.post("/scan/{x}/{y}")
async def request_tile_scan(x: int, y: int):
    """
    Request immediate download of a specific tile.

    Adds the tile (and its neighbours) to the front of the download list.
    Ignored silently if the tile is already downloaded — check the map image
    first (gray = never fetched, anything else = already known).

    If the download queue has more than 50 pending tiles, the request is
    rejected with a 429 status to prevent overwhelming the queue.

    Returns whether the tile was queued or skipped.
    """
    if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
        raise HTTPException(status_code=404, detail="Coordinates out of bounds")
    tm = tile_manager
    dl = downloader
    if tm is None or dl is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    if tm.is_downloaded(x, y):
        return {"queued": False, "reason": "already downloaded"}

    if dl.queue_size > 50:
        raise HTTPException(
            status_code=429,
            detail=f"Queue is full ({dl.queue_size} pending). Try again later.",
        )

    dl.add_to_list_force(x, y)
    return {"queued": True, "x": x, "y": y}


@router.get("/health")
async def health_check():
    tm = tile_manager
    dl = downloader

    scanned = tm.get_scanned_count() if tm else 0
    filled = tm.get_filled_count() if tm else 0
    total = MAP_SIZE * MAP_SIZE

    return {
        "status": "ok" if tm is not None else "degraded",
        "queue_size": dl.queue_size if dl else 0,
        "scanned": scanned,
        "filled": filled,
        "coverage": round(scanned * 100 / total, 2),
        "fill_ratio": round(filled * 100 / total, 2),
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/map")
async def get_current_map():
    """
    Current map as a 4096×4096 8-bit palette PNG.
    Cached for MAP_CACHE_TTL seconds. Encoded in a thread to avoid blocking.
    """
    global _cached_map_png, _cache_expires_at
    tm = tile_manager
    if tm is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    now = time.monotonic()
    if _cached_map_png is not None and now < _cache_expires_at:
        return Response(
            content=_cached_map_png,
            media_type="image/png",
            headers={"Cache-Control": f"public, max-age={int(MAP_CACHE_TTL)}"},
        )
    async with _map_lock:
        now = time.monotonic()
        if _cached_map_png is not None and now < _cache_expires_at:
            return Response(
                content=_cached_map_png,
                media_type="image/png",
                headers={"Cache-Control": f"public, max-age={int(MAP_CACHE_TTL)}"},
            )
        try:
            png = await asyncio.get_event_loop().run_in_executor(
                None, tm.to_png_bytes, 1
            )
        except Exception as e:
            logger.error(f"Failed to generate map PNG: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate map")
        _cached_map_png = png
        _cache_expires_at = time.monotonic() + MAP_CACHE_TTL
    return Response(
        content=_cached_map_png,
        media_type="image/png",
        headers={"Cache-Control": f"public, max-age={int(MAP_CACHE_TTL)}"},
    )


@router.get("/map/{week}")
async def get_archive_map(week: str):
    """Archived map for a specific week."""
    row = get_connection().execute(
        "SELECT image FROM maps WHERE week_key=?", (week,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Week not found")
    return Response(
        content=row["image"],
        media_type="image/png",
        # Archive weeks are immutable, same reasoning as /tile/{x}/{y}/{week}.
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/weeks")
async def get_weeks():
    """List of all available archive weeks."""
    rows = get_connection().execute(
        "SELECT week_key FROM maps ORDER BY week_key"
    ).fetchall()
    return [row["week_key"] for row in rows]


@router.get("/region/{x1}/{y1}/{x2}/{y2}")
async def get_region(x1: int, y1: int, x2: int, y2: int, request: Request):
    """
    Composite image of a rectangular tile region.
    Max 128 tiles. Rate-limited. Optional ?week= param for archive.
    """
    global _last_region_time
    now = datetime.now().timestamp()
    if now - _last_region_time < REGION_RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limited. Try again in {int(REGION_RATE_LIMIT - (now - _last_region_time))}s.",
        )
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if not (0 <= x1 < MAP_SIZE and 0 <= x2 < MAP_SIZE and
            0 <= y1 < MAP_SIZE and 0 <= y2 < MAP_SIZE):
        raise HTTPException(status_code=404, detail="Coordinates out of bounds")
    num_tiles = (x2 - x1 + 1) * (y2 - y1 + 1)
    if num_tiles > MAX_REGION_TILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many tiles ({num_tiles}). Max {MAX_REGION_TILES}.",
        )
    _last_region_time = now
    tm = tile_manager
    if tm is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    week = request.query_params.get("week") or tm.week_key
    tile_map: dict = {}
    if week:
        # For each tile in the region return its state as of `week`:
        # the most recent row with week_key <= week.
        # This matches get_archive_tile behaviour — tiles that were not
        # re-downloaded in a given week still appear with their last known image.
        rows = get_connection().execute(
            "SELECT x, y, image FROM tiles t"
            " WHERE x BETWEEN ? AND ? AND y BETWEEN ? AND ?"
            "   AND week_key = ("
            "       SELECT MAX(week_key) FROM tiles"
            "       WHERE x = t.x AND y = t.y AND week_key <= ?"
            "   )",
            (x1, x2, y1, y2, week),
        ).fetchall()
        tile_map = {(row["x"], row["y"]): bytes(row["image"]) for row in rows}

    return Response(
        content=composite_tiles_to_png(tile_map, x1, y1, x2, y2),
        media_type="image/png",
    )
