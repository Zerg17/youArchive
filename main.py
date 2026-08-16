"""
youArchive - Tile archive server for youplace.live
"""
import asyncio
import logging

from fastapi import FastAPI

from config import HOST, PORT
from database.db import init_db, close_db
from core.tile_manager import TileManager
from core.downloader import TileDownloader
from core.archiver import Archiver
from api.routes import router, setup_routes


class UvicornErrorFilter(logging.Filter):
    """
    Фильтр для подавления сообщений о невалидных HTTP-запросах
    (без заголовка Host), которые приходят от ботов/сканеров.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if "Missing mandatory Host: header" in record.getMessage():
            return False
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

uvicorn_error_logger = logging.getLogger("uvicorn.error")
uvicorn_error_logger.addFilter(UvicornErrorFilter())

tile_manager = TileManager()
downloader = TileDownloader(tile_manager)
archiver = Archiver(tile_manager, downloader)

setup_routes(tile_manager, downloader)
logger.info("Routes initialized at module level")


app = FastAPI(
    title="youArchive",
    description="Tile archive server for youplace.live",
    version="1.0.0",
)

app.include_router(router)


@app.on_event("startup")
async def startup():
    """Startup logic - runs when the server starts accepting requests."""
    logger.info("Starting youArchive server...")

    try:
        init_db()
        logger.info("Database initialized")

        await archiver.start()
        logger.info(f"Archiver started, current week: {archiver.current_week}")

        await downloader.start()
        logger.info("Downloader started")

        logger.info("Server startup complete")

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)


@app.on_event("shutdown")
async def shutdown():
    """Shutdown logic."""
    logger.info("Shutting down...")

    try:
        if archiver.current_week:
            tile_manager.save_to_db(archiver.current_week)
            logger.info(f"Saved map for week {archiver.current_week}")
    except Exception as e:
        logger.error(f"Error saving map on shutdown: {e}")

    try:
        await downloader.stop()
    except Exception as e:
        logger.error(f"Error stopping downloader: {e}")

    try:
        await archiver.stop()
    except Exception as e:
        logger.error(f"Error stopping archiver: {e}")

    try:
        close_db()
    except Exception as e:
        logger.error(f"Error closing database: {e}")

    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
        loop="asyncio",
    )