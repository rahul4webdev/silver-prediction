#!/usr/bin/env python3
"""
Standalone tick collector worker.
Runs continuously to collect real-time tick data from Upstox WebSocket.

Usage:
    python -m workers.tick_collector_worker

This script should be run as a systemd service for production.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.models.database import init_db
from app.services.tick_collector import tick_collector
from app.services.upstox_client import upstox_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/var/log/tick-collector.log") if Path("/var/log").exists() else logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class TickCollectorWorker:
    """
    Worker that manages the tick collector lifecycle.

    Features:
    - Graceful shutdown on SIGTERM/SIGINT
    - Auto-restart on errors
    - Market hours awareness (only collects during trading hours)
    - Periodic aggregation of ticks
    """

    # MCX trading hours (IST)
    MCX_OPEN_HOUR = 9  # 9:00 AM IST
    MCX_CLOSE_HOUR = 23  # 11:55 PM IST
    MCX_CLOSE_MINUTE = 55

    def __init__(self):
        self._running = False
        self._shutdown_event = asyncio.Event()

    def _is_market_hours(self) -> bool:
        """Check if current time is within MCX trading hours."""
        # Get current IST time
        now_utc = datetime.now(timezone.utc)
        ist_offset = 5.5  # IST is UTC+5:30
        now_ist_hour = (now_utc.hour + int(ist_offset)) % 24
        now_ist_minute = now_utc.minute + int((ist_offset % 1) * 60)

        if now_ist_minute >= 60:
            now_ist_hour = (now_ist_hour + 1) % 24
            now_ist_minute -= 60

        # Check if within trading hours
        if now_ist_hour >= self.MCX_OPEN_HOUR and now_ist_hour < self.MCX_CLOSE_HOUR:
            return True
        if now_ist_hour == self.MCX_CLOSE_HOUR and now_ist_minute <= self.MCX_CLOSE_MINUTE:
            return True

        return False

    def _time_until_market_open(self) -> int:
        """Get seconds until market opens."""
        now_utc = datetime.now(timezone.utc)
        ist_offset = 5.5

        # Calculate IST time
        now_ist_hour = (now_utc.hour + int(ist_offset)) % 24
        now_ist_minute = now_utc.minute + int((ist_offset % 1) * 60)

        if now_ist_minute >= 60:
            now_ist_hour = (now_ist_hour + 1) % 24
            now_ist_minute -= 60

        # Calculate time until 9 AM IST
        if now_ist_hour >= self.MCX_CLOSE_HOUR:
            # After market close, wait until tomorrow
            hours_until = 24 - now_ist_hour + self.MCX_OPEN_HOUR
        else:
            hours_until = self.MCX_OPEN_HOUR - now_ist_hour

        minutes_until = -now_ist_minute
        seconds_until = hours_until * 3600 + minutes_until * 60

        return max(seconds_until, 60)  # Minimum 1 minute

    async def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown)

    def _handle_shutdown(self):
        """Handle shutdown signal."""
        logger.info("Received shutdown signal")
        self._running = False
        self._shutdown_event.set()

    async def _check_auth(self) -> bool:
        """Check if Upstox authentication is valid."""
        if not upstox_client.is_authenticated:
            logger.warning("Upstox not authenticated - token not set")
            return False

        try:
            auth_status = await upstox_client.verify_authentication()
            if not auth_status.get("authenticated"):
                logger.warning(f"Upstox token invalid: {auth_status.get('reason')}")
                return False
            return True
        except Exception as e:
            logger.error(f"Auth check failed: {e}")
            return False

    async def _run_aggregation(self):
        """Periodically aggregate tick data."""
        while self._running:
            # Wait 5 minutes between aggregations
            await asyncio.sleep(300)

            if not self._running:
                break

            try:
                # Aggregate at different intervals
                for interval in ["1s", "5s", "10s", "1m"]:
                    count = await tick_collector.aggregate_ticks(interval)
                    if count > 0:
                        logger.info(f"Created {count} {interval} aggregated candles")
            except Exception as e:
                logger.error(f"Aggregation failed: {e}")

    async def run(self):
        """Main worker loop."""
        logger.info("=" * 60)
        logger.info("Starting Tick Collector Worker")
        logger.info(f"Environment: {settings.environment}")
        logger.info("=" * 60)

        # Initialize database
        try:
            await init_db()
            logger.info("Database initialized")
        except Exception as e:
            logger.error(f"Database init failed: {e}")
            return

        # Set up signal handlers
        await self._setup_signal_handlers()

        self._running = True
        aggregation_task = None

        while self._running:
            try:
                # Check authentication
                if not await self._check_auth():
                    logger.warning("Waiting for valid authentication...")
                    await asyncio.sleep(60)
                    continue

                # Check market hours
                if not self._is_market_hours():
                    wait_time = self._time_until_market_open()
                    logger.info(f"Market closed. Waiting {wait_time // 60} minutes until market opens...")
                    await asyncio.sleep(min(wait_time, 300))  # Check every 5 min max
                    continue

                logger.info("Market is open - starting tick collection")

                # Start aggregation task
                if aggregation_task is None or aggregation_task.done():
                    aggregation_task = asyncio.create_task(self._run_aggregation())

                # Run tick collector
                await tick_collector.start()

            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(30)

        # Cleanup
        logger.info("Shutting down worker...")
        await tick_collector.stop()

        if aggregation_task and not aggregation_task.done():
            aggregation_task.cancel()
            try:
                await aggregation_task
            except asyncio.CancelledError:
                pass

        logger.info("Worker stopped")


async def main():
    """Entry point."""
    worker = TickCollectorWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
