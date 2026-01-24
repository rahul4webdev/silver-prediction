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
import os
import signal
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.models.database import init_db
from app.services.tick_collector import tick_collector
from app.services.upstox_client import upstox_client

# Try to import pytz for proper timezone handling
try:
    import pytz
    IST = pytz.timezone('Asia/Kolkata')
except ImportError:
    IST = None

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

    # MCX trading hours (IST) - Monday to Friday only
    MCX_OPEN_HOUR = 9  # 9:00 AM IST
    MCX_CLOSE_HOUR = 23  # 11:55 PM IST
    MCX_CLOSE_MINUTE = 55

    # MCX is closed on Saturday (5) and Sunday (6)
    MCX_TRADING_DAYS = [0, 1, 2, 3, 4]  # Monday=0 to Friday=4

    def __init__(self):
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._last_token_reload = None

    def _get_ist_now(self) -> datetime:
        """Get current IST time with proper timezone handling."""
        now_utc = datetime.now(timezone.utc)
        if IST:
            return now_utc.astimezone(IST)
        else:
            # Manual IST offset (UTC+5:30)
            ist_offset = timedelta(hours=5, minutes=30)
            return now_utc + ist_offset

    def _is_trading_day(self) -> bool:
        """Check if today is a trading day (Monday to Friday)."""
        now_ist = self._get_ist_now()
        return now_ist.weekday() in self.MCX_TRADING_DAYS

    def _is_market_hours(self) -> bool:
        """Check if current time is within MCX trading hours (Mon-Fri, 9 AM - 11:55 PM IST)."""
        now_ist = self._get_ist_now()

        # First check if it's a trading day
        if not self._is_trading_day():
            return False

        now_ist_hour = now_ist.hour
        now_ist_minute = now_ist.minute

        # Check if within trading hours
        if now_ist_hour >= self.MCX_OPEN_HOUR and now_ist_hour < self.MCX_CLOSE_HOUR:
            return True
        if now_ist_hour == self.MCX_CLOSE_HOUR and now_ist_minute <= self.MCX_CLOSE_MINUTE:
            return True

        return False

    def _time_until_market_open(self) -> int:
        """Get seconds until market opens (accounts for weekends)."""
        now_ist = self._get_ist_now()
        now_ist_hour = now_ist.hour
        now_ist_minute = now_ist.minute
        weekday = now_ist.weekday()

        # Calculate days until next trading day
        days_until_trading = 0
        if weekday == 5:  # Saturday
            days_until_trading = 2  # Wait until Monday
        elif weekday == 6:  # Sunday
            days_until_trading = 1  # Wait until Monday
        elif now_ist_hour >= self.MCX_CLOSE_HOUR and now_ist_minute > self.MCX_CLOSE_MINUTE:
            # After market close on a trading day
            if weekday == 4:  # Friday after close
                days_until_trading = 3  # Wait until Monday
            else:
                days_until_trading = 1  # Wait until tomorrow

        # Calculate hours until 9 AM
        if now_ist_hour >= self.MCX_CLOSE_HOUR:
            hours_until = 24 - now_ist_hour + self.MCX_OPEN_HOUR
        elif now_ist_hour < self.MCX_OPEN_HOUR:
            hours_until = self.MCX_OPEN_HOUR - now_ist_hour
        else:
            hours_until = 0  # Already in trading hours

        minutes_until = -now_ist_minute if hours_until > 0 else 0
        seconds_until = (days_until_trading * 24 * 3600) + (hours_until * 3600) + (minutes_until * 60)

        return max(seconds_until, 60)  # Minimum 1 minute

    def _reload_token_from_env(self) -> bool:
        """Reload access token from environment variable if available."""
        try:
            # Re-read from environment (for cases where .env was updated)
            token = os.environ.get('UPSTOX_ACCESS_TOKEN')
            if token and token != upstox_client.access_token:
                upstox_client.set_access_token(token)
                logger.info("Reloaded Upstox access token from environment")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to reload token: {e}")
            return False

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
        """Check if Upstox authentication is valid. Will try to reload token if invalid."""
        # First try to reload token from environment if it may have been updated
        if not upstox_client.is_authenticated:
            self._reload_token_from_env()

        if not upstox_client.is_authenticated:
            logger.warning("Upstox not authenticated - token not set")
            return False

        try:
            auth_status = await upstox_client.verify_authentication()
            if not auth_status.get("authenticated"):
                reason = auth_status.get('reason', 'unknown')
                logger.warning(f"Upstox token invalid: {reason}")

                # If token is expired, try to reload from environment
                if reason in ('token_expired', 'invalid_token'):
                    self._reload_token_from_env()
                    # Retry authentication after reload
                    auth_status = await upstox_client.verify_authentication()
                    if auth_status.get("authenticated"):
                        logger.info("Authentication successful after token reload")
                        return True

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
        now_ist = self._get_ist_now()
        logger.info(f"Current IST time: {now_ist.strftime('%Y-%m-%d %H:%M:%S %A')}")
        logger.info(f"MCX trading hours: 9:00 AM - 11:55 PM IST (Monday-Friday)")
        logger.info(f"Is trading day: {self._is_trading_day()}")
        logger.info(f"Is market hours: {self._is_market_hours()}")
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
        last_status_log = None

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
                    now_ist = self._get_ist_now()

                    # Detailed logging about why market is closed
                    if not self._is_trading_day():
                        day_name = now_ist.strftime('%A')
                        logger.info(f"Market closed - {day_name} is not a trading day. Waiting {wait_time // 3600}h {(wait_time % 3600) // 60}m until Monday 9 AM IST...")
                    else:
                        current_time = now_ist.strftime('%H:%M')
                        logger.info(f"Market closed - current time {current_time} IST is outside trading hours (9:00-23:55). Waiting {wait_time // 60} minutes...")

                    await asyncio.sleep(min(wait_time, 300))  # Check every 5 min max
                    continue

                # Log market status periodically (every hour)
                now = datetime.now(timezone.utc)
                if last_status_log is None or (now - last_status_log).total_seconds() > 3600:
                    now_ist = self._get_ist_now()
                    logger.info(f"Market is open - {now_ist.strftime('%H:%M IST %A')} - starting/continuing tick collection")
                    last_status_log = now

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
