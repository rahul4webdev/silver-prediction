#!/usr/bin/env python3
"""
Notification Worker for Telegram alerts.

Handles:
- Daily Upstox token re-authentication reminder (8:45 AM IST)
- Tick collector health checks
- Platform health status
- Prediction notifications after generation (only during trading hours)
- Daily performance report at 11:30 PM IST

Trading Hours: 9:00 AM - 11:30 PM IST, Monday-Friday
Notifications are skipped on weekends and can be skipped on holidays.

Usage:
    python -m workers.notification_worker

This script should be run as a systemd service for production.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.core.config import settings
from app.models.database import init_db, get_db_session
from app.services.telegram_notifier import telegram_notifier
from app.services.upstox_client import upstox_client

# Try to import pytz for proper timezone handling
try:
    import pytz
    IST = pytz.timezone('Asia/Kolkata')
except ImportError:
    IST = None

# MCX Trading Hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 0
MARKET_CLOSE_HOUR = 23
MARKET_CLOSE_MINUTE = 30

# Known Indian market holidays (can be extended)
# Format: (month, day)
INDIAN_MARKET_HOLIDAYS_2025 = {
    (1, 26),   # Republic Day
    (2, 26),   # Maha Shivaratri
    (3, 14),   # Holi
    (3, 31),   # Id-ul-Fitr (tentative)
    (4, 10),   # Mahavir Jayanti
    (4, 14),   # Dr. Ambedkar Jayanti
    (4, 18),   # Good Friday
    (5, 1),    # May Day
    (6, 7),    # Id-ul-Adha (tentative)
    (7, 6),    # Muharram (tentative)
    (8, 15),   # Independence Day
    (8, 16),   # Parsi New Year
    (9, 5),    # Milad-un-Nabi (tentative)
    (10, 2),   # Gandhi Jayanti
    (10, 21),  # Dussehra
    (10, 22),  # Dussehra
    (11, 5),   # Diwali (Laxmi Puja)
    (11, 6),   # Diwali Balipratipada
    (11, 7),   # Diwali (Bhai Dooj)
    (12, 25),  # Christmas
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class NotificationWorker:
    """
    Worker that sends scheduled Telegram notifications.

    Notifications are sent only during trading hours (9:00 AM - 11:30 PM IST)
    on weekdays. Weekends and holidays are skipped.
    """

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self._running = False
        self._last_tick_check: Optional[datetime] = None
        self._last_tick_count: int = 0
        # Custom holidays can be added here (format: date objects)
        self._custom_holidays: Set[date] = set()

    def _get_ist_now(self) -> datetime:
        """Get current IST time."""
        now_utc = datetime.now(timezone.utc)
        if IST:
            return now_utc.astimezone(IST)
        else:
            ist_offset = timedelta(hours=5, minutes=30)
            return now_utc + ist_offset

    def _is_weekend(self) -> bool:
        """Check if today is a weekend (Saturday=5, Sunday=6)."""
        now_ist = self._get_ist_now()
        return now_ist.weekday() >= 5

    def _is_holiday(self) -> bool:
        """Check if today is a known market holiday."""
        now_ist = self._get_ist_now()
        today = now_ist.date()

        # Check custom holidays first
        if today in self._custom_holidays:
            return True

        # Check known Indian market holidays
        month_day = (today.month, today.day)
        if month_day in INDIAN_MARKET_HOLIDAYS_2025:
            return True

        return False

    def _is_trading_hours(self) -> bool:
        """
        Check if current time is within MCX trading hours (9:00 AM - 11:30 PM IST).
        Returns False on weekends and holidays.
        """
        # Skip weekends
        if self._is_weekend():
            return False

        # Skip holidays
        if self._is_holiday():
            return False

        now_ist = self._get_ist_now()
        hour = now_ist.hour
        minute = now_ist.minute
        current_time = hour * 60 + minute

        market_open = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MINUTE  # 9:00 = 540
        market_close = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE  # 23:30 = 1410

        return market_open <= current_time <= market_close

    def _get_market_closed_reason(self) -> Optional[str]:
        """Get reason why market is closed."""
        if self._is_weekend():
            now_ist = self._get_ist_now()
            day_name = now_ist.strftime("%A")
            return f"Weekend ({day_name}) - Market Closed"

        if self._is_holiday():
            return "Public Holiday - Market Closed"

        now_ist = self._get_ist_now()
        hour = now_ist.hour
        minute = now_ist.minute

        if hour < MARKET_OPEN_HOUR or (hour == MARKET_OPEN_HOUR and minute < MARKET_OPEN_MINUTE):
            return f"Pre-market hours (Market opens at {MARKET_OPEN_HOUR}:{MARKET_OPEN_MINUTE:02d} AM IST)"

        if hour > MARKET_CLOSE_HOUR or (hour == MARKET_CLOSE_HOUR and minute > MARKET_CLOSE_MINUTE):
            return f"Post-market hours (Market closed at {MARKET_CLOSE_HOUR}:{MARKET_CLOSE_MINUTE:02d} PM IST)"

        return None

    def add_holiday(self, holiday_date: date):
        """Add a custom holiday date."""
        self._custom_holidays.add(holiday_date)
        logger.info(f"Added custom holiday: {holiday_date}")

    def remove_holiday(self, holiday_date: date):
        """Remove a custom holiday date."""
        self._custom_holidays.discard(holiday_date)
        logger.info(f"Removed custom holiday: {holiday_date}")

    async def send_auth_reminder(self):
        """Send daily authentication reminder at 8:45 AM IST."""
        # Skip on weekends and holidays
        if self._is_weekend() or self._is_holiday():
            reason = "Weekend" if self._is_weekend() else "Holiday"
            logger.info(f"Skipping auth reminder - {reason}, market is closed")
            return

        logger.info("Sending Upstox auth reminder...")
        try:
            await telegram_notifier.send_auth_reminder()
        except Exception as e:
            logger.error(f"Failed to send auth reminder: {e}")

    async def check_tick_collector(self):
        """Check tick collector health and send notification if issues."""
        # Skip check on weekends and holidays
        if self._is_weekend() or self._is_holiday():
            logger.info("Skipping tick collector check - market is closed")
            return

        logger.info("Checking tick collector health...")
        try:
            from app.services.tick_collector import tick_collector

            is_running = tick_collector.is_running
            last_tick = tick_collector.last_tick_time

            # Calculate ticks per minute if we have previous data
            ticks_per_min = 0
            if self._last_tick_check and self._last_tick_count > 0:
                elapsed = (datetime.now() - self._last_tick_check).total_seconds() / 60
                if elapsed > 0:
                    current_count = tick_collector.tick_count
                    ticks_per_min = (current_count - self._last_tick_count) / elapsed

            self._last_tick_check = datetime.now()
            self._last_tick_count = tick_collector.tick_count

            # Get subscribed contracts count
            contracts = len(tick_collector.subscribed_instruments) if hasattr(tick_collector, 'subscribed_instruments') else 0

            # Determine if there's an issue and get reason
            error_msg = None
            reason = None
            is_market_hours = self._is_trading_hours()

            if not is_market_hours:
                # Market is closed - get the reason
                reason = self._get_market_closed_reason()
                if not is_running:
                    # This is expected when market is closed, no notification needed
                    logger.info(f"Tick collector not running - {reason}")
                    return
            else:
                # During market hours - check for issues
                if not is_running:
                    error_msg = "Tick collector not running during market hours"
                    reason = "Service may have stopped or crashed. Check systemd logs."
                elif last_tick and (datetime.now() - last_tick).total_seconds() > 300:
                    error_msg = "No ticks received in last 5 minutes"
                    reason = "WebSocket connection may be lost or Upstox auth expired."
                elif contracts == 0:
                    error_msg = "No contracts subscribed"
                    reason = "Failed to subscribe to MCX Silver contracts."

            # Only send notification if there's an issue during market hours
            if error_msg:
                await telegram_notifier.send_tick_collector_status(
                    is_running=is_running,
                    contracts_subscribed=contracts,
                    last_tick_time=last_tick,
                    ticks_per_minute=ticks_per_min,
                    error_message=error_msg,
                    reason=reason,
                )

        except Exception as e:
            logger.error(f"Failed to check tick collector: {e}")


    async def send_platform_health(self):
        """Send comprehensive platform health status."""
        # Skip on weekends and holidays
        if self._is_weekend() or self._is_holiday():
            logger.info("Skipping platform health notification - market is closed")
            return

        logger.info("Sending platform health status...")
        try:
            # Check API health
            api_healthy = True  # If we're running, API is healthy

            # Check database
            db_healthy = False
            try:
                async with get_db_session() as db:
                    from sqlalchemy import text
                    await db.execute(text("SELECT 1"))
                    db_healthy = True
            except Exception as e:
                logger.error(f"Database health check failed: {e}")

            # Check Redis
            redis_healthy = False
            try:
                import redis.asyncio as redis
                r = redis.from_url(settings.redis_url)
                await r.ping()
                redis_healthy = True
                await r.close()
            except Exception:
                pass

            # Check Upstox auth
            upstox_authenticated = False
            try:
                auth_status = await upstox_client.verify_authentication()
                upstox_authenticated = auth_status.get("authenticated", False)
            except Exception:
                pass

            # Check tick collector
            tick_collector_running = False
            try:
                from app.services.tick_collector import tick_collector
                tick_collector_running = tick_collector.is_running
            except Exception:
                pass

            # Check scheduler - assume running if this code is executing
            scheduler_running = self._running

            # Check models
            models_trained = {}
            try:
                from app.services.prediction_engine import prediction_engine
                for interval in ["30m", "1h", "4h", "1d"]:
                    ensemble = prediction_engine.get_ensemble(interval)
                    models_trained[interval] = ensemble.is_trained
            except Exception:
                pass

            await telegram_notifier.send_platform_health(
                api_healthy=api_healthy,
                db_healthy=db_healthy,
                redis_healthy=redis_healthy,
                upstox_authenticated=upstox_authenticated,
                tick_collector_running=tick_collector_running,
                scheduler_running=scheduler_running,
                models_trained=models_trained,
            )

        except Exception as e:
            logger.error(f"Failed to send platform health: {e}")

    async def send_predictions_notification(self, interval: str):
        """Generate and send predictions for a specific interval."""
        # Skip on weekends and holidays
        if self._is_weekend() or self._is_holiday():
            logger.info(f"Skipping {interval} predictions - market is closed (weekend/holiday)")
            return

        # Skip outside trading hours (9:00 AM - 11:30 PM IST)
        if not self._is_trading_hours():
            reason = self._get_market_closed_reason()
            logger.info(f"Skipping {interval} predictions - {reason}")
            return

        logger.info(f"Generating and sending {interval} predictions...")
        try:
            from app.services.prediction_engine import prediction_engine

            predictions = []

            # Generate predictions for all MCX contracts
            async with get_db_session() as db:
                # Get MCX contracts
                try:
                    contracts = await upstox_client.get_all_silver_instrument_keys()
                except Exception:
                    contracts = [{"contract_type": "SILVER", "instrument_key": None, "trading_symbol": None}]

                for contract in contracts[:3]:  # Limit to top 3 contracts
                    try:
                        prediction = await prediction_engine.generate_prediction(
                            db,
                            asset="silver",
                            market="mcx",
                            interval=interval,
                            instrument_key=contract.get("instrument_key"),
                            contract_type=contract.get("contract_type"),
                            trading_symbol=contract.get("trading_symbol"),
                        )
                        predictions.append(prediction)
                    except Exception as e:
                        logger.warning(f"Failed to generate MCX prediction for {contract.get('contract_type')}: {e}")

                # Generate COMEX prediction
                try:
                    prediction = await prediction_engine.generate_prediction(
                        db,
                        asset="silver",
                        market="comex",
                        interval=interval,
                    )
                    predictions.append(prediction)
                except Exception as e:
                    logger.warning(f"Failed to generate COMEX prediction: {e}")

            if predictions:
                await telegram_notifier.send_predictions(interval, predictions)

        except Exception as e:
            logger.error(f"Failed to send {interval} predictions: {e}")

    async def send_30m_predictions(self):
        """Send 30-minute predictions."""
        await self.send_predictions_notification("30m")

    async def send_1h_predictions(self):
        """Send 1-hour predictions."""
        await self.send_predictions_notification("1h")

    async def send_4h_predictions(self):
        """Send 4-hour predictions."""
        await self.send_predictions_notification("4h")

    async def send_daily_predictions(self):
        """Send daily predictions."""
        await self.send_predictions_notification("1d")

    async def send_daily_performance_report(self):
        """
        Send daily performance report at 11:30 PM IST.
        Shows accuracy stats for all predictions made today.
        """
        # Skip on weekends and holidays (no trading, no predictions)
        if self._is_weekend() or self._is_holiday():
            logger.info("Skipping daily performance report - market was closed today")
            return

        logger.info("Generating daily performance report...")
        try:
            from sqlalchemy import text

            now_ist = self._get_ist_now()
            today = now_ist.date()

            async with get_db_session() as db:
                # Get all verified predictions for today
                result = await db.execute(text("""
                    SELECT
                        interval,
                        market,
                        COUNT(*) as total,
                        SUM(CASE WHEN is_direction_correct = true THEN 1 ELSE 0 END) as successful,
                        SUM(CASE WHEN is_direction_correct = false THEN 1 ELSE 0 END) as failed
                    FROM predictions
                    WHERE DATE(prediction_time AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Kolkata') = :today
                    AND verified_at IS NOT NULL
                    GROUP BY interval, market
                """), {"today": today})

                rows = result.fetchall()

                if not rows:
                    logger.info("No verified predictions found for today")
                    return

                # Aggregate stats
                total_predictions = 0
                successful_predictions = 0
                failed_predictions = 0
                interval_stats = {}
                market_stats = {}

                for row in rows:
                    interval = row[0]
                    market = row[1]
                    total = row[2]
                    success = row[3] or 0
                    failed = row[4] or 0

                    total_predictions += total
                    successful_predictions += success
                    failed_predictions += failed

                    # Aggregate by interval
                    if interval not in interval_stats:
                        interval_stats[interval] = {"total": 0, "success": 0, "failed": 0}
                    interval_stats[interval]["total"] += total
                    interval_stats[interval]["success"] += success
                    interval_stats[interval]["failed"] += failed

                    # Aggregate by market
                    if market not in market_stats:
                        market_stats[market] = {"total": 0, "success": 0, "failed": 0}
                    market_stats[market]["total"] += total
                    market_stats[market]["success"] += success
                    market_stats[market]["failed"] += failed

                # Send the report
                await telegram_notifier.send_daily_performance_report(
                    date=now_ist,
                    total_predictions=total_predictions,
                    successful_predictions=successful_predictions,
                    failed_predictions=failed_predictions,
                    interval_stats=interval_stats,
                    market_stats=market_stats,
                )

        except Exception as e:
            logger.error(f"Failed to send daily performance report: {e}")

    def start(self):
        """Start the notification scheduler."""
        if self._running:
            logger.warning("Notification worker already running")
            return

        # ==================== Schedule Jobs ====================

        # 1. Daily Upstox auth reminder at 8:45 AM IST (3:15 UTC)
        self.scheduler.add_job(
            self.send_auth_reminder,
            CronTrigger(hour=3, minute=15),  # 8:45 AM IST = 3:15 UTC
            id="auth_reminder",
            name="Daily Upstox Auth Reminder",
            replace_existing=True,
        )

        # 2. Tick collector health check every 5 minutes during market hours
        self.scheduler.add_job(
            self.check_tick_collector,
            IntervalTrigger(minutes=5),
            id="tick_collector_check",
            name="Tick Collector Health Check",
            replace_existing=True,
        )

        # 3. Platform health status every 6 hours
        self.scheduler.add_job(
            self.send_platform_health,
            IntervalTrigger(hours=6),
            id="platform_health",
            name="Platform Health Status",
            replace_existing=True,
        )

        # 4. 30-minute predictions at :00 and :30
        self.scheduler.add_job(
            self.send_30m_predictions,
            CronTrigger(minute="0,30"),
            id="predictions_30m",
            name="30-Minute Predictions",
            replace_existing=True,
        )

        # 5. 1-hour predictions at :00
        self.scheduler.add_job(
            self.send_1h_predictions,
            CronTrigger(minute=0),
            id="predictions_1h",
            name="1-Hour Predictions",
            replace_existing=True,
        )

        # 6. 4-hour predictions at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
        self.scheduler.add_job(
            self.send_4h_predictions,
            CronTrigger(hour="0,4,8,12,16,20", minute=0),
            id="predictions_4h",
            name="4-Hour Predictions",
            replace_existing=True,
        )

        # 7. Daily predictions at market open (3:30 UTC = 9:00 AM IST)
        self.scheduler.add_job(
            self.send_daily_predictions,
            CronTrigger(hour=3, minute=30),
            id="predictions_daily",
            name="Daily Predictions",
            replace_existing=True,
        )

        # 8. Daily performance report at 11:30 PM IST (18:00 UTC)
        self.scheduler.add_job(
            self.send_daily_performance_report,
            CronTrigger(hour=18, minute=0),  # 11:30 PM IST = 18:00 UTC
            id="daily_performance_report",
            name="Daily Performance Report",
            replace_existing=True,
        )

        # Send startup notification
        self.scheduler.add_job(
            self._send_startup_notification,
            id="startup_notification",
            name="Startup Notification",
        )

        self.scheduler.start()
        self._running = True
        logger.info("Notification worker started with all jobs")

    async def _send_startup_notification(self):
        """Send notification that the worker has started."""
        message = f"""
🚀 <b>Notification Worker Started</b>

<b>Scheduled Notifications:</b>
• Auth reminder: 8:45 AM IST daily
• Tick collector: Every 5 min (trading hours)
• Platform health: Every 6 hours
• 30m predictions: Every 30 min (trading hours)
• 1h predictions: Every hour (trading hours)
• 4h predictions: Every 4 hours (trading hours)
• Daily predictions: 9:00 AM IST
• Daily report: 11:30 PM IST

<b>Trading Hours:</b> 9:00 AM - 11:30 PM IST
<b>Market Days:</b> Monday - Friday

ℹ️ Notifications skipped on weekends & holidays

⏰ {self._get_ist_now().strftime('%Y-%m-%d %H:%M:%S IST')}
"""
        await telegram_notifier.send_message(message.strip())

    def stop(self):
        """Stop the notification scheduler."""
        if self._running:
            self.scheduler.shutdown(wait=True)
            self._running = False
            logger.info("Notification worker stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            })

        return {
            "is_running": self._running,
            "jobs": jobs,
        }


# Singleton instance
notification_worker = NotificationWorker()


async def run_notification_worker():
    """Main entry point for running the notification worker."""
    # Initialize database
    await init_db()

    # Check if Telegram is configured
    if not telegram_notifier.is_configured:
        logger.error("Telegram not configured! Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return

    logger.info("=" * 60)
    logger.info("Starting Notification Worker")
    logger.info(f"Telegram Bot: {'Configured' if telegram_notifier.is_configured else 'Not configured'}")
    logger.info(f"Environment: {settings.environment}")
    logger.info("=" * 60)

    notification_worker.start()

    # Set up signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: notification_worker.stop())

    # Keep running
    try:
        while notification_worker._running:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        notification_worker.stop()


if __name__ == "__main__":
    asyncio.run(run_notification_worker())
