"""
Scheduled tasks for data sync, model training, and prediction generation.
Uses APScheduler for reliable scheduling.
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.services.data_sync import data_sync_service
from app.services.prediction_engine import prediction_engine

logger = logging.getLogger(__name__)

# Timezone
IST = ZoneInfo("Asia/Kolkata")

# MCX Market Hours
MCX_MARKET_OPEN = time(9, 0)   # 9:00 AM IST
MCX_MARKET_CLOSE = time(23, 30)  # 11:30 PM IST

# Indian market holidays (month, day)
INDIAN_MARKET_HOLIDAYS = {
    (1, 26), (2, 26), (3, 14), (3, 31), (4, 10), (4, 14), (4, 18),
    (5, 1), (6, 7), (7, 6), (8, 15), (8, 16), (9, 5), (10, 2),
    (10, 21), (10, 22), (11, 5), (11, 6), (11, 7), (12, 25),
}


class SchedulerWorker:
    """
    Scheduler for automated tasks:
    - Data sync (every 30 minutes during market hours)
    - Model training (daily at 6 AM IST)
    - Prediction generation (every 30 minutes)
    - Prediction verification (every 5 minutes)
    """

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.engine = None
        self.async_session = None
        self._is_running = False

    async def initialize(self):
        """Initialize database connection."""
        self.engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_size=5,
            max_overflow=10,
        )
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info("Scheduler database connection initialized")

    async def get_session(self) -> AsyncSession:
        """Get a new database session."""
        return self.async_session()

    def _is_market_open(self) -> bool:
        """Check if MCX market is currently open (9 AM - 11:30 PM IST, Mon-Fri, excluding holidays)."""
        now_ist = datetime.now(IST)

        # Check weekend (Saturday=5, Sunday=6)
        if now_ist.weekday() >= 5:
            return False

        # Check holidays
        month_day = (now_ist.month, now_ist.day)
        if month_day in INDIAN_MARKET_HOLIDAYS:
            return False

        # Check time
        current_time = now_ist.time()
        return MCX_MARKET_OPEN <= current_time <= MCX_MARKET_CLOSE

    def start(self):
        """Start the scheduler with all jobs."""
        if self._is_running:
            logger.warning("Scheduler already running")
            return

        # Schedule data sync every 30 minutes
        self.scheduler.add_job(
            self.sync_all_data,
            IntervalTrigger(minutes=30),
            id="sync_data",
            name="Sync market data",
            replace_existing=True,
        )

        # Schedule model training daily at 6 AM IST (00:30 UTC)
        self.scheduler.add_job(
            self.train_models,
            CronTrigger(hour=0, minute=30),
            id="train_models",
            name="Train ML models",
            replace_existing=True,
        )

        # Schedule 30-minute predictions every 30 minutes (at :00 and :30)
        self.scheduler.add_job(
            self.generate_30m_predictions,
            CronTrigger(minute="0,30"),
            id="generate_30m_predictions",
            name="Generate 30-minute predictions",
            replace_existing=True,
        )

        # Schedule 1-hour predictions every hour (at :00)
        self.scheduler.add_job(
            self.generate_1h_predictions,
            CronTrigger(minute="0"),
            id="generate_1h_predictions",
            name="Generate 1-hour predictions",
            replace_existing=True,
        )

        # Schedule 4-hour predictions every 4 hours (at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC)
        self.scheduler.add_job(
            self.generate_4h_predictions,
            CronTrigger(hour="0,4,8,12,16,20", minute="0"),
            id="generate_4h_predictions",
            name="Generate 4-hour predictions",
            replace_existing=True,
        )

        # Schedule daily predictions once per day at market open (3:30 UTC = 9:00 AM IST)
        self.scheduler.add_job(
            self.generate_daily_predictions,
            CronTrigger(hour="3", minute="30"),
            id="generate_daily_predictions",
            name="Generate daily predictions",
            replace_existing=True,
        )

        # Schedule prediction verification every 5 minutes
        self.scheduler.add_job(
            self.verify_predictions,
            IntervalTrigger(minutes=5),
            id="verify_predictions",
            name="Verify predictions",
            replace_existing=True,
        )

        # Run initial data sync on startup
        self.scheduler.add_job(
            self.sync_all_data,
            id="initial_sync",
            name="Initial data sync",
        )

        self.scheduler.start()
        self._is_running = True
        logger.info("Scheduler started with all jobs")

    def stop(self):
        """Stop the scheduler."""
        if self._is_running:
            self.scheduler.shutdown(wait=True)
            self._is_running = False
            logger.info("Scheduler stopped")

    async def sync_all_data(self):
        """Sync data from all sources."""
        logger.info("Starting scheduled data sync...")

        try:
            async with await self.get_session() as db:
                results = {}

                # Sync COMEX silver (Yahoo Finance - always available)
                for interval in ["30m", "1h", "4h", "1d"]:
                    try:
                        result = await data_sync_service.sync_comex_data(
                            db, "silver", interval, days=60
                        )
                        results[f"comex_{interval}"] = result
                    except Exception as e:
                        logger.error(f"COMEX sync failed for {interval}: {e}")
                        results[f"comex_{interval}"] = {"status": "error", "error": str(e)}

                # Sync MCX silver (if Upstox authenticated)
                for interval in ["30m", "1h", "4h", "1d"]:
                    try:
                        result = await data_sync_service.sync_mcx_data(
                            db, "silver", interval, days=60
                        )
                        results[f"mcx_{interval}"] = result
                    except Exception as e:
                        logger.error(f"MCX sync failed for {interval}: {e}")
                        results[f"mcx_{interval}"] = {"status": "error", "error": str(e)}

                logger.info(f"Data sync complete: {results}")
                return results

        except Exception as e:
            logger.error(f"Data sync failed: {e}")
            return {"status": "error", "error": str(e)}

    async def train_models(self):
        """Train or retrain ML models."""
        logger.info("Starting scheduled model training...")

        try:
            async with await self.get_session() as db:
                results = {}

                for market in ["comex", "mcx"]:
                    for interval in ["30m", "1h", "4h", "1d"]:
                        try:
                            result = await prediction_engine.train_models(
                                db, "silver", market, interval
                            )
                            results[f"{market}_{interval}"] = result
                        except Exception as e:
                            logger.error(f"Training failed for {market}/{interval}: {e}")
                            results[f"{market}_{interval}"] = {"status": "error", "error": str(e)}

                logger.info(f"Model training complete: {results}")
                return results

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _generate_predictions_for_interval(self, interval: str):
        """Generate predictions for a specific interval for both markets."""
        # Skip if market is closed
        if not self._is_market_open():
            now_ist = datetime.now(IST)
            logger.info(
                f"Skipping {interval} predictions - market closed "
                f"(IST: {now_ist.strftime('%Y-%m-%d %H:%M')}, weekday: {now_ist.weekday()})"
            )
            return {"status": "skipped", "reason": "market_closed"}

        logger.info(f"Generating {interval} predictions...")

        try:
            async with await self.get_session() as db:
                results = {}

                for market in ["mcx", "comex"]:
                    try:
                        prediction = await prediction_engine.generate_prediction(
                            db, "silver", market, interval
                        )
                        if prediction:
                            results[f"{market}_{interval}"] = {
                                "status": "success",
                                "prediction_id": prediction.get("id"),
                                "direction": prediction.get("predicted_direction"),
                                "target_time": prediction.get("target_time"),
                            }
                        else:
                            results[f"{market}_{interval}"] = {"status": "no_data"}
                    except Exception as e:
                        logger.error(f"Prediction failed for {market}/{interval}: {e}")
                        results[f"{market}_{interval}"] = {"status": "error", "error": str(e)}

                logger.info(f"{interval} prediction generation complete: {results}")
                return results

        except Exception as e:
            logger.error(f"{interval} prediction generation failed: {e}")
            return {"status": "error", "error": str(e)}

    async def generate_30m_predictions(self):
        """Generate 30-minute predictions for both markets."""
        return await self._generate_predictions_for_interval("30m")

    async def generate_1h_predictions(self):
        """Generate 1-hour predictions for both markets."""
        return await self._generate_predictions_for_interval("1h")

    async def generate_4h_predictions(self):
        """Generate 4-hour predictions for both markets."""
        return await self._generate_predictions_for_interval("4h")

    async def generate_daily_predictions(self):
        """Generate daily predictions for both markets."""
        return await self._generate_predictions_for_interval("1d")

    async def verify_predictions(self):
        """Verify predictions that have reached their target time."""
        logger.info("Starting prediction verification...")

        try:
            async with await self.get_session() as db:
                from workers.prediction_verifier import prediction_verifier
                result = await prediction_verifier.verify_pending_predictions(db)
                logger.info(f"Prediction verification complete: {result}")
                return result

        except Exception as e:
            logger.error(f"Prediction verification failed: {e}")
            return {"status": "error", "error": str(e)}

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
            "is_running": self._is_running,
            "jobs": jobs,
        }


# Singleton instance
scheduler_worker = SchedulerWorker()


async def run_scheduler():
    """Main entry point for running the scheduler."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    await scheduler_worker.initialize()
    scheduler_worker.start()

    # Keep running
    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        scheduler_worker.stop()


if __name__ == "__main__":
    asyncio.run(run_scheduler())
