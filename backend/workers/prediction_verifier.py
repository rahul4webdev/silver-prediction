"""
Background worker for verifying predictions.
Runs periodically to check pending predictions against actual prices.
"""

import asyncio
import logging
from datetime import datetime

from celery import Celery

from app.core.config import settings
from app.models.database import get_db_context
from app.services.prediction_engine import prediction_engine

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "prediction_workers",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max
    worker_prefetch_multiplier=1,
)

# Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    "verify-predictions-every-minute": {
        "task": "workers.prediction_verifier.verify_pending_predictions",
        "schedule": 60.0,  # Every minute
    },
    "generate-30m-predictions": {
        "task": "workers.prediction_verifier.generate_predictions",
        "schedule": 1800.0,  # Every 30 minutes
        "args": ["silver", "mcx", "30m"],
    },
    "generate-1h-predictions": {
        "task": "workers.prediction_verifier.generate_predictions",
        "schedule": 3600.0,  # Every hour
        "args": ["silver", "mcx", "1h"],
    },
    "sync-market-data-every-5-minutes": {
        "task": "workers.prediction_verifier.sync_market_data",
        "schedule": 300.0,  # Every 5 minutes
    },
    "retrain-models-daily": {
        "task": "workers.prediction_verifier.retrain_models",
        "schedule": 86400.0,  # Daily
        "args": ["silver", "mcx", "30m"],
    },
}


def run_async(coro):
    """Helper to run async functions in Celery tasks."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, max_retries=3)
def verify_pending_predictions(self):
    """
    Verify all pending predictions whose target_time has passed.
    Runs every minute.
    """
    async def _verify():
        async with get_db_context() as db:
            result = await prediction_engine.verify_pending_predictions(db)
            return result

    try:
        result = run_async(_verify())
        logger.info(f"Verification complete: {result}")
        return result
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        self.retry(exc=e, countdown=60)


@celery_app.task(bind=True, max_retries=3)
def generate_predictions(self, asset: str, market: str, interval: str):
    """
    Generate prediction for specified asset/market/interval.
    """
    async def _generate():
        async with get_db_context() as db:
            result = await prediction_engine.generate_prediction(
                db, asset, market, interval
            )
            return result

    try:
        result = run_async(_generate())
        logger.info(f"Prediction generated for {asset}/{market}/{interval}")
        return result
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        self.retry(exc=e, countdown=300)


@celery_app.task(bind=True, max_retries=3)
def sync_market_data(self):
    """
    Sync latest market data from Upstox and Yahoo Finance.
    """
    async def _sync():
        from app.services.data_sync import DataSyncService

        sync_service = DataSyncService()

        async with get_db_context() as db:
            # Sync MCX data
            mcx_result = await sync_service.sync_mcx_data(db, "silver", "30m")

            # Sync COMEX data
            comex_result = await sync_service.sync_comex_data(db, "silver", "30m")

            return {
                "mcx": mcx_result,
                "comex": comex_result,
            }

    try:
        result = run_async(_sync())
        logger.info(f"Market data synced: {result}")
        return result
    except Exception as e:
        logger.error(f"Data sync failed: {e}")
        self.retry(exc=e, countdown=60)


@celery_app.task(bind=True, max_retries=1)
def retrain_models(self, asset: str, market: str, interval: str):
    """
    Retrain ML models with latest data.
    Runs daily at configured hour.
    """
    async def _retrain():
        async with get_db_context() as db:
            result = await prediction_engine.train_models(
                db, asset, market, interval
            )
            return result

    try:
        result = run_async(_retrain())
        logger.info(f"Models retrained for {asset}/{market}/{interval}")
        return result
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise


@celery_app.task
def calculate_daily_accuracy():
    """
    Calculate and log daily accuracy metrics.
    """
    async def _calculate():
        async with get_db_context() as db:
            summary = await prediction_engine.get_accuracy_summary(
                db, asset="silver", period_days=1
            )
            return summary

    try:
        result = run_async(_calculate())
        logger.info(f"Daily accuracy: {result}")
        return result
    except Exception as e:
        logger.error(f"Accuracy calculation failed: {e}")
        raise


# Manual trigger functions (for API use)
async def trigger_verification():
    """Trigger verification manually."""
    async with get_db_context() as db:
        return await prediction_engine.verify_pending_predictions(db)


async def trigger_prediction(asset: str, market: str, interval: str):
    """Trigger prediction manually."""
    async with get_db_context() as db:
        return await prediction_engine.generate_prediction(db, asset, market, interval)
