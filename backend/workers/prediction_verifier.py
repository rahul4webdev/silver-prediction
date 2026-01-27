"""
Background worker for verifying predictions.
Runs periodically to check pending predictions against actual prices.
"""

import asyncio
import logging
from datetime import datetime, time
from zoneinfo import ZoneInfo

from celery import Celery

from app.core.config import settings
from app.core.constants import ASSET_CONFIGS, Asset, Market
from app.models.database import get_db_context
from app.services.prediction_engine import prediction_engine

logger = logging.getLogger(__name__)

# Timezone for market hours
IST = ZoneInfo("Asia/Kolkata")
EST = ZoneInfo("America/New_York")

# MCX Market Hours (IST)
MCX_MARKET_OPEN = time(9, 0)   # 9:00 AM IST
MCX_MARKET_CLOSE = time(23, 30)  # 11:30 PM IST

# COMEX is nearly 24 hours (6 PM to 5 PM next day EST)
# We'll consider it always open for simplicity

# Indian market holidays for 2025 & 2026 (month, day)
INDIAN_MARKET_HOLIDAYS = {
    # 2025
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
    # 2026 (add as needed)
    (1, 26),   # Republic Day
}


def is_mcx_market_open() -> bool:
    """Check if MCX market is currently open (9 AM - 11:30 PM IST, Mon-Fri, excluding holidays)."""
    now_ist = datetime.now(IST)

    # Check weekend first (Saturday=5, Sunday=6)
    if now_ist.weekday() >= 5:
        return False

    # Check holidays
    month_day = (now_ist.month, now_ist.day)
    if month_day in INDIAN_MARKET_HOLIDAYS:
        return False

    # Check time
    current_time = now_ist.time()
    return MCX_MARKET_OPEN <= current_time <= MCX_MARKET_CLOSE


def is_comex_market_open() -> bool:
    """Check if COMEX market is open (nearly 24 hours, except weekends)."""
    now_est = datetime.now(EST)
    # COMEX is closed on weekends (Saturday after 5 PM to Sunday 6 PM)
    weekday = now_est.weekday()
    hour = now_est.hour

    # Saturday after 5 PM EST - closed
    if weekday == 5 and hour >= 17:
        return False
    # All day Sunday until 6 PM - closed
    if weekday == 6 and hour < 18:
        return False

    return True


def is_market_open(market: str) -> bool:
    """Check if the specified market is open."""
    if market == "mcx":
        return is_mcx_market_open()
    elif market == "comex":
        return is_comex_market_open()
    return True  # Default to open


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
# Predictions for all intervals and both assets
celery_app.conf.beat_schedule = {
    # Verification - every minute (always runs to catch up on verifications)
    "verify-predictions-every-minute": {
        "task": "workers.prediction_verifier.verify_pending_predictions",
        "schedule": 60.0,
    },

    # Silver MCX - all intervals
    "generate-silver-mcx-30m": {
        "task": "workers.prediction_verifier.generate_predictions",
        "schedule": 1800.0,  # 30 minutes
        "args": ["silver", "mcx", "30m"],
    },
    "generate-silver-mcx-1h": {
        "task": "workers.prediction_verifier.generate_predictions",
        "schedule": 3600.0,  # 1 hour
        "args": ["silver", "mcx", "1h"],
    },
    "generate-silver-mcx-4h": {
        "task": "workers.prediction_verifier.generate_predictions",
        "schedule": 14400.0,  # 4 hours
        "args": ["silver", "mcx", "4h"],
    },
    "generate-silver-mcx-daily": {
        "task": "workers.prediction_verifier.generate_predictions",
        "schedule": 86400.0,  # Daily
        "args": ["silver", "mcx", "1d"],
    },

    # Silver COMEX - all intervals
    "generate-silver-comex-30m": {
        "task": "workers.prediction_verifier.generate_predictions",
        "schedule": 1800.0,
        "args": ["silver", "comex", "30m"],
    },
    "generate-silver-comex-1h": {
        "task": "workers.prediction_verifier.generate_predictions",
        "schedule": 3600.0,
        "args": ["silver", "comex", "1h"],
    },
    "generate-silver-comex-4h": {
        "task": "workers.prediction_verifier.generate_predictions",
        "schedule": 14400.0,
        "args": ["silver", "comex", "4h"],
    },
    "generate-silver-comex-daily": {
        "task": "workers.prediction_verifier.generate_predictions",
        "schedule": 86400.0,
        "args": ["silver", "comex", "1d"],
    },

    # Gold MCX - all intervals
    "generate-gold-mcx-30m": {
        "task": "workers.prediction_verifier.generate_predictions",
        "schedule": 1800.0,
        "args": ["gold", "mcx", "30m"],
    },
    "generate-gold-mcx-1h": {
        "task": "workers.prediction_verifier.generate_predictions",
        "schedule": 3600.0,
        "args": ["gold", "mcx", "1h"],
    },
    "generate-gold-mcx-4h": {
        "task": "workers.prediction_verifier.generate_predictions",
        "schedule": 14400.0,
        "args": ["gold", "mcx", "4h"],
    },
    "generate-gold-mcx-daily": {
        "task": "workers.prediction_verifier.generate_predictions",
        "schedule": 86400.0,
        "args": ["gold", "mcx", "1d"],
    },

    # Gold COMEX - all intervals
    "generate-gold-comex-30m": {
        "task": "workers.prediction_verifier.generate_predictions",
        "schedule": 1800.0,
        "args": ["gold", "comex", "30m"],
    },
    "generate-gold-comex-1h": {
        "task": "workers.prediction_verifier.generate_predictions",
        "schedule": 3600.0,
        "args": ["gold", "comex", "1h"],
    },
    "generate-gold-comex-4h": {
        "task": "workers.prediction_verifier.generate_predictions",
        "schedule": 14400.0,
        "args": ["gold", "comex", "4h"],
    },
    "generate-gold-comex-daily": {
        "task": "workers.prediction_verifier.generate_predictions",
        "schedule": 86400.0,
        "args": ["gold", "comex", "1d"],
    },

    # Data sync - every 5 minutes (only during market hours)
    "sync-market-data-every-5-minutes": {
        "task": "workers.prediction_verifier.sync_market_data",
        "schedule": 300.0,
    },

    # Model retraining - daily for key intervals
    "retrain-silver-mcx-30m": {
        "task": "workers.prediction_verifier.retrain_models",
        "schedule": 86400.0,
        "args": ["silver", "mcx", "30m"],
    },
    "retrain-silver-mcx-1h": {
        "task": "workers.prediction_verifier.retrain_models",
        "schedule": 86400.0,
        "args": ["silver", "mcx", "1h"],
    },
    "retrain-gold-mcx-30m": {
        "task": "workers.prediction_verifier.retrain_models",
        "schedule": 86400.0,
        "args": ["gold", "mcx", "30m"],
    },
    "retrain-gold-mcx-1h": {
        "task": "workers.prediction_verifier.retrain_models",
        "schedule": 86400.0,
        "args": ["gold", "mcx", "1h"],
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
    Only generates during market hours.
    For MCX silver, generates predictions for 3 contracts (SILVER, SILVERM, SILVERMIC).
    """
    # Check if market is open
    if not is_market_open(market):
        now_ist = datetime.now(IST)
        logger.info(
            f"Skipping prediction for {asset}/{market}/{interval} - "
            f"market closed (IST: {now_ist.strftime('%H:%M')})"
        )
        return {"skipped": True, "reason": "market_closed"}

    async def _generate():
        async with get_db_context() as db:
            results = {}

            # For MCX silver, generate predictions for multiple contracts
            if market == "mcx" and asset == "silver":
                try:
                    from app.services.upstox_client import upstox_client

                    # Get all silver contracts sorted by expiry (ascending)
                    silver_contracts = await upstox_client.get_all_silver_instrument_keys()

                    if silver_contracts:
                        # Get unique contract types with nearest expiry (max 3)
                        seen_types = set()
                        contracts_to_use = []

                        for contract in silver_contracts:
                            contract_type = contract.get("contract_type")
                            if contract_type and contract_type not in seen_types:
                                seen_types.add(contract_type)
                                contracts_to_use.append(contract)
                                if len(contracts_to_use) >= 3:
                                    break

                        logger.info(
                            f"Generating MCX predictions for {len(contracts_to_use)} contracts: "
                            f"{[c.get('contract_type') for c in contracts_to_use]}"
                        )

                        # Generate prediction for each contract
                        for contract in contracts_to_use:
                            contract_type = contract.get("contract_type")
                            try:
                                prediction = await prediction_engine.generate_prediction(
                                    db,
                                    asset,
                                    market,
                                    interval,
                                    instrument_key=contract.get("instrument_key"),
                                    contract_type=contract_type,
                                    trading_symbol=contract.get("trading_symbol"),
                                    expiry=contract.get("expiry"),
                                )
                                results[contract_type] = prediction
                            except Exception as e:
                                logger.error(f"Prediction failed for {contract_type}: {e}")
                                results[contract_type] = {"error": str(e)}

                        return results
                except Exception as e:
                    logger.warning(f"Failed to get MCX contracts, using default: {e}")

            # Default: single prediction (COMEX or fallback MCX)
            result = await prediction_engine.generate_prediction(
                db, asset, market, interval
            )
            return result

    try:
        result = run_async(_generate())
        logger.info(f"Prediction generated for {asset}/{market}/{interval}")
        return result
    except Exception as e:
        logger.error(f"Prediction generation failed for {asset}/{market}/{interval}: {e}")
        self.retry(exc=e, countdown=300)


@celery_app.task(bind=True, max_retries=3)
def sync_market_data(self):
    """
    Sync latest market data from Upstox and Yahoo Finance.
    Syncs 30m data and aggregates to 1h and 4h for MCX.
    """
    async def _sync():
        from app.services.data_sync import DataSyncService

        sync_service = DataSyncService()
        results = {}

        async with get_db_context() as db:
            # Sync Silver data - all intervals
            for market in ["mcx", "comex"]:
                if is_market_open(market):
                    for interval in ["30m", "1h", "4h", "1d"]:
                        try:
                            if market == "mcx":
                                result = await sync_service.sync_mcx_data(db, "silver", interval)
                            else:
                                result = await sync_service.sync_comex_data(db, "silver", interval)
                            results[f"silver_{market}_{interval}"] = result
                        except Exception as e:
                            logger.warning(f"Failed to sync silver {market} {interval}: {e}")
                            results[f"silver_{market}_{interval}"] = {"error": str(e)}

            # Sync Gold data - all intervals
            for market in ["mcx", "comex"]:
                if is_market_open(market):
                    for interval in ["30m", "1h", "4h", "1d"]:
                        try:
                            if market == "mcx":
                                result = await sync_service.sync_mcx_data(db, "gold", interval)
                            else:
                                result = await sync_service.sync_comex_data(db, "gold", interval)
                            results[f"gold_{market}_{interval}"] = result
                        except Exception as e:
                            logger.warning(f"Failed to sync gold {market} {interval}: {e}")
                            results[f"gold_{market}_{interval}"] = {"error": str(e)}

            return results

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
