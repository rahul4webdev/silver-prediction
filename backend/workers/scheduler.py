"""
Scheduled tasks for data sync, model training, and prediction generation.
Uses APScheduler for reliable scheduling.

Training Strategy:
- Hourly training during market hours (9 AM - 11:30 PM IST)
- Uses: Historical data + Recent tick data + News sentiment
- Training happens at :15 past each hour (9:15, 10:15, 11:15, etc.)
- Last training at 11:15 PM covers data until close
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select, func, and_
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
    - Model training (daily at 6 AM IST + hourly during market hours)
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

    def _is_trading_day(self) -> bool:
        """Check if today is a trading day (Mon-Fri, excluding holidays)."""
        now_ist = datetime.now(IST)

        # Check weekend
        if now_ist.weekday() >= 5:
            return False

        # Check holidays
        month_day = (now_ist.month, now_ist.day)
        return month_day not in INDIAN_MARKET_HOLIDAYS

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

        # Schedule sentiment sync every 30 minutes
        self.scheduler.add_job(
            self.sync_sentiment_data,
            IntervalTrigger(minutes=30),
            id="sync_sentiment",
            name="Sync news sentiment",
            replace_existing=True,
        )

        # Schedule daily model training at 6 AM IST (00:30 UTC)
        self.scheduler.add_job(
            self.train_models,
            CronTrigger(hour=0, minute=30),
            id="train_models",
            name="Train ML models (daily)",
            replace_existing=True,
        )

        # Schedule hourly model training at :15 past each hour (during trading hours)
        # This runs 9:15, 10:15, 11:15, ... 23:15 IST = 3:45, 4:45, ... 17:45 UTC
        self.scheduler.add_job(
            self.train_models_hourly,
            CronTrigger(minute=15),
            id="train_models_hourly",
            name="Train ML models (hourly)",
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

    async def sync_sentiment_data(self):
        """Sync news sentiment data and save to database."""
        logger.info("Starting sentiment sync...")

        try:
            from app.services.news_sentiment import news_sentiment_service

            async with await self.get_session() as db:
                results = {}

                # Sync sentiment for silver
                result = await news_sentiment_service.fetch_and_save_sentiment(
                    db, asset="silver", lookback_days=3
                )
                results["silver"] = result

                # Sync sentiment for gold
                result = await news_sentiment_service.fetch_and_save_sentiment(
                    db, asset="gold", lookback_days=3
                )
                results["gold"] = result

                logger.info(f"Sentiment sync complete: {results}")
                return results

        except Exception as e:
            logger.error(f"Sentiment sync failed: {e}")
            return {"status": "error", "error": str(e)}

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
        """Train or retrain ML models (daily full training)."""
        logger.info("Starting scheduled daily model training...")

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

                logger.info(f"Daily model training complete: {results}")
                return results

        except Exception as e:
            logger.error(f"Daily model training failed: {e}")
            return {"status": "error", "error": str(e)}

    async def train_models_hourly(self):
        """
        Hourly model training during market hours.

        This trains models with:
        1. Historical data from database
        2. Recent tick data from the last hour (if available)
        3. Latest news sentiment data

        Training schedule:
        - 9:15 AM IST: First training with overnight data
        - 10:15 AM IST: Training with 9:00-10:00 tick data
        - 11:15 AM IST: Training with 10:00-11:00 tick data
        - ... continues hourly ...
        - 11:15 PM IST: Last training with 10:00-11:00 PM tick data
        """
        # Skip if market is closed
        if not self._is_market_open():
            now_ist = datetime.now(IST)
            logger.info(
                f"Skipping hourly training - market closed "
                f"(IST: {now_ist.strftime('%Y-%m-%d %H:%M')}, weekday: {now_ist.weekday()})"
            )
            return {"status": "skipped", "reason": "market_closed"}

        now_ist = datetime.now(IST)
        logger.info(f"Starting hourly model training at {now_ist.strftime('%H:%M')} IST...")

        try:
            async with await self.get_session() as db:
                results = {
                    "timestamp": datetime.now(IST).isoformat(),
                    "training_hour": now_ist.hour,
                    "markets": {},
                }

                # Get sentiment data first (used for all training)
                sentiment_data = await self._get_sentiment_data()
                results["sentiment"] = {
                    "available": sentiment_data is not None,
                    "label": sentiment_data.get("label") if sentiment_data else None,
                    "score": sentiment_data.get("overall") if sentiment_data else None,
                }

                # Get recent tick data from the last hour
                tick_stats = await self._get_recent_tick_stats(db)
                results["tick_data"] = tick_stats

                # Train models for each market/interval combination
                for market in ["mcx", "comex"]:
                    results["markets"][market] = {}

                    for interval in ["30m", "1h", "4h", "1d"]:
                        try:
                            # Train with enhanced data (historical + sentiment)
                            result = await self._train_with_enhanced_data(
                                db, "silver", market, interval, sentiment_data, tick_stats
                            )
                            results["markets"][market][interval] = result
                            logger.info(
                                f"Hourly training {market}/{interval}: "
                                f"{result.get('training_samples', 0)} samples, "
                                f"sentiment: {sentiment_data.get('label') if sentiment_data else 'N/A'}"
                            )
                        except ValueError as e:
                            # Insufficient data - not an error, just skip
                            logger.info(f"Skipped {market}/{interval}: {e}")
                            results["markets"][market][interval] = {"status": "skipped", "reason": str(e)}
                        except Exception as e:
                            logger.error(f"Hourly training failed for {market}/{interval}: {e}")
                            results["markets"][market][interval] = {"status": "error", "error": str(e)}

                logger.info(f"Hourly model training complete at {now_ist.strftime('%H:%M')} IST")
                return results

        except Exception as e:
            logger.error(f"Hourly model training failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _get_sentiment_data(self) -> Optional[Dict[str, Any]]:
        """Fetch current news sentiment for silver."""
        try:
            from app.services.news_sentiment import news_sentiment_service

            sentiment = await news_sentiment_service.get_sentiment(
                asset="silver",
                lookback_days=3,  # Last 3 days of news
            )

            return {
                "overall": sentiment.overall_sentiment,
                "label": sentiment.sentiment_label,
                "confidence": sentiment.confidence,
                "article_count": sentiment.article_count,
                "bullish_count": sentiment.bullish_count,
                "bearish_count": sentiment.bearish_count,
            }
        except Exception as e:
            logger.warning(f"Failed to get sentiment data: {e}")
            return None

    async def _get_recent_tick_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """
        Get statistics from tick data collected in the last hour.

        Returns OHLCV-like stats from real-time tick data.
        """
        try:
            from app.models.tick_data import TickData

            # Get ticks from the last hour
            one_hour_ago = datetime.now(IST) - timedelta(hours=1)

            result = await db.execute(
                select(
                    func.count(TickData.id).label("tick_count"),
                    func.min(TickData.ltp).label("low"),
                    func.max(TickData.ltp).label("high"),
                    func.avg(TickData.ltp).label("avg"),
                    func.sum(TickData.volume).label("total_volume"),
                ).where(
                    and_(
                        TickData.asset == "silver",
                        TickData.market == "mcx",
                        TickData.timestamp >= one_hour_ago,
                    )
                )
            )

            row = result.first()

            if row and row.tick_count > 0:
                # Get first and last tick for open/close
                first_tick = await db.execute(
                    select(TickData.ltp)
                    .where(
                        and_(
                            TickData.asset == "silver",
                            TickData.market == "mcx",
                            TickData.timestamp >= one_hour_ago,
                        )
                    )
                    .order_by(TickData.timestamp.asc())
                    .limit(1)
                )
                last_tick = await db.execute(
                    select(TickData.ltp)
                    .where(
                        and_(
                            TickData.asset == "silver",
                            TickData.market == "mcx",
                            TickData.timestamp >= one_hour_ago,
                        )
                    )
                    .order_by(TickData.timestamp.desc())
                    .limit(1)
                )

                first_price = first_tick.scalar()
                last_price = last_tick.scalar()

                return {
                    "available": True,
                    "tick_count": row.tick_count,
                    "open": float(first_price) if first_price else None,
                    "high": float(row.high) if row.high else None,
                    "low": float(row.low) if row.low else None,
                    "close": float(last_price) if last_price else None,
                    "avg_price": float(row.avg) if row.avg else None,
                    "total_volume": int(row.total_volume) if row.total_volume else 0,
                    "period": "last_hour",
                }

            return {"available": False, "tick_count": 0, "reason": "no_ticks_in_last_hour"}

        except Exception as e:
            logger.warning(f"Failed to get tick stats: {e}")
            return {"available": False, "error": str(e)}

    async def _train_with_enhanced_data(
        self,
        db: AsyncSession,
        asset: str,
        market: str,
        interval: str,
        sentiment_data: Optional[Dict],
        tick_stats: Dict,
    ) -> Dict[str, Any]:
        """
        Train models with enhanced data including sentiment and recent ticks.

        This method:
        1. Gets historical OHLCV data
        2. Adds technical features
        3. Adds historical sentiment features from database
        4. Falls back to current sentiment if no historical data
        5. Trains the ensemble models
        """
        from app.ml.features.technical import add_technical_features
        from app.services.news_sentiment import news_sentiment_service

        logger.debug(f"Training {asset}/{market}/{interval} with enhanced data...")

        # Get historical data
        df = await prediction_engine.get_recent_data(db, asset, market, interval, limit=10000)

        # Minimum samples vary by interval
        min_samples = {
            "30m": 500,
            "1h": 400,
            "4h": 200,
            "1d": 100,
        }
        required = min_samples.get(interval, 500)

        if df.empty or len(df) < required:
            raise ValueError(f"Insufficient data: {len(df)} rows, need at least {required}")

        # Drop columns with all NaN values
        df = df.dropna(axis=1, how='all')

        # Add technical features
        df = add_technical_features(df)

        # Try to add historical sentiment features from database
        historical_sentiment_added = False
        if "timestamp" in df.columns:
            try:
                # Get sentiment features for each row based on its timestamp
                sentiment_features = []
                for _, row in df.iterrows():
                    timestamp = pd.to_datetime(row["timestamp"])
                    features = await news_sentiment_service.get_historical_sentiment_features(
                        db, asset, timestamp, lookback_hours=24
                    )
                    sentiment_features.append(features)

                # Check if we have enough historical sentiment data
                valid_sentiment = [f for f in sentiment_features if f is not None]
                if len(valid_sentiment) >= len(df) * 0.3:  # At least 30% coverage
                    # Add sentiment features to dataframe
                    for i, features in enumerate(sentiment_features):
                        if features:
                            for key, value in features.items():
                                if key not in df.columns:
                                    df[key] = 0.0
                                df.iloc[i, df.columns.get_loc(key)] = value
                    historical_sentiment_added = True
                    logger.info(f"Added historical sentiment to {len(valid_sentiment)}/{len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to add historical sentiment: {e}")

        # Fall back to current sentiment for all rows if no historical data
        if not historical_sentiment_added and sentiment_data:
            df["sentiment_score"] = sentiment_data.get("overall", 0)
            df["sentiment_confidence"] = sentiment_data.get("confidence", 0.5)
            df["news_article_count"] = sentiment_data.get("article_count", 0)
            df["news_bullish_ratio"] = (
                sentiment_data.get("bullish_count", 0) /
                max(sentiment_data.get("article_count", 1), 1)
            )
            df["news_bearish_ratio"] = (
                sentiment_data.get("bearish_count", 0) /
                max(sentiment_data.get("article_count", 1), 1)
            )

            # Sentiment momentum (bullish = 1, bearish = -1, neutral = 0)
            label = sentiment_data.get("label", "neutral")
            if label == "bullish":
                df["sentiment_direction"] = 1
            elif label == "bearish":
                df["sentiment_direction"] = -1
            else:
                df["sentiment_direction"] = 0

        # Add recent tick volatility if available (for MCX)
        if tick_stats.get("available") and market == "mcx":
            # Recent price range as percentage of average
            if tick_stats.get("high") and tick_stats.get("low") and tick_stats.get("avg_price"):
                price_range = tick_stats["high"] - tick_stats["low"]
                df["recent_tick_volatility"] = price_range / tick_stats["avg_price"] * 100
                df["recent_tick_count"] = tick_stats.get("tick_count", 0)

        # Remove rows with NaN from feature calculation
        df = df.dropna()

        if len(df) < required // 2:  # Allow some reduction due to NaN removal
            raise ValueError(f"Too many NaN rows: {len(df)} valid rows after cleaning")

        # Get ensemble and train
        ensemble = prediction_engine.get_ensemble(interval)
        results = ensemble.train_all(df)

        # Save models
        ensemble.save()

        return {
            "status": "success",
            "asset": asset,
            "market": market,
            "interval": interval,
            "training_samples": len(df),
            "features_used": list(df.columns),
            "sentiment_included": sentiment_data is not None or historical_sentiment_added,
            "historical_sentiment": historical_sentiment_added,
            "tick_data_included": tick_stats.get("available", False),
            "results": results,
        }

    async def _generate_predictions_for_interval(self, interval: str):
        """
        Generate predictions for a specific interval for both markets.
        For MCX, generates predictions for 3 contracts (SILVER, SILVERM, SILVERMIC)
        with nearest expiry in ascending order.
        """
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

                # Generate COMEX prediction (single contract)
                try:
                    prediction = await prediction_engine.generate_prediction(
                        db, "silver", "comex", interval
                    )
                    if prediction:
                        results[f"comex_{interval}"] = {
                            "status": "success",
                            "prediction_id": prediction.get("id"),
                            "direction": prediction.get("predicted_direction"),
                            "target_time": prediction.get("target_time"),
                        }
                    else:
                        results[f"comex_{interval}"] = {"status": "no_data"}
                except Exception as e:
                    logger.error(f"Prediction failed for comex/{interval}: {e}")
                    results[f"comex_{interval}"] = {"status": "error", "error": str(e)}

                # Generate MCX predictions for 3 contracts with nearest expiry
                try:
                    from app.services.upstox_client import upstox_client

                    # Get all silver contracts sorted by expiry (ascending)
                    silver_contracts = await upstox_client.get_all_silver_instrument_keys()

                    if not silver_contracts:
                        logger.warning("No MCX silver contracts found, using default prediction")
                        # Fallback to default single prediction
                        prediction = await prediction_engine.generate_prediction(
                            db, "silver", "mcx", interval
                        )
                        if prediction:
                            results[f"mcx_{interval}"] = {
                                "status": "success",
                                "prediction_id": prediction.get("id"),
                                "direction": prediction.get("predicted_direction"),
                                "target_time": prediction.get("target_time"),
                            }
                    else:
                        # Get unique contract types with nearest expiry (max 3)
                        # We want one of each type: SILVER, SILVERM, SILVERMIC
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
                                    "silver",
                                    "mcx",
                                    interval,
                                    instrument_key=contract.get("instrument_key"),
                                    contract_type=contract_type,
                                    trading_symbol=contract.get("trading_symbol"),
                                    expiry=contract.get("expiry"),
                                )
                                if prediction:
                                    results[f"mcx_{contract_type}_{interval}"] = {
                                        "status": "success",
                                        "prediction_id": prediction.get("id"),
                                        "direction": prediction.get("predicted_direction"),
                                        "target_time": prediction.get("target_time"),
                                        "contract_type": contract_type,
                                        "trading_symbol": contract.get("trading_symbol"),
                                    }
                                    logger.info(
                                        f"MCX {contract_type} prediction: {prediction.get('predicted_direction')}"
                                    )
                            except Exception as e:
                                logger.error(f"Prediction failed for mcx/{contract_type}/{interval}: {e}")
                                results[f"mcx_{contract_type}_{interval}"] = {
                                    "status": "error",
                                    "error": str(e),
                                }

                except Exception as e:
                    logger.error(f"Failed to get MCX contracts: {e}")
                    # Fallback to default single prediction
                    try:
                        prediction = await prediction_engine.generate_prediction(
                            db, "silver", "mcx", interval
                        )
                        if prediction:
                            results[f"mcx_{interval}"] = {
                                "status": "success",
                                "prediction_id": prediction.get("id"),
                                "direction": prediction.get("predicted_direction"),
                                "target_time": prediction.get("target_time"),
                            }
                    except Exception as e2:
                        logger.error(f"Fallback MCX prediction also failed: {e2}")
                        results[f"mcx_{interval}"] = {"status": "error", "error": str(e2)}

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
                # Use prediction_engine directly instead of importing from prediction_verifier
                result = await prediction_engine.verify_pending_predictions(db)
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
