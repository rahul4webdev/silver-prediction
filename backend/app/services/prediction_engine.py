"""
Prediction engine service that orchestrates the prediction workflow.
Handles data fetching, model prediction, storage, and verification.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import Asset, Market, PredictionInterval, INTERVAL_CONFIGS
from app.models.predictions import Prediction
from app.models.price_data import PriceData
from app.ml.models.ensemble import EnsemblePredictor
from app.ml.features.technical import add_technical_features

# Optional sentiment imports
try:
    from app.services.news_sentiment import news_sentiment_service
    from app.ml.features.sentiment import sentiment_feature_engine
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# Optional macro data imports
try:
    from app.services.macro_data import macro_data_service
    MACRO_AVAILABLE = True
except ImportError:
    MACRO_AVAILABLE = False

logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Main prediction engine that:
    1. Fetches recent price data
    2. Adds technical features
    3. Runs ensemble prediction
    4. Stores prediction to database
    5. Verifies past predictions
    """

    def __init__(self, models_path: str = "./data/models"):
        self.models_path = models_path
        self.ensembles: Dict[str, EnsemblePredictor] = {}

    def get_ensemble(self, interval: str) -> EnsemblePredictor:
        """Get or create ensemble for interval."""
        if interval not in self.ensembles:
            self.ensembles[interval] = EnsemblePredictor(
                interval=interval,
                models_path=self.models_path,
            )
            # Try to load existing models
            try:
                self.ensembles[interval].load()
            except Exception as e:
                logger.warning(f"Could not load models for {interval}: {e}")

        return self.ensembles[interval]

    async def get_recent_data(
        self,
        db: AsyncSession,
        asset: str,
        market: str,
        interval: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Get recent price data from database.

        Args:
            db: Database session
            asset: Asset code
            market: Market code
            interval: Data interval
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        query = (
            select(PriceData)
            .where(
                and_(
                    PriceData.asset == asset,
                    PriceData.market == market,
                    PriceData.interval == interval,
                )
            )
            .order_by(PriceData.timestamp.desc())
            .limit(limit)
        )

        result = await db.execute(query)
        rows = result.scalars().all()

        if not rows:
            return pd.DataFrame()

        # Convert to DataFrame
        data = [row.to_dict() for row in rows]
        df = pd.DataFrame(data)

        # Sort by timestamp (oldest first)
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    async def get_sentiment_features(self, asset: str = "silver") -> Dict[str, float]:
        """
        Get sentiment features for an asset.

        Args:
            asset: Asset to get sentiment for

        Returns:
            Dict of sentiment features
        """
        if not SENTIMENT_AVAILABLE:
            return {}

        try:
            sentiment = await news_sentiment_service.get_sentiment(asset=asset)
            features = news_sentiment_service.sentiment_to_features(sentiment)
            logger.info(f"Sentiment features for {asset}: {features}")
            return features
        except Exception as e:
            logger.warning(f"Failed to get sentiment features: {e}")
            return {}

    async def generate_prediction(
        self,
        db: AsyncSession,
        asset: str = "silver",
        market: str = "mcx",
        interval: str = "30m",
        horizon: int = 1,
        include_sentiment: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a prediction and store it.

        Args:
            db: Database session
            asset: Asset to predict
            market: Market (mcx or comex)
            interval: Prediction interval
            horizon: Periods ahead
            include_sentiment: Whether to fetch and include sentiment features

        Returns:
            Prediction result dict
        """
        logger.info(f"Generating {interval} prediction for {asset}/{market}")

        # Get recent data
        df = await self.get_recent_data(db, asset, market, interval)

        if df.empty:
            raise ValueError(f"No data available for {asset}/{market}/{interval}")

        if len(df) < 100:
            raise ValueError(f"Insufficient data: {len(df)} rows, need at least 100")

        # Drop columns with all NaN values (like open_interest, vwap)
        df = df.dropna(axis=1, how='all')

        # Add technical features
        df = add_technical_features(df)

        # Add sentiment features (optional)
        sentiment_data = {}
        if include_sentiment and SENTIMENT_AVAILABLE:
            try:
                sentiment = await news_sentiment_service.get_sentiment(asset=asset)
                sentiment_data = {
                    "overall_sentiment": sentiment.overall_sentiment,
                    "sentiment_label": sentiment.sentiment_label,
                    "confidence": sentiment.confidence,
                    "article_count": sentiment.article_count,
                }
                # Add sentiment features to dataframe
                features = news_sentiment_service.sentiment_to_features(sentiment)
                for feature_name, value in features.items():
                    df[feature_name] = value
                logger.info(f"Added sentiment features: {list(features.keys())}")
            except Exception as e:
                logger.warning(f"Failed to add sentiment features: {e}")

        # Add macro data features (DXY, Fear & Greed, etc.)
        macro_data = {}
        if MACRO_AVAILABLE:
            try:
                macro = await macro_data_service.get_all_macro_data(asset=asset)
                macro_data = {
                    "dxy": macro.get("dxy", {}).get("current"),
                    "dxy_trend": macro.get("dxy", {}).get("trend"),
                    "fear_greed": macro.get("fear_greed", {}).get("current"),
                    "fear_greed_label": macro.get("fear_greed", {}).get("classification"),
                    "macro_signal": macro.get("composite_signal", {}).get("direction"),
                    "macro_score": macro.get("composite_signal", {}).get("score"),
                }
                # Add macro features to dataframe
                macro_features = macro_data_service.macro_to_features(macro)
                for feature_name, value in macro_features.items():
                    df[feature_name] = value
                logger.info(f"Added macro features: {list(macro_features.keys())}")
            except Exception as e:
                logger.warning(f"Failed to add macro features: {e}")

        # Remove rows with NaN from feature calculation (keeps rows with 200+ candles of history)
        df = df.dropna()

        # Get current price (latest from data)
        current_price = float(df["close"].iloc[-1])

        # Use actual current time for prediction, not data timestamp
        # This ensures target_time is always in the future
        prediction_time = datetime.now()

        # Get ensemble and predict
        ensemble = self.get_ensemble(interval)

        if not ensemble.is_trained:
            raise RuntimeError(f"Ensemble for {interval} not trained. Run training first.")

        prediction_result = ensemble.predict(df, horizon, current_price)
        ensemble_pred = prediction_result["ensemble"]

        # Calculate target time from NOW (when prediction is made)
        interval_minutes = INTERVAL_CONFIGS[PredictionInterval(interval)].minutes
        target_time = prediction_time + timedelta(minutes=interval_minutes * horizon)

        # Create prediction record
        prediction = Prediction(
            asset=asset,
            market=market,
            interval=interval,
            prediction_time=prediction_time,
            target_time=target_time,
            current_price=Decimal(str(current_price)),
            predicted_price=Decimal(str(ensemble_pred["predicted_price"])),
            predicted_direction=ensemble_pred["direction"],
            direction_confidence=Decimal(str(ensemble_pred["direction_probability"])),
            ci_50_lower=Decimal(str(ensemble_pred["ci_50"]["lower"])),
            ci_50_upper=Decimal(str(ensemble_pred["ci_50"]["upper"])),
            ci_80_lower=Decimal(str(ensemble_pred["ci_80"]["lower"])),
            ci_80_upper=Decimal(str(ensemble_pred["ci_80"]["upper"])),
            ci_95_lower=Decimal(str(ensemble_pred["ci_95"]["lower"])),
            ci_95_upper=Decimal(str(ensemble_pred["ci_95"]["upper"])),
            model_weights=prediction_result["model_weights"],
            features_used={
                "interval": interval,
                "data_points": len(df),
                "sentiment_included": bool(sentiment_data),
                "macro_included": bool(macro_data),
            },
        )

        # Include sentiment and macro in response
        prediction_dict = prediction.to_dict()
        if sentiment_data:
            prediction_dict["sentiment"] = sentiment_data
        if macro_data:
            prediction_dict["macro"] = macro_data

        # Store prediction
        db.add(prediction)
        await db.commit()
        await db.refresh(prediction)

        logger.info(
            f"Prediction saved: {prediction.predicted_direction} "
            f"({prediction.predicted_price}) with {prediction.direction_confidence:.2%} confidence"
        )

        return prediction_dict

    async def verify_pending_predictions(
        self,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """
        Verify all pending predictions whose target_time has passed.

        Returns:
            Summary of verified predictions
        """
        logger.info("Verifying pending predictions...")

        # Get unverified predictions where target_time < now
        query = (
            select(Prediction)
            .where(
                and_(
                    Prediction.verified_at.is_(None),
                    Prediction.target_time < datetime.utcnow(),
                )
            )
            .limit(100)  # Process in batches
        )

        result = await db.execute(query)
        pending = result.scalars().all()

        if not pending:
            return {"verified": 0, "message": "No pending predictions to verify"}

        verified_count = 0
        correct_count = 0

        for prediction in pending:
            try:
                # Get actual price at target time
                actual_price = await self._get_price_at_time(
                    db,
                    prediction.asset,
                    prediction.market,
                    prediction.interval,
                    prediction.target_time,
                )

                if actual_price is None:
                    logger.warning(
                        f"No price data for verification: {prediction.id}"
                    )
                    continue

                # Verify the prediction
                prediction.verify(actual_price)
                verified_count += 1

                if prediction.is_direction_correct:
                    correct_count += 1

                # Update ensemble performance
                ensemble = self.get_ensemble(prediction.interval)
                # Update each model's performance (simplified)
                for model_name in ensemble.weights:
                    ensemble.update_performance(
                        model_name,
                        prediction.is_direction_correct,
                    )

                logger.info(
                    f"Verified prediction {prediction.id[:8]}: "
                    f"{'correct' if prediction.is_direction_correct else 'wrong'} "
                    f"(error: {prediction.price_error_percent:.2f}%)"
                )

            except Exception as e:
                logger.error(f"Error verifying prediction {prediction.id}: {e}")

        await db.commit()

        return {
            "verified": verified_count,
            "correct": correct_count,
            "accuracy": correct_count / verified_count if verified_count > 0 else 0,
        }

    async def _get_price_at_time(
        self,
        db: AsyncSession,
        asset: str,
        market: str,
        interval: str,
        target_time: datetime,
    ) -> Optional[float]:
        """
        Get the price at or closest to the target time.
        """
        # Look for price within a time window
        time_tolerance = timedelta(minutes=60)

        query = (
            select(PriceData)
            .where(
                and_(
                    PriceData.asset == asset,
                    PriceData.market == market,
                    PriceData.interval == interval,
                    PriceData.timestamp >= target_time - time_tolerance,
                    PriceData.timestamp <= target_time + time_tolerance,
                )
            )
            .order_by(func.abs(func.extract('epoch', PriceData.timestamp - target_time)))
            .limit(1)
        )

        result = await db.execute(query)
        price_data = result.scalar_one_or_none()

        if price_data:
            return float(price_data.close)

        return None

    async def get_accuracy_summary(
        self,
        db: AsyncSession,
        asset: str = "silver",
        market: Optional[str] = None,
        interval: Optional[str] = None,
        period_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get prediction accuracy summary.

        Args:
            db: Database session
            asset: Asset to filter
            market: Market to filter (optional)
            interval: Interval to filter (optional)
            period_days: Number of days to analyze

        Returns:
            Accuracy metrics dict
        """
        since = datetime.utcnow() - timedelta(days=period_days)

        # Build filter conditions
        conditions = [
            Prediction.asset == asset,
            Prediction.verified_at.isnot(None),
            Prediction.created_at >= since,
        ]

        if market:
            conditions.append(Prediction.market == market)
        if interval:
            conditions.append(Prediction.interval == interval)

        # Get verified predictions
        query = select(Prediction).where(and_(*conditions))
        result = await db.execute(query)
        predictions = result.scalars().all()

        if not predictions:
            return {
                "total_predictions": 0,
                "message": "No verified predictions in period",
            }

        # Calculate metrics
        total = len(predictions)
        correct = sum(1 for p in predictions if p.is_direction_correct)
        within_ci_50 = sum(1 for p in predictions if p.within_ci_50)
        within_ci_80 = sum(1 for p in predictions if p.within_ci_80)
        within_ci_95 = sum(1 for p in predictions if p.within_ci_95)

        errors = [float(p.price_error_percent) for p in predictions if p.price_error_percent]
        mae = sum(abs(e) for e in errors) / len(errors) if errors else 0

        # By interval breakdown
        by_interval = {}
        for p in predictions:
            if p.interval not in by_interval:
                by_interval[p.interval] = {"total": 0, "correct": 0}
            by_interval[p.interval]["total"] += 1
            if p.is_direction_correct:
                by_interval[p.interval]["correct"] += 1

        for interval_data in by_interval.values():
            interval_data["accuracy"] = (
                interval_data["correct"] / interval_data["total"]
                if interval_data["total"] > 0 else 0
            )

        # By market breakdown
        by_market = {}
        for p in predictions:
            if p.market not in by_market:
                by_market[p.market] = {"total": 0, "correct": 0}
            by_market[p.market]["total"] += 1
            if p.is_direction_correct:
                by_market[p.market]["correct"] += 1

        for market_data in by_market.values():
            market_data["accuracy"] = (
                market_data["correct"] / market_data["total"]
                if market_data["total"] > 0 else 0
            )

        return {
            "total_predictions": total,
            "verified_predictions": total,
            "period_days": period_days,
            "direction_accuracy": {
                "overall": correct / total,
                "correct": correct,
                "wrong": total - correct,
            },
            "confidence_interval_coverage": {
                "ci_50": within_ci_50 / total,
                "ci_80": within_ci_80 / total,
                "ci_95": within_ci_95 / total,
            },
            "error_metrics": {
                "mape": mae,
            },
            "by_interval": by_interval,
            "by_market": by_market,
        }

    async def train_models(
        self,
        db: AsyncSession,
        asset: str = "silver",
        market: str = "mcx",
        interval: str = "30m",
    ) -> Dict[str, Any]:
        """
        Train ensemble models on historical data.

        Args:
            db: Database session
            asset: Asset to train for
            market: Market
            interval: Interval

        Returns:
            Training results
        """
        logger.info(f"Training models for {asset}/{market}/{interval}")

        # Get all historical data
        df = await self.get_recent_data(db, asset, market, interval, limit=10000)

        if df.empty or len(df) < 500:
            raise ValueError(f"Insufficient data for training: {len(df)} rows")

        # Drop columns with all NaN values (like open_interest, vwap)
        df = df.dropna(axis=1, how='all')

        # Add technical features
        df = add_technical_features(df)

        # Remove rows with NaN from feature calculation (keeps rows with 200+ candles of history)
        df = df.dropna()

        # Get ensemble and train
        ensemble = self.get_ensemble(interval)
        results = ensemble.train_all(df)

        # Save models
        ensemble.save()

        return {
            "asset": asset,
            "market": market,
            "interval": interval,
            "training_samples": len(df),
            "results": results,
        }


# Singleton instance
prediction_engine = PredictionEngine()
