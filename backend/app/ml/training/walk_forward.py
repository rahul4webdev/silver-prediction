"""
Walk-forward optimization and model retraining.
Implements rolling window training to prevent overfitting and adapt to market changes.

Walk-forward validation:
1. Train on window [0, N]
2. Validate on [N, N+k]
3. Roll window forward
4. Repeat until end of data
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import PredictionInterval, INTERVAL_CONFIGS
from app.models.predictions import Prediction
from app.models.price_data import PriceData
from app.ml.models.ensemble import EnsemblePredictor
from app.ml.features.technical import add_technical_features

logger = logging.getLogger(__name__)


class WalkForwardOptimizer:
    """
    Walk-forward optimization for ensemble models.

    Features:
    - Rolling window training
    - Automatic weight adjustment based on recent performance
    - Model drift detection
    - Scheduled retraining triggers
    """

    def __init__(
        self,
        models_path: str = "./data/models",
        train_window_days: int = 180,  # 6 months training
        validation_window_days: int = 30,  # 1 month validation
        retrain_threshold: float = 0.05,  # Retrain if accuracy drops 5%
    ):
        self.models_path = Path(models_path)
        self.train_window_days = train_window_days
        self.validation_window_days = validation_window_days
        self.retrain_threshold = retrain_threshold
        self.performance_history: Dict[str, List[float]] = {}

    async def get_training_data(
        self,
        db: AsyncSession,
        asset: str,
        market: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Get price data for training window."""
        query = (
            select(PriceData)
            .where(
                and_(
                    PriceData.asset == asset,
                    PriceData.market == market,
                    PriceData.interval == interval,
                    PriceData.timestamp >= start_date,
                    PriceData.timestamp <= end_date,
                )
            )
            .order_by(PriceData.timestamp)
        )

        result = await db.execute(query)
        rows = result.scalars().all()

        if not rows:
            return pd.DataFrame()

        data = [row.to_dict() for row in rows]
        return pd.DataFrame(data)

    async def run_walk_forward(
        self,
        db: AsyncSession,
        asset: str = "silver",
        market: str = "mcx",
        interval: str = "30m",
    ) -> Dict[str, Any]:
        """
        Run walk-forward optimization.

        Returns:
            Dict with optimization results and recommended weights
        """
        logger.info(f"Starting walk-forward optimization for {asset}/{market}/{interval}")

        # Get all available data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data

        df = await self.get_training_data(db, asset, market, interval, start_date, end_date)

        if df.empty or len(df) < 500:
            return {"error": "Insufficient data for walk-forward optimization"}

        # Add technical features
        df = df.dropna(axis=1, how='all')
        df = add_technical_features(df)
        df = df.dropna()

        # Calculate walk-forward folds
        train_size = int(len(df) * 0.6)  # 60% for initial training
        val_size = int(len(df) * 0.2)    # 20% for validation
        step_size = int(len(df) * 0.1)   # 10% step size

        results = []
        weights_history = []

        fold = 0
        while train_size + val_size <= len(df):
            logger.info(f"Fold {fold}: train[0:{train_size}], val[{train_size}:{train_size + val_size}]")

            train_df = df.iloc[:train_size].copy()
            val_df = df.iloc[train_size:train_size + val_size].copy()

            # Train ensemble on this fold
            ensemble = EnsemblePredictor(interval=interval, models_path=str(self.models_path))
            train_results = ensemble.train_all(train_df)

            # Validate on holdout set
            val_metrics = self._validate_fold(ensemble, val_df)

            results.append({
                "fold": fold,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "train_results": train_results,
                "val_metrics": val_metrics,
                "weights": ensemble.weights.copy(),
            })

            weights_history.append(ensemble.weights.copy())

            # Move window forward
            train_size += step_size
            fold += 1

        # Calculate optimal weights (average of recent successful folds)
        optimal_weights = self._calculate_optimal_weights(results)

        return {
            "asset": asset,
            "market": market,
            "interval": interval,
            "folds": len(results),
            "results": results,
            "optimal_weights": optimal_weights,
            "timestamp": datetime.now().isoformat(),
        }

    def _validate_fold(
        self,
        ensemble: EnsemblePredictor,
        val_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Validate ensemble on holdout data."""
        if not ensemble.is_trained or len(val_df) < 10:
            return {"error": "Cannot validate"}

        predictions = []
        actuals = []

        # Make predictions for each row
        for i in range(50, len(val_df)):
            try:
                context = val_df.iloc[:i].copy()
                actual_next = float(val_df.iloc[i]["close"])

                pred = ensemble.predict(context, horizon=1, current_price=float(context["close"].iloc[-1]))
                predicted = pred["ensemble"]["predicted_price"]
                predicted_direction = pred["ensemble"]["direction"]

                # Calculate actual direction
                current = float(context["close"].iloc[-1])
                actual_direction = "bullish" if actual_next > current else "bearish"

                predictions.append({
                    "predicted": predicted,
                    "predicted_direction": predicted_direction,
                    "actual": actual_next,
                    "actual_direction": actual_direction,
                    "correct": predicted_direction == actual_direction,
                })

            except Exception as e:
                continue

        if not predictions:
            return {"error": "No valid predictions"}

        # Calculate metrics
        correct = sum(1 for p in predictions if p["correct"])
        direction_accuracy = correct / len(predictions)

        errors = [abs(p["predicted"] - p["actual"]) / p["actual"] * 100 for p in predictions]
        mape = np.mean(errors)

        return {
            "direction_accuracy": direction_accuracy,
            "mape": mape,
            "total_predictions": len(predictions),
            "correct_predictions": correct,
        }

    def _calculate_optimal_weights(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Calculate optimal weights based on walk-forward results.

        Uses exponential weighting to favor more recent folds.
        """
        if not results:
            return {}

        # Filter successful folds
        valid_results = [r for r in results if "error" not in r.get("val_metrics", {})]

        if not valid_results:
            return {}

        # Weight recent folds more heavily
        decay = 0.9
        weights_sum = {}
        total_weight = 0

        for i, result in enumerate(valid_results):
            fold_weight = decay ** (len(valid_results) - i - 1)
            fold_accuracy = result.get("val_metrics", {}).get("direction_accuracy", 0.5)

            # Only consider folds with above-random accuracy
            if fold_accuracy > 0.5:
                for model, model_weight in result.get("weights", {}).items():
                    if model not in weights_sum:
                        weights_sum[model] = 0
                    weights_sum[model] += model_weight * fold_weight * fold_accuracy
                    total_weight += fold_weight * fold_accuracy

        # Normalize
        if total_weight > 0:
            return {k: v / total_weight for k, v in weights_sum.items()}

        return {}

    async def check_model_drift(
        self,
        db: AsyncSession,
        asset: str = "silver",
        interval: str = "30m",
        lookback_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Check if model performance has degraded (model drift).

        Returns:
            Dict with drift analysis and retraining recommendation
        """
        since = datetime.utcnow() - timedelta(days=lookback_days)

        # Get recent verified predictions
        query = (
            select(Prediction)
            .where(
                and_(
                    Prediction.asset == asset,
                    Prediction.interval == interval,
                    Prediction.verified_at.isnot(None),
                    Prediction.verified_at >= since,
                )
            )
            .order_by(Prediction.verified_at.desc())
        )

        result = await db.execute(query)
        predictions = result.scalars().all()

        if len(predictions) < 10:
            return {
                "status": "insufficient_data",
                "message": f"Only {len(predictions)} predictions in last {lookback_days} days",
                "retrain_recommended": False,
            }

        # Calculate recent accuracy
        correct = sum(1 for p in predictions if p.is_direction_correct)
        recent_accuracy = correct / len(predictions)

        # Compare to baseline (50% is random)
        baseline = 0.55  # Our target minimum

        # Get historical accuracy
        interval_key = f"{asset}_{interval}"
        if interval_key not in self.performance_history:
            self.performance_history[interval_key] = []

        self.performance_history[interval_key].append(recent_accuracy)

        # Calculate trend
        history = self.performance_history[interval_key]
        if len(history) >= 3:
            recent_avg = np.mean(history[-3:])
            older_avg = np.mean(history[:-3]) if len(history) > 3 else recent_avg
            trend = recent_avg - older_avg
        else:
            trend = 0

        # Determine if retraining is needed
        drift_detected = recent_accuracy < baseline or trend < -self.retrain_threshold
        retrain_recommended = drift_detected

        return {
            "status": "drift_detected" if drift_detected else "stable",
            "recent_accuracy": recent_accuracy,
            "baseline": baseline,
            "trend": trend,
            "predictions_analyzed": len(predictions),
            "lookback_days": lookback_days,
            "retrain_recommended": retrain_recommended,
            "message": (
                f"Performance degraded to {recent_accuracy:.1%}, retraining recommended"
                if retrain_recommended
                else f"Performance stable at {recent_accuracy:.1%}"
            ),
        }

    async def schedule_retrain_if_needed(
        self,
        db: AsyncSession,
        asset: str = "silver",
        market: str = "mcx",
        interval: str = "30m",
    ) -> Optional[Dict[str, Any]]:
        """
        Check drift and retrain if needed.

        Returns:
            Training results if retrained, None otherwise
        """
        drift = await self.check_model_drift(db, asset, interval)

        if not drift.get("retrain_recommended"):
            logger.info(f"No retraining needed for {asset}/{interval}: {drift['message']}")
            return None

        logger.info(f"Retraining triggered for {asset}/{interval}: {drift['message']}")

        # Run walk-forward optimization
        from app.services.prediction_engine import prediction_engine
        results = await prediction_engine.train_models(db, asset, market, interval)

        # Send notification if configured
        try:
            from app.services.notifications import telegram_service
            await telegram_service.send_model_retrain_notification(results)
        except Exception as e:
            logger.warning(f"Failed to send retrain notification: {e}")

        return results


class ConfidenceCalibrator:
    """
    Calibrate prediction confidence intervals.

    Ensures that stated confidence intervals actually contain
    the true percentage of outcomes (e.g., 80% CI should contain 80%).
    """

    def __init__(self):
        self.calibration_data: Dict[str, List[Dict]] = {}

    async def collect_calibration_data(
        self,
        db: AsyncSession,
        asset: str = "silver",
        interval: str = "30m",
        min_samples: int = 50,
    ) -> Dict[str, Any]:
        """
        Collect data for calibration analysis.
        """
        # Get verified predictions
        query = (
            select(Prediction)
            .where(
                and_(
                    Prediction.asset == asset,
                    Prediction.interval == interval,
                    Prediction.verified_at.isnot(None),
                )
            )
            .order_by(Prediction.verified_at.desc())
            .limit(500)
        )

        result = await db.execute(query)
        predictions = result.scalars().all()

        if len(predictions) < min_samples:
            return {
                "status": "insufficient_data",
                "samples": len(predictions),
                "min_required": min_samples,
            }

        # Calculate actual coverage for each CI
        within_50 = sum(1 for p in predictions if p.within_ci_50)
        within_80 = sum(1 for p in predictions if p.within_ci_80)
        within_95 = sum(1 for p in predictions if p.within_ci_95)

        total = len(predictions)

        actual_coverage = {
            "ci_50": within_50 / total,
            "ci_80": within_80 / total,
            "ci_95": within_95 / total,
        }

        target_coverage = {
            "ci_50": 0.50,
            "ci_80": 0.80,
            "ci_95": 0.95,
        }

        # Calculate calibration error
        calibration_errors = {
            ci: actual_coverage[ci] - target_coverage[ci]
            for ci in target_coverage
        }

        # Determine if calibration adjustment needed
        needs_calibration = any(abs(e) > 0.05 for e in calibration_errors.values())

        # Calculate adjustment factors
        # If actual coverage is too low, we need to widen intervals
        adjustment_factors = {}
        for ci in target_coverage:
            if actual_coverage[ci] < target_coverage[ci]:
                # Intervals too narrow, widen them
                adjustment_factors[ci] = target_coverage[ci] / max(actual_coverage[ci], 0.01)
            else:
                # Intervals too wide, narrow them
                adjustment_factors[ci] = target_coverage[ci] / actual_coverage[ci]

        return {
            "status": "calibration_complete",
            "samples": total,
            "actual_coverage": actual_coverage,
            "target_coverage": target_coverage,
            "calibration_errors": calibration_errors,
            "needs_calibration": needs_calibration,
            "adjustment_factors": adjustment_factors,
            "recommendation": (
                "Widen confidence intervals" if any(e < -0.05 for e in calibration_errors.values())
                else "Narrow confidence intervals" if any(e > 0.05 for e in calibration_errors.values())
                else "Calibration is acceptable"
            ),
        }


# Singleton instances
walk_forward_optimizer = WalkForwardOptimizer()
confidence_calibrator = ConfidenceCalibrator()
