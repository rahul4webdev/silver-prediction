"""
Ensemble predictor that combines Prophet, LSTM, and XGBoost models.
Uses weighted voting with dynamic weight adjustment based on recent performance.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.constants import INTERVAL_CONFIGS, PredictionInterval
from app.ml.models.base import BaseModel, PredictionResult
from app.ml.models.prophet_model import ProphetModel
from app.ml.models.lstm_model import LSTMModel
from app.ml.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models.

    Features:
    - Weighted voting across Prophet, LSTM, and XGBoost
    - Dynamic weight adjustment based on recent accuracy
    - Probability distribution output with confidence intervals
    - Direction probability calculation
    """

    def __init__(
        self,
        interval: str = "30m",
        models_path: str = "./data/models",
    ):
        self.interval = interval
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self.prophet = ProphetModel()
        self.lstm = LSTMModel()
        self.xgboost = XGBoostModel()

        # Get default weights for interval
        interval_enum = PredictionInterval(interval) if interval in [e.value for e in PredictionInterval] else PredictionInterval.THIRTY_MIN
        self.weights = INTERVAL_CONFIGS[interval_enum].model_weights.copy()

        # Performance tracking for weight adjustment
        self.model_performance: Dict[str, List[float]] = {
            "prophet": [],
            "lstm": [],
            "xgboost": [],
        }

        self.is_trained = False
        self.last_trained: Optional[datetime] = None

    def train_all(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        validation_split: float = 0.2,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all models in the ensemble.

        Args:
            df: DataFrame with OHLCV and features
            target_col: Column to predict
            validation_split: Fraction for validation

        Returns:
            Dict of model metrics
        """
        logger.info(f"Training ensemble on {len(df)} samples")

        results = {}

        # Train Prophet
        try:
            logger.info("Training Prophet model...")
            results["prophet"] = self.prophet.train(df, target_col, validation_split)
        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            results["prophet"] = {"error": str(e)}

        # Train LSTM (needs features)
        try:
            logger.info("Training LSTM model...")
            results["lstm"] = self.lstm.train(df, target_col, validation_split)
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            results["lstm"] = {"error": str(e)}

        # Train XGBoost
        try:
            logger.info("Training XGBoost model...")
            results["xgboost"] = self.xgboost.train(df, target_col, validation_split)
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            results["xgboost"] = {"error": str(e)}

        self.is_trained = True
        self.last_trained = datetime.now()

        # Update weights based on training performance
        self._update_weights_from_training(results)

        logger.info(f"Ensemble training complete. Updated weights: {self.weights}")

        return results

    def _update_weights_from_training(self, training_results: Dict[str, Dict[str, float]]) -> None:
        """
        Update model weights based on training performance.

        Uses direction accuracy as the primary metric.
        """
        accuracies = {}

        for model_name, metrics in training_results.items():
            if "error" in metrics:
                accuracies[model_name] = 0.5  # Default for failed models
            else:
                accuracies[model_name] = metrics.get("direction_accuracy", 0.5)

        # Normalize to sum to 1
        total = sum(accuracies.values())
        if total > 0:
            for model_name in self.weights:
                if model_name in accuracies:
                    # Blend with default weights (50% each)
                    new_weight = (self.weights[model_name] + accuracies[model_name] / total) / 2
                    self.weights[model_name] = new_weight

        # Ensure weights sum to 1
        total_weights = sum(self.weights.values())
        self.weights = {k: v / total_weights for k, v in self.weights.items()}

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        current_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Make ensemble prediction.

        Args:
            df: Recent data for prediction
            horizon: Periods ahead to predict
            current_price: Current price for direction calculation

        Returns:
            Dict with ensemble prediction and individual model predictions
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble must be trained before prediction")

        if current_price is None:
            current_price = float(df["close"].iloc[-1])

        predictions = {}
        valid_predictions = []

        # Get predictions from each model
        for model_name, model in [
            ("prophet", self.prophet),
            ("lstm", self.lstm),
            ("xgboost", self.xgboost),
        ]:
            try:
                if model.is_trained:
                    pred = model.predict(df, horizon, current_price)
                    predictions[model_name] = pred
                    valid_predictions.append((model_name, pred))
            except Exception as e:
                logger.error(f"Prediction failed for {model_name}: {e}")
                predictions[model_name] = None

        if not valid_predictions:
            raise RuntimeError("All model predictions failed")

        # Calculate weighted ensemble prediction
        ensemble_price = 0.0
        ensemble_std = 0.0
        direction_votes = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
        total_weight = 0.0

        for model_name, pred in valid_predictions:
            weight = self.weights.get(model_name, 0.33)
            ensemble_price += weight * pred.predicted_price
            ensemble_std += weight * pred.std_dev
            direction_votes[pred.direction] += weight * pred.direction_probability
            total_weight += weight

        # Normalize
        if total_weight > 0:
            ensemble_price /= total_weight
            ensemble_std /= total_weight

        # Determine ensemble direction
        ensemble_direction = max(direction_votes, key=direction_votes.get)
        ensemble_direction_prob = direction_votes[ensemble_direction] / sum(direction_votes.values()) if sum(direction_votes.values()) > 0 else 0.5

        # Calculate confidence intervals from ensemble
        ci_50 = (ensemble_price - 0.6745 * ensemble_std, ensemble_price + 0.6745 * ensemble_std)
        ci_80 = (ensemble_price - 1.282 * ensemble_std, ensemble_price + 1.282 * ensemble_std)
        ci_95 = (ensemble_price - 1.96 * ensemble_std, ensemble_price + 1.96 * ensemble_std)

        # Calculate target time
        if "timestamp" in df.columns:
            last_time = pd.to_datetime(df["timestamp"].iloc[-1])
        else:
            last_time = datetime.now()

        # Determine time delta based on interval
        interval_minutes = {
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "daily": 1440,
        }.get(self.interval, 60)

        target_time = last_time + timedelta(minutes=interval_minutes * horizon)

        return {
            "ensemble": {
                "predicted_price": ensemble_price,
                "std_dev": ensemble_std,
                "ci_50": {"lower": ci_50[0], "upper": ci_50[1]},
                "ci_80": {"lower": ci_80[0], "upper": ci_80[1]},
                "ci_95": {"lower": ci_95[0], "upper": ci_95[1]},
                "direction": ensemble_direction,
                "direction_probability": ensemble_direction_prob,
                "current_price": current_price,
                "prediction_time": datetime.now().isoformat(),
                "target_time": target_time.isoformat(),
            },
            "model_weights": self.weights,
            "individual_predictions": {
                name: pred.to_dict() if pred else None
                for name, pred in predictions.items()
            },
        }

    def update_performance(
        self,
        model_name: str,
        was_correct: bool,
        window_size: int = 50,
    ) -> None:
        """
        Update model performance for weight adjustment.

        Args:
            model_name: Name of model
            was_correct: Whether prediction was correct
            window_size: Number of recent predictions to track
        """
        if model_name not in self.model_performance:
            return

        self.model_performance[model_name].append(1.0 if was_correct else 0.0)

        # Keep only recent performance
        if len(self.model_performance[model_name]) > window_size:
            self.model_performance[model_name] = self.model_performance[model_name][-window_size:]

        # Recalculate weights if we have enough data
        if all(len(v) >= 10 for v in self.model_performance.values()):
            self._recalculate_weights()

    def _recalculate_weights(self) -> None:
        """
        Recalculate model weights based on recent performance.
        """
        accuracies = {}
        for model_name, performance in self.model_performance.items():
            if performance:
                accuracies[model_name] = np.mean(performance)
            else:
                accuracies[model_name] = 0.5

        # Softmax-style weighting
        total = sum(np.exp(acc * 2) for acc in accuracies.values())  # Temperature = 0.5
        for model_name in self.weights:
            if model_name in accuracies:
                self.weights[model_name] = np.exp(accuracies[model_name] * 2) / total

        logger.info(f"Updated ensemble weights: {self.weights}")

    def save(self, suffix: str = "") -> None:
        """Save all models."""
        for model_name, model in [
            ("prophet", self.prophet),
            ("lstm", self.lstm),
            ("xgboost", self.xgboost),
        ]:
            path = self.models_path / f"{model_name}_{self.interval}{suffix}.pkl"
            try:
                model.save(str(path))
            except Exception as e:
                logger.error(f"Failed to save {model_name}: {e}")

        # Save ensemble metadata
        import json
        meta_path = self.models_path / f"ensemble_{self.interval}{suffix}_meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "weights": self.weights,
                "interval": self.interval,
                "is_trained": self.is_trained,
                "last_trained": self.last_trained.isoformat() if self.last_trained else None,
                "model_performance": self.model_performance,
            }, f)

        logger.info(f"Ensemble saved to {self.models_path}")

    def load(self, suffix: str = "") -> None:
        """Load all models."""
        for model_name, model in [
            ("prophet", self.prophet),
            ("lstm", self.lstm),
            ("xgboost", self.xgboost),
        ]:
            path = self.models_path / f"{model_name}_{self.interval}{suffix}.pkl"
            try:
                if path.exists():
                    model.load(str(path))
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")

        # Load ensemble metadata
        import json
        meta_path = self.models_path / f"ensemble_{self.interval}{suffix}_meta.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.weights = meta.get("weights", self.weights)
                self.is_trained = meta.get("is_trained", False)
                self.model_performance = meta.get("model_performance", self.model_performance)
                if meta.get("last_trained"):
                    self.last_trained = datetime.fromisoformat(meta["last_trained"])

        logger.info(f"Ensemble loaded from {self.models_path}")

    def get_info(self) -> Dict[str, Any]:
        """Get ensemble information."""
        return {
            "interval": self.interval,
            "is_trained": self.is_trained,
            "last_trained": self.last_trained.isoformat() if self.last_trained else None,
            "weights": self.weights,
            "models": {
                "prophet": self.prophet.get_model_info(),
                "lstm": self.lstm.get_model_info(),
                "xgboost": self.xgboost.get_model_info(),
            },
        }
