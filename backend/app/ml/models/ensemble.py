"""
Ensemble predictor that combines Prophet, LSTM, XGBoost, GRU, and Random Forest models.
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

# Optional imports - these may not be available
try:
    from app.ml.models.prophet_model import ProphetModel
except ImportError:
    ProphetModel = None

try:
    from app.ml.models.lstm_model import LSTMModel
except ImportError:
    LSTMModel = None

try:
    from app.ml.models.xgboost_model import XGBoostModel
except ImportError:
    XGBoostModel = None

try:
    from app.ml.models.gru_model import GRUModel
except ImportError:
    GRUModel = None

try:
    from app.ml.models.random_forest_model import RandomForestModel
except ImportError:
    RandomForestModel = None

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models.

    Features:
    - Weighted voting across Prophet, LSTM, XGBoost, GRU, and Random Forest
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

        # Initialize models (only if available)
        self.prophet = ProphetModel() if ProphetModel else None
        self.lstm = LSTMModel() if LSTMModel else None
        self.xgboost = XGBoostModel() if XGBoostModel else None
        self.gru = GRUModel() if GRUModel else None
        self.random_forest = RandomForestModel() if RandomForestModel else None

        # Log which models are available
        available = []
        if self.prophet: available.append("prophet")
        if self.lstm: available.append("lstm")
        if self.xgboost: available.append("xgboost")
        if self.gru: available.append("gru")
        if self.random_forest: available.append("random_forest")
        logger.info(f"Available models: {available}")

        # Get default weights for interval (expand for new models)
        interval_enum = PredictionInterval(interval) if interval in [e.value for e in PredictionInterval] else PredictionInterval.THIRTY_MIN
        base_weights = INTERVAL_CONFIGS[interval_enum].model_weights.copy()

        # Distribute weights across 5 models
        self.weights = {
            "prophet": base_weights.get("prophet", 0.2),
            "lstm": base_weights.get("lstm", 0.2),
            "xgboost": base_weights.get("xgboost", 0.2),
            "gru": 0.2,
            "random_forest": 0.2,
        }
        # Normalize
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        # Performance tracking for weight adjustment
        self.model_performance: Dict[str, List[float]] = {
            "prophet": [],
            "lstm": [],
            "xgboost": [],
            "gru": [],
            "random_forest": [],
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
        if self.prophet:
            try:
                logger.info("Training Prophet model...")
                results["prophet"] = self.prophet.train(df, target_col, validation_split)
            except Exception as e:
                logger.error(f"Prophet training failed: {e}")
                results["prophet"] = {"error": str(e)}
        else:
            logger.warning("Prophet model not available, skipping")
            results["prophet"] = {"error": "Model not installed"}

        # Train LSTM
        if self.lstm:
            try:
                logger.info("Training LSTM model...")
                results["lstm"] = self.lstm.train(df, target_col, validation_split)
            except Exception as e:
                logger.error(f"LSTM training failed: {e}")
                results["lstm"] = {"error": str(e)}
        else:
            logger.warning("LSTM model not available, skipping")
            results["lstm"] = {"error": "Model not installed"}

        # Train XGBoost
        if self.xgboost:
            try:
                logger.info("Training XGBoost model...")
                results["xgboost"] = self.xgboost.train(df, target_col, validation_split)
            except Exception as e:
                logger.error(f"XGBoost training failed: {e}")
                results["xgboost"] = {"error": str(e)}
        else:
            logger.warning("XGBoost model not available, skipping")
            results["xgboost"] = {"error": "Model not installed"}

        # Train GRU
        if self.gru:
            try:
                logger.info("Training GRU model...")
                results["gru"] = self.gru.train(df, target_col, validation_split)
            except Exception as e:
                logger.error(f"GRU training failed: {e}")
                results["gru"] = {"error": str(e)}
        else:
            logger.warning("GRU model not available, skipping")
            results["gru"] = {"error": "Model not installed"}

        # Train Random Forest
        if self.random_forest:
            try:
                logger.info("Training Random Forest model...")
                results["random_forest"] = self.random_forest.train(df, target_col, validation_split)
            except Exception as e:
                logger.error(f"Random Forest training failed: {e}")
                results["random_forest"] = {"error": str(e)}
        else:
            logger.warning("Random Forest model not available, skipping")
            results["random_forest"] = {"error": "Model not installed"}

        self.is_trained = True
        self.last_trained = datetime.now()

        # Update weights based on training performance
        self._update_weights_from_training(results)

        logger.info(f"Ensemble training complete. Updated weights: {self.weights}")

        return results

    def _update_weights_from_training(self, training_results: Dict[str, Dict[str, float]]) -> None:
        """
        Update model weights based on training performance.

        For returns-based models, we focus on:
        1. Direction accuracy (most important for trading)
        2. MAE (mean absolute error on return predictions)

        Models predicting returns will have MAE in percentage points (e.g., 0.5%).
        """
        scores = {}

        for model_name, metrics in training_results.items():
            if "error" in metrics:
                scores[model_name] = 0.0  # Zero weight for failed models
            else:
                direction_accuracy = metrics.get("direction_accuracy", 0.5)
                mae = metrics.get("mae", 1.0)  # MAE in percentage points for returns

                # For direction-based trading, direction accuracy is paramount
                # A model with 55% direction accuracy is valuable
                # A model with <50% direction accuracy is worse than random

                # Direction score: 50% accuracy = 0, 60% = 0.5, 70% = 1.0
                direction_score = max(0, (direction_accuracy - 0.5) * 5)  # Scale 0.5-0.7 to 0-1

                # MAE score: For returns, MAE < 1% is good, > 2% is poor
                # 0% MAE = 1.0, 1% MAE = 0.5, 2% MAE = 0
                mae_score = max(0, 1 - mae / 2)

                # Weight direction accuracy more heavily (70%) than MAE (30%)
                score = 0.7 * direction_score + 0.3 * mae_score

                # Models with direction accuracy < 50% get minimal weight
                if direction_accuracy < 0.5:
                    score = 0.01
                    logger.warning(f"{model_name} has direction_accuracy={direction_accuracy:.1%}, setting minimal weight")

                scores[model_name] = max(score, 0.01)  # Minimum 1% weight

                logger.info(f"{model_name}: direction={direction_accuracy:.1%}, MAE={mae:.4f}%, score={score:.3f}")

        # Normalize to sum to 1
        total = sum(scores.values())
        if total > 0:
            for model_name in self.weights:
                if model_name in scores:
                    self.weights[model_name] = scores[model_name] / total
                else:
                    self.weights[model_name] = 0.01

        # Ensure weights sum to 1
        total_weights = sum(self.weights.values())
        self.weights = {k: v / total_weights for k, v in self.weights.items()}

        logger.info(f"Updated weights: {self.weights}")

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
            ("gru", self.gru),
            ("random_forest", self.random_forest),
        ]:
            try:
                if model is not None and model.is_trained:
                    pred = model.predict(df, horizon, current_price)
                    predictions[model_name] = pred
                    valid_predictions.append((model_name, pred))
                elif model is None:
                    logger.debug(f"Model {model_name} not available")
                    predictions[model_name] = None
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
            weight = self.weights.get(model_name, 0.2)
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
            ("gru", self.gru),
            ("random_forest", self.random_forest),
        ]:
            if model is None:
                continue
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
            ("gru", self.gru),
            ("random_forest", self.random_forest),
        ]:
            if model is None:
                continue
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
        models_info = {}
        for name, model in [
            ("prophet", self.prophet),
            ("lstm", self.lstm),
            ("xgboost", self.xgboost),
            ("gru", self.gru),
            ("random_forest", self.random_forest),
        ]:
            if model is not None:
                models_info[name] = model.get_model_info()
            else:
                models_info[name] = {"available": False, "error": "Model not installed"}

        return {
            "interval": self.interval,
            "is_trained": self.is_trained,
            "last_trained": self.last_trained.isoformat() if self.last_trained else None,
            "weights": self.weights,
            "models": models_info,
        }
