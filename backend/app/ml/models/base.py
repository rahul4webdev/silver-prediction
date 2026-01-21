"""
Base model interface for all prediction models.
Defines the contract that Prophet, LSTM, and XGBoost models must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PredictionResult:
    """
    Standard prediction result from any model.
    """
    # Point prediction
    predicted_price: float

    # Uncertainty estimates
    std_dev: float
    ci_50: Tuple[float, float]  # 50% confidence interval (lower, upper)
    ci_80: Tuple[float, float]  # 80% confidence interval
    ci_95: Tuple[float, float]  # 95% confidence interval

    # Direction
    direction: str  # bullish, bearish, neutral
    direction_probability: float  # 0.0 to 1.0

    # Metadata
    model_name: str
    prediction_time: datetime
    target_time: datetime
    features_used: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predicted_price": self.predicted_price,
            "std_dev": self.std_dev,
            "ci_50": {"lower": self.ci_50[0], "upper": self.ci_50[1]},
            "ci_80": {"lower": self.ci_80[0], "upper": self.ci_80[1]},
            "ci_95": {"lower": self.ci_95[0], "upper": self.ci_95[1]},
            "direction": self.direction,
            "direction_probability": self.direction_probability,
            "model_name": self.model_name,
            "prediction_time": self.prediction_time.isoformat(),
            "target_time": self.target_time.isoformat(),
            "features_used": self.features_used,
        }


class BaseModel(ABC):
    """
    Abstract base class for all prediction models.

    All models must implement:
    - train(): Train the model on historical data
    - predict(): Make predictions with uncertainty estimates
    - save()/load(): Model persistence
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_trained = False
        self.last_trained: Optional[datetime] = None
        self.training_metrics: Dict[str, float] = {}

    @abstractmethod
    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Train the model on historical data.

        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            validation_split: Fraction of data for validation

        Returns:
            Dict of training metrics (loss, accuracy, etc.)
        """
        pass

    @abstractmethod
    def predict(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        current_price: Optional[float] = None,
    ) -> PredictionResult:
        """
        Make predictions with uncertainty estimates.

        Args:
            df: DataFrame with features (must include recent history)
            horizon: Number of periods to predict ahead
            current_price: Current price for direction calculation

        Returns:
            PredictionResult with prediction and confidence intervals
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass

    def calculate_confidence_intervals(
        self,
        mean: float,
        std: float,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Calculate confidence intervals from mean and standard deviation.

        Uses normal distribution percentiles:
        - 50% CI: +/- 0.6745 std
        - 80% CI: +/- 1.282 std
        - 95% CI: +/- 1.96 std

        Returns:
            Tuple of (ci_50, ci_80, ci_95)
        """
        ci_50 = (mean - 0.6745 * std, mean + 0.6745 * std)
        ci_80 = (mean - 1.282 * std, mean + 1.282 * std)
        ci_95 = (mean - 1.96 * std, mean + 1.96 * std)

        return ci_50, ci_80, ci_95

    def determine_direction(
        self,
        current_price: float,
        predicted_price: float,
        threshold_percent: float = 0.1,
    ) -> Tuple[str, float]:
        """
        Determine price direction and confidence.

        Args:
            current_price: Current price
            predicted_price: Predicted price
            threshold_percent: Minimum change to not be neutral

        Returns:
            Tuple of (direction, probability)
        """
        change_percent = ((predicted_price - current_price) / current_price) * 100

        if abs(change_percent) < threshold_percent:
            return "neutral", 0.5

        direction = "bullish" if change_percent > 0 else "bearish"

        # Simple probability based on change magnitude
        # This is a placeholder - real probability should come from model
        probability = min(0.5 + abs(change_percent) / 10, 0.95)

        return direction, probability

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model_name,
            "is_trained": self.is_trained,
            "last_trained": self.last_trained.isoformat() if self.last_trained else None,
            "training_metrics": self.training_metrics,
        }
