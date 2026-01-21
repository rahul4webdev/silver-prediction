"""
Facebook Prophet model for trend and seasonality prediction.
Excellent at capturing weekly, monthly, and yearly patterns.
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from prophet import Prophet

from app.ml.models.base import BaseModel, PredictionResult

logger = logging.getLogger(__name__)


class ProphetModel(BaseModel):
    """
    Prophet model for time series forecasting.

    Strengths:
    - Captures trend, seasonality, and holiday effects
    - Handles missing data gracefully
    - Provides uncertainty intervals naturally
    - Good for longer-term predictions (daily, weekly)

    Limitations:
    - Less effective for very short-term (intraday) predictions
    - Doesn't capture complex non-linear patterns
    """

    def __init__(
        self,
        seasonality_mode: str = "multiplicative",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
    ):
        super().__init__("prophet")

        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality

        self.model: Optional[Prophet] = None
        self.train_df: Optional[pd.DataFrame] = None

    def _prepare_data(self, df: pd.DataFrame, target_col: str = "close") -> pd.DataFrame:
        """
        Prepare data for Prophet (requires 'ds' and 'y' columns).
        """
        prophet_df = pd.DataFrame()

        # Handle timestamp column
        if "timestamp" in df.columns:
            prophet_df["ds"] = pd.to_datetime(df["timestamp"])
        elif "date" in df.columns:
            prophet_df["ds"] = pd.to_datetime(df["date"])
        elif df.index.name == "timestamp" or isinstance(df.index, pd.DatetimeIndex):
            prophet_df["ds"] = df.index
        else:
            raise ValueError("No timestamp column found in DataFrame")

        # Target column
        prophet_df["y"] = df[target_col].values

        # Remove timezone if present (Prophet prefers tz-naive)
        if prophet_df["ds"].dt.tz is not None:
            prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

        # Drop any NaN values
        prophet_df = prophet_df.dropna()

        return prophet_df

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Train Prophet model.

        Args:
            df: DataFrame with timestamp and price data
            target_col: Column to predict
            validation_split: Fraction for validation

        Returns:
            Training metrics
        """
        logger.info(f"Training Prophet model on {len(df)} samples")

        # Prepare data
        prophet_df = self._prepare_data(df, target_col)

        # Split for validation
        split_idx = int(len(prophet_df) * (1 - validation_split))
        train_data = prophet_df.iloc[:split_idx]
        val_data = prophet_df.iloc[split_idx:]

        # Create and fit model
        self.model = Prophet(
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            interval_width=0.95,  # 95% confidence interval
        )

        # Suppress Prophet's verbose output
        self.model.fit(train_data)

        # Validate
        future = self.model.make_future_dataframe(periods=len(val_data), freq="h")
        forecast = self.model.predict(future)

        # Calculate validation metrics
        val_predictions = forecast.iloc[-len(val_data):]["yhat"].values
        val_actual = val_data["y"].values

        mae = np.mean(np.abs(val_predictions - val_actual))
        rmse = np.sqrt(np.mean((val_predictions - val_actual) ** 2))
        mape = np.mean(np.abs((val_actual - val_predictions) / val_actual)) * 100

        # Direction accuracy
        val_direction_actual = np.sign(np.diff(val_actual))
        val_direction_pred = np.sign(np.diff(val_predictions))
        direction_accuracy = np.mean(val_direction_actual == val_direction_pred)

        self.training_metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "direction_accuracy": float(direction_accuracy),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
        }

        self.is_trained = True
        self.last_trained = datetime.now()
        self.train_df = prophet_df

        logger.info(f"Prophet training complete. MAPE: {mape:.2f}%, Direction: {direction_accuracy:.2%}")

        return self.training_metrics

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        current_price: Optional[float] = None,
    ) -> PredictionResult:
        """
        Make prediction with Prophet.

        Args:
            df: Recent data for context
            horizon: Periods ahead to predict
            current_price: Current price for direction calculation

        Returns:
            PredictionResult with prediction and intervals
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        # Prepare recent data
        prophet_df = self._prepare_data(df)

        # Determine frequency from data
        if len(prophet_df) > 1:
            time_diff = prophet_df["ds"].diff().median()
            if time_diff <= timedelta(minutes=30):
                freq = "30T"
            elif time_diff <= timedelta(hours=1):
                freq = "H"
            elif time_diff <= timedelta(hours=4):
                freq = "4H"
            else:
                freq = "D"
        else:
            freq = "H"

        # Create future dataframe
        last_date = prophet_df["ds"].max()
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
        future = pd.DataFrame({"ds": future_dates})

        # Make prediction
        forecast = self.model.predict(future)

        # Get the prediction for target horizon
        pred_row = forecast.iloc[-1]
        predicted_price = float(pred_row["yhat"])

        # Calculate uncertainty from Prophet's intervals
        ci_lower = float(pred_row["yhat_lower"])
        ci_upper = float(pred_row["yhat_upper"])

        # Prophet provides 95% CI by default, estimate std from it
        std_dev = (ci_upper - ci_lower) / (2 * 1.96)

        # Calculate all confidence intervals
        ci_50, ci_80, ci_95 = self.calculate_confidence_intervals(predicted_price, std_dev)

        # Use Prophet's actual 95% CI
        ci_95 = (ci_lower, ci_upper)

        # Determine direction
        if current_price is None:
            current_price = float(prophet_df["y"].iloc[-1])

        direction, direction_prob = self.determine_direction(current_price, predicted_price)

        return PredictionResult(
            predicted_price=predicted_price,
            std_dev=std_dev,
            ci_50=ci_50,
            ci_80=ci_80,
            ci_95=ci_95,
            direction=direction,
            direction_probability=direction_prob,
            model_name=self.model_name,
            prediction_time=datetime.now(),
            target_time=future_dates[-1].to_pydatetime(),
            features_used=["trend", "seasonality"],
        )

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "is_trained": self.is_trained,
                "last_trained": self.last_trained,
                "training_metrics": self.training_metrics,
                "config": {
                    "seasonality_mode": self.seasonality_mode,
                    "changepoint_prior_scale": self.changepoint_prior_scale,
                    "seasonality_prior_scale": self.seasonality_prior_scale,
                },
            }, f)

        logger.info(f"Prophet model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.is_trained = data["is_trained"]
        self.last_trained = data["last_trained"]
        self.training_metrics = data["training_metrics"]

        logger.info(f"Prophet model loaded from {path}")
