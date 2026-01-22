"""
Facebook Prophet model for trend and seasonality prediction.
Excellent at capturing weekly, monthly, and yearly patterns.

NOTE: This model predicts percentage returns, not absolute prices.
This works better for commodities like silver where the absolute
price level changes over time.
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet

from app.ml.models.base import BaseModel, PredictionResult

logger = logging.getLogger(__name__)


class ProphetModel(BaseModel):
    """
    Prophet model for time series forecasting.

    Predicts percentage returns instead of absolute prices for better
    performance on commodities.

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
        seasonality_mode: str = "additive",  # Additive is better for returns
        changepoint_prior_scale: float = 0.1,  # More flexible for volatile markets
        seasonality_prior_scale: float = 5.0,
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = False,  # Not enough data usually
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
        self._last_price: Optional[float] = None

    def _prepare_data(self, df: pd.DataFrame, target_col: str = "close") -> Tuple[pd.DataFrame, float]:
        """
        Prepare data for Prophet - converts to percentage returns.

        Returns:
            Tuple of (prophet_df, last_price)
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

        # Get price data
        prices = df[target_col].values
        last_price = float(prices[-1])

        # Calculate percentage returns (we predict returns, not absolute prices)
        # This normalizes the scale and works better for volatile commodities
        returns = np.diff(prices) / prices[:-1] * 100  # Percentage returns

        # Align timestamps (first row has no return)
        prophet_df = prophet_df.iloc[1:].copy()
        prophet_df["y"] = returns

        # Remove timezone if present (Prophet prefers tz-naive)
        if prophet_df["ds"].dt.tz is not None:
            prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

        # Drop any NaN values
        prophet_df = prophet_df.dropna()

        return prophet_df, last_price

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Train Prophet model on percentage returns.

        Args:
            df: DataFrame with timestamp and price data
            target_col: Column to predict
            validation_split: Fraction for validation

        Returns:
            Training metrics
        """
        logger.info(f"Training Prophet model on {len(df)} samples (using returns)")

        # Prepare data (converts to returns)
        prophet_df, self._last_price = self._prepare_data(df, target_col)

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

        # Validate - predict returns
        future = self.model.make_future_dataframe(periods=len(val_data), freq="h")
        forecast = self.model.predict(future)

        # Get return predictions and actuals
        val_pred_returns = forecast.iloc[-len(val_data):]["yhat"].values
        val_actual_returns = val_data["y"].values

        # Calculate return-based metrics
        mae = np.mean(np.abs(val_pred_returns - val_actual_returns))
        rmse = np.sqrt(np.mean((val_pred_returns - val_actual_returns) ** 2))

        # For returns, MAPE doesn't make sense (returns can be 0)
        # Use MAE relative to mean return magnitude instead
        mean_return_mag = np.mean(np.abs(val_actual_returns))
        mape = (mae / mean_return_mag * 100) if mean_return_mag > 0 else 100

        # Direction accuracy - this is what matters most
        val_direction_actual = np.sign(val_actual_returns)
        val_direction_pred = np.sign(val_pred_returns)
        direction_accuracy = np.mean(val_direction_actual == val_direction_pred)

        self.training_metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),  # Relative to mean return magnitude
            "direction_accuracy": float(direction_accuracy),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "prediction_type": "returns",
        }

        self.is_trained = True
        self.last_trained = datetime.now()
        self.train_df = prophet_df

        logger.info(f"Prophet training complete. Direction accuracy: {direction_accuracy:.2%}, MAE: {mae:.4f}%")

        return self.training_metrics

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        current_price: Optional[float] = None,
    ) -> PredictionResult:
        """
        Make prediction with Prophet.

        Predicts return and converts back to price using current_price.

        Args:
            df: Recent data for context
            horizon: Periods ahead to predict
            current_price: Current price for conversion and direction

        Returns:
            PredictionResult with prediction and intervals
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        # Prepare recent data
        prophet_df, last_data_price = self._prepare_data(df)

        # Use provided current_price or fall back to last price in data
        if current_price is None:
            current_price = last_data_price

        # Determine frequency from data
        if len(prophet_df) > 1:
            time_diff = prophet_df["ds"].diff().median()
            if time_diff <= timedelta(minutes=30):
                freq = "30min"
            elif time_diff <= timedelta(hours=1):
                freq = "h"
            elif time_diff <= timedelta(hours=4):
                freq = "4h"
            else:
                freq = "D"
        else:
            freq = "h"

        # Create future dataframe
        last_date = prophet_df["ds"].max()
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
        future = pd.DataFrame({"ds": future_dates})

        # Make prediction (returns percentage return)
        forecast = self.model.predict(future)

        # Get the predicted return and confidence intervals
        pred_row = forecast.iloc[-1]
        predicted_return = float(pred_row["yhat"])  # This is percentage return
        ci_return_lower = float(pred_row["yhat_lower"])
        ci_return_upper = float(pred_row["yhat_upper"])

        # Convert return prediction to price
        # If predicted return is +1%, price goes up 1%
        predicted_price = current_price * (1 + predicted_return / 100)

        # Convert confidence intervals to price
        ci_95_lower = current_price * (1 + ci_return_lower / 100)
        ci_95_upper = current_price * (1 + ci_return_upper / 100)

        # Estimate std_dev from return intervals
        return_std = (ci_return_upper - ci_return_lower) / (2 * 1.96)
        std_dev = current_price * return_std / 100

        # Calculate all confidence intervals
        ci_50, ci_80, _ = self.calculate_confidence_intervals(predicted_price, std_dev)
        ci_95 = (ci_95_lower, ci_95_upper)

        # Determine direction
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
            features_used=["trend", "seasonality", "returns"],
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
                "last_price": self._last_price,
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
        self._last_price = data.get("last_price")

        logger.info(f"Prophet model loaded from {path}")
