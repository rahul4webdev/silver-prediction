"""
XGBoost model for feature-based prediction.
Provides interpretable feature importance and handles non-linear relationships.

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
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from app.ml.models.base import BaseModel, PredictionResult

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost model for price prediction.

    Predicts percentage returns instead of absolute prices for better
    performance on commodities.

    Strengths:
    - Handles non-linear relationships
    - Provides feature importance
    - Fast training and inference
    - Robust to overfitting with proper tuning
    - Returns-based prediction normalizes across different price levels

    Limitations:
    - Doesn't capture sequential patterns directly
    - Requires feature engineering
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        lookback_periods: int = 10,
    ):
        super().__init__("xgboost")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.lookback_periods = lookback_periods

        self.model: Optional[xgb.XGBRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self._last_price: Optional[float] = None

    def _create_return_features(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
    ) -> Tuple[pd.DataFrame, np.ndarray, float]:
        """
        Create return-based features for prediction.

        Returns features based on percentage returns, not absolute prices.
        This normalizes the data across different price levels.

        Returns:
            df_features: DataFrame with return-based features
            target_returns: Array of target returns (percentage)
            last_price: Last price in the data
        """
        result = df.copy()
        prices = result[target_col].values
        last_price = float(prices[-1])

        # Calculate target returns (what we want to predict)
        # Returns[i] = (Price[i+1] - Price[i]) / Price[i] * 100
        # We shift by -1 to get next period's return as target
        target_returns = np.zeros(len(prices))
        target_returns[:-1] = np.diff(prices) / prices[:-1] * 100
        target_returns[-1] = 0  # Last row has no next return

        # Historical return lags
        for i in range(1, self.lookback_periods + 1):
            result[f"return_lag_{i}"] = result[target_col].pct_change(i) * 100

        # Rolling return statistics
        returns = result[target_col].pct_change() * 100
        for window in [5, 10, 20]:
            result[f"return_mean_{window}"] = returns.rolling(window).mean()
            result[f"return_std_{window}"] = returns.rolling(window).std()
            result[f"return_min_{window}"] = returns.rolling(window).min()
            result[f"return_max_{window}"] = returns.rolling(window).max()

        # Price position relative to rolling range (normalized 0-1)
        for window in [10, 20]:
            rolling_min = result[target_col].rolling(window).min()
            rolling_max = result[target_col].rolling(window).max()
            rolling_range = rolling_max - rolling_min
            result[f"price_position_{window}"] = (result[target_col] - rolling_min) / (rolling_range + 1e-8)

        # Volatility features (normalized)
        result["daily_range_pct"] = (result["high"] - result["low"]) / result[target_col] * 100
        result["daily_return"] = result[target_col].pct_change() * 100

        # Volume change (if available)
        if "volume" in result.columns:
            result["volume_change_pct"] = result["volume"].pct_change() * 100
            result["volume_change_pct"] = result["volume_change_pct"].clip(-100, 100)

        # Technical indicators (if available) - convert to relative form
        if "rsi_14" in result.columns:
            result["rsi_centered"] = result["rsi_14"] - 50  # Center around 0

        if "macd" in result.columns and "close" in result.columns:
            # Normalize MACD by price
            result["macd_pct"] = result["macd"] / result[target_col] * 100

        # Bollinger band position (already normalized)
        if "bb_upper" in result.columns and "bb_lower" in result.columns:
            bb_range = result["bb_upper"] - result["bb_lower"]
            result["bb_position"] = (result[target_col] - result["bb_lower"]) / (bb_range + 1e-8)

        # Time features (if timestamp available)
        if "timestamp" in result.columns:
            ts = pd.to_datetime(result["timestamp"])
            result["hour"] = ts.dt.hour
            result["day_of_week"] = ts.dt.dayofweek
            # Normalize time features
            result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
            result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
            result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
            result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)

        return result, target_returns, last_price

    def _prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        is_training: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Prepare features for XGBoost.

        Returns:
            X: Feature matrix
            y: Target returns (percentage)
            last_price: Last price in data
        """
        # Create return-based features
        df_features, target_returns, last_price = self._create_return_features(df, target_col)

        # Drop rows with NaN (from lag creation)
        valid_mask = ~df_features.isna().any(axis=1)

        # Select feature columns (exclude non-features)
        exclude_cols = [
            "timestamp", "date", "id", target_col,
            "open", "high", "low", "close", "volume",  # Raw prices
            "hour", "day_of_week", "day_of_month", "month",  # Raw time (use sin/cos instead)
        ]
        feature_cols = [c for c in df_features.columns if c not in exclude_cols]

        # Filter to numeric columns only
        feature_cols = [c for c in feature_cols if df_features[c].dtype in ["float64", "int64", "float32", "int32"]]

        if is_training:
            self.feature_columns = feature_cols

        # Apply valid mask
        df_features = df_features[valid_mask]
        target_returns = target_returns[valid_mask.values]

        # Extract features
        X = df_features[self.feature_columns].values
        y = target_returns

        return X, y, last_price

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Train XGBoost model on percentage returns.

        Args:
            df: DataFrame with features
            target_col: Column to predict
            validation_split: Fraction for validation

        Returns:
            Training metrics
        """
        logger.info(f"Training XGBoost model on {len(df)} samples (using returns)")

        # Prepare features
        X, y, self._last_price = self._prepare_features(df, target_col, is_training=True)

        # Remove last row (target return is 0/unknown)
        X = X[:-1]
        y = y[:-1]

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Create and train model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Get feature importance
        importance = self.model.feature_importances_
        self.feature_importance = {
            col: float(imp) for col, imp in zip(self.feature_columns, importance)
        }

        # Calculate validation metrics on returns
        y_pred = self.model.predict(X_val)

        # MAE on returns (percentage points)
        mae = np.mean(np.abs(y_pred - y_val))

        # RMSE on returns
        rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))

        # MAPE relative to mean return magnitude
        mean_return_mag = np.mean(np.abs(y_val))
        mape = (mae / mean_return_mag * 100) if mean_return_mag > 0 else 100

        # Direction accuracy - this is what matters most
        direction_actual = np.sign(y_val)
        direction_pred = np.sign(y_pred)
        direction_accuracy = np.mean(direction_actual == direction_pred)

        self.training_metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),  # Relative to mean return magnitude
            "direction_accuracy": float(direction_accuracy),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "n_features": len(self.feature_columns),
            "prediction_type": "returns",
        }

        self.is_trained = True
        self.last_trained = datetime.now()

        logger.info(f"XGBoost training complete. Direction accuracy: {direction_accuracy:.2%}, MAE: {mae:.4f}%")
        logger.info(f"Top 5 features: {sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]}")

        return self.training_metrics

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        current_price: Optional[float] = None,
        n_estimators_for_std: int = 50,
    ) -> PredictionResult:
        """
        Make prediction with XGBoost.

        Predicts return and converts back to price using current_price.

        Args:
            df: Recent data with features
            horizon: Periods ahead (note: XGBoost predicts next period return)
            current_price: Current price for conversion and direction
            n_estimators_for_std: Number of estimators for std estimation

        Returns:
            PredictionResult
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        # Prepare features
        X, _, last_data_price = self._prepare_features(df, is_training=False)
        X_scaled = self.scaler.transform(X)

        # Get last sample for prediction
        X_last = X_scaled[-1:, :]

        # Use provided current_price or fall back to last price in data
        if current_price is None:
            current_price = float(df["close"].iloc[-1])

        # Main prediction (predicted return in percentage)
        predicted_return = float(self.model.predict(X_last)[0])

        # Estimate uncertainty using different numbers of trees
        return_predictions = []
        for n_trees in range(max(10, self.n_estimators - n_estimators_for_std), self.n_estimators + 1, 5):
            pred = float(self.model.predict(X_last, iteration_range=(0, n_trees))[0])
            return_predictions.append(pred)

        return_std = float(np.std(return_predictions)) if len(return_predictions) > 1 else abs(predicted_return * 0.5)

        # Convert return prediction to price
        predicted_price = current_price * (1 + predicted_return / 100)

        # Convert return std to price std
        std_dev = current_price * return_std / 100

        # Calculate confidence intervals
        ci_50, ci_80, ci_95 = self.calculate_confidence_intervals(predicted_price, std_dev)

        # Determine direction
        direction, direction_prob = self.determine_direction(current_price, predicted_price)

        # Calculate target time
        if "timestamp" in df.columns:
            last_time = pd.to_datetime(df["timestamp"].iloc[-1])
        else:
            last_time = datetime.now()

        # Determine time delta based on data frequency
        if len(df) > 1 and "timestamp" in df.columns:
            time_diff = pd.to_datetime(df["timestamp"]).diff().median()
            if pd.notna(time_diff):
                target_time = last_time + time_diff * horizon
            else:
                target_time = last_time + timedelta(hours=horizon)
        else:
            target_time = last_time + timedelta(hours=horizon)

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
            target_time=target_time,
            features_used=list(self.feature_importance.keys())[:10] + ["returns"],
        )

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get top N feature importances."""
        sorted_importance = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return dict(sorted_importance[:top_n])

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "feature_importance": self.feature_importance,
                "is_trained": self.is_trained,
                "last_trained": self.last_trained,
                "training_metrics": self.training_metrics,
                "last_price": self._last_price,
                "config": {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "lookback_periods": self.lookback_periods,
                },
            }, f)

        logger.info(f"XGBoost model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_columns = data["feature_columns"]
        self.feature_importance = data["feature_importance"]
        self.is_trained = data["is_trained"]
        self.last_trained = data["last_trained"]
        self.training_metrics = data["training_metrics"]
        self._last_price = data.get("last_price")

        config = data["config"]
        self.n_estimators = config["n_estimators"]
        self.max_depth = config["max_depth"]
        self.learning_rate = config["learning_rate"]
        self.lookback_periods = config["lookback_periods"]

        logger.info(f"XGBoost model loaded from {path}")
